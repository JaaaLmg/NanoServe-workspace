from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 活动请求索引（seq_id -> Sequence）：取消/超时按 id 定位，免于遍历队列。
        # 覆盖 waiting/running 队列成员与被抢占（PREEMPTED）后暂停在队列之外的请求，
        # 生命周期扫描与完成判断都以本索引为权威；终态请求从此处移除
        self.requests: dict[int, Sequence] = {}
        # 活动 request_id 集合：保证外部 ID 在活动期内唯一，取消才能唯一定位。
        # 请求终态（从 requests 移除）后其 ID 允许复用
        self._active_request_ids: set[str] = set()

    def is_finished(self):
        # 以活动索引为权威完成判断：等待/运行/暂停（PREEMPTED）中的请求都算未完成，
        # 只有全部请求终态（索引清空）才视为完成。终态请求不在任何队列，
        # 因此仅凭队列空判断会把"暂停请求未清理"误判为已完成
        return not self.requests

    # ---------- 队列与资源的统一入口（状态、队列、KV 三者在此保持一致） ----------

    def add(self, seq: Sequence):
        """接收新请求：先完成全部校验，再写入索引与队列（失败零污染，不变量 7）。

        - 仅接纳 WAITING 状态的请求（显式异常，不依赖会被 python -O 去除的断言）；
        - 同一 seq_id 不允许两个活动对象；
        - 活动 request_id 必须唯一（自动 ID 与自定义 ID 一视同仁），
          否则取消/查询无法唯一定位请求。
        """
        if seq.status != SequenceStatus.WAITING:
            raise ValueError(f"仅接纳 WAITING 请求，当前状态为 {seq.status.name}")
        if seq.seq_id in self.requests:
            raise ValueError(f"seq_id={seq.seq_id} 的活动请求已存在，禁止重复添加")
        if seq.request_id in self._active_request_ids:
            raise ValueError(f"request_id={seq.request_id!r} 的活动请求已存在，禁止重复注册")
        self.requests[seq.seq_id] = seq
        self._active_request_ids.add(seq.request_id)
        self._enqueue_waiting(seq)

    def _enqueue_waiting(self, seq: Sequence, *, front: bool = False):
        """只接纳 WAITING 请求进入等待队列，并防止重复入队。"""
        if seq.status != SequenceStatus.WAITING:
            raise ValueError(f"仅 WAITING 请求可进入 waiting 队列，当前为 {seq.status.name}")
        if seq in self.waiting:
            return
        self.waiting.appendleft(seq) if front else self.waiting.append(seq)

    def _enqueue_running(self, seq: Sequence):
        """只接纳 RUNNING 请求进入运行队列，并防止同一对象重复出现。"""
        if seq.status != SequenceStatus.RUNNING:
            raise ValueError(f"仅 RUNNING 请求可进入 running 队列，当前为 {seq.status.name}")
        if seq in self.running:
            return
        self.running.append(seq)

    def _remove_from_queues(self, seq: Sequence):
        """从任意队列移除；对象已不在队列时安全返回（幂等）。"""
        try:
            self.waiting.remove(seq)
        except ValueError:
            pass
        try:
            self.running.remove(seq)
        except ValueError:
            pass

    def _release_sequence(self, seq: Sequence):
        """统一释放 KV block 并清空调度临时计数（幂等，不产生 double free）。

        BlockManager.deallocate 以 block_table 为准做引用计数扣减，
        释放后 table 清空，因此重复调用是安全的空操作。
        """
        if seq.block_table:
            self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0

    def _finalize(self, seq: Sequence):
        """终态收尾：移出所有队列 + 释放资源 + 从活动索引删除（幂等）。

        所有权保护：只有当请求仍登记在活动索引中（本次调用拥有该记录）时，
        才移除索引并清除其 request_id 唯一性标记。否则"旧对象的延迟/重复
        收尾"（如对早已完成的批次重复执行 postprocess）会误删复用了同一
        request_id 的新请求的身份记录，破坏活动 ID 唯一性。
        资源释放与队列移除本身幂等，对非所有者调用是安全的空操作。
        """
        owned = self.requests.get(seq.seq_id) is seq
        self._remove_from_queues(seq)
        self._release_sequence(seq)
        if owned:
            del self.requests[seq.seq_id]
            self._active_request_ids.discard(seq.request_id)

    # ---------- 控制入口：取消 / 超时 / 抢占 / 恢复 ----------

    def cancel(self, seq_id: int, reason: str = "client_cancelled",
               now: float | None = None) -> bool:
        """取消请求（统一入口）：设置终态、移出队列并释放资源。

        - 请求不存在返回 False；请求已是终态时返回 False，
          但仍会幂等完成剩余清理（终态对象被外部 mark_* 留在队列/索引中时，
          不能因控制入口"重复调用"而遗留未释放资源）；
        - 首次记录的取消原因不被后续重复取消覆盖；
        - 与 TIMEOUT 的区分保留在 finish_reason 中，便于指标统计。
        """
        seq = self.requests.get(seq_id)
        if seq is None:
            return False
        if seq.is_terminal:
            self._finalize(seq)
            return False
        # 先打标记（原因只记录首次），再以记录下来的原因做终态迁移
        seq.request_cancel(reason)
        seq.mark_cancelled(seq.cancel_reason or reason, now=now)
        self._finalize(seq)
        return True

    def timeout(self, seq_id: int, now: float | None = None,
                reason: str = "deadline_exceeded") -> bool:
        """超时终止（统一入口）：语义与 cancel 相同但 finish_reason 不同。"""
        seq = self.requests.get(seq_id)
        if seq is None:
            return False
        if seq.is_terminal:
            self._finalize(seq)
            return False
        seq.mark_timeout(reason, now=now)
        self._finalize(seq)
        return True

    def check_deadlines(self, now: float | None = None) -> list[Sequence]:
        """在调度边界统一执行 deadline 检查，返回本轮超时的请求。

        以活动索引为扫描范围（覆盖队列成员与暂停请求），使用单调时钟；
        带取消标记的请求跳过——取消优先于超时（客户端显式意图优先），
        由 _purge_cancelled 按 CANCELLED 处理。
        """
        if now is None:
            now = perf_counter()
        timed_out = []
        for seq in list(self.requests.values()):
            if seq.cancel_requested or seq.is_terminal:
                continue
            if seq.deadline is not None and now >= seq.deadline:
                self.timeout(seq.seq_id, now=now)
                timed_out.append(seq)
        return timed_out

    def _purge_cancelled(self, now: float | None = None) -> list[Sequence]:
        """调度边界安全检查：处理模型执行期间被置位取消标记的请求。

        取消可能发生在一次模型执行前后（未来 HTTP/SSE 为异步），
        也可能作用于暂停中的请求；扫描范围是活动索引而非队列。
        """
        cancelled = []
        for seq in list(self.requests.values()):
            if seq.cancel_requested and not seq.is_terminal:
                self.cancel(seq.seq_id, reason=seq.cancel_reason or "client_cancelled", now=now)
                cancelled.append(seq)
        return cancelled

    def _purge_terminal(self) -> list[Sequence]:
        """调度边界兜底：清理已被外部置为终态但仍被持有的请求。

        例如调用方直接使用公开的 mark_cancelled()/mark_timeout()/mark_finished()。
        不做兜底会导致终态请求再次参与调度：WAITING 中的会先分配 block 再在
        迁移时抛异常，RUNNING 中的会再次进入 decode（违反不变量 1/5）。
        """
        finalized = []
        for seq in list(self.requests.values()):
            if seq.is_terminal:
                self._finalize(seq)
                finalized.append(seq)
        return finalized

    def preempt(self, seq: Sequence):
        """抢占：RUNNING -> PREEMPTED，释放 KV block（不回滚已生成 token）。

        只完成抢占本身，不负责重新入队；恢复由 resume() 显式执行。
        暂停中的请求仍保留在活动索引里，参与生命周期扫描与完成判断。
        恢复采用 recompute 方案：物理块释放，token 进度保留，恢复时重新 prefill。
        """
        seq.transition_to(SequenceStatus.PREEMPTED)
        seq.is_prefill = True
        self._remove_from_queues(seq)
        self._release_sequence(seq)

    def resume(self, seq: Sequence):
        """恢复被抢占请求：只允许 PREEMPTED -> WAITING，重新入队等待 recompute。

        front=True 使被抢占请求排在等待队列头部（沿用原抢占行为的优先级）。
        """
        seq.transition_to(SequenceStatus.WAITING)
        seq.is_prefill = True
        self._enqueue_waiting(seq, front=True)

    # ---------- 调度主流程 ----------

    def schedule(self) -> tuple[list[Sequence], bool]:
        # 调度边界：统一生命周期扫描。顺序即优先级——终态兜底最先是纯清理；
        # 取消先于超时（客户端显式意图优先于系统判断）。
        # 扫描范围为活动索引，覆盖暂停（PREEMPTED）在队列之外的请求
        self._purge_terminal()
        self._purge_cancelled()
        self.check_deadlines()

        scheduled_seqs = []
        num_batched_tokens = 0

        # 空队列直接返回空批次，不进入模型执行（空批次契约由 Engine 兜住）
        if not self.waiting and not self.running:
            return scheduled_seqs, False

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # prefill 完成才算被调度接纳：WAITING -> RUNNING 走统一迁移入口
                seq.transition_to(SequenceStatus.RUNNING)
                self.waiting.popleft()
                self._enqueue_running(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 块不足：抢占合法的 RUNNING 请求释放资源。
                # preempt 只做抢占，resume 立即让它以 WAITING 身份回到等待队列
                # （观察点状态为 WAITING，迁移历史记录在 num_preempts/last_preempted_at）
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                    self.resume(victim)
                else:
                    self.preempt(seq)
                    self.resume(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool,
                    now: float | None = None):
        if now is None:
            now = perf_counter()
        for seq, token_id in zip(seqs, token_ids):
            # 安全边界（模型执行返回后、追加 token 与 KV 记账之前）：
            # 1) 终态兜底：外部 mark_* 产生的终态请求直接清理，不参与记账；
            # 2) 取消标记优先于超时（客户端显式意图优先于系统判断）；
            # 3) 模型执行期间跨过 deadline 的请求不得被记为正常完成——
            #    丢弃本轮采样 token，按 TIMEOUT 终止并释放资源。
            # 否则请求会在下一轮 check_deadlines 之前以 FINISHED/length 落账，
            # 从活动索引移除后超时就再也无法纠正。
            if seq.is_terminal:
                self._finalize(seq)
                continue
            if seq.cancel_requested:
                seq.mark_cancelled(seq.cancel_reason or "client_cancelled", now=now)
                self._finalize(seq)
                continue
            if seq.deadline is not None and now >= seq.deadline:
                seq.mark_timeout("deadline_exceeded", now=now)
                self._finalize(seq)
                continue
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            # 完成判定：EOS 触发记为 stop，达到 max_tokens 记为 length；
            # mark_finished 幂等，配合 _finalize 保证不 double free
            if not seq.ignore_eos and token_id == self.eos:
                seq.mark_finished("stop")
                self._finalize(seq)
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.mark_finished("length")
                self._finalize(seq)
