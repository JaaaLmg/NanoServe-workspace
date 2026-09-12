import json
import logging
from collections import deque
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config, validate_positive_int
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

# 模块级 logger：库代码不做 basicConfig，日志开关由调用方（验收脚本/服务层）控制
logger = logging.getLogger(__name__)

# ---------- 本轮决策原因（§5.1 原因分类：不把所有等待都算 budget） ----------
REASON_SCHEDULED = "scheduled"            # 实际接纳正 token 工作（含首请求部分 chunk）
REASON_BUDGET = "budget"                  # 队首/候选需求大于 remaining，或预算已耗尽
REASON_SEQUENCE_CAP = "sequence_cap"      # 序列数上限先阻止接纳（与 budget 同现时优先）
REASON_KV_CAPACITY = "kv_capacity"        # 已查询候选但 KV 容量不足或被 KV 抢占
REASON_HEAD_OF_LINE = "head_of_line"      # 前序请求处停止，尾部未独立检查预算/KV
REASON_PHASE_PRIORITY = "phase_priority"  # 本轮已有 prefill，running 未成为 decode 候选
REASON_PAUSED = "paused"                  # 独立 PREEMPTED，尚未恢复


@dataclass
class RoundDecision:
    """单个请求在本调度轮的决策快照（仅 IDs/计数/原因，不含 prompt 或 token 内容）。"""

    seq_id: int
    request_id: str
    reason: str
    # 本轮希望推进的输入 token 数；未成为候选或 KV 查询失败时为 None（不虚构需求）
    needed_tokens: int | None = None
    # 实际接纳的正 token 数；延后为 0
    scheduled_tokens: int = 0
    # 本轮是否查询过该请求的 KV 可行性（用于区分"预算先阻止"与"KV 已确认不足"）
    kv_checked: bool = False
    # head_of_line 专用：阻塞它的首个未处理请求 seq_id 与其直接原因
    blocked_by_seq_id: int | None = None
    blocking_reason: str | None = None
    # 决策时该请求累计预算等待秒数（已结算 + 当前未结算，§5.3）
    budget_wait_seconds: float = 0.0
    # ---------- Day8 chunk 观测字段（仅 scheduled 决策填充，其余为 None） ----------
    # 本轮接纳的是该请求当前 prefill 阶段的第几个 chunk（1-based，由 Scheduler
    # 的 per-request 计数器维护；prefix 命中/抢占恢复开启新 prefill 阶段时
    # 重新从 1 计数），供日志排查与 chunk 序列连续性验证
    chunk_index: int | None = None
    # 接纳时刻的进度起点（已提交 KV 的上下文 token 数）
    offset_before: int | None = None
    # 本轮是否为该请求当前 prefill 阶段的最后一个 chunk（接纳即完成 prefill）
    is_last_chunk: bool | None = None

    def to_dict(self) -> dict:
        return {
            "seq_id": self.seq_id,
            "request_id": self.request_id,
            "needed_tokens": self.needed_tokens,
            "scheduled_tokens": self.scheduled_tokens,
            "reason": self.reason,
            "kv_checked": self.kv_checked,
            "blocked_by_seq_id": self.blocked_by_seq_id,
            "blocking_reason": self.blocking_reason,
            "budget_wait_seconds": self.budget_wait_seconds,
            "chunk_index": self.chunk_index,
            "offset_before": self.offset_before,
            "is_last_chunk": self.is_last_chunk,
        }


@dataclass
class BudgetWaitStats:
    """请求的预算等待统计（rank 0 Scheduler 所有，按 seq_id 对象所有权管理）。

    只为有过 budget episode 的请求创建记录；终态摘要后删除，避免全历史驻留。
    request_id 可复用，因此不作为唯一 key；新请求对象从 0 开始。
    """

    seq_id: int
    request_id: str
    # 处于 budget / 预算 HOL 原因的调度轮数（请求-轮次口径）
    budget_deferred_rounds: int = 0
    # 当前开放 episode 的起始单调时钟；None 表示无开放 episode
    budget_wait_started_at: float | None = None
    # 已结算（closed）的预算等待秒数；开放时段不重复计入
    budget_wait_seconds: float = 0.0


def _log_event(payload: dict):
    """以 JSON Lines 发出结构化事件；INFO 未开启时不构造字符串，避免无谓开销。"""
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps(payload, ensure_ascii=False))


class Scheduler:

    def __init__(self, config: Config):
        # Day7：Scheduler 面向 SimpleNamespace 等直接构造路径，与 Config 共用同一
        # 校验入口，避免两套口径漂移；校验在创建资源账本（BlockManager）之前完成。
        # 两项约束独立生效：token 预算限制每轮输入 token 总量，max_num_seqs 限制批序列数；
        # 允许 B < max_num_seqs（decode 按预算分批的正常配置）。
        self.max_num_seqs = validate_positive_int(config.max_num_seqs, "max_num_seqs")
        self.max_num_batched_tokens = validate_positive_int(
            config.max_num_batched_tokens, "max_num_batched_tokens")
        # Day8：单请求单轮 prefill query 上限。直接构造路径（SimpleNamespace）
        # 缺省该字段时回退默认值 1024（与 Config 默认一致），测试显式提供。
        # 与 B 相互独立：chunk_size 限制每请求，B 限制每轮总量，互不替代。
        self.chunk_size = validate_positive_int(
            getattr(config, "chunk_size", 1024), "chunk_size")
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

        # ---------- Day7：轮次与预算等待统计（rank 0 内部字段，不进 TP payload） ----------
        # 每次 schedule 自增的单调轮次 ID；空轮也分配，不用外部 ID 充当轮次 ID
        self._round_counter = 0
        # 正在进行的调度轮 ID；轮外操作（如显式 cancel）结算 episode 时为 None
        self._current_round_id: int | None = None
        # 上一轮的调度计划快照（供 Engine 关联 round_id/planned_tokens，内存有界）
        self.last_schedule_stats: dict | None = None
        # 预算等待记录：seq_id -> BudgetWaitStats；只为有过 budget episode 的请求创建
        self.budget_wait: dict[int, BudgetWaitStats] = {}
        # Day8：请求当前 prefill 阶段已接纳的 chunk 计数（rank 0 观测字段，不进
        # TP payload）。首次接纳（无 block_table，即新阶段首个 chunk）重置为 1；
        # 终态收尾时删除记录，避免全历史驻留
        self._prefill_chunk_count: dict[int, int] = {}
        # Scheduler 生命周期标量累计（§5.3）：
        # budget_deferred_unique_requests_total：出现过 budget episode 的唯一请求数
        # budget_deferred_request_rounds_total：处于预算等待原因的请求-轮次总数
        # budget_wait_closed_seconds_total：已结算的预算等待秒数总和
        self.budget_deferred_unique_requests_total = 0
        self.budget_deferred_request_rounds_total = 0
        self.budget_wait_closed_seconds_total = 0.0

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

    def _finalize(self, seq: Sequence, now: float | None = None):
        """终态收尾：移出所有队列 + 释放资源 + 从活动索引删除（幂等）。

        所有权保护：只有当请求仍登记在活动索引中（本次调用拥有该记录）时，
        才移除索引并清除其 request_id 唯一性标记。否则"旧对象的延迟/重复
        收尾"（如对早已完成的批次重复执行 postprocess）会误删复用了同一
        request_id 的新请求的身份记录，破坏活动 ID 唯一性。
        资源释放与队列移除本身幂等，对非所有者调用是安全的空操作。
        预算等待统计同受所有权保护：终态摘要/episode 结算只在拥有记录时执行一次，
        重复收尾不会重复累计或删除新对象的记录。
        """
        owned = self.requests.get(seq.seq_id) is seq
        self._remove_from_queues(seq)
        self._release_sequence(seq)
        if owned:
            del self.requests[seq.seq_id]
            self._active_request_ids.discard(seq.request_id)
            # Day8 chunk 计数随请求终态回收（seq_id 全局唯一，记录不会误伤新请求）
            self._prefill_chunk_count.pop(seq.seq_id, None)
            # 终态预算收尾：结算未关闭 episode、发终态等待摘要并删除记录
            self._settle_terminal_budget_stats(seq, now)

    # ---------- 控制入口：取消 / 超时 / 抢占 / 恢复 ----------

    def cancel(self, seq_id: int, reason: str = "client_cancelled",
               now: float | None = None) -> bool:
        """取消请求（统一入口）：设置终态、移出队列并释放资源。

        - 请求不存在返回 False；请求已是终态时返回 False，
          但仍会幂等完成剩余清理（终态对象被外部 mark_* 留在队列/索引中时，
          不能因控制入口"重复调用"而遗留未释放资源）；
        - 首次记录的取消原因不被后续重复取消覆盖；
        - 与 TIMEOUT 的区分保留在 finish_reason 中，便于指标统计；
        - 终态时刻即预算 episode 结算时刻（显式 now 优先，否则读同一单调时钟）。
        """
        seq = self.requests.get(seq_id)
        if seq is None:
            return False
        if seq.is_terminal:
            self._finalize(seq, now)
            return False
        # 先打标记（原因只记录首次），再以记录下来的原因做终态迁移
        seq.request_cancel(reason)
        seq.mark_cancelled(seq.cancel_reason or reason, now=now)
        self._finalize(seq, now)
        return True

    def timeout(self, seq_id: int, now: float | None = None,
                reason: str = "deadline_exceeded") -> bool:
        """超时终止（统一入口）：语义与 cancel 相同但 finish_reason 不同。"""
        seq = self.requests.get(seq_id)
        if seq is None:
            return False
        if seq.is_terminal:
            self._finalize(seq, now)
            return False
        seq.mark_timeout(reason, now=now)
        self._finalize(seq, now)
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

    def _purge_terminal(self, now: float | None = None) -> list[Sequence]:
        """调度边界兜底：清理已被外部置为终态但仍被持有的请求。

        例如调用方直接使用公开的 mark_cancelled()/mark_timeout()/mark_finished()。
        不做兜底会导致终态请求再次参与调度：WAITING 中的会先分配 block 再在
        迁移时抛异常，RUNNING 中的会再次进入 decode（违反不变量 1/5）。
        结算时刻使用"Scheduler 首次观察时刻"（传入的 now），不倒填外部时间。
        """
        finalized = []
        for seq in list(self.requests.values()):
            if seq.is_terminal:
                self._finalize(seq, now)
                finalized.append(seq)
        return finalized

    def preempt(self, seq: Sequence, now: float | None = None):
        """抢占：RUNNING -> PREEMPTED，释放 KV block（不回滚已生成 token）。

        只完成抢占本身，不负责重新入队；恢复由 resume() 显式执行。
        暂停中的请求仍保留在活动索引里，参与生命周期扫描与完成判断。
        恢复采用 recompute 方案：物理块释放，token 进度保留，恢复时重新 prefill。
        显式抢占同时结束该请求的预算等待 episode：此后等待原因为 paused/KV，
        不应继续计入预算等待。
        """
        if now is None:
            now = perf_counter()
        seq.transition_to(SequenceStatus.PREEMPTED)
        seq.is_prefill = True
        self._remove_from_queues(seq)
        self._release_sequence(seq)
        self._update_budget_wait(seq, deferred=False, now=now,
                                 round_id=self._current_round_id, close_reason="preempted")

    def resume(self, seq: Sequence):
        """恢复被抢占请求：只允许 PREEMPTED -> WAITING，重新入队等待 recompute。

        front=True 使被抢占请求排在等待队列头部（沿用原抢占行为的优先级）。
        """
        seq.transition_to(SequenceStatus.WAITING)
        seq.is_prefill = True
        self._enqueue_waiting(seq, front=True)

    # ---------- Day7：预算等待统计（§5.2/§5.3） ----------

    def _update_budget_wait(self, seq: Sequence, *, deferred: bool, now: float,
                            round_id: int | None, close_reason: str | None = None):
        """按本轮观察原因维护请求的预算等待 episode（幂等）。

        - deferred=True：主原因为 budget（direct 或预算 HOL）——累计请求-轮次；
          无 episode 则开启（started_at=now），已有 episode 不重开（连续等待
          不重置起点）；首次创建记录时唯一请求数 +1。
        - deferred=False：本轮观察原因不再属于 budget（被调度、原因切换、
          终态、抢占）——结算 now - started_at 后关闭 episode；无开放 episode
          时是安全空操作，因此重复调用不会重复累计。
        """
        if deferred:
            stats = self.budget_wait.get(seq.seq_id)
            if stats is None:
                # 首次为该请求对象创建记录：唯一请求数 +1；
                # 复用 request_id 的新请求对象拥有全新记录，从 0 开始
                stats = BudgetWaitStats(seq_id=seq.seq_id, request_id=seq.request_id)
                self.budget_wait[seq.seq_id] = stats
                self.budget_deferred_unique_requests_total += 1
            stats.budget_deferred_rounds += 1
            self.budget_deferred_request_rounds_total += 1
            if stats.budget_wait_started_at is None:
                stats.budget_wait_started_at = now
            return
        stats = self.budget_wait.get(seq.seq_id)
        if stats is None or stats.budget_wait_started_at is None:
            return
        started_at = stats.budget_wait_started_at
        duration = max(0.0, now - started_at)
        stats.budget_wait_seconds += duration
        self.budget_wait_closed_seconds_total += duration
        stats.budget_wait_started_at = None
        # episode 关闭事件：含原始起止时间，验收方可独立重算等待秒数
        _log_event({
            "event": "budget_wait_episode",
            "seq_id": seq.seq_id,
            "request_id": seq.request_id,
            "started_at": started_at,
            "ended_at": now,
            "duration_seconds": duration,
            "close_reason": close_reason,
            "round_id": round_id,
            "observed_at": now,
        })

    def _budget_observed_seconds(self, seq_id: int, now: float) -> float:
        """截至 now 的请求预算等待总时长 = 已结算 closed + 当前开放 episode 的 now-started。

        该实时值仅用于展示/决策快照，不写回 closed 累计（§5.3）。
        """
        stats = self.budget_wait.get(seq_id)
        if stats is None:
            return 0.0
        total = stats.budget_wait_seconds
        if stats.budget_wait_started_at is not None:
            total += max(0.0, now - stats.budget_wait_started_at)
        return total

    def _settle_terminal_budget_stats(self, seq: Sequence, now: float | None):
        """终态预算收尾（仅在拥有活动记录时由 _finalize 调用一次）：
        结算未关闭 episode -> 发终态等待摘要 -> 删除记录（有界保存，§5.3/§6.3）。
        """
        if now is None:
            now = perf_counter()
        stats = self.budget_wait.get(seq.seq_id)
        if stats is not None and stats.budget_wait_started_at is not None:
            self._update_budget_wait(
                seq, deferred=False, now=now, round_id=self._current_round_id,
                close_reason=f"terminal:{seq.finish_reason}")
        # 终态等待摘要：供终态索引删除后的验收核对；无预算延后时输出 0。
        # 所有权保护保证重复收尾不会重复发送该事件
        _log_event({
            "event": "request_budget_wait",
            "seq_id": seq.seq_id,
            "request_id": seq.request_id,
            "budget_deferred_rounds": stats.budget_deferred_rounds if stats else 0,
            "budget_wait_seconds": stats.budget_wait_seconds if stats else 0.0,
            "status": seq.status.name,
            "finish_reason": seq.finish_reason,
            "round_id": self._current_round_id,
            "observed_at": now,
        })
        if stats is not None:
            del self.budget_wait[seq.seq_id]

    # ---------- Day7：prefill 需求估算与停止归因 ----------

    def _estimate_prefill_tokens(self, seq: Sequence) -> tuple[int, int | None]:
        """估算 prefill 本轮输入需求（只读查询，不分配资源、不改引用计数）。

        返回 (分配时沿用的缓存块数, 本轮需求 needed_tokens)；
        needed_tokens=None 表示 KV 容量不足（can_allocate 返回 -1），
        此时调用方不得声称"本次被 budget 拒绝"。
        - 未分配过 block 的请求：需求基于显式进度计算 = 总 token - prefix 命中
          token（命中块本轮不再执行、不占 token 预算）；
        - 已持有 block 的分块/恢复请求：需求 = 总 token - prefill_offset
          （prefill_offset 是已提交 KV 进度的唯一事实源）。
        """
        if not seq.block_table:
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                return 0, None
            return num_cached_blocks, seq.num_tokens - num_cached_blocks * self.block_size
        return 0, seq.num_tokens - seq.prefill_offset

    def _attribute_prefill_stop(self, decisions: list[RoundDecision], decided: set[int], *,
                                reason: str, needed: int | None, kv_checked: bool):
        """prefill 扫描停止：首个未处理请求记直接原因，其余未处理请求记 head_of_line。

        阶段内 FCFS 不跳过放不下的队首去选更短尾部，因此尾部请求并未被
        独立检查预算/KV，不能声称它们各自都放不下；blocked_by_seq_id 指向
        首个未处理请求，blocking_reason 说明停止原因。
        若队首是本轮刚分块、尚未完成 prefill 的请求（已有 scheduled 决策），
        则从其后第一个未处理请求开始归因，避免同一请求获得两条决策。
        """
        first = None
        for seq in self.waiting:
            if seq.seq_id not in decided:
                first = seq
                break
        if first is None:
            return
        decided.add(first.seq_id)
        decisions.append(RoundDecision(
            first.seq_id, first.request_id, reason,
            needed_tokens=needed, kv_checked=kv_checked))
        for seq in list(self.waiting):
            if seq.seq_id in decided:
                continue
            decided.add(seq.seq_id)
            decisions.append(RoundDecision(
                seq.seq_id, seq.request_id, REASON_HEAD_OF_LINE,
                blocking_reason=reason, blocked_by_seq_id=first.seq_id))

    def _record_budget_decisions(self, decisions: list[RoundDecision], now: float,
                                 round_id: int) -> tuple[int, int]:
        """按本轮决策快照更新预算等待统计，返回 (direct, hol) 预算延后人数。

        - direct：主原因即为 budget 的请求数；
        - hol：主原因为 head_of_line 且 blocking_reason=budget 的请求数；
        - 两者之和（本轮天然去重）即 budget_deferred_requests；
        - KV/sequence_cap/phase_priority/paused 不计入预算人数。
        """
        direct = hol = 0
        for d in decisions:
            if d.reason == REASON_BUDGET:
                budget_related = True
                direct += 1
            elif d.reason == REASON_HEAD_OF_LINE and d.blocking_reason == REASON_BUDGET:
                budget_related = True
                hol += 1
            else:
                budget_related = False
            seq = self.requests.get(d.seq_id)
            if seq is None:
                # 防御：决策针对活动请求生成，正常不会缺失；缺失则跳过统计
                continue
            # 原因为 budget 时开启/延续 episode；否则结算并关闭已有 episode
            self._update_budget_wait(
                seq, deferred=budget_related, now=now, round_id=round_id,
                close_reason=None if budget_related else d.reason)
            d.budget_wait_seconds = self._budget_observed_seconds(d.seq_id, now)
        return direct, hol

    def _record_paused_decisions(self, decisions: list[RoundDecision], decided: set[int]):
        """为独立 PREEMPTED（不在任何队列）的请求记录 paused 决策。

        暂停请求本轮不可执行：不计预算等待、不参与执行候选（§5.1）。
        本轮已因 KV 抢占被决策过的请求（已 resume 为 WAITING）不再重复记录。
        """
        for seq in self.requests.values():
            if seq.status == SequenceStatus.PREEMPTED and seq.seq_id not in decided:
                decided.add(seq.seq_id)
                decisions.append(RoundDecision(
                    seq.seq_id, seq.request_id, REASON_PAUSED))

    def _finalize_round(self, batch: list[Sequence], is_prefill: bool, phase: str,
                        decisions: list[RoundDecision], now: float) -> tuple[list[Sequence], bool]:
        """轮末收尾：以 batch 实际字段重算校验 -> 更新统计 -> 记录计划日志。"""
        round_id = self._current_round_id
        # 每轮末以 batch 的实际字段重算，不只信任循环局部变量（§4.2）
        planned = sum(seq.num_scheduled_tokens for seq in batch)
        if batch:
            if any(type(seq.num_scheduled_tokens) is not int
                   or seq.num_scheduled_tokens <= 0 for seq in batch):
                raise ValueError(
                    f"round {round_id}: num_scheduled_tokens 必须为正整数")
            if planned <= 0 or any(seq.num_scheduled_tokens <= 0 for seq in batch):
                raise ValueError(
                    f"round {round_id}: 非空批次存在非正的 num_scheduled_tokens "
                    f"（planned={planned}）")
            if planned > self.max_num_batched_tokens:
                raise ValueError(
                    f"round {round_id}: 计划 token {planned} 超过预算 "
                    f"{self.max_num_batched_tokens}")
            if len(batch) > self.max_num_seqs:
                raise ValueError(
                    f"round {round_id}: 批次序列数 {len(batch)} 超过上限 {self.max_num_seqs}")
            # Day8：chunk_size 上限与预算/序列数独立校验（四条独立，互不替代）。
            # 每请求每轮接纳的 prefill query 数不得超过 chunk_size（decode 恒为 1，
            # 天然满足，无需区分阶段）
            over_chunk = [seq.seq_id for seq in batch
                          if seq.num_scheduled_tokens > self.chunk_size]
            if over_chunk:
                raise ValueError(
                    f"round {round_id}: 序列 {over_chunk} 的单轮接纳数超过 "
                    f"chunk_size={self.chunk_size}")
            seq_ids = [seq.seq_id for seq in batch]
            if len(set(seq_ids)) != len(seq_ids):
                raise ValueError(f"round {round_id}: 批次内 seq_id 重复: {seq_ids}")
        direct, hol = self._record_budget_decisions(decisions, now, round_id)
        stats = {
            "event": "scheduler_round",
            "round_id": round_id,
            "phase": phase,
            "token_budget": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "planned_tokens": planned,
            "scheduled_requests": len(batch),
            "budget_deferred_direct": direct,
            "budget_deferred_hol": hol,
            "budget_deferred_requests": direct + hol,
            "observed_at": now,
            "decisions": [d.to_dict() for d in decisions],
        }
        self.last_schedule_stats = stats
        _log_event(stats)
        self._current_round_id = None
        return batch, is_prefill

    # ---------- 调度主流程 ----------

    def schedule(self, *, now: float | None = None) -> tuple[list[Sequence], bool]:
        """制定一轮调度计划，返回 (批次, 是否 prefill)（接口与 Day6 兼容）。

        预算口径：本轮模型输入 query token 总量 planned_tokens <= B，
        且非空批次每条 num_scheduled_tokens 为正整数；decode 同样受预算约束。
        时间语义：整轮使用一次单调时钟 now（可注入确定值用于测试）。
        """
        if now is None:
            now = perf_counter()
        # 每轮唯一单调 round_id；空轮也分配并记录
        self._round_counter += 1
        round_id = self._current_round_id = self._round_counter

        # 调度边界：统一生命周期扫描。顺序即优先级——终态兜底最先是纯清理；
        # 取消先于超时（客户端显式意图优先于系统判断）。
        # 扫描范围为活动索引，覆盖暂停（PREEMPTED）在队列之外的请求；
        # 终态/取消/到期的请求先按 Day6 清理并退出本轮活动统计
        self._purge_terminal(now)
        self._purge_cancelled(now)
        self.check_deadlines(now)

        budget = self.max_num_batched_tokens
        batch: list[Sequence] = []
        used = 0  # U：本轮已承诺的输入 token 数
        decisions: list[RoundDecision] = []
        decided: set[int] = set()

        # ---------- prefill：按 waiting 当前顺序（阶段内 FCFS）尝试接纳 ----------
        # Day8：扫描位置与队列分离——中间 chunk 的请求保持 WAITING 原地
        # （保留 block_table 与已提交进度），但本轮不再被扫描
        # （每请求每轮至多一个 chunk）；扫描位置前进，后续请求仍按 FCFS 考察
        scan = 0
        while self.waiting and len(batch) < self.max_num_seqs and scan < len(self.waiting):
            seq = self.waiting[scan]
            remaining = budget - used
            if remaining == 0:
                # 预算恰好耗尽（未命中序列数上限）：首个未处理请求记 budget，
                # 无需查询 KV，不虚构其需求与 KV 可行性；尾部记预算 HOL。
                # 若首个未处理请求是本轮刚分块的请求（已有 scheduled 决策），归因自动跳过它
                self._attribute_prefill_stop(decisions, decided, reason=REASON_BUDGET,
                                             needed=None, kv_checked=False)
                break
            num_cached_blocks, needed = self._estimate_prefill_tokens(seq)
            if needed is None:
                # KV 容量不足：停止 prefill 接纳（KV 原因，与预算延后区分）
                self._attribute_prefill_stop(decisions, decided, reason=REASON_KV_CAPACITY,
                                             needed=None, kv_checked=True)
                break
            # Day8：计划 chunk q = min(剩余需求, chunk_size, 剩余预算)。
            # chunk_size 与 B 是两条独立上限：首候选允许 q < 需求（拆分）；
            # 后续候选必须整段放下（q == needed），否则停止扫描（FCFS 不跳过，
            # 不绕过它去接纳更短尾部）。chunk_size 造成的部分推进不产生新的
            # 等待原因——被拆分候选本身记 scheduled，放不下的后续候选沿用
            # Day7 的 budget 归因口径
            q = min(needed, self.chunk_size, remaining)
            if q < needed and batch:
                # 非首候选放不下：整请求延后（记录已算出的需求），停止扫描
                self._attribute_prefill_stop(decisions, decided, reason=REASON_BUDGET,
                                             needed=needed, kv_checked=not seq.block_table)
                break
            # 接纳：此刻才分配 block、设置正 token 数并扣减预算
            # kv_checked 在分配前捕获（allocate 会填充 block_table 并设置初始 offset）
            kv_checked = not seq.block_table
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                # 无 block_table 即新 prefill 阶段（新请求 / 抢占恢复重算）的首个
                # chunk：计数从 1 重新开始；续传 chunk 在旧计数上递增
                chunk_index = 1
            else:
                chunk_index = self._prefill_chunk_count.get(seq.seq_id, 0) + 1
            self._prefill_chunk_count[seq.seq_id] = chunk_index
            offset_before = seq.prefill_offset
            seq.num_scheduled_tokens = q
            used += q
            decided.add(seq.seq_id)
            decisions.append(RoundDecision(
                seq.seq_id, seq.request_id, REASON_SCHEDULED,
                needed_tokens=needed, scheduled_tokens=q,
                kv_checked=kv_checked,
                chunk_index=chunk_index,
                offset_before=offset_before,
                is_last_chunk=offset_before + q == seq.prefill_target))
            if offset_before + q == seq.prefill_target:
                # prefill 完成才算被调度接纳：WAITING -> RUNNING 走统一迁移入口
                # （迁移条件与 Day7 一致：offset + q == prefill_target，
                # 即最后一个 chunk 被接纳）；中间 chunk 仍保持 WAITING
                seq.transition_to(SequenceStatus.RUNNING)
                # 扫描位置可能已越过队首（前面驻留中间 chunk 请求），按对象移除
                self.waiting.remove(seq)
                self._enqueue_running(seq)
            else:
                # 中间 chunk：本轮不再扫描该请求，扫描位置前进越过它
                scan += 1
            batch.append(seq)

        if self.waiting and len(batch) >= self.max_num_seqs:
            # 序列数上限先阻止了继续扫描（与预算同时命中时优先记 sequence_cap）：
            # 队首记 sequence_cap，其余记 sequence_cap HOL，不计入预算人数
            self._attribute_prefill_stop(decisions, decided, reason=REASON_SEQUENCE_CAP,
                                         needed=None, kv_checked=False)

        if batch:
            # 本轮已有 prefill：整轮返回 prefill，running 未参与 decode 候选。
            # 未选 running 是 phase_priority，不是 budget；预算余量可以留空。
            # 注意跳过本轮刚从 waiting 转入 running 的批次成员（已记 scheduled）
            for seq in self.running:
                if seq.seq_id in decided:
                    continue
                decided.add(seq.seq_id)
                decisions.append(RoundDecision(
                    seq.seq_id, seq.request_id, REASON_PHASE_PRIORITY))
            self._record_paused_decisions(decisions, decided)
            return self._finalize_round(batch, True, "prefill", decisions, now)

        # ---------- decode：按 running 队列顺序（阶段内 FCFS）选择 ----------
        # 先检查剩余预算与序列数上限，达标前不 popleft / 不 may_append / 不 preempt；
        # 延后请求原地保留 RUNNING、token、KV 与相对顺序
        while self.running:
            if len(batch) >= self.max_num_seqs:
                # 序列数上限先阻止接纳：其余候选延后（限制来自 cap，不能伪记为 budget）
                for seq in self.running:
                    decided.add(seq.seq_id)
                    decisions.append(RoundDecision(
                        seq.seq_id, seq.request_id, REASON_SEQUENCE_CAP))
                break
            if used >= budget:
                # 预算耗尽：decode 候选的单位需求恒为 1（已知），
                # 其余 RUNNING 候选均按直接 budget 记录，不再查询 KV
                for seq in self.running:
                    decided.add(seq.seq_id)
                    decisions.append(RoundDecision(
                        seq.seq_id, seq.request_id, REASON_BUDGET, needed_tokens=1))
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                # 块不足：抢占合法 RUNNING 请求释放资源（KV 原因，不算预算拒绝）。
                # preempt 只做抢占，resume 立即让它以 WAITING 身份回到等待队列
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim, now=now)
                    self.resume(victim)
                    decided.add(victim.seq_id)
                    decisions.append(RoundDecision(
                        victim.seq_id, victim.request_id, REASON_KV_CAPACITY,
                        kv_checked=True))
                else:
                    self.preempt(seq, now=now)
                    self.resume(seq)
                    decided.add(seq.seq_id)
                    decisions.append(RoundDecision(
                        seq.seq_id, seq.request_id, REASON_KV_CAPACITY,
                        needed_tokens=1, kv_checked=True))
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                used += 1
                decided.add(seq.seq_id)
                decisions.append(RoundDecision(
                    seq.seq_id, seq.request_id, REASON_SCHEDULED,
                    needed_tokens=1, scheduled_tokens=1, kv_checked=True))
                batch.append(seq)
        # 沿用既有规则：已选择的 decode 批次恢复至 running 队首；
        # 延后请求保持相对顺序跟在后面，不引入轮转公平策略
        self.running.extendleft(reversed(batch))

        # 独立 PREEMPTED（不在任何队列）本轮不可执行：记 paused，不算预算等待
        self._record_paused_decisions(decisions, decided)

        phase = "decode" if batch else "idle"
        return self._finalize_round(batch, False, phase, decisions, now)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int] | None,
                    is_prefill: bool, now: float | None = None):
        """模型执行返回后的收尾：安全检查 -> 原子提交 chunk 进度 -> 完成判定。

        Day8 提交契约（§4.5）：
        - token 数显式校验：prefill 轮只有"最后 chunk"（offset + q == target）
          的请求产出采样 token，中间 chunk 不采样；decode 轮每序列 1 token。
          run() 对"本轮无任何采样"返回 None，这里与空列表同价。数量不匹配
          显式抛错，不使用会静默截断的 zip。
        - 批次快照一致（先整批校验再逐个提交，避免半提交状态）：活动请求必须
          带有本轮调度设置的待执行计划（num_scheduled_tokens > 0）；计数已
          清零说明本批次是重复/迟到的旧 postprocess，拒绝推进进度。
        - 原子性：每个 chunk 只被成功执行它的这一轮提交一次；取消/超时/
          终态不推进 offset；提交后立即校验单调有界（offset <= target）。
        """
        if now is None:
            now = perf_counter()
        samples = list(token_ids) if token_ids is not None else []
        # 需采样请求集合：必须在任何状态变更前按调度快照计算——
        # prefill 轮 = offset + q == prefill_target（最后 chunk）；
        # decode 轮 = 全部。安全检查丢弃的采样在 token_ids 中仍占位，
        # 游标推进与安全检查解耦，保证对齐关系不因丢弃而错位
        needs_sample = [
            (not is_prefill)
            or (seq.prefill_offset + seq.num_scheduled_tokens == seq.prefill_target)
            for seq in seqs
        ]
        # 全终态批次（纯迟到重放）不校验 token 数：其请求已被安全路径幂等清理，
        # 携带的 token 无意义；Day6/Day7 所有权测试依赖"全终态旧批次重放是
        # 安全空操作"。含活动请求的批次必须与调度计划严格一致
        has_live = any(not seq.is_terminal for seq in seqs)
        if has_live and len(samples) != sum(needs_sample):
            raise ValueError(
                f"postprocess 采样 token 数与需采样请求数不一致：得到 {len(samples)}，"
                f"期望 {sum(needs_sample)}（is_prefill={is_prefill}，批次 {len(seqs)}）")
        for seq in seqs:
            if not seq.is_terminal and seq.num_scheduled_tokens <= 0:
                raise ValueError(
                    f"request {seq.request_id!r} (seq_id={seq.seq_id}) "
                    f"无待执行的调度计划，疑似重复或迟到的 postprocess，已拒绝提交")
        ti = 0  # token_ids 游标
        for seq, need in zip(seqs, needs_sample):
            token_id = samples[ti] if need else None
            if need:
                ti += 1
            # 安全边界（模型执行返回后、追加 token 与 KV 记账之前）：
            # 1) 终态兜底：外部 mark_* 产生的终态请求直接清理，不参与记账；
            # 2) 取消标记优先于超时（客户端显式意图优先于系统判断）；
            # 3) 模型执行期间跨过 deadline 的请求不得被记为正常完成——
            #    丢弃本轮采样 token，按 TIMEOUT 终止并释放资源。
            # 否则请求会在下一轮 check_deadlines 之前以 FINISHED/length 落账，
            # 从活动索引移除后超时就再也无法纠正。
            # 注意：即使输出被丢弃，本轮已执行的实际输入仍按计划快照计入预算，
            # 预算等待统计的终态结算使用本入口的 now（内部嵌套清理不重复取时）。
            # 安全检查命中的请求不推进 offset（失败不提交，§4.5）
            if seq.is_terminal:
                self._finalize(seq, now)
                continue
            if seq.cancel_requested:
                seq.mark_cancelled(seq.cancel_reason or "client_cancelled", now=now)
                self._finalize(seq, now)
                continue
            if seq.deadline is not None and now >= seq.deadline:
                seq.mark_timeout("deadline_exceeded", now=now)
                self._finalize(seq, now)
                continue
            # ---------- Day8 原子提交（成功路径） ----------
            offset_before = seq.prefill_offset
            q = seq.num_scheduled_tokens
            # hash_blocks 接收显式区间 [offset_before, offset_before + q)：
            # 只登记本次执行写满的完整块（中间 chunk 写满的块同样登记，
            # 尾块永不登记）
            self.block_manager.hash_blocks(seq, offset_before, offset_before + q)
            seq.prefill_offset = offset_before + q
            seq.num_scheduled_tokens = 0
            # 单调有界校验（§5.3 不变量 1）：推进后立即核对，违反即显式抛错
            if seq.prefill_offset > seq.prefill_target:
                raise ValueError(
                    f"request {seq.request_id!r}: 提交后 prefill_offset "
                    f"{seq.prefill_offset} 超过目标 {seq.prefill_target}，进度记账被破坏")
            if is_prefill and seq.prefill_offset < seq.prefill_target:
                # 中间 chunk：丢弃采样结果，保持 WAITING，下一轮延续进度
                continue
            # 最后 chunk（或 decode 轮）：追加采样的首 completion；
            # 完成判定：EOS 触发记为 stop，达到 max_tokens 记为 length；
            # mark_finished 幂等，配合 _finalize 保证不 double free
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.eos:
                seq.mark_finished("stop")
                self._finalize(seq, now)
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.mark_finished("length")
                self._finalize(seq, now)
