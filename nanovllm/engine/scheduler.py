import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config, validate_positive_int
from nanovllm.engine.sequence import (InvalidStateTransition, Sequence,
                                      SequenceStatus)
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest

# 模块级 logger：库代码不做 basicConfig，日志开关由调用方（验收脚本/服务层）控制
logger = logging.getLogger(__name__)

# ---------- 本轮决策原因（§5.1 原因分类：不把所有等待都算 budget） ----------
# Day9 混合轮归因重构：phase_priority（"整轮被另一阶段占用"）随 decode-first
# 调度废除——running decode 与 waiting prefill 可同轮共存；新增 decode_priority
# 表达"预算被同轮 decode 优先占用"这一混合轮特有的让路原因
REASON_SCHEDULED = "scheduled"            # 实际接纳正 token 工作（含首请求部分 chunk）
REASON_BUDGET = "budget"                  # 队首/候选需求大于 remaining，或预算已耗尽
REASON_SEQUENCE_CAP = "sequence_cap"      # 序列数上限先阻止接纳（与 budget 同现时优先）
REASON_KV_CAPACITY = "kv_capacity"        # 已查询候选但 KV 容量不足或被 KV 抢占
REASON_HEAD_OF_LINE = "head_of_line"      # 前序请求处停止，尾部未独立检查预算/KV
REASON_DECODE_PRIORITY = "decode_priority"  # 预算被同轮 decode 优先占用（D>0 且需求<=B）
REASON_PAUSED = "paused"                  # 独立 PREEMPTED，尚未恢复


@dataclass
class RoundDecision:
    """单个请求在本调度轮的决策快照（仅 IDs/计数/原因，不含 prompt 或 token 内容）。"""

    seq_id: int
    request_id: str
    reason: str
    # 本请求在本轮的角色：prefill 候选 / decode 候选（Day9 混合轮逐请求阶段标注；
    # paused 请求恢复后按 prefill recompute，同样记 "prefill"）
    phase: str = "prefill"
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
            "phase": self.phase,
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


@dataclass
class BatchItem:
    """单个请求在本调度轮的执行计划（Day9 混合批次的逐请求阶段标注）。

    - 批次是 items 有序列表：decode item 在前、prefill item 在后（与调度顺序、
      执行子批顺序一致），同 phase 内保持各自队列的 FCFS 顺序；
    - 仅 rank 0 侧结构：经 model_runner.call("run", items) 广播（dataclass 默认
      pickle，seq 字段走 Sequence v3 协议），不进入 Sequence.__getstate__ payload，
      pickle 协议版本不变；
    - phase 是批次阶段的单一权威（Sequence.is_prefill 只是请求侧镜像标记）；
    - needs_sample 在调度接纳时按计划快照冻结（decode 恒 True；prefill 等于
      is_last_chunk，即中间 chunk 不采样），postprocess 以它与 token 注入列表
      一一对齐——不再从全批布尔推导，这是混合轮采样对齐的依据；
    - round_id 用于 postprocess 的迟到/重复结果关联校验（双保险之一）。
    """

    seq: Sequence
    phase: str               # "prefill" | "decode"
    scheduled_tokens: int    # prefill 为 q（0 < q <= chunk_size）；decode 恒为 1
    offset_before: int | None = None   # prefill：接纳时刻的 prefill_offset；decode 为 None
    is_last_chunk: bool = False        # prefill：接纳即完成 prefill；decode 恒为 False
    needs_sample: bool = False         # decode 恒 True；prefill 等于 is_last_chunk
    round_id: int = 0                  # 接纳时的调度轮次 ID


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
        # Day10 控制锁（§3.4）：保护控制面状态（cancel_requested/cancel_reason/
        # requests 索引/队列）与资源账本变更的线性化。request_cancel 信号入口与
        # schedule/postprocess 安全点都在锁内执行控制面读写；锁不跨越 GPU forward
        # （ModelRunner 调用发生在 Engine 的两次安全点之间，不持锁），因此外部取消
        # 不会阻塞在 kernel 上，也不会在 kernel 使用对象期间清空 block table。
        # RLock：安全点内部嵌套调用 cancel/timeout/preempt 等入口需要可重入。
        self._control_lock = threading.RLock()
        # Scheduler 默认读取真实单调时钟；测试通过各公开入口的 now 参数
        # 注入确定性时间，避免与外部直接构造的绝对 deadline 发生时间轴冲突。
        self._clock = perf_counter
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 活动请求索引（seq_id -> Sequence）：取消/超时按 id 定位，免于遍历队列。
        # 覆盖 waiting/running 队列成员与被抢占（PREEMPTED）后暂停在队列之外的请求，
        # 生命周期扫描与完成判断都以本索引为权威；终态请求从此处移除
        self.requests: dict[int, Sequence] = {}
        # 活动 request_id 集合：保证外部 ID 在活动期内唯一，取消才能唯一定位。
        # 请求终态（从 requests 移除）后其 ID 允许复用
        self._active_request_ids: set[str] = set()

        # Day11–12 完成记录通道（rank 0 控制面，不进 TP payload）：终态请求在
        # _finalize 捕获只读记录，由 LLMEngine.pop_completed()/pop_aborted()
        # 排出给服务 worker 消费（幂等 drain：pop 后清空，重复调用返回空）。
        # 离线 generate() 不消费也不受影响——记录随 Engine 生命周期废弃。
        # 队列无上限：服务模式 worker 每轮 step 后即消费；离线模式记录量以
        # 历史请求数为上界，且只保留 completion token 计数级的小对象。
        self.completed_records: deque[CompletedRequest] = deque()
        self.aborted_records: deque[AbortedRequest] = deque()

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
        with self._control_lock:
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

        活动索引的对象身份是唯一所有权凭证。旧批次或其他 Scheduler 的对象
        不得借用同一 seq_id 触碰当前队列和 KV 账本；真正 owner 的重复收尾
        仍由 block_table 清空保证幂等。
        """
        if self.requests.get(seq.seq_id) is not seq:
            return
        # Day11–12：在终态迁移与资源清理之间捕获只读终态记录（§4.4），
        # 保证 request_id、finish_reason 和 token 计数与终态同源。所有权检查
        # 已通过，意味着这是该 seq 的首次（唯一一次）收尾，不会重复记录；
        # 后续对同一 seq 的幂等重复收尾在上方即被所有权检查拦截。
        # 异常路径（abort_all_active）产生的 CANCELLED 记录同样进入通道，
        # 服务层按 finish_reason 映射为失败结果，不伪装成正常 completion。
        if seq.status is SequenceStatus.FINISHED:
            self.completed_records.append(CompletedRequest(
                seq_id=seq.seq_id, request_id=seq.request_id,
                completion_token_ids=tuple(seq.completion_token_ids),
                prompt_tokens=seq.num_prompt_tokens,
                completion_tokens=seq.num_completion_tokens,
                finish_reason=seq.finish_reason or "stop",
                finished_at=seq.finished_at if seq.finished_at is not None
                else (now if now is not None else self._clock()),
            ))
        elif seq.status in (SequenceStatus.CANCELLED, SequenceStatus.TIMEOUT):
            self.aborted_records.append(AbortedRequest(
                seq_id=seq.seq_id, request_id=seq.request_id,
                finish_reason=seq.finish_reason or seq.status.name.lower(),
                finished_at=seq.finished_at if seq.finished_at is not None
                else (now if now is not None else self._clock()),
            ))
        self._remove_from_queues(seq)
        self._release_sequence(seq)
        del self.requests[seq.seq_id]
        self._active_request_ids.discard(seq.request_id)
        # Day8 chunk 计数随请求终态回收（seq_id 全局唯一，记录不会误伤新请求）
        self._prefill_chunk_count.pop(seq.seq_id, None)
        # 终态预算收尾：结算未关闭 episode、发终态等待摘要并删除记录
        self._settle_terminal_budget_stats(seq, now)

    # ---------- 控制入口：取消 / 超时 / 抢占 / 恢复（Day10 §3.4/§4.2） ----------
    #
    # 两层取消语义：
    # - request_cancel() 是控制面信号：只在锁内置位标记，允许在 ModelRunner
    #   forward 期间被外部线程调用，不触碰 block/队列/token；
    # - cancel() 是安全点收尾：要求调度器不在执行当前模型批次，执行终态迁移、
    #   队列移除、KV 释放和活动索引删除。Engine 对外取消入口优先使用信号语义。

    def request_cancel(self, seq_id: int, reason: str = "client_cancelled") -> bool:
        """控制面取消信号：只在锁内置位 cancel_requested 与首次 cancel_reason。

        - 找不到活动请求或已终态：返回 False；
        - 重复调用返回 True，但保持首次记录的原因（Sequence.request_cancel）；
        - 不迁移状态、不动队列、不释放 block、不改 token，因此可以安全地
          在模型执行期间调用；真正收尾发生在调度边界/postprocess 安全点。
        """
        with self._control_lock:
            seq = self.requests.get(seq_id)
            if seq is None or seq.is_terminal:
                return False
            return seq.request_cancel(reason)

    def request_cancel_by_request_id(self, request_id: str,
                                     reason: str = "client_cancelled") -> bool:
        """按 request_id 原子地发出取消信号，消除查找后再按 seq_id 操作的竞态。"""
        with self._control_lock:
            seq = next((item for item in self.requests.values()
                        if item.request_id == request_id), None)
            if seq is None or seq.is_terminal:
                return False
            return seq.request_cancel(reason)

    def get_request(self, request_id: str) -> Sequence | None:
        """在控制锁内按 request_id 读取活动请求。"""
        with self._control_lock:
            return next((seq for seq in self.requests.values()
                         if seq.request_id == request_id), None)

    def _log_control_event(self, action: str, seq: Sequence, from_status: SequenceStatus,
                           reason: str, released_blocks: int, now: float,
                           *, free_before: int | None = None,
                           used_before: int | None = None):
        """发出可独立重算的 request_control 事件。

        before/after 账本快照让验收脚本可以验证 ``released_blocks`` 的真实差值，
        而不是只相信动作代码传入的计划值；事件仍不含 prompt 或 token 明文。
        """
        free_after = len(self.block_manager.free_block_ids)
        used_after = len(self.block_manager.used_block_ids)
        if free_before is None:
            free_before = free_after
        if used_before is None:
            used_before = used_after
        _log_event({
            "event": "request_control",
            "round_id": self._current_round_id,
            "seq_id": seq.seq_id,
            "request_id": seq.request_id,
            "action": action,
            "from_status": from_status.name,
            "to_status": seq.status.name,
            "reason": reason,
            "num_preempts": seq.num_preempts,
            "released_blocks": released_blocks,
            "free_blocks_before": free_before,
            "used_blocks_before": used_before,
            "free_blocks": free_after,
            "used_blocks": used_after,
            "observed_at": now,
        })

    def cancel(self, seq_id: int, reason: str = "client_cancelled",
               now: float | None = None) -> bool:
        """安全点取消收尾：设置终态、移出队列并释放资源（幂等）。

        - 请求不存在返回 False；请求已是终态时返回 False，
          但仍会幂等完成剩余清理（终态对象被外部 mark_* 留在队列/索引中时，
          不能因控制入口"重复调用"而遗留未释放资源）；
        - 首次记录的取消原因不被后续重复取消覆盖；
        - 与 TIMEOUT 的区分保留在 finish_reason 中，便于指标统计；
        - 终态时刻即预算 episode 结算时刻（显式 now 优先，否则读同一单调时钟）。
        每次调用发出 request_control 事件（含终态重放的幂等调用，便于审计
        重复取消不覆盖原因）。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            seq = self.requests.get(seq_id)
            if seq is None:
                return False
            from_status = seq.status
            if seq.is_terminal:
                free_before = len(self.block_manager.free_block_ids)
                used_before = len(self.block_manager.used_block_ids)
                self._finalize(seq, now)
                self._log_control_event(
                    "cancel", seq, from_status, seq.finish_reason or reason,
                    used_before - len(self.block_manager.used_block_ids), now,
                    free_before=free_before, used_before=used_before)
                return False
            # 先打标记（原因只记录首次），再以记录下来的原因做终态迁移
            seq.request_cancel(reason)
            seq.mark_cancelled(seq.cancel_reason or reason, now=now)
            free_before = len(self.block_manager.free_block_ids)
            used_before = len(self.block_manager.used_block_ids)
            self._finalize(seq, now)
            self._log_control_event(
                "cancel", seq, from_status, seq.cancel_reason or reason,
                used_before - len(self.block_manager.used_block_ids), now,
                free_before=free_before, used_before=used_before)
            return True

    def timeout(self, seq_id: int, now: float | None = None,
                reason: str = "deadline_exceeded") -> bool:
        """安全点超时收尾：已有取消信号优先于 deadline。"""
        if now is None:
            now = self._clock()
        with self._control_lock:
            seq = self.requests.get(seq_id)
            if seq is None:
                return False
            # 取消和超时同时到达时，客户端显式意图优先；统一转入 cancel
            # 入口，避免直接 mark_timeout 绕过优先级契约。
            if seq.cancel_requested and not seq.is_terminal:
                return self.cancel(seq_id, reason=seq.cancel_reason or reason, now=now)
            from_status = seq.status
            free_before = len(self.block_manager.free_block_ids)
            used_before = len(self.block_manager.used_block_ids)
            if seq.is_terminal:
                self._finalize(seq, now)
                self._log_control_event(
                    "timeout", seq, from_status, seq.finish_reason or reason,
                    used_before - len(self.block_manager.used_block_ids), now,
                    free_before=free_before, used_before=used_before)
                return False
            seq.mark_timeout(reason, now=now)
            self._finalize(seq, now)
            self._log_control_event(
                "timeout", seq, from_status, seq.finish_reason or reason,
                used_before - len(self.block_manager.used_block_ids), now,
                free_before=free_before, used_before=used_before)
            return True

    def check_deadlines(self, now: float | None = None) -> list[Sequence]:
        """在调度边界统一执行 deadline 检查，返回本轮超时的请求。

        以活动索引为扫描范围（覆盖队列成员与暂停请求），使用单调时钟；
        带取消标记的请求跳过——取消优先于超时（客户端显式意图优先），
        由 _purge_cancelled 按 CANCELLED 处理。同一轮只取一次时钟，
        便于确定性测试（注入 now）。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
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
        仅在已持锁的调度路径内调用。
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
        仅在已持锁的调度路径内调用。
        """
        finalized = []
        for seq in list(self.requests.values()):
            if seq.is_terminal:
                self._finalize(seq, now)
                finalized.append(seq)
        return finalized

    # ---------- Day10 抢占：victim 选择与恢复事务（§4.2.2/§5.3） ----------

    def _pick_victim(self, exclude_seq_ids: set[int],
                     tried_seq_ids: set[int]) -> Sequence | None:
        """按固定策略选择抢占 victim：running 队尾优先，向队首反向搜索。

        合法 victim 条件（§4.2.2 规则 1 的三条排除）：
        - 状态为 RUNNING（终态/等待/暂停对象不可抢占）且仍在 running 队列；
        - 不是本轮已接纳的 BatchItem 对象（exclude_seq_ids，含同轮已接纳的
          decode item 与最后 chunk prefill item）；
        - 无本轮执行计划（num_scheduled_tokens == 0，不是正在执行的对象）。
        tried_seq_ids 排除本轮已尝试过的候选：尝试次数天然有界（<= len(running)），
        不会无界循环。没有合法候选返回 None，由调用方延后/停止，不破坏队列。
        """
        for victim in reversed(self.running):
            if victim.seq_id in exclude_seq_ids or victim.seq_id in tried_seq_ids:
                continue
            if victim.status is not SequenceStatus.RUNNING or victim.is_terminal:
                continue
            if victim.num_scheduled_tokens > 0:
                continue
            return victim
        return None

    def _preempt_victim(self, victim: Sequence, decided: set[int],
                        decisions: list[RoundDecision], now: float, phase: str):
        """自动抢占事务：先完成释放（PREEMPTED + 移出队列 + 释放 block），
        再恢复为 WAITING 插入 waiting 队首（同轮自动路径，§3.3 规则）。
        事件中保留一次抢占记录；victim 以 prefill recompute 身份重新排队。"""
        self.preempt(victim, now=now)
        self.resume(victim, now=now)
        decided.add(victim.seq_id)
        decisions.append(RoundDecision(
            victim.seq_id, victim.request_id, REASON_KV_CAPACITY,
            kv_checked=True, phase=phase))

    def preempt(self, seq: Sequence, now: float | None = None):
        """抢占：RUNNING -> PREEMPTED，释放 KV block（不回滚已生成 token）。

        只完成抢占本身，不负责重新入队；恢复由 resume() 显式执行。
        暂停中的请求仍保留在活动索引里，参与生命周期扫描与完成判断。
        恢复采用 recompute 方案：物理块释放，token 进度保留，恢复时重新 prefill。
        显式抢占同时结束该请求的预算等待 episode：此后等待原因为 paused/KV，
        不应继续计入预算等待。
        非法抢占（对 WAITING/终态对象重复调用）经统一迁移入口显式抛异常，
        不产生任何队列/资源副作用。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            if self.requests.get(seq.seq_id) is not seq:
                raise ValueError(
                    f"只能抢占当前 Scheduler 活动请求: "
                    f"request_id={getattr(seq, 'request_id', None)!r}")
            if seq not in self.running:
                # 已注册对象但状态/队列不满足抢占前置条件，交给状态机统一
                # 抛出显式非法迁移，保持既有调用方的错误契约。
                raise InvalidStateTransition(
                    seq, seq.status, SequenceStatus.PREEMPTED, "request not in running")
            from_status = seq.status
            free_before = len(self.block_manager.free_block_ids)
            used_before = len(self.block_manager.used_block_ids)
            # 迁移时间与注入时钟统一（不变量 11）：last_preempted_at 使用
            # 调用方注入的 now，与事件 observed_at 同源
            seq.transition_to(SequenceStatus.PREEMPTED, now=now)
            seq.is_prefill = True
            self._remove_from_queues(seq)
            self._release_sequence(seq)
            self._update_budget_wait(seq, deferred=False, now=now,
                                     round_id=self._current_round_id, close_reason="preempted")
            self._log_control_event(
                "preempt", seq, from_status, "kv_capacity",
                used_before - len(self.block_manager.used_block_ids), now,
                free_before=free_before, used_before=used_before)

    def resume(self, seq: Sequence, now: float | None = None):
        """恢复被抢占请求：只允许 PREEMPTED -> WAITING，重新入队等待 recompute。

        front=True 使被抢占请求排在等待队列头部（沿用原抢占行为的优先级）。
        恢复不修改逻辑 token 进度（prompt/completion/采样参数），
        有效 KV 进度保持抢占时归零的状态，由重新 prefill 重建。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            if self.requests.get(seq.seq_id) is not seq:
                raise ValueError(
                    f"只能恢复当前 Scheduler 活动索引中的请求: "
                    f"request_id={getattr(seq, 'request_id', None)!r}")
            if seq.status is not SequenceStatus.PREEMPTED:
                # 先让状态机报告重复/非法恢复，保持既有异常契约；该调用
                # 对非法迁移不会产生任何状态或资源副作用。
                seq.transition_to(SequenceStatus.WAITING, now=now)
            if seq in self.waiting or seq in self.running:
                raise ValueError(
                    f"恢复请求已存在于工作队列，拒绝制造重复成员: "
                    f"request_id={getattr(seq, 'request_id', None)!r}")
            from_status = seq.status
            seq.transition_to(SequenceStatus.WAITING, now=now)
            seq.is_prefill = True
            self._enqueue_waiting(seq, front=True)
            self._log_control_event("resume", seq, from_status, "recompute_resume", 0, now)

    # ---------- Day10 异常收尾（§4.2.3/§4.3） ----------

    def _resource_snapshot(self) -> dict:
        """资源快照：活动请求/队列/账本计数，供异常收尾日志与验收交叉核对。"""
        return {
            "active_requests": len(self.requests),
            "waiting": len(self.waiting),
            "running": len(self.running),
            "free_blocks": len(self.block_manager.free_block_ids),
            "used_blocks": len(self.block_manager.used_block_ids),
        }

    def _abort_one(self, seq: Sequence, reason: str, now: float):
        """单个对象的异常收尾（幂等，可重入）：活动则迁移 CANCELLED 再统一 _finalize。

        所有权保护：仅当对象仍登记在活动索引中才做终态迁移——旧批次迟到收尾
        不能把复用了同一 seq_id 的新请求误标记为 CANCELLED；_finalize 自带
        所有权检查，对非所有者是安全空操作。已终态对象不覆盖既有 finish_reason。
        """
        if self.requests.get(seq.seq_id) is not seq:
            return
        from_status = seq.status
        if not seq.is_terminal:
            seq.mark_cancelled(reason, now=now)
        free_before = len(self.block_manager.free_block_ids)
        used_before = len(self.block_manager.used_block_ids)
        self._finalize(seq, now)
        self._log_control_event(
            "abort", seq, from_status, reason,
            used_before - len(self.block_manager.used_block_ids), now,
            free_before=free_before, used_before=used_before)

    def abort_round(self, items, reason: str = "engine_error",
                    now: float | None = None) -> dict:
        """批次异常收尾（§4.2.3）：清除本轮计划并把本轮对象收敛到终态。

        - 仍活动且归属当前索引的 item：取消其本轮计划（num_scheduled_tokens
          清零，不伪造为已执行），迁移 CANCELLED（finish_reason=reason，如
          engine_error/execution_error），释放其全部 block；
        - 已终态对象：只执行幂等 _finalize，不覆盖既有原因；
        - 非所有者（旧批次迟到对象）：_finalize 为安全空操作，不影响新对象；
        - 返回收尾后的资源快照，供错误日志与测试核对。
        异常清理可重复调用，不 double free、不重复删除索引（不变量 9）。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            errors = []
            for it in items:
                try:
                    self._abort_one(it.seq, reason, now)
                except BaseException as exc:
                    errors.append(exc)
            snapshot = self._resource_snapshot()
            if errors:
                raise RuntimeError(
                    f"批次异常收尾失败 {len(errors)} 项，资源快照={snapshot}") from errors[0]
            return snapshot

    def abort_all_active(self, reason: str = "engine_error",
                         now: float | None = None) -> dict:
        """全活动请求异常收尾：Engine 模型执行异常后不可重试，等待/运行/暂停
        中的请求都不能留下"看似可继续"的活动对象（§4.3 step 事务第 4 步）。

        与 abort_round 相同的迁移与释放规则；返回收尾后的资源快照。
        可重复调用（幂等）。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            errors = []
            # 快照保证清理期间即使某个对象异常，也继续处理其余活动请求。
            for seq in list(self.requests.values()):
                try:
                    self._abort_one(seq, reason, now)
                except BaseException as exc:
                    errors.append(exc)
            snapshot = self._resource_snapshot()
            if errors:
                raise RuntimeError(
                    f"活动请求异常收尾失败 {len(errors)} 项，资源快照={snapshot}") from errors[0]
            return snapshot

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
            now = self._clock()
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
        直接原因可能是 decode_priority（Day9：预算被同轮 decode 优先占用），
        尾部 HOL 的 blocking_reason 原样跟随，保证归因可独立复算。
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
            needed_tokens=needed, kv_checked=kv_checked, phase="prefill"))
        for seq in list(self.waiting):
            if seq.seq_id in decided:
                continue
            decided.add(seq.seq_id)
            decisions.append(RoundDecision(
                seq.seq_id, seq.request_id, REASON_HEAD_OF_LINE,
                blocking_reason=reason, blocked_by_seq_id=first.seq_id,
                phase="prefill"))

    def _record_budget_decisions(self, decisions: list[RoundDecision], now: float,
                                 round_id: int) -> tuple[int, int]:
        """按本轮决策快照更新预算等待统计，返回 (direct, hol) 预算延后人数。

        - direct：主原因即为 budget 的请求数；
        - hol：主原因为 head_of_line 且 blocking_reason=budget 的请求数；
        - 两者之和（本轮天然去重）即 budget_deferred_requests；
        - KV/sequence_cap/decode_priority/paused 不计入预算人数
          （decode_priority 是优先级策略的让路，不是预算的锅，§3.5）。
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
                    # paused 请求恢复后按 prefill recompute 重新入队，角色记 "prefill"
                    seq.seq_id, seq.request_id, REASON_PAUSED, phase="prefill"))

    def _finalize_round(self, items: list[BatchItem], phase: str,
                        decisions: list[RoundDecision], now: float,
                        needed_first: int | None = None) -> tuple[list[BatchItem], str]:
        """轮末收尾：以 items 实际字段重算校验 -> 更新统计 -> 记录计划日志。

        Day9 混合预算口径（§3.3/§5.2 不变量 14）：
            planned = sum(prefill q_i) + count(decode items) <= B
        四条上限独立显式校验（互不替代）：q_i 正整数、prefill q_i <= chunk_size、
        总量 <= B、条数 <= max_num_seqs；另校验 decode item 恒为 1、seq_id 去重
        （同一请求每轮至多一个 item，不允许同轮同时 prefill+decode）。
        needed_first 记入事件供验收方独立复算 decode_priority 判定条件（§3.5）。
        """
        round_id = self._current_round_id
        # 每轮末以 items 的实际字段重算，不只信任循环局部变量（§4.2）
        prefill_items = [it for it in items if it.phase == "prefill"]
        decode_items = [it for it in items if it.phase == "decode"]
        planned = sum(it.scheduled_tokens for it in prefill_items) + len(decode_items)
        if items:
            # 阶段合法性：BatchItem.phase 是批次阶段权威，未知值在此显式拒绝
            if any(it.phase not in ("prefill", "decode") for it in items):
                raise ValueError(
                    f"round {round_id}: 批次存在未知 phase 的 item: "
                    f"{sorted({it.phase for it in items} - {'prefill', 'decode'})}")
            if any(type(it.scheduled_tokens) is not int
                   or it.scheduled_tokens <= 0 for it in items):
                raise ValueError(
                    f"round {round_id}: num_scheduled_tokens 必须为正整数")
            if planned <= 0:
                raise ValueError(
                    f"round {round_id}: 非空批次计划总量为 {planned}")
            # decode item 的单位约束：每条每轮恒为 1 token（§3.3）
            bad_decode = [it.seq.seq_id for it in decode_items
                          if it.scheduled_tokens != 1]
            if bad_decode:
                raise ValueError(
                    f"round {round_id}: decode item 的接纳数必须为 1，"
                    f"违规序列 {bad_decode}")
            # 混合预算：decode 占用量 + prefill query 总量不得超过 B
            if planned > self.max_num_batched_tokens:
                raise ValueError(
                    f"round {round_id}: 计划 token {planned} 超过预算 "
                    f"{self.max_num_batched_tokens}")
            if len(items) > self.max_num_seqs:
                raise ValueError(
                    f"round {round_id}: 批次条目数 {len(items)} 超过上限 "
                    f"{self.max_num_seqs}")
            # Day8：chunk_size 上限与预算/序列数独立校验（decode q=1 天然满足，
            # 显式只查 prefill item）
            over_chunk = [it.seq.seq_id for it in prefill_items
                          if it.scheduled_tokens > self.chunk_size]
            if over_chunk:
                raise ValueError(
                    f"round {round_id}: 序列 {over_chunk} 的单轮接纳数超过 "
                    f"chunk_size={self.chunk_size}")
            seq_ids = [it.seq.seq_id for it in items]
            if len(set(seq_ids)) != len(seq_ids):
                raise ValueError(f"round {round_id}: 批次内 seq_id 重复: {seq_ids}")
            # 计划快照与 seq 实际字段一致性：BatchItem 是调度产物，二者不允许漂移
            mismatched = [it.seq.seq_id for it in items
                          if it.seq.num_scheduled_tokens != it.scheduled_tokens]
            if mismatched:
                raise ValueError(
                    f"round {round_id}: BatchItem 计划量与 seq.num_scheduled_tokens "
                    f"不一致: {mismatched}")
        direct, hol = self._record_budget_decisions(decisions, now, round_id)
        stats = {
            "event": "scheduler_round",
            "round_id": round_id,
            "phase": phase,
            "token_budget": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "planned_tokens": planned,
            # Day9 分阶段计划量：decode_tokens 即 decode 条数（每条 1 token）
            "prefill_tokens": sum(it.scheduled_tokens for it in prefill_items),
            "decode_tokens": len(decode_items),
            "prefill_items": len(prefill_items),
            "decode_items": len(decode_items),
            "scheduled_requests": len(items),
            # Day9 归因依据：本轮首个被考察的 prefill 候选的需求（不变量 18 的
            # needed_first），供 decode_priority 判定条件独立复算；无成功估算时为 None
            "needed_first": needed_first,
            "budget_deferred_direct": direct,
            "budget_deferred_hol": hol,
            "budget_deferred_requests": direct + hol,
            "observed_at": now,
            "decisions": [d.to_dict() for d in decisions],
        }
        self.last_schedule_stats = stats
        _log_event(stats)
        self._current_round_id = None
        return items, phase

    # ---------- 调度主流程 ----------

    def _schedule_decode_phase(self, *, round_id: int, budget: int, used: int,
                               decisions: list[RoundDecision], decided: set[int],
                               now: float) -> tuple[list[BatchItem], int]:
        """decode 阶段（decode-first，§3.4 阶段 1）：先占预算与名额。

        按 running 队列顺序（阶段内 FCFS）选择；先检查剩余预算与序列数上限，
        达标前不 popleft / 不 may_append / 不抢占；延后请求原地保留 RUNNING、
        token、KV 与相对顺序。返回 (decode items, 更新后的 used)。
        decisions/decided 就地追加（与 _attribute_prefill_stop 同一风格）。
        """
        items: list[BatchItem] = []
        while self.running:
            if len(items) >= self.max_num_seqs:
                # 序列数上限先阻止接纳：其余候选延后（限制来自 cap，不能伪记为 budget）
                for seq in self.running:
                    decided.add(seq.seq_id)
                    decisions.append(RoundDecision(
                        seq.seq_id, seq.request_id, REASON_SEQUENCE_CAP, phase="decode"))
                break
            if used >= budget:
                # 预算耗尽：decode 候选的单位需求恒为 1（已知），
                # 其余 RUNNING 候选均按直接 budget 记录，不再查询 KV
                for seq in self.running:
                    decided.add(seq.seq_id)
                    decisions.append(RoundDecision(
                        seq.seq_id, seq.request_id, REASON_BUDGET,
                        needed_tokens=1, phase="decode"))
                break
            seq = self.running.popleft()
            # Day10 §4.2.2：块不足时按"队尾优先、反向搜索、有限尝试"选择合法
            # victim 抢占让块；victim 排除本轮已接纳 item（admitted_ids，含已
            # 接纳的 decode item 与最后 chunk prefill item）、终态对象和已有
            # 本轮执行计划的对象。抢占释放后重新查询 can_append，仍不足则
            # 继续选下一个候选（tried 保证有界），无合法 victim 时延后候选。
            admitted_ids = {it.seq.seq_id for it in items}
            tried: set[int] = set()
            deferred = False
            while not self.block_manager.can_append(seq):
                victim = self._pick_victim(admitted_ids, tried)
                if victim is None:
                    # 无合法 victim（§4.2.2 规则 5）：延后候选，不破坏队列——
                    # 候选保留 RUNNING/KV 回到队首原位，下一轮容量允许时直接
                    # 继续 decode。不做 Day9 的"自抢占"：自抢占不释放任何新
                    # 容量（自己释放又自己重算），只会白白损失有效 KV 进度。
                    deferred = True
                    break
                self._preempt_victim(victim, decided, decisions, now, phase="decode")
            if deferred:
                self.running.appendleft(seq)
                decided.add(seq.seq_id)
                decisions.append(RoundDecision(
                    seq.seq_id, seq.request_id, REASON_KV_CAPACITY,
                    needed_tokens=1, kv_checked=True, phase="decode"))
                # FCFS 停止：队首无法推进，其余未考察成员按 HOL 归因
                # （它们未被独立检查，归因跟随队首的 kv_capacity）
                for blocked in list(self.running):
                    if blocked.seq_id in decided:
                        continue
                    decided.add(blocked.seq_id)
                    decisions.append(RoundDecision(
                        blocked.seq_id, blocked.request_id, REASON_HEAD_OF_LINE,
                        blocking_reason=REASON_KV_CAPACITY,
                        blocked_by_seq_id=seq.seq_id, phase="decode"))
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            used += 1
            decided.add(seq.seq_id)
            decisions.append(RoundDecision(
                seq.seq_id, seq.request_id, REASON_SCHEDULED,
                needed_tokens=1, scheduled_tokens=1, kv_checked=True,
                phase="decode"))
            items.append(BatchItem(
                seq=seq, phase="decode", scheduled_tokens=1,
                offset_before=None, is_last_chunk=False,
                needs_sample=True, round_id=round_id))
        # 沿用既有规则：已选择的 decode 批次恢复至 running 队首；
        # 延后请求保持相对顺序跟在后面，不引入轮转公平策略
        self.running.extendleft(reversed([it.seq for it in items]))
        return items, used

    def schedule(self, *, now: float | None = None) -> tuple[list[BatchItem], str]:
        """制定一轮调度计划，返回 (items, phase)（Day9 混合批次接口）。

        混合预算口径（§3.3）：
            planned_tokens = sum(prefill q_i) + count(decode items) <= B
        且非空批次每条 num_scheduled_tokens 为正整数；decode item 恒为 1。

        调度策略 decode-first（§3.4）：同一轮内 decode 先于 prefill 分配预算与
        序列名额——已有 decode 请求的推进不被长 prefill 独占 GPU 挤掉（结构性
        保底，无需保留比例配置）；prefill 使用 decode 之后的剩余预算，接纳规则
        （首候选拆分/后续整段/FCFS 不跳过/扫描位置与队列分离）沿用 Day8。
        phase ∈ {"prefill", "decode", "mixed", "idle"}，纯阶段是混合的退化特例。
        归因（§3.5/§5.2 不变量 18）：prefill 候选延后时，decode_priority 仅当
        "D > 0 且 needed_first <= B"——needed_first 是本轮首个被考察的 prefill
        候选的需求（含未被接纳即停止的情形），不是被延后候选自己的需求。
        时间语义：整轮使用一次单调时钟 now（可注入确定值用于测试）。

        控制锁（Day10 §3.4）：整轮调度是纯 CPU 记账的安全点，全程持锁保证
        与外部 request_cancel 信号及账本变更线性化；锁不跨越 GPU forward。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            return self._schedule(now=now)

    def _schedule(self, *, now: float) -> tuple[list[BatchItem], str]:
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
        items: list[BatchItem] = []
        used = 0  # U：本轮已承诺的输入 token 数（decode + prefill 同一口径）
        decisions: list[RoundDecision] = []
        decided: set[int] = set()

        # ---------- 阶段 1：decode（decode-first，先占预算与名额，§3.4） ----------
        decode_items, used = self._schedule_decode_phase(
            round_id=round_id, budget=budget, used=used,
            decisions=decisions, decided=decided, now=now)
        items.extend(decode_items)

        # ---------- 阶段 2：prefill（使用 decode 之后的剩余预算与名额） ----------
        # decode_used 是 decode_priority 归因的判据：本轮 decode 实际占用量
        decode_used = used
        prefill_admitted = False  # prefill 阶段是否已有接纳（首候选才允许拆分）
        # needed_first（§3.5/不变量 18）：本轮首个被考察的 prefill 候选的需求。
        # 在第一次成功估算时记录（含后续未接纳即停止的情形）；为 None 表示本轮
        # 尚无成功估算（首个候选即 KV 不足停止，此时不会有 budget/decode_priority
        # 归因发生）。decode_priority 判定统一使用该值，与设计字面规则一致
        needed_first: int | None = None
        # Day8：扫描位置与队列分离——中间 chunk 的请求保持 WAITING 原地
        # （保留 block_table 与已提交进度），但本轮不再被扫描
        # （每请求每轮至多一个 chunk）；扫描位置前进，后续请求仍按 FCFS 考察
        scan = 0
        while self.waiting and len(items) < self.max_num_seqs and scan < len(self.waiting):
            seq = self.waiting[scan]
            remaining = budget - used
            if remaining == 0:
                # 预算恰好耗尽（未命中序列数上限）：首个未处理请求归因。
                # D==0 时沿用 Day7 口径：budget、不查询 KV、不虚构需求；
                # D>0 时按不变量 18 判定：needed_first <= B 记 decode_priority
                #（预算短缺纯由 decode 优先造成），否则记 budget（首候选自身
                # 规模已超预算，延后并非 decode 造成）。
                # 若该候选本身就是本轮首个被考察的候选（尚无估算），此刻补一次
                # 只读估算，其需求如实入档；若 needed_first 已来自更早的候选，
                # 本候选未做估算，needed 记 None（不虚构需求）。KV 不足仍记
                # kv_capacity
                if decode_used > 0:
                    estimated_here = needed_first is None
                    if estimated_here:
                        _, needed_first = self._estimate_prefill_tokens(seq)
                        if needed_first is None:
                            self._attribute_prefill_stop(decisions, decided,
                                                         reason=REASON_KV_CAPACITY,
                                                         needed=None, kv_checked=True)
                            break
                    reason = (REASON_DECODE_PRIORITY if needed_first <= budget
                              else REASON_BUDGET)
                    self._attribute_prefill_stop(
                        decisions, decided, reason=reason,
                        needed=needed_first if estimated_here else None,
                        kv_checked=estimated_here and not seq.block_table)
                    break
                self._attribute_prefill_stop(decisions, decided, reason=REASON_BUDGET,
                                             needed=None, kv_checked=False)
                break
            num_cached_blocks, needed = self._estimate_prefill_tokens(seq)
            if needed is None:
                # KV 容量不足（§4.2.2 规则 5/§5.3）：先按固定策略抢占 running
                # victim 释放容量，再对同一候选重新查询 can_allocate（"C 重新
                # 查询容量并继续调度"）。victim 恢复为 WAITING 并插入 waiting
                # 队首（优先重算），队首插入使扫描索引整体后移一位，因此每次
                # 插入后 scan += 1 补偿，保证后续迭代仍指向同一候选及其后继。
                # tried 集合保证尝试次数有界（<= len(running)），不无界循环。
                admitted_ids = {it.seq.seq_id for it in items}
                tried: set[int] = set()
                while needed is None:
                    victim = self._pick_victim(admitted_ids, tried)
                    if victim is None:
                        break
                    tried.add(victim.seq_id)
                    self._preempt_victim(victim, decided, decisions, now, phase="decode")
                    scan += 1  # waiting 队首插入的扫描索引补偿
                    num_cached_blocks, needed = self._estimate_prefill_tokens(seq)
            if needed is None:
                # 无合法 victim 或释放后仍不足：停止 prefill 接纳（KV 原因，
                # 与预算延后区分），不得破坏队列
                self._attribute_prefill_stop(decisions, decided, reason=REASON_KV_CAPACITY,
                                             needed=None, kv_checked=True)
                break
            if needed_first is None:
                needed_first = needed
            # Day8：计划 chunk q = min(剩余需求, chunk_size, 剩余预算)。
            # chunk_size 与 B 是两条独立上限：首候选允许 q < 需求（拆分）；
            # 后续候选必须整段放下（q == needed），否则停止扫描（FCFS 不跳过，
            # 不绕过它去接纳更短尾部）。chunk_size 造成的部分推进不产生新的
            # 等待原因——被拆分候选本身记 scheduled；放不下的后续候选按
            # 不变量 18 归因（decode_priority 仅当 D>0 且 needed_first<=B）
            q = min(needed, self.chunk_size, remaining)
            if q < needed and prefill_admitted:
                # 非首候选放不下：整请求延后（记录该候选自己的需求），停止扫描
                reason = (REASON_DECODE_PRIORITY
                          if (decode_used > 0 and needed_first is not None
                              and needed_first <= budget) else REASON_BUDGET)
                self._attribute_prefill_stop(decisions, decided, reason=reason,
                                             needed=needed,
                                             kv_checked=not seq.block_table)
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
            seq.is_prefill = True
            used += q
            prefill_admitted = True
            decided.add(seq.seq_id)
            # "本轮接纳即完成 prefill"判定改用 Sequence 具名谓词（Day9 单一权威）
            is_last = seq.is_last_chunk_scheduled
            decisions.append(RoundDecision(
                seq.seq_id, seq.request_id, REASON_SCHEDULED,
                needed_tokens=needed, scheduled_tokens=q,
                kv_checked=kv_checked,
                chunk_index=chunk_index,
                offset_before=offset_before,
                is_last_chunk=is_last, phase="prefill"))
            items.append(BatchItem(
                seq=seq, phase="prefill", scheduled_tokens=q,
                offset_before=offset_before, is_last_chunk=is_last,
                needs_sample=is_last, round_id=round_id))
            if is_last:
                # prefill 完成才算被调度接纳：WAITING -> RUNNING 走统一迁移入口
                # （迁移条件与 Day7/8 一致：最后一个 chunk 被接纳）；中间 chunk
                # 仍保持 WAITING。本轮不再有该请求的 decode item——其首个
                # completion token 由最后 chunk 的采样产生（§3.4 设计要点 3）
                seq.transition_to(SequenceStatus.RUNNING)
                # 扫描位置可能已越过队首（前面驻留中间 chunk 请求），按对象移除
                self.waiting.remove(seq)
                self._enqueue_running(seq)
            else:
                # 中间 chunk：本轮不再扫描该请求，扫描位置前进越过它
                scan += 1

        if self.waiting and len(items) >= self.max_num_seqs:
            # 序列数上限先阻止了继续扫描（与预算同时命中时优先记 sequence_cap）：
            # 队首记 sequence_cap，其余记 sequence_cap HOL，不计入预算人数
            self._attribute_prefill_stop(decisions, decided, reason=REASON_SEQUENCE_CAP,
                                         needed=None, kv_checked=False)

        # 独立 PREEMPTED（不在任何队列）本轮不可执行：记 paused，不算预算等待
        self._record_paused_decisions(decisions, decided)

        # 轮级 phase 标签：混合轮 = decode 与 prefill item 同时存在（§3.1）
        has_prefill = any(it.phase == "prefill" for it in items)
        has_decode = any(it.phase == "decode" for it in items)
        if has_prefill and has_decode:
            phase = "mixed"
        elif has_prefill:
            phase = "prefill"
        elif has_decode:
            phase = "decode"
        else:
            phase = "idle"
        return self._finalize_round(items, phase, decisions, now,
                                    needed_first=needed_first)

    def postprocess(self, items: list[BatchItem], token_ids: list[int] | None,
                    now: float | None = None):
        """模型执行返回后的收尾（Day9 逐 item 提交契约 §4.3）。

        - round 关联校验（双保险之一）：items 携带的 round_id 必须与最近一次
          调度轮一致；含活动请求的批次不匹配即显式拒绝——修复 Day8 审查 §4.4
          指出的"旧批次重放时同 seq 已被重新规划，会以当前计划二次提交"缺陷。
          全终态批次豁免该校验：其请求已被安全路径幂等清理，携带的 token 无意义，
          重放是安全空操作（沿用 Day6/7 所有权测试依赖的契约）。
        - token 数显式校验：与 sum(item.needs_sample) 一致。needs_sample 是调度
          接纳时冻结的快照（decode 恒 True；prefill 等于 is_last_chunk），与
          ModelRunner 合并结果的顺序契约共享同一权威，不使用会静默截断的 zip。
          run() 对"本轮无任何采样"返回 None，这里与空列表同价。
        - 原子性/幂等（沿用 Day8 §4.5）：先整批校验再逐个提交；取消/超时/终态
          不推进 offset；提交后立即校验单调有界；重复/迟到收尾被拒。
        - 混合轮新增约束：单个 item 的终态不影响同轮其他 item 的提交。

        控制锁（Day10 §3.4）：postprocess 是模型返回后的 CPU 记账安全点，全程
        持锁保证与外部 request_cancel 信号线性化；锁不跨越 GPU forward。
        """
        if now is None:
            now = self._clock()
        with self._control_lock:
            return self._postprocess(items, token_ids, now=now)

    def _postprocess(self, items: list[BatchItem], token_ids: list[int] | None, *,
                     now: float):
        seqs = [it.seq for it in items]
        # 含活动请求标记：round 关联校验与 token 数校验共用同一谓词（审查 §4.2）
        has_live = any(not s.is_terminal for s in seqs)
        # ---------- round 关联校验（仅对含活动请求的批次强制，§4.3 前置校验 1） ----------
        if has_live:
            last_round = (self.last_schedule_stats or {}).get("round_id")
            if last_round is None or any(it.round_id != last_round for it in items):
                raise ValueError(
                    "postprocess 批次与最近调度轮不匹配（round_id 关联校验失败），"
                    "疑似重复或迟到的旧批次结果，已拒绝提交")
        samples = list(token_ids) if token_ids is not None else []
        # 需采样集合：按调度快照 BatchItem.needs_sample 逐 item 对齐——
        # 安全检查丢弃的采样在 token_ids 中仍占位，游标推进与安全检查解耦，
        # 保证对齐关系不因丢弃而错位（Day8 原则的逐 item 版本）
        needs_sample = [it.needs_sample for it in items]
        # 全终态批次（纯迟到重放）不校验 token 数：其请求已被安全路径幂等清理，
        # 携带的 token 无意义；Day6/Day7 所有权测试依赖"全终态旧批次重放是
        # 安全空操作"。含活动请求的批次必须与调度计划严格一致
        if has_live and len(samples) != sum(needs_sample):
            raise ValueError(
                f"postprocess 采样 token 数与需采样请求数不一致：得到 {len(samples)}，"
                f"期望 {sum(needs_sample)}（批次 {len(items)}，"
                f"decode {sum(1 for it in items if it.phase == 'decode')} 条）")
        for it in items:
            if not it.seq.is_terminal and it.seq.num_scheduled_tokens <= 0:
                raise ValueError(
                    f"request {it.seq.request_id!r} (seq_id={it.seq.seq_id}) "
                    f"无待执行的调度计划，疑似重复或迟到的 postprocess，已拒绝提交")
        ti = 0  # token_ids 游标
        for it, need in zip(items, needs_sample):
            token_id = samples[ti] if need else None
            if need:
                ti += 1
            seq = it.seq
            # 安全边界（模型执行返回后、追加 token 与 KV 记账之前）：
            # 1) 终态兜底：外部 mark_* 产生的终态请求直接清理，不参与记账，
            #    且不影响同轮其他 item 的提交（§4.6 同轮隔离）；
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
                # 取消优先于超时优先于正常提交（§4.4）：走统一 cancel 入口，
                # 终态迁移 + 队列/索引/资源收尾与 request_control 事件同源
                self.cancel(seq.seq_id, reason=seq.cancel_reason or "client_cancelled", now=now)
                continue
            if seq.deadline is not None and now >= seq.deadline:
                # 模型执行期间跨过 deadline：丢弃本轮采样，按 TIMEOUT 终止
                self.timeout(seq.seq_id, now=now)
                continue
            # ---------- 原子提交（成功路径，prefill/decode 统一） ----------
            offset_before = seq.prefill_offset
            q = seq.num_scheduled_tokens
            # hash_blocks 接收显式区间 [offset_before, offset_before + q)：
            # 只登记本次执行写满的完整块（中间 chunk 写满的块同样登记，
            # 尾块永不登记）；decode 轮区间为 [len-1, len)，与 Day7 逐 token
            # 登记行为逐位一致
            self.block_manager.hash_blocks(seq, offset_before, offset_before + q)
            seq.prefill_offset = offset_before + q
            seq.num_scheduled_tokens = 0
            # 单调有界校验（§5.3 不变量 1）：推进后立即核对，违反即显式抛错
            if seq.prefill_offset > seq.prefill_target:
                raise ValueError(
                    f"request {seq.request_id!r}: 提交后 prefill_offset "
                    f"{seq.prefill_offset} 超过目标 {seq.prefill_target}，进度记账被破坏")
            if it.phase == "prefill" and seq.prefill_offset < seq.prefill_target:
                # 中间 chunk：丢弃采样结果，保持 WAITING，下一轮延续进度
                # （阶段分支由 item.phase 决定，替代 Day8 的全批 is_prefill）
                continue
            # 最后 chunk（或 decode item）：追加采样的首 completion；
            # 完成判定：EOS 触发记为 stop，达到 max_tokens 记为 length；
            # mark_finished 幂等，配合 _finalize 保证不 double free
            seq.append_token(token_id)
            if not seq.ignore_eos and token_id == self.eos:
                seq.mark_finished("stop")
                self._finalize(seq, now)
            elif seq.num_completion_tokens == seq.max_tokens:
                seq.mark_finished("length")
                self._finalize(seq, now)
