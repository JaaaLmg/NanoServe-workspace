from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """请求生命周期六状态（Day6 状态机，语义见 docs/request-lifecycle.md 3.1）。

    - WAITING:   已接收但尚未完成首轮 prefill，或抢占后等待恢复（非终态）
    - RUNNING:   已被调度，可能参加下一轮执行（非终态）
    - PREEMPTED: 因 KV 资源不足被主动暂停，等待恢复（非终态）
    - FINISHED:  正常生成结束（EOS 或达到 max_tokens），终态
    - CANCELLED: 用户/客户端主动取消，终态
    - TIMEOUT:   超过 deadline 被系统终止，终态
    """

    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


# 终态集合：进入后不允许任何迁移；完成/取消/超时的清理动作必须幂等
TERMINAL_STATUSES = frozenset({
    SequenceStatus.FINISHED,
    SequenceStatus.CANCELLED,
    SequenceStatus.TIMEOUT,
})

# 合法迁移表：状态机规则的唯一权威定义（对应文档 3.2 迁移图）。
# Scheduler 等调用方必须通过 transition_to() 迁移，不得绕过本表直接赋值状态。
VALID_TRANSITIONS: dict[SequenceStatus, frozenset[SequenceStatus]] = {
    SequenceStatus.WAITING: frozenset({
        SequenceStatus.RUNNING,
        SequenceStatus.CANCELLED,
        SequenceStatus.TIMEOUT,
    }),
    SequenceStatus.RUNNING: frozenset({
        SequenceStatus.PREEMPTED,
        SequenceStatus.FINISHED,
        SequenceStatus.CANCELLED,
        SequenceStatus.TIMEOUT,
    }),
    SequenceStatus.PREEMPTED: frozenset({
        SequenceStatus.WAITING,
        SequenceStatus.CANCELLED,
        SequenceStatus.TIMEOUT,
    }),
    # 终态：空集合，任何迁出都非法
    SequenceStatus.FINISHED: frozenset(),
    SequenceStatus.CANCELLED: frozenset(),
    SequenceStatus.TIMEOUT: frozenset(),
}

# 进入终态时未显式给出 reason 时使用的默认 finish_reason
_DEFAULT_FINISH_REASONS = {
    SequenceStatus.FINISHED: "stop",
    SequenceStatus.CANCELLED: "cancelled",
    SequenceStatus.TIMEOUT: "timeout",
}


class InvalidStateTransition(Exception):
    """非法状态迁移异常。

    属于编程错误（如 FINISHED -> RUNNING），应尽早失败而不是悄悄把
    队列和 KV 记账弄乱。错误信息包含请求 ID、原状态、目标状态和原因，
    便于直接定位调用点。
    """

    def __init__(self, seq, old_status, new_status, reason=None):
        self.request_id = getattr(seq, "request_id", None)
        self.seq_id = getattr(seq, "seq_id", None)
        self.old_status = old_status
        self.new_status = new_status
        self.reason = reason
        message = (
            f"request {self.request_id!r} (seq_id={self.seq_id}): "
            f"illegal state transition {old_status.name} -> {new_status.name}"
        )
        if reason is not None:
            message += f", reason={reason!r}"
        super().__init__(message)


class Sequence:
    block_size = 256
    counter = count()
    # 序列化格式版本：v1 为旧 6 元组（不含控制面字段）；
    # v2 为 (版本号, 字段名字典)，新增控制面字段且可向后兼容 v1
    STATE_VERSION = 2

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(),
                 request_id: str | None = None, deadline: float | None = None):
        self.seq_id = next(Sequence.counter)
        # 对外稳定的请求 ID：日志串联、Engine 取消入口使用；保留 seq_id 兼容内部排序
        self.request_id = request_id if request_id is not None else f"req-{self.seq_id}"
        self.status = SequenceStatus.WAITING
        # 时间信息统一使用单调时钟 perf_counter（不可回拨），deadline 判断才可靠
        self.created_at = perf_counter()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.deadline = deadline
        # 取消标记：request_cancel() 只打标记；真正的 CANCELLED 迁移发生在调度边界
        self.cancel_requested = False
        self.cancel_reason: str | None = None
        # 终态原因：stop/length/cancelled/timeout 等
        self.finish_reason: str | None = None
        # 抢占历史：区分"排队等待"与"资源不足导致的等待"，供日志与指标使用
        self.num_preempts = 0
        self.last_preempted_at: float | None = None
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.seed = sampling_params.seed
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    # ---------- 只读状态属性：减少调用方的重复判断 ----------

    @property
    def is_terminal(self):
        """是否处于终态（FINISHED/CANCELLED/TIMEOUT）。"""
        return self.status in TERMINAL_STATUSES

    @property
    def is_finished(self):
        """是否已结束：Day6 起覆盖三个终态。

        需要精确区分"正常完成"时，应直接比较 status == FINISHED
        （如 LLMEngine.step() 收集输出时）。
        """
        return self.is_terminal

    @property
    def is_active(self):
        """是否仍在调度生命周期内（未进入终态）。"""
        return not self.is_terminal

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ---------- 状态机统一迁移入口 ----------

    def transition_to(self, new_status: SequenceStatus, *, reason: str | None = None,
                      now: float | None = None) -> None:
        """统一迁移入口：校验合法性并维护时间戳、reason 等派生字段。

        非法迁移直接抛 InvalidStateTransition，且不产生任何副作用，
        保证不会出现"队列已改但状态未变"的半完成操作。
        """
        old_status = self.status
        if new_status not in VALID_TRANSITIONS[old_status]:
            raise InvalidStateTransition(self, old_status, new_status, reason)
        self.status = new_status
        if now is None:
            now = perf_counter()
        if new_status == SequenceStatus.RUNNING:
            # started_at 只记录首次进入 RUNNING 的时间；抢占恢复后不重置
            if self.started_at is None:
                self.started_at = now
        elif new_status == SequenceStatus.PREEMPTED:
            # 记录抢占历史，供日志/指标区分等待原因
            self.num_preempts += 1
            self.last_preempted_at = now
        elif new_status in TERMINAL_STATUSES:
            self.finished_at = now
            # 终态 reason：显式指定优先，否则使用状态对应的默认值
            self.finish_reason = reason if reason is not None else _DEFAULT_FINISH_REASONS[new_status]

    def request_cancel(self, reason: str = "client_cancelled") -> bool:
        """记录取消请求：只打标记，不迁移状态。

        真正的 CANCELLED 迁移和资源释放在调度边界执行（Scheduler.cancel
        或 postprocess 的安全检查），避免在模型执行中途强改请求。
        - 已终态返回 False；
        - 重复取消不覆盖首次记录的原因。
        """
        if self.is_terminal:
            return False
        if not self.cancel_requested:
            self.cancel_requested = True
            self.cancel_reason = reason
        return True

    def mark_finished(self, reason: str = "stop", now: float | None = None) -> bool:
        """正常完成（幂等）：已终态返回 False，不重复清理资源。"""
        if self.is_terminal:
            return False
        self.transition_to(SequenceStatus.FINISHED, reason=reason, now=now)
        return True

    def mark_cancelled(self, reason: str = "client_cancelled", now: float | None = None) -> bool:
        """标记为已取消（幂等）：已终态返回 False。"""
        if self.is_terminal:
            return False
        self.transition_to(SequenceStatus.CANCELLED, reason=reason, now=now)
        return True

    def mark_timeout(self, reason: str = "deadline_exceeded", now: float | None = None) -> bool:
        """标记为超时终止（幂等）：已终态返回 False。"""
        if self.is_terminal:
            return False
        self.transition_to(SequenceStatus.TIMEOUT, reason=reason, now=now)
        return True

    # ---------- 张量并行序列化协议 ----------

    def __getstate__(self):
        # TP>1 时 Sequence 经共享内存 pickle 传输到子进程。
        # 数据面字段（进度/block_table/last_state）照旧；
        # 控制面必须同步 status、cancel_requested 和生成进度（is_prefill），
        # 时间戳等仅 rank 0 关心的字段不传输，由 __setstate__ 给默认值兜底。
        # 注意：版本化只保证"新版进程可读取旧格式"的单向兼容，
        # 不承诺任意新旧进程混合部署下的双向转发能力
        last_state = self.last_token if not self.is_prefill else self.token_ids
        payload = {
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_scheduled_tokens": self.num_scheduled_tokens,
            "block_table": self.block_table,
            "last_state": last_state,
            "status": self.status,
            "is_prefill": self.is_prefill,
            "cancel_requested": self.cancel_requested,
        }
        return (Sequence.STATE_VERSION, payload)

    def __setstate__(self, state):
        # v2 格式：(版本号, 字段名字典)。版本号显式参与校验：
        # 未知版本立刻报清晰错误，而不是静默按当前结构误读字段
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            version, payload = state
            if version != Sequence.STATE_VERSION:
                raise ValueError(
                    f"不支持的 Sequence 状态格式版本: {version!r}（当前支持 {Sequence.STATE_VERSION}）"
                )
            self.num_tokens = payload["num_tokens"]
            self.num_prompt_tokens = payload["num_prompt_tokens"]
            self.num_cached_tokens = payload["num_cached_tokens"]
            self.num_scheduled_tokens = payload["num_scheduled_tokens"]
            self.block_table = payload["block_table"]
            self.status = payload.get("status", SequenceStatus.WAITING)
            self.is_prefill = payload.get("is_prefill", True)
            self.cancel_requested = payload.get("cancel_requested", False)
            last_state = payload["last_state"]
        else:
            # v1 兼容：旧 6 元组仅做单向读取（不承诺双向转发）。
            # 控制面字段回退默认值；last_state 为列表即 prefill 模式、
            # 为标量即 decode 模式，据此恢复 is_prefill，
            # 保证恢复后的对象可以再次 pickle 往返而不退化成空 prefill payload
            (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
             self.num_scheduled_tokens, self.block_table, last_state) = state
            self.is_prefill = isinstance(last_state, list)
            self.status = SequenceStatus.WAITING
            self.cancel_requested = False
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
        # 子进程不使用的字段补默认值，避免反序列化后访问时 AttributeError
        _defaults = (
            ("seq_id", -1),
            ("request_id", "unpickled"),
            ("created_at", 0.0),
            ("started_at", None),
            ("finished_at", None),
            ("deadline", None),
            ("cancel_reason", None),
            ("finish_reason", None),
            ("num_preempts", 0),
            ("last_preempted_at", None),
        )
        for attr, default in _defaults:
            if not hasattr(self, attr):
                setattr(self, attr, default)
