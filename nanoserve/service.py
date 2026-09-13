"""服务层：统一内部请求、请求管理、错误类型与结果 DTO（Day11–12 §4/§5.4）。

职责分离：
- build_sampling_params / build_completion_request / build_chat_request：
  HTTP 请求到统一 InternalRequest 的唯一转换路径（采样映射、chat template、
  上下文长度预算检查都在这里完成，且发生在 engine.add_request 之前）；
- RequestHandle / CompletionResult：等待中的句柄与只读结果 DTO，不保存
  可变 Sequence；
- RequestManager：句柄登记、完成/失败收口。pending 表只在 worker 线程弹出、
  HTTP 线程只插入，resolve/fail 借助 dict.pop 保证每个句柄恰好收口一次。

线程模型：HTTP 路由为同步 def（FastAPI 线程池执行），等待使用
concurrent.futures.Future；worker 线程负责 resolve/fail，天然 loop-safe，
不跨线程触碰事件循环。
"""

import logging
import os
import queue
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Iterator, Literal

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest

try:  # 底层 Day13 DTO 可先后落地；服务层对旧版本保持可导入。
    from nanovllm.engine.completed_request import TokenEvent
except ImportError:  # pragma: no cover - 被底层实现覆盖时不走此兼容分支
    @dataclass(frozen=True, slots=True)
    class TokenEvent:
        seq_id: int
        request_id: str
        round_id: int
        token_ids: tuple[int, ...]
        completion_index: int
        emitted_at: float
        is_first_token: bool
        is_final: bool = False
        finish_reason: str | None = None
from nanoserve.schemas import (ChatCompletionRequest, CompletionRequest,
                               DEFAULT_MAX_TOKENS)
try:
    from nanoserve.observability import Observability, RequestTimeline
except ImportError:  # pragma: no cover - keeps low-level service imports lightweight
    Observability = object
    RequestTimeline = object

logger = logging.getLogger(__name__)

InternalKind = Literal["completion", "chat"]


# ============================== 统一错误类型 ==============================

class APIError(Exception):
    """所有由 API 主动返回的错误的基类：携带 HTTP 状态码与 OpenAI 风格错误体。

    - type：OpenAI 错误大类（invalid_request_error / service_error / server_error）；
    - code：稳定错误码（用于客户端分支与测试断言）；
    - request_id：请求已入队后发生的错误会带上请求 ID，用于 x-request-id
      响应头关联；校验阶段（尚未生成请求 ID）的错误由响应处理器生成随机 ID。
    消息内容禁止包含 prompt/token 明文、block table 或内部堆栈。
    """

    status_code: int = 500
    error_type: str = "server_error"
    code: str = "internal_error"

    def __init__(self, message: str, *, param: str | None = None,
                 request_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.request_id = request_id

    def to_payload(self) -> dict:
        """转换为 OpenAI 风格错误体（不含堆栈）。"""
        return {"error": {
            "message": self.message,
            "type": self.error_type,
            "param": self.param,
            "code": self.code,
        }}


class InvalidRequestError(APIError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request_error"


class ContextLengthExceededError(APIError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "context_length_exceeded"


class ModelNotFoundError(APIError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class StreamNotImplementedError(APIError):
    status_code = 501
    error_type = "invalid_request_error"
    code = "stream_not_implemented"


class ServiceNotReadyError(APIError):
    status_code = 503
    error_type = "service_error"
    code = "service_not_ready"


class ServiceDrainingError(APIError):
    status_code = 503
    error_type = "service_error"
    code = "service_draining"


class RequestTimeoutError(APIError):
    status_code = 504
    error_type = "service_error"
    code = "request_timeout"


class EngineError(APIError):
    status_code = 500
    error_type = "server_error"
    code = "engine_error"


# ============================== 统一内部请求 ==============================

@dataclass(frozen=True, slots=True)
class InternalRequest:
    """进入 Engine 前的统一内部请求（completion/chat 共用同一种表示）。

    - prompt_token_ids 是 completion 编码或 chat template 的唯一产物，
      Engine 收到后不再重复应用模板或编码；
    - kind 只用于选择响应包装，不参与 Scheduler 状态机；
    - request_id 由服务层生成（cmpl-<uuid> / chatcmpl-<uuid>）并传给
      add_request，完成后不会重新生成；
    - deadline 使用 perf_counter 单调时钟（与 Day10 一致）；HTTP 响应里的
      Unix created 时间戳只用于展示，不参与超时判断；
    - 不保存 Sequence、block table 或可变 token 列表引用。
    """

    request_id: str
    kind: InternalKind
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    model_id: str
    created_at: float                 # perf_counter 单调时钟
    deadline: float | None


@dataclass(frozen=True)
class CompletionResult:
    """完成结果的只读 DTO：worker 按完成记录解码后交给路由包装。"""

    request_id: str
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class RequestHandle:
    """等待中的请求句柄：路由线程持有并等待 Future，worker 线程收口。"""

    request_id: str
    kind: InternalKind
    created_at: float
    # Engine admission 后绑定的 seq_id；完成记录必须同时匹配 request_id
    # 与 seq_id，防止旧轮次迟到记录完成同 ID 复用的新句柄。
    seq_id: int | None = None
    future: Future = field(default_factory=Future)
    stream: "StreamHandle | None" = None
    admitted_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    timeline: object | None = None
    next_completion_index: int = 0


class StreamBackpressureError(APIError):
    """流客户端消费太慢；失败该流而不是让 Engine 线程阻塞。"""

    status_code = 503
    error_type = "service_error"
    code = "stream_backpressure"


@dataclass(frozen=True, slots=True)
class StreamTerminal:
    """有界流的终态 envelope；正常值为 CompletedRequest，异常值为 APIError。"""

    record: object | None = None
    error: BaseException | None = None


class StreamHandle:
    """独立、有界、线程安全的事件 sink。

    publish 永不等待：队列满即以 stream_backpressure 收口。终态 envelope
    按 FIFO 写入，保证已有 token 先于 finish；close/cancel 重复调用无副作用。
    """

    def __init__(self, request_id: str, kind: InternalKind, created_at: float,
                 *, maxsize: int | None = None):
        size = maxsize if maxsize is not None else int(
            os.getenv("NANOSERVE_STREAM_EVENT_QUEUE_SIZE", "16"))
        if size <= 0:
            raise ValueError("stream event queue size must be positive")
        self.request_id, self.kind, self.created_at = request_id, kind, created_at
        self.seq_id: int | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=size)
        self._lock = threading.Lock()
        self._closed = False
        self._terminal: StreamTerminal | None = None
        # admission 与首个输出分别通知：路由返回 StreamingResponse 前要先
        # 给 add/首轮 step 一个有限观察窗口，避免错误已经发生却先发 200 首帧。
        self._ready = threading.Event()
        self._output_ready = threading.Event()
        self._error: BaseException | None = None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def terminal_error(self) -> BaseException | None:
        with self._lock:
            return self._error

    def bind_seq_id(self, seq_id: int) -> bool:
        with self._lock:
            if self.seq_id is not None:
                return self.seq_id == seq_id
            self.seq_id = seq_id
            self._ready.set()
            return True

    def wait_admission(self, timeout: float = 5.0) -> bool:
        """等待 admission 或首个终态；用于首 SSE frame 前暴露提交错误。"""
        if self._ready.wait(timeout=max(0.0, timeout)):
            return True
        with self._lock:
            return self.seq_id is not None or self._terminal is not None

    def publish(self, event: TokenEvent) -> bool:
        """非阻塞写入 token；失败时立即终止该 sink。"""
        with self._lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(event)
                self._output_ready.set()
                return True
            except queue.Full:
                pass
        self.fail(StreamBackpressureError(
            "stream consumer is too slow; request cancelled"))
        return False

    def finish(self, record: CompletedRequest) -> bool:
        return self._put_terminal(StreamTerminal(record=record))

    def abort(self, record: AbortedRequest) -> bool:
        return self._put_terminal(StreamTerminal(record=record))

    def fail(self, error: BaseException) -> bool:
        return self._put_terminal(StreamTerminal(error=error))

    # Explicit aliases make the sink usable by sync and async protocol adapters
    # without exposing its internal queue.
    publish_token = publish

    def _put_terminal(self, terminal: StreamTerminal) -> bool:
        with self._lock:
            if self._closed:
                return False
            # 终态必须排在所有已发布 token 后面；队列满时不能丢 token
            # 再伪造成功 finish，改为保留错误终态并让消费者先排空。
            if self._queue.full():
                self._closed = False
                self._error = StreamBackpressureError(
                    "stream consumer is too slow; request cancelled")
                self._terminal = StreamTerminal(error=self._error)
                self._closed = True
                self._ready.set()
                self._output_ready.set()
                return False
            self._closed = True
            self._terminal = terminal
            self._error = terminal.error
            self._queue.put_nowait(terminal)
            self._ready.set()
            self._output_ready.set()
            return True

    def next_event(self, timeout: float | None = None):
        """有限等待取得 token 或终态；终态缓存后可安全重复读取。"""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                terminal = self._terminal
            if terminal is None:
                raise TimeoutError("timed out waiting for stream event")
            item = terminal
        if isinstance(item, StreamTerminal):
            with self._lock:
                self._terminal = item
            if item.error is not None:
                raise item.error
            return item.record
        return item

    def get(self, timeout: float | None = None):
        return self.next_event(timeout)

    receive = next_event

    def iter_events(self, timeout: float = 1.0) -> Iterator:
        while True:
            item = self.next_event(timeout)
            if isinstance(item, (CompletedRequest, AbortedRequest)):
                return
            yield item

    def close(self, error: BaseException | None = None) -> bool:
        return self.fail(error or ServiceDrainingError("stream closed"))


# ============================== 请求构建（唯一转换路径） ==============================

def build_sampling_params(*, max_tokens: int, temperature: float,
                          top_p: float) -> SamplingParams:
    """唯一采样参数映射入口：completion 与 chat 复用，边界口径一致。

    schema 层已做第一道校验；SamplingParams.__post_init__ 仍是最终校验入口。
    """
    return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                          top_p=top_p)


def _normalize_template_output(template, tokenizer) -> list[int]:
    """把 apply_chat_template 的返回值规范化为 token id 列表。

    不同 transformers 版本的返回类型不同（§3.3 "以实际 tokenizer 兼容性为准"）：
    - list[int]：直接使用（主路径，tokenize=True 的预期产物）；
    - BatchEncoding/dict（含 input_ids）：取 input_ids；批维度取首条
      （单条对话请求只有一条序列）；
    - list[list[int]]：同上，取首条；
    - str：tokenizer 只能返回模板字符串时，用同一 tokenizer 再编码恰好一次，
      避免换 tokenizer 造成不一致。
    """
    if isinstance(template, str):
        return list(tokenizer.encode(template))
    # BatchEncoding（transformers）继承 UserDict 而非 dict，因此用键存在性
    # 做鸭子类型判断，而不是 isinstance(..., dict)
    if template is None:
        raise ValueError("chat template returned None")
    if not isinstance(template, str) and hasattr(template, "keys") \
            and "input_ids" in template:
        template = template["input_ids"]
    if isinstance(template, (list, tuple)) and template \
            and isinstance(template[0], (list, tuple)):
        template = template[0]
    if not isinstance(template, (list, tuple)):
        raise ValueError("chat template did not return token IDs")
    if any(type(token) is not int for token in template):
        raise ValueError("chat template returned invalid token IDs")
    return list(template)


def _check_context_length(prompt_tokens: int, max_tokens: int,
                          max_model_len: int) -> None:
    """上下文预算检查：prompt token 数 + max_tokens 不得超过 max_model_len。

    检查发生在 engine.add_request（KV 分配）之前，避免运行期才因容量不足
    进入无进展状态；错误消息只含计数，不含 token 明文。
    """
    if prompt_tokens + max_tokens > max_model_len:
        raise ContextLengthExceededError(
            f"prompt_tokens ({prompt_tokens}) + max_tokens ({max_tokens}) "
            f"exceeds max_model_len ({max_model_len})",
            param="max_tokens")


def build_completion_request(request: CompletionRequest, tokenizer,
                             *, model_id: str, max_model_len: int,
                             max_request_seconds: float | None = None) -> InternalRequest:
    """completion 字符串 prompt → 统一内部请求。

    使用 Engine 同一 tokenizer 编码（服务层不创建第二个 tokenizer）；
    编码结果为空（无法定义 prefill 目标）显式拒绝。
    """
    try:
        token_ids = tokenizer.encode(request.prompt)
    except Exception as exc:
        raise InvalidRequestError(
            "prompt encoding failed", param="prompt") from exc
    if not isinstance(token_ids, (list, tuple)) or \
            any(type(token) is not int for token in token_ids):
        raise InvalidRequestError(
            "prompt encoding produced invalid token IDs", param="prompt")
    if not token_ids:
        raise InvalidRequestError("prompt produced no tokens", param="prompt")
    _check_context_length(len(token_ids), request.max_tokens, max_model_len)
    now = perf_counter()
    return InternalRequest(
        request_id=f"cmpl-{uuid.uuid4().hex}",
        kind="completion",
        prompt_token_ids=tuple(token_ids),
        sampling_params=build_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p),
        model_id=model_id,
        created_at=now,
        deadline=now + max_request_seconds
        if max_request_seconds is not None else None,
    )


def build_chat_request(request: ChatCompletionRequest, tokenizer,
                       *, model_id: str, max_model_len: int,
                       max_request_seconds: float | None = None) -> InternalRequest:
    """messages → chat template token 列表 → 统一内部请求。

    模板始终由 tokenizer 提供（不手写 ChatML 或假定固定模型格式），调用
    apply_chat_template(tokenize=True, add_generation_prompt=True) 得到的
    token 列表直接作为 prompt_token_ids，避免模板字符串再次编码造成不一致；
    若 tokenizer 只能返回字符串，则用同一 tokenizer 再编码一次（固定路径）。
    模板抛出的任何异常都转换为可理解的 400，不向客户端泄漏 traceback。
    """
    messages = [{"role": m.role, "content": m.content}
                for m in request.messages]
    try:
        template = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
    except Exception as exc:  # 模板失败属于请求侧错误，转 400 并吞掉原始堆栈
        raise InvalidRequestError(
            "chat template failed for the loaded tokenizer",
            param="messages") from exc
    try:
        token_ids = _normalize_template_output(template, tokenizer)
    except Exception as exc:
        raise InvalidRequestError(
            "chat template produced invalid token IDs", param="messages") \
            from exc
    if not token_ids:
        raise InvalidRequestError(
            "chat template produced no tokens", param="messages")
    _check_context_length(len(token_ids), request.max_tokens, max_model_len)
    now = perf_counter()
    return InternalRequest(
        request_id=f"chatcmpl-{uuid.uuid4().hex}",
        kind="chat",
        prompt_token_ids=tuple(token_ids),
        sampling_params=build_sampling_params(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p),
        model_id=model_id,
        created_at=now,
        deadline=now + max_request_seconds
        if max_request_seconds is not None else None,
    )


# ============================== 请求管理 ==============================

def _set_future_result(future: Future, value) -> None:
    """幂等收口工具：Future 已完成时不重复设置（防御迟到重复记录）。"""
    if not future.done():
        future.set_result(value)


def _set_future_exception(future: Future, exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


class RequestManager:
    """请求句柄登记与结果收口（线程安全）。

    - submit：HTTP 线程生成句柄并排队到 worker（worker 由 app 注入）；
    - resolve_completed / resolve_aborted：仅 worker 线程调用，按 request_id
      弹出句柄并收口 Future；未知/迟到记录记日志后丢弃，不会误完成复用
      同一 ID 的新请求（ID 由服务层用 uuid 生成，活动期唯一）；
    - fail / fail_all：把底层取消/超时/Engine 异常转为服务异常；
    - stop_accepting：优雅关闭第一步，之后 submit 抛 ServiceDrainingError。
    """

    def __init__(self, tokenizer, observer=None):
        # 文本解码必须使用与 Engine 编码相同的 tokenizer（app 启动时注入）
        self._tokenizer = tokenizer
        # 观测器是旁路依赖；None 保持 Day11–12 直接构造 RequestManager 的兼容性。
        self._observer = observer
        self._cancel_requested: dict[str, str] = {}
        # pending 同时被 HTTP 线程和 worker/关闭线程访问；即使 CPython 的
        # 单次 dict 操作受 GIL 保护，也不能用它替代跨操作的一致性锁。
        self._pending: dict[str, RequestHandle] = {}
        self._lock = threading.Lock()
        self._accepting = True
        self._worker = None

    def attach_worker(self, worker) -> None:
        """注入 EngineWorker（app 启动时调用一次；submit 依赖其命令队列）。"""
        self._worker = worker

    # ---------- HTTP 线程侧 ----------

    def submit(self, request: InternalRequest, *, stream: bool = False,
               stream_queue_size: int | None = None):
        """登记句柄并排队到 worker；停止接收后抛 ServiceDrainingError。

        stream=True 时先创建独立 sink 再入队，避免 submit 与 worker admission
        之间出现无法交付事件的窗口；默认路径仍返回原 RequestHandle。
        """
        # 登记和命令入队必须在同一临界区完成；shutdown 只能在线性化点
        # 之前或之后发生，不能把已登记但未入队的 Future 留给无人消费。
        with self._lock:
            if not self._accepting:
                raise ServiceDrainingError(
                    "server is shutting down and not accepting requests")
            if self._worker is None:
                raise ServiceNotReadyError("engine worker is not ready")
            if request.request_id in self._pending:
                # 服务层 uuid 生成的 ID 不应冲突；命中即为编程错误，尽早失败
                raise EngineError("duplicate active request id",
                                  request_id=request.request_id)
            stream_handle = StreamHandle(
                request.request_id, request.kind, request.created_at,
                maxsize=stream_queue_size) if stream else None
            handle = RequestHandle(request_id=request.request_id,
                                   kind=request.kind,
                                   created_at=request.created_at,
                                   stream=stream_handle)
            if self._observer is not None:
                try:
                    handle.timeline = self._observer.start_request(
                        request.request_id, kind=request.kind,
                        prompt_tokens=len(request.prompt_token_ids))
                    handle.timeline.mark_submitted(request.created_at)
                    self._observer.emit("request_submitted",
                                       request_id=request.request_id,
                                       kind=request.kind,
                                       observed_at=request.created_at,
                                       queue_origin="http")
                except Exception:
                    handle.timeline = None
            self._pending[request.request_id] = handle
            try:
                self._worker.submit(request)
            except BaseException:
                # 入队失败必须回收句柄，避免 Future 永久悬挂
                self._pending.pop(request.request_id, None)
                raise
        return handle

    def wait(self, handle: RequestHandle) -> CompletionResult:
        """等待 worker 收口；失败以 APIError 子类异常透传给路由。"""
        return handle.future.result()

    def stop_accepting(self) -> None:
        """优雅关闭第一步：新请求一律 503（与 worker 退出线性化）。"""
        with self._lock:
            self._accepting = False

    # ---------- worker 线程侧 ----------

    def bind_seq_id(self, request_id: str, seq_id: int) -> bool:
        """绑定 Engine admission 后的 seq_id，供迟到记录做世代校验。"""
        with self._lock:
            handle = self._pending.get(request_id)
            if handle is None or handle.seq_id is not None:
                return False
            handle.seq_id = seq_id
            handle.admitted_at = perf_counter()
            if handle.stream is not None:
                handle.stream.bind_seq_id(seq_id)
            timeline = getattr(handle, "timeline", None)
            if timeline is not None:
                try:
                    timeline.seq_id = seq_id
                    timeline.mark_admitted(handle.admitted_at)
                    self._observer.emit("request_admitted",
                                       request_id=request_id, seq_id=seq_id,
                                       kind=handle.kind,
                                       observed_at=handle.admitted_at)
                except Exception:
                    pass
            return True

    def stream_handle(self, request_id: str) -> StreamHandle | None:
        with self._lock:
            handle = self._pending.get(request_id)
            return handle.stream if handle is not None else None

    def resolve_token_event(self, event: TokenEvent) -> bool:
        """校验 request+seq 世代并记录 token；流请求再发布到 sink。"""
        with self._lock:
            handle = self._pending.get(event.request_id)
            if handle is None or handle.seq_id != event.seq_id:
                logger.warning("忽略未知/迟到 token 事件: request_id=%s seq_id=%s",
                               event.request_id, event.seq_id)
                return False
            expected = handle.next_completion_index
            if event.completion_index != expected:
                logger.warning("忽略非连续 token 事件: request_id=%s seq_id=%s index=%s expected=%s",
                               event.request_id, event.seq_id,
                               event.completion_index, expected)
                return False
            if event.is_first_token != (expected == 0):
                return False
            handle.next_completion_index = expected + len(event.token_ids)
            if handle.first_token_at is None and event.token_ids:
                handle.first_token_at = event.emitted_at
            if event.token_ids:
                handle.last_token_at = event.emitted_at
            timeline = getattr(handle, "timeline", None)
            if timeline is not None and event.token_ids:
                try:
                    for _ in event.token_ids:
                        timeline.mark_token(event.emitted_at)
                    if self._observer is not None:
                        self._observer.emit(
                            "request_first_token" if event.is_first_token else "request_token",
                            request_id=event.request_id, seq_id=event.seq_id,
                            round_id=event.round_id, completion_index=event.completion_index,
                            phase=getattr(event, "phase", "decode"),
                        observed_at=event.emitted_at)
                except Exception:
                    pass
            sink = handle.stream
        # 非流式请求也要消费事件以记录指标，但不需要入队。
        if sink is None:
            return True
        ok = sink.publish(event)
        return bool(ok)

    def cancel(self, request_id: str, reason: str = "client_cancelled") -> bool:
        """取消请求并覆盖尚未 admission 的窗口。"""
        with self._lock:
            handle = self._pending.get(request_id)
            if handle is None:
                return False
            self._cancel_requested[request_id] = reason
            worker = self._worker
        if worker is None:
            return False
        return worker.cancel(request_id, reason)

    def is_cancel_requested(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._cancel_requested

    def pending_request_ids(self) -> list[str]:
        """当前未收口句柄的 ID 快照（worker 关闭阶段发送取消信号用）。"""
        with self._lock:
            return list(self._pending.keys())

    def resolve_completed(self, record: CompletedRequest) -> None:
        """正常完成记录 → 解码文本并成功收口。

        只解码 completion token IDs（与 Engine 编码同一 tokenizer）；
        未知/迟到记录（句柄已收口或不存在）记日志丢弃，不完成另一个
        复用 ID 的请求。
        """
        with self._lock:
            handle = self._pending.get(record.request_id)
            if handle is None or (handle.seq_id is not None
                                  and handle.seq_id != record.seq_id):
                handle = None
            else:
                handle = self._pending.pop(record.request_id)
        if handle is None:
            logger.warning("忽略未知/迟到完成记录: request_id=%s seq_id=%s",
                           record.request_id, record.seq_id)
            return
        if handle.stream is not None:
            # 流式句柄必须保留已排队 token，并把终态放在其后；HTTP
            # generator 会据此发送 finish chunk，不能把 CompletedRequest
            # 当作普通 Future 结果直接丢掉。
            handle.stream.finish(record)
            self._record_observer_terminal(handle, record, "completed")
            with self._lock:
                self._cancel_requested.pop(record.request_id, None)
            return
        try:
            text = self._tokenizer.decode(list(record.completion_token_ids))
            result = CompletionResult(
                request_id=record.request_id,
                text=text,
                finish_reason=record.finish_reason,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
            )
        except Exception:
            # 句柄已从 pending 移出后也必须得到终态；否则解码器异常会让
            # HTTP Future 永久等待，并阻塞服务关闭。原始异常不返回客户端。
            _set_future_exception(handle.future,
                                  EngineError("failed to decode completion"))
            logger.exception("完成结果解码失败: request_id=%s",
                             record.request_id)
            return
        self._record_observer_terminal(handle, record, "completed")
        _set_future_result(handle.future, result)

    def resolve_aborted(self, record: AbortedRequest) -> None:
        """取消/超时/异常记录 → 按原因映射为失败收口（不伪装成成功）。"""
        with self._lock:
            handle = self._pending.get(record.request_id)
            if handle is None or (handle.seq_id is not None
                                  and handle.seq_id != record.seq_id):
                handle = None
            else:
                handle = self._pending.pop(record.request_id)
        if handle is None:
            logger.warning("忽略未知/迟到中止记录: request_id=%s seq_id=%s reason=%s",
                           record.request_id, record.seq_id, record.finish_reason)
            return
        error = self._abort_error(record, handle.request_id)
        if handle.stream is not None:
            handle.stream.abort(record)
        else:
            _set_future_exception(handle.future, error)
        self._record_observer_terminal(handle, record, "aborted")
        with self._lock:
            self._cancel_requested.pop(record.request_id, None)

    def _record_observer_terminal(self, handle, record, status: str) -> None:
        timeline = getattr(handle, "timeline", None)
        if timeline is None or self._observer is None:
            return
        try:
            timeline.mark_finished(getattr(record, "finished_at", None))
            reason = getattr(record, "finish_reason", "") or ""
            if status == "completed":
                metric_status = "completed"
            elif "timeout" in reason or "deadline" in reason:
                metric_status = "timeout"
            elif reason in ("engine_error", "execution_error"):
                metric_status = "engine_error"
            elif "shutdown" in reason or reason == "engine_exit":
                metric_status = "server_shutdown"
            else:
                metric_status = "cancelled"
            self._observer.record_timeline(
                timeline, status=metric_status, finish_reason=reason,
                log_event=True)
            if metric_status != "completed":
                self._observer.emit("request_aborted", request_id=timeline.request_id,
                                   seq_id=timeline.seq_id, kind=timeline.kind,
                                   status=metric_status, finish_reason=reason,
                                   latency_seconds=timeline.latency_seconds)
        except Exception:
            logger.exception("request observability update failed")

    @staticmethod
    def _abort_error(record: AbortedRequest, request_id: str) -> APIError:
        """终态原因 → 服务错误映射：timeout→504，engine 侧→500，其余→503。"""
        reason = record.finish_reason or "cancelled"
        if reason == "timeout" or reason == "deadline_exceeded":
            return RequestTimeoutError(
                "request exceeded its deadline", request_id=request_id)
        if reason in ("engine_error", "execution_error"):
            return EngineError("engine execution failed",
                               request_id=request_id)
        # cancelled / client_cancelled / server_shutdown 等：服务取消/收尾
        return ServiceDrainingError(
            "request was cancelled before completion",
            request_id=request_id)

    def fail_request(self, request_id: str,
                     make_error: Callable[[], APIError]) -> None:
        """失败单个句柄（worker 处理 submit 命令异常时调用）。

        make_error 为工厂函数：每个句柄获得独立的异常实例，可携带各自的
        request_id 供 x-request-id 关联。
        """
        with self._lock:
            handle = self._pending.pop(request_id, None)
        if handle is None:
            return
        exc = make_error()
        if exc.request_id is None:
            exc.request_id = request_id
        if handle.stream is not None:
            handle.stream.fail(exc)
        else:
            _set_future_exception(handle.future, exc)
        self._record_observer_failure(handle, exc)
        with self._lock:
            self._cancel_requested.pop(request_id, None)

    def _record_observer_failure(self, handle, exc) -> None:
        if self._observer is None or getattr(handle, "timeline", None) is None:
            return
        try:
            timeline = handle.timeline
            timeline.mark_finished(perf_counter())
            reason = getattr(exc, "code", "engine_error")
            status = "server_shutdown" if "draining" in reason else "engine_error"
            self._observer.record_timeline(timeline, status=status, log_event=True)
            self._observer.emit("request_aborted", request_id=handle.request_id,
                               seq_id=handle.seq_id, kind=handle.kind,
                               status=status, finish_reason=reason,
                               observed_at=timeline.finished_at)
        except Exception:
            pass

    def fail_all(self, make_error: Callable[[], APIError]) -> None:
        """失败全部未收口句柄（Engine 异常/关闭兜底；恰好一次语义由 pop 保证）。"""
        with self._lock:
            request_ids = list(self._pending)
        for request_id in request_ids:
            self.fail_request(request_id, make_error)
