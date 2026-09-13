"""EngineWorker：唯一直接调用 engine.step() 的专用线程（Day11–12 §4.3）。

为什么不用 asyncio.to_thread(engine.step)：多个 HTTP handler 仍可能并发
进入同一个 Engine，破坏 Scheduler/ModelRunner 的单引擎串行假设。本 worker
用单一专用线程串行驱动 add_request/step/exit，请求处理器只做校验、入队
和等待结果；多个请求先进入 waiting，再由 continuous batching 统一调度。

命令队列语义：
- submit / shutdown 走命令队列，与 step 循环在同一线程线性化；
- cancel 直接调用 engine.cancel_request()：Day10 已把该入口设计为线程安全的
  信号语义（只置位标记，可在模型 forward 期间发起，不触碰 block/队列/token），
  由 Scheduler 控制锁与 worker 线程线性化，无需跨线程应答；
- 退出时每个句柄恰好被 resolve、fail 或 cancel 一次；join/exit 可重复调用，
  不重复调用 engine.exit() 或重复设置 Future。

无进展语义（§4.3）：step() 返回空批次但仍有活动请求（等待队列因 KV 容量
不足无法接纳且无 running 请求可推进，或全部处于暂停）时，与 Day10 的
generate() 无进展错误契约一致：worker 视 Engine 为不可推进，失败所有等待
句柄并退出，避免无界忙循环。

对内禁止暴露：engine.scheduler、Sequence、block_table、model_runner、
可写状态枚举。
"""

import logging
import queue
import threading
from enum import Enum, auto
from time import perf_counter
from typing import Callable, NamedTuple

from nanoserve.service import (APIError, EngineError, InternalRequest,
                               RequestManager, ServiceDrainingError,
                               TokenEvent)

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """worker 生命周期状态（只读视图供健康检查消费）。"""

    NOT_STARTED = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class _SubmitCommand(NamedTuple):
    request: InternalRequest


class _ShutdownCommand:
    """关闭哨兵：唤醒空闲阻塞的命令等待，携带退出语义。"""


class _ConsumeCommand:
    """记录消费哨兵：空闲期的取消在 Engine 安全点立即收尾并产生中止记录，
    此时没有 step 驱动，需要唤醒 worker 排出记录、收口对应 Future，
    否则已取消请求的句柄会悬挂到关闭才被收口。"""


class EngineWorker:

    # 优雅关闭时等待活动请求收尾（step 推进）的上限；超时后由 engine.exit()
    # 的 abort_all_active 兜底清理，不强杀线程、不在 GPU kernel 中途改表
    DRAIN_TIMEOUT_SECONDS = 10.0

    def __init__(self, engine, manager: RequestManager):
        self._engine = engine
        self._manager = manager
        self._commands: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="nanoserve-engine-worker", daemon=True)
        self._status = WorkerStatus.NOT_STARTED
        self._status_lock = threading.Lock()
        # 关闭/退出只执行一次的幂等标记
        self._shutdown_requested = False
        self._engine_exited = False
        self._exit_failed = False
        self._run_started = threading.Event()

    # ==================== 对外线程安全接口（§5.5） ====================

    def start(self) -> None:
        """启动 worker 线程；重复调用是安全空操作。"""
        with self._status_lock:
            if self._status is not WorkerStatus.NOT_STARTED:
                return
            self._status = WorkerStatus.RUNNING
        self._thread.start()
        # 只在 worker 已进入运行函数后返回，避免 lifespan 在启动竞态窗口
        # 误把尚未真正运行的线程报告为 ready；若启动瞬间失败，交给 lifespan
        # 走启动失败路径，而不是先报告 ready。
        self._run_started.wait()
        if self.status is WorkerStatus.FAILED:
            raise RuntimeError("engine worker failed during startup")

    def submit(self, request: InternalRequest) -> None:
        """排队 submit 命令；与 shutdown 哨兵在线性化锁内排序。"""
        with self._status_lock:
            if self._shutdown_requested or self._status in (
                    WorkerStatus.STOPPING, WorkerStatus.STOPPED,
                    WorkerStatus.FAILED):
                raise ServiceDrainingError(
                    "engine worker is stopping and not accepting requests")
            self._commands.put(_SubmitCommand(request))

    def cancel(self, request_id: str, reason: str = "client_cancelled") -> bool:
        """取消请求：只进入 Day10 控制入口（signal 语义），不改 Sequence。

        engine.cancel_request 本身线程安全，可从任意线程调用；已完成/未知
        请求返回 False（迟到 cancel 幂等）。取消成功后入队消费哨兵：Engine
        空闲收尾产生的中止记录由 worker 线程排出（记录消费只发生在
        worker 线程，保持单所有者）。
        """
        result = self._engine.cancel_request(request_id, reason)
        if result:
            self._commands.put(_ConsumeCommand())
        return result

    def begin_shutdown(self) -> None:
        """请求 worker 退出：幂等；通过哨兵命令唤醒空闲等待。"""
        with self._status_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self._commands.put(_ShutdownCommand())

    def join(self, timeout: float | None = None) -> None:
        """等待 worker 线程退出；可重复调用。"""
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def status(self) -> WorkerStatus:
        with self._status_lock:
            return self._status

    @property
    def alive(self) -> bool:
        """线程是否仍在运行（健康检查用：FAILED/退出即不 ready）。"""
        return self._thread.is_alive()

    # ==================== worker 线程内部 ====================

    def _run(self) -> None:
        self._run_started.set()
        try:
            self._loop()
        except BaseException as exc:
            # Engine 异常：先切换为 FAILED，使健康检查立即停止接收请求；
            # 随后再做记录消费和资源收尾，避免失败窗口继续接纳 Future。
            with self._status_lock:
                self._status = WorkerStatus.FAILED
            logger.error("Engine worker 异常退出: %s: %s",
                         type(exc).__name__, exc)
            self._manager.stop_accepting()
            # 单条完成记录损坏时仍需继续处理其他记录，并让对应句柄收口；
            # 即使底层 pop 本身抛错，也不能跳过 fail_all/Engine exit。
            try:
                self._consume_records()
            except BaseException as consume_error:
                logger.error("异常记录消费失败: %s: %s",
                             type(consume_error).__name__, consume_error)
            self._manager.fail_all(self._make_engine_error_factory())
            self._teardown(mark_failed=True)
            return
        # 正常退出路径（收到 shutdown 命令）
        self._shutdown_phase()

    @staticmethod
    def _make_engine_error_factory() -> Callable[[], APIError]:
        return lambda: EngineError("engine execution failed")

    def _loop(self) -> None:
        while True:
            if self._shutdown_requested:
                return
            if self._engine.has_active_requests():
                # 活动期：非阻塞取命令（submit/cancel 需要及时线性化），随后
                # 串行驱动 step——同一命令队列保证 add_request 与 step 的
                # 所有者是同一线程；命令最多延迟一轮 step 被处理
                self._drain_commands()
                if self._shutdown_requested:
                    return
                if self._engine.has_active_requests():
                    outputs, num_tokens = self._engine.step()
                    # 增量必须先于终态记录进入 sink；否则最后一个 token
                    # 可能排在 finish 后，客户端会看到乱序流。
                    self._consume_records()
                    if not outputs and num_tokens == 0 \
                            and self._engine.has_active_requests():
                        # Day10 无进展错误语义：不把无进展伪装成等待，也不无界
                        # 循环；让所有等待中的 HTTP 请求得到明确失败
                        raise EngineError(
                            "engine made no progress: active requests cannot "
                            "be scheduled (usually KV capacity insufficient); "
                            "failing pending requests")
            else:
                # 空闲期：阻塞等待命令，不能 busy loop；shutdown 哨兵保证
                # 关闭请求能立即唤醒
                command = self._commands.get()
                if isinstance(command, _ShutdownCommand):
                    return
                self._handle_command(command)
                self._drain_commands()

    def _drain_commands(self) -> None:
        """非阻塞排空命令队列；遇到 shutdown 哨兵即停止（外层循环检测标志）。"""
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if isinstance(command, _ShutdownCommand):
                return
            self._handle_command(command)

    def _handle_command(self, command) -> None:
        """处理命令：consume 哨兵只消费记录；submit 在 worker 线程内调用
        add_request，保证控制面串行所有者。单个 submit 失败只失败该句柄，
        不影响 worker 与其他请求。"""
        if isinstance(command, _ConsumeCommand):
            self._consume_records()
            return
        request = command.request
        try:
            # 断连先于 admission 时不再把请求送入 Engine，避免 cancel(False)
            # 后出现“取消已发生但请求仍开始生成”的窗口。
            if self._manager.is_cancel_requested(request.request_id):
                self._manager.fail_request(
                    request.request_id,
                    lambda: ServiceDrainingError(
                        "request disconnected before admission"))
                return
            # InternalRequest.prompt_token_ids 是不可变 tuple（设计 §4.1）；
            # Engine 的 Sequence 会在生成过程中原地 append token，
            # 因此在 Engine 边界转换为 list
            seq_id = self._engine.add_request(
                list(request.prompt_token_ids), request.sampling_params,
                request.request_id, request.deadline)
            # 真实 Engine 当前返回 request_id；服务测试桩可返回 seq_id。
            # 若 Engine 只返回 request_id，则完成记录的 seq_id 校验在
            # record 通道中退化为 request_id 校验，保持旧接口兼容。
            if isinstance(seq_id, int):
                bound = self._manager.bind_seq_id(request.request_id, seq_id)
            else:
                # LLMEngine 为兼容旧 API 返回 request_id；在 admission 后
                # 立即读取一次内部 seq_id 仅用于世代绑定，不向服务层暴露 Sequence。
                get_request = getattr(self._engine, "get_request", None)
                admitted = get_request(request.request_id) \
                    if get_request is not None else None
                bound = (self._manager.bind_seq_id(request.request_id,
                                                    admitted.seq_id)
                         if admitted is not None else False)
            if not bound:
                self._manager.fail_request(
                    request.request_id,
                    lambda: EngineError("engine did not provide request identity"))
                try:
                    self._engine.cancel_request(request.request_id,
                                                reason="engine_error")
                except BaseException:
                    pass
                return
            # admission 与断连可能并发：add/bind 完成后再次检查取消标志，
            # 仍只通过 Engine cancel 入口处理，不直接触碰底层资源。
            if self._manager.is_cancel_requested(request.request_id):
                self.cancel(request.request_id, reason="client_disconnected")
        except BaseException as exc:
            logger.error("add_request 失败: request_id=%s, %s: %s",
                         request.request_id, type(exc).__name__, exc)
            self._manager.fail_request(
                request.request_id,
                lambda: EngineError(
                    "request rejected by engine before scheduling"))

    def _consume_records(self) -> None:
        """按 token → completed → aborted 固定顺序排出控制面记录。"""
        pop_events = getattr(self._engine, "pop_token_events", None)
        if pop_events is not None:
            try:
                token_events = pop_events()
            except BaseException as exc:
                logger.error("读取 token 事件失败: %s: %s", type(exc).__name__, exc)
                token_events = []
            for event in token_events:
                try:
                    if not isinstance(event, TokenEvent):
                        # 兼容底层 dataclass 版本差异：以字段契约适配 DTO。
                        event = TokenEvent(
                            seq_id=event.seq_id, request_id=event.request_id,
                            round_id=event.round_id, token_ids=tuple(event.token_ids),
                            completion_index=event.completion_index,
                            emitted_at=event.emitted_at,
                            is_first_token=event.is_first_token,
                            is_final=getattr(event, "is_final", False),
                            finish_reason=getattr(event, "finish_reason", None),
                            phase=getattr(event, "phase", "decode"))
                    event_status = self._manager.resolve_token_event_status(event)
                    if event_status != "accepted":
                        # 未知/迟到/序号错误世代只能丢弃；只有当前句柄明确
                        # 报告背压时才取消，避免旧事件按 request_id 误伤新请求。
                        logger.warning("token event 未被流句柄接收: request_id=%s seq_id=%s status=%s",
                                       event.request_id, event.seq_id, event_status)
                        if event_status == "backpressure":
                            self.cancel(event.request_id, reason="stream_backpressure")
                except BaseException as exc:
                    logger.error("token 事件消费失败: %s: %s", type(exc).__name__, exc)
                    self._manager.fail_request(
                        getattr(event, "request_id", ""),
                        lambda: EngineError("failed to consume token event"))
        try:
            completed = self._engine.pop_completed()
        except BaseException as exc:
            logger.error("读取完成记录失败: %s: %s", type(exc).__name__, exc)
            completed = []
        for record in completed:
            try:
                self._manager.resolve_completed(record)
            except BaseException as exc:
                logger.error("完成记录消费失败: request_id=%s, %s: %s",
                             record.request_id, type(exc).__name__, exc)
                self._manager.fail_request(
                    record.request_id,
                    lambda: EngineError("failed to consume completion record"))
        try:
            aborted = self._engine.pop_aborted()
        except BaseException as exc:
            logger.error("读取中止记录失败: %s: %s", type(exc).__name__, exc)
            aborted = []
        for record in aborted:
            try:
                self._manager.resolve_aborted(record)
            except BaseException as exc:
                logger.error("中止记录消费失败: request_id=%s, %s: %s",
                             record.request_id, type(exc).__name__, exc)
                self._manager.fail_request(
                    record.request_id,
                    lambda: EngineError("failed to consume abort record"))

    def _shutdown_phase(self) -> None:
        """优雅退出：不再接受新命令 → 取消活动请求 → 有限 drain → engine.exit()。"""
        with self._status_lock:
            self._status = WorkerStatus.STOPPING
        self._manager.stop_accepting()
        # 1) 排空命令队列：排队中的 submit 未被 Engine 接受，直接失败收口，
        #    保证没有句柄因关闭而悬挂
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, _SubmitCommand):
                self._manager.fail_request(
                    command.request.request_id,
                    lambda: ServiceDrainingError(
                        "server is shutting down and not accepting requests"))
        # 2) 对仍未收口的句柄发送取消信号（Day10 signal 入口；真正的终态
        #    迁移与资源释放在调度边界/exit 兜底完成）
        for request_id in self._manager.pending_request_ids():
            try:
                self._engine.cancel_request(request_id, reason="server_shutdown")
            except BaseException as exc:
                logger.error("关闭阶段取消请求失败: request_id=%s, %s",
                             request_id, type(exc).__name__)
        # 3) 有限 drain：继续驱动 step 让取消/完成自然收口
        deadline = perf_counter() + self.DRAIN_TIMEOUT_SECONDS
        while self._engine.has_active_requests():
            if perf_counter() >= deadline:
                break
            try:
                self._engine.step()
            except BaseException as exc:
                logger.error("关闭阶段 step 异常: %s: %s",
                             type(exc).__name__, exc)
                break
            self._consume_records()
        # 4) engine.exit() 兜底清理（幂等；异常不吞掉，标记 worker 失败）
        self._exit_engine()
        # 5) 兜底收口：drain 中已完成/取消的句柄已被 resolve/fail，这里只
        #    处理仍未收口的（例如 drain 超时被 exit 中止的请求）
        self._consume_records()
        self._manager.fail_all(
            lambda: ServiceDrainingError(
                "server is shutting down and not accepting requests"))
        with self._status_lock:
            # exit 失败的关闭不能报告为干净停止（健康检查须保持非 ready）
            self._status = WorkerStatus.FAILED if self._exit_failed \
                else WorkerStatus.STOPPED

    def _teardown(self, *, mark_failed: bool) -> None:
        """异常退出路径的资源收尾：Engine 的 Day10 清理已在 step 事务内执行，
        这里调用 exit() 释放 runner/子进程，并失败剩余句柄。"""
        self._exit_engine()
        self._consume_records()
        self._manager.fail_all(self._make_engine_error_factory())
        with self._status_lock:
            self._status = WorkerStatus.FAILED if mark_failed \
                else WorkerStatus.STOPPED

    def _exit_engine(self) -> None:
        """幂等调用 engine.exit()；异常记录并标记失败，但不中断退出收尾。

        exit 异常不能吞掉（不能假装资源已释放）：记录日志并置 _exit_failed，
        worker 终态据此保持 FAILED，健康检查报告非 ready。
        """
        if self._engine_exited:
            return
        self._engine_exited = True
        try:
            self._engine.exit()
        except BaseException as exc:
            self._exit_failed = True
            logger.error("engine.exit() 失败: %s: %s", type(exc).__name__, exc)
