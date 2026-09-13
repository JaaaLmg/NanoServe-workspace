"""FastAPI app factory、lifespan 与统一异常处理（Day11–12 §5.2）。

create_app(engine_factory=None, server_config=None) 同时服务生产启动与测试注入：
- 生产：lifespan 调用默认工厂创建真实 LLMEngine 与 tokenizer；
- 测试：注入 fake factory，不访问 CUDA、NCCL、权重或 HuggingFace 网络。

startup 顺序：解析配置 → 创建 Engine → 创建 RequestManager/Worker →
worker ready → readiness=true。任意初始化异常：记录不含 prompt 的错误摘要、
readiness=false，并让异常继续向上抛出（uvicorn 感知启动失败，不会返回一个
表面 200 但内部没有 Engine 的服务）。

shutdown 顺序：停止接收新请求 → 请求 worker drain/cancel → engine.exit() →
join worker → 清理 lifespan 状态。

app state 只保存服务对象和只读配置，不保存单个请求的 Sequence 引用。
"""

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, NamedTuple

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from nanoserve import api
from nanoserve.observability import Observability
from nanoserve.config import ServerConfig
from nanoserve.service import APIError, EngineError, RequestManager
from nanoserve.worker import EngineWorker

logger = logging.getLogger(__name__)

# 生命周期状态常量：单一事实源在 ServiceState，路由/健康检查只读消费
STATUS_NOT_READY = "not_ready"
STATUS_READY = "ready"
STATUS_DRAINING = "draining"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


class EngineBundle(NamedTuple):
    """工厂产物：Engine、tokenizer 与只读的上下文上限快照。

    max_model_len 在工厂构造时一次性读取（只读配置），路由层不需要、也
    不允许再触碰 engine.model_runner 或 scheduler。
    """

    engine: Any
    tokenizer: Any
    max_model_len: int


def default_engine_factory(server_config: ServerConfig) -> EngineBundle:
    """生产工厂：创建真实 LLMEngine（含 tokenizer）。

    Engine 参数（TP/eager）由 ServerConfig 传入；max_model_len 等其余参数
    仍由 nanovllm.config.Config 的默认值与显式校验负责。
    """
    from nanovllm.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        server_config.model,
        tensor_parallel_size=server_config.tensor_parallel_size,
        enforce_eager=server_config.enforce_eager,
    )
    return EngineBundle(engine=engine, tokenizer=engine.tokenizer,
                        max_model_len=engine.max_model_len)


class ServiceState:
    """服务生命周期与只读依赖的单一事实源（app.state.service）。

    状态迁移：not_ready → ready → draining → stopped；任意阶段可进入 failed。
    健康语义（不变量 13）：只有 Engine 与 worker 都 ready 才返回 ok；
    draining/failed/stopped 一律报告非 ready。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.status = STATUS_NOT_READY
        self.model_id: str | None = None
        self.tokenizer = None
        self.manager: RequestManager | None = None
        self.worker = None
        self.engine = None
        self.server_config: ServerConfig | None = None
        # 只读服务依赖快照（路由构建请求用），startup 成功后不可变
        self.max_model_len = 0
        self.max_request_seconds: float | None = None
        # /v1/models 的 created 时间戳：进程内稳定
        self.created_at = int(time.time())
        self.observability = None

    def configure(self, *, model_id: str, tokenizer, manager: RequestManager,
                  worker, engine, server_config: ServerConfig,
                  max_model_len: int, observability=None) -> None:
        with self._lock:
            self.model_id = model_id
            self.tokenizer = tokenizer
            self.manager = manager
            self.worker = worker
            self.engine = engine
            self.server_config = server_config
            self.max_request_seconds = server_config.max_request_seconds
            self.max_model_len = max_model_len
            self.observability = observability

    def mark_ready(self) -> None:
        with self._lock:
            self.status = STATUS_READY

    def mark_draining(self) -> None:
        with self._lock:
            self.status = STATUS_DRAINING

    def mark_stopped(self) -> None:
        with self._lock:
            self.status = STATUS_STOPPED

    def mark_failed(self) -> None:
        with self._lock:
            self.status = STATUS_FAILED

    def health_snapshot(self) -> tuple[str, bool, str | None]:
        """返回 (health status, accepting_requests, model_id)。

        ready 需同时满足：状态为 ready 且 worker 线程仍存活且未失败——
        Engine 异常后 worker 退出，健康检查立即回落为 not_ready，不报告
        虚假 ready。
        """
        with self._lock:
            status = self.status
            worker = self.worker
            model_id = self.model_id
        if status == STATUS_READY and worker is not None \
                and worker.alive and worker.status.name == "RUNNING":
            return "ok", True, model_id
        if status == STATUS_READY and worker is not None \
                and worker.status.name == "FAILED":
            return "not_ready", False, model_id
        if status in (STATUS_DRAINING, STATUS_STOPPED):
            return "draining", False, model_id
        return "not_ready", False, model_id


class InvalidRequestShim(APIError):
    """校验错误的轻量构造（400 / invalid_request_error）。"""

    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request_error"


def create_app(engine_factory: Callable[[ServerConfig], EngineBundle] | None = None,
               server_config: ServerConfig | None = None) -> FastAPI:
    """创建 FastAPI 应用；engine_factory 供测试注入 fake Engine。"""
    factory = engine_factory or default_engine_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = ServiceState()
        app.state.service = state
        engine = manager = worker = None
        observer = Observability()
        try:
            config = server_config or ServerConfig.from_env()
            bundle = factory(config)
            engine = bundle.engine
            manager = RequestManager(bundle.tokenizer, observer=observer)
            worker = EngineWorker(engine, manager)
            manager.attach_worker(worker)
            state.configure(model_id=config.public_model_id,
                            tokenizer=bundle.tokenizer, manager=manager,
                            worker=worker, engine=engine,
                            server_config=config,
                            max_model_len=bundle.max_model_len,
                            observability=observer)
            worker.start()
            state.mark_ready()
            logger.info("NanoServe 就绪: model_id=%s, max_model_len=%d, "
                        "tensor_parallel_size=%d, enforce_eager=%s",
                        config.public_model_id, bundle.max_model_len,
                        config.tensor_parallel_size, config.enforce_eager)
            yield
        except BaseException as exc:
            # 启动失败：记录不含 prompt 的错误摘要，readiness 保持 false，
            # 异常继续上抛让 uvicorn 感知
            state.mark_failed()
            logger.error("服务启动失败: %s: %s", type(exc).__name__, exc)
            raise
        finally:
            # 启动失败已经是 FAILED；关闭清理不能把失败状态覆盖成
            # draining/stopped，否则健康检查会掩盖根因。
            startup_failed = state.status == STATUS_FAILED
            if not startup_failed:
                # 优雅关闭：停止接收 → worker drain/cancel → engine.exit() → join
                state.mark_draining()
            if manager is not None:
                manager.stop_accepting()
            if worker is not None:
                worker.begin_shutdown()
                worker.join(timeout=worker.DRAIN_TIMEOUT_SECONDS + 5.0)
            if worker is not None and worker.alive:
                # Engine 的所有破坏性操作必须由 worker 线程拥有；join 超时后
                # 不从 lifespan 线程并发调用 engine.exit()，避免与 step/exit 竞争。
                state.mark_failed()
                logger.error("worker 未在关闭期限内退出，无法安全回收 Engine")
            elif worker is not None and worker.status.name == "FAILED":
                # worker 已完成失败收尾时仍保留 FAILED，不能被 shutdown 的
                # 正常 STOPPED 状态覆盖。
                state.mark_failed()
            elif engine is not None and not getattr(worker, "_engine_exited", False):
                try:
                    # worker 未启动或启动前失败时没有 worker 所有者，只能由
                    # lifespan 做一次性兜底；正常 worker 路径已自行 exit。
                    engine.exit()
                except BaseException as exc:
                    state.mark_failed()
                    logger.error("engine.exit() 失败: %s: %s",
                                 type(exc).__name__, exc)
            if state.status != STATUS_FAILED:
                state.mark_stopped()
            logger.info("NanoServe 已停止")

    app = FastAPI(title="NanoServe OpenAI-Compatible API",
                  version="0.1.0", lifespan=lifespan)
    app.include_router(api.router)

    @app.get("/metrics")
    async def metrics(request: Request):
        service = request.app.state.service
        observer = getattr(service, "observability", None)
        if observer is None:
            return JSONResponse(status_code=503,
                                content={"error": {"message": "metrics unavailable"}})
        # 只读取 Engine 提供的公开标量快照，不触碰 Scheduler 私有对象。
        try:
            snapshot = service.engine.resource_snapshot()
            observer.refresh_resource_snapshot(snapshot)
        except Exception:
            # metrics 是旁路能力；Engine 快照失败不能影响抓取或请求处理。
            pass
        from fastapi.responses import Response
        return Response(content=observer.render_metrics(),
                        media_type="text/plain; version=0.0.4")

    # ---------- 统一异常处理（§3.5：同一输入错误不因分支不同而协议不一致） ----------

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        service = getattr(request.app.state, "service", None)
        observer = getattr(service, "observability", None)
        if observer is not None:
            try:
                observer.emit("request_rejected",
                              request_id=exc.request_id,
                              stage="service", error_code=exc.code)
            except Exception:
                pass
        return api._error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request,
                                       exc: RequestValidationError):
        # FastAPI 默认的 Pydantic 422 统一转换为 OpenAI 风格 400。
        # 只暴露字段路径与错误类型，不回显输入值（防止 prompt 明文泄漏）。
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(part) for part in first.get("loc", [])
                       if part != "body") or "request"
        error = InvalidRequestShim(
            f"invalid value for field {loc!r}: {first.get('type', 'invalid')}",
            param=loc)
        try:
            observer = getattr(request.app.state.service, "observability", None)
            if observer is not None:
                observer.emit("request_rejected", stage="validation",
                              error_code=error.code)
        except Exception:
            pass
        return api._error_response(error)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # 兜底 500：不向客户端返回内部堆栈；日志只记异常类型与消息
        logger.error("未处理异常: %s: %s", type(exc).__name__, exc)
        error = EngineError("internal server error")
        return api._error_response(error)

    return app
