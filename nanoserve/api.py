"""HTTP 路由：/health、/v1/models 与两个非流式 POST 路由（Day11–12 §3）。

路由层只做：就绪检查 → 模型匹配 → stream 边界 → 构建统一内部请求 →
提交并等待结果 → 包装 OpenAI 风格响应。不触碰 Sequence.status、
block_table、BlockManager 或 Scheduler 队列——状态与资源仍由 Day10 的
底层单一权威管理。

x-request-id 关联规则（固定且可测试）：
- 成功响应：x-request-id == 响应体 id（即传给 Engine 的 request_id）；
- 请求已生成 ID 后失败的错误：x-request-id == 该请求 ID；
- 校验阶段（尚未生成 ID）的错误：x-request-id 为随机 req-<hex>。

路由为同步 def：FastAPI 在线程池中执行，阻塞等待 Future 不会卡死事件循环；
Engine worker 仍只有一个 step 驱动者（continuous batching 不被 HTTP 拆散）。
"""

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from nanoserve.schemas import (ChatCompletionRequest, ChatChoice,
                               ChatAssistantMessage, ChatCompletionResponse,
                               CompletionChoice, CompletionRequest,
                               CompletionResponse, Usage)
from nanoserve.service import (APIError, ModelNotFoundError,
                               ServiceDrainingError, ServiceNotReadyError,
                               StreamNotImplementedError, build_chat_request,
                               build_completion_request)

router = APIRouter()

# 生命周期状态常量（与 app.ServiceState 对齐的只读字符串视图）
_STATUS_READY = "ready"
_STATUS_DRAINING = "draining"
_STATUS_NOT_READY = "not_ready"


def _service_state(request: Request):
    """读取 app.state 中由 lifespan 装配的服务状态对象。"""
    return request.app.state.service


def _ensure_ready(service) -> None:
    """就绪检查与健康检查共享 worker 状态，避免死线程仍接纳请求。"""
    health_status, accepting, _ = service.health_snapshot()
    if health_status == "ok" and accepting:
        return
    if service.status in (_STATUS_DRAINING, "stopped"):
        raise ServiceDrainingError(
            "server is shutting down and not accepting requests")
    # worker FAILED 或线程意外退出属于服务不可用，而不是正常排空。
    raise ServiceNotReadyError("service engine is not ready")


def _ensure_model(service, model: str) -> None:
    """模型匹配检查：不支持请求内动态切换模型。"""
    if model != service.model_id:
        raise ModelNotFoundError(f"model {model!r} not found", param="model")


def _ensure_not_stream(stream: bool) -> None:
    """stream 边界：Day11–12 只交付 stream=false；true 显式 501，绝不静默
    退化为非流式响应。"""
    if stream:
        raise StreamNotImplementedError(
            "streaming responses are not implemented yet (planned for a "
            "later release)", param="stream")


def _error_response(exc: APIError) -> JSONResponse:
    """统一错误响应：OpenAI 风格错误体 + x-request-id 头（规则见模块 docstring）。"""
    request_id = exc.request_id or f"req-{uuid.uuid4().hex}"
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload(),
                        headers={"x-request-id": request_id})


@router.get("/health")
def health(request: Request):
    """健康检查：只反映服务生命周期状态，不把单个请求的完成当作 readiness。

    - Engine/worker 均就绪：200 status="ok"；
    - 正在优雅关闭或已停止：503 status="draining"；
    - 尚未启动完成或启动/运行失败：503 status="not_ready"。
    """
    service = _service_state(request)
    status, accepting, model_id = service.health_snapshot()
    body = {"status": status, "model": model_id,
            "accepting_requests": accepting}
    if status == "ok":
        return JSONResponse(status_code=200, content=body)
    return JSONResponse(status_code=503, content=body)


@router.get("/v1/models")
def models(request: Request):
    """当前单模型的 OpenAI 风格列表；不提供动态模型加载/删除。"""
    service = _service_state(request)
    _ensure_ready(service)
    return {
        "object": "list",
        "data": [{
            "id": service.model_id,
            "object": "model",
            "created": service.created_at,
            "owned_by": "nanoserve",
        }],
    }


@router.post("/v1/completions")
def completions(body: CompletionRequest, request: Request):
    """非流式 text completion（Day11）。"""
    service = _service_state(request)
    _ensure_ready(service)
    _ensure_model(service, body.model)
    _ensure_not_stream(body.stream)
    internal = build_completion_request(
        body, service.tokenizer, model_id=service.model_id,
        max_model_len=service.max_model_len,
        max_request_seconds=service.max_request_seconds)
    handle = service.manager.submit(internal)
    result = service.manager.wait(handle)
    response = CompletionResponse(
        id=result.request_id,
        created=int(time.time()),
        model=service.model_id,
        choices=[CompletionChoice(text=result.text,
                                  finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens
                    + result.completion_tokens),
    )
    return JSONResponse(content=response.model_dump(),
                        headers={"x-request-id": result.request_id})


@router.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest, request: Request):
    """非流式 chat completion（Day12）：模板转换 + assistant message 包装。"""
    service = _service_state(request)
    _ensure_ready(service)
    _ensure_model(service, body.model)
    _ensure_not_stream(body.stream)
    internal = build_chat_request(
        body, service.tokenizer, model_id=service.model_id,
        max_model_len=service.max_model_len,
        max_request_seconds=service.max_request_seconds)
    handle = service.manager.submit(internal)
    result = service.manager.wait(handle)
    response = ChatCompletionResponse(
        id=result.request_id,
        created=int(time.time()),
        model=service.model_id,
        choices=[ChatChoice(
            message=ChatAssistantMessage(content=result.text),
            finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens
                    + result.completion_tokens),
    )
    return JSONResponse(content=response.model_dump(),
                        headers={"x-request-id": result.request_id})
