"""HTTP 路由与 OpenAI 兼容 SSE 流式输出。

Engine 仍只由 EngineWorker 驱动；SSE generator 只消费 RequestManager 的
StreamHandle，并在连接断开时通过 worker.cancel 发出 Day10 signal-only 信号。
"""

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanoserve.schemas import (ChatCompletionRequest, ChatChoice,
                               ChatAssistantMessage, ChatCompletionResponse,
                               CompletionChoice, CompletionRequest,
                               CompletionResponse, Usage)
from nanoserve.service import (APIError, ModelNotFoundError,
                               ServiceDrainingError, ServiceNotReadyError,
                               StreamNotImplementedError, TokenEvent,
                               build_chat_request, build_completion_request)
from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest

router = APIRouter()

_STATUS_READY = "ready"
_STATUS_DRAINING = "draining"


def _service_state(request: Request):
    return request.app.state.service


def _ensure_ready(service) -> None:
    health_status, accepting, _ = service.health_snapshot()
    if health_status == "ok" and accepting:
        return
    if service.status in (_STATUS_DRAINING, "stopped"):
        raise ServiceDrainingError(
            "server is shutting down and not accepting requests")
    raise ServiceNotReadyError("service engine is not ready")


def _ensure_model(service, model: str) -> None:
    if model != service.model_id:
        raise ModelNotFoundError(f"model {model!r} not found", param="model")


def _ensure_not_stream(stream: bool) -> None:
    """保留兼容符号；stream=true 现在由各路由进入 SSE 分支。"""
    if stream:
        raise StreamNotImplementedError(
            "streaming responses are not implemented yet", param="stream")


def _error_response(exc: APIError) -> JSONResponse:
    request_id = exc.request_id or f"req-{uuid.uuid4().hex}"
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload(),
                        headers={"x-request-id": request_id})


def _sse(payload: dict) -> str:
    """SSE data frame；JSON 中不允许换行穿透 frame 边界。"""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n"


def _done() -> str:
    return "data: [DONE]\n\n"


def _decode_delta(tokenizer, event: TokenEvent) -> str:
    # 使用同一 tokenizer 解码真实增量 token；不记录 token 内容到日志。
    return tokenizer.decode(list(event.token_ids))


def _completion_chunk(request_id: str, model: str, created: int, text: str = "",
                      finish_reason: str | None = None, usage: dict | None = None):
    payload = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _chat_chunk(request_id: str, model: str, created: int, delta: dict,
                finish_reason: str | None = None, usage: dict | None = None):
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


async def _stream_body(request: Request, service, internal, handle, *, chat: bool):
    """消费一个 StreamHandle 并序列化 SSE；不在事件循环中阻塞队列。"""
    created = int(time.time())
    terminal_seen = False
    first = (_chat_chunk(internal.request_id, service.model_id, created,
                         {"role": "assistant", "content": ""})
             if chat else
             _completion_chunk(internal.request_id, service.model_id, created))
    try:
        # admission 先于首帧，避免 add_request/worker 失败后仍返回 200 首帧。
        if not handle.wait_admission(timeout=5.0):
            return
        if handle.terminal_error is not None:
            return
        yield _sse(first)
        while True:
            if await request.is_disconnected():
                break
            try:
                item = await asyncio.to_thread(handle.next_event, 0.25)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except APIError:
                # 首 chunk 已经发出，不能再追加成功 finish 或 [DONE]。
                terminal_seen = True
                return
            if isinstance(item, TokenEvent):
                text = _decode_delta(service.tokenizer, item)
                if not text:
                    continue
                payload = (_chat_chunk(internal.request_id, service.model_id, created,
                                       {"content": text}) if chat else
                           _completion_chunk(internal.request_id, service.model_id,
                                             created, text=text))
                yield _sse(payload)
                continue
            if isinstance(item, CompletedRequest):
                terminal_seen = True
                usage = {
                    "prompt_tokens": item.prompt_tokens,
                    "completion_tokens": item.completion_tokens,
                    "total_tokens": item.prompt_tokens + item.completion_tokens,
                }
                payload = (_chat_chunk(internal.request_id, service.model_id, created,
                                       {}, item.finish_reason, usage) if chat else
                           _completion_chunk(internal.request_id, service.model_id,
                                             created, finish_reason=item.finish_reason,
                                             usage=usage))
                yield _sse(payload)
                yield _done()
                return
            if isinstance(item, AbortedRequest):
                terminal_seen = True
                return
            # StreamHandle 的兼容实现可能返回未知终态；安全结束而不伪造成功。
            terminal_seen = True
            return
    finally:
        # generator 被客户端取消/断连时只走 worker 的公开取消入口；完成或
        # 已中止的请求再次 cancel 是幂等的，避免网络线程触碰 Scheduler/KV。
        if not terminal_seen:
            try:
                service.manager.cancel(internal.request_id,
                                       reason="client_disconnected")
                observer = getattr(service, "observability", None)
                if observer is not None:
                    observer.emit("request_disconnected", request_id=internal.request_id,
                                  kind=internal.kind, reason="client_disconnected")
            except Exception:
                pass


def _streaming_response(request: Request, service, internal, *, chat: bool):
    handle = service.manager.submit(internal, stream=True)
    stream = handle.stream
    if stream is None:
        raise RuntimeError("stream handle was not created")
    # 路由在线程池中运行；在创建 StreamingResponse 前等待 admission，
    # 让 add_request/首轮 Engine 错误仍能映射为普通 JSON 错误。
    if not stream.wait_admission(timeout=5.0):
        service.manager.cancel(internal.request_id, reason="admission_timeout")
        raise ServiceNotReadyError("request admission timed out")
    if stream.terminal_error is not None:
        error = stream.terminal_error
        if isinstance(error, APIError):
            raise error
        raise ServiceNotReadyError("request was rejected before streaming")
    return StreamingResponse(
        _stream_body(request, service, internal, stream, chat=chat),
        media_type="text/event-stream",
        headers={"x-request-id": internal.request_id,
                 "Cache-Control": "no-cache",
                 "Connection": "keep-alive"},
    )


@router.get("/health")
def health(request: Request):
    service = _service_state(request)
    status, accepting, model_id = service.health_snapshot()
    body = {"status": status, "model": model_id,
            "accepting_requests": accepting}
    return JSONResponse(status_code=200 if status == "ok" else 503,
                        content=body)


@router.get("/v1/models")
def models(request: Request):
    service = _service_state(request)
    _ensure_ready(service)
    return {
        "object": "list",
        "data": [{"id": service.model_id, "object": "model",
                  "created": service.created_at, "owned_by": "nanoserve"}],
    }


@router.post("/v1/completions")
def completions(body: CompletionRequest, request: Request):
    service = _service_state(request)
    _ensure_ready(service)
    _ensure_model(service, body.model)
    internal = build_completion_request(
        body, service.tokenizer, model_id=service.model_id,
        max_model_len=service.max_model_len,
        max_request_seconds=service.max_request_seconds)
    observer = getattr(service, "observability", None)
    if observer is not None:
        observer.emit("request_received", request_id=internal.request_id,
                      kind=internal.kind, model_id=internal.model_id,
                      prompt_tokens=len(internal.prompt_token_ids),
                      max_tokens=body.max_tokens, observed_at=internal.created_at)
    if body.stream:
        return _streaming_response(request, service, internal, chat=False)
    handle = service.manager.submit(internal)
    result = service.manager.wait(handle)
    response = CompletionResponse(
        id=result.request_id, created=int(time.time()), model=service.model_id,
        choices=[CompletionChoice(text=result.text,
                                  finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens),
    )
    return JSONResponse(content=response.model_dump(),
                        headers={"x-request-id": result.request_id})


@router.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest, request: Request):
    service = _service_state(request)
    _ensure_ready(service)
    _ensure_model(service, body.model)
    internal = build_chat_request(
        body, service.tokenizer, model_id=service.model_id,
        max_model_len=service.max_model_len,
        max_request_seconds=service.max_request_seconds)
    observer = getattr(service, "observability", None)
    if observer is not None:
        observer.emit("request_received", request_id=internal.request_id,
                      kind=internal.kind, model_id=internal.model_id,
                      prompt_tokens=len(internal.prompt_token_ids),
                      max_tokens=body.max_tokens, observed_at=internal.created_at)
    if body.stream:
        return _streaming_response(request, service, internal, chat=True)
    handle = service.manager.submit(internal)
    result = service.manager.wait(handle)
    response = ChatCompletionResponse(
        id=result.request_id, created=int(time.time()), model=service.model_id,
        choices=[ChatChoice(message=ChatAssistantMessage(content=result.text),
                             finish_reason=result.finish_reason)],
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.prompt_tokens + result.completion_tokens),
    )
    return JSONResponse(content=response.model_dump(),
                        headers={"x-request-id": result.request_id})
