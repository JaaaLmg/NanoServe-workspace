"""Day11–12 HTTP 路由、错误处理、生命周期与 OpenAI Python Client 集成测试
（docs/openai-api-day11-12.md §8.5 与 §8.6）。

覆盖范围：
- /health 在 ready/draining/failed 状态的状态码与 body；启动失败不报告 ready；
- /v1/models 返回公开 model ID，未知 model 不被动态加载；
- completion/chat 非流式响应的 OpenAI 字段、usage 守恒、finish_reason、
  x-request-id 与 response id 的固定关联规则；
- 非法字段/role、上下文超限（KV 分配前拒绝）、模型不匹配、模板错误、
  stream=true 的 501、Engine 异常的 500（无内部堆栈）；
- 错误 body 与捕获日志不含 prompt 明文；
- 优雅关闭：停止接收后新请求 503，活动请求不永久悬挂；
- OpenAI Python Client 指向本地 uvicorn 服务的真实 completion/chat 调用
  （CPU fake Engine；stream=True 的客户端行为留给 Day13，不在本任务断言）。

无 GPU/模型依赖：注入 FakeEngine/FakeTokenizer 工厂，不加载权重、不初始化
CUDA；OpenAI 集成测试通过真实 uvicorn 端口访问本服务。

重复执行命令：
    python -m pytest tests/test_service_api.py -q
    python -O -m pytest tests/test_service_api.py -q
"""

import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.testing import FakeEngine, FakeTokenizer
from nanoserve.testing import make_fake_factory
from nanoserve.worker import WorkerStatus

MODEL_ID = "Qwen3-0.6B"
SECRET_PROMPT = "SECRET-PROMPT-CONTENT-应不泄漏"


def make_config(**kwargs) -> ServerConfig:
    return ServerConfig(model="/tmp/fake-model", model_id=MODEL_ID, **kwargs)


def make_client(factory=None, config=None) -> TestClient:
    app = create_app(engine_factory=factory or make_fake_factory(),
                     server_config=config or make_config())
    return TestClient(app)


class TestHealthAndModels:
    """健康检查与模型信息（§8.5 前两组）。"""

    def test_health_ready(self):
        with make_client() as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body == {"status": "ok", "model": MODEL_ID,
                            "accepting_requests": True}

    def test_health_draining_and_not_ready(self):
        with make_client() as client:
            service = client.app.state.service
            service.mark_draining()
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "draining"
            assert resp.json()["accepting_requests"] is False
            service.mark_failed()
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "not_ready"

    def test_health_not_ok_after_engine_failure(self):
        """Engine 异常导致 worker 退出后：请求 500，健康检查立即回落非 ready。"""
        engine = FakeEngine(step_script=[RuntimeError("boom")])
        with make_client(make_fake_factory(engine=engine)) as client:
            resp = client.post("/v1/completions",
                               json={"model": MODEL_ID, "prompt": "x"})
            assert resp.status_code == 500
            assert resp.json()["error"]["code"] == "engine_error"
            resp = client.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "not_ready"

    def test_startup_failure_propagates(self):
        """工厂抛错：lifespan 启动失败向上传播，不产生表面可用的服务。"""

        def failing_factory(config):
            raise RuntimeError("model load failed")

        with pytest.raises(RuntimeError, match="model load failed"):
            with make_client(failing_factory):
                pass

    def test_models_lists_public_id_only(self):
        with make_client() as client:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "list"
            assert [item["id"] for item in body["data"]] == [MODEL_ID]
            assert body["data"][0]["object"] == "model"
            assert body["data"][0]["owned_by"] == "nanoserve"


class TestCompletionAndChatSuccess:
    """非流式成功响应契约。"""

    def test_completion_non_stream_shape(self):
        engine = FakeEngine()  # 默认 completion tokens 3 个 < max_tokens → stop
        with make_client(make_fake_factory(engine=engine)) as client:
            resp = client.post("/v1/completions", json={
                "model": MODEL_ID, "prompt": "hello", "max_tokens": 16,
                "temperature": 0.0})
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "text_completion"
            assert body["model"] == MODEL_ID
            assert body["id"].startswith("cmpl-")
            (choice,) = body["choices"]
            assert choice["index"] == 0
            assert choice["text"] == "text<3>"
            assert choice["logprobs"] is None
            assert choice["finish_reason"] == "stop"
            usage = body["usage"]
            assert usage["prompt_tokens"] == 5
            assert usage["completion_tokens"] == 3
            assert usage["total_tokens"] == \
                usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completion_length_finish_reason(self):
        engine = FakeEngine(completion_tokens=(1, 2, 3, 4))
        with make_client(make_fake_factory(engine=engine)) as client:
            resp = client.post("/v1/completions", json={
                "model": MODEL_ID, "prompt": "hello", "max_tokens": 4})
            (choice,) = resp.json()["choices"]
            assert choice["finish_reason"] == "length"
            assert resp.json()["usage"]["completion_tokens"] == 4

    def test_chat_non_stream_shape(self):
        with make_client() as client:
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": "你是简洁的助手。"},
                    {"role": "user", "content": "什么是 continuous batching？"},
                ],
                "max_tokens": 16, "temperature": 0.0})
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "chat.completion"
            assert body["model"] == MODEL_ID
            assert isinstance(body["created"], int)
            assert body["id"].startswith("chatcmpl-")
            (choice,) = body["choices"]
            assert choice["index"] == 0
            assert choice["message"]["role"] == "assistant"
            assert choice["message"]["content"] == "text<3>"
            # chat 响应不携带 completion 专属字段
            assert "logprobs" not in choice
            assert body["usage"]["total_tokens"] == \
                body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]

    def test_x_request_id_matches_response_id(self):
        """关联规则固定：成功响应 x-request-id == body.id。"""
        with make_client() as client:
            resp = client.post("/v1/completions",
                               json={"model": MODEL_ID, "prompt": "x"})
            assert resp.headers["x-request-id"] == resp.json()["id"]
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "x"}]})
            assert resp.headers["x-request-id"] == resp.json()["id"]

    def test_prompt_token_ids_passed_to_engine(self):
        """统一内部请求：Engine 收到 token 列表而非字符串（不重复编码）。"""
        engine = FakeEngine()
        with make_client(make_fake_factory(engine=engine)) as client:
            client.post("/v1/completions",
                        json={"model": MODEL_ID, "prompt": "abc"})
            (tokens, params, rid, deadline), = engine.added_requests
            assert tokens == tuple(FakeTokenizer().encode("abc"))
            assert deadline is None


class TestErrorContract:
    """错误响应：状态码、error code 与隐私边界。"""

    def test_model_mismatch_404(self):
        with make_client() as client:
            resp = client.post("/v1/completions",
                               json={"model": "other-model", "prompt": "x"})
            assert resp.status_code == 404
            assert resp.json()["error"]["code"] == "model_not_found"
            # 未支持的模型不会被动态加载
            models = client.get("/v1/models").json()
            assert [i["id"] for i in models["data"]] == [MODEL_ID]

    @pytest.mark.parametrize("route,payload", [
        ("/v1/completions", {"model": MODEL_ID, "prompt": "x", "n": 2}),
        ("/v1/completions", {"model": MODEL_ID}),
        ("/v1/completions", {"model": MODEL_ID, "prompt": ""}),
        ("/v1/completions", {"model": MODEL_ID, "prompt": "x",
                             "max_tokens": 0}),
        ("/v1/chat/completions", {"model": MODEL_ID, "messages": []}),
        ("/v1/chat/completions", {"model": MODEL_ID, "messages": [
            {"role": "tool", "content": "x"}]}),
        ("/v1/chat/completions", {"model": MODEL_ID, "messages": [
            {"role": "user", "content": "x"}, {"role": "user"}]}),
    ])
    def test_invalid_requests_return_unified_400(self, route, payload):
        """非法字段/role/空 messages 等统一 400 OpenAI 错误结构（422 被转换）。"""
        with make_client() as client:
            resp = client.post(route, json=payload)
            assert resp.status_code == 400
            error = resp.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert error["code"] == "invalid_request_error"
            assert error["param"]
            assert "x-request-id" in resp.headers

    def test_context_length_exceeded_before_kv_allocation(self):
        """上下文超限：400 context_length_exceeded，且请求不进入 Engine。"""
        engine = FakeEngine()
        factory = make_fake_factory(engine=engine, max_model_len=32)
        with make_client(factory) as client:
            resp = client.post("/v1/completions", json={
                "model": MODEL_ID, "prompt": "a" * 40, "max_tokens": 8})
            assert resp.status_code == 400
            assert resp.json()["error"]["code"] == "context_length_exceeded"
            assert engine.added_requests == []

    def test_stream_true_returns_sse_both_routes(self):
        """stream=true 进入 SSE，而不是静默退化或返回旧的 501。"""
        tokenizer = FakeTokenizer(token_text={21: "A", 22: "B", 23: "C"})
        engine = FakeEngine(tokenizer=tokenizer, completion_tokens=(21, 22, 23))
        with make_client(make_fake_factory(engine=engine)) as client:
            for route, payload, prefix in [
                ("/v1/completions", {"model": MODEL_ID, "prompt": "x",
                                     "stream": True}, "cmpl-"),
                ("/v1/chat/completions", {"model": MODEL_ID, "messages": [
                    {"role": "user", "content": "x"}], "stream": True},
                 "chatcmpl-"),
            ]:
                resp = client.post(route, json=payload)
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                assert resp.text.count("data: [DONE]") == 1
                assert prefix in resp.text

    def test_template_error_maps_to_400(self):
        tokenizer = FakeTokenizer(
            template_error=RuntimeError("jinja internal SECRET-DETAIL"))
        factory = make_fake_factory(tokenizer=tokenizer)
        with make_client(factory) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "x"}]})
            assert resp.status_code == 400
            error = resp.json()["error"]
            assert error["code"] == "invalid_request_error"
            assert error["param"] == "messages"
            # 模板异常内容不外泄
            assert "SECRET" not in resp.text

    def test_not_ready_returns_503(self):
        with make_client() as client:
            client.app.state.service.mark_failed()
            resp = client.post("/v1/completions",
                               json={"model": MODEL_ID, "prompt": "x"})
            assert resp.status_code == 503
            assert resp.json()["error"]["code"] == "service_not_ready"

    def test_engine_exception_returns_500_without_stack(self):
        """Engine 异常：500 engine_error，body 不含内部堆栈与 prompt 明文。"""
        engine = FakeEngine(step_script=[RuntimeError("boom")])
        with make_client(make_fake_factory(engine=engine)) as client:
            resp = client.post("/v1/completions", json={
                "model": MODEL_ID, "prompt": SECRET_PROMPT})
            assert resp.status_code == 500
            error = resp.json()["error"]
            assert error["code"] == "engine_error"
            assert "Traceback" not in resp.text
            assert "boom" not in resp.text

    def test_error_bodies_and_logs_contain_no_prompt(self, caplog):
        """错误 body 与捕获日志都不含 prompt/token 明文或 block table。"""
        import logging

        engine = FakeEngine(step_script=[RuntimeError("boom")])
        with make_client(make_fake_factory(engine=engine)) as client:
            with caplog.at_level(logging.ERROR):
                resp = client.post("/v1/completions", json={
                    "model": MODEL_ID, "prompt": SECRET_PROMPT})
            assert resp.status_code == 500
            assert SECRET_PROMPT not in resp.text
            assert SECRET_PROMPT not in caplog.text
            assert "block_table" not in caplog.text


class TestLifecycleViaHTTP:
    """HTTP 视角的生命周期：关闭后 503、lifespan 幂等。"""

    def test_draining_rejects_new_requests(self):
        with make_client() as client:
            service = client.app.state.service
            service.mark_draining()
            service.manager.stop_accepting()
            resp = client.post("/v1/completions",
                               json={"model": MODEL_ID, "prompt": "x"})
            assert resp.status_code == 503
            assert resp.json()["error"]["code"] == "service_draining"

    def test_lifespan_shutdown_cleans_up_once(self):
        engine = FakeEngine()
        with make_client(make_fake_factory(engine=engine)) as client:
            service = client.app.state.service
            handle_target = service
        # with 退出后：worker 已停止，engine.exit() 恰好一次（幂等契约，
        # lifespan 兜底调用对已退出的 Engine 是安全空操作）
        assert handle_target.worker.status is WorkerStatus.STOPPED
        assert engine.exit_calls == 1

    def test_shutdown_during_active_request_does_not_hang(self):
        """关闭时活动请求得到明确错误，HTTP 等待不永久悬挂。"""
        engine = FakeEngine(step_script=[("pending",)] * 50, step_delay=0.01)

        result: dict = {}

        def issue_request(client):
            try:
                resp = client.post("/v1/completions", json={
                    "model": MODEL_ID, "prompt": "x"})
                result["status"] = resp.status_code
            except Exception as exc:  # TestClient 在 lifespan 关闭时可能抛错
                result["error"] = type(exc).__name__

        app = create_app(engine_factory=make_fake_factory(engine=engine),
                         server_config=make_config())
        with TestClient(app) as client:
            worker = client.app.state.service.worker
            thread = threading.Thread(target=issue_request, args=(client,))
            thread.start()
            # 等待请求进入 Engine 后触发优雅关闭
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not engine.has_active_requests():
                time.sleep(0.005)
            client.app.state.service.mark_draining()
            client.app.state.service.manager.stop_accepting()
            worker.begin_shutdown()
            worker.join(timeout=5)
            thread.join(timeout=5)
        assert not thread.is_alive()
        # 请求最终得到 503 draining（或 lifespan 关闭导致的连接终止），绝非悬挂
        assert result.get("status") == 503 or "error" in result


@pytest.fixture(scope="module")
def openai_server():
    """启动一次本地 uvicorn 服务（fake Engine），供 OpenAI Client 集成测试复用。"""
    pytest.importorskip("openai")
    import uvicorn

    app = create_app(engine_factory=make_fake_factory(),
                     server_config=make_config())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn 未在时限内启动"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


class TestOpenAIClientIntegration:
    """OpenAI Python Client 指向本地 uvicorn 服务（§8.6，CPU fake Engine）。"""

    def test_openai_client_completion(self, openai_server):
        from openai import OpenAI

        client = OpenAI(base_url=f"{openai_server}/v1", api_key="test")
        completion = client.completions.create(
            model=MODEL_ID, prompt="hello", max_tokens=8, temperature=0)
        assert completion.object == "text_completion"
        assert completion.model == MODEL_ID
        assert completion.choices[0].text == "text<3>"
        assert completion.usage.total_tokens == \
            completion.usage.prompt_tokens + completion.usage.completion_tokens

    def test_openai_client_chat(self, openai_server):
        from openai import OpenAI

        client = OpenAI(base_url=f"{openai_server}/v1", api_key="test")
        chat = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8, temperature=0)
        assert chat.object == "chat.completion"
        assert chat.choices[0].message.role == "assistant"
        assert chat.choices[0].message.content == "text<3>"
        assert chat.choices[0].finish_reason == "stop"
