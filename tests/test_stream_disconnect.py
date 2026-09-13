"""Day13 断连/取消安全边界的 CPU 测试（无 GPU/模型）。

重复执行：python -m pytest tests/test_stream_disconnect.py -q
"""
import time

from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

MODEL = "Qwen3-0.6B"


def test_worker_level_disconnect_signal_and_cleanup():
    """断连只经过 manager/worker 高层入口，最终 active 归零。"""
    from nanoserve.service import RequestManager, build_completion_request
    from nanoserve.schemas import CompletionRequest
    from nanoserve.worker import EngineWorker

    engine = FakeEngine(step_delay=0.01, step_script=[("pending",)] * 20)
    manager = RequestManager(engine.tokenizer)
    worker = EngineWorker(engine, manager)
    manager.attach_worker(worker)
    worker.start()
    request = build_completion_request(
        CompletionRequest(model=MODEL, prompt="x"), engine.tokenizer,
        model_id=MODEL, max_model_len=4096)
    handle = manager.submit(request, stream=True)
    try:
        deadline = time.monotonic() + 2
        while not engine.active and time.monotonic() < deadline:
            time.sleep(0.005)
        assert worker.cancel(request.request_id, "client_disconnected")
        deadline = time.monotonic() + 2
        while engine.active and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not engine.active
        assert engine.cancel_calls[-1] == (request.request_id, "client_disconnected")
        deadline = time.monotonic() + 2
        while manager.pending_request_ids() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert manager.pending_request_ids() == []
    finally:
        worker.begin_shutdown()
        worker.join(timeout=3)
        assert not worker.alive


def test_sse_stream_can_be_closed_without_leaking_worker():
    tok = FakeTokenizer(token_text={21: "A"})
    engine = FakeEngine(tokenizer=tok, completion_tokens=(21,), step_delay=0.02,
                        step_script=[("pending",)] * 20)
    factory = make_fake_factory(engine=engine)
    app = create_app(engine_factory=factory,
                     server_config=ServerConfig(model="/tmp/fake", model_id=MODEL))
    with TestClient(app) as c:
        with c.stream("POST", "/v1/completions", json={
                "model": MODEL, "prompt": "x", "stream": True}) as response:
            assert response.status_code == 200
            next(response.iter_lines())
        # TestClient 的连接模拟不保证立即产生 disconnect，至少确认关闭路径
        # 不会让生命周期 shutdown 悬挂。
    assert factory.engine.exit_calls == 1
