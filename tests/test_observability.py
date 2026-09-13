"""Day14 metrics 与结构化日志的 CPU/ASGI 验收测试。

重复执行：python -m pytest tests/test_observability.py -q
"""
import json
import logging
import pytest

from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.observability import Observability, RequestTimeline
from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

MODEL = "Qwen3-0.6B"


def test_metrics_contains_required_names_after_request():
    engine = FakeEngine(tokenizer=FakeTokenizer(token_text={21: "A"}),
                        completion_tokens=(21,))
    app = create_app(engine_factory=make_fake_factory(engine=engine),
                     server_config=ServerConfig(model="/tmp/fake", model_id=MODEL))
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "max_tokens": 1})
        assert response.status_code == 200
        metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    for name in ("request_queue_time_seconds", "time_to_first_token_seconds",
                 "time_per_output_token_seconds", "request_latency_seconds",
                 "prompt_tokens_total", "generation_tokens_total",
                 "kv_cache_utilization", "prefix_cache_hit_rate",
                 "running_requests"):
        assert f"# HELP {name}" in text


def test_timeline_does_not_fabricate_itl():
    timeline = RequestTimeline.start(kind="completion", prompt_tokens=2,
                                     clock=lambda: 1.0)
    timeline.mark_admitted(1.2)
    timeline.mark_token(1.5)
    assert timeline.time_to_first_token_seconds == 0.5
    assert timeline.inter_token_latencies == ()
    timeline.mark_token(1.8)
    assert timeline.inter_token_latencies == pytest.approx((0.3,))


def test_structured_log_drops_sensitive_fields():
    records = []
    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    log = logging.getLogger("test-observability")
    log.handlers[:] = [Handler()]
    log.setLevel(logging.INFO)
    observer = Observability(log=log)
    observer.emit("request_received", request_id="req-1", prompt="secret",
                  text="secret-text", token_ids=[1, 2], kind="completion")
    payload = json.loads(records[-1])
    assert payload["request_id"] == "req-1"
    assert "prompt" not in payload and "text" not in payload and "token_ids" not in payload
