"""Day13–14 高风险边界回归：异常、序号、背压和 prefix metrics。

CPU-only；重复执行：python -m pytest tests/test_day13_14_edge_cases.py -q
"""
import json
import time

import pytest
from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.service import StreamBackpressureError, StreamHandle, TokenEvent
from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

MODEL = "Qwen3-0.6B"


def make_app(engine):
    return create_app(engine_factory=make_fake_factory(engine=engine),
                     server_config=ServerConfig(model="/tmp/fake", model_id=MODEL))


def test_admission_failure_is_json_before_sse_frame():
    engine = FakeEngine(fail_add_request=True)
    with TestClient(make_app(engine)) as client:
        response = client.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "stream": True})
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "engine_error"
    assert "data:" not in response.text


def test_stream_handle_keeps_terminal_replayable():
    handle = StreamHandle("cmpl-test", "completion", 1.0, maxsize=2)
    record = type("Record", (), {})()
    terminal = __import__("nanoserve.service", fromlist=["StreamTerminal"]).StreamTerminal(record=record)
    assert handle._put_terminal(terminal)
    assert handle.next_event(timeout=0.1) is record
    assert handle.next_event(timeout=0.1) is record


def test_stream_backpressure_does_not_drop_buffered_token():
    handle = StreamHandle("cmpl-test", "completion", 1.0, maxsize=1)
    event = TokenEvent(1, "cmpl-test", 1, (21,), 0, 1.0, True)
    assert handle.publish(event)
    assert not handle.finish(type("Record", (), {})())
    assert handle.next_event(timeout=0.1) is event
    with pytest.raises(StreamBackpressureError):
        handle.next_event(timeout=0.1)


def test_metrics_prefix_counters_are_refreshed_from_engine_snapshot():
    engine = FakeEngine()
    # Fake resource snapshot exposes the same scalar names as the real Engine.
    engine.resource.update({
        "prefix_cache_lookups": 4,
        "prefix_cache_hits": 2,
        "prefix_cache_misses": 1,
        "prefix_cache_capacity_failures": 1,
    })
    with TestClient(make_app(engine)) as client:
        metrics = client.get("/metrics").text
    assert "prefix_cache_lookups_total 4.0" in metrics
    assert "prefix_cache_hits_total 2.0" in metrics
    assert "prefix_cache_misses_total 1.0" in metrics
    assert "prefix_cache_capacity_failures_total 1.0" in metrics


def test_token_event_index_gap_is_rejected():
    from nanoserve.service import RequestManager
    manager = RequestManager(FakeTokenizer())
    class Worker:
        def submit(self, request): pass
    manager.attach_worker(Worker())
    from nanoserve.schemas import CompletionRequest
    from nanoserve.service import build_completion_request
    request = build_completion_request(CompletionRequest(model=MODEL, prompt="x"),
                                       FakeTokenizer(), model_id=MODEL,
                                       max_model_len=4096)
    handle = manager.submit(request, stream=True)
    assert manager.bind_seq_id(request.request_id, 1)
    gap = TokenEvent(1, request.request_id, 1, (21,), 1, 1.0, False)
    assert not manager.resolve_token_event(gap)
    assert not handle.stream.closed
