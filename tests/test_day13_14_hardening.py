"""Day13–14 验收加固：世代、竞态、有限等待和观测收口。

CPU-only；重复执行：python -m pytest tests/test_day13_14_hardening.py -q
"""
import time
from types import SimpleNamespace

import pytest

from nanoserve.observability import Observability
from nanoserve.schemas import CompletionRequest
from nanoserve.service import (EngineError, RequestManager, RequestTimeoutError,
                               StreamHandle, TokenEvent, build_completion_request)
from nanoserve.testing import FakeEngine, FakeTokenizer
from nanoserve.worker import EngineWorker
from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest
from nanovllm.sampling_params import SamplingParams

MODEL = "Qwen3-0.6B"


def request(rid="req-1"):
    return build_completion_request(
        CompletionRequest(model=MODEL, prompt="x", max_tokens=2),
        FakeTokenizer(), model_id=MODEL, max_model_len=4096)


def test_stale_event_does_not_cancel_reused_generation():
    engine = FakeEngine(step_script=[("pending",), []])
    manager = RequestManager(engine.tokenizer)
    worker = EngineWorker(engine, manager)
    manager.attach_worker(worker)
    worker.start()
    first = request("reuse")
    h1 = manager.submit(first, stream=True)
    assert manager.bind_seq_id(first.request_id, 1)
    # 收口第一代，通过直接终态记录模拟 worker drain。
    manager.resolve_completed(CompletedRequest(
        seq_id=1, request_id="reuse", completion_token_ids=(21,),
        prompt_tokens=1, completion_tokens=1, finish_reason="stop", finished_at=1.0))
    second = request("reuse")
    h2 = manager.submit(second, stream=True)
    assert manager.bind_seq_id(second.request_id, 2)
    engine.token_events.append(TokenEvent(1, "reuse", 1, (21,), 0, 1.0, True))
    worker._consume_records()
    assert not h2.stream.closed
    assert not any(rid == "reuse" for rid, _ in engine.cancel_calls)
    worker.begin_shutdown()
    worker.join(timeout=3)
    assert not worker.alive


def test_unbound_stream_record_is_ignored_without_resolution():
    manager = RequestManager(FakeTokenizer())
    manager.attach_worker(SimpleNamespace(submit=lambda request: None))
    req = request("unbound")
    handle = manager.submit(req, stream=True)
    manager.resolve_completed(CompletedRequest(
        seq_id=9, request_id=req.request_id, completion_token_ids=(1,),
        prompt_tokens=1, completion_tokens=1, finish_reason="stop", finished_at=1.0))
    assert not handle.future.done()
    assert req.request_id in manager.pending_request_ids()


def test_wait_timeout_is_bounded_and_removes_handle():
    manager = RequestManager(FakeTokenizer())
    manager.attach_worker(SimpleNamespace(
        submit=lambda request: None,
        cancel=lambda request_id, reason: False))
    handle = manager.submit(request("timeout"))
    started = time.monotonic()
    with pytest.raises(RequestTimeoutError):
        manager.wait(handle, timeout=0.01)
    assert time.monotonic() - started < 1.5
    assert manager.pending_request_ids() == []


def test_prompt_counter_is_recorded_at_admission():
    observer = Observability()
    engine = FakeEngine(step_script=[("pending",)] * 10)
    manager = RequestManager(engine.tokenizer, observer=observer)
    worker = EngineWorker(engine, manager)
    manager.attach_worker(worker)
    worker.start()
    handle = manager.submit(request("admit"))
    deadline = time.monotonic() + 2
    while handle.seq_id is None and time.monotonic() < deadline:
        time.sleep(0.005)
    rendered = observer.render_metrics().decode()
    assert 'prompt_tokens_total{kind="completion"} 1.0' in rendered
    worker.cancel("admit", "client_cancelled")
    worker.begin_shutdown()
    worker.join(timeout=3)
    assert not worker.alive
