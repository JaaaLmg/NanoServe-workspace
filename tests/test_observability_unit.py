"""Day14 observability 纯 CPU 单元测试。

不加载模型/GPU；重复执行：``python -m pytest tests/test_observability_unit.py -q``。
覆盖独立 registry、时间线口径、资源快照、低基数标签、日志脱敏和异常隔离。
"""

import json
import logging

import pytest
from prometheus_client import CollectorRegistry

from nanoserve.observability import (METRIC_NAMES, LifecycleLogger,
                                     Observability, ResourceSnapshot,
                                     RequestTimeline, lifecycle_envelope,
                                     validate_labels)


def _samples(metrics: Observability) -> str:
    return metrics.render_metrics().decode("utf-8")


def test_registry_isolated_and_exports_planned_metrics():
    first = Observability()
    second = Observability()
    assert first.registry is not second.registry
    text = _samples(first)
    assert all(name in text for name in METRIC_NAMES)


def test_timeline_uses_monotonic_points_and_no_fake_itl():
    metrics = Observability(clock=lambda: 100.0)
    timeline = RequestTimeline(request_id="r", kind="completion",
                               submitted_at=10.0, prompt_tokens=4)
    timeline.mark_admitted(12.0)
    timeline.mark_first_token(15.0)
    timeline.mark_finished(20.0)
    metrics.record_timeline(timeline)
    text = _samples(metrics)
    assert timeline.queue_wait_seconds == 2.0
    assert timeline.time_to_first_token_seconds == 5.0
    assert timeline.inter_token_latencies == ()
    assert 'request_queue_time_seconds_count{kind="completion"} 1.0' in text
    # 没有第二个真实 token 时，不创建/观测伪造的 ITL 样本。
    assert 'time_per_output_token_seconds_count{kind="completion"}' not in text
    assert 'prompt_tokens_total{kind="completion"} 4.0' in text
    assert 'generation_tokens_total{kind="completion"} 1.0' in text


def test_repeated_record_does_not_double_count_and_real_tokens_have_itl():
    metrics = Observability()
    timeline = metrics.start_request("r", prompt_tokens=2)
    timeline.mark_admitted(1.0)
    timeline.mark_token(2.0)
    timeline.mark_token(2.25)
    timeline.mark_finished(3.0)
    metrics.record_timeline(timeline)
    metrics.record_timeline(timeline)
    text = _samples(metrics)
    assert 'prompt_tokens_total{kind="completion"} 2.0' in text
    assert 'generation_tokens_total{kind="completion"} 2.0' in text
    assert 'time_per_output_token_seconds_count{kind="completion"} 1.0' in text


def test_resource_and_prefix_snapshot_semantics():
    metrics = Observability()
    snapshot = metrics.refresh_resource_snapshot(
        {"running": 3, "used_blocks": 4, "total_blocks": 8})
    assert snapshot.kv_cache_utilization == 0.5
    metrics.record_prefix_lookup(hit=True)
    metrics.record_prefix_lookup(hit=False)
    metrics.record_prefix_lookup(capacity_failure=True)
    text = _samples(metrics)
    assert 'running_requests 3.0' in text
    assert 'kv_cache_utilization 0.5' in text
    assert 'prefix_cache_hit_rate 0.3333333333333333' in text
    assert 'prefix_cache_capacity_failures_total 1.0' in text
    assert 'prefix_cache_misses_total 1.0' in text

    zero = ResourceSnapshot(used_blocks=99, total_blocks=0)
    assert zero.kv_cache_utilization == 0.0


def test_labels_and_json_envelope_are_low_cardinality_and_redacted():
    validate_labels(kind="chat", status="cancelled", phase="decode")
    with pytest.raises(ValueError):
        validate_labels(kind="request-id-is-not-a-label")
    payload = lifecycle_envelope(
        "request_token", observed_at=42.0, request_id="r", seq_id=1,
        prompt="do-not-log", text="do-not-log", token_id=8, phase="decode")
    assert payload == {
        "schema_version": 1, "event": "request_token", "observed_at": 42.0,
        "request_id": "r", "seq_id": 1, "phase": "decode",
    }
    encoded = json.dumps(payload)
    assert "do-not-log" not in encoded


def test_logger_handler_failure_is_isolated():
    class BrokenLogger:
        def info(self, *args, **kwargs):
            raise RuntimeError("handler down")

    assert LifecycleLogger(BrokenLogger()).emit("request_received",
                                                  request_id="r") is None


def test_metric_update_failure_is_isolated(caplog):
    metrics = Observability()
    metrics.running_requests.set = lambda value: (_ for _ in ()).throw(
        RuntimeError("metric backend down"))
    with caplog.at_level(logging.WARNING):
        metrics.refresh_resource_snapshot(ResourceSnapshot(running=1))
    assert "observability metric update failed" in caplog.text
