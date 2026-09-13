"""NanoServe Day14 observability primitives.

本模块只提供观测旁路，不依赖 Service/Worker/Engine 的私有实现：上层可以把
``RequestTimeline`` 作为请求级 DTO，在 admission、token 和终态边界调用记录
方法，再把只读资源快照交给 ``Observability.refresh_resource_snapshot``。

所有耗时使用同一个可注入的单调时钟（默认 ``perf_counter``）。Prometheus
指标注册到实例自己的 ``CollectorRegistry``，因此测试和多 app 进程不会污染
全局 registry。观测失败只会记录 warning，不能影响请求收口。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import Any, Callable, Mapping

try:
    from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                                   generate_latest)
except ImportError as exc:  # pragma: no cover - exercised only without optional env
    raise ImportError(
        "NanoServe observability requires 'prometheus-client'. Install the runtime "
        "dependencies with `pip install -e .` or `pip install prometheus-client`."
    ) from exc

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_HISTOGRAM_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
                             0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

METRIC_NAMES = (
    "request_queue_time_seconds",
    "time_to_first_token_seconds",
    "time_per_output_token_seconds",
    "request_latency_seconds",
    "prompt_tokens_total",
    "generation_tokens_total",
    "kv_cache_utilization",
    "prefix_cache_hit_rate",
    "running_requests",
)

_ALLOWED_KINDS = frozenset(("completion", "chat"))
_ALLOWED_STATUSES = frozenset((
    "completed", "cancelled", "timeout", "engine_error", "server_shutdown",
    "aborted", "rejected",
))
_ALLOWED_PHASES = frozenset(("prefill", "decode", "mixed"))


def _safe_nonnegative(value: Any) -> float | None:
    """把外部时间/计数转换成非负有限数；坏观测不应打断业务。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return max(0.0, number)


def _safe_count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _duration(end: float | None, start: float | None) -> float | None:
    if end is None or start is None:
        return None
    try:
        difference = end - start
    except (TypeError, ValueError, OverflowError):
        return None
    return _safe_nonnegative(difference)


def validate_labels(*, kind: str | None = None, status: str | None = None,
                    phase: str | None = None) -> None:
    """校验 Prometheus 的低基数标签；request/seq/prompt/token 不在此 API 中。"""
    if kind is not None and kind not in _ALLOWED_KINDS:
        raise ValueError(f"unsupported observability kind: {kind!r}")
    if status is not None and status not in _ALLOWED_STATUSES:
        raise ValueError(f"unsupported observability status: {status!r}")
    if phase is not None and phase not in _ALLOWED_PHASES:
        raise ValueError(f"unsupported observability phase: {phase!r}")


@dataclass(slots=True)
class RequestTimeline:
    """一个请求的单调时间线和 token 计数 DTO。

    ``request_id``/``seq_id`` 只用于日志关联，绝不会成为 Prometheus label。
    ``token_timestamps`` 只保存 token 可被上层消费的时刻，不保存 token 内容。
    ``mark_*`` 方法是幂等的，便于 worker 在重复 drain/race 场景安全调用。
    """

    request_id: str | None = None
    kind: str = "completion"
    submitted_at: float | None = None
    admitted_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    seq_id: int | None = None
    token_timestamps: list[float] = field(default_factory=list)
    _clock: Callable[[], float] = field(default=perf_counter, repr=False,
                                        compare=False)
    _queue_recorded: bool = field(default=False, init=False, repr=False,
                                    compare=False)
    _ttft_recorded: bool = field(default=False, init=False, repr=False,
                                  compare=False)
    _itl_recorded: bool = field(default=False, init=False, repr=False,
                                 compare=False)
    _latency_recorded: bool = field(default=False, init=False, repr=False,
                                     compare=False)
    _prompt_recorded: bool = field(default=False, init=False, repr=False,
                                    compare=False)
    _tokens_recorded: int = field(default=0, init=False, repr=False,
                                   compare=False)

    def __post_init__(self) -> None:
        validate_labels(kind=self.kind)
        self.prompt_tokens = _safe_count(self.prompt_tokens)
        self.generation_tokens = _safe_count(self.generation_tokens)

    @classmethod
    def start(cls, request_id: str | None = None, *, kind: str = "completion",
              prompt_tokens: int = 0, seq_id: int | None = None,
              clock: Callable[[], float] = perf_counter) -> "RequestTimeline":
        """在请求提交边界创建 DTO，避免把 HTTP Unix 时间当作耗时基线。"""
        return cls(request_id=request_id, kind=kind, seq_id=seq_id,
                   prompt_tokens=prompt_tokens, submitted_at=clock(),
                   _clock=clock)

    def _now(self, at: float | None) -> float:
        return self._clock() if at is None else at

    def mark_submitted(self, at: float | None = None) -> float:
        if self.submitted_at is None:
            self.submitted_at = self._now(at)
        return self.submitted_at

    def mark_admitted(self, at: float | None = None) -> float:
        if self.admitted_at is None:
            self.admitted_at = self._now(at)
        return self.admitted_at

    def mark_first_token(self, at: float | None = None) -> float:
        """记录首个真实 token 的可消费时刻，并计入一个 completion token。

        若调用方希望同时记录首 token 和后续 token，后续只调用 ``mark_token``；
        首 token 不会因重复 drain 被重复追加。
        """
        timestamp = self._now(at)
        if self.first_token_at is None:
            self.first_token_at = timestamp
            self.last_token_at = timestamp
            self.token_timestamps.append(timestamp)
            self.generation_tokens = max(1, self.generation_tokens)
        return self.first_token_at

    def mark_token(self, at: float | None = None) -> float:
        """记录一个真实可消费 token 的时刻，不记录 token ID 或文本。"""
        timestamp = self._now(at)
        if self.first_token_at is None:
            return self.mark_first_token(timestamp)
        self.token_timestamps.append(timestamp)
        self.last_token_at = timestamp
        self.generation_tokens += 1
        return timestamp

    def mark_finished(self, at: float | None = None) -> float:
        if self.finished_at is None:
            self.finished_at = self._now(at)
        return self.finished_at

    # 语义化别名使 worker 接入时不需要了解 DTO 内部字段。
    record_admission = mark_admitted
    record_first_token = mark_first_token
    record_token = mark_token
    record_finished = mark_finished

    @property
    def queue_wait_seconds(self) -> float | None:
        return _duration(self.admitted_at, self.submitted_at)

    @property
    def time_to_first_token_seconds(self) -> float | None:
        return _duration(self.first_token_at, self.submitted_at)

    @property
    def admission_time_to_first_token_seconds(self) -> float | None:
        return _duration(self.first_token_at, self.admitted_at)

    @property
    def latency_seconds(self) -> float | None:
        return _duration(self.finished_at, self.submitted_at)

    @property
    def inter_token_latencies(self) -> tuple[float, ...]:
        """相邻 token 的 ITL；少于两个真实 token 时为空。"""
        if len(self.token_timestamps) < 2:
            return ()
        values: list[float] = []
        for previous, current in zip(self.token_timestamps,
                                     self.token_timestamps[1:]):
            duration = _duration(current, previous)
            if duration is not None:
                values.append(duration)
        return tuple(values)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Engine/Scheduler 在锁内复制后交给观测层的只读资源快照。"""

    running: int = 0
    waiting: int = 0
    paused: int = 0
    active: int = 0
    used_blocks: int = 0
    total_blocks: int = 0
    prefix_cache_lookups: int | None = None
    prefix_cache_hits: int | None = None
    prefix_cache_misses: int | None = None
    prefix_cache_capacity_failures: int | None = None

    @property
    def kv_cache_utilization(self) -> float:
        total = _safe_count(self.total_blocks)
        return min(1.0, _safe_count(self.used_blocks) / total) if total else 0.0

    @property
    def prefix_cache_hit_rate(self) -> float | None:
        if self.prefix_cache_lookups is None:
            return None
        lookups = _safe_count(self.prefix_cache_lookups)
        return (_safe_count(self.prefix_cache_hits) / lookups
                if lookups else 0.0)


class LifecycleLogger:
    """输出版本化 JSON lifecycle envelope 的轻量适配器。"""

    # 采用白名单而非黑名单，防止未来新增字段时意外泄漏用户输入。
    _SAFE_FIELDS = frozenset({
        "request_id", "seq_id", "round_id", "kind", "model_id", "queue_origin",
        "admitted_at", "completion_index", "phase", "status", "finish_reason",
        "prompt_tokens", "completion_tokens", "generation_tokens",
        "queue_wait_seconds", "ttft_seconds", "latency_seconds", "reason",
        "stage", "error_code", "error_type", "observed_at",
    })

    def __init__(self, log: logging.Logger | None = None, *, schema_version: int = 1):
        self.logger = log or logger
        self.schema_version = int(schema_version)

    def envelope(self, event: str, *, observed_at: float | None = None,
                 **fields: Any) -> dict[str, Any]:
        if not isinstance(event, str) or not event:
            raise ValueError("lifecycle event must be a non-empty string")
        envelope: dict[str, Any] = {
            "schema_version": self.schema_version,
            "event": event,
            "observed_at": perf_counter() if observed_at is None else observed_at,
        }
        for key, value in fields.items():
            if key not in self._SAFE_FIELDS or key in {"prompt", "token", "text"}:
                continue
            # 只接受标量；嵌套 payload 容易绕过字段白名单带入 prompt/messages。
            if value is None or isinstance(value, (str, int, float, bool)):
                envelope[key] = value
        return envelope

    def emit(self, event: str, *, observed_at: float | None = None,
             **fields: Any) -> dict[str, Any] | None:
        """发出一条 JSON 日志；序列化/handler 失败均隔离。"""
        try:
            payload = self.envelope(event, observed_at=observed_at, **fields)
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.logger.info("%s", encoded)
            return payload
        except Exception as exc:  # logging handler 不能阻断 Engine 收口
            try:
                logger.warning("observability lifecycle log failed: %s: %s",
                               type(exc).__name__, str(exc)[:160])
            except Exception:
                pass
            return None

    log = emit


def lifecycle_envelope(event: str, *, observed_at: float | None = None,
                       schema_version: int = SCHEMA_VERSION,
                       **fields: Any) -> dict[str, Any]:
    """函数式 envelope API，方便 service/worker 无需持有 logger 对象。"""
    return LifecycleLogger(schema_version=schema_version).envelope(
        event, observed_at=observed_at, **fields)


class Observability:
    """独立 registry、请求时间线记录和资源 gauge 的统一入口。"""

    def __init__(self, *, registry: CollectorRegistry | None = None,
                 log: logging.Logger | None = None,
                 clock: Callable[[], float] = perf_counter,
                 histogram_buckets: tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS):
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.clock = clock
        self.lifecycle = LifecycleLogger(log, schema_version=SCHEMA_VERSION)
        buckets = tuple(histogram_buckets)
        # label 只选择稳定业务维度；严禁 request_id/seq_id/prompt/token。
        self.request_queue_time_seconds = Histogram(
            "request_queue_time_seconds", "Request queue wait time in seconds.",
            ["kind"], registry=self.registry, buckets=buckets)
        self.time_to_first_token_seconds = Histogram(
            "time_to_first_token_seconds", "Service time to first token in seconds.",
            ["kind"], registry=self.registry, buckets=buckets)
        self.time_per_output_token_seconds = Histogram(
            "time_per_output_token_seconds", "Inter-token output time in seconds.",
            ["kind"], registry=self.registry, buckets=buckets)
        self.request_latency_seconds = Histogram(
            "request_latency_seconds", "End-to-end request latency in seconds.",
            ["kind", "status"], registry=self.registry, buckets=buckets)
        self.prompt_tokens_total = Counter(
            "prompt_tokens_total", "Prompt tokens admitted to the engine.",
            ["kind"], registry=self.registry)
        self.generation_tokens_total = Counter(
            "generation_tokens_total", "Completion tokens actually generated.",
            ["kind"], registry=self.registry)
        self.kv_cache_utilization = Gauge(
            "kv_cache_utilization", "Physical KV cache utilization in [0, 1].",
            registry=self.registry)
        self.prefix_cache_hit_rate = Gauge(
            "prefix_cache_hit_rate", "Request-level prefix cache hit rate.",
            registry=self.registry)
        self.running_requests = Gauge(
            "running_requests", "Requests currently in RUNNING state.",
            registry=self.registry)
        # 推荐的累计计数，供后续 service/worker 接入，不增加高基数标签。
        self.prefix_cache_lookups_total = Counter(
            "prefix_cache_lookups_total", "Prefix cache lookup requests.",
            registry=self.registry)
        self.prefix_cache_hits_total = Counter(
            "prefix_cache_hits_total", "Prefix cache hit requests.",
            registry=self.registry)
        self.prefix_cache_misses_total = Counter(
            "prefix_cache_misses_total", "Prefix cache miss requests.",
            registry=self.registry)
        self.prefix_cache_capacity_failures_total = Counter(
            "prefix_cache_capacity_failures_total", "Prefix lookup KV capacity failures.",
            registry=self.registry)
        self._prefix_lock = threading.Lock()
        self._prefix_lookups = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._prefix_capacity_failures = 0
        self._prefix_exported = (0, 0, 0, 0)

    def start_request(self, request_id: str | None = None, *, kind: str = "completion",
                      prompt_tokens: int = 0, seq_id: int | None = None) -> RequestTimeline:
        return RequestTimeline.start(request_id, kind=kind,
                                     prompt_tokens=prompt_tokens, seq_id=seq_id,
                                     clock=self.clock)

    def _observe(self, action: Callable[[], Any], description: str) -> None:
        try:
            action()
        except Exception as exc:
            try:
                logger.warning("observability metric update failed (%s): %s: %s",
                               description, type(exc).__name__, str(exc)[:160])
            except Exception:
                pass

    def record_timeline(self, timeline: RequestTimeline, *, status: str = "completed",
                        log_event: bool = False,
                        finish_reason: str | None = None) -> RequestTimeline:
        """记录时间线的可用部分；缺失首 token 不会伪造 TTFT/TPOT。"""
        validate_labels(kind=timeline.kind, status=status)
        kind = timeline.kind
        if timeline.admitted_at is not None and not timeline._prompt_recorded:
            self._observe(lambda: self.prompt_tokens_total.labels(kind=kind).inc(
                _safe_count(timeline.prompt_tokens)), "prompt tokens")
            timeline._prompt_recorded = True
        queue_wait = timeline.queue_wait_seconds
        if queue_wait is not None and not timeline._queue_recorded:
            self._observe(lambda: self.request_queue_time_seconds.labels(kind=kind)
                          .observe(queue_wait), "queue time")
            timeline._queue_recorded = True
        ttft = timeline.time_to_first_token_seconds
        if ttft is not None and not timeline._ttft_recorded:
            self._observe(lambda: self.time_to_first_token_seconds.labels(kind=kind)
                          .observe(ttft), "TTFT")
            timeline._ttft_recorded = True
        latency = timeline.latency_seconds
        if latency is not None and not timeline._latency_recorded:
            self._observe(lambda: self.request_latency_seconds.labels(
                kind=kind, status=status).observe(latency), "latency")
            timeline._latency_recorded = True
        if timeline.token_timestamps and not timeline._itl_recorded:
            self._observe(lambda: [self.time_per_output_token_seconds.labels(kind=kind)
                                   .observe(value)
                                   for value in timeline.inter_token_latencies], "ITL")
            timeline._itl_recorded = True
        new_tokens = max(0, _safe_count(timeline.generation_tokens)
                         - timeline._tokens_recorded)
        if new_tokens:
            self._observe(lambda: self.generation_tokens_total.labels(kind=kind).inc(
                new_tokens), "generation tokens")
            timeline._tokens_recorded += new_tokens
        if log_event:
            event_name = "request_finished" if status == "completed" else "request_aborted"
            self.lifecycle.emit(event_name, observed_at=timeline.finished_at,
                                request_id=timeline.request_id, seq_id=timeline.seq_id,
                                kind=kind, status=status,
                                finish_reason=finish_reason or status,
                                prompt_tokens=timeline.prompt_tokens,
                                completion_tokens=timeline.generation_tokens,
                                queue_wait_seconds=timeline.queue_wait_seconds,
                                ttft_seconds=timeline.time_to_first_token_seconds,
                                latency_seconds=timeline.latency_seconds)
        return timeline

    record_request = record_timeline

    def record_prefix_lookup(self, hit: bool = False, *, capacity_failure: bool = False) -> None:
        """记录首次请求级 lookup；容量失败单独分类且不计入 miss。"""
        with self._prefix_lock:
            self._prefix_lookups += 1
            lookups = self._prefix_lookups
            if capacity_failure:
                self._prefix_capacity_failures += 1
            elif hit:
                self._prefix_hits += 1
            else:
                self._prefix_misses += 1
            hits = self._prefix_hits
        self._observe(lambda: self.prefix_cache_lookups_total.inc(), "prefix lookup")
        if capacity_failure:
            self._observe(lambda: self.prefix_cache_capacity_failures_total.inc(),
                          "prefix capacity failure")
        elif hit:
            self._observe(lambda: self.prefix_cache_hits_total.inc(), "prefix hit")
        else:
            self._observe(lambda: self.prefix_cache_misses_total.inc(), "prefix miss")
        self._observe(lambda: self.prefix_cache_hit_rate.set(
            hits / lookups if lookups else 0.0), "prefix hit rate")

    def refresh_resource_snapshot(self, snapshot: ResourceSnapshot | Mapping[str, Any] | Any,
                                  **kwargs: Any) -> ResourceSnapshot:
        """刷新 running/KV/prefix gauge；输入可为 DTO、mapping 或简单对象。"""
        if kwargs:
            if isinstance(snapshot, Mapping):
                values = dict(snapshot)
                values.update(kwargs)
                snapshot = ResourceSnapshot(**values)
            else:
                values = {item.name: getattr(snapshot, item.name, item.default)
                          for item in fields(ResourceSnapshot)
                          if not item.name.startswith("_")}
                values.update(kwargs)
                snapshot = ResourceSnapshot(**values)
        elif not isinstance(snapshot, ResourceSnapshot):
            field_names = {
                item.name for item in fields(ResourceSnapshot)
                if not item.name.startswith("_")
            }
            if isinstance(snapshot, Mapping):
                snapshot = ResourceSnapshot(**{
                    name: snapshot[name] for name in field_names if name in snapshot
                })
            else:
                snapshot = ResourceSnapshot(**{
                    item.name: getattr(snapshot, item.name, item.default)
                    for item in fields(ResourceSnapshot)
                    if not item.name.startswith("_")
                })
        self._observe(lambda: self.running_requests.set(_safe_count(snapshot.running)),
                      "running requests")
        self._observe(lambda: self.kv_cache_utilization.set(
            snapshot.kv_cache_utilization), "KV utilization")
        hit_rate = snapshot.prefix_cache_hit_rate
        if hit_rate is not None:
            self._observe(lambda: self.prefix_cache_hit_rate.set(
                min(1.0, max(0.0, hit_rate))), "prefix snapshot hit rate")
        counts = (snapshot.prefix_cache_lookups or 0,
                  snapshot.prefix_cache_hits or 0,
                  snapshot.prefix_cache_misses or 0,
                  snapshot.prefix_cache_capacity_failures or 0)
        previous = self._prefix_exported
        deltas = tuple(max(0, current - old)
                       for current, old in zip(counts, previous))
        counters = (self.prefix_cache_lookups_total,
                    self.prefix_cache_hits_total,
                    self.prefix_cache_misses_total,
                    self.prefix_cache_capacity_failures_total)
        for counter, delta in zip(counters, deltas):
            if delta:
                self._observe(lambda c=counter, d=delta: c.inc(d),
                              "prefix snapshot counter")
        self._prefix_exported = counts
        return snapshot

    refresh_snapshot = refresh_resource_snapshot

    def emit(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.lifecycle.emit(event, **fields)

    def render_metrics(self) -> bytes:
        """生成独立 registry 的 Prometheus exposition 文本。"""
        try:
            return generate_latest(self.registry)
        except Exception as exc:
            try:
                logger.warning("observability metrics rendering failed: %s: %s",
                               type(exc).__name__, str(exc)[:160])
            except Exception:
                pass
            return b""

    metrics = render_metrics


# 对接代码可选择更语义化的别名；它们仍共享同一实现和 registry 约定。
Metrics = Observability
RequestMetrics = Observability

__all__ = [
    "DEFAULT_HISTOGRAM_BUCKETS", "LifecycleLogger", "METRIC_NAMES", "Metrics",
    "Observability", "RequestMetrics", "RequestTimeline", "ResourceSnapshot",
    "SCHEMA_VERSION", "lifecycle_envelope", "validate_labels",
]
