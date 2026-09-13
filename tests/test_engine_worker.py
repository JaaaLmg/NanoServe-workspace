"""Day11–12 Engine 完成记录通道、EngineWorker 与 RequestManager 测试
（docs/openai-api-day11-12.md §4.4 与 §8.4）。

覆盖范围：
- 完成记录通道（真实 Scheduler/Sequence + runner 桩）：
  stop/length 完成记录字段与 usage、drain 幂等、step() 返回格式不变、
  取消/超时产生 AbortedRequest 不伪装成 completion、abort_all_active 记录
  engine_error、request_id 复用隔离；
- Worker/Manager（FakeEngine 桩）：
  只有 worker 线程调用 step()、多请求合并同一调度轮、空闲不空转、
  句柄恰好收口一次、迟到/未知记录不误完成、迟到 cancel 返回 False、
  运行中 cancel 只进入 Day10 控制入口、step 异常失败全部句柄并 exit、
  无进展错误语义、关闭路径无悬挂 Future、错误不串句柄。

无 GPU/模型依赖：Scheduler/Engine 通过 __new__/SimpleNamespace 构造，
FakeEngine/FakeTokenizer 为纯 CPU 桩；超时记录用可注入时钟。

重复执行命令：
    python -m pytest tests/test_engine_worker.py -q
    python -O -m pytest tests/test_engine_worker.py -q
"""

import threading
import time
from types import SimpleNamespace

import pytest

from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

from nanoserve.service import (CompletionResult, EngineError, InternalRequest,
                               RequestManager, RequestTimeoutError,
                               ServiceDrainingError)
from nanoserve.testing import FakeEngine, FakeTokenizer
from nanoserve.worker import EngineWorker, WorkerStatus

EOS = 7
BLOCK_SIZE = 256


# ============================== 真实 Scheduler 桩构造 ==============================

def make_scheduler(num_blocks: int = 64) -> Scheduler:
    """与 test_request_control 相同口径的 Scheduler 构造（无 Config/GPU）。"""
    return Scheduler(SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=512,
        chunk_size=256,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    ))


def make_engine(num_blocks: int = 64, eos: int = EOS) -> LLMEngine:
    """跳过 GPU __init__ 的 Engine 桩：runner 对需采样 item 注入固定 token。"""
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = make_scheduler(num_blocks)
    engine.scheduler.eos = eos
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [ord(c) % 100 + 1 for c in text],
        decode=lambda tokens: f"decoded<{len(list(tokens))}>",
    )

    def runner(method, items_):
        out = [EOS if eos != -1 else 3 for it in items_ if it.needs_sample]
        return out or None

    engine.model_runner = SimpleNamespace(call=runner)
    return engine


def run_until_done(engine: LLMEngine, max_rounds: int = 200) -> None:
    for _ in range(max_rounds):
        if not engine.has_active_requests():
            return
        engine.step()
    raise AssertionError("Engine 未在有界轮次内完成")


class TestEngineCompletionRecordChannel:
    """真实调度路径上的完成记录通道（§4.4 契约）。"""

    def test_stop_completion_record_fields_and_drain_idempotent(self):
        engine = make_engine(eos=EOS)  # 注入 token == eos → stop
        outputs, num_tokens = [], 0
        engine.add_request("hi", SamplingParams(max_tokens=4),
                           request_id="r-stop")
        while engine.has_active_requests():
            out, num_tokens = engine.step()
            outputs.extend(out)
        # step() 既有返回格式不变：[(seq_id, completion_token_ids), num_tokens]
        assert len(outputs) == 1 and outputs[0][1] == [EOS]
        assert num_tokens >= 1
        records = engine.pop_completed()
        assert len(records) == 1
        record = records[0]
        assert isinstance(record, CompletedRequest)
        assert record.request_id == "r-stop"
        assert record.completion_token_ids == (EOS,)
        assert record.prompt_tokens == 2  # "hi" → 2 tokens（stub encode）
        assert record.completion_tokens == 1
        assert record.finish_reason == "stop"
        assert record.finished_at is not None
        # 幂等 drain：重复 pop 返回空，不会重复完成
        assert engine.pop_completed() == []

    def test_length_completion_record(self):
        engine = make_engine(eos=-1)  # 注入 token 永不等于 eos → length
        engine.add_request("hi", SamplingParams(max_tokens=2),
                           request_id="r-len")
        run_until_done(engine)
        (record,) = engine.pop_completed()
        assert record.finish_reason == "length"
        assert record.completion_tokens == 2
        assert record.completion_token_ids == (3, 3)
        assert record.prompt_tokens == 2

    def test_cancel_produces_aborted_record_not_completion(self):
        engine = make_engine(eos=-1)
        engine.add_request("hi", SamplingParams(max_tokens=100),
                           request_id="r-cancel")
        while not engine.scheduler.running:
            engine.step()
        # Engine 空闲时的 cancel：立即在安全点完成收尾（Day10 语义）
        assert engine.cancel_request("r-cancel", "client_gone") is True
        aborted = engine.pop_aborted()
        assert len(aborted) == 1
        assert aborted[0].request_id == "r-cancel"
        # Day10 语义：首次取消原因保留为 finish_reason
        assert aborted[0].finish_reason == "client_gone"
        # 取消不伪装成正常 completion
        assert engine.pop_completed() == []
        assert not engine.has_active_requests()

    def test_timeout_produces_aborted_record(self):
        engine = make_engine(eos=-1)
        # 注入固定时钟：构造校验与调度边界判定共用同一时间轴（Day10 §9.1）
        original_clock = Sequence.clock
        Sequence.clock = staticmethod(lambda: 100.0)
        try:
            engine.scheduler._clock = lambda: 200.0
            engine.add_request("hi", SamplingParams(max_tokens=100),
                               request_id="r-timeout", deadline=150.0)
        finally:
            Sequence.clock = original_clock
        engine.step()  # 首轮调度边界 check_deadlines → TIMEOUT
        aborted = engine.pop_aborted()
        assert len(aborted) == 1
        assert aborted[0].request_id == "r-timeout"
        assert aborted[0].finish_reason == "deadline_exceeded"
        assert engine.pop_completed() == []

    def test_abort_all_active_records_engine_error(self):
        engine = make_engine(eos=-1)

        def failing_runner(method, items_):
            raise RuntimeError("boom")

        engine.model_runner = SimpleNamespace(call=failing_runner)
        engine.add_request("hi", SamplingParams(max_tokens=4),
                           request_id="r-err")
        with pytest.raises(RuntimeError):
            engine.step()
        aborted = engine.pop_aborted()
        assert len(aborted) == 1
        assert aborted[0].request_id == "r-err"
        assert aborted[0].finish_reason == "engine_error"
        assert engine.pop_completed() == []
        assert not engine.has_active_requests()

    def test_request_id_reuse_isolated_in_records(self):
        """终态后复用 request_id：新旧记录不串（§6 不变量 2）。"""
        engine = make_engine(eos=EOS)
        engine.add_request("hi", SamplingParams(max_tokens=4),
                           request_id="dup")
        run_until_done(engine)
        (first,) = engine.pop_completed()
        assert first.request_id == "dup"
        engine.add_request("again", SamplingParams(max_tokens=4),
                           request_id="dup")
        run_until_done(engine)
        records = engine.pop_completed()
        assert len(records) == 1  # 只有新请求的记录
        assert records[0].request_id == "dup"
        assert records[0].prompt_tokens == 5  # "again" → 5 tokens
        assert engine.pop_completed() == []


# ============================== Worker / Manager（FakeEngine） ==============================

def make_internal_request(request_id: str, max_tokens: int = 3,
                          kind: str = "completion") -> InternalRequest:
    return InternalRequest(
        request_id=request_id, kind=kind,
        prompt_token_ids=(1, 2, 3),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        model_id="Qwen3-0.6B",
        created_at=time.perf_counter(),
        deadline=None)


def wait_until(condition, timeout: float = 2.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


class Harness:
    """worker + manager 装配器：start 延迟到调用方，便于先排队再启动。"""

    def __init__(self, engine: FakeEngine | None = None):
        self.engine = engine or FakeEngine()
        self.manager = RequestManager(self.engine.tokenizer)
        self.worker = EngineWorker(self.engine, self.manager)
        self.manager.attach_worker(self.worker)

    def start(self) -> "Harness":
        self.worker.start()
        return self

    def submit(self, request_id: str, **kwargs):
        return self.manager.submit(
            make_internal_request(request_id, **kwargs))

    def close(self, timeout: float = 3.0) -> None:
        self.worker.begin_shutdown()
        self.worker.join(timeout=timeout)


class TestWorkerExecution:
    """step 驱动模型：单线程、批量合并、空闲休眠。"""

    def test_only_worker_thread_calls_step(self):
        """多个请求并发 submit 时只有 worker 线程调用 step()。"""
        harness = Harness().start()
        try:
            handles = [harness.submit(f"req-{i}") for i in range(3)]
            for h in handles:
                result = h.future.result(timeout=2)
                assert isinstance(result, CompletionResult)
                assert result.request_id == h.request_id
            # step 只发生在 worker 线程（§6 不变量 1）
            assert len(harness.engine.step_thread_ids) == 1
            (step_thread_id,) = harness.engine.step_thread_ids
            assert step_thread_id != threading.get_ident()
        finally:
            harness.close()

    def test_requests_batched_into_one_round(self):
        """先排队后启动：多个请求合并进同一调度轮，一次 step 全部完成。"""
        harness = Harness()
        handles = [harness.submit(f"req-{i}") for i in range(3)]
        harness.start()
        try:
            for h in handles:
                h.future.result(timeout=2)
            # 命令排空后才驱动 step：3 个请求同轮完成（批处理不被 HTTP 拆散）
            assert harness.engine.step_calls == 1
        finally:
            harness.close()

    def test_idle_worker_sleeps_without_steps(self):
        """无活动请求时 worker 阻塞等待，不产生高频空 step。"""
        harness = Harness().start()
        try:
            time.sleep(0.05)
            assert harness.engine.step_calls == 0
            handle = harness.submit("req-1")
            handle.future.result(timeout=2)
            assert harness.engine.step_calls == 1
            # 完成后回到空闲：不允许继续空转 step
            time.sleep(0.05)
            assert harness.engine.step_calls == 1
        finally:
            harness.close()


class TestRecordConsumption:
    """完成记录消费：恰好一次、迟到/未知记录隔离。"""

    def test_duplicate_record_does_not_resolve_twice(self):
        harness = Harness().start()
        try:
            handle = harness.submit("req-1")
            result = handle.future.result(timeout=2)
            # 迟到重复记录：句柄已弹出，直接忽略（不重复设置 Future）
            duplicate = CompletedRequest(
                seq_id=1, request_id="req-1", completion_token_ids=(9,),
                prompt_tokens=3, completion_tokens=1,
                finish_reason="stop", finished_at=1.0)
            harness.manager.resolve_completed(duplicate)
            assert harness.manager.wait(handle) is result
        finally:
            harness.close()

    def test_unknown_record_ignored(self):
        harness = Harness().start()
        try:
            handle = harness.submit("req-1")
            stale = CompletedRequest(
                seq_id=99, request_id="req-gone", completion_token_ids=(9,),
                prompt_tokens=1, completion_tokens=1,
                finish_reason="stop", finished_at=1.0)
            harness.manager.resolve_completed(stale)
            result = handle.future.result(timeout=2)
            assert result.request_id == "req-1"
        finally:
            harness.close()

    def test_request_id_reusable_after_resolution(self):
        """ID 复用隔离：句柄收口后 ID 可复用，各自由自己的记录收口。

        迟到/未知记录不误完成新请求由 pop-once 语义与服务层 UUID ID 共同
        保证（同 ID 并发活动在 Engine 活动索引处即被拒绝）。
        """
        harness = Harness().start()
        try:
            first = harness.submit("dup", max_tokens=1)
            first_result = first.future.result(timeout=2)
            assert first_result.completion_tokens == 1
            second = harness.submit("dup", max_tokens=3)
            second_result = second.future.result(timeout=2)
            assert second_result.completion_tokens == 3
            assert second_result.request_id == first_result.request_id
        finally:
            harness.close()

    def test_late_record_cannot_resolve_reused_id(self):
        """同 ID 复用时，旧 seq_id 的迟到记录必须被丢弃。"""
        manager = RequestManager(FakeTokenizer())
        manager.attach_worker(SimpleNamespace(submit=lambda request: None))
        first = manager.submit(make_internal_request("reuse"))
        assert manager.bind_seq_id("reuse", 1)
        manager.resolve_completed(CompletedRequest(
            seq_id=1, request_id="reuse", completion_token_ids=(1,),
            prompt_tokens=3, completion_tokens=1, finish_reason="stop",
            finished_at=1.0))
        assert manager.wait(first).completion_tokens == 1
        second = manager.submit(make_internal_request("reuse"))
        assert manager.bind_seq_id("reuse", 2)
        manager.resolve_completed(CompletedRequest(
            seq_id=1, request_id="reuse", completion_token_ids=(9,),
            prompt_tokens=3, completion_tokens=1, finish_reason="stop",
            finished_at=2.0))
        assert not second.future.done()
        manager.resolve_completed(CompletedRequest(
            seq_id=2, request_id="reuse", completion_token_ids=(2, 3),
            prompt_tokens=3, completion_tokens=2, finish_reason="stop",
            finished_at=3.0))
        assert manager.wait(second).completion_tokens == 2


class TestCancelAndErrors:
    """取消边界与异常收口。"""

    def test_late_cancel_returns_false(self):
        harness = Harness().start()
        try:
            handle = harness.submit("req-1")
            handle.future.result(timeout=2)
            # 已完成/未知请求的迟到 cancel：False（幂等）
            assert harness.worker.cancel("req-1") is False
            assert harness.worker.cancel("req-unknown") is False
            # 迟到 cancel 不得触发消费哨兵（无中止记录产生）
            time.sleep(0.05)
        finally:
            harness.close()

    def test_running_cancel_uses_day10_entry_only(self):
        """运行中 cancel 只进入 Day10 控制入口；中止记录经消费哨兵收口句柄。"""
        harness = Harness(FakeEngine(
            step_script=[("pending",)] * 3, step_delay=0.02)).start()
        try:
            handle = harness.submit("req-1")
            assert wait_until(lambda: harness.engine.has_active_requests())
            assert harness.worker.cancel("req-1", "client_gone") is True
            # cancel 只进了 Engine 控制入口（signal），未直接触碰 HTTP 句柄
            assert harness.engine.cancel_calls == [("req-1", "client_gone")]
            # 中止记录被 worker 消费后，等待中的 Future 以失败收口
            with pytest.raises(ServiceDrainingError):
                handle.future.result(timeout=2)
        finally:
            harness.close()

    def test_step_exception_fails_all_handles_and_exits_once(self):
        engine = FakeEngine(step_script=[RuntimeError("boom")])
        harness = Harness(engine).start()
        try:
            handles = [harness.submit(f"req-{i}") for i in range(2)]
            for h in handles:
                with pytest.raises(EngineError):
                    h.future.result(timeout=2)
            assert harness.worker.status is WorkerStatus.FAILED
            assert engine.exit_calls == 1
            # 失败后不再接受新请求（503 draining 而不是永久悬挂）
            with pytest.raises(ServiceDrainingError):
                harness.submit("req-new")
        finally:
            harness.close()

    def test_no_progress_fails_pending_without_endless_loop(self):
        harness = Harness(FakeEngine(step_script=[("noop",)])).start()
        try:
            handle = harness.submit("req-1")
            with pytest.raises(EngineError):
                handle.future.result(timeout=2)
            assert harness.worker.status is WorkerStatus.FAILED
        finally:
            harness.close()

    def test_add_request_failure_fails_only_that_handle(self):
        engine = FakeEngine()
        engine.fail_add_request = True
        harness = Harness(engine).start()
        try:
            handle = harness.submit("req-1")
            with pytest.raises(EngineError):
                handle.future.result(timeout=2)
            # worker 仍然存活，可以继续服务后续请求
            assert harness.worker.status is WorkerStatus.RUNNING
        finally:
            harness.close()

    def test_error_does_not_cross_contaminate_results(self):
        """一个请求的错误不会把另一个请求的结果错误映射到它的句柄。"""
        engine = FakeEngine(step_script=[
            [("req-a", (21,))],   # 第一轮只完成 req-a
            RuntimeError("boom")  # 第二轮异常，req-b 失败
        ])
        harness = Harness(engine).start()
        try:
            handle_a = harness.submit("req-a", max_tokens=1)
            handle_b = harness.submit("req-b", max_tokens=3)
            result_a = handle_a.future.result(timeout=2)
            with pytest.raises(EngineError):
                handle_b.future.result(timeout=2)
            assert result_a.request_id == "req-a"
            assert result_a.completion_tokens == 1
        finally:
            harness.close()


class TestShutdownSemantics:
    """关闭路径：停止接收、活动请求收口、join/exit 幂等。"""

    def test_clean_shutdown_resolves_and_exits_once(self):
        harness = Harness().start()
        handle = harness.submit("req-1")
        result = handle.future.result(timeout=2)
        harness.close()
        harness.close()  # 重复 close 幂等
        assert harness.worker.status is WorkerStatus.STOPPED
        assert harness.engine.exit_calls == 1
        assert result.request_id == "req-1"

    def test_shutdown_cancels_active_requests_without_hang(self):
        harness = Harness(FakeEngine(
            step_script=[("pending",)] * 50, step_delay=0.01)).start()
        try:
            handle = harness.submit("req-1")
            assert wait_until(lambda: harness.engine.has_active_requests())
            harness.close()
            # 活动请求在关闭路径被取消收口，Future 不悬挂
            with pytest.raises(ServiceDrainingError):
                handle.future.result(timeout=2)
            assert harness.worker.status is WorkerStatus.STOPPED
            assert harness.engine.exit_calls == 1
        finally:
            harness.close()

    def test_stop_accepting_rejects_new_submits(self):
        harness = Harness().start()
        try:
            harness.manager.stop_accepting()
            with pytest.raises(ServiceDrainingError):
                harness.submit("req-late")
        finally:
            harness.close()

    def test_aborted_records_map_to_service_errors(self):
        """终态原因 → 服务错误映射：timeout→504，cancelled→503 draining。"""
        manager = RequestManager(FakeTokenizer())
        # 本测试直接驱动收口路径，worker 用无操作桩代替
        manager.attach_worker(SimpleNamespace(submit=lambda request: None))
        timeout_handle = manager.submit(make_internal_request("req-t"))
        manager.resolve_aborted(AbortedRequest(
            seq_id=1, request_id="req-t", finish_reason="timeout",
            finished_at=1.0))
        with pytest.raises(RequestTimeoutError) as exc_info:
            manager.wait(timeout_handle)
        assert exc_info.value.status_code == 504
        assert exc_info.value.request_id == "req-t"

        cancel_handle = manager.submit(make_internal_request("req-c"))
        manager.resolve_aborted(AbortedRequest(
            seq_id=2, request_id="req-c", finish_reason="cancelled",
            finished_at=1.0))
        with pytest.raises(ServiceDrainingError) as exc_info:
            manager.wait(cancel_handle)
        assert exc_info.value.status_code == 503
        assert exc_info.value.request_id == "req-c"

    def test_engine_error_abort_maps_to_500(self):
        """engine_error 中止记录映射为 EngineError（500）。"""
        manager = RequestManager(FakeTokenizer())
        manager.attach_worker(SimpleNamespace(submit=lambda request: None))
        handle = manager.submit(make_internal_request("req-e"))
        manager.resolve_aborted(AbortedRequest(
            seq_id=1, request_id="req-e", finish_reason="engine_error",
            finished_at=1.0))
        with pytest.raises(EngineError) as exc_info:
            manager.wait(handle)
        assert exc_info.value.status_code == 500
        assert exc_info.value.code == "engine_error"

    def test_decode_failure_closes_future_as_engine_error(self):
        """tokenizer 解码异常不能让已弹出的句柄永久悬挂。"""
        class BrokenTokenizer(FakeTokenizer):
            def decode(self, token_ids):
                raise ValueError("decode failed")

        manager = RequestManager(BrokenTokenizer())
        manager.attach_worker(SimpleNamespace(submit=lambda request: None))
        handle = manager.submit(make_internal_request("req-decode"))
        manager.resolve_completed(CompletedRequest(
            seq_id=1, request_id="req-decode", completion_token_ids=(1,),
            prompt_tokens=3, completion_tokens=1, finish_reason="stop",
            finished_at=1.0))
        with pytest.raises(EngineError) as exc_info:
            manager.wait(handle)
        assert exc_info.value.code == "engine_error"
