"""Day 6 验收测试：请求生命周期与状态机（docs/request-lifecycle.md 第 8 节）。

覆盖范围：
- 8.1 状态机单元测试：六状态、合法/非法迁移、幂等终态、取消标记语义；
- 8.2 Scheduler 集成测试：队列一致性、KV 释放、抢占/恢复、deadline、重复操作；
- 张量并行序列化协议：v2 字段化状态 + v1 旧元组兼容；
- LLMEngine 控制面：add_request 返回可追踪 ID、get_request / cancel_request 代理。

全部为纯 CPU 测试，不加载模型：Scheduler 只读取 Config 的少数字段，
用 SimpleNamespace 构造；LLMEngine 通过 __new__ 跳过需要 GPU 的 __init__，
仅验证与调度器相关的控制面逻辑。

Day9 适配说明（docs/mixed-prefill-decode.md）：schedule() 返回 (items, phase)、
postprocess() 逐 item 提交；共享前缀等"已 RUNNING + 新 waiting"场景的轮次
变为混合轮（decode 在前、prefill 在后），断言按 item 序列改写。
状态机、控制面与序列化协议的断言原样保持。

重复执行命令：
    python -m pytest tests/test_request_lifecycle.py -q
"""

import pickle
from time import perf_counter
from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    InvalidStateTransition,
    Sequence,
    SequenceStatus,
)
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 8
EOS = 999_999  # 测试中不会出现的 token id，配合 ignore_eos 使用

# 真实运行时由 LLMEngine 设置（nanovllm/engine/llm_engine.py），测试手动对齐
Sequence.block_size = BLOCK_SIZE

# Day10：Sequence 构造校验 deadline >= created_at（单调时钟，§4.1/§6.2）。
# 本文件在虚拟时间轴上注入小数值 now（如 10.0/12.0），因此把 Sequence.clock
# 注入为固定 0.0，使 created_at 与虚拟时间轴同轴、构造校验可确定性通过；
# fixture 结束时恢复真实时钟，避免跨测试泄漏。
@pytest.fixture(autouse=True)
def _fixed_sequence_clock():
    original = Sequence.clock
    Sequence.clock = staticmethod(lambda: 1.0)
    yield
    Sequence.clock = original


WAITING = SequenceStatus.WAITING
RUNNING = SequenceStatus.RUNNING
PREEMPTED = SequenceStatus.PREEMPTED
FINISHED = SequenceStatus.FINISHED
CANCELLED = SequenceStatus.CANCELLED
TIMEOUT = SequenceStatus.TIMEOUT


def make_seq(num_tokens: int = 4, max_tokens: int = 8, deadline: float | None = None,
             request_id: str | None = None) -> Sequence:
    return Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        request_id=request_id,
        deadline=deadline,
    )


def make_scheduler(num_blocks: int = 8, max_num_batched_tokens: int = 10**6) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=512,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def schedule_round(sched: Scheduler, *args, **kwargs):
    """Day9 适配：schedule() 返回 (items, phase)；展开为 (seqs, items, is_prefill)。

    items 是 Day9 postprocess 的入参（携带 needs_sample 快照与 round_id）；
    is_prefill 仅服务旧断言（phase == "prefill"，decode/mixed 轮为 False）。
    """
    items, phase = sched.schedule(*args, **kwargs)
    return [it.seq for it in items], items, phase == "prefill"


def tokens_for_items(items, token: int) -> list[int]:
    """按 Day9 提交契约构造 postprocess 的 token 注入列表（按 needs_sample 快照对齐）。"""
    return [token for it in items if it.needs_sample]


def run_round(sched: Scheduler, sampled_token_ids: list[int]):
    """执行一轮 schedule + postprocess，模拟一次引擎 step（不含模型）。"""
    seqs, items, is_prefill = schedule_round(sched)
    sched.postprocess(items, sampled_token_ids)
    return seqs, is_prefill


# ============================== 8.1 状态机单元测试 ==============================

class TestStateMachineDefinition:
    def test_six_states_defined(self):
        """SequenceStatus 必须包含六种计划要求的状态。"""
        assert {s.name for s in SequenceStatus} == {
            "WAITING", "RUNNING", "PREEMPTED", "FINISHED", "CANCELLED", "TIMEOUT",
        }
        assert TERMINAL_STATUSES == {FINISHED, CANCELLED, TIMEOUT}

    def test_transition_table_matches_state_diagram(self):
        """迁移表与文档 3.2 状态迁移图一致：每个状态都有定义，终态迁出为空。"""
        assert set(VALID_TRANSITIONS) == set(SequenceStatus)
        assert VALID_TRANSITIONS[WAITING] == {RUNNING, CANCELLED, TIMEOUT}
        assert VALID_TRANSITIONS[RUNNING] == {PREEMPTED, FINISHED, CANCELLED, TIMEOUT}
        assert VALID_TRANSITIONS[PREEMPTED] == {WAITING, CANCELLED, TIMEOUT}
        for terminal in (FINISHED, CANCELLED, TIMEOUT):
            assert VALID_TRANSITIONS[terminal] == set()


class TestStateTransitions:
    def test_initial_state_and_fields(self):
        """初始状态为 WAITING，时间戳与控制字段初始化正确。"""
        seq = make_seq()
        assert seq.status == WAITING
        assert seq.request_id == f"req-{seq.seq_id}"  # 未指定时由 seq_id 派生
        assert seq.created_at > 0
        assert seq.started_at is None
        assert seq.finished_at is None
        assert seq.deadline is None
        assert seq.cancel_requested is False
        assert seq.cancel_reason is None
        assert seq.finish_reason is None
        assert seq.is_active and not seq.is_finished and not seq.is_terminal
        assert seq.num_preempts == 0

    def test_waiting_to_running_sets_started_at_once(self):
        """WAITING -> RUNNING 合法，started_at 只设置一次。"""
        seq = make_seq()
        seq.transition_to(RUNNING)
        started = seq.started_at
        assert started is not None and started >= seq.created_at
        # 抢占恢复后再次进入 RUNNING，started_at 不重置
        seq.transition_to(PREEMPTED)
        seq.transition_to(WAITING)
        seq.transition_to(RUNNING)
        assert seq.started_at == started

    @pytest.mark.parametrize("target,reason", [
        (FINISHED, "stop"),
        (CANCELLED, "client_cancelled"),
        (TIMEOUT, "deadline_exceeded"),
    ])
    def test_running_to_terminal_records_reason(self, target, reason):
        """RUNNING -> 三个终态均合法，并记录 finish_reason 与 finished_at。"""
        seq = make_seq()
        seq.transition_to(RUNNING)
        seq.transition_to(target, reason=reason)
        assert seq.status == target
        assert seq.finish_reason == reason
        assert seq.finished_at is not None
        assert seq.is_terminal and seq.is_finished and not seq.is_active

    def test_running_to_preempted_to_waiting_preserves_progress(self):
        """RUNNING -> PREEMPTED -> WAITING 合法，生成 token 和进度保持不变。"""
        seq = make_seq(num_tokens=6, max_tokens=8)
        seq.append_token(7000)
        seq.transition_to(RUNNING)
        tokens_before = list(seq.token_ids)
        completion_before = seq.num_completion_tokens

        seq.transition_to(PREEMPTED)
        assert seq.num_preempts == 1
        assert seq.last_preempted_at is not None
        seq.transition_to(WAITING)

        assert seq.token_ids == tokens_before
        assert seq.num_completion_tokens == completion_before
        assert seq.num_prompt_tokens == 6  # prompt 进度不回滚

    @pytest.mark.parametrize("from_status,to_status", [
        (WAITING, CANCELLED), (WAITING, TIMEOUT),
        (PREEMPTED, CANCELLED), (PREEMPTED, TIMEOUT),
    ])
    def test_waiting_or_preempted_to_cancel_timeout(self, from_status, to_status):
        """WAITING/PREEMPTED -> CANCELLED/TIMEOUT 均合法。"""
        seq = make_seq()
        if from_status == PREEMPTED:
            seq.transition_to(RUNNING)
            seq.transition_to(PREEMPTED)
        seq.transition_to(to_status)
        assert seq.status == to_status

    def test_request_cancel_flag_semantics(self):
        """取消标记可被读取；原因不被后续重复取消覆盖；未终态时返回 True。"""
        seq = make_seq()
        assert seq.request_cancel("client_gone") is True
        assert seq.cancel_requested is True
        assert seq.cancel_reason == "client_gone"
        # 重复取消：仍返回 True（请求确实待取消），但原因保持首次记录
        assert seq.request_cancel("another_reason") is True
        assert seq.cancel_reason == "client_gone"

    @pytest.mark.parametrize("terminal", [FINISHED, CANCELLED, TIMEOUT])
    def test_terminal_states_are_idempotent(self, terminal):
        """终态重复完成、重复取消、重复超时均幂等返回 False，状态不再变化。"""
        seq = make_seq()
        seq.transition_to(RUNNING)
        if terminal == FINISHED:
            seq.mark_finished("stop")
        elif terminal == CANCELLED:
            seq.mark_cancelled()
        else:
            seq.mark_timeout()
        assert seq.is_terminal

        assert seq.mark_finished("stop") is False
        assert seq.mark_cancelled("again") is False
        assert seq.mark_timeout("again") is False
        assert seq.request_cancel("again") is False  # 终态后取消请求被拒绝
        assert seq.status == terminal
        assert seq.finished_at is not None  # 时间戳不被重复调用覆盖

    @pytest.mark.parametrize("old,new", [
        (FINISHED, RUNNING),
        (CANCELLED, WAITING),
        (TIMEOUT, PREEMPTED),
        (WAITING, FINISHED),   # 未经过执行不能直接"完成"
        (RUNNING, WAITING),    # 运行中不能直接退回等待，必须先经 PREEMPTED
        (CANCELLED, RUNNING),
    ])
    def test_invalid_transitions_raise(self, old, new):
        """非法迁移抛 InvalidStateTransition，且状态保持不变（原子性）。"""
        seq = make_seq()
        seq.status = old  # 测试直接构造起始状态
        with pytest.raises(InvalidStateTransition) as excinfo:
            seq.transition_to(new)
        # 错误信息可定位请求
        assert seq.request_id in str(excinfo.value)
        assert old.name in str(excinfo.value) and new.name in str(excinfo.value)
        assert seq.status == old  # 无副作用

    def test_invalid_transition_error_context(self):
        """异常对象保留 request_id、原状态、目标状态和原因。"""
        seq = make_seq(request_id="req-42")
        seq.transition_to(RUNNING)
        seq.mark_finished("stop")
        with pytest.raises(InvalidStateTransition) as excinfo:
            seq.transition_to(RUNNING, reason="oops")
        assert excinfo.value.request_id == "req-42"
        assert excinfo.value.old_status == FINISHED
        assert excinfo.value.new_status == RUNNING
        assert excinfo.value.reason == "oops"


# ============================== 8.2 Scheduler 集成测试 ==============================

class TestSchedulerBasics:
    def test_empty_queues_schedule_nothing(self):
        """空队列不会调度，返回空批次而不是崩溃。"""
        sched = make_scheduler()
        seqs, items, is_prefill = schedule_round(sched)
        assert seqs == [] and is_prefill is False
        assert sched.is_finished()

    def test_duplicate_add_rejected(self):
        """同一 seq_id 不会创建两个活动对象。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        with pytest.raises(ValueError):
            sched.add(seq)

    def test_single_request_prefill_enters_running(self):
        """单请求 prefill 后进入 RUNNING，并出现在 running 队列。"""
        sched = make_scheduler()
        seq = make_seq(4, max_tokens=4)
        sched.add(seq)
        seqs, items, is_prefill = schedule_round(sched)
        assert is_prefill is True and seqs == [seq]
        assert seq.status == RUNNING
        assert list(sched.running) == [seq]
        assert list(sched.waiting) == []
        assert sched.requests[seq.seq_id] is seq

    def test_finish_by_max_tokens_releases_blocks(self):
        """达到 max_tokens 后：从 running 移除、block 全部释放、活动索引清空。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(4, max_tokens=2)
        sched.add(seq)

        run_round(sched, [100])   # prefill，产出首个 completion token
        run_round(sched, [101])   # decode，第 2 个 token 达到 max_tokens

        assert seq.status == FINISHED
        assert seq.finish_reason == "length"
        assert seq.block_table == []
        assert seq.is_terminal
        assert list(sched.running) == [] and list(sched.waiting) == []
        assert sched.requests == {}
        assert sched.is_finished()
        assert len(bm.free_block_ids) == 8

    def test_finish_by_eos_records_stop(self):
        """EOS 触发完成时 finish_reason 为 stop（与 length 区分）。"""
        sched = make_scheduler(num_blocks=8)
        eos = 7777
        sched.eos = eos
        seq = Sequence(list(range(1, 5)), SamplingParams(max_tokens=8, ignore_eos=False))
        sched.add(seq)

        run_round(sched, [200])          # prefill
        run_round(sched, [eos])          # decode 采样到 EOS

        assert seq.status == FINISHED
        assert seq.finish_reason == "stop"
        assert seq.block_table == []
        assert sched.is_finished()


class TestCancellation:
    def test_cancel_waiting_request_never_runs(self):
        """waiting 中取消请求：不会进入模型执行，资源从未分配。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(4, max_tokens=4)
        sched.add(seq)

        assert sched.cancel(seq.seq_id, reason="client_gone") is True
        assert seq.status == CANCELLED
        assert seq.finish_reason == "client_gone"
        assert seq.block_table == []  # 从未分配
        assert list(sched.waiting) == [] and list(sched.running) == []
        assert sched.is_finished()

        # 取消后的请求不参与调度
        seqs, items, is_prefill = schedule_round(sched)
        assert seqs == []
        assert len(bm.free_block_ids) == 8

    def test_cancel_running_request_between_steps(self):
        """running 中取消请求：下一步调度不再选中它，block 已释放。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [100])   # prefill -> RUNNING
        assert len(seq.block_table) == 1

        assert sched.cancel(seq.seq_id) is True
        assert seq.status == CANCELLED and seq.block_table == []
        seqs, items, _ = schedule_round(sched)
        assert seqs == []
        assert sched.is_finished()

    def test_cancel_flag_during_model_execution_drops_token(self):
        """模拟模型执行期间收到取消：postprocess 不追加 token，直接清理。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [100])                      # prefill
        seqs, items, is_prefill = schedule_round(sched)          # decode 批次已选出
        assert is_prefill is False

        # "模型执行期间"客户端置位取消标记（只打标记，不迁移）
        assert seq.request_cancel("client_disconnected") is True

        sched.postprocess(items, [5555])  # 本轮采样的 token 必须被丢弃
        assert 5555 not in seq.token_ids             # 取消之后的 token 未追加
        assert seq.status == CANCELLED
        assert seq.finish_reason == "client_disconnected"
        assert seq.block_table == []
        assert list(sched.running) == []
        assert sched.is_finished()

    def test_repeated_cancel_is_idempotent(self):
        """多次调用 cancel 不造成队列重复或 free block 数量异常。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [100])   # prefill，占用 1 块
        assert len(bm.free_block_ids) == 7

        assert sched.cancel(seq.seq_id) is True
        free_after_first = len(bm.free_block_ids)
        assert free_after_first == 8

        # 重复取消 / 不存在的 id：返回 False，不改变任何状态
        assert sched.cancel(seq.seq_id) is False
        assert sched.cancel(12345) is False
        assert seq.status == CANCELLED
        assert len(bm.free_block_ids) == free_after_first
        assert sched.cancel(seq.seq_id, reason="other") is False
        assert seq.finish_reason != "other"  # 原因不被覆盖


class TestTimeout:
    def test_expired_deadline_in_waiting_is_purged_by_schedule(self):
        """超时请求即使尚未运行也会在调度边界从 waiting 清除。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(4, max_tokens=4, deadline=perf_counter() - 1)  # 已过期
        sched.add(seq)

        seqs, items, _ = schedule_round(sched)  # schedule 内部执行 deadline 检查
        assert seqs == []
        assert seq.status == TIMEOUT
        assert seq.finish_reason == "deadline_exceeded"
        assert seq.block_table == []
        assert sched.is_finished()
        assert len(bm.free_block_ids) == 8

    def test_running_deadline_enforced_at_boundary(self):
        """运行中请求超过 deadline：在安全边界停止并释放资源。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8, deadline=perf_counter() + 3600)
        sched.add(seq)
        run_round(sched, [100])   # prefill -> RUNNING
        assert seq.status == RUNNING

        # 时间推进到 deadline 之后（用固定 now 保证测试确定性）
        now = perf_counter() + 7200
        timed_out = sched.check_deadlines(now=now)
        assert timed_out == [seq]
        assert seq.status == TIMEOUT
        assert seq.block_table == []
        assert list(sched.running) == []
        assert sched.is_finished()

    def test_timeout_then_cancel_is_idempotent(self):
        """先超时后取消（或反过来）：终态不被改写，不重复释放。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=4, deadline=perf_counter() - 1)
        sched.add(seq)
        assert sched.timeout(seq.seq_id) is True
        assert sched.cancel(seq.seq_id) is False
        assert sched.timeout(seq.seq_id) is False
        assert seq.status == TIMEOUT  # 先到先得，终态不被覆盖


class TestPreemption:
    def test_preemption_goes_through_preempted_state(self):
        """块不足时只抢占合法的 running 请求；观察点为恢复后的 WAITING，
        但迁移历史（num_preempts/last_preempted_at）记录了 PREEMPTED 事件；
        被抢占请求不参与当前 decode。"""
        sched = make_scheduler(num_blocks=2)
        a = make_seq(8, max_tokens=3)
        b = make_seq(8, max_tokens=3)
        sched.add(a)
        sched.add(b)

        seqs, items, is_prefill = schedule_round(sched)      # prefill 两条，空闲块耗尽
        assert is_prefill is True and seqs == [a, b]
        sched.postprocess(items, [7000, 7001])

        seqs, items, is_prefill = schedule_round(sched)      # decode：b 被抢占让块给 a
        assert is_prefill is False and seqs == [a]
        # b 经历了 PREEMPTED -> WAITING（resume 在同一次调度内完成）
        assert b.status == WAITING
        assert b.num_preempts == 1
        assert b.last_preempted_at is not None
        assert b.block_table == []               # 抢占即释放物理块
        assert list(sched.waiting) == [b]        # 回到等待队列队首等待 recompute
        assert a in sched.running                # 被抢占者之外仍正常 decode

    def test_resume_preserves_generated_tokens(self):
        """恢复后 token 序列、num_completion_tokens 和最终输出不丢失。"""
        sched = make_scheduler(num_blocks=2)
        a = make_seq(8, max_tokens=2)
        b = make_seq(8, max_tokens=3)
        sched.add(a)
        sched.add(b)

        run_round(sched, [7000, 7001])           # prefill：a、b 各产出 1 个 token
        run_round(sched, [7002])                 # decode a（b 被抢占）
        assert b.num_preempts == 1
        assert b.num_completion_tokens == 1
        assert 7001 in b.token_ids               # 已生成 token 保留

        # 反复调度直到全部结束：b 走重新 prefill 恢复
        for _ in range(20):
            if sched.is_finished():
                break
            seqs, items, is_prefill = schedule_round(sched)
            sched.postprocess(items, tokens_for_items(items, 8000))

        assert sched.is_finished()
        assert b.is_finished and b.num_completion_tokens == 3
        assert b.token_ids[b.num_prompt_tokens:] == [7001, 8000, 8000]  # 进度未丢失
        assert a.is_finished and a.token_ids[a.num_prompt_tokens:] == [7000, 7002]

    def test_preempt_then_cancel_from_preempted(self):
        """PREEMPTED 状态的请求可以被取消（中间态 -> 终态合法）。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [1])            # prefill -> RUNNING
        assert seq.status == RUNNING

        sched.preempt(seq)               # 直接抢占：RUNNING -> PREEMPTED
        assert seq.status == PREEMPTED
        assert seq.block_table == []     # 抢占即释放

        assert sched.cancel(seq.seq_id, reason="give_up") is True
        assert seq.status == CANCELLED
        assert seq.finish_reason == "give_up"
        assert list(sched.waiting) == [] and list(sched.running) == []
        assert sched.is_finished()


class TestInvariants:
    def test_no_duplicate_queue_membership_after_mixed_ops(self):
        """混合 cancel 与调度操作后，队列无重复成员、KV 账本保持平衡。"""
        sched = make_scheduler(num_blocks=4)
        bm = sched.block_manager
        seqs = [make_seq(4, max_tokens=8, request_id=f"req-{i}") for i in range(3)]
        for s in seqs:
            sched.add(s)

        run_round(sched, [10, 11, 12])           # prefill 3 条，各占 1 块
        sched.cancel(seqs[0].seq_id)             # 取消 a，释放其块
        run_round(sched, [13, 14])               # b、c 继续 decode
        run_round(sched, [15, 16])

        for queue in (sched.waiting, sched.running):
            assert len(queue) == len(set(queue))  # 对象唯一
        for s in seqs:
            if s.is_terminal:
                assert s.block_table == []
                assert s.seq_id not in sched.requests
        # 空闲块数 + 被占用块数 == 总块数（账本平衡）
        assert len(bm.free_block_ids) + len(bm.used_block_ids) == 4

    def test_all_finished_leaves_no_residue(self):
        """所有活动请求结束后 is_finished() 为真，且没有残留队列成员。"""
        sched = make_scheduler(num_blocks=8)
        a = make_seq(4, max_tokens=2)
        b = make_seq(6, max_tokens=2)
        sched.add(a)
        sched.add(b)

        # b 中途取消，a 自然完成
        run_round(sched, [100, 101])
        sched.cancel(b.seq_id)
        rounds = 0
        while not sched.is_finished():
            seqs, items, is_prefill = schedule_round(sched)
            sched.postprocess(items, tokens_for_items(items, 9000))
            rounds += 1
            assert rounds < 20

        assert list(sched.waiting) == [] and list(sched.running) == []
        assert sched.requests == {}
        assert a.status == FINISHED and b.status == CANCELLED


# ============================== 序列化协议（张量并行） ==============================

class TestSerialization:
    def test_pickle_roundtrip_keeps_control_plane(self):
        """v2 序列化往返：status/cancel_requested/生成进度必须同步到子进程。"""
        seq = make_seq(6, max_tokens=4)
        seq.transition_to(RUNNING)
        seq.append_token(4242)
        seq.request_cancel("mid_run")

        clone = pickle.loads(pickle.dumps(seq))
        assert clone.status == RUNNING
        assert clone.cancel_requested is True
        assert clone.num_tokens == 7
        assert clone.num_prompt_tokens == 6
        assert clone.is_prefill is True
        # 子进程不使用的字段有默认值，访问不抛 AttributeError
        assert clone.started_at is None or isinstance(clone.started_at, float)
        assert clone.request_id is not None

    def test_setstate_accepts_legacy_tuple(self):
        """旧版本 v1 六元组可被安全恢复：数据面照旧，控制面回退默认值。"""
        legacy = (6, 6, 0, 0, [0, 1], [1, 2, 3, 4, 5, 6])  # 旧格式 prefill 模式
        seq = Sequence.__new__(Sequence)  # 模拟反序列化：不经过 __init__
        seq.__setstate__(legacy)
        assert seq.num_tokens == 6
        assert seq.num_prompt_tokens == 6
        assert seq.token_ids == [1, 2, 3, 4, 5, 6]
        assert seq.status == WAITING
        assert seq.is_prefill is True
        assert seq.cancel_requested is False
        assert seq.cancel_requested is not None  # 控制面字段存在且有默认值

    def test_decode_mode_roundtrip(self):
        """decode 模式（last_state 为单个 token）的序列化保持不变。"""
        seq = make_seq(6, max_tokens=4)
        seq.transition_to(RUNNING)
        seq.is_prefill = False
        seq.append_token(77)
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.last_token == 77
        assert clone.token_ids == []  # decode 模式不传全量 token_ids（与旧行为一致）
        assert clone.num_tokens == 7
        assert clone.status == RUNNING


# ============================== LLMEngine 控制面 ==============================

class TestEngineControlPlane:
    @staticmethod
    def make_engine() -> LLMEngine:
        """跳过需要 GPU/模型的 __init__，仅装配调度器与 tokenizer 桩。"""
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = make_scheduler(num_blocks=8)
        engine.tokenizer = SimpleNamespace(encode=lambda text: [ord(c) % 1000 + 1 for c in text])
        return engine

    def test_add_request_returns_trackable_id(self):
        """add_request 返回可追踪 request_id，请求进入 waiting 队列。"""
        engine = self.make_engine()
        rid = engine.add_request("你好", SamplingParams(max_tokens=2))
        assert isinstance(rid, str) and rid
        seq = engine.get_request(rid)
        assert seq is not None and seq.request_id == rid
        assert seq.status == WAITING
        assert list(engine.scheduler.waiting) == [seq]

    def test_custom_request_id_is_preserved(self):
        engine = self.make_engine()
        rid = engine.add_request("hello", SamplingParams(), request_id="http-req-1")
        assert rid == "http-req-1"
        assert engine.get_request("http-req-1") is not None

    def test_get_request_unknown_returns_none(self):
        engine = self.make_engine()
        assert engine.get_request("no-such-id") is None

    def test_cancel_request_via_engine(self):
        """Engine 取消代理：成功返回 True；重复取消/未知 id 返回 False。"""
        engine = self.make_engine()
        rid = engine.add_request("hello", SamplingParams(max_tokens=2))
        assert engine.cancel_request(rid, reason="client_abort") is True
        seq = engine.scheduler.waiting[0] if engine.scheduler.waiting else None
        assert seq is None  # 已从队列清除
        assert engine.get_request(rid) is None
        assert engine.cancel_request(rid) is False
        assert engine.cancel_request("no-such-id") is False
        assert engine.scheduler.is_finished()


# ================== Day6 审查修复回归（docs/day6-review.md R1–R6 / N1–N2） ==================

class TestEngineStepEmptyBatch:
    """R1：schedule() 清理最后一个请求后返回空批次，Engine 不得把空批次交给 ModelRunner。"""

    @staticmethod
    def make_engine():
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = make_scheduler(num_blocks=8)
        calls = []

        def fake_call(method, items_):
            calls.append((method, len(items_)))
            out = tokens_for_items(items_, 100)
            return out if out else None

        engine.model_runner = SimpleNamespace(call=fake_call)
        return engine, calls

    def test_step_with_expired_last_request_returns_empty(self):
        engine, calls = self.make_engine()
        seq = make_seq(4, max_tokens=8, deadline=perf_counter() - 1)  # 已过期
        engine.scheduler.add(seq)
        assert engine.is_finished() is False

        assert engine.step() == ([], 0)      # 空批次契约：零输出、零工作量
        assert calls == []                   # ModelRunner 未被调用
        assert seq.status == TIMEOUT
        assert engine.is_finished() is True

    def test_step_with_cancel_flagged_last_request_returns_empty(self):
        engine, calls = self.make_engine()
        seq = make_seq(4, max_tokens=8)
        engine.scheduler.add(seq)
        seq.request_cancel("client_gone")    # 模拟执行前被标记取消

        assert engine.step() == ([], 0)
        assert calls == []
        assert seq.status == CANCELLED
        assert engine.is_finished() is True

    def test_engine_drives_request_to_completion(self):
        """正常路径：stub runner 注入采样 token，Engine 可驱动请求至终态并收集输出。"""
        engine, calls = self.make_engine()
        seq = make_seq(4, max_tokens=2)
        engine.scheduler.add(seq)

        outputs = []
        while not engine.is_finished():
            step_outputs, _ = engine.step()
            outputs.extend(step_outputs)

        assert calls and all(n > 0 for _, n in calls)      # 从未出现空批次
        assert outputs == [(seq.seq_id, [100, 100])]       # stub 每轮注入同一 token
        assert seq.status == FINISHED and seq.finish_reason == "length"


class TestPostprocessDeadline:
    """R2：模型执行期间跨过 deadline 的请求不得被记为正常完成。

    同轮同时出现取消与超时时的优先规则：取消（客户端显式意图）优先于超时。
    """

    def test_timeout_during_execution_wins_over_length(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=1, deadline=perf_counter() + 3600)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched)          # 调度时尚未超时
        assert is_prefill is True

        # 模型执行期间跨过 deadline，且本轮采样 token 恰好达到 max_tokens
        now = seq.deadline + 1
        sched.postprocess(items, [42], now=now)

        assert seq.status == TIMEOUT
        assert seq.finish_reason == "deadline_exceeded"
        assert seq.finished_at == now
        assert 42 not in seq.token_ids                # 本轮采样 token 被丢弃
        assert seq.block_table == []
        assert sched.requests == {} and sched.is_finished()

    def test_timeout_during_execution_wins_over_eos(self):
        sched = make_scheduler(num_blocks=8)
        eos = 7777
        sched.eos = eos
        seq = Sequence(list(range(1, 5)), SamplingParams(max_tokens=8, ignore_eos=False),
                       deadline=perf_counter() + 3600)
        sched.add(seq)

        batch, items, is_prefill = schedule_round(sched)
        sched.postprocess(items, [200])   # prefill（真实时钟未超时）
        batch, items, is_prefill = schedule_round(sched)
        assert is_prefill is False
        sched.postprocess(items, [eos], now=seq.deadline + 1)  # 采样到 EOS 同轮超时

        assert seq.status == TIMEOUT                  # 不是 FINISHED/stop
        assert seq.finish_reason == "deadline_exceeded"

    def test_timeout_during_chunked_prefill_releases_blocks(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        seq = make_seq(6, max_tokens=4, deadline=perf_counter() + 3600)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched)          # 第 1 个 chunk，仍为 WAITING
        assert is_prefill is True and seq.status == WAITING

        # Day8 提交契约：中间 chunk 执行不产生采样 token（真实 runner 返回 None）；
        # 即使 deadline 已过，安全路径同样丢弃输出、不推进 offset，按 TIMEOUT 终止
        sched.postprocess(items, [], now=seq.deadline + 1)

        assert seq.status == TIMEOUT
        assert seq.block_table == []                  # 已分配的 chunk block 被回收
        assert seq.num_cached_tokens == 0
        assert sched.is_finished()

    def test_cancel_wins_over_timeout_when_both_fire(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8, deadline=perf_counter() + 3600)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched)
        seq.request_cancel("client_gone")             # 取消标记与过期 deadline 同时存在

        sched.postprocess(items, [42], now=seq.deadline + 1)

        assert seq.status == CANCELLED                # 取消优先于超时
        assert seq.finish_reason == "client_gone"


class TestTerminalPurgeAtScheduleBoundary:
    """R3：外部公开方法（mark_*）产生的终态请求，在调度边界必须被兜底清理。"""

    def test_terminal_in_running_queue_purged_before_decode(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])                        # prefill -> RUNNING
        seq.mark_timeout()                            # 公开方法，绕过 Scheduler 入口

        batch, items, _ = schedule_round(sched)
        assert batch == []                            # 终态请求不再进入 decode
        assert seq.status == TIMEOUT
        assert seq.block_table == []
        assert sched.requests == {} and sched.is_finished()

    def test_terminal_in_waiting_queue_purged_before_allocate(self):
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        seq.mark_cancelled("external")                # WAITING 中被外部置为终态

        batch, items, _ = schedule_round(sched)                   # 不得先分配 block 再抛迁移异常
        assert batch == []
        assert seq.block_table == []
        assert len(bm.free_block_ids) == 8            # 资源从未分配
        assert sched.requests == {} and sched.is_finished()

    def test_timeout_entry_cleanups_terminal_held_seq(self):
        """对已终态但仍被持有的请求，控制入口返回 False 且完成剩余清理。"""
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])
        seq.mark_timeout()

        assert sched.timeout(seq.seq_id) is False
        assert sched.cancel(seq.seq_id) is False
        assert seq.block_table == [] and list(sched.running) == []
        assert sched.requests == {}

    def test_mark_finished_seq_in_running_cleaned(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])
        seq.mark_finished("stop")

        batch, items, _ = schedule_round(sched)
        assert batch == [] and seq.block_table == [] and sched.is_finished()


class TestRequestIdUniqueness:
    """R4：活动 request_id 必须唯一，取消才能唯一定位；拒绝时不得污染状态。"""

    def test_duplicate_request_id_rejected_atomically(self):
        sched = make_scheduler(num_blocks=8)
        s1 = make_seq(request_id="same")
        sched.add(s1)
        s2 = make_seq(request_id="same")

        with pytest.raises(ValueError):
            sched.add(s2)

        assert list(sched.requests.values()) == [s1]  # 失败零污染
        assert list(sched.waiting) == [s1] and not sched.running
        assert s2.seq_id not in sched.requests

    def test_custom_id_colliding_with_auto_id_rejected(self):
        sched = make_scheduler(num_blocks=8)
        s1 = make_seq()                               # 自动 ID：req-<seq_id>
        sched.add(s1)
        s2 = make_seq(request_id=s1.request_id)       # 自定义 ID 与自动 ID 碰撞
        with pytest.raises(ValueError):
            sched.add(s2)

    def test_engine_rejects_duplicate_custom_id(self):
        engine = TestEngineControlPlane.make_engine()
        rid = engine.add_request("a", SamplingParams(max_tokens=2), request_id="same")
        with pytest.raises(ValueError):
            engine.add_request("b", SamplingParams(max_tokens=2), request_id="same")

        # 取消仍能唯一定位第一个请求；重复取消返回 False
        assert engine.cancel_request(rid) is True
        assert engine.cancel_request(rid) is False
        assert engine.get_request(rid) is None

    def test_id_reusable_after_terminal(self):
        """终态请求从活动索引移除后，其 request_id 允许复用（Day6 策略）。"""
        sched = make_scheduler(num_blocks=8)
        s1 = make_seq(request_id="dup")
        sched.add(s1)
        sched.cancel(s1.seq_id)
        s2 = make_seq(request_id="dup")
        sched.add(s2)                                 # 不抛异常
        assert sched.requests[s2.seq_id] is s2


class TestPausedRequests:
    """R5：独立 PREEMPTED（未 resume）的请求仍是非终态，参与扫描与完成判断。"""

    def test_paused_request_counts_as_unfinished(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])
        sched.preempt(seq)                            # 只抢占，不 resume

        assert seq.status == PREEMPTED
        assert seq.seq_id in sched.requests
        assert sched.is_finished() is False           # 暂停不是完成

    def test_paused_request_deadline_enforced(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])
        sched.preempt(seq)
        seq.deadline = perf_counter() - 1             # 暂停期间 deadline 过期

        batch, items, _ = schedule_round(sched)                   # 扫描范围为活动索引，覆盖暂停请求
        assert batch == []
        assert seq.status == TIMEOUT
        assert sched.requests == {} and sched.is_finished()

    def test_paused_request_cancel_flag_processed(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        run_round(sched, [10])
        sched.preempt(seq)
        seq.request_cancel("gone")

        batch, items, _ = schedule_round(sched)
        assert batch == []
        assert seq.status == CANCELLED and seq.finish_reason == "gone"
        assert sched.is_finished()

    def test_paused_request_resume_and_complete(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=2)
        sched.add(seq)
        run_round(sched, [10])                        # prefill，产出首个 completion token
        sched.preempt(seq)                            # 暂停
        sched.resume(seq)                             # 显式恢复 -> WAITING
        assert seq.status == WAITING

        while not sched.is_finished():
            batch, items, is_prefill = schedule_round(sched)
            sched.postprocess(items, tokens_for_items(items, 60))

        assert seq.is_finished
        assert seq.token_ids[seq.num_prompt_tokens:] == [10, 60]  # 进度未丢失


class TestAddAtomicity:
    """R6：add() 校验失败时，索引、队列与资源必须与调用前完全一致。"""

    @staticmethod
    def snapshot(sched):
        return (
            dict(sched.requests),
            list(sched.waiting),
            list(sched.running),
            len(sched.block_manager.free_block_ids),
            sched.is_finished(),
        )

    def test_add_rejects_terminal_seq_without_pollution(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        seq.mark_cancelled("already_dead")            # 终态对象不允许入队
        before = self.snapshot(sched)

        with pytest.raises(ValueError):
            sched.add(seq)

        assert self.snapshot(sched) == before         # 失败零污染

    def test_add_rejects_running_seq_without_pollution(self):
        sched_a = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched_a.add(seq)
        run_round(sched_a, [10])                      # seq 已是 RUNNING
        sched_b = make_scheduler(num_blocks=8)
        before = self.snapshot(sched_b)

        with pytest.raises(ValueError):
            sched_b.add(seq)

        assert self.snapshot(sched_b) == before

    def test_duplicate_seq_id_rejected_without_pollution(self):
        sched = make_scheduler(num_blocks=8)
        seq = make_seq(4, max_tokens=8)
        sched.add(seq)
        before = self.snapshot(sched)

        with pytest.raises(ValueError):
            sched.add(seq)

        assert self.snapshot(sched) == before
        assert len(sched.requests) == 1


class TestSharedPrefixCancellation:
    """共享前缀块的取消语义：单个持有者取消不回收共享块，最后持有者回收后缓存哈希保留。"""

    def test_cancel_one_holder_keeps_shared_blocks(self):
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        prompt = list(range(1, 9))                    # 1 个完整块作为共享前缀
        a = Sequence(prompt + [100], SamplingParams(max_tokens=2, ignore_eos=True))
        b = Sequence(prompt + [200], SamplingParams(max_tokens=3, ignore_eos=True))

        sched.add(a)
        run_round(sched, [10])                        # a prefill，block0 哈希登记
        sched.add(b)
        # Day9 decode-first：a 已 RUNNING，本轮先 decode a，再接纳 b 的 prefill
        # （b 命中 a 的前缀块，共享 ref_count=2）
        batch, items, is_prefill = schedule_round(sched)
        assert [(it.phase, it.seq.seq_id) for it in items] == [
            ("decode", a.seq_id), ("prefill", b.seq_id)]
        assert b.num_cached_tokens == 8
        assert b.block_table[0] == a.block_table[0]
        shared = a.block_table[0]
        assert bm.blocks[shared].ref_count == 2

        sched.cancel(a.seq_id)                        # 取消一个持有者
        assert bm.blocks[shared].ref_count == 1       # 共享块不回收
        assert b.block_table[0] == shared

        while not sched.is_finished():
            batch, items, is_prefill = schedule_round(sched)
            sched.postprocess(items, tokens_for_items(items, 30))

        assert b.is_finished
        assert bm.blocks[shared].ref_count == 0       # 最后持有者完成后回收
        assert len(bm.hash_to_block_id) == 1          # 缓存哈希保留


class TestSerializationRobustness:
    """N1/N2：版本号显式校验 + legacy decode 元组的 is_prefill 恢复。"""

    def test_unknown_version_rejected(self):
        seq = make_seq(6, max_tokens=4)
        version, payload = seq.__getstate__()
        assert version == Sequence.STATE_VERSION
        clone = Sequence.__new__(Sequence)
        with pytest.raises(ValueError):
            clone.__setstate__((999, payload))        # 未知版本报清晰错误

    def test_legacy_decode_tuple_restores_prefill_mode(self):
        legacy = (3, 2, 2, 1, [0], 77)                # 旧 decode 格式：last_state 为标量
        seq = Sequence.__new__(Sequence)
        seq.__setstate__(legacy)
        assert seq.is_prefill is False                # 按 last_state 类型恢复
        assert seq.last_token == 77
        assert seq.token_ids == []

        # 修复前：is_prefill 被错误置 True，再次 pickle 往返会以空列表作
        # prefill payload，在 last_token = token_ids[-1] 处抛 IndexError
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.is_prefill is False
        assert clone.last_token == 77
        assert clone.status == WAITING


# ================== Day6 二轮审查修复回归（day6-review.md 二轮 B / C） ==================

class TestStaleCleanupIdOwnership:
    """二轮 B：旧终态对象的延迟/重复收尾不得误删复用 ID 的新请求唯一性标记。

    触发路径均为公开入口：取消后 ID 复用再收到旧批次的后处理，
    或正常完成后 ID 复用再重复执行同一批次的后处理。
    """

    @staticmethod
    def make_engine() -> LLMEngine:
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = make_scheduler(num_blocks=8)
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text],
            decode=lambda tokens: "stub",
        )
        engine.model_runner = SimpleNamespace(
            call=lambda m, items_: tokens_for_items(items_, 10))
        return engine

    @pytest.mark.parametrize("ending", ["cancel", "finish"])
    def test_stale_postprocess_after_id_reuse_keeps_new_owner(self, ending):
        engine = self.make_engine()
        rid = engine.add_request([1, 2], SamplingParams(max_tokens=1, ignore_eos=True),
                                 request_id="same")
        a = engine.get_request(rid)
        batch, items, is_prefill = schedule_round(engine.scheduler)

        if ending == "cancel":
            engine.cancel_request(rid)                    # A 取消，ID 标记释放
        else:
            engine.scheduler.postprocess(items, [10])  # A 正常完成
        assert a.is_terminal

        # 复用 ID 注册 B（公开允许的策略）
        b_rid = engine.add_request([3, 4], SamplingParams(max_tokens=8, ignore_eos=True),
                                   request_id="same")
        b = engine.get_request(b_rid)
        assert b is not a and b.request_id == "same"

        # 对 A 的旧批次执行延迟（cancel 路径）/重复（finish 路径）后处理
        engine.scheduler.postprocess(items, [10])

        # B 的身份记录不受旧对象收尾影响
        assert engine.get_request("same") is b
        assert engine.scheduler._active_request_ids == {"same"}

        # C 复用同一 ID 仍被拒绝（B 仍活动）
        with pytest.raises(ValueError):
            engine.add_request([5, 6], SamplingParams(max_tokens=8, ignore_eos=True),
                               request_id="same")

        # B 只能被取消一次，且作用于 B 本身
        assert engine.cancel_request("same") is True
        assert b.status == CANCELLED
        assert engine.cancel_request("same") is False

    def test_finalize_ownership_is_bound_to_index_registration(self):
        """调度器级直接验证：_finalize 对已移出索引的对象不触碰活动 ID 集合。"""
        sched = make_scheduler(num_blocks=8)
        s1 = make_seq(request_id="dup")
        sched.add(s1)
        sched.cancel(s1.seq_id)                # s1 收尾，ID 标记释放
        assert sched._active_request_ids == set()

        s2 = make_seq(request_id="dup")        # 复用 ID
        sched.add(s2)
        sched._finalize(s1)                    # 旧对象重复收尾
        assert sched._active_request_ids == {"dup"}   # 新所有者的标记保持
        assert sched.requests[s2.seq_id] is s2
        sched._finalize(s2)                    # 所有者收尾才清除
        assert sched._active_request_ids == set() and sched.requests == {}


class TestGeneratePausedContract:
    """二轮 C：同步 generate() 遇到健康暂停请求时显式报错，不忙循环、不伪装完成。

    覆盖：健康暂停、暂停后到期、暂停后取消标记、显式恢复四种路径，
    均通过真实 generate() 驱动（仅 stub 注入采样 token）。
    """

    @staticmethod
    def make_engine() -> LLMEngine:
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = make_scheduler(num_blocks=8)
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text],
            decode=lambda tokens: "stub",
        )
        engine.model_runner = SimpleNamespace(call=lambda m, items_: tokens_for_items(items_, 7))
        return engine

    @staticmethod
    def pause_running(engine):
        """用真实 step() 把请求推进到 RUNNING 后暂停（公开 preempt 入口）。"""
        rid = engine.add_request([1, 2], SamplingParams(max_tokens=8, ignore_eos=True))
        seq = engine.get_request(rid)
        engine.step()                          # prefill -> RUNNING
        assert seq.status == RUNNING
        engine.scheduler.preempt(seq)          # 暂停，不 resume
        assert seq.status == PREEMPTED
        return rid, seq

    def test_healthy_paused_request_raises_instead_of_busy_loop(self):
        engine = self.make_engine()
        rid, seq = self.pause_running(engine)

        with pytest.raises(RuntimeError) as excinfo:
            engine.generate([[3, 4]], SamplingParams(max_tokens=1, ignore_eos=True),
                            use_tqdm=False)

        # 错误信息可定位需要处理的暂停请求；状态未被伪装
        assert seq.request_id in str(excinfo.value)
        assert seq.status == PREEMPTED
        assert engine.is_finished() is False

    def test_paused_request_deadline_resolved_by_generate(self):
        engine = self.make_engine()
        _, seq = self.pause_running(engine)
        seq.deadline = perf_counter() - 1      # 暂停期间到期

        outputs = engine.generate([[3, 4]], SamplingParams(max_tokens=1, ignore_eos=True),
                                  use_tqdm=False)

        assert seq.status == TIMEOUT and seq.finish_reason == "deadline_exceeded"
        assert len(outputs) == 1               # 新请求正常完成
        assert engine.is_finished() is True

    def test_paused_request_cancel_flag_resolved_by_generate(self):
        engine = self.make_engine()
        _, seq = self.pause_running(engine)
        seq.request_cancel("client_gone")

        outputs = engine.generate([[3, 4]], SamplingParams(max_tokens=1, ignore_eos=True),
                                  use_tqdm=False)

        assert seq.status == CANCELLED and seq.finish_reason == "client_gone"
        assert len(outputs) == 1 and engine.is_finished() is True

    def test_paused_request_explicit_resume_completes(self):
        engine = self.make_engine()
        rid, seq = self.pause_running(engine)
        engine.scheduler.resume(seq)           # 显式恢复 -> WAITING，参与后续调度

        outputs = engine.generate([[3, 4]], SamplingParams(max_tokens=1, ignore_eos=True),
                                  use_tqdm=False)

        assert seq.is_finished and seq.num_completion_tokens == 8  # 恢复后跑满 max_tokens
        assert len(outputs) == 2               # 暂停恢复的请求与新请求都有输出
        assert engine.is_finished() is True
