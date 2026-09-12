"""Day 10 验收测试：取消、超时、抢占与恢复（docs/request-cancellation-preemption.md §9.1 测试矩阵）。

覆盖范围（对应文档 §9.1 表格逐行）：
- 控制信号：waiting/running/PREEMPTED 请求取消；重复取消；自定义 reason；
  request_cancel 信号只置位不改状态/队列/KV；首次原因不被覆盖；
- deadline：deadline 恰好等于 now、早/晚 1 tick；waiting/running/暂停全覆盖；
  构造入口校验 deadline >= created_at（单调时钟，可注入固定时钟）；
- 优先级：cancel 与 timeout 同时命中 -> CANCELLED；postprocess 前置取消/超时
  不追加 token、不推进 offset；已提交 token 不被迟到取消撤销；
- 混合轮：decode/prefill item 单独取消/超时，其他 item 正常提交、采样对齐；
- 抢占状态机：RUNNING→PREEMPTED；重复抢占/终态/WAITING 抢占显式异常；
  resume 只允许 PREEMPTED -> WAITING；重复恢复拒绝；
- 恢复进度：抢占前后 prompt/completion token、计数、采样参数不变；
  恢复重新 prefill 后不重复追加 completion；
- victim：队尾优先；排除本轮已接纳 item/有计划对象；候选不足继续搜索；
  无合法 victim 时延后（保留 RUNNING/KV）；prefill 侧抢占后重新查询容量；
- KV 账本：cancel/timeout/finish/preempt/resume 重复组合后 free/used 互斥、
  总数守恒、ref_count 不为负、无 double free（check_ledger）；
- 异常清理：runner 抛错、postprocess 抛错、异常重复清理 -> 全活动请求收尾、
  队列/索引为空、Engine 禁止重试；abort 尊重对象所有权（ID 复用不误伤）；
- ID/迟到：终态后复用 request_id；旧 item 迟到 postprocess 被 round_id 拒绝；
- 序列化：v1/v2/v3 读取、当前版本往返、取消/状态字段不静默丢失；
- 无进展：全部 PREEMPTED 或 KV 无法接纳时 generate() 显式报错，不忙循环。

无 GPU / 模型依赖：Scheduler/BlockManager/Sequence 为纯 CPU 逻辑；LLMEngine 通过
__new__ 跳过 GPU __init__，runner 为契约桩；时间使用可注入的固定时钟
（Sequence.clock + schedule/postprocess 的 now 关键字），不加载权重、不 sleep、
不访问 CUDA。采样 token 由测试直接注入，Sequence.block_size 手动对齐。

重复执行命令：
    python -m pytest tests/test_request_control.py -q
    python -O -m pytest tests/test_request_control.py -q
"""

import json
import logging
import pickle
from types import SimpleNamespace

import pytest

from nanovllm.engine import model_runner as model_runner_module
from nanovllm.engine.block_manager import BlockManager
from scripts.validate_request_control import replay_control_transitions
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import (InvalidStateTransition, Sequence,
                                      SequenceStatus)
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 8
EOS = 999_999  # 测试中不会出现的 token id，配合 ignore_eos 使用

# 真实运行时由 LLMEngine 设置（nanovllm/engine/llm_engine.py），测试手动对齐
Sequence.block_size = BLOCK_SIZE

WAITING = SequenceStatus.WAITING
RUNNING = SequenceStatus.RUNNING
PREEMPTED = SequenceStatus.PREEMPTED
FINISHED = SequenceStatus.FINISHED
CANCELLED = SequenceStatus.CANCELLED
TIMEOUT = SequenceStatus.TIMEOUT


# ============================== 公共夹具与构造工具 ==============================

@pytest.fixture(autouse=True)
def test_clock():
    """可注入固定时钟（§6.2 "测试可注入固定时钟"）。

    Sequence 构造校验 deadline >= created_at（单调时钟）。把 Sequence.clock
    注入为可控虚拟时间轴（默认 1.0），使构造校验与调度注入的 now 同轴；
    测试内可通过 holder["now"] = x 推进时间。结束恢复真实时钟防泄漏。
    """
    holder = {"now": 1.0}
    original = Sequence.clock
    Sequence.clock = staticmethod(lambda: holder["now"])
    yield holder
    Sequence.clock = original


def make_seq(num_tokens: int = 4, max_tokens: int = 8, request_id: str | None = None,
             deadline: float | None = None, temperature: float = 1.0,
             seed: int | None = None) -> Sequence:
    return Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True,
                       temperature=temperature, seed=seed),
        request_id=request_id,
        deadline=deadline,
    )


def make_scheduler(num_blocks: int = 32, max_num_batched_tokens: int = 10 ** 6,
                   max_num_seqs: int = 512, chunk_size: int = 1024) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        chunk_size=chunk_size,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def tokens_for(items, token: int = 7) -> list[int]:
    """按 Day9 提交契约构造 postprocess 注入列表（与 needs_sample 快照共享谓词）。"""
    return [token for it in items if it.needs_sample]


def drive_round(sched: Scheduler, now: float, token: int = 7):
    """执行一轮 schedule + postprocess（注入按 needs_sample 对齐），返回 (items, phase)。"""
    items, phase = sched.schedule(now=now)
    sched.postprocess(items, tokens_for(items, token), now=now)
    return items, phase


def decision_of(stats: dict, seq: Sequence) -> dict:
    for d in stats["decisions"]:
        if d["seq_id"] == seq.seq_id:
            return d
    raise AssertionError(f"seq_id={seq.seq_id} 不在本轮决策中: {stats['decisions']}")


def setup_decode_source(sched: Scheduler, count: int = 1, now: float = 2.0,
                        max_tokens: int = 100, num_tokens: int = 1) -> list[Sequence]:
    """构造 count 条 RUNNING 的 decode 源（prefill 完成即 RUNNING），返回 seqs。

    有界轮次内驱动（防挂）；last-chunk prefill 采样使每条已带 1 个 completion，
    因此 setup 结束后 len = num_tokens + 1。
    """
    seqs = []
    t = now
    for _ in range(count):
        seq = make_seq(num_tokens, max_tokens=max_tokens)
        sched.add(seq)
        for _ in range(200):
            drive_round(sched, t)
            t += 1.0
            if seq.status == RUNNING:
                break
        else:
            raise AssertionError("setup 未在有界轮次内完成")
        seqs.append(seq)
    return seqs


def make_engine(sched: Scheduler, runner=None):
    """构造跳过 GPU __init__ 的 Engine 桩；runner 缺省为按 needs_sample 注入 7。"""
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = sched
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: [ord(c) % 100 + 1 for c in text],
        decode=lambda tokens: "stub",
    )
    if runner is None:
        def runner(method, items_):
            out = tokens_for(items_, 7)
            return out if out else None
    engine.model_runner = SimpleNamespace(call=runner)
    return engine


# ============================== 控制信号（§9.1 行 1） ==============================

class TestControlSignal:
    """request_cancel 信号语义与 cancel 安全点收尾。"""

    def test_request_cancel_signal_only_waiting(self):
        """waiting 请求的取消信号：只置位，不改状态/队列/资源。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        assert sched.request_cancel(seq.seq_id, "client_gone") is True
        assert seq.cancel_requested and seq.cancel_reason == "client_gone"
        assert seq.status == WAITING and seq in sched.waiting
        assert seq.block_table == [] and sched.block_manager.used_block_ids == set()

    def test_request_cancel_signal_only_running(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        used = set(sched.block_manager.used_block_ids)
        assert sched.request_cancel(seq.seq_id, "why") is True
        assert seq.cancel_requested and seq.status == RUNNING
        assert seq in sched.running and seq.block_table
        assert sched.block_manager.used_block_ids == used  # 账本未被信号改动

    def test_request_cancel_signal_only_preempted(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        sched.preempt(seq, now=3.0)     # 显式抢占：暂停在队列之外
        assert sched.request_cancel(seq.seq_id, "pause_cancel") is True
        assert seq.status == PREEMPTED and seq.cancel_requested
        assert seq.seq_id in sched.requests and seq not in sched.waiting
        assert seq not in sched.running

    def test_request_cancel_unknown_or_terminal(self):
        sched = make_scheduler()
        assert sched.request_cancel(424242) is False
        seq = make_seq()
        sched.add(seq)
        sched.cancel(seq.seq_id, now=2.0)
        assert seq.is_terminal
        assert sched.request_cancel(seq.seq_id) is False  # 终态：信号拒绝

    def test_repeat_cancel_keeps_first_reason(self):
        """重复取消返回成功但不覆盖首次原因；终态原因沿用首次记录。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        assert sched.request_cancel(seq.seq_id, "first") is True
        assert sched.request_cancel(seq.seq_id, "second") is True
        assert seq.cancel_reason == "first"
        assert sched.cancel(seq.seq_id, "third", now=2.0) is True
        assert seq.status == CANCELLED and seq.finish_reason == "first"
        # 终态后重复取消：返回 False，原因仍不被覆盖
        assert sched.cancel(seq.seq_id, "fourth", now=3.0) is False
        assert seq.finish_reason == "first"

    def test_sequence_direct_mark_cancelled_keeps_signal_reason(self):
        seq = make_seq()
        seq.request_cancel("direct_reason")
        assert seq.mark_cancelled(now=2.0) is True
        assert seq.finish_reason == "direct_reason"

    @pytest.mark.parametrize("state", ["waiting", "running", "preempted"])
    def test_cancel_from_each_active_state(self, state):
        """waiting/running/PREEMPTED 三类活动请求均可经 cancel() 安全点收尾。"""
        sched = make_scheduler()
        seq = make_seq(16, max_tokens=4)
        sched.add(seq)
        if state == "running":
            drive_round(sched, 2.0)          # 完成 prefill -> RUNNING
        elif state == "preempted":
            drive_round(sched, 2.0)
            sched.preempt(seq, now=3.0)      # 暂停在队列之外
        used_before = len(sched.block_manager.used_block_ids)
        assert sched.cancel(seq.seq_id, "bye", now=4.0) is True
        assert seq.status == CANCELLED and seq.finish_reason == "bye"
        assert seq.finished_at == 4.0
        assert seq.seq_id not in sched.requests
        assert seq not in sched.waiting and seq not in sched.running
        assert len(sched.block_manager.used_block_ids) < used_before or used_before == 0
        assert sched.is_finished()
        sched.block_manager.check_ledger()

    def test_cancel_releases_exclusive_blocks(self):
        """取消释放全部可回收 block：事件与账本差值一致（released_blocks）。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)   # 1 个 prompt 块
        held = len(seq.block_table)
        used_before = len(sched.block_manager.used_block_ids)
        sched.cancel(seq.seq_id, now=3.0)
        assert len(sched.block_manager.used_block_ids) == used_before - held
        assert seq.block_table == [] and seq.prefill_offset == 0


# ============================== deadline / 超时（§9.1 行 2） ==============================

class TestDeadline:
    """单调时钟 deadline 语义：安全点检查、覆盖三类活动状态、构造校验。"""

    def test_constructor_validates_deadline_not_before_created_at(self, test_clock):
        test_clock["now"] = 10.0
        with pytest.raises(ValueError, match="deadline"):
            make_seq(4, deadline=9.0)            # 早于 created_at：拒绝
        assert make_seq(4, deadline=10.0).deadline == 10.0   # 等于 created_at：允许
        assert make_seq(4, deadline=None).deadline is None
        assert make_seq(4, deadline=11.0).deadline == 11.0

    def test_deadline_equal_now_times_out(self):
        """now == deadline 即超时（now >= deadline 语义，§9.1 行 2）。"""
        sched = make_scheduler()
        seq = make_seq(4, deadline=5.0)
        sched.add(seq)
        timed_out = sched.check_deadlines(now=5.0)
        assert timed_out == [seq]
        assert seq.status == TIMEOUT and seq.finish_reason == "deadline_exceeded"
        assert seq.seq_id not in sched.requests

    def test_deadline_one_tick_boundary(self):
        sched = make_scheduler()
        seq = make_seq(4, deadline=5.0)
        sched.add(seq)
        assert sched.check_deadlines(now=4.999) == []   # 未到期：保留
        assert seq.status == WAITING
        assert sched.check_deadlines(now=5.001) == [seq]  # 过期 1 tick：终止
        assert seq.status == TIMEOUT

    @pytest.mark.parametrize("state", ["waiting", "running", "paused"])
    def test_deadline_covers_waiting_running_paused(self, state):
        """check_deadlines 扫描活动索引：waiting/running/暂停请求都被覆盖。"""
        sched = make_scheduler()
        seq = make_seq(16, deadline=50.0)
        sched.add(seq)
        if state in ("running", "paused"):
            drive_round(sched, 2.0)          # -> RUNNING
        if state == "paused":
            sched.preempt(seq, now=3.0)      # 暂停在队列之外
        timed_out = sched.check_deadlines(now=50.0)
        assert timed_out == [seq]
        assert seq.status == TIMEOUT
        assert seq not in sched.waiting and seq not in sched.running
        assert sched.requests == {}          # 活动索引清空
        sched.block_manager.check_ledger()

    def test_running_timeout_at_schedule_boundary(self):
        """运行中跨 deadline：下一轮 schedule 边界统一清理并释放资源。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched, num_tokens=4, max_tokens=8)
        seq.deadline = 100.0
        drive_round(sched, 3.0)              # 正常 decode
        assert seq.status == RUNNING
        items, phase = sched.schedule(now=101.0)   # 调度边界发现超时
        assert all(it.seq is not seq for it in items)
        assert seq.status == TIMEOUT and sched.requests == {}
        assert seq.block_table == []
        sched.block_manager.check_ledger()


# ============================== 优先级（§9.1 行 3） ==============================

class TestPriority:
    """取消 > 超时 > 正常提交；命中后不追加 token、不推进 KV。"""

    def test_cancel_beats_timeout_when_both_hit(self):
        """取消信号与过期 deadline 同时存在：CANCELLED 优先，原因不被改写。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        seq.deadline = 2.0                   # 已过期
        sched.request_cancel(seq.seq_id, "client_wins")
        items, phase = sched.schedule(now=3.0)
        assert all(it.seq is not seq for it in items)
        assert seq.status == CANCELLED       # 取消优先
        assert seq.finish_reason == "client_wins"
        assert seq.finish_reason != "deadline_exceeded"

    def test_cancel_flag_beats_deadline_in_postprocess(self, test_clock):
        """postprocess 安全检查：cancel_requested 优先于 deadline 检查。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        items, _ = sched.schedule(now=10.0)
        seq.deadline = 9.0                   # 模型执行期间跨过 deadline
        sched.request_cancel(seq.seq_id, "both")  # 同时置位取消
        sched.postprocess(items, tokens_for(items), now=11.0)
        assert seq.status == CANCELLED and seq.finish_reason == "both"
        # 本轮采样未提交（completion 计数不变）；终态释放后有效 KV 进度归零
        assert seq.num_completion_tokens == 1
        assert seq.prefill_offset == 0

    def test_postprocess_cancel_discards_sample(self):
        """postprocess 前置取消：不 hash、不推进 offset、不追加 token。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        comp_before = seq.num_completion_tokens
        items, _ = sched.schedule(now=10.0)
        assert any(it.seq is seq for it in items)
        sched.request_cancel(seq.seq_id, "mid_exec")
        sched.postprocess(items, tokens_for(items), now=10.0)
        assert seq.status == CANCELLED and seq.finish_reason == "mid_exec"
        assert seq.num_completion_tokens == comp_before
        # 本轮 offset 未推进（释放后归零是设计内行为，非本轮提交所得）
        assert seq.prefill_offset == 0
        assert seq.seq_id not in sched.requests
        sched.block_manager.check_ledger()

    def test_postprocess_timeout_discards_sample(self):
        """postprocess 前置超时：模型执行跨过 deadline 的请求不得记为正常完成。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        items, _ = sched.schedule(now=10.0)
        seq.deadline = 10.5                  # schedule 后、提交前跨过
        sched.postprocess(items, tokens_for(items), now=11.0)
        assert seq.status == TIMEOUT and seq.finish_reason == "deadline_exceeded"
        assert len(seq.completion_token_ids) == 1   # 无本轮 token
        sched.block_manager.check_ledger()

    def test_direct_timeout_respects_existing_cancel_signal(self):
        """直接调用 timeout 也必须遵守 CANCELLED 优先于 TIMEOUT。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        assert sched.request_cancel(seq.seq_id, "client_wins") is True
        assert sched.timeout(seq.seq_id, now=2.0) is True
        assert seq.status == CANCELLED
        assert seq.finish_reason == "client_wins"

    def test_committed_finish_not_revoked_by_late_cancel(self):
        """已进入提交区并完成 token 追加的请求可正常结束：迟到取消不撤销。

        提交完成后再置取消信号：本轮结果不回滚；下一轮调度边界才按
        CANCELLED 收尾（安全点线性化，§4.4）。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        items, _ = sched.schedule(now=10.0)
        sched.postprocess(items, tokens_for(items), now=10.0)
        assert seq.status == RUNNING         # 正常提交，未完成
        comp = seq.num_completion_tokens
        sched.request_cancel(seq.seq_id, "too_late")
        # 信号不追溯已提交 token：状态与计数保持
        assert seq.status == RUNNING
        assert seq.num_completion_tokens == comp
        # 下一轮调度边界：取消收尾，token 既不撤销也不追加
        sched.schedule(now=11.0)
        assert seq.status == CANCELLED and seq.finish_reason == "too_late"
        assert seq.num_completion_tokens == comp


# ============================== 混合轮隔离（§9.1 行 4） ==============================

class TestMixedRoundIsolation:
    """混合轮单 item 取消/超时不影响其他 item 的提交与采样对齐。"""

    def test_cancelled_decode_item_keeps_alignment(self):
        """decode item 取消：占位 token 被丢弃，同轮 prefill 最后 chunk 正常提交。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched)
        c = make_seq(7, max_tokens=4)        # 7 token <= 剩余预算：本轮最后 chunk
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        assert [it.phase for it in items] == ["decode", "prefill"]
        it_c = next(it for it in items if it.phase == "prefill")
        assert it_c.is_last_chunk            # c 最后 chunk 需采样
        sched.request_cancel(a.seq_id, "drop_a")   # 模拟 forward 期间取消
        # 注入按 needs_sample 对齐：decode a 与 prefill c 各占 1 个位置
        samples = [70, 80]
        sched.postprocess(items, samples, now=10.0)
        assert a.status == CANCELLED
        assert a.num_completion_tokens == 1  # a 的采样被丢弃（无追加）
        assert c.status == RUNNING
        assert c.token_ids[-1] == 80         # c 消费自己占位的采样结果
        assert c.num_completion_tokens == 1
        sched.block_manager.check_ledger()

    def test_timeout_mid_chunk_other_items_commit(self):
        """mid-chunk prefill 超时：输出丢弃、其他 item 正常提交（同轮隔离）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched)
        c = make_seq(20, max_tokens=4, deadline=13.0)
        sched.add(c)
        items, _ = sched.schedule(now=10.0)  # c 中间 chunk（q=7，不采样）
        assert any(it.seq is c for it in items)
        sched.postprocess(items, tokens_for(items), now=11.0)
        assert c.status == WAITING and c.prefill_offset == 7
        items, _ = sched.schedule(now=12.0)  # c 续 chunk 被接纳
        assert any(it.seq is c for it in items)
        sched.postprocess(items, tokens_for(items), now=14.0)  # 已跨 deadline
        assert c.status == TIMEOUT and c.prefill_offset == 0
        assert a.num_completion_tokens == 3  # 同轮 decode 正常提交
        sched.block_manager.check_ledger()


# ============================== 抢占状态机（§9.1 行 5） ==============================

class TestPreemptStateMachine:
    """preempt/resume 的合法迁移与显式拒绝。"""

    def test_preempt_running_to_preempted(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        held = len(seq.block_table)
        used_before = len(sched.block_manager.used_block_ids)
        sched.preempt(seq, now=5.0)
        assert seq.status == PREEMPTED and seq.num_preempts == 1
        assert seq.last_preempted_at == 5.0
        assert seq.is_prefill is True        # 恢复后按 prefill recompute
        assert seq not in sched.running and seq not in sched.waiting
        assert seq.seq_id in sched.requests  # 仍在活动索引（参与扫描/完成判断）
        assert len(seq.block_table) == 0
        assert len(sched.block_manager.used_block_ids) == used_before - held
        assert seq.prefill_offset == 0       # 有效 KV 进度随释放归零

    def test_preempt_preserves_logical_tokens(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        tokens_snapshot = list(seq.token_ids)
        prompt_snapshot = list(seq.prompt_token_ids)
        comp_before = seq.num_completion_tokens
        sched.preempt(seq, now=5.0)
        assert seq.token_ids == tokens_snapshot
        assert seq.prompt_token_ids == prompt_snapshot
        assert seq.num_completion_tokens == comp_before
        assert seq.temperature == 1.0 and seq.max_tokens == 100 and seq.ignore_eos

    def test_repeat_preempt_rejected_without_side_effects(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        sched.preempt(seq, now=5.0)
        used = set(sched.block_manager.used_block_ids)
        with pytest.raises(InvalidStateTransition):
            sched.preempt(seq, now=6.0)      # PREEMPTED -> PREEMPTED 非法
        assert seq.num_preempts == 1
        assert seq.status == PREEMPTED
        assert sched.block_manager.used_block_ids == used

    @pytest.mark.parametrize("setup_state", ["waiting", "finished", "cancelled"])
    def test_preempt_non_running_rejected(self, setup_state):
        """WAITING/终态对象抢占被显式拒绝（合法 victim 只能是 RUNNING）。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        if setup_state in ("finished", "cancelled"):
            drive_round(sched, 2.0)          # 先到 RUNNING（正常完成的前提）
            if setup_state == "finished":
                seq.mark_finished("stop", now=3.0)
            else:
                seq.mark_cancelled("x", now=3.0)
        with pytest.raises(InvalidStateTransition):
            sched.preempt(seq, now=4.0)

    def test_resume_only_from_preempted_and_front_insert(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        older = make_seq(2)
        sched.add(older)                     # waiting: [older]
        sched.preempt(seq, now=5.0)
        sched.resume(seq, now=5.0)
        assert seq.status == WAITING
        assert seq is sched.waiting[0]       # 恢复请求插入 waiting 队首（优先重算）
        with pytest.raises(InvalidStateTransition):
            sched.resume(seq, now=6.0)       # WAITING -> WAITING 非法（重复恢复）

    def test_unregistered_preempt_and_resume_have_no_side_effect(self):
        """外部对象不得借 Scheduler 入口改变状态或污染 waiting 队列。"""
        sched = make_scheduler()
        foreign = make_seq()
        foreign.transition_to(RUNNING, now=2.0)
        with pytest.raises(ValueError):
            sched.preempt(foreign, now=2.0)
        assert foreign.status == RUNNING and foreign not in sched.waiting
        foreign.status = PREEMPTED
        with pytest.raises(ValueError):
            sched.resume(foreign, now=3.0)
        assert foreign not in sched.waiting

    def test_resume_rejects_cross_queue_duplicate(self):
        """已注册暂停对象若残留于 running，resume 必须拒绝双队列。"""
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        sched.preempt(seq, now=5.0)
        sched.running.append(seq)  # 注入损坏状态
        with pytest.raises(ValueError, match="重复成员"):
            sched.resume(seq, now=6.0)
        assert seq in sched.running and seq not in sched.waiting


# ============================== 恢复进度（§9.1 行 6） ==============================

class TestRecomputeRecovery:
    """抢占 -> 恢复 -> 重新 prefill：逻辑进度保留、completion 不重复。"""

    def test_full_recovery_keeps_progress_and_continues(self):
        sched = make_scheduler(num_blocks=8)
        seq, = setup_decode_source(sched, num_tokens=8, max_tokens=6)
        # decode 2 步后抢占
        drive_round(sched, 3.0)
        drive_round(sched, 4.0)
        assert seq.num_completion_tokens == 3   # prefill 采样 1 + decode 2
        prompt = list(seq.prompt_token_ids)
        comp = seq.num_completion_tokens
        tail = seq.token_ids[-comp:]
        sched.preempt(seq, now=5.0)
        sched.resume(seq)
        assert seq.status == WAITING and seq.block_table == []
        assert seq.prefill_offset == 0 and seq.prefill_target == seq.num_tokens
        # 恢复轮：混合 prefill 重算（前缀命中自身缓存），重新 prefill 不追加 token
        items, phase = sched.schedule(now=6.0)
        it = next(it for it in items if it.seq is seq)
        assert it.phase == "prefill" and it.is_last_chunk
        assert it.scheduled_tokens == seq.num_tokens - seq.prefill_offset
        sched.postprocess(items, tokens_for(items), now=6.0)
        assert seq.status == RUNNING
        assert seq.num_completion_tokens == comp + 1      # 恰好多 1 个新 token
        assert seq.prompt_token_ids == prompt
        assert seq.completion_token_ids[:comp] == tail    # 历史进度不丢不重
        sched.block_manager.check_ledger()

    def test_repeated_preempt_resume_cycles_are_safe(self):
        """多次抢占-恢复循环：计数正确、账本守恒、无重复追加。"""
        sched = make_scheduler(num_blocks=8)
        seq, = setup_decode_source(sched, num_tokens=8, max_tokens=10)
        for i in range(3):
            drive_round(sched, 10.0 + i)
            sched.preempt(seq, now=20.0 + i)
            sched.resume(seq)
        assert seq.num_preempts == 3
        comp = seq.num_completion_tokens
        assert len(seq.completion_token_ids) == comp
        sched.block_manager.check_ledger()


# ============================== victim 选择（§9.1 行 7） ==============================

class TestVictimSelection:
    """队尾优先、排除规则、有限尝试、无 victim 延后、prefill 侧抢占重查。"""

    def test_victim_is_running_tail(self):
        """decode 候选块不足：默认抢占 running 队尾（FCFS 最后被服务者）。

        块数推演（pool=6，三条 seq 各 8 token prompt）：setup 后各 len=9、
        各持 1 块、free=3。t=10 轮三条 decode 各需 +1 块（9%8==1）-> free=0；
        t=11..16 轮 len 11..16 不跨块边界。t=17 轮三条 len=17 又各需 +1 块：
        pop a 后无块 -> 抢占队尾 c（释放 2 块）-> a、b 依次接纳，c 回 waiting 队首。
        """
        sched = make_scheduler(num_blocks=6)
        a, b, c = setup_decode_source(sched, count=3, num_tokens=8, max_tokens=50)
        items = None
        for i in range(8):
            items, _ = drive_round(sched, 10.0 + i)
        assert c.num_preempts == 1 and c.status == WAITING
        assert sched.waiting[0] is c
        assert b.num_preempts == 0 and a.num_preempts == 0
        admitted = [it.seq.seq_id for it in items if it.phase == "decode"]
        assert a.seq_id in admitted and b.seq_id in admitted
        assert c.seq_id not in admitted
        sched.block_manager.check_ledger()

    def test_no_victim_defers_candidate(self):
        """无合法 victim：候选延后（保留 RUNNING/KV），不抢占、不破坏队列。"""
        sched = make_scheduler(num_blocks=2)
        seq, = setup_decode_source(sched, num_tokens=16, max_tokens=50)  # 2 块占满
        assert seq.block_table and len(sched.block_manager.free_block_ids) == 0
        # decode 需要 +1 块：free=0 且无其他 running -> 延后
        items, phase = sched.schedule(now=10.0)
        assert items == [] and phase == "idle"
        assert seq.status == RUNNING and seq.block_table
        assert seq.num_preempts == 0
        d = decision_of(sched.last_schedule_stats, seq)
        assert d["reason"] == "kv_capacity" and d["kv_checked"] is True
        sched.block_manager.check_ledger()

    def test_prefill_side_preempt_then_requery_succeeds(self):
        """§5.3 时序：waiting 队首 d 需新块 -> 抢占合法 victim（队尾方向）->
        d 重新查询容量并继续调度；victim 恢复 waiting 队首优先重算。

        构造（pool=6）：a/b（8-token，各 2 块）、c（24-token，3 块，其中
        首块与 a 的首块 prefix 共享、独占 2 块）均已 RUNNING，free=0。触发
        轮 decode a、b 不需新块（正常接纳），c 需要新块且 free=0、running
        中已无合法 victim -> c 被 decode 侧延后（保留 3 块）；随后 prefill
        d（20-token）can_allocate 失败 -> 反向搜索跳过本轮已接纳的 a/b、
        抢占 c（其独占 2 块回到 free）-> d 重新查询成功并被接纳。
        """
        sched = make_scheduler(num_blocks=6)
        a, b = setup_decode_source(sched, count=2, num_tokens=8, max_tokens=50)
        c, = setup_decode_source(sched, count=1, num_tokens=24, max_tokens=50)
        assert (len(a.block_table), len(b.block_table), len(c.block_table)) == (2, 2, 3)
        assert len(sched.block_manager.free_block_ids) == 0
        d = make_seq(20, max_tokens=4)       # 3 块
        sched.add(d)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        # decode 侧：a、b 接纳；c 需新块但无合法 victim -> 延后（kv_capacity）
        assert all(it.seq is not c for it in items)
        # prefill 侧：d 的 KV 检查失败 -> 已接纳的 a/b 被排除，c 被抢占让块
        assert a.num_preempts == 0 and b.num_preempts == 0
        assert c.num_preempts == 1 and c.status == WAITING
        assert sched.waiting[0] is c         # victim 恢复到 waiting 队首优先重算
        assert len(d.block_table) == 3       # 抢占让块后分配成功
        it_d = next(it for it in items if it.seq is d)
        assert it_d.phase == "prefill" and it_d.is_last_chunk
        sched.block_manager.check_ledger()

    def test_prefill_preempt_exhausted_stops_without_breaking_queue(self):
        """抢占全部合法 victim 后仍不足：有限尝试后按 kv_capacity 停止，
        被抢占者已恢复 waiting（合法状态），队列不被破坏。"""
        sched = make_scheduler(num_blocks=4)
        a, b = setup_decode_source(sched, count=2, num_tokens=8, max_tokens=50)
        c = make_seq(40, max_tokens=4)       # 5 块 > 池总量，怎么释放都不够
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        # a/b 至少其一被抢占尝试让块；尝试有界（测试正常结束即为证）
        preempted = [s for s in (a, b) if s.num_preempts == 1]
        assert all(s.status == WAITING for s in preempted)
        assert c.status == WAITING and not c.block_table
        dd = decision_of(sched.last_schedule_stats, c)
        assert dd["reason"] == "kv_capacity"
        assert c in sched.waiting            # 队列完整
        sched.block_manager.check_ledger()


# ============================== KV 账本（§9.1 行 8） ==============================

class TestLedgerConservation:
    """控制动作重复组合下的账本守恒（check_ledger 交叉核对）。"""

    def test_control_ops_keep_ledger_consistent(self):
        rng = __import__("random").Random(20260912)
        sched = make_scheduler(num_blocks=16, max_num_batched_tokens=12,
                               chunk_size=8)
        id_pool = [f"ctl-{i}" for i in range(5)]
        now = 1.0
        for round_no in range(120):
            now += 1.0
            op = rng.random()
            if op < 0.30:
                seq = make_seq(rng.randint(1, 12), max_tokens=rng.randint(1, 4),
                               request_id=rng.choice(id_pool))
                try:
                    sched.add(seq)
                except ValueError:
                    pass                      # 活动 ID 冲突：预期拒绝
            elif op < 0.42 and sched.requests:
                sched.cancel(rng.choice(list(sched.requests)), now=now)
            elif op < 0.50 and sched.requests:
                sched.timeout(rng.choice(list(sched.requests)), now=now)
            elif op < 0.58 and sched.running:
                victim = rng.choice(list(sched.running))
                if victim.status == RUNNING:
                    sched.preempt(victim, now=now)
            elif op < 0.64 and sched.requests:
                paused = [s for s in sched.requests.values()
                          if s.status == PREEMPTED]
                if paused:
                    sched.resume(rng.choice(paused))
            elif op < 0.70 and sched.running:
                rng.choice(list(sched.running)).mark_finished("stop", now=now)
            drive_round(sched, now, token=rng.randint(1, 100))
            # 每个控制动作后账本必须守恒（互斥/总数/引用计数）
            sched.block_manager.check_ledger()
        # 收尾：全部终态后资源完全释放
        for sid in list(sched.requests):
            sched.cancel(sid, now=now)
        assert sched.requests == {} and not sched.waiting and not sched.running
        assert sched.block_manager.used_block_ids == set()
        assert len(sched.block_manager.free_block_ids) == 16
        sched.block_manager.check_ledger()

    def test_finalize_repeat_no_double_free(self):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        sched.cancel(seq.seq_id, now=3.0)
        free_after = list(sched.block_manager.free_block_ids)
        for _ in range(3):
            sched.cancel(seq.seq_id, now=4.0)     # 重复取消：幂等
            sched._finalize(seq)                   # 重复收尾：幂等
        assert list(sched.block_manager.free_block_ids) == free_after
        assert seq.block_table == []
        sched.block_manager.check_ledger()

    def test_ledger_rejects_duplicate_or_invalid_free_ids(self):
        """账本检查必须检测 deque 重复和越界 ID，而不只是集合关系。"""
        sched = make_scheduler(num_blocks=2)
        sched.block_manager.free_block_ids.append(0)
        with pytest.raises(ValueError, match="重复"):
            sched.block_manager.check_ledger()
        sched.block_manager.free_block_ids.pop()
        sched.block_manager.free_block_ids.append(99)
        with pytest.raises(ValueError, match="越界"):
            sched.block_manager.check_ledger()

    def test_deallocate_rejects_corrupt_table_without_mutation(self):
        """非法/重复 block_table 在任何 ref_count 修改前被拒绝。"""
        bm = BlockManager(2, BLOCK_SIZE)
        seq = make_seq(4)
        seq.block_table = [0, 0]
        before = (list(bm.free_block_ids), set(bm.used_block_ids),
                  [b.ref_count for b in bm.blocks])
        with pytest.raises(ValueError, match="重复"):
            bm.deallocate(seq)
        assert list(bm.free_block_ids) == before[0]
        assert bm.used_block_ids == before[1]
        assert [b.ref_count for b in bm.blocks] == before[2]
        seq.block_table = [99]
        with pytest.raises(ValueError, match="非法"):
            bm.deallocate(seq)
        assert list(bm.free_block_ids) == before[0]
        assert bm.used_block_ids == before[1]
        assert [b.ref_count for b in bm.blocks] == before[2]

    def test_allocate_rejects_corrupt_free_head_without_mutation(self):
        """free 队首损坏时分配前拒绝，不能丢失 deque 条目。"""
        bm = BlockManager(2, BLOCK_SIZE)
        bm.free_block_ids[0] = 99
        seq = make_seq(4)
        before = list(bm.free_block_ids)
        with pytest.raises(ValueError, match="账本破坏"):
            bm._allocate_block()
        assert list(bm.free_block_ids) == before

    def test_allocate_exception_rolls_back_everything(self, monkeypatch):
        """分配过程中出现异常时，KV 账本和请求 table 恢复到调用前。"""
        bm = BlockManager(3, BLOCK_SIZE)
        seq = make_seq(16)
        original = bm._allocate_block
        calls = {"n": 0}
        def fail_second():
            calls["n"] += 1
            block = original()
            if calls["n"] == 2:
                raise RuntimeError("injected allocation failure")
            return block
        monkeypatch.setattr(bm, "_allocate_block", fail_second)
        with pytest.raises(RuntimeError, match="allocation failure"):
            bm.allocate(seq, 0)
        assert seq.block_table == [] and seq.prefill_offset == 0
        bm.check_ledger()

    def test_non_owner_finalize_does_not_touch_scheduler(self):
        """旧对象收尾不能删除当前 Scheduler 的新对象或其账本。"""
        sched = make_scheduler()
        old = make_seq(request_id="reuse")
        sched.add(old)
        sched.cancel(old.seq_id, now=2.0)
        new = make_seq(request_id="reuse")
        sched.add(new)
        waiting_before = list(sched.waiting)
        sched._finalize(old, now=3.0)
        assert list(sched.waiting) == waiting_before
        assert sched.requests.get(new.seq_id) is new


# ============================== 异常清理（§9.1 行 9） ==============================

class TestExceptionCleanup:
    """Engine step 事务：runner/postprocess 异常 -> 批次回滚 + 全活动收尾。"""

    def _three_live(self, sched):
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        return a, b, c

    def test_runner_exception_aborts_everything(self, caplog):
        sched = make_scheduler()
        a, b, c = self._three_live(sched)
        used_before = len(sched.block_manager.used_block_ids)
        engine = make_engine(sched, runner=lambda m, items_: (_ for _ in ()).throw(
            RuntimeError("cuda oom")))
        with caplog.at_level(logging.INFO):
            with pytest.raises(RuntimeError):
                engine.step()
        # 全部活动请求以 CANCELLED/engine_error 收尾
        for seq in (a, b, c):
            assert seq.status == CANCELLED and seq.finish_reason == "engine_error"
            assert seq.block_table == []
        assert sched.requests == {} and not sched.waiting and not sched.running
        assert sched.block_manager.used_block_ids == set()
        assert len(sched.block_manager.free_block_ids) == sched.block_manager.blocks.__len__()
        # Engine 锁定：后续 step 立即拒绝
        with pytest.raises(RuntimeError, match="禁止重试"):
            engine.step()
        # 事件：error 轮 + 清理摘要
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        assert any(e["event"] == "engine_round" and e["outcome"] == "error"
                   and e["executed_tokens"] is None for e in events)
        summary = [e for e in events if e["event"] == "engine_abort_summary"]
        assert summary and summary[-1]["final_snapshot"]["active_requests"] == 0

    def test_postprocess_exception_aborts_everything(self):
        """token 数不匹配使 postprocess 在提交前抛错：同样触发全活动收尾。"""
        sched = make_scheduler()
        a, b, c = self._three_live(sched)
        engine = make_engine(sched)
        used_blocks = len(sched.block_manager.used_block_ids)
        with pytest.raises(ValueError):
            # 注入错误长度的采样列表 -> postprocess 前置校验抛 ValueError
            engine.model_runner = SimpleNamespace(call=lambda m, items_: [7, 7])
            engine.step()
        for seq in (a, b, c):
            assert seq.status == CANCELLED and seq.finish_reason == "engine_error"
        assert sched.requests == {}
        assert len(sched.block_manager.used_block_ids) < used_blocks or used_blocks == 0
        sched.block_manager.check_ledger()

    def test_abort_entries_idempotent(self):
        """abort_round/abort_all_active 可重复调用：不 double free、不重复删索引。"""
        sched = make_scheduler()
        a, b, c = self._three_live(sched)
        items, _ = sched.schedule(now=10.0)
        snap1 = sched.abort_round(items, reason="engine_error")
        snap2 = sched.abort_round(items, reason="engine_error")
        snap3 = sched.abort_all_active(reason="engine_error")
        snap4 = sched.abort_all_active(reason="engine_error")
        assert snap1["active_requests"] == 0
        assert snap2 == snap1 and snap3 == snap2 and snap4 == snap3
        assert sched.block_manager.used_block_ids == set()
        sched.block_manager.check_ledger()

    def test_abort_respects_ownership_on_id_reuse(self):
        """旧批次迟到 abort 不能误删复用了同一 request_id 的新请求。"""
        sched = make_scheduler()
        old = make_seq(4, request_id="dup")
        sched.add(old)
        items, _ = sched.schedule(now=2.0)
        sched.cancel(old.seq_id, now=3.0)    # 旧请求终态、ID 释放
        new = make_seq(4, request_id="dup")  # 复用 ID
        sched.add(new)
        sched.abort_round(items, reason="engine_error")   # 迟到收尾旧批次
        assert new.status == WAITING and sched.requests.get(new.seq_id) is new
        assert "dup" in sched._active_request_ids
        sched.block_manager.check_ledger()

    def test_abort_all_active_continues_after_one_cleanup_failure(self, monkeypatch):
        """单对象清理失败不能阻断其余活动请求的最终收尾。"""
        sched = make_scheduler()
        a, b, c = self._three_live(sched)
        original = sched._abort_one
        failed = {a.seq_id}

        def flaky(seq, reason, now):
            if seq.seq_id in failed:
                failed.remove(seq.seq_id)
                raise RuntimeError("injected cleanup failure")
            return original(seq, reason, now)

        monkeypatch.setattr(sched, "_abort_one", flaky)
        with pytest.raises(RuntimeError, match="异常收尾失败"):
            sched.abort_all_active(reason="engine_error", now=20.0)
        # 失败对象可被后续幂等调用补清，其余对象已经不应残留。
        sched.abort_all_active(reason="engine_error", now=21.0)
        assert not sched.requests and not sched.waiting and not sched.running
        sched.block_manager.check_ledger()

    def test_replay_rejects_broken_history_and_ledger(self):
        """事件审计器必须发现跨事件断裂、身份改写和错误资源总量。"""
        base = {
            "event": "request_control", "round_id": None, "seq_id": 1,
            "request_id": "r", "action": "cancel", "from_status": "RUNNING",
            "to_status": "CANCELLED", "reason": "x", "num_preempts": 0,
            "released_blocks": 1, "free_blocks_before": 1, "used_blocks_before": 3,
            "free_blocks": 2, "used_blocks": 1,
            "observed_at": 1.0,
        }
        events = [dict(base), dict(base, action="timeout", from_status="RUNNING",
                                   to_status="TIMEOUT", request_id="changed")]
        problems = replay_control_transitions(events, total_blocks=4)
        assert any("历史断裂" in p for p in problems)
        assert any("request_id 被改写" in p for p in problems)
        assert any("free+used" in p for p in problems)

    def test_replay_accepts_unlogged_internal_transition_baseline(self):
        """控制事件的首个 from_status 可作为基线，后续控制必须连续。"""
        events = [{
            "event": "request_control", "round_id": 2, "seq_id": 9,
            "request_id": "r9", "action": "preempt", "from_status": "RUNNING",
            "to_status": "PREEMPTED", "reason": "kv_capacity", "num_preempts": 1,
            "released_blocks": 1, "free_blocks_before": 2, "used_blocks_before": 2,
            "free_blocks": 3, "used_blocks": 1,
            "observed_at": 2.0,
        }, {
            "event": "request_control", "round_id": 2, "seq_id": 9,
            "request_id": "r9", "action": "resume", "from_status": "PREEMPTED",
            "to_status": "WAITING", "reason": "recompute_resume", "num_preempts": 1,
            "released_blocks": 0, "free_blocks_before": 3, "used_blocks_before": 1,
            "free_blocks": 3, "used_blocks": 1,
            "observed_at": 2.0,
        }]
        assert replay_control_transitions(events, total_blocks=4) == []

    def test_model_runner_resets_context_after_subbatch_failure(self, monkeypatch):
        """decode/prefill 子批异常时均必须执行 reset_context。"""
        runner = model_runner_module.ModelRunner.__new__(model_runner_module.ModelRunner)
        runner.rank = 0
        runner.prepare_decode = lambda seqs: (None, None)
        runner.prepare_prefill = lambda seqs: (None, None)
        runner.run_model = lambda *args: (_ for _ in ()).throw(RuntimeError("forward"))
        resets = []
        monkeypatch.setattr(model_runner_module, "reset_context", lambda: resets.append(True))
        item = SimpleNamespace(phase="decode", seq=make_seq())
        with pytest.raises(RuntimeError):
            runner.run([item])
        assert resets == [True]

        resets.clear()
        item = SimpleNamespace(phase="prefill", seq=make_seq())
        with pytest.raises(RuntimeError):
            runner.run([item])
        assert resets == [True]

    def test_schedule_exception_aborts_active_requests(self, monkeypatch, caplog):
        """schedule 半分配后异常时 Engine 也必须进入失败态并清账。"""
        sched = make_scheduler()
        seq = make_seq(4)
        sched.add(seq)
        original = sched.block_manager.allocate

        def allocate_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected allocate failure")

        monkeypatch.setattr(sched.block_manager, "allocate", allocate_then_fail)
        engine = make_engine(sched)
        with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="allocate failure"):
            engine.step()
        assert engine._execution_failed is True
        assert not sched.requests and not sched.waiting and not sched.running
        assert not sched.block_manager.used_block_ids
        sched.block_manager.check_ledger()
        events = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
        assert any(e.get("event") == "engine_round" and e.get("outcome") == "error"
                   and e.get("executed_tokens") is None for e in events)

    def test_may_append_exception_aborts_active_requests(self, monkeypatch):
        """decode 追加 block 后抛错时，Engine 仍清理被移出 running 的请求。"""
        sched = make_scheduler(num_blocks=8)
        seq, = setup_decode_source(sched, num_tokens=8, max_tokens=20)
        original = sched.block_manager.may_append

        def append_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected append failure")

        monkeypatch.setattr(sched.block_manager, "may_append", append_then_fail)
        engine = make_engine(sched)
        with pytest.raises(RuntimeError, match="append failure"):
            engine.step()
        assert engine._execution_failed is True
        assert not sched.requests and not sched.waiting and not sched.running
        assert not sched.block_manager.used_block_ids
        sched.block_manager.check_ledger()

    def test_exit_failure_still_releases_runner_and_workers(self):
        """runner exit 异常不能跳过活动请求清理、引用释放和 worker join。"""
        sched = make_scheduler()
        seq = make_seq()
        sched.add(seq)
        joined = []
        class Worker:
            def join(self):
                joined.append(True)
        class Runner:
            def call(self, method):
                assert method == "exit"
                raise RuntimeError("injected exit failure")
        engine = make_engine(sched, runner=Runner().call)
        engine.model_runner = Runner()
        engine.ps = [Worker()]
        with pytest.raises(RuntimeError, match="exit failure"):
            engine.exit()
        assert getattr(engine, "model_runner", None) is None
        assert joined == [True]
        assert not sched.requests and not sched.block_manager.used_block_ids


# ============================== ID 复用与迟到批次（§9.1 行 10） ==============================

class TestIdentityAndLateBatch:
    def test_request_id_reuse_after_terminal(self):
        sched = make_scheduler()
        s1 = make_seq(4, request_id="req-dup")
        sched.add(s1)
        sched.cancel(s1.seq_id, now=2.0)
        s2 = make_seq(4, request_id="req-dup")
        sched.add(s2)                         # 终态后 ID 可复用
        assert sched.requests.get(s2.seq_id) is s2
        # 按 ID 取消作用于新对象
        assert sched.cancel(s2.seq_id, "again", now=3.0) is True
        assert s2.status == CANCELLED and s1.status == CANCELLED

    def test_late_postprocess_rejected_by_round_id(self):
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched)
        c = make_seq(4, max_tokens=4)
        sched.add(c)
        old_items, _ = sched.schedule(now=10.0)
        sched.postprocess(old_items, tokens_for(old_items), now=10.0)
        new_items, _ = sched.schedule(now=11.0)   # 新轮次：同 seq 重新规划
        with pytest.raises(ValueError, match="round_id"):
            sched.postprocess(old_items, tokens_for(old_items), now=11.0)
        # 新批次正常提交
        sched.postprocess(new_items, tokens_for(new_items), now=11.0)


# ============================== 序列化（§9.1 行 11） ==============================

class TestSerialization:
    """v1/v2/v3 读取兼容、当前版本往返、控制字段不静默丢失。"""

    def test_current_version_roundtrip_keeps_control_fields(self):
        seq = make_seq(6, max_tokens=5, request_id="tp-1")
        seq.transition_to(RUNNING, now=2.0)
        seq.is_prefill = False               # decode 模式（调度接纳时由 Scheduler 置位）
        seq.request_cancel("client_gone")
        seq.prefill_offset = 3
        seq.num_scheduled_tokens = 0
        seq.block_table = [4, 5]
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.status == RUNNING
        assert clone.cancel_requested is True
        assert clone.is_prefill is False
        assert clone.prefill_offset == 3
        assert clone.num_tokens == 6 and clone.num_prompt_tokens == 6
        assert clone.last_token == seq.last_token

    def test_v3_payload_carries_control_plane(self):
        seq = make_seq(4)
        seq.request_cancel("why")
        version, payload = seq.__getstate__()
        assert version == Sequence.STATE_VERSION == 3
        assert payload["status"] == WAITING
        assert payload["cancel_requested"] is True
        assert payload["prefill_offset"] == 0
        assert payload["is_prefill"] is True

    def test_v2_payload_read_maps_legacy_offset_field(self):
        seq = Sequence.__new__(Sequence)
        v2_payload = {
            "num_tokens": 5, "num_prompt_tokens": 4,
            "num_cached_tokens": 2,           # v2 字段名（Day8 语义同 prefill_offset）
            "num_scheduled_tokens": 1, "block_table": [1],
            "last_state": 9,                  # 标量 -> decode 模式
            "status": RUNNING, "is_prefill": False, "cancel_requested": True,
        }
        seq.__setstate__((2, v2_payload))
        assert seq.prefill_offset == 2        # 单向映射
        assert seq.status == RUNNING and seq.cancel_requested is True
        assert seq.is_prefill is False and seq.last_token == 9
        # 控制字段缺省兜底：不因缺失而异常
        v2_payload.pop("cancel_requested")
        seq2 = Sequence.__new__(Sequence)
        seq2.__setstate__((2, dict(v2_payload)))
        assert seq2.cancel_requested is False

    def test_v1_legacy_tuple_read(self):
        seq = Sequence.__new__(Sequence)
        legacy = (5, 4, 2, 1, [1], [1, 2, 3, 4, 5])   # prefill 模式（last_state 为列表）
        seq.__setstate__(legacy)
        assert seq.prefill_offset == 2
        assert seq.is_prefill is True
        assert seq.status == WAITING and seq.cancel_requested is False
        assert seq.token_ids == [1, 2, 3, 4, 5]
        # 反序列化后的对象可再次 pickle 往返（不退化为空 payload）
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.is_prefill and clone.token_ids == [1, 2, 3, 4, 5]

    def test_v1_decode_mode_read(self):
        seq = Sequence.__new__(Sequence)
        seq.__setstate__((5, 4, 2, 1, [1], 9))        # last_state 标量 -> decode
        assert seq.is_prefill is False and seq.last_token == 9
        assert seq.token_ids == []

    def test_unknown_version_rejected(self):
        seq = make_seq(4)
        _, payload = seq.__getstate__()
        with pytest.raises(ValueError, match="版本"):
            Sequence.__new__(Sequence).__setstate__((999, payload))


# ============================== 无进展契约（§9.1 行 12） ==============================

class TestNoProgressContract:
    """同步 generate() 遇到无法推进的状态：显式报错，不忙循环、不伪装完成。"""

    def test_generate_raises_when_all_paused(self):
        sched = make_scheduler(num_blocks=8)
        engine = make_engine(sched)
        rid = engine.add_request([1, 2, 3], SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()                         # prefill -> RUNNING
        seq = engine.get_request(rid)
        sched.preempt(seq, now=2.0)           # 显式暂停，无外部 resume
        with pytest.raises(RuntimeError, match="PREEMPTED"):
            engine.generate([[4, 5]], SamplingParams(max_tokens=1, ignore_eos=True),
                            use_tqdm=False)
        assert seq.status == PREEMPTED and not engine.is_finished()

    def test_generate_raises_when_kv_blocked(self):
        """KV 无法接纳（decode 延后 + prefill 抢占后仍不足）：idle 轮显式报错。"""
        sched = make_scheduler(num_blocks=2)
        engine = make_engine(sched)
        rid = engine.add_request(list(range(1, 17)),
                                 SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()                         # a 占满 2 块（16 token）-> len 17
        a = engine.get_request(rid)
        b_rid = engine.add_request(list(range(1, 21)),
                                   SamplingParams(max_tokens=1, ignore_eos=True))
        with pytest.raises(RuntimeError, match="KV 容量"):
            engine.generate([], SamplingParams(max_tokens=1, ignore_eos=True),
                            use_tqdm=False)
        # a 未被伪装完成；轮次无进展但资源账本一致
        assert a.num_completion_tokens == 1
        sched.block_manager.check_ledger()
        assert b_rid  # b 已登记（可能已被抢占让块后仍不足）

    def test_idle_step_returns_without_model_call(self):
        """idle 轮：step 返回空且不调用 ModelRunner（decode 组装会崩溃）。"""
        sched = make_scheduler(num_blocks=2)
        engine = make_engine(sched)
        called = []

        def spy_runner(method, items_):
            # 按 Day9 提交契约返回采样；同时记录调用次数
            called.append(items_)
            out = tokens_for(items_, 7)
            return out if out else None

        engine.model_runner = SimpleNamespace(call=spy_runner)
        rid = engine.add_request(list(range(1, 17)),
                                 SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()
        engine.step()                         # decode 需新块、free=0 -> idle
        assert called and len(called) == 1    # 第二轮未调用 runner
        assert engine.step() == ([], 0)


# ============================== Engine 取消入口（§4.3） ==============================

class TestEngineCancelRequest:
    """Engine.cancel_request：空闲立即收尾；step 期间只置位信号。"""

    def test_cancel_request_idle_finalizes_immediately(self):
        sched = make_scheduler()
        engine = make_engine(sched)
        rid = engine.add_request([1, 2, 3], SamplingParams(max_tokens=4, ignore_eos=True))
        assert engine.cancel_request(rid, "client_gone") is True
        seq = sched.requests.get(1)
        assert seq is None                    # 已从活动索引移除（立即收尾）
        # 通过重新注册查询：终态对象已不在活动索引，get_request 返回 None
        assert engine.get_request(rid) is None
        assert engine.cancel_request(rid) is False
        sched.block_manager.check_ledger()

    def test_cancel_request_during_step_only_signals(self):
        """forward 期间取消：只置位信号；postprocess 安全点完成收尾。"""
        sched = make_scheduler()
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(encode=lambda t: [1])
        observed = {}

        def runner(method, items_):
            # 模拟另一线程在 forward 期间发起取消：信号只置位
            for it in items_:
                if it.seq.request_id == "victim":
                    observed["during_forward"] = engine.cancel_request(
                        "victim", "mid_forward")
            return tokens_for(items_, 7)

        engine.model_runner = SimpleNamespace(call=runner)
        rid = engine.add_request([1, 2, 3], SamplingParams(max_tokens=4, ignore_eos=True),
                                 request_id="victim")
        engine.step()
        assert observed["during_forward"] is True
        seq = None
        assert engine.get_request("victim") is None   # postprocess 已收尾
        # 终态确认：重新加入同 ID 前先检查收尾事件的最终状态
        # （对象已从索引移除，直接断言收尾后的资源与返回值）
        assert sched.block_manager.check_ledger() is None
        assert engine.cancel_request("victim") is False

    def test_cancel_request_unknown_false(self):
        engine = make_engine(make_scheduler())
        assert engine.cancel_request("nope") is False


# ============================== 控制事件契约（§6.3） ==============================

class TestControlEvents:
    """request_control 事件：字段白名单、released_blocks 为账本差值、无明文。"""

    ACTION_WHITELIST = {"cancel", "timeout", "preempt", "resume", "abort"}
    STATUS_WHITELIST = {"WAITING", "RUNNING", "PREEMPTED",
                        "FINISHED", "CANCELLED", "TIMEOUT"}

    def _collect(self, caplog):
        return [json.loads(r.message) for r in caplog.records
                if r.message.startswith("{")]

    def test_control_events_fully_checkable(self, caplog):
        sched = make_scheduler()
        seq = make_seq(8, max_tokens=50, request_id="ev-1")
        sched.add(seq)
        drive_round(sched, 2.0)              # -> RUNNING
        with caplog.at_level(logging.INFO):
            sched.preempt(seq, now=10.0)      # action=preempt
            sched.resume(seq)                 # action=resume
            sched.request_cancel(seq.seq_id, "ev")
            sched.cancel(seq.seq_id, "ev", now=11.0)   # action=cancel
            seq2 = make_seq(4, deadline=12.0)
            sched.add(seq2)
            sched.check_deadlines(now=13.0)   # action=timeout
        events = [e for e in self._collect(caplog) if e["event"] == "request_control"]
        actions = [e["action"] for e in events]
        for action in ("preempt", "resume", "cancel", "timeout"):
            assert action in actions
        for e in events:
            assert set(e) == {"event", "round_id", "seq_id", "request_id",
                              "action", "from_status", "to_status", "reason",
                              "num_preempts", "released_blocks", "free_blocks_before",
                              "used_blocks_before", "free_blocks", "used_blocks",
                              "observed_at"}
            assert e["action"] in self.ACTION_WHITELIST
            assert e["from_status"] in self.STATUS_WHITELIST
            assert e["to_status"] in self.STATUS_WHITELIST
            assert e["free_blocks"] + e["used_blocks"] == len(sched.block_manager.blocks)
            assert isinstance(e["released_blocks"], int) and e["released_blocks"] >= 0
            assert "token_ids" not in e and "prompt" not in e
        # 抢占/恢复按同一 seq_id 配对，且抢占释放量 > 0
        preempt = next(e for e in events if e["action"] == "preempt")
        resume = next(e for e in events if e["action"] == "resume")
        cancel = next(e for e in events if e["action"] == "cancel")
        assert preempt["seq_id"] == resume["seq_id"] == cancel["seq_id"]
        assert preempt["to_status"] == "PREEMPTED"
        assert resume["from_status"] == "PREEMPTED" and resume["to_status"] == "WAITING"
        assert resume["num_preempts"] == preempt["num_preempts"] == 1
        assert cancel["to_status"] == "CANCELLED"
        assert preempt["released_blocks"] > 0
        # 超时事件
        timeout = next(e for e in events if e["action"] == "timeout")
        assert timeout["to_status"] == "TIMEOUT"

    def test_cancel_event_released_matches_ledger(self, caplog):
        sched = make_scheduler()
        seq, = setup_decode_source(sched)
        held = len(seq.block_table)
        with caplog.at_level(logging.INFO):
            sched.cancel(seq.seq_id, now=5.0)
        cancel = next(e for e in self._collect(caplog)
                      if e["event"] == "request_control" and e["action"] == "cancel")
        assert cancel["released_blocks"] == held
        assert cancel["reason"] == "client_cancelled"
