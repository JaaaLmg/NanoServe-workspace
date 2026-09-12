"""Day 9 验收测试：混合 Prefill 与 Decode（docs/mixed-prefill-decode.md §9.1 测试矩阵）。

覆盖范围（对应文档 §9.1 表格逐行）：
- 混合表示：items 同时含 decode+prefill item；phase 四值（prefill/decode/mixed/idle）；
  decode 在前 prefill 在后；同 seq 唯一 item；最后 chunk 接纳的请求无同轮 decode item；
- 混合预算：sum(prefill q)+D <= B 逐轮断言；D 先占预算；四条上限独立；
  decode item 恒为 1；分阶段统计量（prefill_tokens/decode_tokens）可重算；
- decode 保底与归因：decode_priority 三分支（D>0 挤占 / needed>B 仍 budget / D==0）、
  不计 episode、HOL 跟随、phase_priority 无残留；
- chunk 行为：长 prompt 与 decode 同轮逐块推进、区间连续、首候选拆分按 remaining、
  扫描位置与队列分离在混合轮不回归；
- 采样对齐：中间 chunk 不采样不耗 RNG；decode 与最后 chunk 同轮各 1 token；
  token 数不匹配显式抛错；注入助手与 needs_sample 共享谓词；
- 执行契约：runner spy 核对两个子批成员/顺序/展平输入长度 == 计划
  （runner_input_len 对账的 CPU 侧；GPU 侧由脚本 runner 包装独立观测）；
- KV 与资源：decode may_append 与 prefill allocate 同轮不超卖；同轮完成释放幂等；
  KV 不足时 decode 抢占与 prefill 停止归因独立；
- 生命周期：混合轮中单 item cancel/timeout/完成不影响其他 item；抢占恢复；
- 迟到与重复：round_id 关联校验（同 seq 重新规划后旧批次拒绝）、全终态重放安全；
- 日志证据：事件字段白名单、分阶段计数可重算、无 prompt/token 明文；
- 随机混合：固定 seed 交错操作 + 轮次上限 + 终态全平衡。

无 GPU / 模型依赖：Scheduler/BlockManager/Sequence 为纯 CPU 逻辑；LLMEngine 通过
__new__ 跳过 GPU __init__，runner 为契约桩；时间使用注入的确定值（now 关键字），
不加载权重、不 sleep、不访问 CUDA。真实 Sampler RNG 顺序（decode 子批先采样）
属 GPU 执行路径，由 GPU 验收覆盖，此处以桩顺序契约钉死合并顺序。

重复执行命令：
    python -m pytest tests/test_mixed_batch.py -q
    python -O -m pytest tests/test_mixed_batch.py -q
"""

import json
import logging
import random
from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import BatchItem, Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
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


# ============================== 公共构造工具 ==============================

def make_seq(num_tokens: int = 4, max_tokens: int = 8, request_id: str | None = None,
             deadline: float | None = None) -> Sequence:
    return Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
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


def item_view(items):
    """批次结构摘要：(phase, seq_id, scheduled_tokens) 序列。"""
    return [(it.phase, it.seq.seq_id, it.scheduled_tokens) for it in items]


def decision_of(stats: dict, seq: Sequence) -> dict:
    for d in stats["decisions"]:
        if d["seq_id"] == seq.seq_id:
            return d
    raise AssertionError(f"seq_id={seq.seq_id} 不在本轮决策中: {stats['decisions']}")


def assert_mixed_round_invariants(sched: Scheduler, items, phase: str):
    """混合轮硬约束（§5.2 不变量 14-17 的逐轮断言）。

    - planned == sum(prefill q) + decode 条数，且 <= B；
    - len(items) <= max_num_seqs；seq_id 不重复；
    - decode item 恒 q==1、prefill item 0<q<=chunk_size；
    - decode item 在前、prefill item 在后；分阶段统计量可重算；
    - phase 标签与 items 构成一致。
    """
    stats = sched.last_schedule_stats
    prefill = [it for it in items if it.phase == "prefill"]
    decode = [it for it in items if it.phase == "decode"]
    assert stats["phase"] == phase
    assert stats["prefill_items"] == len(prefill)
    assert stats["decode_items"] == len(decode)
    assert stats["prefill_tokens"] == sum(it.scheduled_tokens for it in prefill)
    assert stats["decode_tokens"] == len(decode)
    planned = stats["prefill_tokens"] + stats["decode_tokens"]
    assert stats["planned_tokens"] == planned
    assert 0 <= planned <= sched.max_num_batched_tokens
    assert len(items) <= sched.max_num_seqs
    seq_ids = [it.seq.seq_id for it in items]
    assert len(set(seq_ids)) == len(seq_ids), "同一请求每轮至多一个 item"
    assert [it.phase for it in items] == ["decode"] * len(decode) + ["prefill"] * len(prefill)
    for it in decode:
        assert it.scheduled_tokens == 1 and it.needs_sample
        assert it.seq.num_scheduled_tokens == 1
    for it in prefill:
        assert 0 < it.scheduled_tokens <= sched.chunk_size
        assert it.needs_sample == it.is_last_chunk
        assert it.offset_before == it.seq.prefill_offset


def setup_decode_source(sched: Scheduler, count: int = 2, now: float = 1.0,
                        max_tokens: int = 100) -> list[Sequence]:
    """构造 count 条 RUNNING 的 decode 源（逐条 prefill 完成），返回 seqs。

    请求 prompt 均为 1 token：第一条单独 prefill；其后每条的接纳轮会混合
    更早请求的 decode（Day9 语义），故 max_tokens 需预留这些额外 decode 配额
    （默认 100 足够）。
    """
    seqs = []
    t = now
    for _ in range(count):
        seq = make_seq(1, max_tokens=max_tokens)
        sched.add(seq)
        for _ in range(200):
            items, phase = sched.schedule(now=t)
            sched.postprocess(items, tokens_for(items), now=t)
            t += 1.0
            if seq.status == RUNNING:
                break
        else:
            raise AssertionError("setup 未在有界轮次内完成")
        seqs.append(seq)
    return seqs


# ============================== 混合表示 ==============================

class TestMixedBatchRepresentation:
    """混合批次的逐请求阶段标注与轮级 phase 标签（§3.1/§3.2）。"""

    def test_mixed_round_contains_both_phases(self):
        """设计 §5.1 时序：decode a,b 在前，prefill c（q=B-D）在后，phase=mixed。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        assert item_view(items) == [("decode", a.seq_id, 1), ("decode", b.seq_id, 1),
                                    ("prefill", c.seq_id, 6)]
        assert_mixed_round_invariants(sched, items, phase)
        sched.postprocess(items, tokens_for(items), now=10.0)
        # decode 先推进（setup 期间 a 已因 b 的接纳多 decode 1 次），c 保持 WAITING
        assert a.num_completion_tokens == 3 and b.num_completion_tokens == 2
        assert c.status == WAITING and c.prefill_offset == 6

    def test_phase_labels_for_four_round_shapes(self):
        """phase 四值：mixed / prefill / decode / idle 与批次构成一一对应。"""
        # idle
        sched = make_scheduler()
        items, phase = sched.schedule(now=1.0)
        assert items == [] and phase == "idle"
        # prefill（无 running）
        c = make_seq(4, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=2.0)
        assert phase == "prefill" and [it.phase for it in items] == ["prefill"]
        sched.postprocess(items, tokens_for(items), now=2.0)
        # decode（无 waiting）
        items, phase = sched.schedule(now=3.0)
        assert phase == "decode" and [it.phase for it in items] == ["decode"]
        sched.postprocess(items, tokens_for(items), now=3.0)
        # mixed：新 waiting + 既有 running
        d = make_seq(4, max_tokens=4)
        sched.add(d)
        items, phase = sched.schedule(now=4.0)
        assert phase == "mixed"

    def test_last_chunk_admission_excludes_same_round_decode(self):
        """最后 chunk 接纳的请求本轮不再以 decode item 出现（§3.4 设计要点 3）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(2, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        assert item_view(items) == [("decode", a.seq_id, 1), ("prefill", c.seq_id, 2)]
        sched.postprocess(items, tokens_for(items), now=10.0)
        # c 的首个 completion 来自最后 chunk 采样，同轮无 c 的 decode item
        assert c.status == RUNNING and c.num_completion_tokens == 1
        assert a.num_completion_tokens == 2

    def test_mixed_round_invariants_hold_across_drive(self):
        """逐轮断言：混合/纯阶段轮的结构与预算不变量恒成立。"""
        sched = make_scheduler(num_blocks=32, max_num_batched_tokens=10,
                               max_num_seqs=4, chunk_size=5)
        for i in range(8):
            sched.add(make_seq(1 + i, max_tokens=3))
        t = 1.0
        for _ in range(120):
            if sched.is_finished():
                break
            items, phase = sched.schedule(now=t)
            assert_mixed_round_invariants(sched, items, phase)
            sched.postprocess(items, tokens_for(items), now=t)
            t += 1.0
        assert sched.is_finished()


# ============================== 混合预算 ==============================

class TestMixedBudget:
    """混合预算公式与分阶段统计（§3.3/§5.2 不变量 14）。"""

    def test_mixed_budget_formula_and_stats_recomputable(self):
        """planned = sum(prefill q) + decode 条数；事件分阶段量可由 items 重算。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        stats = sched.last_schedule_stats
        assert stats["planned_tokens"] == 2 + 6
        assert stats["prefill_tokens"] == 6 and stats["decode_tokens"] == 2
        assert stats["prefill_items"] == 1 and stats["decode_items"] == 2
        assert stats["token_budget"] == 8
        assert_mixed_round_invariants(sched, items, phase)

    def test_decode_budget_reserved_before_prefill(self):
        """decode 先占预算：prefill 首候选拆分阈值 = B - D（§3.4 结构性保底）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        # q = min(20, chunk_size, 8-2) = 6：decode 占用的 2 个预算先行扣除
        assert [it.scheduled_tokens for it in items if it.phase == "prefill"] == [6]
        sched.postprocess(items, tokens_for(items), now=10.0)

    def test_budget_exactly_filled_by_mixed_round(self):
        """decode 条数 + prefill q 恰好等于 B：逐轮不越界。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        srcs = setup_decode_source(sched, count=3)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        for t in range(10, 20):
            items, phase = sched.schedule(now=float(t))
            assert_mixed_round_invariants(sched, items, phase)
            sched.postprocess(items, tokens_for(items), now=float(t))
            if sched.is_finished():
                break

    def test_mixed_round_respects_sequence_cap(self):
        """序列数名额跨阶段共享：decode 占满 cap 后 prefill 不再接纳。"""
        sched = make_scheduler(max_num_batched_tokens=64, max_num_seqs=2)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(4, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert len(items) == 2
        assert decision_of(sched.last_schedule_stats, c)["reason"] == "sequence_cap"


# ============================== decode 保底与归因 ==============================

class TestDecodePriorityAttribution:
    """decode_priority 三分支判定与预算统计口径（§3.5）。"""

    def test_decode_priority_when_decode_consumes_budget(self):
        """分支一：D>0 且 needed<=B -> decode_priority（需求已估算、KV 已查询）。"""
        sched = make_scheduler(max_num_batched_tokens=2)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(1, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "decode_priority"
        assert dc["needed_tokens"] == 1
        assert dc["kv_checked"] is True
        assert dc["phase"] == "prefill"
        # 不计入预算等待
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0
        assert c.seq_id not in sched.budget_wait

    def test_budget_when_needed_exceeds_budget_despite_decode(self):
        """分支二：D>0 但 needed>B -> budget（延后是请求自身规模所致）。"""
        sched = make_scheduler(max_num_batched_tokens=2)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(4, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "budget"
        assert dc["needed_tokens"] == 4
        # budget 原因计入预算等待
        assert sched.last_schedule_stats["budget_deferred_requests"] == 1
        assert c.seq_id in sched.budget_wait

    def test_budget_when_no_decode_this_round(self):
        """分支三：D==0 -> budget（Day7 口径不变，且不查询 KV）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, b, c = (make_seq(3, 8), make_seq(5, 8), make_seq(2, 8))
        for s in (a, b, c):
            sched.add(s)
        items, phase = sched.schedule(now=1.0)
        assert phase == "prefill"
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "budget"
        assert dc["needed_tokens"] is None      # 未查询 KV，不虚构需求
        assert dc["kv_checked"] is False

    def test_budget_when_needed_first_exceeds_budget(self):
        """不变量 18 字面规则（审查 §4.1 反例）：D>0、首候选 needed=20>B=8 拆分
        接纳后，后续小候选（自身 needed=5<=B）延后仍记 budget——无 decode 时
        首候选会整占预算，延后并非 decode 造成。事件记录 needed_first 供复算。"""
        sched = make_scheduler(max_num_batched_tokens=8, chunk_size=8)
        a, b = setup_decode_source(sched, count=2)
        p1 = make_seq(20, max_tokens=4)
        p2 = make_seq(5, max_tokens=4)
        sched.add(p1)
        sched.add(p2)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        assert item_view(items) == [("decode", a.seq_id, 1), ("decode", b.seq_id, 1),
                                    ("prefill", p1.seq_id, 6)]
        dp2 = decision_of(sched.last_schedule_stats, p2)
        assert dp2["reason"] == "budget"
        assert dp2["needed_tokens"] is None   # p2 未被估算（needed_first 已由 p1 提供）
        assert sched.last_schedule_stats["needed_first"] == 20
        assert sched.last_schedule_stats["budget_deferred_requests"] == 1
        assert p2.seq_id in sched.budget_wait

    def test_budget_when_needed_first_exceeds_budget_nonfirst_scan_stop(self):
        """同上反例走"非首候选整段放不下"路径：B=12、D=2、p1(20) 接纳 8 后余 2，
        p2(5) 放不下即停——needed_first=20>B=12 记 budget，不误记 decode_priority。"""
        sched = make_scheduler(max_num_batched_tokens=12, chunk_size=8)
        a, b = setup_decode_source(sched, count=2)
        p1 = make_seq(20, max_tokens=4)
        p2 = make_seq(5, max_tokens=4)
        sched.add(p1)
        sched.add(p2)
        items, phase = sched.schedule(now=10.0)
        assert phase == "mixed"
        assert item_view(items) == [("decode", a.seq_id, 1), ("decode", b.seq_id, 1),
                                    ("prefill", p1.seq_id, 8)]
        dp2 = decision_of(sched.last_schedule_stats, p2)
        assert dp2["reason"] == "budget"
        assert dp2["needed_tokens"] == 5      # 非首候选路径：候选自身需求如实入档
        assert sched.last_schedule_stats["needed_first"] == 20
        assert sched.last_schedule_stats["budget_deferred_requests"] == 1

    def test_hol_follows_decode_priority(self):
        """被 decode 挤掉候选的后续请求记 HOL，blocking_reason 原样跟随。"""
        sched = make_scheduler(max_num_batched_tokens=2)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(1, max_tokens=4)
        d = make_seq(1, max_tokens=4)
        sched.add(c)
        sched.add(d)
        items, phase = sched.schedule(now=10.0)
        assert decision_of(sched.last_schedule_stats, c)["reason"] == "decode_priority"
        dd = decision_of(sched.last_schedule_stats, d)
        assert dd["reason"] == "head_of_line"
        assert dd["blocking_reason"] == "decode_priority"
        assert dd["blocked_by_seq_id"] == c.seq_id
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0

    def test_decode_never_deferred_while_waiting_prefill_exists(self):
        """decode 保底的直接证据：长 prefill 存在时 decode 每轮照样推进。"""
        sched = make_scheduler(max_num_batched_tokens=8, chunk_size=4)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(40, max_tokens=4)
        sched.add(c)
        for t in range(10, 20):
            items, phase = sched.schedule(now=float(t))
            decode_tokens = [it.scheduled_tokens for it in items if it.phase == "decode"]
            assert decode_tokens == [1, 1], "已有 decode 请求必须每轮推进"
            sched.postprocess(items, tokens_for(items), now=float(t))
            if c.status == RUNNING:
                break
        assert c.status == RUNNING


# ============================== chunk 行为（混合轮） ==============================

class TestMixedChunkBehavior:
    """长 prompt 与 decode 同轮逐块推进；Day8 chunk 规则在混合轮内不回归。"""

    def test_long_prompt_chunked_alongside_decode(self):
        """设计 §5.1 时序：c 的 chunk 区间连续，a/b 的 decode 从未中断。"""
        sched = make_scheduler(max_num_batched_tokens=8, chunk_size=8)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        intervals = []
        decode_rounds = 0
        t = 10.0
        for _ in range(10):
            # 状态快照必须在 schedule 之前：最后 chunk 接纳会把 c 迁移为 RUNNING
            c_waiting = c.status == WAITING
            items, phase = sched.schedule(now=t)
            if c_waiting and any(it.seq is c for it in items):
                it_c = next(it for it in items if it.seq is c)
                intervals.append((it_c.offset_before, it_c.offset_before + it_c.scheduled_tokens))
            if any(it.phase == "decode" for it in items):
                decode_rounds += 1
            sched.postprocess(items, tokens_for(items), now=t)
            t += 1.0
            if c.status == RUNNING and c.num_completion_tokens >= 1:
                break
        # 区间 [0,6) [6,12) [12,18) [18,20)：连续、单调、无重叠，并集 [0,20)
        pos = 0
        for start, end in intervals:
            assert start == pos
            pos = end
        assert pos == 20
        # c 的 4 个 chunk 轮中 decode 全部参与（decode-first）
        assert decode_rounds == 4

    def test_first_candidate_splits_by_remaining_after_decode(self):
        """首候选拆分按 B-D 计算；chunk_size 与 B 两条上限同时生效。"""
        sched = make_scheduler(max_num_batched_tokens=10, chunk_size=3)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        # q = min(20, chunk_size=3, 10-1) = 3：chunk_size 主导
        assert [it.scheduled_tokens for it in items if it.phase == "prefill"] == [3]
        sched.postprocess(items, tokens_for(items), now=10.0)

    def test_scan_position_separation_in_mixed_round(self):
        """中间 chunk 请求原地保留，后续整段请求同轮接纳（Day8 规则在混合轮不回归）。"""
        sched = make_scheduler(max_num_batched_tokens=64, chunk_size=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)   # 分块请求（队首）
        d = make_seq(4, max_tokens=4)    # 整段请求
        sched.add(c)
        sched.add(d)
        items, phase = sched.schedule(now=10.0)
        # r1（mixed）：decode a + prefill c chunk[0,8) + prefill d 整段完成
        assert [(it.phase, it.seq.seq_id) for it in items] == [
            ("decode", a.seq_id), ("prefill", c.seq_id), ("prefill", d.seq_id)]
        assert c.status == WAITING and d.status == RUNNING
        sched.postprocess(items, tokens_for(items), now=10.0)
        # r2（mixed）：decode a、decode d（新 RUNNING 请求），c 原地保留续 chunk 2
        items, phase = sched.schedule(now=11.0)
        assert [(it.phase, it.seq.seq_id) for it in items] == [
            ("decode", a.seq_id), ("decode", d.seq_id), ("prefill", c.seq_id)]
        assert list(sched.waiting) == [c]
        it_c = next(it for it in items if it.seq is c)
        assert it_c.offset_before == 8
        # chunk_index 是调度决策快照字段（BatchItem 不携带，走 decision_of）
        assert decision_of(sched.last_schedule_stats, c)["chunk_index"] == 2
        sched.postprocess(items, tokens_for(items), now=11.0)


# ============================== 采样对齐 ==============================

class TestMixedSamplingAlignment:
    """混合轮采样契约：needs_sample 快照对齐、中间 chunk 不采样（§4.5）。"""

    def test_mixed_round_token_alignment(self):
        """decode 与最后 chunk 同轮各 1 token，按 items 顺序注入与消费。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(2, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert [it.needs_sample for it in items] == [True, True]
        sched.postprocess(items, [101, 102], now=10.0)
        # items 顺序 [decode a, prefill c]：a 得 101、c 得 102
        assert a.last_token == 101
        assert c.last_token == 102 and c.num_completion_tokens == 1

    def test_mid_chunk_no_token_in_mixed_round(self):
        """混合轮中的中间 chunk 不产出采样 token：注入数 = decode 条数。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        assert [it.needs_sample for it in items] == [True, False]
        assert [it.scheduled_tokens for it in items if it.phase == "prefill"] == [7]
        # 只注入 decode 的 token：数量正确，postprocess 正常提交
        sched.postprocess(items, [201], now=10.0)
        assert a.last_token == 201
        assert c.num_completion_tokens == 0 and c.prefill_offset == 7

    def test_token_count_mismatch_rejected_in_mixed_round(self):
        """token 数与需采样集合不一致：显式抛错，不静默截断。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(2, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [7], now=10.0)   # 缺 1 个（期望 2）
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [7, 8, 9], now=10.0)  # 多 1 个

    def test_mid_chunk_rejects_token_even_in_mixed_round(self):
        """中间 chunk 传入采样 token（注入错位）：显式拒绝，decode token 不受影响。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        # 错误注入：给中间 chunk 也配上 token（2 个 != 期望 1 个）
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [7, 8], now=10.0)
        assert c.prefill_offset == 0 and c.num_scheduled_tokens == 7


# ============================== 执行契约（runner spy） ==============================

class TestRunnerSubBatchContract:
    """同轮分组执行契约：子批顺序、展平输入长度与计划对账（§4.1）。"""

    @staticmethod
    def make_engine(sched: Scheduler):
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text])
        calls = []

        def fake_call(method, items_):
            # runner_input_len 对账（CPU 侧）：decode 展平 = 条数，prefill 展平 = sum(q)
            decode_len = sum(1 for it in items_ if it.phase == "decode")
            prefill_len = sum(it.scheduled_tokens for it in items_ if it.phase == "prefill")
            calls.append({"decode_len": decode_len, "prefill_len": prefill_len,
                          "phases": [it.phase for it in items_],
                          "seq_ids": [it.seq.seq_id for it in items_]})
            out = tokens_for(items_, 7)
            return out if out else None

        engine.model_runner = SimpleNamespace(call=fake_call)
        return engine, calls

    def test_runner_receives_decode_before_prefill(self):
        """同一调度轮内 decode 子批先于 prefill 子批（items 有序）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        engine, calls = self.make_engine(sched)
        engine.add_request(list(range(1, 21)), SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()
        record = calls[0]
        assert record["phases"] == ["decode", "decode", "prefill"]
        assert record["seq_ids"][:2] == [a.seq_id, b.seq_id]
        assert record["seq_ids"][2] == c.seq_id

    def test_runner_input_len_matches_plan(self):
        """runner_input_len 对账：decode 条数 + prefill sum(q) == 计划总量。"""
        sched = make_scheduler(max_num_batched_tokens=12, chunk_size=5)
        a, b = setup_decode_source(sched, count=2)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        engine, calls = self.make_engine(sched)
        engine.add_request(list(range(1, 21)), SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()
        record = calls[0]
        stats = sched.last_schedule_stats
        assert record["decode_len"] == stats["decode_tokens"] == 2
        assert record["prefill_len"] == stats["prefill_tokens"] == 5
        assert record["decode_len"] + record["prefill_len"] == stats["planned_tokens"] == 7

    def test_mixed_engine_step_returns_total_workload(self):
        """step() 第二返回值 = 本轮总 query token 数（恒非负，§4.4）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(6, max_tokens=4)
        sched.add(c)
        engine, calls = self.make_engine(sched)
        engine.add_request(list(range(1, 7)), SamplingParams(max_tokens=4, ignore_eos=True))
        outputs, num_tokens = engine.step()
        assert num_tokens == 1 + 6   # decode 1 条 + prefill 6 token
        assert sched.last_schedule_stats["phase"] == "mixed"

    def test_engine_rejects_malformed_mixed_plan(self):
        """调用前显式校验：decode item 计数 != 1 / seq 重复 / 未知 phase 均拒绝。"""
        sched = make_scheduler(max_num_batched_tokens=32)
        engine, calls = self.make_engine(sched)
        engine.add_request("hi", SamplingParams(max_tokens=4, ignore_eos=True))
        seq = sched.add.__self__ if False else None  # 占位：下方直接构造
        good = make_seq(4, max_tokens=4)
        real_schedule = sched.schedule

        # decode item 计数 != 1
        bad_decode = BatchItem(seq=good, phase="decode", scheduled_tokens=2,
                               needs_sample=True, round_id=99)
        sched.schedule = lambda **kw: ([bad_decode], "decode")
        sched.last_schedule_stats = {"round_id": 99, "token_budget": 32,
                                     "planned_tokens": 2}
        with pytest.raises(ValueError, match="decode item 的接纳数必须为 1"):
            engine.step()
        # 每次计划校验失败都会锁定该 Engine；后续 malformed case 使用独立
        # Engine，保持失败锁语义而不让测试依赖同一实例重试。
        sched2 = make_scheduler(max_num_batched_tokens=32)
        engine2, calls2 = self.make_engine(sched2)
        good2 = make_seq(4, max_tokens=4)
        dup1 = BatchItem(seq=good2, phase="decode", scheduled_tokens=1,
                         needs_sample=True, round_id=99)
        dup2 = BatchItem(seq=good2, phase="decode", scheduled_tokens=1,
                         needs_sample=True, round_id=99)
        sched2.schedule = lambda **kw: ([dup1, dup2], "decode")
        sched2.last_schedule_stats = {"round_id": 99, "token_budget": 32,
                                      "planned_tokens": 2}
        with pytest.raises(ValueError, match="seq_id 重复"):
            engine2.step()
        sched3 = make_scheduler(max_num_batched_tokens=32)
        engine3, calls3 = self.make_engine(sched3)
        good3 = make_seq(4, max_tokens=4)
        weird = BatchItem(seq=good3, phase="train", scheduled_tokens=2,
                          needs_sample=False, round_id=99)
        sched3.schedule = lambda **kw: ([weird], "train")
        sched3.last_schedule_stats = {"round_id": 99, "token_budget": 32,
                                      "planned_tokens": 2}
        with pytest.raises(ValueError, match="未知 phase"):
            engine3.step()
        assert calls == [] and calls2 == [] and calls3 == []  # 模型从未被调用


# ============================== KV 与资源 ==============================

class TestMixedKVAndResources:
    """混合轮 KV 记账：不超卖、同轮完成释放、KV 不足归因独立（§4.6）。"""

    def test_kv_not_oversold_in_mixed_round(self):
        """decode may_append 与 prefill allocate 同轮消耗同一 free 池，恰好用尽不超卖。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=10 ** 6)
        a = make_seq(8, max_tokens=3)   # prefill 1 块；len=9 时 decode 需第 2 块
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        sched.postprocess(items, tokens_for(items), now=1.0)   # a: 1 块，len=9
        b = make_seq(8, max_tokens=3)   # prefill 1 块
        c = make_seq(8, max_tokens=3)   # prefill 1 块
        sched.add(b)
        sched.add(c)
        bm = sched.block_manager
        # a.len=9 -> 本轮 decode 需要 1 个新块；b、c 各分配 1 块：恰好用完 4 块
        items, phase = sched.schedule(now=2.0)
        assert phase == "mixed"
        assert len(bm.used_block_ids) == 4
        assert len(bm.free_block_ids) == 0
        sched.postprocess(items, tokens_for(items), now=2.0)

    def test_same_round_completion_releases_blocks(self):
        """同轮 decode item 完成与 prefill 最后 chunk 完成：两者释放均幂等且不串扰。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8)
        a = make_seq(4, max_tokens=1)   # prefill 后立即完成（comp 1 == max_tokens）
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        sched.postprocess(items, tokens_for(items), now=1.0)
        assert a.status == FINISHED
        b = make_seq(4, max_tokens=1)   # 同样一轮完成
        sched.add(b)
        items, phase = sched.schedule(now=2.0)
        assert phase == "prefill"
        sched.postprocess(items, tokens_for(items), now=2.0)
        assert b.status == FINISHED
        bm = sched.block_manager
        assert len(bm.used_block_ids) == 0
        assert len(bm.free_block_ids) == 8

    def test_decode_finish_does_not_affect_prefill_item(self):
        """混合轮内 decode item 达到 max_tokens 终止，prefill item 正常提交。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a = make_seq(1, max_tokens=2)
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        sched.postprocess(items, tokens_for(items), now=1.0)   # a comp=1
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=2.0)
        assert phase == "mixed"
        sched.postprocess(items, tokens_for(items), now=2.0)
        # a 完成（setup 后 comp 1 + 本轮 decode = comp 2），c 的 chunk 正常推进
        assert a.status == FINISHED and a.num_completion_tokens == 2
        assert c.status == WAITING and c.prefill_offset == 7
        assert a.block_table == [] and c.block_table   # a 释放、c 持有

    def test_kv_shortage_decode_defers_prefill_stops(self):
        """KV 不足：decode 侧无合法 victim 时延后（Day10 §4.2.2 规则 5：候选保留
        RUNNING/KV 回队首原位，不再做无谓的自抢占）；延后原因记 kv_capacity，
        独立于预算归因。prefill 侧停止接纳与 victim 抢占由请求控制测试覆盖。"""
        sched = make_scheduler(num_blocks=3, max_num_batched_tokens=10 ** 6)
        a = make_seq(8, max_tokens=5)   # len=9 时 decode 需第 2 块
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        sched.postprocess(items, tokens_for(items), now=1.0)   # a 1 块，len=9
        b = make_seq(8, max_tokens=5)
        sched.add(b)
        # 混合轮：decode a 补第 2 块（free 2->1），prefill b 分配 1 块（free 1->0）
        items, phase = sched.schedule(now=2.0)
        assert phase == "mixed"
        sched.postprocess(items, tokens_for(items), now=2.0)
        # 下一轮：decode a（len=10）不需要新块，正常接纳；decode b（len=9）
        # 写位置 8 需新块且 free=0、running 中无合法 victim -> b 延后：
        # 保留 RUNNING 与已提交 KV 回到队首原位，不抢占、不释放
        items, phase = sched.schedule(now=3.0)
        db = decision_of(sched.last_schedule_stats, b)
        assert db["reason"] == "kv_capacity"
        assert b.num_preempts == 0
        assert b.status == RUNNING and b.block_table
        assert [it.seq.seq_id for it in items] == [a.seq_id]
        # KV 原因不算预算拒绝（决策归因与预算统计独立）
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0
        assert b.seq_id not in sched.budget_wait
        sched.block_manager.check_ledger()


# ============================== 生命周期（混合轮） ==============================

class TestMixedLifecycle:
    """混合轮中的取消/超时/抢占恢复与 ID 复用（§4.6）。"""

    def test_cancel_mid_chunk_in_mixed_round(self):
        """prefill item 执行期间取消：该 item 不提交，同轮 decode item 正常落账。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        c.request_cancel("mid_exec")   # 模拟执行期间取消
        sched.postprocess(items, tokens_for(items), now=10.0)
        assert c.status == CANCELLED and c.prefill_offset == 0
        assert a.num_completion_tokens == 2   # decode item 不受影响
        assert not sched.block_manager.used_block_ids & set()

    def test_timeout_mid_chunk_in_mixed_round(self):
        """prefill item 执行期间跨 deadline：输出丢弃、按 TIMEOUT 终止；decode item 正常。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4, deadline=12.0)
        sched.add(c)
        items, phase = sched.schedule(now=10.0)
        sched.postprocess(items, tokens_for(items), now=10.0)
        assert c.prefill_offset == 7
        items, phase = sched.schedule(now=11.0)   # 调度时尚未超时
        assert any(it.seq is c for it in items)
        sched.postprocess(items, tokens_for(items), now=13.0)   # 执行返回时已跨过 deadline
        assert c.status == TIMEOUT and c.prefill_offset == 0
        assert a.num_completion_tokens == 3   # 同轮 decode 正常提交

    def test_preempt_resume_recompute_in_mixed_flow(self):
        """抢占恢复的请求以 WAITING 参与 prefill 扫描，混合轮内按 prefix 重算接纳。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=10 ** 6)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(16, max_tokens=8)
        sched.add(c)
        items, _ = sched.schedule(now=10.0)
        sched.postprocess(items, tokens_for(items), now=10.0)   # c 完成 prefill，len=17
        items, _ = sched.schedule(now=11.0)
        sched.postprocess(items, tokens_for(items), now=11.0)   # c decode 1 步，len=18
        sched.preempt(c, now=12.0)
        sched.resume(c)
        assert c.status == WAITING and not c.block_table
        # 混合轮：decode a + prefill c 重算（前 2 块命中自身缓存，需求 2）
        items, phase = sched.schedule(now=13.0)
        assert phase == "mixed"
        it_c = next(it for it in items if it.seq is c)
        assert it_c.phase == "prefill" and it_c.scheduled_tokens == 2
        assert it_c.offset_before == 16 and it_c.is_last_chunk
        sched.postprocess(items, tokens_for(items), now=13.0)
        assert c.status == RUNNING and c.prefill_offset == 18
        assert c.token_ids[-3:] == [7, 7, 7]

    def test_new_request_visible_next_round_not_current(self):
        """本轮批次已定后加入的新请求，下一轮才可见（动态加入语义）。"""
        sched = make_scheduler(max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        items, phase = sched.schedule(now=10.0)
        c = make_seq(4, max_tokens=4)
        sched.add(c)   # 本轮 schedule 已发生
        assert all(it.seq is not c for it in items)
        sched.postprocess(items, tokens_for(items), now=10.0)
        items, phase = sched.schedule(now=11.0)
        assert any(it.seq is c for it in items)   # 下一轮可见


# ============================== 迟到与重复（round 关联） ==============================

class TestRoundAssociation:
    """round_id 关联校验：迟到/重复结果双保险（§4.3/§5.2 不变量 19）。"""

    def test_round_id_unique_and_monotonic(self):
        sched = make_scheduler(max_num_batched_tokens=32)
        a = make_seq(4, max_tokens=2)
        sched.add(a)
        seen = []
        t = 1.0
        for _ in range(10):
            items, phase = sched.schedule(now=t)
            for it in items:
                seen.append(it.round_id)
            sched.postprocess(items, tokens_for(items), now=t)
            t += 1.0
            if sched.is_finished():
                break
        assert seen == sorted(seen) and len(set(seen)) == len(seen)

    def test_stale_batch_after_replan_rejected(self):
        """同 seq 被重新规划后，旧 round 的批次重放被 round 校验拒绝（Day9 加固）。"""
        sched = make_scheduler(max_num_batched_tokens=8, chunk_size=1024)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        items1, _ = sched.schedule(now=1.0)
        sched.postprocess(items1, [], now=1.0)     # chunk 1 提交
        items2, _ = sched.schedule(now=2.0)        # 同 seq 被重新规划
        assert items2[0].round_id != items1[0].round_id
        with pytest.raises(ValueError, match="迟到"):
            sched.postprocess(items1, [], now=2.0)  # 旧批次重放
        sched.postprocess(items2, [], now=2.0)      # 新计划正常提交
        assert c.prefill_offset == 16

    def test_terminal_batch_replay_is_safe_noop(self):
        """全终态旧批次重放：round 校验豁免，安全空操作（Day6/7 契约保持）。"""
        sched = make_scheduler(max_num_batched_tokens=32)
        a = make_seq(1, max_tokens=1)
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        sched.postprocess(items, tokens_for(items), now=1.0)
        assert a.status == FINISHED
        sched.schedule(now=2.0)   # 推进调度轮次（空轮）
        sched.postprocess(items, tokens_for(items), now=3.0)   # 重放：不抛错
        assert sched.budget_wait_closed_seconds_total == 0.0

    def test_live_batch_with_wrong_round_rejected(self):
        """含活动请求的批次 round_id 与最近调度轮不匹配：显式拒绝。"""
        sched = make_scheduler(max_num_batched_tokens=32)
        a = make_seq(4, max_tokens=4)
        sched.add(a)
        items, _ = sched.schedule(now=1.0)
        forged = [BatchItem(seq=it.seq, phase=it.phase,
                            scheduled_tokens=it.scheduled_tokens,
                            offset_before=it.offset_before,
                            is_last_chunk=it.is_last_chunk,
                            needs_sample=it.needs_sample,
                            round_id=it.round_id + 100) for it in items]
        with pytest.raises(ValueError, match="迟到"):
            sched.postprocess(forged, tokens_for(forged), now=1.0)


# ============================== 日志证据 ==============================

class TestMixedLogEvidence:
    """事件字段白名单、分阶段计数可重算、round 关联、无明文（§6.4）。"""

    def test_mixed_round_events_fully_checkable(self, caplog):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8)
        a, = setup_decode_source(sched, count=1)
        c = make_seq(20, max_tokens=4)
        sched.add(c)
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(encode=lambda t: [1])
        engine.model_runner = SimpleNamespace(
            call=lambda m, items_: tokens_for(items_, 7) or None)
        with caplog.at_level(logging.INFO):
            for _ in range(30):
                if sched.is_finished():
                    break
                engine.step()
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        sched_rounds = [e for e in events if e["event"] == "scheduler_round"]
        engine_rounds = [e for e in events if e["event"] == "engine_round"]
        assert sched_rounds and engine_rounds
        mixed = [e for e in sched_rounds if e["phase"] == "mixed"]
        assert mixed, "本场景必须产生混合轮"
        for e in sched_rounds:
            # 分阶段计数可重算：prefill_tokens + decode_tokens == planned
            assert e["prefill_tokens"] + e["decode_tokens"] == e["planned_tokens"]
            assert e["prefill_items"] + e["decode_items"] == e["scheduled_requests"]
            assert e["phase"] in ("prefill", "decode", "mixed", "idle")
            assert 0 <= e["planned_tokens"] <= e["token_budget"]
            for d in e["decisions"]:
                assert d["phase"] in ("prefill", "decode")
                assert d["reason"] in {"scheduled", "budget", "sequence_cap",
                                       "kv_capacity", "head_of_line",
                                       "decode_priority", "paused"}
        # 混合轮：decode item 在前（decisions 顺序与 items 顺序一致）
        for e in mixed:
            phases = [d["phase"] for d in e["decisions"] if d["reason"] == "scheduled"]
            assert phases == sorted(phases, key=lambda p: 0 if p == "decode" else 1)
        # scheduler/engine 事件按 round_id 关联且 planned 一致
        sched_by_id = {e["round_id"]: e for e in sched_rounds}
        for e in engine_rounds:
            if e["outcome"] == "completed":
                assert e["round_id"] in sched_by_id
                assert sched_by_id[e["round_id"]]["planned_tokens"] == e["planned_tokens"]
                assert e["executed_tokens"] == e["planned_tokens"]
                assert e["prefill_tokens"] + e["decode_tokens"] == e["planned_tokens"]
                assert e["prefill_items"] == e["prefill_chunks"]
            elif e["outcome"] == "idle":
                assert e["model_called"] is False and e["prefill_tokens"] == 0
        # 无 prompt/token 明文
        assert all("token_ids" not in e and "prompt" not in e for e in events)

    def test_log_level_off_by_default_no_output(self):
        """库代码不做 basicConfig：默认级别下无输出。"""
        import io
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            sched = make_scheduler()
            setup_decode_source(sched, count=2)
            sched.schedule(now=9.0)
            assert stream.getvalue() == ""
        finally:
            root.removeHandler(handler)


# ============================== 随机混合 ==============================

class TestMixedRandomScenario:
    """固定 seed 交错 add/cancel/timeout/preempt/resume/mark_*/旧批次重放；
    轮次上限防挂；身份/队列/引用计数/free-used 全平衡（§9.1 随机混合行）。"""

    def test_mixed_ops_balance_and_termination(self):
        rng = random.Random(20260913)
        sched = make_scheduler(num_blocks=24, max_num_batched_tokens=16,
                               max_num_seqs=8, chunk_size=4)
        id_pool = [f"mix9-{i}" for i in range(6)]
        stale_batches = []   # (items, token_ids)：迟到重放候选
        now = 100.0
        decode_priority_seen = False
        for _ in range(150):
            now += 1.0
            op = rng.random()
            if op < 0.25:
                seq = make_seq(rng.randint(1, 18), max_tokens=rng.randint(1, 4),
                               request_id=rng.choice(id_pool))
                try:
                    sched.add(seq)
                except ValueError:
                    pass  # 活动 ID 冲突：符合预期的原子拒绝
            elif op < 0.35 and sched.requests:
                sched.cancel(rng.choice(list(sched.requests)), now=now)
            elif op < 0.42 and sched.requests:
                sched.timeout(rng.choice(list(sched.requests)), now=now)
            elif op < 0.48 and sched.running:
                seq = rng.choice(list(sched.running))
                if seq.status == RUNNING:
                    sched.preempt(seq, now=now)
                    sched.resume(seq)
            elif op < 0.54 and sched.requests:
                victim = rng.choice(list(sched.requests.values()))
                if victim.status == RUNNING:
                    victim.mark_finished("stop")
            elif op < 0.60 and stale_batches:
                stale_items, tokens = stale_batches.pop(rng.randrange(len(stale_batches)))
                if all(it.seq.is_terminal for it in stale_items):
                    sched.postprocess(stale_items, tokens, now=now)
            # 驱动一轮
            running_before = [s.seq_id for s in sched.running]
            preempt_counts = {s.seq_id: s.num_preempts for s in sched.running}
            items, phase = sched.schedule(now=now)
            assert_mixed_round_invariants(sched, items, phase)
            # decode-first：轮前 RUNNING 请求要么被 decode 接纳，要么因 KV 不足
            # 让块（合法例外：有 victim 时被抢占回 waiting；无 victim 时延后、
            # 保留 RUNNING/KV 且必须有 kv_capacity 归因，Day10 §4.2.2 规则 5），
            # 不允许因预算/名额被延后
            decode_ids = [it.seq.seq_id for it in items if it.phase == "decode"]
            preempted, kv_deferred = [], []
            for sid in running_before:
                if sid in decode_ids:
                    continue   # 已被本轮 decode 接纳（可能已完成并被收尾）
                s = sched.requests.get(sid)
                if s is None or (s.status == WAITING and s.num_preempts > 0):
                    preempted.append(sid)
                elif s.status == RUNNING:
                    d = decision_of(sched.last_schedule_stats, s)
                    assert d["reason"] == "kv_capacity"
                    assert s.num_preempts == preempt_counts[sid], "延后不应伴随抢占"
                    kv_deferred.append(sid)
            assert (set(decode_ids) | set(preempted) | set(kv_deferred)
                    == set(running_before))
            # phase_priority 全仓库无残留（随机流程中不应出现未知原因）
            for d in sched.last_schedule_stats["decisions"]:
                if d["reason"] == "decode_priority":
                    decode_priority_seen = True
            token_ids = [rng.randint(1, 100) for it in items if it.needs_sample]
            sched.postprocess(items, token_ids, now=now)
            if items:
                stale_batches.append((items, token_ids))
                if len(stale_batches) > 4:
                    stale_batches.pop(0)
            # 轻量不变量：队列互斥、状态与索引一致、chunk 进度单调有界
            assert not (set(sched.waiting) & set(sched.running))
            for s in sched.waiting:
                assert s.status == WAITING and sched.requests.get(s.seq_id) is s
                assert 0 <= s.prefill_offset <= s.prefill_target
            for s in sched.running:
                assert s.status == RUNNING and sched.requests.get(s.seq_id) is s
        assert decode_priority_seen, "随机流程应观察到 decode-first 让路"
        # 收尾：有界清场
        for _ in range(300):
            if sched.is_finished():
                break
            for sid in list(sched.requests):
                sched.cancel(sid, now=now)
            now += 1.0
        assert sched.is_finished()
        bm = sched.block_manager
        free_set = set(bm.free_block_ids)
        assert free_set | set(bm.used_block_ids) == set(range(len(bm.blocks)))
        assert not (free_set & set(bm.used_block_ids))
        assert all(bm.blocks[i].ref_count == 0 for i in free_set)
        assert not sched.waiting and not sched.running and not sched.requests
        assert not sched.budget_wait
        assert not sched._prefill_chunk_count

    def test_every_round_honors_mixed_budget(self):
        """第二轮 seed 交叉验证：混合预算公式与序列上限逐轮恒成立。"""
        rng = random.Random(9999)
        sched = make_scheduler(num_blocks=16, max_num_batched_tokens=9,
                               max_num_seqs=4, chunk_size=3)
        now = 0.0
        for _ in range(100):
            now += 1.0
            if rng.random() < 0.5:
                try:
                    sched.add(make_seq(rng.randint(1, 14), max_tokens=rng.randint(1, 3)))
                except ValueError:
                    pass
            items, phase = sched.schedule(now=now)
            assert_mixed_round_invariants(sched, items, phase)
            sched.postprocess(items, tokens_for(items, 7), now=now)
        for sid in list(sched.requests):
            sched.cancel(sid, now=now)
        assert sched.is_finished()
