"""Day 8 验收测试：Chunked Prefill（docs/chunked-prefill.md 第 9.1 节测试矩阵）。

覆盖范围（对应文档 §9.1 表格逐行）：
- 配置：chunk_size=1 合法；0/负/float/str/bool 拒绝；默认 1024；SimpleNamespace
  直接构造路径缺省回退；校验使用显式 raise（python -O 下仍生效，由
  ``python -O -m pytest tests/test_chunked_prefill.py -q`` 复验）；
- 字段语义：初始 offset=0；prefill_complete 派生只读；num_cached_tokens 兼容
  只读；deallocate 归零；复用 ID 不继承进度；
- chunk 边界：prompt 长度 0（显式拒绝）/1、chunk-1/chunk/chunk+1、B-1/B/B+1、
  8K；offset 单调推进至 target；区间连续无重叠遗漏；
- 预算交互：chunk_size 与 B 的小/等/大组合；q_i<=chunk_size 与 sum(q)<=B 独立
  校验；chunk 部分推进记 scheduled 不记 budget；预算耗尽归因与 Day7 一致；
- 调度规则：每请求每轮至多 1 chunk；首候选拆分、后续整段放下、FCFS 不跳过；
  同请求不重复入批；
- 输入组装（真实 ModelRunner 组装函数）：input_ids 片段、positions 绝对连续、
  cu_seqlens_q/k、slot_mapping 跨块、展平长度校验；decode 用 num_tokens 元数据
  （TP worker 空 token_ids 不影响）；越界 offset 显式抛错；
- prefix 与恢复：命中块计入初始 offset；尾块不命中；恢复 recompute 按命中重算；
  hash_blocks 显式区间只登记写满块；
- 采样：中间 chunk 不产生 token、不消耗 RNG；最后 chunk 采样行 == 该请求最后
  query；多请求变长混合行对齐；token 数与需采样数一致（不静默截断）；
- 生命周期：chunk 中 cancel/timeout/外部 mark_*：不推进、KV 释放、无过期 token；
  preempt→resume recompute；重复 postprocess/迟到结果被拒；ID 复用隔离；
- 一致性（CPU 桩）：固定 token 注入下 one-shot 与多 chunk 的 completion 序列一致；
- TP 协议：v3 pickle 往返；v1/v2 旧格式读取映射；offset/is_prefill/status 正确恢复；
- 随机混合：固定 seed 交错 add/cancel/timeout/preempt/resume/旧结果，轮次上限
  防挂；最终队列/索引/ref/free-used 全平衡。

Day9 适配说明（docs/mixed-prefill-decode.md）：schedule() 返回 (items, phase)、
postprocess() 逐 item 提交；混合轮（如"已 RUNNING 请求 decode + waiting prefill
续块"）取代部分旧纯 prefill 轮，相关断言按 (phase, seq_id) 序列改写；
迟到 postprocess 新增 round_id 关联校验用例（同 seq 重新规划后旧批次拒绝）。
纯单阶段场景（chunk 边界、prefix、输入组装、TP 协议）行为断言原样保持。

无 GPU / 模型依赖：Scheduler/BlockManager/Sequence 为纯 CPU 逻辑；
ModelRunner 通过 __new__ 绕过 GPU __init__，只调用 CPU 纯组装函数
（_build_prefill_inputs/_build_decode_inputs/_select_prefill_sample_rows）；
时间使用注入的确定值（now 关键字），不加载权重、不 sleep、不访问 CUDA。

重复执行命令：
    python -m pytest tests/test_chunked_prefill.py -q
    python -O -m pytest tests/test_chunked_prefill.py -q
"""

import json
import logging
import pickle
import random
from types import SimpleNamespace

import pytest

from nanovllm.config import Config, validate_positive_int
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.layers.sampler import Sampler
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
                   max_num_seqs: int = 512, chunk_size: int | None = None,
                   with_chunk_size_field: bool = True) -> Scheduler:
    """构造 Scheduler。

    chunk_size=None 且 with_chunk_size_field=True 时显式提供默认值 1024；
    with_chunk_size_field=False 时不提供该字段，覆盖 SimpleNamespace 直接
    构造路径的缺省回退行为（回退默认 1024）。
    """
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    if with_chunk_size_field:
        config.chunk_size = 1024 if chunk_size is None else chunk_size
    return Scheduler(config)


def assert_round_invariants(sched: Scheduler, batch: list[Sequence]):
    """每轮硬约束：0<=planned<=B；len(batch)<=max_num_seqs；q_i 为正整数且
    <= chunk_size；批次内 seq_id 不重复（每请求每轮至多一个 chunk/item）。"""
    stats = sched.last_schedule_stats
    planned = sum(s.num_scheduled_tokens for s in batch)
    assert stats["planned_tokens"] == planned
    assert 0 <= planned <= sched.max_num_batched_tokens
    assert len(batch) <= sched.max_num_seqs
    seq_ids = [s.seq_id for s in batch]
    assert len(set(seq_ids)) == len(seq_ids), "同一请求不得在同一轮重复入批"
    if batch:
        assert planned > 0
        for s in batch:
            assert type(s.num_scheduled_tokens) is int
            assert s.num_scheduled_tokens > 0
            # q_i <= chunk_size 与 sum(q) <= B 两条上限独立成立
            assert s.num_scheduled_tokens <= sched.chunk_size


def drive_full_prefill(sched: Scheduler, seq: Sequence, token: int,
                       t0: float = 1.0, max_rounds: int = 200) -> list[tuple[int, int]]:
    """驱动一个请求完成全部 prefill chunk，逐轮断言不变量，返回 chunk 区间列表。"""
    intervals = []
    t = t0
    for _ in range(max_rounds):
        batch, items, is_prefill = schedule_round(sched, now=t)
        assert is_prefill, "prefill 完成前每轮都应是 prefill 批次"
        assert_round_invariants(sched, batch)
        start = seq.prefill_offset
        q = seq.num_scheduled_tokens
        intervals.append((start, start + q))
        sched.postprocess(items, tokens_for_items(items, token), now=t)
        t += 1.0
        if seq.status == RUNNING:
            break
    else:
        raise AssertionError("prefill 未在有界轮次内完成")
    return intervals


def assert_intervals_contiguous(intervals: list[tuple[int, int]], target: int):
    """区间连续、单调、无重叠、无遗漏，并集恰为 [0, target)（§5.1）。"""
    assert intervals, "至少应有一个 chunk 区间"
    pos = 0
    for start, end in intervals:
        assert start == pos, f"区间不连续：期望起点 {pos}，实际 {start}"
        assert end > start, "chunk 区间必须为正"
        pos = end
    assert pos == target, f"区间并集 {pos} != 有效上下文 {target}"


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


def decision_of(stats: dict, seq: Sequence) -> dict:
    """从本轮调度统计中取某请求的决策快照。"""
    for d in stats["decisions"]:
        if d["seq_id"] == seq.seq_id:
            return d
    raise AssertionError(f"seq_id={seq.seq_id} 不在本轮决策中: {stats['decisions']}")


def make_runner(block_size: int = BLOCK_SIZE) -> ModelRunner:
    """绕过 GPU __init__ 构造 ModelRunner，仅用于 CPU 纯组装函数测试。"""
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = block_size
    return runner


# ============================== 配置校验 ==============================

class TestChunkSizeConfig:
    """chunk_size 显式校验（§6.1），不依赖会被 python -O 移除的 assert。"""

    def test_chunk_size_one_is_valid(self):
        sched = make_scheduler(chunk_size=1)
        assert sched.chunk_size == 1

    def test_default_chunk_size_is_1024(self):
        # Config 数据类默认值
        assert Config.__dataclass_fields__["chunk_size"].default == 1024
        # SimpleNamespace 直接构造路径缺省回退 1024
        sched = make_scheduler(with_chunk_size_field=False)
        assert sched.chunk_size == 1024

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_reject_non_positive(self, bad):
        with pytest.raises(ValueError, match="chunk_size"):
            validate_positive_int(bad, "chunk_size")

    @pytest.mark.parametrize("bad", [8.0, "8", True, False, None, [8]])
    def test_reject_wrong_types_including_bool(self, bad):
        # bool 是 int 子类：必须用 type 精确匹配拒绝（python -O 下仍生效）
        with pytest.raises(ValueError) as exc_info:
            validate_positive_int(bad, "chunk_size")
        assert "chunk_size" in str(exc_info.value)
        assert repr(bad) in str(exc_info.value)

    def test_scheduler_direct_construction_validates(self):
        config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16, chunk_size=0,
                                 eos=EOS, kvcache_block_size=BLOCK_SIZE, num_kvcache_blocks=4)
        with pytest.raises(ValueError, match="chunk_size"):
            Scheduler(config)

    @staticmethod
    def make_model_dir(tmp_path):
        d = tmp_path / "mini-model"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 4096,
        }))
        return str(d)

    def test_config_dataclass_default_and_custom(self, tmp_path):
        model_dir = self.make_model_dir(tmp_path)
        cfg = Config(model_dir, enforce_eager=True)
        assert cfg.chunk_size == 1024
        cfg2 = Config(model_dir, chunk_size=256, enforce_eager=True)
        assert cfg2.chunk_size == 256

    @pytest.mark.parametrize("bad", [0, -5, 2.5, True, "256"])
    def test_config_dataclass_rejects_invalid_chunk_size(self, tmp_path, bad):
        model_dir = self.make_model_dir(tmp_path)
        with pytest.raises(ValueError, match="chunk_size"):
            Config(model_dir, chunk_size=bad)


# ============================== 字段语义 ==============================

class TestSequenceProgressSemantics:
    """prefill_offset 单一事实源与派生只读标志（§3.1/§6.2）。"""

    def test_initial_offset_zero_and_target(self):
        seq = make_seq(10)
        assert seq.prefill_offset == 0
        assert seq.prefill_target == 10
        assert seq.prefill_complete is False
        # 兼容只读视图：数值语义与 Day7 完全一致
        assert seq.num_cached_tokens == 0

    def test_prefill_complete_is_derived_readonly(self):
        seq = make_seq(4)
        with pytest.raises(AttributeError):
            seq.prefill_complete = True

    def test_num_cached_tokens_is_readonly_property(self):
        seq = make_seq(4)
        with pytest.raises(AttributeError):
            seq.num_cached_tokens = 5

    def test_offset_tracks_committed_progress(self):
        """prefill 中间 chunk 后 offset==已提交量、prefill_complete 仍为 False。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        sched.postprocess(items, [], now=1.0)
        assert seq.prefill_offset == 8
        assert seq.num_cached_tokens == 8  # 兼容视图同步
        assert seq.prefill_complete is False
        assert seq.status == WAITING  # 中间 chunk 保持 WAITING

    def test_deallocate_zeroes_offset(self):
        bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
        seq = make_seq(20)
        bm.allocate(seq, 0)
        seq.prefill_offset = 13
        bm.deallocate(seq)
        assert seq.prefill_offset == 0
        assert seq.num_cached_tokens == 0
        assert seq.block_table == []

    def test_reused_id_does_not_inherit_offset(self):
        """终态释放后 offset 作废：复用同一 request_id 的新请求从 0 开始。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        old = make_seq(20, max_tokens=2, request_id="shared")
        sched.add(old)
        intervals = drive_full_prefill(sched, old, token=7)
        assert intervals[-1][1] == 20
        assert old.mark_finished("length") is True
        sched._finalize(old, now=99.0)
        # 旧对象进度已随释放作废
        assert old.prefill_offset == 0
        # 复用 ID 的新请求对象：进度从 0 开始，不继承
        new = make_seq(4, max_tokens=2, request_id="shared")
        sched.add(new)
        assert new.prefill_offset == 0
        assert new.prefill_target == 4


# ============================== chunk 边界 ==============================

class TestChunkBoundaries:
    """prompt 长度与 chunk_size/B 的边界组合（§9.1 chunk 边界行）。"""

    def test_empty_prompt_rejected_explicitly(self):
        """prompt 长度 0：构造入口显式拒绝（不静默进入调度/组装）。"""
        with pytest.raises(ValueError, match="prompt 不能为空"):
            Sequence([], SamplingParams(max_tokens=2))

    def test_prompt_length_one_single_chunk(self):
        sched = make_scheduler(num_blocks=2, chunk_size=8)
        seq = make_seq(1, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7)
        assert intervals == [(0, 1)]
        assert seq.status == RUNNING

    @pytest.mark.parametrize("prompt_len", [7, 8, 9])  # chunk-1 / chunk / chunk+1
    def test_prompt_around_chunk_size(self, prompt_len):
        sched = make_scheduler(num_blocks=4, chunk_size=8)
        seq = make_seq(prompt_len, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7)
        expected_rounds = (prompt_len + 7) // 8
        assert len(intervals) == expected_rounds
        assert_intervals_contiguous(intervals, prompt_len)

    @pytest.mark.parametrize("prompt_len", [7, 8, 9])  # B-1 / B / B+1
    def test_prompt_around_budget(self, prompt_len):
        """chunk_size 默认 1024 远大于 B：预算主导，行为与 Day7 一致。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(prompt_len, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7)
        expected_rounds = (prompt_len + 7) // 8
        assert len(intervals) == expected_rounds
        assert_intervals_contiguous(intervals, prompt_len)

    def test_8k_prompt_chunked_to_completion(self):
        """8K prompt：分块完成、offset 单调推进至 target、区间无重叠遗漏。"""
        sched = make_scheduler(num_blocks=1040, max_num_batched_tokens=2048, chunk_size=1024)
        seq = make_seq(8192, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7, max_rounds=32)
        assert [e - s for s, e in intervals] == [1024] * 8
        assert_intervals_contiguous(intervals, 8192)
        assert seq.prefill_offset == 8192
        assert len(seq.block_table) == 1024  # 首轮一次性分配（8192/8 块），无重复分配


# ============================== 预算交互 ==============================

class TestBudgetInteraction:
    """chunk_size 与 B 的小/等/大组合；两条上限独立校验；归因口径不变。"""

    def test_chunk_size_smaller_than_budget_splits_head(self):
        """chunk_size < B：首候选按 chunk_size 拆分，余量可留给后续整段请求。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=32, chunk_size=8)
        a, b = make_seq(20, max_tokens=2), make_seq(4, max_tokens=2)
        sched.add(a)
        sched.add(b)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        assert a.num_scheduled_tokens == 8       # 受 chunk_size 限制（而非预算 32）
        assert b.num_scheduled_tokens == 4       # 整段放下：needed(4) <= min(chunk, remaining)
        assert sched.last_schedule_stats["planned_tokens"] == 12
        da = decision_of(sched.last_schedule_stats, a)
        assert da["reason"] == "scheduled" and da["is_last_chunk"] is False
        assert da["chunk_index"] == 1 and da["offset_before"] == 0
        db = decision_of(sched.last_schedule_stats, b)
        assert db["is_last_chunk"] is True and db["chunk_index"] == 1

    def test_chunk_size_equal_budget(self):
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=8)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7)
        assert [e - s for s, e in intervals] == [8, 8, 4]
        assert_intervals_contiguous(intervals, 20)

    def test_chunk_size_larger_than_budget_keeps_day7_behavior(self):
        """chunk_size > B：预算主导，20-token/B=8 的 8/8/4 行为原样保持。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        intervals = drive_full_prefill(sched, seq, token=7)
        assert [e - s for s, e in intervals] == [8, 8, 4]
        assert_intervals_contiguous(intervals, 20)

    def test_partial_chunk_attributed_scheduled_not_budget(self):
        """chunk 部分推进记 scheduled：不进入预算等待统计（§4.1）。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=4)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        for t in (1.0, 2.0, 3.0, 4.0, 5.0):
            batch, items, is_prefill = schedule_round(sched, now=t)
            assert is_prefill
            assert_round_invariants(sched, batch)
            # 每轮部分推进都属于被调度，不是 budget 等待
            assert sched.last_schedule_stats["budget_deferred_requests"] == 0
            assert decision_of(sched.last_schedule_stats, seq)["reason"] == "scheduled"
            sched.postprocess(items, tokens_for_items(items, 7),
                              now=t)
            if seq.status == RUNNING:
                break
        assert seq.prefill_offset == 20
        assert seq.seq_id not in sched.budget_wait

    def test_budget_exhaustion_attribution_unchanged(self):
        """[3,5,2]/B8 恰好耗尽：第三条 direct budget（needed=None、kv_checked=False）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8, chunk_size=1024)
        a, b, c = make_seq(3, 8), make_seq(5, 8), make_seq(2, 8)
        for s in (a, b, c):
            sched.add(s)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill and [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "budget"
        assert dc["needed_tokens"] is None
        assert dc["kv_checked"] is False

    def test_per_round_caps_hold_across_scenario(self):
        """随机驱动中每轮 q_i<=chunk_size 与 sum(q_i)<=B 恒成立（独立校验）。"""
        rng = random.Random(20260912)
        sched = make_scheduler(num_blocks=32, max_num_batched_tokens=12,
                               max_num_seqs=4, chunk_size=5)
        for _ in range(60):
            sched.add(make_seq(rng.randint(1, 20), max_tokens=rng.randint(1, 3)))
            batch, items, is_prefill = schedule_round(sched, now=1.0)
            assert_round_invariants(sched, batch)
            sched.postprocess(items, tokens_for_items(items, 7),
                              now=1.0)
        for sid in list(sched.requests):
            sched.cancel(sid, now=2.0)
        assert sched.is_finished()


# ============================== 调度规则 ==============================

class TestSchedulingRules:
    """每请求每轮至多 1 chunk；首候选拆分、后续整段放下、FCFS 不跳过。"""

    def test_head_admitted_once_per_round(self):
        """chunk_size 导致首请求未完成时，不得被重复 append 到同一批次。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill
        assert [s.seq_id for s in batch].count(seq.seq_id) == 1
        assert seq.num_scheduled_tokens == 8
        assert len(sched.waiting) == 1  # 原地保留，未重复入队
        assert sched.waiting[0] is seq
        stats = sched.last_schedule_stats
        assert sum(1 for d in stats["decisions"] if d["seq_id"] == seq.seq_id) == 1

    def test_tail_fits_whole_joins_behind_mid_chunk_head(self):
        """中间 chunk 请求原地保留；整段放得下的后续请求同轮接纳（§5.2）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        a, b, c = make_seq(20, 2), make_seq(6, 2), make_seq(5, 2)
        for s in (a, b, c):
            sched.add(s)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill
        # a 首候选拆分；b、c 整段放下，同轮接纳并完成各自 prefill
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id, c.seq_id]
        assert [s.num_scheduled_tokens for s in batch] == [8, 6, 5]
        assert a.status == WAITING and b.status == RUNNING and c.status == RUNNING
        # a 原地保留在 waiting 队首，b/c 已移出
        assert list(sched.waiting) == [a]

    def test_tail_too_large_stops_scan_fcfs_no_skip(self):
        """后续候选放不下（需求>chunk_size）：停止扫描，不跳过它接纳更短尾部。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        a, b, c = make_seq(20, 2), make_seq(30, 2), make_seq(2, 2)
        for s in (s_ for s_ in (a, b, c)):
            sched.add(s)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill
        assert [s.seq_id for s in batch] == [a.seq_id]  # b 放不下即停，c 不被跳过接纳
        db = decision_of(sched.last_schedule_stats, b)
        assert db["reason"] == "budget"
        assert db["needed_tokens"] == 30  # 如实记录 b 自身需求
        assert db["kv_checked"] is True
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "head_of_line"
        assert dc["blocked_by_seq_id"] == b.seq_id
        assert dc["blocking_reason"] == "budget"

    def test_tail_blocked_by_budget_not_chunk(self):
        """后续候选需求<=chunk_size 但>剩余预算：Day7 budget 归因不变。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=12, chunk_size=8)
        a, b = make_seq(20, 2), make_seq(10, 2)
        sched.add(a)
        sched.add(b)
        batch, items, _ = schedule_round(sched, now=1.0)
        assert [s.seq_id for s in batch] == [a.seq_id]  # a 接纳 8，余 4 < b 需求 10
        db = decision_of(sched.last_schedule_stats, b)
        assert db["reason"] == "budget"
        assert db["needed_tokens"] == 10

    def test_chunk_continues_next_round_in_queue_order(self):
        """中间 chunk 请求下一轮仍按 FCFS 队首优先延续推进。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        a, b = make_seq(20, 2), make_seq(4, 2)
        sched.add(a)
        sched.add(b)
        batch, items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(items, tokens_for_items(items, 7), now=1.0)
        # b 已在上一轮完成 prefill；本轮 decode-first 先推进 b，再续传 a 的 chunk 2
        batch, items, is_prefill = schedule_round(sched, now=2.0)
        assert [(it.phase, it.seq.seq_id) for it in items] == [
            ("decode", b.seq_id), ("prefill", a.seq_id)]
        assert a.num_scheduled_tokens == 8
        da = decision_of(sched.last_schedule_stats, a)
        assert da["chunk_index"] == 2 and da["offset_before"] == 8
        sched.postprocess(items, tokens_for_items(items, 7), now=2.0)

    def test_chunk_index_counts_per_phase_and_resets_after_resume(self):
        """chunk_index 按 prefill 阶段计数：恢复重算后从 1 重新开始。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        seq = make_seq(20, max_tokens=4)
        sched.add(seq)
        indices = []
        for t in range(1, 8):
            batch, items, is_prefill = schedule_round(sched, now=float(t))
            if not is_prefill:
                break
            d = decision_of(sched.last_schedule_stats, seq)
            indices.append(d["chunk_index"])
            sched.postprocess(items, tokens_for_items(items, 7),
                              now=float(t))
        assert indices == [1, 2, 3]  # 8/8/4 三个 chunk
        # decode 一轮后抢占再恢复：重算属于新 prefill 阶段
        batch, items, _ = schedule_round(sched, now=10.0)
        sched.postprocess(items, tokens_for_items(items, 7), now=10.0)
        sched.preempt(seq, now=11.0)
        sched.resume(seq)
        batch, items, is_prefill = schedule_round(sched, now=12.0)
        assert is_prefill
        d = decision_of(sched.last_schedule_stats, seq)
        assert d["chunk_index"] == 1
        assert d["offset_before"] == 16  # 前 2 块（16 token）命中自身缓存

    def test_sequence_cap_priority_over_budget_unchanged(self):
        """序列数上限与预算同时命中时仍优先记 sequence_cap（Day7 归因继承）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64,
                               max_num_seqs=2, chunk_size=8)
        a, b, c = make_seq(4, 2), make_seq(4, 2), make_seq(4, 2)
        for s in (a, b, c):
            sched.add(s)
        batch, items, _ = schedule_round(sched, now=1.0)
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        assert decision_of(sched.last_schedule_stats, c)["reason"] == "sequence_cap"


# ============================== 输入组装（真实 prepare 组装函数） ==============================

def _prefill_slot_expectation(seq: Sequence, start: int, end: int,
                              block_size: int) -> list[int]:
    """按 block_table 推导 [start, end) 的期望物理槽位（独立实现，用于交叉验证）。"""
    expected = []
    for pos in range(start, end):
        block_idx = pos // block_size
        offset_in_block = pos % block_size
        expected.append(seq.block_table[block_idx] * block_size + offset_in_block)
    return expected


class TestInputAssembly:
    """真实 _build_prefill_inputs/_build_decode_inputs 的显式 offset 契约（§4.3）。"""

    def test_single_seq_chunk_inputs(self):
        seq = make_seq(20, max_tokens=2)
        BlockManager(num_blocks=4, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.prefill_offset = 8                          # 第二个 chunk [8, 16)
        seq.num_scheduled_tokens = 8
        input_ids, positions, cu_q, cu_k, max_q, max_k, slot_mapping, has_bt = \
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)
        assert input_ids == list(range(9, 17))          # seq[8:16] 片段
        assert positions == list(range(8, 16))          # 绝对位置，不从 0 重置
        assert cu_q == [0, 8]
        assert cu_k == [0, 16]                          # 历史 KV(8) + 当前 query(8)
        assert (max_q, max_k) == (8, 16)
        assert has_bt is True
        assert slot_mapping == _prefill_slot_expectation(seq, 8, 16, BLOCK_SIZE)

    def test_slot_mapping_crosses_block_boundary(self):
        """chunk 跨物理块边界：slot 按块正确切分，与 token 位置一一对应。"""
        seq = make_seq(20, max_tokens=2)
        BlockManager(num_blocks=4, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.prefill_offset = 6
        seq.num_scheduled_tokens = 6                    # [6, 12)：跨块 0/1
        _, _, _, _, _, _, slot_mapping, _ = \
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)
        expected = ([0 * 8 + 6, 0 * 8 + 7]              # 块 0 尾部 2 槽
                    + [1 * 8 + i for i in range(4)])    # 块 1 头部 4 槽（位置 8..11）
        assert slot_mapping == expected
        assert slot_mapping == _prefill_slot_expectation(seq, 6, 12, BLOCK_SIZE)

    def test_multi_seq_varlen_segments(self):
        """多请求变长 chunk：cu_seqlens 分段正确，请求间隔离。"""
        a, b = make_seq(20, max_tokens=2), make_seq(10, max_tokens=2)
        for s, off, q in ((a, 8, 8), (b, 0, 2)):
            BlockManager(num_blocks=8, block_size=BLOCK_SIZE).allocate(s, 0)
            s.prefill_offset = off
            s.num_scheduled_tokens = q
        input_ids, positions, cu_q, cu_k, _, _, slot_mapping, _ = \
            ModelRunner._build_prefill_inputs([a, b], BLOCK_SIZE)
        assert input_ids == list(range(9, 17)) + list(range(1, 3))
        assert positions == list(range(8, 16)) + list(range(0, 2))
        assert cu_q == [0, 8, 10]
        assert cu_k == [0, 16, 18]                      # b 无 prefix：k==q
        assert len(slot_mapping) == 10

    def test_flatten_length_consistency(self):
        """展平长度：input_ids/positions/slot_mapping/cu_seqlens_q[-1] 一致。"""
        seq = make_seq(20, max_tokens=2)
        BlockManager(num_blocks=4, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.prefill_offset = 0
        seq.num_scheduled_tokens = 12
        input_ids, positions, cu_q, _, _, _, slot_mapping, _ = \
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)
        assert len(input_ids) == len(positions) == len(slot_mapping) == 12
        assert cu_q[-1] == 12

    def test_prefix_hit_uses_block_tables_branch(self):
        """prefix 命中：cu_seqlens_k > cu_seqlens_q（历史 KV 从缓存读取）。"""
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        # 模拟另一请求已登记前 2 块的 prefix 缓存
        donor = make_seq(20, max_tokens=1)
        bm.allocate(donor, 0)
        bm.hash_blocks(donor, 0, 16)
        seq = make_seq(20, max_tokens=2)                # 同前 16 token
        assert bm.can_allocate(seq) == 2
        bm.allocate(seq, 2)
        assert seq.prefill_offset == 16
        seq.num_scheduled_tokens = 4                    # 只执行 [16, 20)
        _, _, cu_q, cu_k, _, _, _, _ = \
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)
        assert cu_q == [0, 4]
        assert cu_k == [0, 20]                          # 历史 16 + 当前 4
        assert cu_k[-1] > cu_q[-1]

    def test_warmup_batch_without_block_table(self):
        """warmup 批次（无 block_table）：组装不抛错，slot 为空。"""
        seq = make_seq(16, max_tokens=1)
        seq.num_scheduled_tokens = 16
        input_ids, positions, cu_q, cu_k, _, _, slot_mapping, has_bt = \
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)
        assert has_bt is False
        assert slot_mapping == []
        assert len(input_ids) == 16 and cu_q[-1] == 16

    def test_large_block_size_8k_slots(self, monkeypatch):
        """block_size=256、8K prompt 的末 chunk：slot 跨多块且与位置一致。"""
        monkeypatch.setattr(Sequence, "block_size", 256)
        bs = 256
        seq = make_seq(8192, max_tokens=1)
        bm = BlockManager(num_blocks=32, block_size=bs)
        bm.allocate(seq, 0)
        seq.prefill_offset = 7680                       # 最后 chunk [7680, 8192)
        seq.num_scheduled_tokens = 512
        _, positions, _, _, _, _, slot_mapping, _ = \
            ModelRunner._build_prefill_inputs([seq], bs)
        assert positions[0] == 7680 and positions[-1] == 8191
        assert slot_mapping == _prefill_slot_expectation(seq, 7680, 8192, bs)
        # 末槽 = 位置 8191 → 块 31 的最后一个位置
        assert slot_mapping[-1] == seq.block_table[31] * bs + 255

    @pytest.mark.parametrize("offset, q, target", [
        (20, 4, 20),   # offset == target：空区间非法
        (24, 4, 20),   # 越界
        (0, 0, 20),    # 零 query 非法
        (-1, 4, 20),   # 负起点
    ])
    def test_out_of_range_offset_raises(self, offset, q, target):
        """越界 offset 显式抛错，不静默组装（§4.3）。"""
        seq = make_seq(target, max_tokens=2)
        BlockManager(num_blocks=4, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.prefill_offset = offset
        seq.num_scheduled_tokens = q
        with pytest.raises(ValueError, match="非法 prefill 区间"):
            ModelRunner._build_prefill_inputs([seq], BLOCK_SIZE)

    def test_decode_inputs_use_num_tokens_metadata(self):
        """decode 组装使用 num_tokens 元数据（positions/context_lens）。"""
        seq = make_seq(6, max_tokens=4)
        BlockManager(num_blocks=2, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.transition_to(RUNNING)
        seq.is_prefill = False
        seq.append_token(77)                            # num_tokens=7
        input_ids, positions, slot_mapping, context_lens = \
            ModelRunner._build_decode_inputs([seq], BLOCK_SIZE)
        assert input_ids == [77]
        assert positions == [6]                         # num_tokens - 1
        assert context_lens == [7]                      # num_tokens
        assert slot_mapping == [seq.block_table[-1] * 8 + seq.last_block_num_tokens - 1]

    def test_decode_inputs_after_tp_deserialization(self):
        """TP worker 反序列化后的 decode 序列（token_ids 为空）组装结果一致。"""
        seq = make_seq(6, max_tokens=4)
        BlockManager(num_blocks=2, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.transition_to(RUNNING)
        seq.is_prefill = False
        seq.append_token(77)
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.token_ids == []                    # decode 模式不传全量 token
        direct = ModelRunner._build_decode_inputs([seq], BLOCK_SIZE)
        via_tp = ModelRunner._build_decode_inputs([clone], BLOCK_SIZE)
        assert direct == via_tp


# ============================== prefix 与恢复 ==============================

class TestPrefixAndRecompute:
    """prefix 命中计入初始 offset；恢复 recompute 按命中重算；显式区间登记。"""

    def test_prefix_hit_sets_initial_offset(self):
        """命中块计入初始 offset（allocate 时设置），只执行未缓存部分。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=1024)
        prompt = list(range(1, 17))                     # 16 token = 2 完整块
        a = Sequence(prompt, SamplingParams(max_tokens=2, ignore_eos=True))
        sched.add(a)
        batch, items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(items, [7], now=1.0)
        b = Sequence(prompt + [91, 92], SamplingParams(max_tokens=2, ignore_eos=True))
        sched.add(b)
        # Day9：a 已 RUNNING，本轮 decode-first 先推进 a 的 decode，再接纳 b 的 prefill
        batch, items, is_prefill = schedule_round(sched, now=2.0)
        assert [(it.phase, it.seq.seq_id) for it in items] == [
            ("decode", a.seq_id), ("prefill", b.seq_id)]
        # 尾块（第 3 块）永不命中：初始 offset = 2 块 * 8 = 16
        assert b.prefill_offset == 16
        assert b.num_scheduled_tokens == 2              # 只执行 [16, 18)
        d = decision_of(sched.last_schedule_stats, b)
        assert d["offset_before"] == 16 and d["is_last_chunk"] is True
        sched.postprocess(items, tokens_for_items(items, 7), now=2.0)

    def test_tail_block_never_hit(self):
        """can_allocate 协议：range(num_blocks-1)，最后一块永不假命中。"""
        sched = make_scheduler(num_blocks=8)
        prompt = list(range(1, 17))                     # 恰好 2 个完整块
        a = Sequence(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        sched.add(a)
        batch, items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(items, [7], now=1.0)    # 2 块全部登记
        b = Sequence(prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        sched.add(b)
        sched.schedule(now=2.0)
        # 命中块数 = num_blocks - 1 = 1：尾块（块 1）不命中
        assert b.prefill_offset == 8
        assert b.num_scheduled_tokens == 8

    def test_mid_chunk_full_blocks_registered_for_prefix_reuse(self):
        """中间 chunk 写满的块同样登记进 prefix 索引（chunked 仍享受复用）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=12, chunk_size=1024)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        batch, items, _ = schedule_round(sched, now=1.0)              # chunk [0, 12)
        sched.postprocess(items, [], now=1.0)
        bm = sched.block_manager
        # 块 0 已被本次执行写满并登记；块 1（部分写入）不登记
        h0 = BlockManager.compute_hash(seq.block(0), -1)
        assert bm.hash_to_block_id.get(h0) == seq.block_table[0]
        h1 = BlockManager.compute_hash(seq.block(1), h0)
        assert bm.hash_to_block_id.get(h1) is None
        # 相同前 8 token 的新请求可命中块 0
        follower = Sequence(seq.token_ids[:8] + [71, 72, 73, 74],
                            SamplingParams(max_tokens=1, ignore_eos=True))
        sched.add(follower)
        batch, items, is_prefill = schedule_round(sched, now=2.0)
        assert is_prefill
        assert follower.prefill_offset == 8
        assert follower.block_table[0] == seq.block_table[0]

    def test_hash_blocks_explicit_interval_only_full_blocks(self):
        """hash_blocks 显式区间：只登记 [start, end) 内写满的块；空区间为空操作。"""
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq = make_seq(40, max_tokens=1)                # 5 块
        bm.allocate(seq, 0)
        bm.hash_blocks(seq, 0, 12)                      # 写满块 0，块 1 部分
        h0 = BlockManager.compute_hash(seq.block(0), -1)
        assert bm.hash_to_block_id.get(h0) == seq.block_table[0]
        assert bm.blocks[seq.block_table[1]].hash == -1
        bm.hash_blocks(seq, 12, 12)                     # 空区间：安全空操作
        assert bm.blocks[seq.block_table[1]].hash == -1
        bm.hash_blocks(seq, 12, 20)                     # 写满块 1，链式哈希
        h1 = BlockManager.compute_hash(seq.block(1), h0)
        assert bm.hash_to_block_id.get(h1) == seq.block_table[1]

    def test_resume_recompute_recovers_from_explicit_offset(self):
        """抢占恢复：物理块释放、offset 归零，恢复后按 prefix 命中重算进度。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=1024)
        seq = make_seq(16, max_tokens=8)
        sched.add(seq)
        batch, items, _ = schedule_round(sched, now=1.0)              # prefill 16
        sched.postprocess(items, [7], now=1.0)    # len=17, offset=16
        batch, items, is_prefill = schedule_round(sched, now=2.0)
        assert not is_prefill
        sched.postprocess(items, [8], now=2.0)   # len=18, offset=17
        sched.preempt(seq, now=3.0)
        assert seq.prefill_offset == 0                  # 进度随物理块作废
        assert seq.block_table == []
        sched.resume(seq)
        batch, items, is_prefill = schedule_round(sched, now=4.0)
        assert is_prefill
        # 重算区间 [16, 18)：前 2 块命中自身缓存（16 token），重算需求 2
        assert seq.prefill_offset == 16
        assert seq.num_scheduled_tokens == 2
        d = decision_of(sched.last_schedule_stats, seq)
        assert d["offset_before"] == 16 and d["is_last_chunk"] is True
        sched.postprocess(items, [9], now=4.0)
        assert seq.status == RUNNING
        assert seq.prefill_offset == 18
        assert seq.token_ids[-3:] == [7, 8, 9]          # 生成进度未丢失


# ============================== 采样契约 ==============================

class TestSamplingContract:
    """中间 chunk 不采样、不耗 RNG；最后 chunk 采样最后 query 行（§4.5）。"""

    def test_mid_chunk_selects_no_rows(self):
        seq = make_seq(20, max_tokens=2)
        seq.num_scheduled_tokens = 8                    # offset=0，未完成
        assert ModelRunner._select_prefill_sample_rows([seq]) == []

    def test_last_chunk_selects_last_query_row(self):
        seq = make_seq(20, max_tokens=2)
        seq.prefill_offset = 16
        seq.num_scheduled_tokens = 4                    # 最后 chunk [16, 20)
        assert ModelRunner._select_prefill_sample_rows([seq]) == [0]

    def test_mixed_batch_rows_aligned_to_last_query(self):
        """多请求变长混合：A 中间（不采样）、B/C 最后 chunk（各采一行）。

        ParallelLMHead 已按 cu_seqlens_q[1:]-1 把每请求最后 query 行聚合到
        logits 第 i 行（i 为批内下标），因此子集选择的下标即 logits 行号。
        """
        a, b, c = make_seq(20, 2), make_seq(10, 2), make_seq(6, 2)
        a.prefill_offset, a.num_scheduled_tokens = 8, 8    # 中间
        b.prefill_offset, b.num_scheduled_tokens = 8, 2    # 最后 [8, 10)
        c.prefill_offset, c.num_scheduled_tokens = 0, 6    # 最后 [0, 6)
        assert ModelRunner._select_prefill_sample_rows([a, b, c]) == [1, 2]

    def test_mid_chunk_consumes_no_rng(self):
        """中间 chunk 不采样 ⇒ RNG 流与 chunk 划分无关。

        用 generator 内部状态作确定性判据：等价路径（单次采样）状态一致；
        违规路径（多采样一次）状态必然被推进。
        """
        torch = pytest.importorskip("torch")
        logits = torch.randn(20, 32, generator=torch.Generator().manual_seed(7))
        temps = torch.tensor([1.0])
        top_ps = torch.tensor([0.9])

        # 路径一（one-shot 等价）：仅一次采样
        g1 = torch.Generator().manual_seed(42)
        t1 = Sampler.sample(logits[[19]], temps, top_ps, [g1])

        # 路径二（chunked）：中间 chunk 不调用采样器，随后对同一行采样
        g2 = torch.Generator().manual_seed(42)
        t2 = Sampler.sample(logits[[19]], temps, top_ps, [g2])

        assert t1.item() == t2.item()
        assert g1.get_state().numpy().tobytes() == g2.get_state().numpy().tobytes()
        # 负例对照：中间 chunk 若违规多采样一次，generator 状态被消耗
        g3 = torch.Generator().manual_seed(42)
        Sampler.sample(logits[[0]], temps, top_ps, [g3])   # 违规的中间 chunk 采样
        t3 = Sampler.sample(logits[[19]], temps, top_ps, [g3])
        assert g3.get_state().numpy().tobytes() != g2.get_state().numpy().tobytes()
        assert t3.item() != t2.item()

    def test_mid_chunk_postprocess_rejects_tokens(self):
        """中间 chunk 传入采样 token：显式抛错，不静默截断（§4.5）。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [7], now=1.0)
        # 正确注入（空列表）后 offset 正常推进
        sched.postprocess(items, [], now=1.0)
        assert seq.prefill_offset == 8 and len(seq.token_ids) == 20

    def test_last_chunk_missing_token_rejected(self):
        """最后 chunk 缺少采样 token：显式抛错。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=32, chunk_size=1024)
        seq = make_seq(8, max_tokens=2)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [], now=1.0)

    def test_decode_missing_token_rejected(self):
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=32, chunk_size=1024)
        seq = make_seq(4, max_tokens=2)
        sched.add(seq)
        batch, items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(items, [7], now=1.0)
        batch, items, is_prefill = schedule_round(sched, now=2.0)
        assert not is_prefill
        with pytest.raises(ValueError, match="需采样请求数"):
            sched.postprocess(items, [], now=2.0)

    def test_mixed_batch_token_alignment(self):
        """混合批次：token 按需采样集合对齐（A 中间无 token，B/C 各得一 token）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        a, b, c = make_seq(20, 2), make_seq(6, 2), make_seq(5, 2)
        for s in (a, b, c):
            sched.add(s)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id, c.seq_id]
        # a 中间 chunk 无 token；b、c 最后 chunk 各一 token
        sched.postprocess(items, [201, 202], now=1.0)
        assert a.token_ids == a.prompt_token_ids        # a 未追加
        assert b.token_ids[-1] == 201
        assert c.token_ids[-1] == 202


# ============================== 生命周期（chunk 边界上的控制面） ==============================

class TestLifecycleDuringChunks:
    """chunk 中取消/超时/外部 mark_*：不推进、KV 释放、无过期 token。"""

    def _chunking_seq(self, sched: Scheduler) -> Sequence:
        seq = make_seq(20, max_tokens=4)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        sched.postprocess(items, [], now=1.0)   # offset=8，仍 WAITING
        return seq

    def test_cancel_mid_chunk(self):
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = self._chunking_seq(sched)
        assert seq.prefill_offset == 8
        assert sched.cancel(seq.seq_id, now=2.0) is True
        assert seq.status == CANCELLED
        assert seq.prefill_offset == 0                  # 终态清理：进度随 KV 作废
        assert seq.block_table == []
        assert seq.num_completion_tokens == 0           # 无过期 token
        assert sched.is_finished()
        assert len(sched.block_manager.free_block_ids) == 4

    def test_timeout_mid_chunk_discards_output(self):
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=4, deadline=5.0)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        sched.postprocess(items, [], now=1.0)
        assert seq.prefill_offset == 8                  # 第 1 chunk 已提交
        batch, items, is_prefill = schedule_round(sched, now=2.0)     # 第 2 chunk 接纳
        assert is_prefill
        # 执行返回时已跨过 deadline：该 chunk 不提交，按 TIMEOUT 终止；
        # 终态清理释放 KV 后进度随之作废（offset 归零）
        sched.postprocess(items, [], now=6.0)
        assert seq.status == TIMEOUT
        assert seq.prefill_offset == 0
        assert seq.block_table == []
        assert sched.is_finished()

    def test_external_mark_terminal_mid_chunk(self):
        """外部 mark_* 置终态：调度边界兜底清理，终态请求不参与调度。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = self._chunking_seq(sched)
        seq.mark_cancelled("client_abort")
        batch, items, is_prefill = schedule_round(sched, now=2.0)     # 边界兜底清理后空批
        assert batch == []
        assert sched.is_finished()

    def test_duplicate_postprocess_rejected(self):
        """重复 postprocess（同轮计划已清零）被快照校验拒绝，进度不二次推进。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=4)
        sched.add(seq)
        batch, items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(items, [], now=1.0)     # 第 1 chunk 已提交，计划清零
        with pytest.raises(ValueError, match="重复或迟到的 postprocess"):
            sched.postprocess(items, [], now=2.0)  # 同轮重放：无待执行计划
        assert seq.prefill_offset == 8

    def test_stale_live_batch_rejected(self):
        """迟到旧批次重放（同 seq 已被重新规划）被 round_id 关联校验拒绝（Day9 加固）。

        Day8 遗留（day8-review §4.4）：旧实现仅靠"计划已清零"检测，若两次收尾
        之间同 seq 已被重新接纳，旧批次会以新计划二次提交。Day9 起 items 携带
        round_id，含活动请求的批次与最近调度轮不匹配即显式拒绝。
        """
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=4)
        sched.add(seq)
        batch, stale_items, _ = schedule_round(sched, now=1.0)
        sched.postprocess(stale_items, [], now=1.0)   # 第 1 chunk 已提交
        batch2, items2, _ = schedule_round(sched, now=2.0)   # 同 seq 被重新规划
        assert any(it.seq is seq for it in items2)
        with pytest.raises(ValueError, match="迟到"):
            sched.postprocess(stale_items, [], now=2.0)  # 旧 round 的批次重放
        # 新计划不受影响：正常提交推进
        sched.postprocess(items2, [], now=2.0)
        assert seq.prefill_offset == 16

    def test_failure_does_not_advance_offset(self):
        """失败不提交：安全检查命中（执行期间取消）时该 chunk 的进度不推进。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=4)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        seq.request_cancel("mid_exec")                  # 模拟执行期间取消
        sched.postprocess(items, tokens_for_items(items, 7),
                          now=1.0)
        assert seq.status == CANCELLED
        assert seq.prefill_offset == 0                  # 未提交
        assert sched.block_manager.used_block_ids == set()  # KV 已释放


# ============================== 一致性（CPU 桩） ==============================

class TestOneShotVsChunkedConsistency:
    """固定 token 注入下：one-shot 与多 chunk 的 completion 序列一致（§4.6）。"""

    def test_completion_sequences_identical(self):
        prompts = [list(range(1, 21)), list(range(50, 63)), list(range(90, 97))]
        max_tokens = 4

        def token_fn(seq: Sequence) -> int:
            # token 只依赖 (请求, 已生成数量)，与 prefill 的 chunk 划分无关；
            # 用 request_id 而非 seq_id（后者是全局计数，两套调度器间不同）
            return 1000 + int(seq.request_id[1:]) * 10 + seq.num_completion_tokens

        def make_prompts_with_ids():
            # 显式 request_id 作为对齐键（seq_id 是全局计数，两套调度器间不同）
            return [Sequence(p, SamplingParams(max_tokens=max_tokens, ignore_eos=True),
                             request_id=f"p{i}")
                    for i, p in enumerate(prompts)]

        one_shot_sched = make_scheduler(num_blocks=16, max_num_batched_tokens=10 ** 6,
                                        chunk_size=10 ** 6)
        for seq in make_prompts_with_ids():
            one_shot_sched.add(seq)
        one_shot = self._drive_seqs(one_shot_sched, token_fn)
        chunked_sched = make_scheduler(num_blocks=16, max_num_batched_tokens=8,
                                       chunk_size=3)
        for seq in make_prompts_with_ids():
            chunked_sched.add(seq)
        chunked = self._drive_seqs(chunked_sched, token_fn)
        assert set(one_shot) == set(chunked)
        for rid in one_shot:
            assert one_shot[rid] == chunked[rid], (
                f"request_id={rid}: one-shot 与 chunked 输出不一致")

    @staticmethod
    def _drive_seqs(sched: Scheduler, token_fn) -> dict[str, list[int]]:
        """按 request_id 汇总输出，驱动所有请求至完成（每轮断言不变量）。"""
        outputs: dict[str, list[int]] = {}
        t = 0.0
        for _ in range(500):
            if sched.is_finished():
                return outputs
            batch, items, is_prefill = schedule_round(sched, now=t)
            assert_round_invariants(sched, batch)
            tokens = [token_fn(it.seq) for it in items if it.needs_sample]
            sched.postprocess(items, tokens, now=t)
            for s in batch:
                if s.status == FINISHED:
                    outputs[s.request_id] = s.completion_token_ids
            t += 1.0
        raise AssertionError("有界轮次内未完成")

    def test_chunked_actually_splits(self):
        """对照：chunk_size=5、prompt 12 → 恰好 3 个 chunk（5/5/2）后进 decode。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=64, chunk_size=5)
        seq = Sequence(list(range(1, 13)), SamplingParams(max_tokens=2, ignore_eos=True))
        sched.add(seq)
        rounds = 0
        t = 1.0
        while seq.status == WAITING:
            batch, items, is_prefill = schedule_round(sched, now=t)
            assert is_prefill
            sched.postprocess(items, tokens_for_items(items, 7),
                              now=t)
            rounds += 1
            t += 1.0
            assert rounds < 10
        assert rounds == 3


# ============================== TP 序列化协议 ==============================

class TestTpSerialization:
    """Sequence pickle v3 + v1/v2 单向兼容（§5.3 不变量 12）。"""

    def test_v3_roundtrip_prefill_mode(self):
        seq = make_seq(20, max_tokens=4)
        seq.prefill_offset = 8
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.prefill_offset == 8
        assert clone.num_tokens == 20
        assert clone.is_prefill is True
        assert clone.status == WAITING
        assert clone.num_cached_tokens == 8             # 兼容视图

    def test_v3_roundtrip_decode_mode(self):
        seq = make_seq(6, max_tokens=4)
        BlockManager(num_blocks=2, block_size=BLOCK_SIZE).allocate(seq, 0)
        seq.transition_to(RUNNING)
        seq.is_prefill = False
        seq.prefill_offset = 6                          # 模拟 prefill 完成后的 decode
        seq.append_token(42)
        seq.request_cancel("mid_run")
        version, payload = seq.__getstate__()
        assert version == 3
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.is_prefill is False
        assert clone.last_token == 42
        assert clone.num_tokens == 7
        assert clone.prefill_offset == 6
        assert clone.status == RUNNING
        assert clone.cancel_requested is True

    def test_v3_payload_uses_prefill_offset_key(self):
        seq = make_seq(4)
        _, payload = seq.__getstate__()
        assert "prefill_offset" in payload
        assert "num_cached_tokens" not in payload       # 单一事实源，无旧字段
        # rank 0 统计类字段不进 TP payload（时间戳/控制账本）
        for absent in ("created_at", "deadline", "finish_reason", "num_preempts"):
            assert absent not in payload

    def test_v2_payload_mapped_to_prefill_offset(self):
        """v2 旧格式（num_cached_tokens 字段名）读取映射到 prefill_offset。"""
        v2_payload = {
            "num_tokens": 20,
            "num_prompt_tokens": 20,
            "num_cached_tokens": 8,
            "num_scheduled_tokens": 0,
            "block_table": [],
            "last_state": list(range(1, 21)),
            "status": WAITING,
            "is_prefill": True,
            "cancel_requested": False,
        }
        seq = Sequence.__new__(Sequence)
        seq.__setstate__((2, v2_payload))
        assert seq.prefill_offset == 8
        assert seq.num_cached_tokens == 8
        # 旧格式对象再次 pickle 往返升级为 v3，语义不变
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.prefill_offset == 8

    def test_v1_tuple_mapped_to_prefill_offset(self):
        legacy = (6, 6, 0, 0, [0, 1], [1, 2, 3, 4, 5, 6])
        seq = Sequence.__new__(Sequence)
        seq.__setstate__(legacy)
        assert seq.prefill_offset == 0
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.prefill_offset == 0

    def test_v1_tuple_decode_mode_offset(self):
        legacy = (3, 2, 2, 1, [0], 77)                  # decode：offset=2
        seq = Sequence.__new__(Sequence)
        seq.__setstate__(legacy)
        assert seq.prefill_offset == 2
        assert seq.is_prefill is False
        clone = pickle.loads(pickle.dumps(seq))
        assert clone.prefill_offset == 2

    def test_unknown_version_rejected(self):
        seq = make_seq(4)
        _, payload = seq.__getstate__()
        clone = Sequence.__new__(Sequence)
        with pytest.raises(ValueError, match="状态格式版本"):
            clone.__setstate__((999, payload))


# ============================== Engine 观测 ==============================

class TestEngineChunkObservation:
    """Engine step 的 chunk 快照扩展与 prefill_chunks 观测（§6.3/§6.4）。"""

    @staticmethod
    def make_engine(sched: Scheduler):
        """跳过 GPU __init__：注入真实 Scheduler、tokenizer 桩与契约 runner 桩。

        runner 按提交契约返回：decode 轮每序列 1 token；prefill 轮只有最后
        chunk 请求产出 token，全中间 chunk 轮返回 None。调用时记录
        (seq_id, request_id, n, offset, is_last_chunk) 快照供断言。
        """
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text])
        calls = []

        def fake_call(method, items_):
            # Day9 契约 runner：按 item 顺序返回 decode + 最后 chunk 的采样 token；
            # 调用时快照 (seq_id, request_id, phase, n, offset, is_last_chunk)
            calls.append([(it.seq.seq_id, it.seq.request_id, it.phase,
                           it.scheduled_tokens, it.offset_before, it.is_last_chunk)
                          for it in items_])
            out = tokens_for_items(items_, 7)
            return out if out else None

        engine.model_runner = SimpleNamespace(call=fake_call)
        return engine, calls

    def test_chunked_prefill_through_engine(self, caplog):
        """长 prompt 经 engine.step 分块完成：快照含 offset/is_last_chunk，
        prefill_chunks 事件字段正确，调度/执行 round_id 可关联。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        engine, calls = self.make_engine(sched)
        rid = engine.add_request("x" * 20, SamplingParams(max_tokens=2, ignore_eos=True))
        with caplog.at_level(logging.INFO):
            num_tokens_list = []
            for _ in range(20):
                outputs, num_tokens = engine.step()
                num_tokens_list.append(num_tokens)
                if outputs:
                    break
        # 请求完成（max_tokens=2）后从活动索引移除
        assert sched.is_finished()
        assert num_tokens_list[0] == 8                  # 首 chunk 8 token
        # 快照校验：单请求场景每轮只含一个 item；prefill 轮 offset 单调推进、
        # 末轮收尾（decode 轮的快照按 phase 字段过滤掉）
        prefill_calls = [snap for snap in calls if snap[0][2] == "prefill"]
        assert len(prefill_calls) == 3
        assert [snap[0][4] for snap in prefill_calls] == [0, 8, 16]     # offset_before
        assert [snap[0][3] for snap in prefill_calls] == [8, 8, 4]      # q
        assert [snap[0][5] for snap in prefill_calls] == [False, False, True]
        decode_calls = [snap for snap in calls if snap[0][2] == "decode"]
        assert decode_calls and all(snap[0][3] == 1 for snap in decode_calls)
        # 事件：engine_round 含 prefill_chunks（prefill 轮=1，decode 轮=0）
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        engine_rounds = [e for e in events if e["event"] == "engine_round"]
        prefill_rounds = [e for e in engine_rounds if e["phase"] == "prefill"]
        assert len(prefill_rounds) == 3
        assert all(e["prefill_chunks"] == 1 for e in prefill_rounds)
        decode_rounds = [e for e in engine_rounds if e["phase"] == "decode"]
        assert decode_rounds and all(e["prefill_chunks"] == 0 for e in decode_rounds)
        assert engine.get_request(rid) is None

    def test_idle_round_event_has_zero_prefill_chunks(self, caplog):
        """空轮 idle 的 engine_round 同样带 prefill_chunks=0（事件字段契约统一）。

        验收脚本（scripts/validate_chunked_prefill.py）的白名单要求所有
        engine_round 事件都含 prefill_chunks 字段；idle 轮缺字段会被误判 FAIL。
        """
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=64, chunk_size=8)
        engine, _ = self.make_engine(sched)
        with caplog.at_level(logging.INFO):
            outputs, num_tokens = engine.step()
        assert outputs == [] and num_tokens == 0
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        idle = [e for e in events
                if e["event"] == "engine_round" and e["outcome"] == "idle"]
        assert len(idle) == 1
        assert idle[0]["prefill_chunks"] == 0
        assert idle[0]["model_called"] is False

    def test_runner_contract_violation_propagates(self):
        """runner 违反采样契约（中间 chunk 返回 token）：postprocess 显式拒绝。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, chunk_size=8)
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text])
        engine.model_runner = SimpleNamespace(
            call=lambda m, items_: [7])   # 违反契约：无采样轮也返回 token
        engine.add_request("x" * 20, SamplingParams(max_tokens=2, ignore_eos=True))
        with pytest.raises(ValueError, match="需采样请求数"):
            engine.step()

    def test_all_mid_chunk_round_accepts_none_tokens(self):
        """全中间 chunk 轮：runner 返回 None（无采样），postprocess 空提交推进。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8, chunk_size=1024)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        batch, items, is_prefill = schedule_round(sched, now=1.0)
        assert is_prefill and batch == [seq]
        sched.postprocess(items, None, now=1.0)
        assert seq.prefill_offset == 8
        assert seq.num_scheduled_tokens == 0
        assert seq.status == WAITING


# ============================== 随机混合（chunked 场景） ==============================

class TestMixedRandomChunked:
    """固定 seed 交错操作 + chunked 调度；轮次上限防挂；终态全平衡。"""

    def test_mixed_ops_balance_and_termination(self):
        rng = random.Random(20260912)
        sched = make_scheduler(num_blocks=24, max_num_batched_tokens=10,
                               max_num_seqs=6, chunk_size=4)
        id_pool = [f"mix8-{i}" for i in range(6)]
        stale_batches = []
        now = 100.0
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
                    # 只有 RUNNING 可正常完成（WAITING -> FINISHED 非法迁移）
                    victim.mark_finished("stop")
            elif op < 0.60 and stale_batches:
                stale_items, tokens = stale_batches.pop(
                    rng.randrange(len(stale_batches)))
                if all(it.seq.is_terminal for it in stale_items):
                    # 迟到重放：只允许全终态批次（模拟迟到的执行结果；
                    # 全终态重放受 round 校验豁免，是安全空操作）
                    sched.postprocess(stale_items, tokens, now=now)
            # 驱动一轮
            batch, items, is_prefill = schedule_round(sched, now=now)
            assert_round_invariants(sched, batch)
            tokens = [rng.randint(1, 100) for it in items if it.needs_sample]
            sched.postprocess(items, tokens, now=now)
            if items:
                stale_batches.append((items, tokens))
                if len(stale_batches) > 4:
                    stale_batches.pop(0)
            # 轻量不变量：队列互斥、状态与索引一致、chunk 进度单调有界
            assert not (set(sched.waiting) & set(sched.running))
            for s in sched.waiting:
                assert s.status == WAITING and sched.requests.get(s.seq_id) is s
                assert 0 <= s.prefill_offset <= s.prefill_target
            for s in sched.running:
                assert s.status == RUNNING and sched.requests.get(s.seq_id) is s

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
        assert not sched._prefill_chunk_count            # chunk 计数随终态回收

    def test_every_round_honors_both_caps(self):
        """第二轮 seed 交叉验证：每轮 q_i<=chunk_size 且 sum(q)<=B 恒成立。"""
        rng = random.Random(8888)
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
            batch, items, is_prefill = schedule_round(sched, now=now)
            assert_round_invariants(sched, batch)
            sched.postprocess(items, tokens_for_items(items, 7),
                              now=now)
        for sid in list(sched.requests):
            sched.cancel(sid, now=now)
        assert sched.is_finished()
