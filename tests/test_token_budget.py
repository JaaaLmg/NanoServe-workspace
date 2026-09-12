"""Day 7 验收测试：每轮 Token Budget 与 FCFS 调度（docs/token-budget.md 第 8 节）。

覆盖范围（对应文档 §8.1 测试矩阵）：
- 配置：B=1 / B<max_num_seqs 合法；0/负数/float/str/bool 拒绝；优化模式显式验证；
- 空与终止：空轮 idle、终态/取消/超时清理、健康暂停不忙循环；
- 单 prefill：小于/等于/大于 B（分块推进、中间片段丢弃采样、block 不重复分配）；
- 多 prefill：[3,5,2]/B8 刚好、[6,4,1]/B8 不跳过、prefix 命中与 resume 重算按未缓存计费；
- Decode 硬上限：多轮 prefill 造出 N>B 的 RUNNING 后只选 B 条；
- Decode 副作用：未选中请求零副作用、B=1 不为预算抢占、KV 不足按原逻辑抢占；
- FCFS 与最终完成：running [A,B,C,D]/B2 有界推进；
- 相互限制：cap 先于 budget、prefill 优先、KV 队首不足、HOL 原因；
- 统计人数/时间：direct 与 HOL 去重、唯一请求、t10/t12/t15/t20/t23 时间线、
  原因切换/取消/超时/抢占/直接 mark_*/重复清理均正确结算；
- 所有权回归：ID 复用 + 迟到 postprocess 不污染新对象；
- Engine 观测：计划与 runner 输入相等、postprocess 清零不丢证据、取消丢输出不减
  executed、runner 抛错 executed=null；
- 日志：JSON 事件可解析、round_id 唯一、执行 <=B、计数可重算、无 prompt 明文；
- 长序列混合：固定 seed 交错操作 + 轮次上限 + 资源/身份平衡。

Day8 适配说明（docs/chunked-prefill.md）：决策快照新增 chunk_index/offset_before/
is_last_chunk 三个字段（精确字典断言同步扩展）；postprocess 提交契约收紧为
"中间 chunk 不采样"——分块 prefill 的中间轮 postprocess 不再传入 token，
混合随机场景的 token 注入改为按需采样集合注入。除注入方式外，预算行为断言
（如 20-token/B=8 的 8/8/4）在默认 chunk_size=1024 下原样保持。

无 GPU / 模型依赖：Scheduler 只读取 Config 的少数字段（SimpleNamespace 构造），
LLMEngine 通过 __new__ 跳过需要 GPU 的 __init__；时间使用注入的确定值（now 关键字），
不加载权重、不 sleep。

重复执行命令：
    python -m pytest tests/test_token_budget.py -q
    python -O -m pytest tests/test_token_budget.py -q
"""

import json
import logging
import random
from types import SimpleNamespace

import pytest

from nanovllm.config import Config, validate_positive_int
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
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


# ============================== 公共构造工具 ==============================

def make_seq(num_tokens: int = 4, max_tokens: int = 8, request_id: str | None = None,
             deadline: float | None = None) -> Sequence:
    return Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        request_id=request_id,
        deadline=deadline,
    )


def make_scheduler(num_blocks: int = 8, max_num_batched_tokens: int = 10 ** 6,
                   max_num_seqs: int = 512) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def setup_running(sched: Scheduler, specs, now: float = 1.0):
    """逐条添加请求并各用一轮 prefill 转为 RUNNING（避免 setup 期间出现预算延后）。

    specs: [(prompt_len, max_tokens), ...]；要求 prompt_len <= 本轮预算，
    使每条请求一轮 prefill 完成。返回 (seqs, 下一可用时间)。
    """
    seqs = []
    t = now
    for prompt_len, max_tokens in specs:
        seq = make_seq(prompt_len, max_tokens=max_tokens)
        sched.add(seq)
        batch, is_prefill = sched.schedule(now=t)
        assert batch and is_prefill, "setup：prefill 应接纳该请求"
        sched.postprocess(batch, [7] * len(batch), True, now=t)
        assert seq.status == RUNNING
        seqs.append(seq)
        t += 1.0
    return seqs, t


def decision_of(stats: dict, seq: Sequence) -> dict:
    """从本轮调度统计中取某请求的决策快照。"""
    for d in stats["decisions"]:
        if d["seq_id"] == seq.seq_id:
            return d
    raise AssertionError(f"seq_id={seq.seq_id} 不在本轮决策中: {stats['decisions']}")


def assert_round_invariants(sched: Scheduler, batch: list[Sequence]):
    """每轮通用硬约束：0 <= planned <= B；len(batch) <= max_num_seqs；非空批次计数为正。"""
    stats = sched.last_schedule_stats
    planned = sum(seq.num_scheduled_tokens for seq in batch)
    assert stats["planned_tokens"] == planned
    assert 0 <= planned <= sched.max_num_batched_tokens
    assert len(batch) <= sched.max_num_seqs
    if batch:
        assert planned > 0
        assert all(seq.num_scheduled_tokens > 0 for seq in batch)


# ============================== 配置校验 ==============================

class TestBudgetConfigValidation:
    """B 与 max_num_seqs 的显式校验（§4.1），不依赖会被 python -O 移除的 assert。"""

    def test_validate_positive_int_accepts_positive_ints(self):
        assert validate_positive_int(1, "max_num_batched_tokens") == 1
        assert validate_positive_int(16384, "max_num_seqs") == 16384

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_reject_non_positive(self, bad):
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            validate_positive_int(bad, "max_num_batched_tokens")

    @pytest.mark.parametrize("bad", [8.0, "8", True, False, None, [8]])
    def test_reject_wrong_types_including_bool(self, bad):
        # bool 是 int 子类：isinstance 会放过 True/False，必须用 type 精确匹配拒绝
        with pytest.raises(ValueError) as exc_info:
            validate_positive_int(bad, "max_num_batched_tokens")
        # 错误信息带字段名和实际值，便于定位配置错误
        assert "max_num_batched_tokens" in str(exc_info.value)
        assert repr(bad) in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0, -5, 2.5, True])
    def test_reject_invalid_max_num_seqs(self, bad):
        with pytest.raises(ValueError, match="max_num_seqs"):
            validate_positive_int(bad, "max_num_seqs")

    def test_scheduler_direct_construction_validates(self):
        """Scheduler 面向 SimpleNamespace 直接构造路径复用同一校验入口，且先于资源账本创建。"""
        base = dict(eos=EOS, kvcache_block_size=BLOCK_SIZE, num_kvcache_blocks=4)
        for field, bad in [("max_num_batched_tokens", 0), ("max_num_batched_tokens", True),
                           ("max_num_seqs", 0), ("max_num_seqs", 1.5)]:
            config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16, **base)
            setattr(config, field, bad)
            with pytest.raises(ValueError, match=field):
                Scheduler(config)

    @staticmethod
    def make_model_dir(tmp_path):
        """构造最小本地模型目录（只含 config.json），供真实 Config 数据类校验使用。"""
        d = tmp_path / "mini-model"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "max_position_embeddings": 4096,
        }))
        return str(d)

    def test_config_dataclass_allows_B1_and_B_lt_max_num_seqs(self, tmp_path):
        """B=1 与 B < max_num_seqs 都是合法配置（decode 按预算分批的正常形态）。"""
        model_dir = self.make_model_dir(tmp_path)
        cfg = Config(model_dir, max_num_batched_tokens=1, max_num_seqs=16,
                     enforce_eager=True)
        assert cfg.max_num_batched_tokens == 1 and cfg.max_num_seqs == 16

    @pytest.mark.parametrize("bad", [0, -3, 8.0, "8", True])
    def test_config_dataclass_rejects_invalid_budget(self, tmp_path, bad):
        model_dir = self.make_model_dir(tmp_path)
        with pytest.raises(ValueError, match="max_num_batched_tokens"):
            Config(model_dir, max_num_batched_tokens=bad)

    def test_scheduler_rejects_valid_type_but_nonpositive_budget(self):
        with pytest.raises(ValueError):
            make_scheduler(max_num_batched_tokens=0)


# ============================== 空与终止 ==============================

class TestEmptyAndTermination:
    """空轮 idle 记录、终态清理与健康暂停契约（§8.1「空与终止」行）。"""

    def test_initial_empty_round_is_idle(self):
        sched = make_scheduler()
        batch, is_prefill = sched.schedule(now=1.0)
        assert batch == [] and is_prefill is False
        stats = sched.last_schedule_stats
        assert stats["phase"] == "idle"
        assert stats["planned_tokens"] == 0
        assert stats["scheduled_requests"] == 0
        assert stats["round_id"] == 1
        assert stats["decisions"] == []
        assert sched.is_finished()

    def test_last_request_cancelled_then_clean(self):
        sched = make_scheduler()
        seq = make_seq(4)
        sched.add(seq)
        assert sched.cancel(seq.seq_id, now=1.0) is True
        batch, _ = sched.schedule(now=2.0)
        assert batch == []
        assert sched.is_finished()

    def test_external_mark_terminal_purged_at_boundary(self):
        """外部直接 mark_* 的终态请求在调度边界被兜底清理，不参与调度。"""
        sched = make_scheduler()
        seq = make_seq(4)
        sched.add(seq)
        assert seq.mark_timeout("deadline_exceeded") is True
        batch, _ = sched.schedule(now=5.0)
        assert batch == []
        assert sched.is_finished()

    def test_deadline_expiry_cleaned_at_schedule_boundary(self):
        sched = make_scheduler()
        seq = make_seq(4, deadline=10.0)
        sched.add(seq)
        batch, _ = sched.schedule(now=11.0)
        assert batch == []
        assert seq.status == TIMEOUT
        assert sched.is_finished()

    def test_healthy_pause_returns_idle_with_paused_decision(self):
        """健康暂停：schedule 返回空轮，暂停请求记 paused（不是 budget）。"""
        sched = make_scheduler()
        (seqs, _) = setup_running(sched, [(4, 8)])
        sched.preempt(seqs[0], now=10.0)
        batch, is_prefill = sched.schedule(now=11.0)
        assert batch == [] and is_prefill is False
        stats = sched.last_schedule_stats
        assert stats["phase"] == "idle"
        d = decision_of(stats, seqs[0])
        assert d["reason"] == "paused"
        assert stats["budget_deferred_requests"] == 0
        assert not sched.is_finished()  # 暂停请求仍活动，不能伪装成已完成

    def test_generate_raises_on_healthy_pause_no_busy_loop(self):
        """同步驱动的暂停契约保持 Day6 语义：RuntimeError 拒绝，不忙循环。"""
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = make_scheduler()
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text])
        rid = engine.add_request("hello", SamplingParams(max_tokens=4, ignore_eos=True))
        engine.model_runner = SimpleNamespace(
            call=lambda m, seqs, pre: [7] * len(seqs))  # 桩 runner：支撑 prefill step
        batch, is_prefill = engine.step()  # 先完成 prefill：请求进入 RUNNING
        seq = engine.get_request(rid)
        assert seq.status == RUNNING
        engine.scheduler.preempt(seq, now=1.0)  # 健康暂停（仅 RUNNING 可抢占）
        # generate 不再新增请求：唯一请求处于暂停态，step 无进展必须显式报错
        with pytest.raises(RuntimeError, match="PREEMPTED"):
            engine.generate([], SamplingParams(max_tokens=4, ignore_eos=True),
                            use_tqdm=False)


# ============================== 单 prefill ==============================

class TestSinglePrefill:
    """单请求 prefill：小于/等于/大于 B 的计费与分块推进。"""

    def test_needed_less_than_budget(self):
        sched = make_scheduler(max_num_batched_tokens=8)
        seq = make_seq(3, max_tokens=4)
        sched.add(seq)
        batch, is_prefill = sched.schedule(now=1.0)
        assert is_prefill and [s.seq_id for s in batch] == [seq.seq_id]
        assert seq.num_scheduled_tokens == 3
        assert sched.last_schedule_stats["planned_tokens"] == 3
        d = decision_of(sched.last_schedule_stats, seq)
        # Day8 新增 chunk 观测字段：单 chunk 决策即最后 chunk（chunk_index 1-based）
        assert d == {"seq_id": seq.seq_id, "request_id": seq.request_id,
                     "needed_tokens": 3, "scheduled_tokens": 3, "reason": "scheduled",
                     "kv_checked": True, "blocked_by_seq_id": None,
                     "blocking_reason": None, "budget_wait_seconds": 0.0,
                     "chunk_index": 1, "offset_before": 0, "is_last_chunk": True}
        sched.postprocess(batch, [7], True, now=1.0)
        assert seq.status == RUNNING

    def test_needed_exactly_budget(self):
        sched = make_scheduler(max_num_batched_tokens=8)
        seq = make_seq(8, max_tokens=4)
        sched.add(seq)
        batch, is_prefill = sched.schedule(now=1.0)
        assert is_prefill and seq.num_scheduled_tokens == 8
        assert sched.last_schedule_stats["planned_tokens"] == 8
        sched.postprocess(batch, [7], True, now=1.0)
        assert seq.status == RUNNING

    def test_needed_over_budget_chunked_to_completion(self):
        """需求大于 B：保留既有首请求 chunking，多轮推进直至完成，不永久等待。"""
        sched = make_scheduler(max_num_batched_tokens=8, num_blocks=4)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        chunks = []
        token_lens = []
        for t in (1.0, 2.0, 3.0):
            batch, is_prefill = sched.schedule(now=t)
            assert is_prefill and len(batch) == 1
            d = decision_of(sched.last_schedule_stats, seq)
            chunks.append((d["needed_tokens"], seq.num_scheduled_tokens))
            # Day8 提交契约：中间 chunk 不采样（不传 token），最后 chunk 才传入
            last = seq.prefill_offset + seq.num_scheduled_tokens == seq.prefill_target
            sched.postprocess(batch, [7] if last else [], is_prefill, now=t)
            token_lens.append(len(seq.token_ids))
        # 需求按未缓存进度递减：20 -> 12 -> 4；接纳数每轮 8、8、4
        assert chunks == [(20, 8), (12, 8), (4, 4)]
        # 中间 chunk 不追加采样结果（20、20），最后 chunk 才 append completion（21）
        assert token_lens == [20, 20, 21]
        assert seq.status == RUNNING
        assert seq.num_cached_tokens == 20

    def test_chunked_prefill_no_double_block_allocation(self):
        """分块推进期间 block 只在首次分配，不重复分配；结束时恰好 3 块。"""
        sched = make_scheduler(max_num_batched_tokens=8, num_blocks=4)
        seq = make_seq(20, max_tokens=2)
        sched.add(seq)
        block_counts = []
        for t in (1.0, 2.0, 3.0):
            batch, is_prefill = sched.schedule(now=t)
            block_counts.append(len(seq.block_table))
            last = seq.prefill_offset + seq.num_scheduled_tokens == seq.prefill_target
            sched.postprocess(batch, [7] if last else [], is_prefill, now=t)
        assert block_counts == [3, 3, 3]  # 首轮一次性分配全部 3 块
        bm = sched.block_manager
        assert len(bm.used_block_ids) == 3
        assert sum(b.ref_count for b in bm.blocks) == 3


# ============================== 多 prefill 与 FCFS ==============================

class TestMultiPrefill:
    """文档 §4.3 的 prefill 场景与计费口径。"""

    def test_352_exact_budget(self):
        """B=8、需求 [3,5,2]：前两条刚好耗尽，第三条 direct budget（HOL=0）。"""
        sched = make_scheduler(max_num_batched_tokens=8, num_blocks=8)
        a, b, c = make_seq(3, 8), make_seq(5, 8), make_seq(2, 8)
        for s in (a, b, c):
            sched.add(s)
        batch, is_prefill = sched.schedule(now=1.0)
        assert is_prefill and [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        assert sched.last_schedule_stats["planned_tokens"] == 8
        assert sched.last_schedule_stats["budget_deferred_direct"] == 1
        assert sched.last_schedule_stats["budget_deferred_hol"] == 0
        assert sched.last_schedule_stats["budget_deferred_requests"] == 1
        dc = decision_of(sched.last_schedule_stats, c)
        # 恰好耗尽规则：needed=null、kv_checked=false（未查询 KV，不虚构需求）
        assert dc["reason"] == "budget"
        assert dc["needed_tokens"] is None
        assert dc["kv_checked"] is False
        # 尾部没有第四条：HOL=0
        assert not any(d["reason"] == "head_of_line"
                       for d in sched.last_schedule_stats["decisions"])

    def test_641_does_not_skip_second(self):
        """B=8、需求 [6,4,1]：第二条放不下即停，不跳过它选第三条；余量 2 可不用。"""
        sched = make_scheduler(max_num_batched_tokens=8, num_blocks=8)
        a, b, c = make_seq(6, 8), make_seq(4, 8), make_seq(1, 8)
        for s in (a, b, c):
            sched.add(s)
        batch, is_prefill = sched.schedule(now=1.0)
        assert is_prefill and [s.seq_id for s in batch] == [a.seq_id]
        assert sched.last_schedule_stats["planned_tokens"] == 6
        db = decision_of(sched.last_schedule_stats, b)
        assert db["reason"] == "budget"
        assert db["needed_tokens"] == 4  # 已计算出的需求如实记录
        assert db["kv_checked"] is True  # 预算检查前已查询 KV 容量
        dc = decision_of(sched.last_schedule_stats, c)
        assert dc["reason"] == "head_of_line"
        assert dc["blocking_reason"] == "budget"
        assert dc["blocked_by_seq_id"] == b.seq_id
        assert sched.last_schedule_stats["budget_deferred_direct"] == 1
        assert sched.last_schedule_stats["budget_deferred_hol"] == 1

    def test_prefix_hit_billed_by_uncached_tokens(self):
        """prefix 命中：只按实际未缓存 query token 计费，缓存块不占预算。"""
        sched = make_scheduler(max_num_batched_tokens=32, num_blocks=8)
        prompt = list(range(1, 17))  # 16 token = 2 块
        a = Sequence(prompt, SamplingParams(max_tokens=2, ignore_eos=True))
        b = Sequence(prompt, SamplingParams(max_tokens=2, ignore_eos=True))
        sched.add(a)
        batch, _ = sched.schedule(now=1.0)
        sched.postprocess(batch, [7], True, now=1.0)
        assert a.status == RUNNING
        sched.add(b)
        batch, is_prefill = sched.schedule(now=2.0)
        assert is_prefill
        db = decision_of(sched.last_schedule_stats, b)
        # 第二块（最后一块）按协议不缓存：命中 1 块 = 8 token，需求 16-8=8
        assert db["needed_tokens"] == 8
        assert b.num_scheduled_tokens == 8
        assert sched.last_schedule_stats["planned_tokens"] == 8
        sched.postprocess(batch, [7], True, now=2.0)
        assert b.status == RUNNING

    def test_resume_recompute_billed_by_recompute_input(self):
        """抢占恢复：按 prompt+已生成 token 的未重算部分计费，不是只算剩余 prompt。"""
        sched = make_scheduler(max_num_batched_tokens=32, num_blocks=8)
        seq = make_seq(16, max_tokens=8)
        sched.add(seq)
        batch, _ = sched.schedule(now=1.0)  # prefill 16 token
        sched.postprocess(batch, [7], True, now=1.0)  # len=17
        batch, is_prefill = sched.schedule(now=2.0)
        assert not is_prefill  # decode 1 步
        sched.postprocess(batch, [7], False, now=2.0)  # len=18
        # 显式抢占 + 恢复：KV 释放、进度保留，等待 recompute
        sched.preempt(seq, now=3.0)
        sched.resume(seq)
        assert seq.status == WAITING and not seq.block_table
        batch, is_prefill = sched.schedule(now=4.0)
        assert is_prefill
        d = decision_of(sched.last_schedule_stats, seq)
        # 总长 18 = prompt16 + 生成2；前 2 块命中自身缓存（16 token），重算需求 = 2。
        # 若按"剩余 prompt"口径会误记为 0
        assert d["needed_tokens"] == 2
        assert seq.num_scheduled_tokens == 2
        sched.postprocess(batch, [7], True, now=4.0)
        assert seq.status == RUNNING

    def test_tail_not_counted_when_queue_empty(self):
        """[3,5,2] 后若尾部还有第四条，HOL=1、预算总人数=2（§5.1 精确计数）。"""
        sched = make_scheduler(max_num_batched_tokens=8, num_blocks=8)
        a, b, c, d = (make_seq(3, 8), make_seq(5, 8), make_seq(2, 8), make_seq(7, 8))
        for s in (a, b, c, d):
            sched.add(s)
        batch, _ = sched.schedule(now=1.0)
        stats = sched.last_schedule_stats
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        assert stats["budget_deferred_direct"] == 1  # c
        assert stats["budget_deferred_hol"] == 1     # d 被 c 阻塞
        assert stats["budget_deferred_requests"] == 2
        dd = decision_of(stats, d)
        assert dd["reason"] == "head_of_line"
        assert dd["blocked_by_seq_id"] == c.seq_id
        assert dd["blocking_reason"] == "budget"


# ============================== Decode 硬上限 ==============================

class TestDecodeBudgetCap:
    """decode 同样受预算约束：N>B 时每轮只选 B 条（构造不热改预算）。"""

    def test_decode_selects_exactly_B_of_N(self):
        """B=2、6 条短请求多轮 prefill 全部转 RUNNING 后，decode 每轮只选 2 条。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        seqs = [make_seq(1, max_tokens=4) for _ in range(6)]
        for s in seqs:
            sched.add(s)
        # 多轮 prefill：每轮按预算接纳 2 条，全部转 RUNNING 后 waiting 清空
        for t in (1.0, 2.0, 3.0):
            batch, is_prefill = sched.schedule(now=t)
            assert is_prefill
            assert_round_invariants(sched, batch)
            sched.postprocess(batch, [7] * len(batch), True, now=t)
        assert all(s.status == RUNNING for s in seqs)
        assert not sched.waiting

        # decode：恰好选中队首 2 条，其余按 budget 延后
        batch, is_prefill = sched.schedule(now=4.0)
        assert not is_prefill
        assert [s.seq_id for s in batch] == [seqs[0].seq_id, seqs[1].seq_id]
        assert sched.last_schedule_stats["planned_tokens"] == 2
        for s in seqs[2:]:
            d = decision_of(sched.last_schedule_stats, s)
            assert d["reason"] == "budget"
            assert d["needed_tokens"] == 1  # decode 候选单位需求已知
            assert d["scheduled_tokens"] == 0
        assert sched.last_schedule_stats["budget_deferred_direct"] == 4
        sched.postprocess(batch, [7] * len(batch), is_prefill, now=4.0)  # t4 轮落地

        # 有界推进直至全部完成：每轮 planned <= B，无重复入队
        finished_order = []
        for t in range(5, 40):
            if sched.is_finished():
                break
            batch, is_prefill = sched.schedule(now=float(t))
            assert_round_invariants(sched, batch)
            ids = [s.seq_id for s in batch]
            assert len(set(ids)) == len(ids)
            assert all(s.status == RUNNING for s in batch)
            sched.postprocess(batch, [7] * len(batch), is_prefill, now=float(t))
            for s in batch:
                if s.status == FINISHED:
                    finished_order.append(s.seq_id)
        assert sched.is_finished()
        assert len(finished_order) == 6
        # 统计：唯一请求数=4（C/D/E/F 有过预算 episode）；
        # 请求-轮次=24：prefill r1（c direct + d/e/f HOL）=4、r2（e direct + f HOL）=2、
        # decode t4-6（c/d/e/f 各 3 轮）=12、t7-9（e/f 各 3 轮）=6
        assert sched.budget_deferred_unique_requests_total == 4
        assert sched.budget_deferred_request_rounds_total == 24
        assert sched.budget_wait_closed_seconds_total > 0.0

    def test_decode_planned_never_exceeds_budget_across_rounds(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=3)
        seqs = [make_seq(1, max_tokens=3) for _ in range(5)]
        for s in seqs:
            sched.add(s)
        planned_list = []
        for t in range(1, 60):
            if sched.is_finished():
                break
            batch, is_prefill = sched.schedule(now=float(t))
            planned_list.append(sched.last_schedule_stats["planned_tokens"])
            assert sched.last_schedule_stats["planned_tokens"] <= 3
            sched.postprocess(batch, [7] * len(batch), is_prefill, now=float(t))
        assert sched.is_finished()
        assert planned_list and max(planned_list) <= 3


# ============================== Decode 副作用 ==============================

class TestDecodeSideEffects:
    """预算延后的零资源副作用；KV 不足仍走原抢占路径且不混记为 budget。"""

    def test_deferred_request_untouched(self):
        """未选中请求：状态/token/KV/引用计数/队列位置全部不变。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a, b, c = make_seq(1, 8), make_seq(1, 8), make_seq(1, 8)
        for s in (a, b, c):
            sched.add(s)
            batch, is_prefill = sched.schedule(now=1.0)
            sched.postprocess(batch, [7] * len(batch), is_prefill, now=1.0)
        assert all(s.status == RUNNING for s in (a, b, c))
        c_blocks = list(c.block_table)
        c_ref = sched.block_manager.blocks[c_blocks[0]].ref_count
        batch, _ = sched.schedule(now=9.0)
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        assert c.status == RUNNING
        assert c.num_tokens == 2 and len(c.token_ids) == 2
        assert c.block_table == c_blocks
        assert sched.block_manager.blocks[c_blocks[0]].ref_count == c_ref
        assert c.num_scheduled_tokens == 0
        assert c not in batch
        # 相对顺序稳定：被选批次回到队首，延后请求按原序跟在后面
        assert list(sched.running) == [a, b, c]

    def test_B1_boundary_does_not_preempt_for_budget(self):
        """B=1：每轮只选 1 条；不得为预算原因抢占队尾（资源无副作用）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=1)
        seqs, _ = setup_running(sched, [(1, 8), (1, 8), (1, 8)])
        free_before = len(sched.block_manager.free_block_ids)
        batch, is_prefill = sched.schedule(now=9.0)
        assert not is_prefill and len(batch) == 1
        assert all(s.num_preempts == 0 for s in seqs)
        assert len(sched.block_manager.free_block_ids) == free_before
        stats = sched.last_schedule_stats
        assert stats["budget_deferred_direct"] == 2
        assert not any(d["reason"] == "kv_capacity" for d in stats["decisions"])

    def test_kv_shortage_preempts_with_kv_reason_not_budget(self):
        """资源不足时按原 KV 抢占逻辑执行，原因记 kv_capacity，不算预算拒绝。"""
        sched = make_scheduler(num_blocks=3, max_num_batched_tokens=10 ** 6)
        a = make_seq(1, 100)  # prefill 后 len=2；decode 到 len=9 时需要新块
        b = make_seq(1, 100)
        sched.add(a)
        batch, _ = sched.schedule(now=1.0)
        sched.postprocess(batch, [7], True, now=1.0)
        sched.add(b)
        batch, _ = sched.schedule(now=2.0)
        sched.postprocess(batch, [7], True, now=2.0)
        assert len(sched.block_manager.free_block_ids) == 1  # 池 3 块已用 2
        # decode：a 先拿走最后 1 块（len=2 -> 需要? len=2, 2%8!=1 不需要块）
        # a,b 的 len=2：decode 不需要新块；先各 decode 7 轮到 len=9
        for t in range(3, 10):
            batch, is_prefill = sched.schedule(now=float(t))
            sched.postprocess(batch, [7] * len(batch), is_prefill, now=float(t))
        assert a.num_tokens == 9 and b.num_tokens == 9
        # 下一轮 decode：a 队首需要新块（9%8==1），池已满 -> 抢占 b
        stats_before = sched.budget_deferred_request_rounds_total
        batch, is_prefill = sched.schedule(now=10.0)
        assert is_prefill is False
        assert [s.seq_id for s in batch] == [a.seq_id]
        db = decision_of(sched.last_schedule_stats, b)
        assert db["reason"] == "kv_capacity"
        assert b.num_preempts == 1
        assert b.status == WAITING
        assert not b.block_table
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0
        # b 从未因预算被延后：没有预算等待记录
        assert b.seq_id not in sched.budget_wait
        assert sched.budget_deferred_request_rounds_total == stats_before


# ============================== FCFS 与最终完成 ==============================

class TestFCFSCompletion:
    """running [A,B,C,D]/B2：A/B 有界完成后 C/D 推进；不重复入队、不逆序。"""

    def test_tail_progresses_after_head_completes(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a, b = make_seq(1, 3), make_seq(1, 3)
        c, d = make_seq(1, 3), make_seq(1, 3)
        for s in (a, b, c, d):
            sched.add(s)
            batch, is_prefill = sched.schedule(now=float(len(sched.waiting)))
            sched.postprocess(batch, [7] * len(batch), is_prefill,
                              now=float(len(sched.waiting)))
        assert all(s.status == RUNNING for s in (a, b, c, d))
        # decode 推进：每轮 2 条；C/D 延后但保持相对顺序
        completed = []
        for t in range(9, 40):
            if sched.is_finished():
                break
            batch, is_prefill = sched.schedule(now=float(t))
            assert_round_invariants(sched, batch)
            ids = [s.seq_id for s in batch]
            assert len(set(ids)) == len(ids), "同一请求不得在同一轮重复入队"
            sched.postprocess(batch, [7] * len(batch), is_prefill, now=float(t))
            completed += [s.seq_id for s in batch if s.status == FINISHED]
        assert sched.is_finished()
        # 完成顺序即 FCFS：A,B 先于 C,D（每组内部按队首优先）
        assert completed == [a.seq_id, b.seq_id, c.seq_id, d.seq_id]
        # C/D 在 A/B 的 2 个 decode 轮中被预算延后（setup 逐条 prefill 无延后）；
        # 请求完成终态后记录已删除，这里用累计计数核对（C/D 各 2 轮，唯一请求 2 个）
        assert sched.budget_deferred_request_rounds_total == 4
        assert sched.budget_deferred_unique_requests_total == 2


# ============================== 相互限制 ==============================

class TestMutualLimits:
    """max_num_seqs 与预算的相互限制、prefill 优先、KV 队首不足与 HOL 归因。"""

    def test_sequence_cap_limits_decode_first(self):
        """B=8、cap=2、4 条 running：限制来自 cap，不得伪记为 budget。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8, max_num_seqs=2)
        seqs, _ = setup_running(sched, [(1, 8)] * 4)
        batch, _ = sched.schedule(now=9.0)
        assert len(batch) == 2
        stats = sched.last_schedule_stats
        for s in seqs[2:]:
            assert decision_of(stats, s)["reason"] == "sequence_cap"
        assert stats["budget_deferred_requests"] == 0

    def test_cap_and_budget_simultaneous_cap_wins(self):
        """B=2、cap=2、4 条 running：两者同时命中时统一优先记 sequence_cap。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2, max_num_seqs=2)
        seqs, _ = setup_running(sched, [(1, 8)] * 4)
        batch, _ = sched.schedule(now=9.0)
        assert len(batch) == 2
        stats = sched.last_schedule_stats
        for s in seqs[2:]:
            assert decision_of(stats, s)["reason"] == "sequence_cap"
        assert stats["budget_deferred_requests"] == 0

    def test_prefill_priority_suppresses_decode_as_phase_priority(self):
        """有可接纳 prefill 时整轮 prefill；running 未成为候选记 phase_priority。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8)
        (running, _) = setup_running(sched, [(1, 8), (1, 8)])
        newcomer = make_seq(2, 8)
        sched.add(newcomer)
        batch, is_prefill = sched.schedule(now=9.0)
        assert is_prefill
        assert [s.seq_id for s in batch] == [newcomer.seq_id]
        for s in running:
            assert decision_of(sched.last_schedule_stats, s)["reason"] == "phase_priority"
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0

    def test_kv_head_insufficient_yields_idle_with_hol(self):
        """KV 根本不足：队首记 kv_capacity，尾部记 HOL；返回空轮不忙循环。"""
        sched = make_scheduler(num_blocks=2, max_num_batched_tokens=32)
        x = make_seq(24, 8)  # 需要 3 块 > 池 2 块
        y = make_seq(4, 8)
        sched.add(x)
        sched.add(y)
        batch, is_prefill = sched.schedule(now=1.0)
        assert batch == [] and is_prefill is False
        stats = sched.last_schedule_stats
        assert stats["phase"] == "idle"
        assert decision_of(stats, x)["reason"] == "kv_capacity"
        dy = decision_of(stats, y)
        assert dy["reason"] == "head_of_line"
        assert dy["blocking_reason"] == "kv_capacity"
        assert dy["blocked_by_seq_id"] == x.seq_id
        assert stats["budget_deferred_requests"] == 0

    def test_prefill_sequence_cap_hol_not_budget(self):
        """prefill 阶段 cap 耗尽：首个未处理记 sequence_cap，其余为 cap HOL（不计预算）。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=64, max_num_seqs=2)
        a, b, c = make_seq(3, 8), make_seq(5, 8), make_seq(2, 8)
        for s in (a, b, c):
            sched.add(s)
        batch, _ = sched.schedule(now=1.0)
        assert [s.seq_id for s in batch] == [a.seq_id, b.seq_id]
        stats = sched.last_schedule_stats
        assert decision_of(stats, c)["reason"] == "sequence_cap"
        assert stats["budget_deferred_requests"] == 0


# ============================== 统计人数 ==============================

class TestBudgetWaitCounts:
    """direct/HOL 去重、唯一请求数、部分 chunk 不计入、非预算原因不混算。"""

    def test_unique_counted_once_across_rounds(self):
        """同一请求多轮延后：request_rounds 累加，unique 只算 1。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a, b, c = make_seq(1, 8), make_seq(1, 8), make_seq(1, 8)
        for s in (a, b):
            sched.add(s)
            batch, is_prefill = sched.schedule(now=float(len(sched.waiting)))
            sched.postprocess(batch, [7] * len(batch), is_prefill,
                              now=float(len(sched.waiting)))
        sched.add(c)
        batch, is_prefill = sched.schedule(now=5.0)  # prefill c
        sched.postprocess(batch, [7], is_prefill, now=5.0)
        # 三轮 decode：c 每轮延后
        for t in (6.0, 7.0, 8.0):
            batch, _ = sched.schedule(now=t)
            sched.postprocess(batch, [7] * len(batch), False, now=t)
        assert sched.budget_deferred_unique_requests_total == 1
        assert sched.budget_deferred_request_rounds_total == 3
        assert sched.budget_wait[c.seq_id].budget_deferred_rounds == 3
        assert a.seq_id not in sched.budget_wait and b.seq_id not in sched.budget_wait

    def test_partial_chunk_not_counted_as_deferred(self):
        """分块 prefill 每轮都有推进：不把剩余 token 记成整请求预算等待。"""
        sched = make_scheduler(num_blocks=4, max_num_batched_tokens=8)
        seq = make_seq(20, 2)
        sched.add(seq)
        for t in (1.0, 2.0, 3.0):
            batch, is_prefill = sched.schedule(now=t)
            assert is_prefill
            assert sched.last_schedule_stats["budget_deferred_requests"] == 0
            assert decision_of(sched.last_schedule_stats, seq)["reason"] == "scheduled"
            last = seq.prefill_offset + seq.num_scheduled_tokens == seq.prefill_target
            sched.postprocess(batch, [7] if last else [], is_prefill, now=t)
        assert seq.seq_id not in sched.budget_wait

    def test_non_budget_reasons_not_mixed_into_counts(self):
        """sequence_cap / phase_priority / paused / kv_capacity 都不计入预算人数。"""
        # sequence_cap（decode）
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=8, max_num_seqs=2)
        seqs, _ = setup_running(sched, [(1, 8)] * 3)
        sched.schedule(now=9.0)
        assert sched.last_schedule_stats["budget_deferred_requests"] == 0
        # phase_priority（prefill 优先）
        sched2 = make_scheduler(num_blocks=8, max_num_batched_tokens=8)
        (running, _) = setup_running(sched2, [(1, 8)])
        sched2.add(make_seq(2, 8))
        sched2.schedule(now=9.0)
        assert sched2.last_schedule_stats["budget_deferred_requests"] == 0
        # paused（健康暂停）
        sched3 = make_scheduler(num_blocks=8, max_num_batched_tokens=8)
        (paused_seqs, _) = setup_running(sched3, [(1, 8)])
        sched3.preempt(paused_seqs[0], now=10.0)
        sched3.schedule(now=11.0)
        assert sched3.last_schedule_stats["budget_deferred_requests"] == 0


# ============================== 统计时间（确定时钟时间线） ==============================

class TestBudgetWaitTiming:
    """§5.2 时间线：t10/t12/t15 与 t20/t23 精确为 rounds=3、seconds=8；无 sleep。"""

    def test_timeline_exact_rounds_and_seconds(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a = make_seq(1, 100)
        b = make_seq(1, 100)
        c = make_seq(1, 100, request_id="victim")  # C 的 ID 稍后被新对象复用
        # 逐条 prefill，避免 setup 期间出现预算延后
        for seq in (a, b, c):
            sched.add(seq)
            batch, is_prefill = sched.schedule(now=1.0)
            sched.postprocess(batch, [7], is_prefill, now=1.0)
        # t=10：A,B 被调度，C 首次 budget（episode 开启）
        batch, _ = sched.schedule(now=10.0)
        sched.postprocess(batch, [7] * len(batch), False, now=10.0)
        rec = sched.budget_wait[c.seq_id]
        assert rec.budget_deferred_rounds == 1
        assert rec.budget_wait_started_at == 10.0
        assert rec.budget_wait_seconds == 0.0
        # t=12：再 budget（episode 不重开）
        batch, _ = sched.schedule(now=12.0)
        sched.postprocess(batch, [7] * len(batch), False, now=12.0)
        rec = sched.budget_wait[c.seq_id]
        assert rec.budget_deferred_rounds == 2
        assert rec.budget_wait_started_at == 10.0  # 连续等待不重置起点
        # t=15：新增请求触发 prefill 轮，C 原因切换为 phase_priority -> 结算 5 秒
        p = make_seq(1, 100)
        sched.add(p)
        batch, is_prefill = sched.schedule(now=15.0)
        assert is_prefill
        sched.postprocess(batch, [7] * len(batch), is_prefill, now=15.0)
        rec = sched.budget_wait[c.seq_id]
        assert rec.budget_deferred_rounds == 2
        assert rec.budget_wait_started_at is None
        assert rec.budget_wait_seconds == 5.0
        assert sched.budget_wait_closed_seconds_total == 5.0
        # t=20：decode 轮 C 再次 budget（episode 重开，unique 不再+1）
        batch, _ = sched.schedule(now=20.0)
        sched.postprocess(batch, [7] * len(batch), False, now=20.0)
        rec = sched.budget_wait[c.seq_id]
        assert rec.budget_deferred_rounds == 3
        assert rec.budget_wait_started_at == 20.0
        assert rec.budget_wait_seconds == 5.0  # 已结算部分不重复累计
        assert sched.budget_deferred_unique_requests_total == 2  # C 与 P
        # t=23：cancel C -> 结算 3 秒；总 rounds=3、closed=8
        assert sched.cancel(c.seq_id, now=23.0) is True
        assert c.seq_id not in sched.budget_wait
        assert sched.budget_wait_closed_seconds_total == 8.0
        assert sched.budget_deferred_request_rounds_total == 4  # C:3 + P:1
        # 重复清理：不改变 8
        assert sched.cancel(c.seq_id, now=24.0) is False
        sched._finalize(c, now=25.0)
        assert sched.budget_wait_closed_seconds_total == 8.0
        assert sched.budget_deferred_unique_requests_total == 2
        # t=23.5：取消 P（其 episode 于 t=20 开启）：结算 3.5 秒后 P 退出统计
        assert sched.cancel(p.seq_id, now=23.5) is True
        assert sched.budget_wait_closed_seconds_total == 11.5  # 8.0 + 3.5
        # 复用 request_id 的新对象从 0 开始
        f = make_seq(1, 100, request_id="victim")
        sched.add(f)
        batch, is_prefill = sched.schedule(now=26.0)  # prefill f
        sched.postprocess(batch, [7] * len(batch), is_prefill, now=26.0)
        batch, _ = sched.schedule(now=27.0)  # decode：A,B 选中，f 延后
        assert decision_of(sched.last_schedule_stats, f)["reason"] == "budget"
        rec_f = sched.budget_wait[f.seq_id]
        assert rec_f.budget_deferred_rounds == 1
        assert rec_f.request_id == "victim"
        assert sched.budget_deferred_unique_requests_total == 3
        assert sched.budget_deferred_request_rounds_total == 5

    def test_timeout_settles_episode_at_operation_time(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        seqs, _ = setup_running(sched, [(1, 100), (1, 100), (1, 100)])
        batch, _ = sched.schedule(now=10.0)  # c 延后，episode 开启
        sched.postprocess(batch, [7] * len(batch), False, now=10.0)
        c = seqs[2]
        assert c.seq_id in sched.budget_wait
        sched.timeout(c.seq_id, now=13.5)
        assert sched.budget_wait_closed_seconds_total == 3.5
        assert c.seq_id not in sched.budget_wait

    def test_external_mark_settles_at_first_observation(self):
        """外部 mark_* 不倒填时间：按 Scheduler 首次观察时刻结算。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        seqs, _ = setup_running(sched, [(1, 100), (1, 100), (1, 100)])
        batch, _ = sched.schedule(now=100.0)  # c 延后
        sched.postprocess(batch, [7] * len(batch), False, now=100.0)
        c = seqs[2]
        c.mark_cancelled("client_abort")  # 外部直接标记，无调度方参与
        # 观察发生在 now=105：episode 按 105 结算，而非 mark_* 的真实墙钟
        sched.schedule(now=105.0)
        assert sched.budget_wait_closed_seconds_total == 5.0
        assert c.seq_id not in sched.budget_wait
        assert c.status == CANCELLED

    def test_explicit_preempt_closes_episode(self):
        """显式抢占结束预算 episode：此后等待为 paused/KV，不再计预算。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        seqs, _ = setup_running(sched, [(1, 100), (1, 100), (1, 100)])
        batch, _ = sched.schedule(now=10.0)  # c 延后
        sched.postprocess(batch, [7] * len(batch), False, now=10.0)
        c = seqs[2]
        assert sched.budget_wait[c.seq_id].budget_wait_started_at == 10.0
        sched.preempt(c, now=12.0)
        rec = sched.budget_wait[c.seq_id]  # 记录保留（非终态），episode 已关
        assert rec.budget_wait_started_at is None
        assert rec.budget_wait_seconds == 2.0
        assert sched.budget_wait_closed_seconds_total == 2.0

    def test_postprocess_deadline_discard_settles_at_entry_now(self):
        """postprocess 内嵌终态清理沿用其入口 now：b 延后两轮后于 t=11.9 被接纳
        （episode 就此结算），模型执行返回时已跨过 deadline——输出丢弃、按 TIMEOUT 终止。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a = make_seq(1, 100)
        c = make_seq(1, 100)
        b = make_seq(1, 100, deadline=12.0)
        # setup 顺序 a,c,b：decode 时 b 排在队尾，被预算延后
        for s in (a, c, b):
            sched.add(s)
            batch, is_prefill = sched.schedule(now=1.0)
            sched.postprocess(batch, [7], is_prefill, now=1.0)
        # t=10 / t=11：a,c 选中，b 延后（episode 开启于 t=10）
        for t in (10.0, 11.0):
            batch, _ = sched.schedule(now=t)
            assert decision_of(sched.last_schedule_stats, b)["reason"] == "budget"
            sched.postprocess(batch, [7] * len(batch), False, now=t)
        assert sched.budget_wait[b.seq_id].budget_wait_started_at == 10.0
        # t=11.5：取消 a,c -> b 独占；t=11.9（deadline 之前）decode 接纳 b，
        # 接纳即原因切换：episode 在调度边界结算（11.9 - 10.0 = 1.9）
        sched.cancel(a.seq_id, now=11.5)
        sched.cancel(c.seq_id, now=11.5)
        batch, _ = sched.schedule(now=11.9)
        assert [s.seq_id for s in batch] == [b.seq_id]
        assert sched.budget_wait_closed_seconds_total == pytest.approx(1.9)
        assert sched.budget_wait[b.seq_id].budget_wait_started_at is None
        # 模型执行跨过 deadline（12.0）：postprocess 入口 now=13.0 丢弃输出并 TIMEOUT，
        # 终态收尾使用入口 now（finished_at=13.0），不产生新的 episode 结算
        sched.postprocess(batch, [7] * len(batch), False, now=13.0)
        assert b.status == TIMEOUT
        assert b.finished_at == 13.0
        assert b.seq_id not in sched.requests
        assert b.seq_id not in sched.budget_wait
        assert sched.budget_wait_closed_seconds_total == pytest.approx(1.9)


# ============================== 所有权回归 ==============================

class TestOwnershipAndIdReuse:
    """ID 复用 + 迟到 postprocess：新对象身份与预算记录不被旧对象污染。"""

    def _make_reuse_scene(self, mode: str):
        """构造复用场景：A（request_id='shared'）以 length/cancel 方式结束，
        E 先占住 decode 队首，使复用同一 ID 的 B 被预算延后一次（episode 于 t=4 开启）。
        返回 (sched, seq_a, seq_b, e, stale_batch)；stale_batch 是 A 的旧 prefill 批次。
        """
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=1)
        seq_a = make_seq(1, max_tokens=1 if mode == "length" else 99,
                         request_id="shared")
        sched.add(seq_a)
        batch, is_prefill = sched.schedule(now=1.0)  # prefill A（1 token，一轮完成）
        stale_batch = (list(batch), is_prefill)
        sched.postprocess(batch, [7], True, now=1.0)
        if mode == "length":
            assert seq_a.status == FINISHED  # completion=1 == max_tokens
        else:
            assert sched.cancel(seq_a.seq_id, now=1.5) is True
        e = make_seq(1, max_tokens=9)
        sched.add(e)
        batch, is_prefill = sched.schedule(now=2.0)  # prefill E
        sched.postprocess(batch, [7], is_prefill, now=2.0)
        seq_b = make_seq(1, max_tokens=9, request_id="shared")
        sched.add(seq_b)
        batch, is_prefill = sched.schedule(now=3.0)  # prefill B（E phase_priority）
        sched.postprocess(batch, [7], is_prefill, now=3.0)
        batch, _ = sched.schedule(now=4.0)  # decode B=1：E 队首选中，B 延后
        assert decision_of(sched.last_schedule_stats, seq_b)["reason"] == "budget"
        return sched, seq_a, seq_b, e, stale_batch

    @pytest.mark.parametrize("mode", ["length", "cancel"])
    def test_stale_postprocess_after_id_reuse_keeps_new_owner(self, mode):
        """两种结束方式（正常结束 / 取消）下：旧 A 的迟到 postprocess
        不得污染复用了 request_id 的新对象 B 的身份与预算等待记录。"""
        sched, seq_a, seq_b, e, stale_batch = self._make_reuse_scene(mode)
        rounds_before = sched.budget_deferred_request_rounds_total
        unique_before = sched.budget_deferred_unique_requests_total
        closed_before = sched.budget_wait_closed_seconds_total
        # 迟到重放 A 的旧 prefill 批次（A 已终态）
        sched.postprocess(stale_batch[0], [7] * len(stale_batch[0]),
                          stale_batch[1], now=5.0)
        # B 的身份与等待记录不变，累计不重复
        assert sched.requests[seq_b.seq_id] is seq_b
        assert seq_b.request_id == "shared"
        assert "shared" in sched._active_request_ids
        assert sched.budget_wait[seq_b.seq_id].budget_deferred_rounds == 1
        assert sched.budget_deferred_request_rounds_total == rounds_before
        assert sched.budget_deferred_unique_requests_total == unique_before
        assert sched.budget_wait_closed_seconds_total == closed_before
        # B 被调度接纳：episode 恰好结算一次（t=4 开启，t=8 关闭）
        sched.cancel(e.seq_id, now=6.0)
        batch, _ = sched.schedule(now=8.0)
        assert [s.seq_id for s in batch] == [seq_b.seq_id]
        rec = sched.budget_wait[seq_b.seq_id]
        assert rec.budget_wait_started_at is None
        assert rec.budget_wait_seconds == 4.0
        assert sched.budget_wait_closed_seconds_total - closed_before == 4.0
        # 再次迟到重放：累计不变（幂等）
        sched.postprocess(stale_batch[0], [7] * len(stale_batch[0]),
                          stale_batch[1], now=9.0)
        assert sched.budget_wait_closed_seconds_total - closed_before == 4.0

    def test_finalize_ownership_is_bound_to_index_registration(self):
        """非所有者重复 _finalize 是安全空操作，不删除新对象的预算记录。"""
        sched, seq_a, seq_b, e, stale_batch = self._make_reuse_scene("length")
        rec_before = sched.budget_wait[seq_b.seq_id]
        sched._finalize(seq_a, now=6.0)  # 旧对象重复收尾（非所有者）
        assert sched.budget_wait[seq_b.seq_id] is rec_before
        assert sched.requests[seq_b.seq_id] is seq_b


# ============================== Engine 观测 ==============================

class TestEngineObservation:
    """计划与执行按 round_id 关联；postprocess 清零不丢证据。"""

    @staticmethod
    def make_engine(sched: Scheduler, runner):
        """跳过 GPU __init__：注入真实 Scheduler、tokenizer 桩与 spy ModelRunner 桩。

        spy 在调用时刻对 (seq_id, request_id, num_scheduled_tokens) 做快照——
        postprocess 会把临时计数清零，事后不能从引用对象倒推执行工作量。
        """
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(
            encode=lambda text: [ord(c) % 100 + 1 for c in text])
        calls = []

        def fake_call(method, seqs, is_prefill):
            calls.append(([(s.seq_id, s.request_id, s.num_scheduled_tokens)
                           for s in seqs], is_prefill))
            return runner(seqs, is_prefill)

        engine.model_runner = SimpleNamespace(call=fake_call)
        return engine, calls

    def test_prefill_plan_matches_runner_input(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=32)
        engine, calls = self.make_engine(sched, lambda seqs, pre: [7] * len(seqs))
        engine.add_request("hello", SamplingParams(max_tokens=4, ignore_eos=True))
        outputs, num_tokens = engine.step()
        snapshot, is_prefill = calls[0]
        assert is_prefill is True
        planned = sum(n for _, _, n in snapshot)
        assert planned == sched.last_schedule_stats["planned_tokens"] == 5  # "hello"->5 token
        assert num_tokens == 5  # prefill 返回正数（历史口径）
        assert outputs == []

    def test_decode_plan_matches_runner_input_and_negative_step(self):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=32)
        engine, calls = self.make_engine(sched, lambda seqs, pre: [7] * len(seqs))
        engine.add_request("hi", SamplingParams(max_tokens=2, ignore_eos=True))
        engine.step()  # prefill
        outputs, num_tokens = engine.step()  # decode
        snapshot, is_prefill = calls[1]
        assert is_prefill is False
        # decode 的预算消耗 = 序列数（每条 1 token）
        assert sched.last_schedule_stats["planned_tokens"] == len(snapshot) == 1
        assert num_tokens == -1  # decode 返回负数（历史口径）
        # postprocess 已把临时计数清零，但本轮执行证据不受影响（调用前快照）
        assert all(n == 1 for _, _, n in snapshot)
        assert sched.last_schedule_stats["planned_tokens"] == 1
        # "hi" 2 token：prefill 后完成数 1，decode 后达到 max_tokens=2 -> 正常完成
        assert len(outputs) == 1 and outputs[0][1] == [7, 7]

    def test_cancel_during_execution_discards_output_keeps_executed(self, caplog):
        """执行返回期间被取消：采样输出丢弃，但本轮已执行输入仍计入预算。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=32)

        def runner(seqs, is_prefill):
            if not is_prefill:
                seqs[0].request_cancel("client_abort")
            return [7] * len(seqs)

        engine, calls = self.make_engine(sched, runner)
        rid = engine.add_request("hi", SamplingParams(max_tokens=4, ignore_eos=True))
        engine.step()  # prefill
        with caplog.at_level(logging.INFO):
            outputs, num_tokens = engine.step()  # decode：执行期间取消
        seq = engine.get_request(rid)
        assert seq is None  # 已终态清理
        assert outputs == []
        # 执行事件记录的是调用前快照，不因取消而缩减
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        engine_rounds = [e for e in events if e["event"] == "engine_round"]
        decode_rounds = [e for e in engine_rounds if e["phase"] == "decode"]
        assert decode_rounds[-1]["executed_tokens"] == 1
        assert decode_rounds[-1]["planned_tokens"] == 1
        assert decode_rounds[-1]["outcome"] == "completed"

    def test_runner_error_logs_null_executed(self, caplog):
        """模型异常不假记成功：executed_tokens=null、outcome=error，异常继续传播。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=32)
        engine, calls = self.make_engine(
            sched, lambda seqs, pre: (_ for _ in ()).throw(RuntimeError("boom")))
        engine.add_request("hi", SamplingParams(max_tokens=4, ignore_eos=True))
        with caplog.at_level(logging.INFO):
            with pytest.raises(RuntimeError, match="boom"):
                engine.step()
        events = [json.loads(r.message) for r in caplog.records
                  if r.message.startswith("{")]
        err = [e for e in events if e["event"] == "engine_round" and e["outcome"] == "error"]
        assert len(err) == 1
        assert err[0]["executed_tokens"] is None
        assert err[0]["planned_tokens"] == 2  # "hi" -> 2 token
        assert err[0]["model_called"] is True

    def test_step_rejects_overbudget_plan_before_model_call(self):
        """调用前显式校验（不依赖 assert）：计划超预算时拒绝并给出 round_id/需求/预算。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=4)
        engine, calls = self.make_engine(sched, lambda seqs, pre: [7] * len(seqs))
        seq = make_seq(8, max_tokens=4)
        sched.add(seq)
        sched.schedule(now=1.0)
        # 模拟调度器内部错误：伪造越界的批次计数与计划快照，验证 Engine 调用前拦截
        seq.num_scheduled_tokens = 100
        sched.schedule = lambda **kwargs: ([seq], True)  # type: ignore[assignment]
        sched.last_schedule_stats = {"round_id": 99, "token_budget": 4,
                                     "planned_tokens": 100}
        with pytest.raises(ValueError, match="round 99"):
            engine.step()
        assert calls == []  # 模型未被调用


# ============================== 日志证据 ==============================

class TestLogEvidence:
    """JSON 事件可解析、round_id 唯一、执行 <=B、计数可重算、无 prompt 明文。"""

    def test_json_events_fully_checkable(self, caplog):
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=2)
        a, b = make_seq(1, 3), make_seq(1, 3)
        c = make_seq(1, 3)
        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = sched
        engine.tokenizer = SimpleNamespace(encode=lambda t: [1] * 1)
        engine.model_runner = SimpleNamespace(
            call=lambda m, seqs, pre: [7] * len(seqs))
        SENTINEL = "CLASSIFIED_PROMPT_TEXT"
        with caplog.at_level(logging.INFO):
            # 混合驱动：prefill、decode、空轮
            sched.add(a)
            sched.add(b)
            batch, pre = sched.schedule(now=1.0)
            sched.postprocess(batch, [7] * len(batch), pre, now=1.0)
            batch, pre = sched.schedule(now=2.0)
            sched.postprocess(batch, [7] * len(batch), pre, now=2.0)
            batch, pre = sched.schedule(now=3.0)
            sched.postprocess(batch, [7] * len(batch), pre, now=3.0)
            sched.add(make_seq(1, 3))
            engine.step()  # 空轮或执行轮（视队列状态）
            sched.schedule(now=4.0)
        raw_lines = [r.message for r in caplog.records if r.message.startswith("{")]
        assert raw_lines, "应产生结构化事件"
        events = [json.loads(line) for line in raw_lines]
        assert SENTINEL not in caplog.text  # 无 prompt 明文

        rounds = [e for e in events if e["event"] == "scheduler_round"]
        assert rounds, "每轮（含空轮）都应有 scheduler_round 事件"
        round_ids = [e["round_id"] for e in rounds]
        assert round_ids == sorted(round_ids) and len(set(round_ids)) == len(round_ids)
        for e in rounds:
            assert e["phase"] in ("prefill", "decode", "idle")
            assert 0 <= e["planned_tokens"] <= e["token_budget"]
            assert e["scheduled_requests"] <= e["max_num_seqs"]
            assert e["budget_deferred_requests"] == (
                e["budget_deferred_direct"] + e["budget_deferred_hol"])
            known = {"scheduled", "budget", "sequence_cap", "kv_capacity",
                     "head_of_line", "phase_priority", "paused"}
            for d in e["decisions"]:
                assert d["reason"] in known
            # 人数可从 decisions 重算
            direct = sum(1 for d in e["decisions"] if d["reason"] == "budget")
            hol = sum(1 for d in e["decisions"]
                      if d["reason"] == "head_of_line" and d["blocking_reason"] == "budget")
            assert (direct, hol) == (e["budget_deferred_direct"], e["budget_deferred_hol"])

        engine_rounds = [e for e in events if e["event"] == "engine_round"]
        for e in engine_rounds:
            if e["outcome"] == "completed":
                assert e["model_called"] is True
                assert e["executed_tokens"] == e["planned_tokens"] <= e["token_budget"]
            elif e["outcome"] == "idle":
                assert e["model_called"] is False and e["executed_tokens"] == 0
            else:
                assert e["outcome"] == "error" and e["executed_tokens"] is None
        # 计划与执行按 round_id 关联
        sched_by_id = {e["round_id"]: e for e in rounds}
        for e in engine_rounds:
            if e["outcome"] == "completed":
                assert e["round_id"] in sched_by_id
                assert sched_by_id[e["round_id"]]["planned_tokens"] == e["planned_tokens"]

        # episode 事件：起止时间可独立重算等待秒数
        episodes = [e for e in events if e["event"] == "budget_wait_episode"]
        for e in episodes:
            assert e["duration_seconds"] == pytest.approx(
                e["ended_at"] - e["started_at"])
            assert e["duration_seconds"] >= 0.0
        summaries = [e for e in events if e["event"] == "request_budget_wait"]
        for e in summaries:
            assert e["status"] in ("FINISHED", "CANCELLED", "TIMEOUT")
        # 本场景全部 episode 均在终态前关闭：总时长可由事件重算
        assert sched.budget_wait_closed_seconds_total == pytest.approx(
            sum(e["duration_seconds"] for e in episodes))
        assert sched.budget_wait_closed_seconds_total == pytest.approx(
            sum(e["budget_wait_seconds"] for e in summaries))

    def test_log_level_off_by_default_no_output(self):
        """库代码不做 basicConfig：默认级别下不输出（由调用方控制）。"""
        import io
        from nanovllm.engine import scheduler as sched_mod
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            sched = make_scheduler()
            seqs, _ = setup_running(sched, [(1, 3), (1, 3), (1, 3)])
            sched.schedule(now=9.0)
            assert stream.getvalue() == ""  # 未开启 INFO 时无输出
        finally:
            root.removeHandler(handler)


# ============================== 长序列混合 ==============================

class TestMixedRandomScenario:
    """固定 seed 交错 add/cancel/timeout/preempt/resume/mark_*/迟到重放，
    轮次上限防挂死；身份/队列/引用计数/free-used 全部平衡。"""

    def test_mixed_ops_balance_and_termination(self):
        rng = random.Random(20260911)
        sched = make_scheduler(num_blocks=24, max_num_batched_tokens=16, max_num_seqs=8)
        id_pool = [f"mixed-{i}" for i in range(6)]
        stale_batches = []  # (seqs, is_prefill, token_ids)：迟到重放候选
        now = 100.0
        for _ in range(120):
            now += 1.0
            op = rng.random()
            if op < 0.22:
                rid = rng.choice(id_pool)
                seq = make_seq(rng.randint(1, 12), max_tokens=rng.randint(1, 5),
                               request_id=rid)
                try:
                    sched.add(seq)
                except ValueError:
                    pass  # 活动 ID 冲突：符合预期的原子拒绝
            elif op < 0.32 and sched.requests:
                sched.cancel(rng.choice(list(sched.requests)), now=now)
            elif op < 0.39 and sched.requests:
                sched.timeout(rng.choice(list(sched.requests)), now=now)
            elif op < 0.45 and sched.running:
                seq = rng.choice(list(sched.running))
                if seq.status == RUNNING:
                    sched.preempt(seq, now=now)
                    sched.resume(seq)
            elif op < 0.51 and sched.requests:
                rng.choice(list(sched.requests.values())).mark_finished("stop")
            elif op < 0.57 and stale_batches:
                seqs, is_pre, tokens = stale_batches.pop(
                    rng.randrange(len(stale_batches)))
                if all(s.is_terminal for s in seqs):
                    # 迟到重放：只允许全终态批次（模拟迟到的执行结果）
                    sched.postprocess(seqs, tokens, is_pre, now=now)
            # 驱动一轮
            batch, is_prefill = sched.schedule(now=now)
            assert_round_invariants(sched, batch)
            # Day8 提交契约：prefill 轮只有最后 chunk 的请求产出采样 token
            token_ids = [rng.randint(1, 100) for s in batch
                         if not is_prefill or (s.prefill_offset + s.num_scheduled_tokens
                                               == s.prefill_target)]
            sched.postprocess(batch, token_ids, is_prefill, now=now)
            if batch:
                stale_batches.append((list(batch), is_prefill, token_ids))
                if len(stale_batches) > 4:
                    stale_batches.pop(0)
            # 轻量不变量：队列互斥、状态与索引一致
            assert not (set(sched.waiting) & set(sched.running))
            for s in sched.waiting:
                assert s.status == WAITING and sched.requests.get(s.seq_id) is s
            for s in sched.running:
                assert s.status == RUNNING and sched.requests.get(s.seq_id) is s

        # 收尾：有界清场——全部取消后引擎必须可终止
        for _ in range(300):
            if sched.is_finished():
                break
            for sid in list(sched.requests):
                sched.cancel(sid, now=now)
            now += 1.0
        assert sched.is_finished()
        # 资源与索引平衡
        bm = sched.block_manager
        free_set = set(bm.free_block_ids)
        assert free_set | set(bm.used_block_ids) == set(range(len(bm.blocks)))
        assert not (free_set & set(bm.used_block_ids))
        assert all(bm.blocks[i].ref_count == 0 for i in free_set)
        assert not sched.waiting and not sched.running and not sched.requests
        assert not sched.budget_wait  # 终态记录全部回收
        # 每轮计划都未越界（从统计累计验证）
        assert sched.max_num_batched_tokens == 16

    def test_mixed_ops_decode_budget_always_respected(self):
        """随机场景中每轮 planned_tokens 不超预算（第二轮 seed 交叉验证）。"""
        rng = random.Random(777)
        sched = make_scheduler(num_blocks=16, max_num_batched_tokens=8, max_num_seqs=4)
        now = 0.0
        for _ in range(80):
            now += 1.0
            if rng.random() < 0.4:
                sched.add(make_seq(rng.randint(1, 9), max_tokens=rng.randint(1, 4)))
            if rng.random() < 0.15 and sched.requests:
                sched.cancel(rng.choice(list(sched.requests)), now=now)
            batch, is_prefill = sched.schedule(now=now)
            planned = sched.last_schedule_stats["planned_tokens"]
            assert 0 <= planned <= 8
            assert len(batch) <= 4
            # Day8 提交契约：prefill 轮只有最后 chunk 的请求产出采样 token
            tokens = [rng.randint(1, 100) for s in batch
                      if not is_prefill or (s.prefill_offset + s.num_scheduled_tokens
                                            == s.prefill_target)]
            sched.postprocess(batch, tokens, is_prefill, now=now)
        for sid in list(sched.requests):
            sched.cancel(sid, now=now)
        assert sched.is_finished()
