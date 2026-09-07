"""Day 3 验收测试：KV Cache 在 Scheduler 驱动下的完整生命周期。

覆盖 plan.md Day 3 任务 2 的路径：申请（allocate）、写入（postprocess 中的
hash_blocks 记账）、复用（prefix cache）、释放（FINISHED）与 preemption。

不加载模型：schedule()/postprocess() 是纯 CPU 逻辑，本轮采样 token 由测试
直接注入（真实运行时由 ModelRunner + Sampler 产生）。Scheduler 只读取
Config 的少数字段，用 SimpleNamespace 即可构造，避免依赖模型目录。
"""

from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 8
EOS = 999_999  # 测试中不会出现的 token id，配合 ignore_eos 使用

# 真实运行时由 LLMEngine 设置（nanovllm/engine/llm_engine.py:21），测试手动对齐
Sequence.block_size = BLOCK_SIZE


def make_scheduler(num_blocks: int, max_num_batched_tokens: int = 10**6) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=512,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config)


def make_seq(num_tokens: int, max_tokens: int) -> Sequence:
    return Sequence(
        list(range(1, num_tokens + 1)),
        SamplingParams(max_tokens=max_tokens, ignore_eos=True),
    )


class TestFullLifecycle:
    def test_prefill_then_decode_until_finish(self):
        """申请 → 写入（哈希登记）→ decode 增长 → FINISHED 释放。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        seq = make_seq(10, max_tokens=2)  # 2 块（第 2 块半空）
        sched.add(seq)

        # 第 1 轮：完整 prefill
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True
        assert seqs == [seq]
        assert seq.num_scheduled_tokens == 10
        assert len(seq.block_table) == 2  # 申请：整个 prompt 的块一次分配
        assert seq.status == SequenceStatus.RUNNING
        assert list(sched.running) == [seq]

        sched.postprocess(seqs, [5000], True)
        assert seq.num_cached_tokens == 10
        assert len(bm.hash_to_block_id) == 1  # 只有完整块 0 登记了哈希
        assert seq.num_completion_tokens == 1
        assert seq.last_token == 5000
        assert not seq.is_finished

        # 第 2 轮：decode，token 落在原块内，无需新块
        seqs, is_prefill = sched.schedule()
        assert is_prefill is False
        assert seqs == [seq]
        assert seq.num_scheduled_tokens == 1
        assert len(seq.block_table) == 2

        sched.postprocess(seqs, [5001], False)
        assert seq.num_completion_tokens == seq.max_tokens
        assert seq.is_finished  # 达到 max_tokens
        assert seq.block_table == []  # 释放
        assert sched.is_finished()
        assert len(bm.free_block_ids) == 8  # 全部回到空闲池
        assert len(bm.hash_to_block_id) == 1  # 哈希保留，供后续前缀复用

    def test_chunked_prefill_allocates_once_and_progresses(self):
        """长 prompt 分块 prefill：块一次分配，进度由 num_cached_tokens 推进。"""
        sched = make_scheduler(num_blocks=8, max_num_batched_tokens=5)
        seq = make_seq(12, max_tokens=2)
        sched.add(seq)

        # 第 1 chunk：5 token
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True
        assert seq.num_scheduled_tokens == 5
        assert len(seq.block_table) == 2  # 整个 prompt 的 2 块一次性分配
        assert seq.status == SequenceStatus.WAITING  # prefill 未完成，仍在 waiting
        assert list(sched.waiting) == [seq] and not sched.running

        sched.postprocess(seqs, [6000], True)
        assert seq.num_cached_tokens == 5
        assert seq.num_completion_tokens == 0  # 中间 chunk 的采样 token 被丢弃
        assert 6000 not in seq.token_ids

        # 第 2 chunk：5 token
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True and seq.num_scheduled_tokens == 5
        sched.postprocess(seqs, [6001], True)
        assert seq.num_cached_tokens == 10

        # 第 3 chunk：最后 2 token，prefill 完成
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True and seq.num_scheduled_tokens == 2
        assert seq.status == SequenceStatus.RUNNING
        sched.postprocess(seqs, [6002], True)
        assert seq.num_completion_tokens == 1  # prefill 完成时产出首个 completion token

        # 第 4 轮：decode 出最后一个 token
        seqs, is_prefill = sched.schedule()
        assert is_prefill is False
        sched.postprocess(seqs, [6003], False)
        assert seq.is_finished and sched.is_finished()


class TestPreemption:
    def test_preemption_frees_blocks_and_reuses_them(self):
        """块不足 → 抢占队尾请求 → 释放的块立即被其他请求复用 → 被抢占请求恢复。"""
        sched = make_scheduler(num_blocks=2)  # 总容量 2 块
        bm = sched.block_manager
        sp = SamplingParams(max_tokens=3, ignore_eos=True)
        a = Sequence(list(range(1, 9)), sp)  # 8 token → 1 块
        b = Sequence(list(range(101, 109)), sp)  # 8 token → 1 块
        sched.add(a)
        sched.add(b)

        # 一轮 prefill 同时调度 a、b，各占 1 块，空闲块耗尽
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True and seqs == [a, b]
        sched.postprocess(seqs, [7000, 7001], True)
        assert len(a.block_table) == 1 and len(b.block_table) == 1
        assert len(bm.free_block_ids) == 0
        b_block = b.block_table[0]
        assert a.num_completion_tokens == 1 and b.num_completion_tokens == 1
        # 两条序列长度均为 9：下一个 decode token 都需要 1 个新块（9 % 8 == 1）

        # decode 轮：a 在队首被调度；b（队尾）被抢占，其块立即被 a 复用
        seqs, is_prefill = sched.schedule()
        assert is_prefill is False
        assert seqs == [a]
        assert b.status == SequenceStatus.WAITING  # 被抢占，回到 waiting 队首
        assert list(sched.waiting) == [b]
        assert b.block_table == []  # b 的块被释放
        assert len(a.block_table) == 2
        assert a.block_table[1] == b_block  # a 复用了 b 刚释放的物理块

        sched.postprocess(seqs, [7002], False)
        assert a.num_completion_tokens == 2
        assert not a.is_finished

        # 下一轮：b 的 prefill 因 can_allocate == -1 无法进行，只有 a 继续 decode
        seqs, is_prefill = sched.schedule()
        assert is_prefill is False and seqs == [a]
        sched.postprocess(seqs, [7003], False)
        assert a.is_finished  # a 达到 max_tokens=3
        assert a.block_table == []
        assert len(bm.free_block_ids) == 2  # a 的块全部释放

        # b 恢复：重新 prefill（prompt + 已生成的 1 个 token），进度不丢失
        assert 7001 in b.token_ids
        for _ in range(10):
            if sched.is_finished():
                break
            seqs, is_prefill = sched.schedule()
            sched.postprocess(seqs, [8000] * len(seqs), is_prefill)
        assert b.is_finished
        assert b.num_completion_tokens == 3
        assert 7001 in b.token_ids  # 被抢占前生成的 token 仍在
        assert sched.is_finished()


class TestPrefixReuseAcrossRequests:
    def test_finished_prefix_blocks_are_reused_by_next_request(self):
        """请求完成后哈希保留；相同前缀的新请求复用物理块，只 prefill 增量 token。"""
        sched = make_scheduler(num_blocks=8)
        bm = sched.block_manager
        sp = SamplingParams(max_tokens=1, ignore_eos=True)
        prompt = list(range(1, 17))  # 恰好 2 个完整块
        a = Sequence(prompt, sp)
        sched.add(a)

        seqs, is_prefill = sched.schedule()
        assert seqs == [a]
        a_table = list(a.block_table)
        sched.postprocess(seqs, [4000], True)
        assert a.is_finished  # max_tokens=1，prefill 后立即完成
        assert a.block_table == []
        assert len(bm.hash_to_block_id) == 2  # 2 个完整块的哈希保留

        c = Sequence(prompt + [9001, 9002], sp)  # 相同前缀 + 2 个新 token
        sched.add(c)
        seqs, is_prefill = sched.schedule()
        assert is_prefill is True and seqs == [c]
        assert c.num_scheduled_tokens == 2  # 只调度未命中的 2 个 token
        assert c.block_table[:2] == a_table[:2]  # 复用 a 留下的物理块
        assert c.num_cached_tokens == 16

        sched.postprocess(seqs, [4001], True)
        assert c.is_finished and sched.is_finished()
