"""Day 3 验收测试：BlockManager 的分配 / 释放 / 复用行为。

对应 plan.md Day 3 任务 4：
- 验证 block 不足时的返回值：``can_allocate`` 返回 -1、``can_append`` 返回 False；
- 验证释放后可复用性：块回到空闲池可被重新分配，且 prefix 哈希仍能命中并
  直接复用同一物理块。

BlockManager / Sequence 是纯 CPU 逻辑，不需要 GPU、模型权重和 torch。
真实运行时 ``Sequence.block_size`` 由 LLMEngine 按 config 设置
（nanovllm/engine/llm_engine.py:21），测试环境在模块导入时手动对齐。
"""

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 8
Sequence.block_size = BLOCK_SIZE


def make_seq(num_tokens: int) -> Sequence:
    # token id 从 1 开始，避免与未初始化块的 token_ids=[] 混淆
    return Sequence(list(range(1, num_tokens + 1)), SamplingParams())


def run_prefill_and_hash(bm: BlockManager, seq: Sequence):
    """模拟调度器对一次完整 prefill 的记账：先分配，postprocess 时登记块哈希。"""
    num_cached = bm.can_allocate(seq)
    assert num_cached != -1
    bm.allocate(seq, num_cached)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)


class TestAllocate:
    def test_allocate_takes_exactly_num_blocks(self):
        bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
        seq = make_seq(20)  # ceil(20/8) = 3 个块
        assert bm.can_allocate(seq) == 0  # 空池无缓存可命中
        bm.allocate(seq, 0)
        assert len(seq.block_table) == 3
        assert len(set(seq.block_table)) == 3  # 3 个不同的物理块
        assert all(bm.blocks[b].ref_count == 1 for b in seq.block_table)
        assert seq.num_cached_tokens == 0
        assert set(seq.block_table) == bm.used_block_ids
        assert len(bm.free_block_ids) == 4 - 3

    def test_can_allocate_returns_minus1_when_blocks_insufficient(self):
        bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
        assert bm.can_allocate(make_seq(20)) == -1  # 需要 3 块 > 空闲 2 块
        # 边界：恰好等于空闲块数时不返回 -1，而是可正常分配
        assert bm.can_allocate(make_seq(16)) == 0

    def test_insufficient_blocks_caused_by_occupancy(self):
        bm = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
        holder = make_seq(8)
        bm.allocate(holder, 0)  # 占用 1 块，剩 2 块
        assert bm.can_allocate(make_seq(24)) == -1  # 需要 3 块 > 空闲 2 块
        assert bm.can_allocate(make_seq(16)) == 0


class TestDeallocate:
    def test_deallocate_returns_blocks_to_free_pool(self):
        bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
        seq = make_seq(12)  # 2 块
        bm.allocate(seq, 0)
        bm.deallocate(seq)
        assert seq.block_table == []
        assert seq.num_cached_tokens == 0
        assert bm.used_block_ids == set()
        assert len(bm.free_block_ids) == 4

    def test_blocks_reusable_after_deallocate(self):
        """释放后的块可被后续更大的请求重新分配（释放后可复用性）。"""
        bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
        seq1 = make_seq(12)  # 2 块
        bm.allocate(seq1, 0)
        bm.deallocate(seq1)
        seq2 = make_seq(32)  # 4 块，只有全部释放才能满足
        assert bm.can_allocate(seq2) == 0
        bm.allocate(seq2, 0)
        assert len(seq2.block_table) == 4

    def test_deallocate_shared_block_only_frees_last_reference(self):
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq1 = make_seq(20)
        run_prefill_and_hash(bm, seq1)  # 2 个完整块已登记哈希，seq1 仍在运行
        seq2 = Sequence(seq1.token_ids[:16] + [51, 52, 53, 54, 55], SamplingParams())
        assert bm.can_allocate(seq2) == 2
        bm.allocate(seq2, 2)
        # 前两个块被两条序列共享
        assert seq2.block_table[:2] == seq1.block_table[:2]
        assert all(bm.blocks[b].ref_count == 2 for b in seq1.block_table[:2])

        bm.deallocate(seq1)  # 第一条结束，共享块引用数减一但不释放
        assert all(bm.blocks[b].ref_count == 1 for b in seq2.block_table[:2])
        assert set(seq2.block_table[:2]) <= bm.used_block_ids

        bm.deallocate(seq2)  # 最后一个引用释放，块才真正回到空闲池
        assert bm.used_block_ids == set()
        assert len(bm.free_block_ids) == 8


class TestPrefixCache:
    def test_hash_blocks_only_hashes_full_blocks_with_chained_hash(self):
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq = make_seq(20)  # 2 个完整块 + 1 个半空块
        bm.allocate(seq, 0)
        seq.num_scheduled_tokens = 20
        bm.hash_blocks(seq)

        h0 = BlockManager.compute_hash(seq.block(0), -1)
        b0, b1, b2 = (bm.blocks[b] for b in seq.block_table)
        assert b0.hash == h0
        # 块哈希把前一块的哈希混入，是位置相关的链式哈希
        assert b1.hash == BlockManager.compute_hash(seq.block(1), h0)
        assert bm.hash_to_block_id[b0.hash] == b0.block_id
        assert bm.hash_to_block_id[b1.hash] == b1.block_id
        # 尾部半空块不登记哈希（内容还会继续追加）
        assert b2.hash == -1 and b2.token_ids == []
        assert b2.block_id not in bm.hash_to_block_id.values()

    def test_finished_sequence_prefix_is_reused_by_hash(self):
        """请求结束后块被释放，但哈希保留；相同前缀的新请求直接复用物理块。"""
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq1 = make_seq(20)
        run_prefill_and_hash(bm, seq1)
        table1 = list(seq1.block_table)
        bm.deallocate(seq1)  # 模拟 FINISHED

        assert len(bm.free_block_ids) == 8  # 块全部释放
        assert len(bm.hash_to_block_id) == 2  # 哈希仍保留

        seq2 = Sequence(seq1.token_ids[:16] + [101, 102, 103, 104, 105], SamplingParams())
        assert bm.can_allocate(seq2) == 2  # 命中 2 个缓存块
        bm.allocate(seq2, 2)
        assert seq2.block_table[:2] == table1[:2]  # 复用同一物理块
        assert seq2.num_cached_tokens == 16
        # 复用块的内容保持不变：物理 KV 无需重写，prefill 只需计算未命中的 token
        for i in range(2):
            assert bm.blocks[seq2.block_table[i]].token_ids == seq2.block(i)

    def test_prefix_match_breaks_on_first_mismatched_block(self):
        """链式哈希要求从头连续匹配：首块不同，即使后续块内容相同也不命中。"""
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq1 = make_seq(20)
        run_prefill_and_hash(bm, seq1)

        # 首块换成不同 token，第二块与 seq1 相同
        seq2 = Sequence(list(range(50, 58)) + seq1.token_ids[8:16] + [1, 2, 3], SamplingParams())
        assert bm.can_allocate(seq2) == 0  # 无命中，但也不是 -1
        bm.allocate(seq2, 0)
        assert set(seq2.block_table).isdisjoint(seq1.block_table[:2])

    def test_token_ids_mismatch_rejected_even_if_hash_matches(self):
        """哈希命中后还会逐块比对 token_ids，防止哈希碰撞或脏块导致错误复用。"""
        bm = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq = make_seq(8)
        run_prefill_and_hash(bm, seq)

        other = make_seq(8)  # 内容不同的另一个首块
        h = BlockManager.compute_hash(other.block(0), -1)
        # 白盒注入：人为制造"哈希指向内容不符的块"
        bm.hash_to_block_id[h] = seq.block_table[0]
        assert bm.can_allocate(other) == 0  # token_ids 校验失败，不复用

    def test_shared_prefix_block_does_not_consume_free_capacity(self):
        """命中"仍在使用"的共享块只增加引用计数，不消耗新的空闲块。"""
        bm = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
        seq1 = make_seq(20)
        run_prefill_and_hash(bm, seq1)  # 占 3 块，剩 1 块
        seq2 = Sequence(seq1.token_ids[:16] + [61, 62, 63, 64, 65], SamplingParams())  # 3 块
        # 若共享块也要消耗空闲容量，则 1 < 3 会返回 -1；实际共享块打折后只需 1 块
        assert bm.can_allocate(seq2) == 2


class TestDecodeAppend:
    def test_can_append_boundary_conditions(self):
        bm = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
        seq = make_seq(8)  # 恰好 1 个整块
        bm.allocate(seq, 0)

        assert bm.can_append(seq) is True  # len=8: 8%8==0，本步无需新块
        bm.may_append(seq)
        assert len(seq.block_table) == 1

        seq.append_token(999)  # len=9: 9%8==1，下一个 decode token 落入新块
        assert bm.can_append(seq) is True  # 还有 1 个空闲块
        bm.may_append(seq)
        assert len(seq.block_table) == 2

        seq.append_token(998)  # len=10: 10%8==2，本块内还有空位
        assert bm.can_append(seq) is True

    def test_can_append_false_when_no_free_block(self):
        """decode 需要新块但池已空：调度器据此触发抢占。"""
        bm = BlockManager(num_blocks=1, block_size=BLOCK_SIZE)
        seq = make_seq(8)
        bm.allocate(seq, 0)  # 唯一的块已被占用
        seq.append_token(999)  # len=9 → 需要新块
        assert bm.can_append(seq) is False
