from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        # 先验证完整账本，再窥视并校验 free 队首；坏账本不能在抛错前
        # 通过 popleft() 被进一步破坏。
        self.check_ledger()
        if not self.free_block_ids:
            raise ValueError("无法分配 block：free block 为空")
        block_id = self.free_block_ids[0]
        if type(block_id) is not int or not 0 <= block_id < len(self.blocks):
            raise ValueError(f"无法分配非法 block ID: {block_id!r}")
        if block_id in self.used_block_ids:
            raise ValueError(f"无法分配同时出现在 used 集合中的 block {block_id}")
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise ValueError(f"无法分配 block {block_id}: ref_count={block.ref_count}")
        self.free_block_ids.popleft()
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        if self.blocks[block_id].ref_count != 0:
            raise ValueError(
                f"无法释放 block {block_id}: ref_count={self.blocks[block_id].ref_count}")
        if block_id not in self.used_block_ids:
            raise ValueError(f"无法释放未占用的 block {block_id}")
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为请求分配 KV；验证和实际记账失败时恢复完整账本快照。"""
        snapshot = self._snapshot_state(seq)
        try:
            self._allocate_unchecked(seq, num_cached_blocks)
        except BaseException:
            self._restore_state(seq, snapshot)
            raise

    def _snapshot_state(self, seq: Sequence) -> dict:
        return {
            "free": list(self.free_block_ids),
            "used": set(self.used_block_ids),
            "hash": dict(self.hash_to_block_id),
            "blocks": [(b.ref_count, b.hash, list(b.token_ids)) for b in self.blocks],
            "table": list(seq.block_table),
            "offset": seq.prefill_offset,
        }

    def _restore_state(self, seq: Sequence, snapshot: dict):
        self.free_block_ids.clear()
        self.free_block_ids.extend(snapshot["free"])
        self.used_block_ids.clear()
        self.used_block_ids.update(snapshot["used"])
        self.hash_to_block_id.clear()
        self.hash_to_block_id.update(snapshot["hash"])
        for block, state in zip(self.blocks, snapshot["blocks"]):
            block.ref_count, block.hash, token_ids = state
            block.token_ids = list(token_ids)
        seq.block_table[:] = snapshot["table"]
        seq.prefill_offset = snapshot["offset"]

    def _allocate_unchecked(self, seq: Sequence, num_cached_blocks: int):
        if seq.block_table:
            raise ValueError("只能为尚未持有 block 的请求分配 KV")
        if type(num_cached_blocks) is not int or not 0 <= num_cached_blocks <= seq.num_blocks:
            raise ValueError(f"非法 prefix 命中块数: {num_cached_blocks!r}")
        # 先验证所有 prefix 命中块和所需新块的容量，再开始任何 ref_count 变更。
        h = -1
        cached_ids = []
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is None or type(block_id) is not int or not 0 <= block_id < len(self.blocks):
                raise ValueError(f"prefix cache 返回非法 block ID: {block_id!r}")
            if self.blocks[block_id].token_ids != token_ids:
                raise ValueError(f"prefix cache block {block_id} 内容不匹配")
            cached_ids.append(block_id)
        new_count = seq.num_blocks - num_cached_blocks
        if len(set(cached_ids)) != len(cached_ids):
            raise ValueError("prefix cache 命中 block 重复，拒绝分配")
        cached_free_count = sum(block_id not in self.used_block_ids for block_id in cached_ids)
        if len(self.free_block_ids) < new_count + cached_free_count:
            raise ValueError("KV 空闲 block 不足，无法原子分配请求")
        for block_id in cached_ids:
            if block_id not in self.used_block_ids and block_id not in self.free_block_ids:
                raise ValueError(f"prefix cache block {block_id} 不在 free/used 账本中")
        h = -1
        for block_id in cached_ids:
            token_ids = seq.block(len(seq.block_table))
            h = self.compute_hash(token_ids, h)
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                self.free_block_ids.remove(block_id)
                block.ref_count = 1
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(new_count):
            seq.block_table.append(self._allocate_block())
        seq.prefill_offset = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放请求持有的 block；完整校验后再修改账本，失败可回滚。"""
        snapshot = self._snapshot_state(seq)
        try:
            block_ids = list(seq.block_table)
            if len(set(block_ids)) != len(block_ids):
                raise ValueError(f"请求 block_table 存在重复 block ID: {block_ids}")
            invalid = [bid for bid in block_ids
                       if type(bid) is not int or not 0 <= bid < len(self.blocks)]
            if invalid:
                raise ValueError(f"请求 block_table 存在非法 block ID: {invalid}")
            not_used = [bid for bid in block_ids if bid not in self.used_block_ids]
            if not_used:
                raise ValueError(f"请求 block_table 包含未占用 block: {not_used}")
            bad_ref = [bid for bid in block_ids if self.blocks[bid].ref_count <= 0]
            if bad_ref:
                raise ValueError(f"请求 block_table 包含无有效引用 block: {bad_ref}")
            for block_id in reversed(block_ids):
                block = self.blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            seq.prefill_offset = 0
            seq.block_table.clear()
        except BaseException:
            self._restore_state(seq, snapshot)
            raise

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def check_ledger(self):
        """CPU 可核对的账本守恒检查（Day10 §4.5）。

        校验四条不变量，任何违反立即显式报错（不用 assert——python -O 会剥离
        断言，验收路径必须始终生效）：
        1. free/used 集合互斥；
        2. len(free) + len(used) == len(blocks)（总数守恒，无 block 凭空出现/丢失）；
        3. 每个 used block 的 ref_count > 0（持有者账目一致，防 double free 后残留）；
        4. 每个 free block 的 ref_count == 0（释放必然清空引用）。
        hash_to_block_id 允许指向 free block（抢占释放后 prefix 元数据暂留是
        设计内行为，§4.5），因此不做哈希映射一致性检查。
        """
        free_list = list(self.free_block_ids)
        free_ids = set(free_list)
        used_ids = set(self.used_block_ids)
        valid_ids = set(range(len(self.blocks)))
        if len(free_ids) != len(free_list):
            raise ValueError("KV 账本破坏：free block deque 存在重复 ID")
        invalid_free = free_ids - valid_ids
        invalid_used = used_ids - valid_ids
        if invalid_free or invalid_used:
            raise ValueError(
                f"KV 账本破坏：存在越界 block，free={sorted(invalid_free)}, "
                f"used={sorted(invalid_used)}")
        overlap = free_ids & used_ids
        if overlap:
            raise ValueError(f"KV 账本破坏：free/used 集合不互斥，重叠 {sorted(overlap)}")
        if free_ids | used_ids != valid_ids:
            missing = valid_ids - (free_ids | used_ids)
            raise ValueError(f"KV 账本破坏：缺少 block ID {sorted(missing)}")
        if len(free_ids) + len(used_ids) != len(self.blocks):
            raise ValueError(
                f"KV 账本破坏：free({len(free_ids)}) + used({len(used_ids)}) "
                f"!= 总块数({len(self.blocks)})")
        bad_used = sorted(b for b in used_ids if self.blocks[b].ref_count <= 0)
        if bad_used:
            raise ValueError(f"KV 账本破坏：used block 引用计数 <= 0: {bad_used}")
        bad_free = sorted(b for b in free_ids if self.blocks[b].ref_count != 0)
        if bad_free:
            raise ValueError(f"KV 账本破坏：free block 仍被引用: {bad_free}")

    def hash_blocks(self, seq: Sequence, start: int, end: int):
        """把本次执行写满的完整块登记进 prefix 索引（Day8 显式区间契约）。

        start/end 是本次成功执行的 query 区间 [start, end)（token 下标），
        由调用方（Scheduler.postprocess）以 offset_before/offset_before+q
        显式给出，不再从 num_cached_tokens/num_scheduled_tokens 内部推导。
        只有被本次执行写满的块（块尾 <= end）才登记；尾块永不登记，
        沿用 can_allocate 的 range(num_blocks - 1) 协议，避免尾块假命中。
        中间 chunk 写满的块同样登记——这是 chunked prefill 仍能享受
        prefix 复用的前提。start==end（空区间）是安全空操作。
        """
        start_block = start // self.block_size
        # floor 除：未写满的尾块所在块不进入登记范围
        end_block = end // self.block_size
        if start_block == end_block: return
        # 链式哈希从前一个已登记块延续；start_block>0 时该块必然已写满且
        # 已登记（chunk 区间连续），或来自 prefix 命中（命中块自带哈希）
        h = self.blocks[seq.block_table[start_block - 1]].hash if start_block > 0 else -1
        for i in range(start_block, end_block):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
