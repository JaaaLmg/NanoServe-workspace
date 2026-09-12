import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.scheduler import BatchItem
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.generators = {}
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # Day9 签名适配：run() 消费 BatchItem 列表。warmup 语义与 Day8 逐位一致：
        # prefill_offset=0、num_scheduled_tokens=seq_len、prefill_target=seq_len，
        # 每条序列都是"最后 chunk"（谓词成立，全部行采样）。warmup 不经过
        # Scheduler/postprocess，round_id=0 仅作占位（不会被消费）。
        items = [BatchItem(seq=seq, phase="prefill",
                           scheduled_tokens=seq.num_scheduled_tokens,
                           offset_before=0, is_last_chunk=True,
                           needs_sample=True, round_id=0)
                 for seq in seqs]
        self.run(items)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        (input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
         slot_mapping, has_block_table) = self._build_prefill_inputs(seqs, self.block_size)
        block_tables = None
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache：历史 KV 直接读缓存
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @staticmethod
    def _build_prefill_inputs(seqs: list[Sequence], block_size: int):
        """CPU 纯组装逻辑（无 CUDA 依赖，可被单元测试直接调用）。

        Day8 显式 offset 契约：对每个 seq 消费
          start = seq.prefill_offset        # 已提交 KV 的上下文长度（唯一进度事实源）
          q     = seq.num_scheduled_tokens  # 本轮接纳的 chunk 大小
          end   = start + q
        并显式校验 0 <= start < end <= prefill_target（否则抛错，不静默组装）。
        - input_ids  取 seq[start:end]，跨 chunk 时是正确的后续片段；
        - positions  使用绝对位置 range(start, end)，绝不从 0 重置；
        - cu_seqlens_k 每段为 end = 历史有效 KV + 当前 query，causal 语义由
          flash_attn_varlen_func 的 varlen 约定保证；
        - slot_mapping 覆盖 [start, end) 的物理槽位，按 block_table 跨块切分。
        返回 (input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_q, max_k,
              slot_mapping, has_block_table)。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        has_block_table = True
        for seq in seqs:
            start = seq.prefill_offset
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            # 显式范围校验：区间必须落在有效上下文内且非空，否则拒绝组装
            if not (0 <= start < end <= seq.prefill_target):
                raise ValueError(
                    f"seq_id={seq.seq_id}: 非法 prefill 区间 [{start}, {end})，"
                    f"prefill_target={seq.prefill_target}，"
                    f"num_scheduled_tokens={seqlen_q}")
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            if not seq.block_table:    # warmup：无物理块，不组装 slot
                has_block_table = False
                continue
            start_block = start // block_size
            end_block = (end + block_size - 1) // block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * block_size
                if i == start_block:
                    slot_start += start % block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * block_size + block_size
                else:
                    slot_end = seq.block_table[i] * block_size + end - i * block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # 展平长度校验（显式 raise，python -O 下仍生效，§4.3）：
        # input_ids / positions / cu_seqlens_q[-1] 必须一致；
        # slot_mapping 仅对真实批次（全部持有 block_table）要求一一对应，
        # warmup 批次不组装 slot，豁免该校验
        total_q = cu_seqlens_q[-1]
        if len(input_ids) != total_q or len(positions) != total_q:
            raise ValueError(
                f"prefill 输入长度不一致：input_ids={len(input_ids)}, "
                f"positions={len(positions)}, cu_seqlens_q[-1]={total_q}")
        if has_block_table and len(slot_mapping) != total_q:
            raise ValueError(
                f"slot_mapping 长度 {len(slot_mapping)} 与 query 数 {total_q} 不一致")
        return (input_ids, positions, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, slot_mapping, has_block_table)

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping, context_lens = self._build_decode_inputs(seqs, self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @staticmethod
    def _build_decode_inputs(seqs: list[Sequence], block_size: int):
        """CPU 纯组装逻辑（无 CUDA 依赖，可被单元测试直接调用）。

        Day8 元数据修正：positions/context_lens 一律使用序列元数据
        num_tokens，而不是可能依赖 token_ids 长度的口径——TP worker
        反序列化后的 decode 序列 token_ids 为空（仅保留 last_token），
        任何基于 token_ids 的推导都存在潜在错误。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(seq.num_tokens - 1)
            context_lens.append(seq.num_tokens)
            slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
        return input_ids, positions, slot_mapping, context_lens

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        generators = []
        for seq in seqs:
            if seq.seed is None:
                generators.append(None)
                continue
            generator = self.generators.get(seq.seq_id)
            if generator is None:
                generator = torch.Generator(device="cuda").manual_seed(seq.seed)
                self.generators[seq.seq_id] = generator
            generators.append(generator)
        return temperatures, top_ps, generators

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, items: list):
        """Day9 混合轮执行：同一调度轮内按 phase 分组，先 decode 子批后 prefill 子批。

        §2.2 设计决策（不实现单次 fused mixed attention）：Context/Attention/
        ParallelLMHead 仍按单阶段批次消费——两个子批各自构造输入、执行前向并
        reset_context，互不残留 Context 状态；decode 子批沿用既有 CUDA Graph
        路径（enforce_eager=False 且 bs<=512 时）。

        采样结果合并契约（§4.1）：items 有序（decode 在前、prefill 在后），
        返回的 token 列表 = decode 子批采样结果 + prefill 最后 chunk 采样结果，
        与 items 中 needs_sample 的出现顺序一致——postprocess 按同一顺序消费，
        两端共享同一对齐契约，任何一端不得隐式重排。
        整轮无任何采样 item 时返回 None（沿用 Day8"无采样返回 None"契约）。
        TP>1：所有 rank 都要执行两个子批的前向（KV 写入必须全 rank 一致），
        仅采样与 logits 消费限定 rank 0。
        """
        token_ids: list[int] = []
        decode_items = [it for it in items if it.phase == "decode"]
        prefill_items = [it for it in items if it.phase == "prefill"]
        if len(decode_items) + len(prefill_items) != len(items):
            unknown = {it.phase for it in items} - {"prefill", "decode"}
            raise ValueError(f"批次存在未知 phase 的 item: {sorted(unknown)}")
        # ---------- decode 子批（在前）：last_token 单步前向 ----------
        if decode_items:
            try:
                seqs = [it.seq for it in decode_items]
                input_ids, positions = self.prepare_decode(seqs)
                logits = self.run_model(input_ids, positions, False)
                if self.rank == 0:
                    temperatures, top_ps, generators = self.prepare_sample(seqs)
                    token_ids.extend(
                        self.sampler(logits, temperatures, top_ps, generators).tolist())
            finally:
                # 无论组装、前向还是采样是否异常，都不能把 decode Context
                # 泄漏给后续 prefill 子批或下一轮。
                reset_context()
        # ---------- prefill 子批（在后）：显式 offset 契约组装 ----------
        if prefill_items:
            try:
                seqs = [it.seq for it in prefill_items]
                input_ids, positions = self.prepare_prefill(seqs)
                logits = self.run_model(input_ids, positions, True)
                if self.rank == 0:
                    # ---------- prefill 采样契约（Day8 沿用） ----------
                    # 1) 行选择：ParallelLMHead 在 prefill 分支已按 cu_seqlens_q[1:]-1
                    #    把每个请求的"最后一个 query 行"聚合为 [len(seqs), vocab]；
                    #    显式校验行数与请求一一对应，形状错位不再静默。
                    # 2) 中间 chunk 不采样、不消耗 RNG：torch.multinomial 会推进
                    #    generator 状态，若中间 chunk 也采样，chunk 划分不同就会改变
                    #    RNG 流，"chunked 与 one-shot 输出一致"在随机采样下不可达。
                    #    因此只有最后 chunk 才进入采样。
                    # 3) 行选择以 BatchItem.needs_sample（调度冻结快照）为准，并与
                    #    Sequence 具名谓词交叉校验——两者同源，不一致说明调度与执行
                    #    之间请求状态被破坏，尽早显式失败。
                    if logits.shape[0] != len(seqs):
                        raise ValueError(
                            f"prefill logits 行数 {logits.shape[0]} 与请求数 {len(seqs)} 不一致，"
                            "每请求应恰好聚合出其最后 query 行")
                    sample_idx = [i for i, it in enumerate(prefill_items) if it.needs_sample]
                    predicate_idx = self._select_prefill_sample_rows(seqs)
                    if sample_idx != predicate_idx:
                        raise ValueError(
                            f"BatchItem.needs_sample 快照 {sample_idx} 与执行侧谓词 "
                            f"{predicate_idx} 不一致，调度与执行之间的请求状态被破坏")
                    if sample_idx:
                        temperatures, top_ps, generators = self.prepare_sample(seqs)
                        token_ids.extend(self.sampler(
                            logits[sample_idx],
                            temperatures[sample_idx],
                            top_ps[sample_idx],
                            [generators[i] for i in sample_idx],
                        ).tolist())
            finally:
                # 同样保证 prefill 异常不会把全局 Context 带入下一轮。
                reset_context()
        if self.rank != 0:
            # worker 只负责前向与 KV 写入，不消费 logits/不采样（TP 语义沿用）
            return None
        return token_ids if token_ids else None

    @staticmethod
    def _select_prefill_sample_rows(seqs: list[Sequence]):
        """CPU 纯逻辑（可被单元测试直接调用）：选出本轮需采样的请求下标。

        只有最后 chunk（"本轮接纳即完成 prefill"具名谓词，Day9 单一权威）的
        请求才采样。返回的批内下标同时是聚合后 logits [len(seqs), vocab]
        的行号：ParallelLMHead 已按 cu_seqlens_q[1:]-1 把每个请求的最后
        query 行放到第 i 行，无需再按扁平 token 位偏移选行。
        """
        sample_idx = []
        for i, seq in enumerate(seqs):
            if seq.is_last_chunk_scheduled:
                sample_idx.append(i)
        return sample_idx

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
