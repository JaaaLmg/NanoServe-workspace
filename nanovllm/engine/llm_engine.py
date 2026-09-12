import atexit
import json
import logging
from dataclasses import fields
from time import perf_counter
from typing import NamedTuple

from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

# Engine 侧结构化日志：与 Scheduler 共用调用方控制的日志配置，库内不做 basicConfig
logger = logging.getLogger(__name__)


def _log_event(payload: dict):
    """以 JSON Lines 发出 Engine 事件；INFO 未开启时不构造字符串。"""
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps(payload, ensure_ascii=False))


class _ItemSnapshot(NamedTuple):
    """模型调用前逐 item 快照（§4.2，rank 0 统计用途，不进 TP payload）。

    postprocess 会把临时计数清零，不能在其后倒推执行工作量，因此在模型调用
    前固化。offset_before/is_last_chunk 为 Day8 chunk 观测字段（is_last_chunk
    决定该请求本轮是否产出采样 token），round_id 供日志关联与迟到结果核对。
    用 NamedTuple 而非裸元组：字段自解释、按名访问，抗后续字段增删。
    """
    seq_id: int
    request_id: str
    phase: str
    scheduled_tokens: int
    offset_before: int | None
    is_last_chunk: bool
    round_id: int


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 模型执行异常后不支持在同一 Engine 上重试：避免复用未完成的 KV
        # 或重复追加 block；调用方应销毁引擎并重新创建实例。
        self._execution_failed = False
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,
                    request_id: str | None = None, deadline: float | None = None) -> str:
        """注册请求并返回可追踪的 request_id。

        HTTP 层（Day11+）可凭该 ID 取消请求、关联结果和串联日志。
        deadline 使用单调时钟（perf_counter）语义。
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, request_id=request_id, deadline=deadline)
        self.scheduler.add(seq)
        return seq.request_id

    def get_request(self, request_id: str) -> Sequence | None:
        """按 request_id 查找活动请求；请求进入终态后已被清理，返回 None。"""
        for seq in self.scheduler.requests.values():
            if seq.request_id == request_id:
                return seq
        return None

    def cancel_request(self, request_id: str, reason: str = "client_cancelled") -> bool:
        """取消指定请求（调度层幂等）：请求不存在或已终态返回 False。"""
        seq = self.get_request(request_id)
        if seq is None:
            return False
        return self.scheduler.cancel(seq.seq_id, reason=reason)

    def step(self):
        if getattr(self, "_execution_failed", False):
            raise RuntimeError(
                "模型执行已失败，当前 Engine 禁止重试；请销毁并重新创建 Engine")
        items, phase = self.scheduler.schedule()
        # 调度计划快照：round_id / planned_tokens / 分阶段量取自 Scheduler 的本轮
        # 统计，是计划与执行日志按 round_id 关联的唯一权威（rank 0 为统计所有者）
        sched_stats = self.scheduler.last_schedule_stats or {}
        round_id = sched_stats.get("round_id")
        budget = sched_stats.get("token_budget", self.scheduler.max_num_batched_tokens)
        if not items:
            # 空批次契约：schedule() 在调度边界清理（超时/取消/终态兜底）后
            # 可能清空队列，或等待队列因 KV 不足暂不可执行。此时本轮无任何
            # 可执行请求，必须直接返回、不调用 ModelRunner（decode 组装
            # block table 会崩溃），并记录 idle 执行事件说明原因。
            # 事件字段按"事件类型"穷举（Day8 教训）：分阶段字段在 idle 轮补 0，
            # 保证所有 engine_round 事件字段集合一致（验收脚本白名单校验）
            _log_event({
                "event": "engine_round", "round_id": round_id, "phase": "idle",
                "token_budget": budget, "planned_tokens": 0,
                "executed_tokens": 0, "model_called": False, "outcome": "idle",
                # 空轮无任何子批：各分阶段字段与 decode 轮同为 0，保持字段契约统一
                "prefill_chunks": 0,
                "prefill_items": 0, "decode_items": 0,
                "prefill_tokens": 0, "decode_tokens": 0,
                "observed_at": perf_counter(),
            })
            return [], 0
        # 调用前快照：postprocess 会把临时计数清零，不能在其后据此倒推执行工作量；
        # 因此在模型调用前保存逐 item 的 _ItemSnapshot（字段说明见其 docstring）
        snapshot = [_ItemSnapshot(it.seq.seq_id, it.seq.request_id, it.phase,
                                  it.scheduled_tokens, it.offset_before,
                                  it.is_last_chunk, it.round_id) for it in items]
        prefill_item_count = sum(1 for s in snapshot if s.phase == "prefill")
        prefill_tokens = sum(s.scheduled_tokens for s in snapshot
                             if s.phase == "prefill")
        decode_count = len(snapshot) - prefill_item_count
        planned = prefill_tokens + decode_count
        # 调用前显式校验批次（混合口径 §4.2，不依赖会被 python -O 移除的 assert）：
        # 非空批次每个计数必须为正整数、phase 合法、decode 恒为 1、seq_id 不重复、
        # 且"prefill token 总量 + decode 条数"不超过本轮 token 预算
        if any(type(s.scheduled_tokens) is not int or s.scheduled_tokens <= 0
               for s in snapshot):
            raise ValueError(
                f"round {round_id}: 非空批次存在非正的 scheduled_tokens: {snapshot}")
        if any(s.phase not in ("prefill", "decode") for s in snapshot):
            raise ValueError(
                f"round {round_id}: 批次存在未知 phase 的 item: {snapshot}")
        if any(s.scheduled_tokens != 1 for s in snapshot if s.phase == "decode"):
            raise ValueError(
                f"round {round_id}: decode item 的接纳数必须为 1: {snapshot}")
        seq_ids = [s.seq_id for s in snapshot]
        if len(set(seq_ids)) != len(seq_ids):
            raise ValueError(f"round {round_id}: 批次内 seq_id 重复: {seq_ids}")
        if planned > budget:
            raise ValueError(
                f"round {round_id}: 计划输入 token {planned} 超过预算 {budget}")
        # Day9 起工作量为本轮总 query token 数（恒非负），不再用符号编码阶段；
        # 分阶段工作量见 engine_round 事件的 prefill_tokens/decode_tokens
        num_tokens = planned
        try:
            token_ids = self.model_runner.call("run", items)
        except Exception:
            # 模型异常不假记成功执行：记录 error 与 planned_tokens，
            # executed_tokens=null（工作量未知）；异常继续向上传播。
            # 失败锁定避免调用方在未完成 KV 上直接重试。
            self._execution_failed = True
            _log_event({
                "event": "engine_round", "round_id": round_id, "phase": phase,
                "token_budget": budget, "planned_tokens": planned,
                "executed_tokens": None, "model_called": True, "outcome": "error",
                "prefill_chunks": prefill_item_count,
                "prefill_items": prefill_item_count,
                "decode_items": decode_count,
                "prefill_tokens": prefill_tokens, "decode_tokens": decode_count,
                "observed_at": perf_counter(),
            })
            raise
        # 正常返回：executed 取自调用前快照（模型实际输入 query token 数），
        # 即使 postprocess 因取消/超时丢弃采样输出，已执行输入仍计入本轮预算
        executed = planned
        _log_event({
            "event": "engine_round", "round_id": round_id, "phase": phase,
            "token_budget": budget, "planned_tokens": planned,
            "executed_tokens": executed, "model_called": True, "outcome": "completed",
            # Day8：本轮 prefill 子批的 chunk 数（decode/idle 轮为 0）；
            # Day9：prefill_items 与 prefill_chunks 同值（同一口径的规范名与别名）
            "prefill_chunks": prefill_item_count,
            "prefill_items": prefill_item_count,
            "decode_items": decode_count,
            "prefill_tokens": prefill_tokens, "decode_tokens": decode_count,
            "observed_at": perf_counter(),
        })
        self.scheduler.postprocess(items, token_ids)
        # 只把正常完成的请求当作 completion 汇报；
        # CANCELLED/TIMEOUT 请求由后续 API 层根据 finish_reason 决定响应
        outputs = [(it.seq.seq_id, it.seq.completion_token_ids) for it in items
                   if it.seq.status == SequenceStatus.FINISHED]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if not output and num_tokens == 0 and not self.is_finished():
                # 同步驱动的暂停契约：step 无进展且仍有活动请求，说明本轮没有
                # 可执行候选。两种来源：剩余请求全部处于暂停（PREEMPTED）状态、
                # 等待显式 resume()/cancel_request()；或等待队列因 KV 容量不足
                # 无法接纳且无 running 请求可推进。既不能把无进展伪装成已完成，
                # 也不能无界忙循环消耗 CPU——显式报错并指出需要处理的请求。
                paused = [seq.request_id for seq in self.scheduler.requests.values()
                          if seq.status == SequenceStatus.PREEMPTED]
                if paused:
                    raise RuntimeError(
                        "generate() 无法推进：存在暂停（PREEMPTED）请求，而同步驱动"
                        "没有恢复机制。请先调用 scheduler.resume() 或 cancel_request()"
                        f" 处理这些请求后再重新驱动: {paused}"
                    )
                raise RuntimeError(
                    "generate() 无法推进：活动请求本轮无可执行候选（通常是 KV 容量"
                    "不足以接纳等待队列队首且无 running 请求可推进）。"
                    "请增大 num_kvcache_blocks / gpu_memory_utilization，或调用 "
                    f"scheduler.cancel_request() 移除无法容纳的请求。活动请求: "
                    f"{[seq.request_id for seq in self.scheduler.requests.values()]}"
                )
            # Day9：吞吐按分阶段计划量计算（num_tokens 不再用符号编码阶段，
            # 混合轮的 prefill/decode 工作量分别计入两个吞吐口径）
            stats = self.scheduler.last_schedule_stats or {}
            elapsed = perf_counter() - t
            if stats.get("prefill_tokens"):
                prefill_throughput = stats["prefill_tokens"] / elapsed
            if stats.get("decode_tokens"):
                decode_throughput = stats["decode_tokens"] / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
