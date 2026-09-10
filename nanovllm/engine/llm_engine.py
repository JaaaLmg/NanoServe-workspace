import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


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
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            # 空批次契约：schedule() 在调度边界清理（超时/取消/终态兜底）后
            # 可能清空队列。此时本轮无任何可执行请求，必须直接返回，
            # 不得把空批次交给 ModelRunner（decode 组装 block table 会崩溃）
            return [], 0
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只把正常完成的请求当作 completion 汇报；
        # CANCELLED/TIMEOUT 请求由后续 API 层根据 finish_reason 决定响应
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs
                   if seq.status == SequenceStatus.FINISHED]
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
                # 同步驱动的暂停契约：step 无进展且仍有活动请求，说明剩余请求
                # 全部处于暂停（PREEMPTED）状态、等待显式 resume()/cancel_request()。
                # 既不能把暂停伪装成已完成，也不能无界忙循环消耗 CPU——
                # 显式报错并指出需要处理的请求。
                paused = [seq.request_id for seq in self.scheduler.requests.values()
                          if seq.status == SequenceStatus.PREEMPTED]
                raise RuntimeError(
                    "generate() 无法推进：存在暂停（PREEMPTED）请求，而同步驱动"
                    "没有恢复机制。请先调用 scheduler.resume() 或 cancel_request()"
                    f" 处理这些请求后再重新驱动: {paused}"
                )
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
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
