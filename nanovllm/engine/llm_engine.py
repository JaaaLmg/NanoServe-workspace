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
from nanovllm.engine.completed_request import AbortedRequest, CompletedRequest

# 模块重导出：服务层（Day11+）只依赖 LLMEngine 与这两类只读记录，
# 不需要感知 Scheduler 内部结构
__all__ = ["LLMEngine", "CompletedRequest", "AbortedRequest"]

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
        """幂等释放 Engine 资源；退出异常不能跳过 Scheduler/worker 清理。"""
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return
        errors = []
        # Scheduler 清理、runner 退出和 worker join 相互隔离：任一步失败都不能
        # 跳过后续资源回收，也不能覆盖最先发生的退出异常。
        try:
            scheduler = getattr(self, "scheduler", None)
            if scheduler is not None and scheduler.requests:
                scheduler.abort_all_active(reason="engine_exit")
        except BaseException as exc:
            errors.append(exc)
            logger.exception("Engine 退出时 Scheduler 清理失败")
        try:
            runner.call("exit")
        except BaseException as exc:
            errors.append(exc)
            logger.exception("ModelRunner 退出失败")
        finally:
            try:
                del self.model_runner
            except AttributeError:
                pass
            for process in getattr(self, "ps", []):
                try:
                    process.join()
                except BaseException as exc:
                    errors.append(exc)
                    logger.exception("张量并行 worker 回收失败")
            self.ps = []
        if errors:
            raise errors[0]

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
        return self.scheduler.get_request(request_id)

    @property
    def max_model_len(self) -> int:
        """只读暴露上下文上限：服务层在分配 KV 之前做长度预算检查（Day11–12）。"""
        return self.model_runner.config.max_model_len

    def has_active_requests(self) -> bool:
        """只读查询：是否存在未进入终态的请求。

        服务 worker 以此决定是否驱动 step()；无活动请求时必须休眠等待，
        不能连续空转 step。
        """
        return bool(self.scheduler.requests)

    def pop_completed(self) -> list[CompletedRequest]:
        """排出并清空正常完成记录（Day11–12 完成记录通道）。

        - step() 的既有 (outputs, num_tokens) 返回格式不变，本通道是并行的
          只读补充，不改变 generate() 等既有调用方的行为；
        - 控制锁内 drain：与 _finalize 的记录写入互斥，重复调用返回空列表
          （幂等），同一记录不会被返回两次；
        - 取消/超时/异常请求不出现在本通道，见 pop_aborted()。
        """
        with self.scheduler._control_lock:
            records = list(self.scheduler.completed_records)
            self.scheduler.completed_records.clear()
            return records

    def pop_aborted(self) -> list[AbortedRequest]:
        """排出并清空取消/超时/异常收尾记录，语义与 pop_completed() 相同。

        服务层据此把等待中的请求收口为明确失败（504/503/500），避免
        Future 永久悬挂；记录不含任何 token 明文。
        """
        with self.scheduler._control_lock:
            records = list(self.scheduler.aborted_records)
            self.scheduler.aborted_records.clear()
            return records

    def cancel_request(self, request_id: str, reason: str = "client_cancelled") -> bool:
        """取消指定请求（Day10 §4.3 信号语义）：请求不存在或已终态返回 False。

        两段式语义：
        - 先调用 Scheduler.request_cancel() 只置位控制面信号——该调用线程安全、
          可在模型 forward 期间发起，不触碰 block/队列/token；
        - Engine 空闲（当前无 step() 在驱动）时，立即在安全点完成 CANCELLED
          迁移与资源释放；Engine 正在 step() 时只保持信号，由本次 step 的
          postprocess 安全检查或下一轮 schedule() 边界扫描完成收尾——
          保证绝不中途修改正在被执行批次使用的 block_table。
        """
        # 查找与 signal 在 Scheduler 同一把锁内完成，避免 request_id 复用时
        # 先查到旧对象、后按 seq_id 操作的 TOCTOU 竞态。
        with self.scheduler._control_lock:
            seq = self.scheduler.get_request(request_id)
            if seq is None:
                return False
            signalled = seq.request_cancel(reason)
            if signalled and not getattr(self, "_in_step", False):
                self.scheduler.cancel(seq.seq_id, reason=reason)
            return signalled

    def step(self):
        """执行一轮调度 + 模型执行 + 逐 item 提交（Day10 §4.3 step 事务）。

        事务顺序：
        1. 检查 Engine 是否已失败（执行异常后禁止隐式重试，避免在不完整 KV 上重试）；
        2. scheduler.schedule()：调度边界清理取消/超时/终态，生成 BatchItem + round_id；
        3. 保存批次快照并显式校验计划；
        4. try: ModelRunner 调用 + postprocess 逐 item 提交
           except BaseException: abort_round（本轮回滚）+ abort_all_active（全活动收尾）
                                + 锁定 Engine + error 事件，原异常继续向上传播
           finally: 清除本轮临时计划与 round 标记（幂等兜底，保证不跨轮残留）；
        5. 仅返回 FINISHED 的正常 outputs；取消/超时不伪装为正常 completion。

        _in_step 全程置位（控制锁内翻转，与 cancel_request 的空闲判定共用同一
        锁序）：step 期间外部 cancel_request 只置位信号，不做破坏性清理，
        保证绝不中途修改正在被执行批次使用的 block_table。
        """
        if getattr(self, "_execution_failed", False):
            raise RuntimeError(
                "模型执行已失败，当前 Engine 禁止重试；请销毁并重新创建 Engine")
        with self.scheduler._control_lock:
            self._in_step = True
        try:
            return self._step()
        finally:
            with self.scheduler._control_lock:
                self._in_step = False

    def _step(self):
        """step 外层事务：连 schedule 阶段异常也必须进入失败收尾。"""
        try:
            return self._step_once()
        except BaseException as original_error:
            # 无论 _step_once 是否已经尝试过清理，都再执行一次幂等兜底；
            # 这样清理异常不会让残留 requests/blocks 被失败锁遮蔽。
            cleanup_errors = []
            if not getattr(self, "_execution_failed", False):
                self._execution_failed = True
                round_id = self.scheduler._current_round_id
                try:
                    _log_event({
                        "event": "engine_round", "round_id": round_id,
                        # 调度阶段尚未形成可执行 batch，使用合法的 idle phase；
                        # outcome=error 与 model_called=False 区分其异常语义。
                        "phase": "idle", "token_budget": self.scheduler.max_num_batched_tokens,
                        "planned_tokens": None, "executed_tokens": None,
                        "model_called": False, "outcome": "error",
                        "prefill_chunks": 0, "prefill_items": 0, "decode_items": 0,
                        "prefill_tokens": 0, "decode_tokens": 0,
                        "observed_at": perf_counter(),
                    })
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            try:
                if self.scheduler.requests:
                    self.scheduler.abort_all_active(reason="engine_error")
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.scheduler.block_manager.check_ledger()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            for cleanup_error in cleanup_errors:
                logger.error("调度异常清理失败（原始异常将保留）: %r", cleanup_error)
            raise
        finally:
            with self.scheduler._control_lock:
                self.scheduler._current_round_id = None

    def _step_once(self):
        # schedule 也属于 step 事务：allocate/may_append/状态迁移可能在
        # 修改账本后抛错，外层 _step 会统一执行失败收尾。
        items = []
        phase = "idle"
        try:
            items, phase = self.scheduler.schedule()
        except BaseException as original_error:
            # 保留本轮 schedule 已产生的半成品，由外层 abort_all_active 统一清理。
            raise original_error
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
            # 模型返回后的逐 item 提交也在同一事务内：postprocess 抛错同样
            # 触发批次回滚与全活动收尾（§4.3 异常安全目标——forward、采样、
            # postprocess 或资源记账异常都不泄漏资源）
            self.scheduler.postprocess(items, token_ids)
        except BaseException as original_error:
            # 模型/采样/postprocess/记账异常（§4.3/§5.4）：Engine 异常后明确
            # 不可重试。每个清理步骤独立隔离，任何清理异常都不能遮蔽原始异常，
            # 也不能阻止后续 abort_all_active 继续释放其余请求。
            self._execution_failed = True
            round_snapshot = None
            final_snapshot = None
            cleanup_errors = []

            try:
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
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                round_snapshot = self.scheduler.abort_round(items, reason="engine_error")
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                # 即使本轮对象清理失败，也必须继续清理等待/运行/暂停请求。
                final_snapshot = self.scheduler.abort_all_active(reason="engine_error")
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self.scheduler.block_manager.check_ledger()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                _log_event({
                    "event": "engine_abort_summary", "round_id": round_id,
                    "reason": "engine_error", "execution_failed": True,
                    "round_snapshot": round_snapshot,
                    "final_snapshot": final_snapshot,
                    "cleanup_errors": len(cleanup_errors),
                    "observed_at": perf_counter(),
                })
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            for cleanup_error in cleanup_errors:
                logger.error("Engine 异常清理失败（原始异常将保留）: %r", cleanup_error)
            raise original_error
        finally:
            # 幂等兜底（不变量 6/9）：无论成功或异常，本轮临时计划不得跨轮
            # 残留；状态和轮次标记也通过控制锁复位，避免与外部控制面并发写入。
            with self.scheduler._control_lock:
                for it in items:
                    it.seq.num_scheduled_tokens = 0
                self.scheduler._current_round_id = None
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
            # 离线 generate() 仍以 step() 返回值为结果来源；同时排出服务层
            # 完成记录，避免长期离线调用让控制面 deque 无界增长。
            self.pop_completed()
            self.pop_aborted()
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
                        "没有恢复机制。请先调用 scheduler.resume() 或 "
                        "engine.cancel_request()"
                        f" 处理这些请求后再重新驱动: {paused}"
                    )
                raise RuntimeError(
                    "generate() 无法推进：活动请求本轮无可执行候选（通常是 KV 容量"
                    "不足以接纳等待队列队首且无 running 请求可推进）。"
                    "请增大 num_kvcache_blocks / gpu_memory_utilization，或调用 "
                    f"engine.cancel_request() 移除无法容纳的请求。活动请求: "
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
