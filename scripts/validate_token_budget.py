#!/usr/bin/env python
"""Day 7 每轮 Token Budget 验收脚本（docs/token-budget.md §6.4）。

两种模式：
- cpu（默认）：真实 Scheduler + 真实 LLMEngine.step 契约，ModelRunner 用桩替换
  （固定注入采样 token），固定单调时钟、固定请求集，无模型初始化、无 GPU 依赖；
- gpu：真实本地模型（不自动下载），参数沿用 Config；budget 只控制请求迭代，
  不改变物理 block 大小（默认 256）。

脚本职责：
1. 开启 nanovllm.engine 的 INFO 日志并以 JSONL 落盘全部结构化事件
   （scheduler_round / engine_round / budget_wait_episode / request_budget_wait）；
2. 独立解析日志校验：round_id 唯一、每轮 planned/executed <= 预算、
   人数口径可重算、episode 秒数可由起止时间重算、事件字段白名单；
3. 终端总结轮次数、非空轮次数、最大 executed_tokens、预算延后人数/轮次数/秒数、
   结束时 free/used block；日志解析失败或任一非空成功轮超限退出非零；
4. GPU 模式可选 --greedy-compare：对固定 prompt 对比充裕预算与受限预算的
   greedy 输出（结果如实记录，不伪造一致）。

示例：
    python scripts/validate_token_budget.py --mode cpu --output /tmp/day7-budget-cpu.jsonl
    python scripts/validate_token_budget.py --mode gpu --model "$MODEL_DIR" \\
        --token-budget 8 --max-num-seqs 16 --max-new-tokens 4 \\
        --output /tmp/day7-budget-gpu.jsonl --greedy-compare
"""

import argparse
import atexit
import gc
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# 允许从任意工作目录直接执行：把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

# 验收脚本负责开启 INFO（库代码不做 basicConfig，§6）
SCHED_LOGGER = logging.getLogger("nanovllm.engine.scheduler")
ENGINE_LOGGER = logging.getLogger("nanovllm.engine.llm_engine")

EVENT_FIELDS = {
    "scheduler_round": {"event", "round_id", "phase", "token_budget", "max_num_seqs",
                        "planned_tokens", "scheduled_requests", "budget_deferred_direct",
                        "budget_deferred_hol", "budget_deferred_requests",
                        "observed_at", "decisions"},
    "engine_round": {"event", "round_id", "phase", "token_budget", "planned_tokens",
                     "executed_tokens", "model_called", "outcome", "observed_at"},
    "budget_wait_episode": {"event", "seq_id", "request_id", "started_at", "ended_at",
                            "duration_seconds", "close_reason", "round_id", "observed_at"},
    "request_budget_wait": {"event", "seq_id", "request_id", "budget_deferred_rounds",
                            "budget_wait_seconds", "status", "finish_reason", "round_id", "observed_at"},
}
KNOWN_REASONS = {"scheduled", "budget", "sequence_cap", "kv_capacity",
                 "head_of_line", "phase_priority", "paused"}


class JsonlCapture(logging.Handler):
    """捕获引擎结构化日志并逐行写入 JSONL 文件（原始事件留档，供独立解析）。"""

    def __init__(self, path: Path):
        super().__init__(level=logging.INFO)
        self.path = path
        self.events = []
        self._raw = path.open("w", encoding="utf-8")

    def emit(self, record):
        message = record.getMessage()
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return  # 非 JSON 记录不属于结构化事件
        self.events.append(payload)
        self._raw.write(message + "\n")
        self._raw.flush()

    def close(self):
        self._raw.close()
        super().close()

    def write_record(self, payload: dict):
        """脚本自身的补充记录（run_config/汇总等）也写入同一 JSONL。"""
        self.events.append(payload)
        self._raw.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._raw.flush()


class FixedClock:
    """确定性单调时钟：CPU 模式注入 Scheduler，保证 trace 可复现。"""

    def __init__(self, start: float = 1000.0, step: float = 0.5):
        self.value = start
        self.step = step

    def tick(self) -> float:
        self.value += self.step
        return self.value

    def now(self) -> float:
        return self.value


def make_stub_runner(sampled_token: int = 7):
    """ModelRunner 桩：不加载权重，固定返回采样 token（ignore_eos 下由 max_tokens 终止）。"""
    def call(method, seqs, is_prefill):
        return [sampled_token] * len(seqs)
    return call


def build_cpu_engine(token_budget: int, max_num_seqs: int, num_blocks: int = 64,
                     block_size: int = 8) -> tuple[LLMEngine, FixedClock]:
    """CPU 模式：真实 Scheduler + 真实 step 契约，runner 为桩、时钟为固定值。"""
    Sequence.block_size = block_size
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        eos=-1,  # 桩采样恒为 7，不会命中 EOS
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    sched = Scheduler(config)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = sched
    engine.model_runner = SimpleNamespace(call=make_stub_runner())
    # 固定时钟注入：schedule 每调用一次推进一拍；postprocess 沿用同一时刻
    clock = FixedClock()
    real_schedule = sched.schedule
    real_postprocess = sched.postprocess

    def schedule_with_clock(**kwargs):
        return real_schedule(now=clock.tick())

    def postprocess_with_clock(seqs, token_ids, is_prefill, **kwargs):
        return real_postprocess(seqs, token_ids, is_prefill, now=clock.now())

    sched.schedule = schedule_with_clock          # type: ignore[assignment]
    sched.postprocess = postprocess_with_clock    # type: ignore[assignment]
    return engine, clock


def drive_engine(engine: LLMEngine, max_rounds: int) -> dict[int, list[int]]:
    """驱动引擎直至全部请求终态（有轮次上限防挂死），返回 seq_id -> completion tokens。"""
    outputs: dict[int, list[int]] = {}
    for _ in range(max_rounds):
        if engine.is_finished():
            break
        step_outputs, _ = engine.step()
        for seq_id, token_ids in step_outputs:
            outputs[seq_id] = token_ids
    if not engine.is_finished():
        # 有界驱动失败必须显式暴露，不能通过取消剩余请求伪造自然完成。
        raise RuntimeError(f"引擎在 {max_rounds} 轮内未完成，剩余请求: "
                           f"{list(engine.scheduler.requests)}")
    return outputs


def cpu_requests():
    """固定请求集：精确填满 / 不足延后 / 超预算分块 / decode 数量大于预算。"""
    specs = []
    # [3,5,2]：前两条恰好耗尽 B=8（第三条 direct budget）
    for n in (3, 5, 2):
        specs.append((list(range(1, n + 1)), 3))
    # [6,4,1]：第二条放不下即停，不跳过它选第三条
    for n in (6, 4, 1):
        specs.append((list(range(10, 10 + n)), 3))
    # 20 token：跨越两轮以上预算的分块 prefill
    specs.append((list(range(20, 40)), 2))
    # 12 条 1 token 短请求：全部 RUNNING 后 decode 数量 12 > B=8
    for _ in range(12):
        specs.append(([1], 4))
    return specs


def gpu_requests(token_budget: int, max_new_tokens: int, block_size: int):
    """GPU 固定请求集：含 B-1/B/B+1、跨多轮分块、prefix 复用与 decode 数量大于预算。

    - decode 证据：短请求数量随预算扩展（始终 > B），保证存在 decode 预算延后；
    - prefix 证据：共享前缀覆盖 2 个完整物理块（首块可缓存、末块按协议不缓存），
      第二条请求的 needed 应为 共享长度 - block_size（只计实际未缓存 token）。
    """
    specs = []
    for n in (token_budget - 1, token_budget, token_budget + 1):
        n = max(1, n)
        specs.append((list(range(100, 100 + n)), 2))       # prefill 计数与继承 chunking
    specs.append((list(range(200, 200 + 2 * token_budget + 4)), 2))  # 跨多轮分块
    shared = list(range(300, 300 + 2 * block_size))
    specs.append((shared, 2))  # prefix 复用对
    specs.append((shared, 2))
    for _ in range(max(12, token_budget + 4)):
        specs.append(([1], max_new_tokens))
    return specs


def add_requests(engine: LLMEngine, specs) -> list[int]:
    """注册请求（token 输入，绕过 tokenizer），返回注册顺序的 seq_id 列表。"""
    known = {seq.request_id: seq.seq_id for seq in engine.scheduler.requests.values()}
    seq_ids = []
    for token_ids, max_tokens in specs:
        rid = engine.add_request(token_ids, SamplingParams(
            max_tokens=max_tokens, ignore_eos=True, temperature=0.0))
        known_new = {seq.request_id: seq.seq_id
                     for seq in engine.scheduler.requests.values()}
        seq_ids.append(known_new[rid])
        known = known_new
    return seq_ids


def check_kv_capacity(engine: LLMEngine, specs):
    """GPU 自检：输入样例必须能容纳于物理 KV 池（容量不足不冒充预算问题）。"""
    block_size = engine.scheduler.block_size
    free = len(engine.scheduler.block_manager.free_block_ids)
    max_blocks = max((len(t) + max(1, m) + block_size - 1) // block_size
                     for t, m in specs)
    if max_blocks > free:
        raise RuntimeError(
            f"KV 容量自检失败：最长样例需要 {max_blocks} 块，池仅 {free} 块；"
            "请增大 gpu_memory_utilization 或缩短样例")


def validate_events(events: list[dict]) -> list[str]:
    """独立校验结构化事件（§6）：口径可重算、上限不越界、字段白名单。"""
    problems = []
    round_ids = [e["round_id"] for e in events if e.get("event") == "scheduler_round"]
    if len(set(round_ids)) != len(round_ids):
        problems.append("round_id 存在重复")
    if round_ids != sorted(round_ids):
        problems.append("round_id 非单调递增")
    required_events = {kind: set(fields) for kind, fields in EVENT_FIELDS.items()}
    scheduler_by_round = {e.get("round_id"): e for e in events
                          if e.get("event") == "scheduler_round"}
    engine_by_round = {e.get("round_id"): e for e in events
                       if e.get("event") == "engine_round"}
    for e in events:
        kind = e.get("event")
        if kind not in EVENT_FIELDS:
            continue
        missing = required_events[kind] - set(e)
        if missing:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 缺少字段 {sorted(missing)}")
        extra = set(e) - EVENT_FIELDS[kind]
        if extra:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 含未知字段 {sorted(extra)}")
        if kind == "scheduler_round":
            if not 0 <= e["planned_tokens"] <= e["token_budget"]:
                problems.append(f"round {e['round_id']}: planned_tokens={e['planned_tokens']} 超出预算 {e['token_budget']}")
            if e["scheduled_requests"] > e["max_num_seqs"]:
                problems.append(f"round {e['round_id']}: 批序列数超出 max_num_seqs")
            if e["budget_deferred_requests"] != e["budget_deferred_direct"] + e["budget_deferred_hol"]:
                problems.append(f"round {e['round_id']}: 预算人数口径不可重算")
            if sum(d["scheduled_tokens"] for d in e["decisions"]) != e["planned_tokens"]:
                problems.append(f"round {e['round_id']}: decisions 求和与 planned_tokens 不一致")
            engine = engine_by_round.get(e["round_id"])
            if engine is None:
                problems.append(f"round {e['round_id']}: 缺少对应 engine_round")
            elif (engine.get("phase") != e["phase"]
                  and not (e["phase"] == "idle" and engine.get("phase") == "idle")):
                problems.append(f"round {e['round_id']}: scheduler/engine phase 不一致")
            elif engine.get("planned_tokens") != e["planned_tokens"]:
                problems.append(f"round {e['round_id']}: scheduler/engine planned_tokens 不一致")
            for d in e["decisions"]:
                if d["reason"] not in KNOWN_REASONS:
                    problems.append(f"round {e['round_id']}: 未知原因 {d['reason']}")
                if d["scheduled_tokens"] < 0:
                    problems.append(f"round {e['round_id']}: scheduled_tokens 为负")
        elif kind == "engine_round":
            if e["outcome"] == "completed":
                if not (e["model_called"] and e["executed_tokens"] is not None
                        and e["executed_tokens"] == e["planned_tokens"]
                        and 0 < e["executed_tokens"] <= e["token_budget"]):
                    problems.append(f"round {e['round_id']}: 成功执行轮 executed={e['executed_tokens']} 与 planned={e['planned_tokens']}、budget={e['token_budget']} 不一致")
            elif e["outcome"] == "idle":
                if e["model_called"] or e["executed_tokens"] != 0:
                    problems.append(f"round {e['round_id']}: idle 轮不应调用模型")
            elif e["outcome"] == "error" and e["executed_tokens"] is not None:
                problems.append(f"round {e['round_id']}: error 轮不得虚报 executed_tokens")
        elif kind == "budget_wait_episode":
            if (e["duration_seconds"] < 0
                    or abs(e["duration_seconds"] - (e["ended_at"] - e["started_at"])) > 1e-6):
                problems.append(f"episode seq_id={e['seq_id']}: 秒数与起止时间不可重算")
    return problems


def validate_resources(sched: Scheduler, *, final: bool = False) -> list[str]:
    """校验活动索引、队列和物理块账本的一致性；资源泄漏必须使验收失败。"""
    problems = []
    free = list(sched.block_manager.free_block_ids)
    used = set(sched.block_manager.used_block_ids)
    all_ids = set(range(len(sched.block_manager.blocks)))
    if len(free) != len(set(free)):
        problems.append("free block 列表存在重复")
    if set(free) & used or set(free) | used != all_ids:
        problems.append("free/used block 集合不完整或相交")
    refs = {block.block_id: 0 for block in sched.block_manager.blocks}
    for seq in sched.requests.values():
        for block_id in seq.block_table:
            refs[block_id] += 1
    for block_id, expected in refs.items():
        if sched.block_manager.blocks[block_id].ref_count != expected:
            problems.append(f"block {block_id}: ref_count 与活动引用不一致")
    if final:
        if sched.requests or sched.waiting or sched.running or sched.budget_wait:
            problems.append("最终活动请求/队列/预算统计未清空")
        if used or any(block.ref_count != 0 for block in sched.block_manager.blocks):
            problems.append("最终仍有 used block 或非零引用")
        if set(free) != all_ids:
            problems.append("最终 free block 未覆盖全部物理块")
        active_ids = {seq.request_id for seq in sched.requests.values()}
        if sched._active_request_ids != active_ids:
            problems.append("活动 request_id 索引不一致")
    return problems


def summarize(capture: JsonlCapture, sched: Scheduler, mode: str) -> tuple[bool, str]:
    """独立校验 + 终端总结；返回 (是否通过, 摘要文本)。"""
    events = capture.events
    problems = validate_events(events)
    problems.extend(validate_resources(sched, final=True))
    rounds = [e for e in events if e.get("event") == "scheduler_round"]
    nonempty = [e for e in rounds if e["planned_tokens"] > 0]
    engine_rounds = [e for e in events if e.get("event") == "engine_round"]
    completed = [e for e in engine_rounds if e.get("outcome") == "completed"]
    max_executed = max((e["executed_tokens"] for e in completed), default=0)
    decode_deferred = [e for e in rounds
                       if e["phase"] == "decode" and e["budget_deferred_requests"] > 0]
    free = len(sched.block_manager.free_block_ids)
    used = len(sched.block_manager.used_block_ids)
    lines = [
        f"[{mode}] 调度轮次: {len(rounds)}（非空 {len(nonempty)}）",
        f"[{mode}] 成功执行轮: {len(completed)}，最大 executed_tokens: {max_executed}",
        f"[{mode}] 预算延后: 唯一请求 {sched.budget_deferred_unique_requests_total} 个 / "
        f"请求-轮次 {sched.budget_deferred_request_rounds_total} 次 / "
        f"已结算等待 {sched.budget_wait_closed_seconds_total:.3f} 秒",
        f"[{mode}] decode 受预算分批证据: {len(decode_deferred)} 轮存在 budget 延后",
        f"[{mode}] 结束时 KV block free/used: {free}/{used}",
    ]
    if not decode_deferred:
        problems.append("没有 decode 轮因预算延后：缺少 decode 预算生效证据")
    ok = not problems
    lines.append(f"[{mode}] 校验结果: {'PASS' if ok else 'FAIL'}")
    lines += [f"  - {p}" for p in problems]
    return ok, "\n".join(lines)


def teardown_engine(engine: LLMEngine):
    """显式释放模型与进程组，并注销 atexit 钩子避免二次调用。

    exit() 会删除 model_runner（权重与 KV cache 随之解除引用），
    再强制 gc + empty_cache 把显存真正归还设备——否则分配器缓存
    继续占用显存，后续引擎初始化会因配额不足而失败。
    """
    atexit.unregister(engine.exit)
    engine.exit()
    gc.collect()
    torch.cuda.empty_cache()


def write_totals(capture: JsonlCapture, sched: Scheduler, mode: str):
    capture.write_record({
        "event": "budget_totals", "mode": mode,
        "unique_requests": sched.budget_deferred_unique_requests_total,
        "request_rounds": sched.budget_deferred_request_rounds_total,
        "closed_seconds": sched.budget_wait_closed_seconds_total,
    })


def run_cpu(args, capture: JsonlCapture) -> bool:
    engine, _clock = build_cpu_engine(args.token_budget, args.max_num_seqs)
    specs = cpu_requests()
    capture.write_record({
        "event": "run_config", "mode": "cpu",
        "token_budget": args.token_budget, "max_num_seqs": args.max_num_seqs,
        "num_kvcache_blocks": 64, "block_size": 8, "request_count": len(specs),
    })
    add_requests(engine, specs)
    drive_engine(engine, max_rounds=2000)
    ok, text = summarize(capture, engine.scheduler, "cpu")
    write_totals(capture, engine.scheduler, "cpu")
    print(text)
    return ok


def run_gpu(args, capture: JsonlCapture) -> bool:
    engine = LLMEngine(args.model, max_num_batched_tokens=args.token_budget,
                       max_num_seqs=args.max_num_seqs, enforce_eager=True,
                       tensor_parallel_size=1)
    specs = gpu_requests(args.token_budget, args.max_new_tokens,
                         engine.scheduler.block_size)
    check_kv_capacity(engine, specs)
    capture.write_record({
        "event": "run_config", "mode": "gpu", "model": args.model,
        "token_budget": args.token_budget, "max_num_seqs": args.max_num_seqs,
        "max_new_tokens": args.max_new_tokens,
        "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
        "block_size": engine.scheduler.block_size,
        "enforce_eager": True, "tensor_parallel_size": 1,
        "request_count": len(specs),
    })
    add_requests(engine, specs)
    drive_engine(engine, max_rounds=5000)
    ok, text = summarize(capture, engine.scheduler, "gpu")
    write_totals(capture, engine.scheduler, "gpu")
    print(text)
    teardown_engine(engine)
    if args.greedy_compare:
        ok = greedy_compare(args, capture) and ok
    return ok


def greedy_compare(args, capture: JsonlCapture) -> bool:
    """§8.3：固定 prompt 在受限预算与充裕预算下的 greedy 输出对比（如实记录）。"""
    prompts = [
        "The capital of France is",
        "1 2 3 4 5 6 7 8 9",
        "Deep learning is a branch of",
        "Qwen3 is a large language model",
        "To be or not to be, that is",
    ]

    def run_once(token_budget: int) -> dict[str, list[int]]:
        """单次运行：注册顺序即输出顺序（同一引擎内 seq_id 单调），返回 prompt -> token_ids。"""
        engine = LLMEngine(args.model, max_num_batched_tokens=token_budget,
                           max_num_seqs=args.max_num_seqs, enforce_eager=True,
                           tensor_parallel_size=1)
        seq_to_prompt: dict[int, str] = {}
        for prompt in prompts:
            rid = engine.add_request(prompt, SamplingParams(
                max_tokens=args.max_new_tokens, temperature=0.0))
            for seq in engine.scheduler.requests.values():
                if seq.request_id == rid:
                    seq_to_prompt[seq.seq_id] = prompt
        outputs: dict[str, list[int]] = {}
        for _ in range(500):
            if engine.is_finished():
                break
            step_outputs, _ = engine.step()
            for seq_id, token_ids in step_outputs:
                outputs[seq_to_prompt[seq_id]] = token_ids
        teardown_engine(engine)
        return outputs

    constrained = run_once(args.token_budget)
    generous = run_once(4096)
    records = []
    for prompt in prompts:
        a, b = constrained.get(prompt), generous.get(prompt)
        records.append({"prompt_index": prompts.index(prompt), "match": a == b,
                        "tokens_constrained": a, "tokens_generous": b})
    missing = [r["prompt_index"] for r in records
               if r["tokens_constrained"] is None or r["tokens_generous"] is None]
    if missing:
        raise RuntimeError(f"greedy 对比缺少输出 prompt_index={missing}")
    match_count = sum(1 for r in records if r["match"])
    capture.write_record({"event": "greedy_compare", "budget": args.token_budget,
                          "prompt_count": len(prompts), "results": records})
    print(f"[gpu] greedy 对比（B={args.token_budget} vs 充裕预算 4096）: "
          f"{match_count}/{len(prompts)} 一致"
          f"（数值差异如实记录，不伪造一致）")
    return True  # 对比结果只记录，不单独作为通过条件（§8.3：出现差异分析并记录）


def main():
    parser = argparse.ArgumentParser(description="Day7 token budget 验收脚本")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", default="/tmp/day7-budget.jsonl")
    parser.add_argument("--model", default=None, help="GPU 模式的本地模型目录")
    parser.add_argument("--token-budget", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--greedy-compare", action="store_true",
                        help="GPU 模式：对比充裕预算与受限预算的 greedy 输出")
    args = parser.parse_args()
    if args.token_budget <= 0 or args.max_num_seqs <= 0:
        parser.error("--token-budget 与 --max-num-seqs 必须为正整数")
    if args.mode == "gpu" and not args.model:
        parser.error("gpu 模式必须提供 --model（本地模型目录，不自动下载）")

    capture = JsonlCapture(Path(args.output))
    SCHED_LOGGER.setLevel(logging.INFO)
    ENGINE_LOGGER.setLevel(logging.INFO)
    SCHED_LOGGER.addHandler(capture)
    ENGINE_LOGGER.addHandler(capture)
    SCHED_LOGGER.propagate = False
    ENGINE_LOGGER.propagate = False
    try:
        ok = run_cpu(args, capture) if args.mode == "cpu" else run_gpu(args, capture)
    finally:
        capture.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
