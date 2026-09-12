"""Day 8 Chunked Prefill 验收脚本（docs/chunked-prefill.md §9.3/§9.4）。

GPU 模式覆盖：
- 8K 主场景：8192-token prompt 在 chunk_size=256/512/1024 三档下分块完成且不
  OOM（KV 容量自检先行）；offset 事件序列连续覆盖 [0, 8192)；
- greedy 一致性：至少 5 个固定 prompt，one-shot（chunk_size 覆盖 prompt 长度）
  vs chunked（受限 chunk_size）最终 token_ids 对比（差异如实记录，不伪造一致）；
- prefix 命中：共享前缀覆盖 2 个完整物理块，第二条请求初始 offset 正确、
  只执行未缓存部分；
- 性能矩阵（§9.4）：chunk_size 256/512/1024 与 one-shot 对照的端到端 wall time、
  TTFT（首 token 观测点为最后 chunk 完成轮）、torch.cuda 显存峰值、完成状态与
  轮次数，原始数据写入 JSONL（只报告观测值，不承诺改善方向）。

CPU 模式：真实 Scheduler + 桩 runner 的分块回归（离线 sanity，无 GPU/权重）。

证据：结构化事件（scheduler_round/engine_round，含 Day8 的 chunk_index/
offset_before/is_last_chunk 决策字段与 prefill_chunks 观测）+ 脚本补充记录
（run_config/offset 轨迹/一致性对比/性能矩阵）写入同一 JSONL。
事件不含 prompt/token 明文；token 对比记录仅含 token id 序列。

用法：
    python scripts/validate_chunked_prefill.py --mode cpu --output /tmp/day8.jsonl
    python scripts/validate_chunked_prefill.py --mode gpu --model <本地模型目录> \
        --output docs/evidence/day8/day8-gpu.jsonl
"""

import argparse
import atexit
import gc
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

SCHED_LOGGER = logging.getLogger("nanovllm.engine.scheduler")
ENGINE_LOGGER = logging.getLogger("nanovllm.engine.llm_engine")

# 事件字段白名单（Day8 版本：engine_round 增加 prefill_chunks）
EVENT_FIELDS = {
    "scheduler_round": {"event", "round_id", "phase", "token_budget", "max_num_seqs",
                        "planned_tokens", "scheduled_requests", "budget_deferred_direct",
                        "budget_deferred_hol", "budget_deferred_requests",
                        "observed_at", "decisions"},
    "engine_round": {"event", "round_id", "phase", "token_budget", "planned_tokens",
                     "executed_tokens", "model_called", "outcome", "observed_at",
                     "prefill_chunks"},
    "budget_wait_episode": {"event", "seq_id", "request_id", "started_at", "ended_at",
                            "duration_seconds", "close_reason", "round_id", "observed_at"},
    "request_budget_wait": {"event", "seq_id", "request_id", "budget_deferred_rounds",
                            "budget_wait_seconds", "status", "finish_reason",
                            "round_id", "observed_at"},
}
KNOWN_REASONS = {"scheduled", "budget", "sequence_cap", "kv_capacity",
                 "head_of_line", "phase_priority", "paused"}


class JsonlCapture(logging.Handler):
    """捕获引擎结构化日志并逐行写入 JSONL（原始事件留档，供独立解析）。"""

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
            return
        self.events.append(payload)
        self._raw.write(message + "\n")
        self._raw.flush()

    def close(self):
        self._raw.close()
        super().close()

    def write_record(self, payload: dict):
        """脚本自身的补充记录（run_config/轨迹/对比/矩阵）写入同一 JSONL。"""
        self.events.append(payload)
        self._raw.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._raw.flush()


def make_stub_runner(sampled_token: int = 7):
    def call(method, seqs, is_prefill):
        # Day8 提交契约：中间 chunk 无采样 token；最后 chunk / decode 每序列 1 token
        if not is_prefill:
            return [sampled_token] * len(seqs)
        n = sum(1 for s in seqs
                if s.prefill_offset + s.num_scheduled_tokens == s.prefill_target)
        return [sampled_token] * n if n else None
    return call


def build_cpu_engine(max_num_batched_tokens: int, chunk_size: int,
                     num_blocks: int = 256, block_size: int = 8):
    """CPU 模式：真实 Scheduler + 桩 runner；chunk_size 显式提供。"""
    Sequence.block_size = block_size
    config = SimpleNamespace(
        max_num_seqs=64,
        max_num_batched_tokens=max_num_batched_tokens,
        chunk_size=chunk_size,
        eos=-1,  # 桩采样恒为 7，不会命中 EOS
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    sched = Scheduler(config)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = sched
    engine.model_runner = SimpleNamespace(call=make_stub_runner())
    return engine


def collect_offset_trajectory(engine: LLMEngine, seq_id: int) -> list[dict]:
    """从本轮调度统计提取该请求的 prefill chunk 决策快照（附 round_id）。

    只保留带 chunk 观测字段（offset_before 非空）的 scheduled 决策——
    decode 轮的 scheduled 决策没有这些字段。
    """
    stats = engine.scheduler.last_schedule_stats or {}
    round_id = stats.get("round_id")
    return [dict(d, round_id=round_id) for d in stats.get("decisions", [])
            if d.get("seq_id") == seq_id and d.get("offset_before") is not None]


def validate_events(events: list[dict], chunk_size: int | None = None) -> list[str]:
    """独立校验结构化事件：字段白名单、预算上限、chunk 上限、offset 连续性。"""
    problems = []
    required = {kind: set(fields) for kind, fields in EVENT_FIELDS.items()}
    sched_rounds = {}
    for e in events:
        kind = e.get("event")
        if kind not in EVENT_FIELDS:
            continue
        missing = required[kind] - set(e)
        if missing:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 缺少字段 {sorted(missing)}")
        extra = set(e) - EVENT_FIELDS[kind]
        if extra:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 含未知字段 {sorted(extra)}")
        if kind == "scheduler_round":
            if not 0 <= e["planned_tokens"] <= e["token_budget"]:
                problems.append(f"round {e['round_id']}: planned_tokens 超出预算")
            if sum(d["scheduled_tokens"] for d in e["decisions"]) != e["planned_tokens"]:
                problems.append(f"round {e['round_id']}: decisions 求和与 planned 不一致")
            for d in e["decisions"]:
                if d["reason"] not in KNOWN_REASONS:
                    problems.append(f"round {e['round_id']}: 未知原因 {d['reason']}")
                if d["reason"] == "scheduled" and chunk_size is not None:
                    if not 0 < d["scheduled_tokens"] <= chunk_size:
                        problems.append(
                            f"round {e['round_id']}: q={d['scheduled_tokens']} "
                            f"超过 chunk_size={chunk_size} 或非正")
            sched_rounds[e["round_id"]] = e
        if kind == "engine_round" and e["outcome"] == "completed":
            rid = e["round_id"]
            s = sched_rounds.get(rid)
            expected_chunks = sum(1 for d in (s or {}).get("decisions", [])
                                  if d["reason"] == "scheduled") if s and s["phase"] == "prefill" else 0
            if e["prefill_chunks"] != expected_chunks:
                problems.append(
                    f"round {rid}: prefill_chunks={e['prefill_chunks']} 与调度决策 {expected_chunks} 不一致")

    # offset 连续性：同一请求各轮 prefill chunk 区间 [offset_before, offset_before+q)
    # 按轮次排序后必须连续、单调、无重叠、无遗漏；且至多一个 is_last_chunk。
    # decode 轮的 scheduled 决策无 chunk 观测字段（offset_before=None），跳过
    trajectory: dict[int, list[tuple]] = {}
    for e in events:
        if e.get("event") != "scheduler_round":
            continue
        for d in e["decisions"]:
            if d["reason"] != "scheduled" or d["offset_before"] is None:
                continue
            trajectory.setdefault(d["seq_id"], []).append(
                (e["round_id"], d["offset_before"], d["scheduled_tokens"],
                 d["is_last_chunk"], d["chunk_index"]))
    for seq_id, chunks in trajectory.items():
        chunks.sort()
        pos = None
        last_chunk_count = 0
        chunk_indices = []
        for _rid, offset_before, q, is_last, chunk_index in chunks:
            if pos is not None and offset_before != pos:
                problems.append(
                    f"seq_id={seq_id}: chunk 区间不连续（期望起点 {pos}，实际 {offset_before}）")
            pos = offset_before + q
            last_chunk_count += 1 if is_last else 0
            chunk_indices.append(chunk_index)
        if last_chunk_count != 1:
            problems.append(
                f"seq_id={seq_id}: is_last_chunk 出现 {last_chunk_count} 次（应为 1）")
        if chunk_indices != list(range(1, len(chunk_indices) + 1)):
            problems.append(
                f"seq_id={seq_id}: chunk_index 序列 {chunk_indices} 不是从 1 连续递增")
    return problems


def check_kv_capacity(engine: LLMEngine, total_tokens: int):
    """GPU 容量自检：8K 样例必须能容纳于物理 KV 池（不足时报配置错误，§8 风险表）。"""
    block_size = engine.scheduler.block_size
    need_blocks = (total_tokens + block_size - 1) // block_size
    free = len(engine.scheduler.block_manager.free_block_ids)
    if need_blocks > free:
        raise RuntimeError(
            f"KV 容量自检失败：{total_tokens} token 需要 {need_blocks} 块，池仅 {free} 块；"
            "请增大 gpu_memory_utilization 或 num_kvcache_blocks")


def teardown_engine(engine: LLMEngine):
    """显式释放模型与进程组并归还显存（分配器缓存不清空会挤占后续引擎初始化）。"""
    atexit.unregister(engine.exit)
    engine.exit()
    gc.collect()
    torch.cuda.empty_cache()


def gpu_env_record() -> dict:
    """环境记录口径（GPU 型号/显存/torch/驱动等）。"""
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 2 ** 30, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }


def scenario_8k(args, capture: JsonlCapture) -> bool:
    """8K 主场景 + §9.4 性能矩阵：chunk_size 256/512/1024 与 one-shot 对照。

    one-shot 对照：B 与 chunk_size 均为 8192（>= prompt 长度），单轮完成 prefill；
    chunked 档位：B=args.token_budget，chunk_size 依次 1024/512/256。
    """
    problems = []
    env = gpu_env_record()
    capture.write_record({"event": "run_config", "scenario": "8k", **env,
                          "prompt_tokens": 8192, "max_new_tokens": 2,
                          "max_num_batched_tokens": args.token_budget,
                          "temperature": 0.0})
    matrix = []
    # (chunk_size, token_budget, is_one_shot)：one-shot 需要预算覆盖整段 prompt
    runs = [(8192, 8192, True)] + [(cs, args.token_budget, False)
                                   for cs in (1024, 512, 256)]
    for chunk_size, budget, is_one_shot in runs:
        engine = LLMEngine(args.model, max_num_batched_tokens=budget,
                           max_num_seqs=args.max_num_seqs, enforce_eager=True,
                           tensor_parallel_size=1, chunk_size=chunk_size,
                           max_model_len=args.max_model_len)
        block_size = engine.scheduler.block_size
        total = 8192 + 2
        check_kv_capacity(engine, total)
        capture.write_record({
            "event": "run_config", "scenario": "8k", "chunk_size": chunk_size,
            "token_budget": budget, "is_one_shot": is_one_shot,
            "block_size": block_size,
            "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
            "kv_capacity_check_tokens": total,
        })
        torch.cuda.reset_peak_memory_stats()
        prompt = list(range(100, 100 + 8192))
        rid = engine.add_request(prompt, SamplingParams(max_tokens=2, temperature=0.0))
        seq_id = next(s.seq_id for s in engine.scheduler.requests.values()
                      if s.request_id == rid)
        # 逐轮记录 offset 轨迹（决策快照不含 prompt/token 明文）
        trajectory = []
        t0 = time.perf_counter()
        ttft = None
        rounds = 0
        finished = False
        for _ in range(200):
            if engine.is_finished():
                finished = True
                break
            engine.step()
            rounds += 1
            for d in collect_offset_trajectory(engine, seq_id):
                trajectory.append({"round_id": d["round_id"],
                                   "chunk_index": d["chunk_index"],
                                   "offset_before": d["offset_before"],
                                   "scheduled_tokens": d["scheduled_tokens"],
                                   "is_last_chunk": d["is_last_chunk"]})
            seq = engine.scheduler.requests.get(seq_id)
            if ttft is None and (seq is None or seq.num_completion_tokens >= 1):
                ttft = time.perf_counter() - t0
        wall = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        capture.write_record({"event": "offset_trajectory", "scenario": "8k",
                              "chunk_size": chunk_size, "seq_id": seq_id,
                              "chunks": trajectory})
        capture.write_record({"event": "perf_matrix", "scenario": "8k",
                              "chunk_size": chunk_size, "token_budget": budget,
                              "is_one_shot": is_one_shot,
                              "wall_time": wall, "ttft": ttft, "rounds": rounds,
                              "finished": finished,
                              "max_memory_allocated": peak_alloc,
                              "max_memory_reserved": peak_reserved})
        matrix.append({"chunk_size": chunk_size, "token_budget": budget,
                       "is_one_shot": is_one_shot,
                       "wall_time": wall, "ttft": ttft, "rounds": rounds,
                       "finished": finished,
                       "max_memory_allocated": peak_alloc,
                       "max_memory_reserved": peak_reserved})
        # 独立重算 offset 连续性：区间并集必须恰为 [0, 8192)
        pos = 0
        for c in trajectory:
            if c["offset_before"] != pos:
                problems.append(
                    f"chunk_size={chunk_size}: offset 不连续（期望 {pos}，"
                    f"实际 {c['offset_before']}）")
            pos = c["offset_before"] + c["scheduled_tokens"]
        if pos != 8192:
            problems.append(f"chunk_size={chunk_size}: 区间并集 {pos} != 8192")
        if not finished:
            problems.append(f"chunk_size={chunk_size}: 8K 请求未完成")
        print(f"[gpu-8k] chunk_size={chunk_size} B={budget}"
              f"{'（one-shot 对照）' if is_one_shot else ''}: "
              f"rounds={rounds} wall={wall:.2f}s ttft={ttft:.2f}s "
              f"peak_alloc={peak_alloc / 2 ** 20:.0f}MiB")
        teardown_engine(engine)
    capture.write_record({"event": "perf_matrix_summary", "scenario": "8k",
                          "matrix": matrix})
    print(f"[gpu-8k] 校验结果: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"  - {p}")
    return not problems


GREEDY_PROMPTS = [
    "The capital of France is",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20",
    "Deep learning is a branch of machine learning that",
    "Qwen3 is a large language model developed by Alibaba",
    "To be or not to be, that is the question",
    "Artificial intelligence has transformed many industries including",
]


def scenario_greedy(args, capture: JsonlCapture) -> bool:
    """greedy 一致性：one-shot vs chunked 的最终 token_ids 逐 token 对比。"""
    capture.write_record({"event": "run_config", "scenario": "greedy",
                          "chunk_size_one_shot": 8192,
                          "chunk_size_chunked": args.greedy_chunk_size,
                          "prompt_count": len(GREEDY_PROMPTS),
                          "max_new_tokens": args.max_new_tokens, "temperature": 0.0,
                          **gpu_env_record()})

    def run_once(chunk_size: int) -> dict[int, list[int]]:
        """单次运行：返回 prompt_index -> token_ids（seq_id 跨引擎不通用，按下标对齐）。"""
        engine = LLMEngine(args.model, max_num_batched_tokens=args.token_budget,
                           max_num_seqs=args.max_num_seqs, enforce_eager=True,
                           tensor_parallel_size=1, chunk_size=chunk_size,
                           max_model_len=args.max_model_len)
        seq_to_index: dict[int, int] = {}
        for i, prompt in enumerate(GREEDY_PROMPTS):
            rid = engine.add_request(prompt, SamplingParams(
                max_tokens=args.max_new_tokens, temperature=0.0))
            seq_to_index[next(s.seq_id for s in engine.scheduler.requests.values()
                              if s.request_id == rid)] = i
        outputs: dict[int, list[int]] = {}
        for _ in range(500):
            if engine.is_finished():
                break
            step_outputs, _ = engine.step()
            for seq_id, token_ids in step_outputs:
                outputs[seq_to_index[seq_id]] = token_ids
        teardown_engine(engine)
        return outputs

    one_shot = run_once(8192)                       # chunk_size 覆盖所有 prompt
    chunked = run_once(args.greedy_chunk_size)      # 受限 chunk_size：多轮拆分
    records = []
    for i in range(len(GREEDY_PROMPTS)):
        a, b = one_shot.get(i), chunked.get(i)
        records.append({"prompt_index": i, "match": a == b,
                        "tokens_one_shot": a, "tokens_chunked": b})
    missing = [r["prompt_index"] for r in records
               if r["tokens_one_shot"] is None or r["tokens_chunked"] is None]
    if missing:
        raise RuntimeError(f"greedy 对比缺少输出 prompt_index={missing}")
    match_count = sum(1 for r in records if r["match"])
    capture.write_record({"event": "greedy_compare",
                          "chunk_size_chunked": args.greedy_chunk_size,
                          "prompt_count": len(GREEDY_PROMPTS), "results": records})
    print(f"[gpu-greedy] chunked(chunk_size={args.greedy_chunk_size}) vs one-shot: "
          f"{match_count}/{len(GREEDY_PROMPTS)} 一致（差异如实记录，不伪造一致）")
    return True  # 对比结果只记录；出现差异时在验收记录中分析（§4.6）


def scenario_prefix(args, capture: JsonlCapture) -> bool:
    """prefix 命中：共享前缀 2 个完整物理块，第二条请求只执行未缓存部分。"""
    engine = LLMEngine(args.model, max_num_batched_tokens=args.token_budget,
                       max_num_seqs=args.max_num_seqs, enforce_eager=True,
                       tensor_parallel_size=1, chunk_size=args.chunk_size,
                       max_model_len=args.max_model_len)
    block_size = engine.scheduler.block_size
    shared_len = 2 * block_size                     # 2 个完整物理块
    shared = list(range(500, 500 + shared_len))
    check_kv_capacity(engine, shared_len + block_size + 4)
    capture.write_record({"event": "run_config", "scenario": "prefix",
                          "chunk_size": args.chunk_size, "block_size": block_size,
                          "shared_prefix_tokens": shared_len, **gpu_env_record()})
    # 第一条请求完成 prefill，使共享块进入 prefix 索引
    rid_a = engine.add_request(shared + [9001, 9002],
                               SamplingParams(max_tokens=2, temperature=0.0))
    sid_a = next(s.seq_id for s in engine.scheduler.requests.values()
                 if s.request_id == rid_a)
    for _ in range(50):
        if engine.is_finished():
            break
        engine.step()
    # 第二条请求共享同一前缀：初始 offset 应为 2*block_size，只执行新增部分
    rid_b = engine.add_request(shared + [9101, 9102, 9103],
                               SamplingParams(max_tokens=2, temperature=0.0))
    sid_b = next(s.seq_id for s in engine.scheduler.requests.values()
                 if s.request_id == rid_b)
    trajectory = []
    for _ in range(50):
        if engine.is_finished():
            break
        engine.step()
        for d in collect_offset_trajectory(engine, sid_b):
            trajectory.append(d)
    initial = trajectory[0] if trajectory else {}
    expected_executed = 3                            # B 比 A 多 3 个新 token
    ok = (initial.get("offset_before") == shared_len
          and sum(d["scheduled_tokens"] for d in trajectory) == expected_executed)
    capture.write_record({"event": "prefix_check", "seq_id": sid_b,
                          "initial_offset": initial.get("offset_before"),
                          "expected_initial_offset": shared_len,
                          "executed_tokens": sum(d["scheduled_tokens"] for d in trajectory),
                          "expected_executed": expected_executed, "pass": ok,
                          "chunks": trajectory})
    print(f"[gpu-prefix] 初始 offset={initial.get('offset_before')}（期望 {shared_len}），"
          f"执行 token={sum(d['scheduled_tokens'] for d in trajectory)}"
          f"（期望 {expected_executed}）: "
          f"{'PASS' if ok else 'FAIL'}")
    teardown_engine(engine)
    return ok


def run_cpu(args, capture: JsonlCapture) -> bool:
    """CPU 模式：桩 runner 驱动分块场景，校验事件契约与 offset 连续性。"""
    engine = build_cpu_engine(args.token_budget, args.chunk_size)
    capture.write_record({"event": "run_config", "mode": "cpu",
                          "token_budget": args.token_budget,
                          "chunk_size": args.chunk_size,
                          "num_kvcache_blocks": 256, "block_size": 8})
    specs = [list(range(1, 101)), list(range(200, 213)), [1], [2]]
    for tokens in specs:
        engine.add_request(tokens, SamplingParams(max_tokens=3, ignore_eos=True))
    for _ in range(500):
        if engine.is_finished():
            break
        engine.step()
    problems = validate_events(capture.events, chunk_size=args.chunk_size)
    if not engine.is_finished():
        problems.append("CPU 引擎未在有界轮次内完成")
    print(f"[cpu] 校验结果: {'PASS' if not problems else 'FAIL'}")
    for p in problems:
        print(f"  - {p}")
    return not problems


def run_gpu(args, capture: JsonlCapture) -> bool:
    selected = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    unknown = selected - {"8k", "greedy", "prefix"}
    if unknown:
        parser_error = f"未知场景: {sorted(unknown)}"
        raise SystemExit(parser_error)
    ok = True
    if "8k" in selected:
        ok = scenario_8k(args, capture) and ok
    if "greedy" in selected:
        ok = scenario_greedy(args, capture) and ok
    if "prefix" in selected:
        ok = scenario_prefix(args, capture) and ok
    # 全场景事件独立校验（chunk_size 各档不同，这里只做与档位无关的口径校验）
    problems = validate_events(capture.events)
    if problems:
        print("[gpu] 事件校验 FAIL:")
        for p in problems:
            print(f"  - {p}")
        ok = False
    else:
        print("[gpu] 事件校验 PASS")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Day8 chunked prefill 验收脚本")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output", default="/tmp/day8-chunked.jsonl")
    parser.add_argument("--model", default=None, help="GPU 模式的本地模型目录")
    parser.add_argument("--token-budget", type=int, default=2048,
                        help="max_num_batched_tokens（>= 最大 chunk_size）")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=16, help="CPU 模式 chunk_size")
    parser.add_argument("--greedy-chunk-size", type=int, default=16,
                        help="greedy 一致性场景的受限 chunk_size")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=12288,
                        help="8192-token prompt 需在 max_model_len 之内（§9.3）")
    parser.add_argument("--scenarios", default="8k,greedy,prefix",
                        help="逗号分隔的场景子集（8k/greedy/prefix），供补充分析单独重跑")
    args = parser.parse_args()
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
