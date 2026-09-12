"""Day 9 混合 Prefill/Decode 验收脚本（docs/mixed-prefill-decode.md §9.3）。

CPU 模式（真实 Scheduler + 桩 runner，无 GPU/权重）覆盖：
- 混合轮事件契约：字段白名单、round_id 唯一单调、phase 四值、分阶段计数可重算；
- 混合预算公式：prefill_tokens + decode_tokens == planned <= B，decisions 可重算；
- decode-first 证据：存在 decode_priority 让路轮；phase_priority 无残留；
  decode_priority 判定条件（D>0 且 needed<=B）独立复算；
- runner_input_len 对账（Day8 遗留收口）：脚本侧 runner 包装在每轮记录
  decode/prefill 子批展平输入长度，并与调度计划交叉核对；
- 主负载（plan.md Day9）：1 × 8192-token prompt + 32 × 128-token prompt，
  验证长 prompt 逐块推进期间 decode 请求持续产出（decode 保底）。

证据：结构化事件（scheduler_round/engine_round/episode）+ 脚本补充记录
（run_config/runner_input_len/mixed_summary）写入同一 JSONL。
事件不含 prompt/token 明文。

用法：
    PYTHONPATH=. python scripts/validate_mixed_batch.py \
        --output docs/evidence/day9/day9-cpu.jsonl
    PYTHONPATH=. python scripts/validate_mixed_batch.py --mode gpu \
        --model <Qwen3-0.6B 本地快照目录> \
        --output docs/evidence/day9/day9-gpu.jsonl
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

# 事件字段白名单（Day9 版本：scheduler_round/engine_round 增加分阶段字段）
EVENT_FIELDS = {
    "scheduler_round": {"event", "round_id", "phase", "token_budget", "max_num_seqs",
                        "planned_tokens", "prefill_tokens", "decode_tokens",
                        "prefill_items", "decode_items", "scheduled_requests",
                        "needed_first",
                        "budget_deferred_direct", "budget_deferred_hol",
                        "budget_deferred_requests", "observed_at", "decisions"},
    "engine_round": {"event", "round_id", "phase", "token_budget", "planned_tokens",
                     "executed_tokens", "model_called", "outcome", "observed_at",
                     "prefill_chunks", "prefill_items", "decode_items",
                     "prefill_tokens", "decode_tokens"},
    "budget_wait_episode": {"event", "seq_id", "request_id", "started_at", "ended_at",
                            "duration_seconds", "close_reason", "round_id", "observed_at"},
    "request_budget_wait": {"event", "seq_id", "request_id", "budget_deferred_rounds",
                            "budget_wait_seconds", "status", "finish_reason",
                            "round_id", "observed_at"},
}
KNOWN_REASONS = {"scheduled", "budget", "sequence_cap", "kv_capacity",
                 "head_of_line", "decode_priority", "paused"}


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
        """脚本自身的补充记录（run_config/对账/汇总）写入同一 JSONL。"""
        self.events.append(payload)
        self._raw.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._raw.flush()


def make_stub_runner(sampled_token: int = 7, ledgers: list | None = None):
    """契约桩 runner：按 needs_sample 返回采样 token。

    ledgers 非空时做 runner_input_len 观测（Day8 遗留收口）：每轮记录
    decode/prefill 子批的展平输入长度（decode=条数，prefill=sum(q)），
    供与调度计划交叉核对。"""
    def call(method, items_):
        decode_len = sum(1 for it in items_ if it.phase == "decode")
        prefill_len = sum(it.scheduled_tokens for it in items_ if it.phase == "prefill")
        if ledgers is not None:
            ledgers.append({"decode_len": decode_len, "prefill_len": prefill_len})
        out = [sampled_token for it in items_ if it.needs_sample]
        return out if out else None
    return call


def build_cpu_engine(token_budget: int, chunk_size: int, max_num_seqs: int = 64,
                     num_blocks: int = 256, block_size: int = 8,
                     ledgers: list | None = None) -> LLMEngine:
    """CPU 模式：真实 Scheduler + 桩 runner。"""
    Sequence.block_size = block_size
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        chunk_size=chunk_size,
        eos=-1,  # 桩采样恒为 7，不会命中 EOS
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    sched = Scheduler(config)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = sched
    engine.model_runner = SimpleNamespace(call=make_stub_runner(ledgers=ledgers))
    return engine


def validate_events(events: list[dict], runner_ledgers: list[dict]) -> list[str]:
    """独立校验：白名单、混合预算、归因复算、runner_input_len 对账。"""
    problems = []
    sched_rounds = [e for e in events if e.get("event") == "scheduler_round"]
    engine_by_round = {e["round_id"]: e for e in events
                       if e.get("event") == "engine_round"}
    round_ids = [e["round_id"] for e in sched_rounds]
    if len(set(round_ids)) != len(round_ids):
        problems.append("round_id 存在重复")
    if round_ids != sorted(round_ids):
        problems.append("round_id 非单调递增")

    # runner_input_len 对账（Day8 遗留收口）：每次模型调用一条
    # {decode_len, prefill_len}（CPU 桩按 items 派生；GPU spy 按 prepare_*
    # 实际组装的 input_ids 长度捕获），与非空调度轮一一对应后与计划核对
    nonempty = [e for e in sched_rounds if e["planned_tokens"] > 0]
    if len(runner_ledgers) != len(nonempty):
        problems.append(f"runner 调用次数 {len(runner_ledgers)} 与非空调度轮数 "
                        f"{len(nonempty)} 不一致")
    else:
        for e, ledger in zip(nonempty, runner_ledgers):
            if ledger.get("decode_len", 0) != e["decode_tokens"]:
                problems.append(f"round {e['round_id']}: runner decode 输入长度 "
                                f"{ledger.get('decode_len')} != 计划 {e['decode_tokens']}")
            if ledger.get("prefill_len", 0) != e["prefill_tokens"]:
                problems.append(f"round {e['round_id']}: runner prefill 输入长度 "
                                f"{ledger.get('prefill_len')} != 计划 {e['prefill_tokens']}")

    for e in events:
        kind = e.get("event")
        if kind not in EVENT_FIELDS:
            continue
        missing = set(EVENT_FIELDS[kind]) - set(e)
        if missing:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 缺少字段 {sorted(missing)}")
        extra = set(e) - EVENT_FIELDS[kind]
        if extra:
            problems.append(f"round {e.get('round_id')}: 事件 {kind} 含未知字段 {sorted(extra)}")
        if kind == "scheduler_round":
            # 混合预算公式独立重算：prefill + decode == planned <= B
            if e["prefill_tokens"] + e["decode_tokens"] != e["planned_tokens"]:
                problems.append(f"round {e['round_id']}: 分阶段量与 planned 不一致")
            if not 0 <= e["planned_tokens"] <= e["token_budget"]:
                problems.append(f"round {e['round_id']}: planned 超出预算")
            if e["prefill_items"] + e["decode_items"] != e["scheduled_requests"]:
                problems.append(f"round {e['round_id']}: 分阶段条数与批次不一致")
            if e["budget_deferred_requests"] != (e["budget_deferred_direct"]
                                                 + e["budget_deferred_hol"]):
                problems.append(f"round {e['round_id']}: 预算人数口径不可重算")
            if sum(d["scheduled_tokens"] for d in e["decisions"]) != e["planned_tokens"]:
                problems.append(f"round {e['round_id']}: decisions 求和与 planned 不一致")
            # phase 标签与构成一致
            expected_phase = ("mixed" if e["prefill_items"] and e["decode_items"]
                              else "prefill" if e["prefill_items"]
                              else "decode" if e["decode_items"] else "idle")
            if e["phase"] != expected_phase:
                problems.append(f"round {e['round_id']}: phase={e['phase']} 与构成不符")
            # 混合轮内 decode item 在前、prefill item 在后
            phases = [d["phase"] for d in e["decisions"] if d["reason"] == "scheduled"]
            if phases != sorted(phases, key=lambda p: 0 if p == "decode" else 1):
                problems.append(f"round {e['round_id']}: scheduled 决策的 phase 顺序错误")
            for d in e["decisions"]:
                if d["reason"] not in KNOWN_REASONS:
                    problems.append(f"round {e['round_id']}: 未知原因 {d['reason']}")
                # decode_priority 判定条件独立复算（不变量 18：D>0 且
                # needed_first<=B，needed_first 为本轮首个 prefill 候选的需求）
                if d["reason"] == "decode_priority":
                    if not (e["decode_tokens"] > 0 and e["needed_first"] is not None
                            and e["needed_first"] <= e["token_budget"]):
                        problems.append(
                            f"round {e['round_id']}: decode_priority 判定条件不成立 "
                            f"(D={e['decode_tokens']}, needed_first={e['needed_first']})")
        elif kind == "engine_round":
            if e["outcome"] == "completed":
                if not (e["model_called"] and e["executed_tokens"] == e["planned_tokens"]
                        and 0 < e["executed_tokens"] <= e["token_budget"]):
                    problems.append(f"round {e['round_id']}: engine 执行量与计划不一致")
                s = next((r for r in sched_rounds if r["round_id"] == e["round_id"]), None)
                if s is None or s["planned_tokens"] != e["planned_tokens"]:
                    problems.append(f"round {e['round_id']}: scheduler/engine 计划不一致")
                if e["prefill_items"] != e["prefill_chunks"]:
                    problems.append(f"round {e['round_id']}: prefill_items/prefill_chunks 不一致")
            elif e["outcome"] == "idle" and (e["model_called"] or e["executed_tokens"] != 0):
                problems.append(f"round {e['round_id']}: idle 轮不应调用模型")
        elif kind == "budget_wait_episode":
            if abs(e["duration_seconds"] - (e["ended_at"] - e["started_at"])) > 1e-6:
                problems.append(f"episode seq_id={e['seq_id']}: 秒数与起止时间不可重算")
    return problems


def run_main_workload(capture: JsonlCapture) -> tuple[list[str], list[dict]]:
    """主负载：1 × 8192 + 32 × 128，验证 decode 持续产出与预算约束。"""
    problems = []
    token_budget, chunk_size = 2048, 1024
    ledgers: list[dict] = []
    engine = build_cpu_engine(token_budget, chunk_size, num_blocks=1400,
                              ledgers=ledgers)
    capture.write_record({"event": "run_config", "scenario": "main",
                          "mode": "cpu", "token_budget": token_budget,
                          "chunk_size": chunk_size, "max_num_seqs": 64,
                          "num_kvcache_blocks": 1400, "block_size": 8,
                          "workload": "1x8192 + 32x128"})
    long_seq = engine.add_request(list(range(100, 100 + 8192)),
                                  SamplingParams(max_tokens=2, ignore_eos=True,
                                                 temperature=0.0))
    long_seq_id = next(s.seq_id for s in engine.scheduler.requests.values()
                       if s.request_id == long_seq)
    # max_tokens 取 16：保证 32 条短请求贯穿 8K prompt 的整个分块期，
    # 使"decode 持续产出"的证据覆盖每一轮混合调度
    # 32 条互不相同的 128-token 短 prompt（与 GPU 口径一致）：若共用同一
    # prompt，prefix 命中会把后续短请求折叠成 8-token prefill，偏离计划负载
    for i in range(32):
        engine.add_request(list(range(20_000 + 1000 * i, 20_000 + 1000 * i + 128)),
                           SamplingParams(max_tokens=16, ignore_eos=True,
                                          temperature=0.0))

    rounds = 0
    long_prefill_rounds = 0          # 8K 请求仍在 prefill 的轮数
    decode_expected_rounds = 0       # 其中已有 decode 源（轮前有 RUNNING）的轮数
    decode_active_rounds = 0         # 其中 decode 持续产出的轮数
    ttft_round = None                # 8K 请求产出首 completion 的轮次
    decode_during_long_prefill = []  # 各轮 decode 条数（持续产出证据）
    for _ in range(2000):
        if engine.is_finished():
            break
        running_before = {s.seq_id for s in engine.scheduler.running}
        engine.step()
        rounds += 1
        stats = engine.scheduler.last_schedule_stats or {}
        long_in_prefill = any(d["seq_id"] == long_seq_id and d["phase"] == "prefill"
                              for d in stats.get("decisions", []))
        if long_in_prefill:
            long_prefill_rounds += 1
            # decode-first 持续产出：轮前已有 RUNNING（decode 源）时，
            # 本轮 decode 条数必须等于源数量（首个 prefill 轮无源，属正常）
            if running_before:
                decode_expected_rounds += 1
                if stats["decode_items"] == len(running_before):
                    decode_active_rounds += 1
                else:
                    problems.append(
                        f"round {stats['round_id']}: 长 prompt prefill 期间 decode "
                        f"产出 {stats['decode_items']} != decode 源 {len(running_before)}")
            decode_during_long_prefill.append(stats["decode_items"])
        seq = engine.scheduler.requests.get(long_seq_id)
        if ttft_round is None and (seq is None or seq.num_completion_tokens >= 1):
            ttft_round = rounds
    if not engine.is_finished():
        problems.append("主负载未在有界轮次内完成")
    if long_prefill_rounds == 0:
        problems.append("未观察到长 prompt 分块 prefill 轮")
    elif decode_expected_rounds == 0:
        problems.append("长 prompt prefill 期间未观察到任何 decode 源")
    capture.write_record({
        "event": "mixed_summary", "scenario": "main",
        "rounds": rounds, "long_prefill_rounds": long_prefill_rounds,
        "decode_expected_rounds": decode_expected_rounds,
        "decode_active_rounds": decode_active_rounds,
        "decode_items_per_long_prefill_round": decode_during_long_prefill,
        "ttft_round_of_long_request": ttft_round,
        "finished": engine.is_finished(),
    })
    print(f"[main] 轮数={rounds} 长prompt分块轮={long_prefill_rounds} "
          f"其中decode产出轮={decode_active_rounds}/{decode_expected_rounds} "
          f"8K请求首token轮={ttft_round}")
    print(f"[main] 校验结果: {'PASS' if not problems else 'FAIL'}")
    return problems, ledgers




# ============================== GPU 模式（§9.4） ==============================

def gpu_env_record() -> dict:
    """环境记录口径（GPU 型号/显存/torch/驱动），与 Day7/8 验收一致。"""
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 2 ** 30, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }


def check_kv_capacity(engine: LLMEngine, total_tokens: int):
    """GPU 容量自检：主负载必须能容纳于物理 KV 池（不足时报配置错误）。"""
    block_size = engine.scheduler.block_size
    need_blocks = (total_tokens + block_size - 1) // block_size
    free = len(engine.scheduler.block_manager.free_block_ids)
    if need_blocks > free:
        raise RuntimeError(
            f"KV 容量自检失败：{total_tokens} token 需要 {need_blocks} 块，池仅 {free} 块；"
            "请增大 gpu_memory_utilization 或 num_kvcache_blocks")


def attach_runner_spies(engine: LLMEngine, ledgers: list[dict]):
    """runner 包装（§6.3/§9.4）：runner_input_len 独立观测（Day8 遗留收口）。

    在 run() 边界记账（每次模型调用恰好一条），decode/prefill 子批的真实展平
    input_ids 长度由 prepare_decode/prepare_prefill 包装捕获——长度来自实际
    组装的张量，而非调度计划同源字段，属独立交叉核对。warmup 发生在引擎
    构造期内（本包装 attach 之前），不计入账本。
    """
    runner = engine.model_runner
    orig_run = runner.run
    captured = {"decode": None, "prefill": None}
    orig_pp = runner.prepare_prefill
    orig_pd = runner.prepare_decode

    def spy_pp(seqs):
        input_ids, positions = orig_pp(seqs)
        captured["prefill"] = len(input_ids)
        return input_ids, positions

    def spy_pd(seqs):
        input_ids, positions = orig_pd(seqs)
        captured["decode"] = len(input_ids)
        return input_ids, positions

    runner.prepare_prefill = spy_pp
    runner.prepare_decode = spy_pd

    def wrapped_run(items):
        captured["decode"] = None
        captured["prefill"] = None
        out = orig_run(items)
        ledgers.append({"decode_len": captured["decode"] or 0,
                        "prefill_len": captured["prefill"] or 0})
        return out

    runner.run = wrapped_run


def percentiles(values: list[float], ps=(50, 95)):
    """就近秩百分位（观测值汇报，不做分布假设）。"""
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        out[f"p{p}"] = ordered[idx]
    return out


def scenario_main(args, capture: JsonlCapture) -> tuple[bool, list[dict], list[tuple]]:
    """返回 (是否通过, 一致性对比记录, [(variant 事件切片, variant 账本切片), ...])。

    两个 variant 各自独立建引擎，round_id 均从 1 计数，事件校验必须按 variant
    分段进行（全局合并会因 round_id 重叠而误报计划不一致）。
    """
    """§9.4 主负载：1 × 8192 + 32 × 128（互不相同的短 prompt），greedy。

    两组对照：one-shot（chunk_size=B=8192，长 prompt 单轮 prefill，"原始一次性
    Prefill"基线）与 Day9 mixed（chunk_size=1024、B=2048）。返回
    (是否通过, one-shot 输出, mixed 输出) 供一致性对比。
    """
    all_ledgers: list[dict] = []
    segments: list[tuple] = []
    long_tokens = list(range(100, 100 + 8192))
    # 32 条互不相同的 128-token 短 prompt：保持每条 128 token 的真实 prefill
    # 工作量（若共用同一 prompt，prefix 命中会把后续短请求折叠成 8-token
    # prefill，偏离计划负载的意图）。token id 取词表内（Qwen3 vocab 151936）
    # 且与长 prompt 区间 [100, 8292) 不重叠
    short_tokens_list = [list(range(20_000 + 1000 * i, 20_000 + 1000 * i + 128))
                         for i in range(32)]
    results = {}
    ok = True
    for chunk_size, budget, is_one_shot in [(8192, 8192, True), (1024, 2048, False)]:
        tag = "one_shot" if is_one_shot else "mixed"
        engine = LLMEngine(args.model, max_num_batched_tokens=budget,
                           max_num_seqs=64, enforce_eager=True,
                           tensor_parallel_size=1, chunk_size=chunk_size,
                           max_model_len=12288)
        atexit.unregister(engine.exit)
        check_kv_capacity(engine, 8192 + 16 + 32)
        capture.write_record({
            "event": "run_config", "scenario": "main", "variant": tag,
            "chunk_size": chunk_size, "token_budget": budget,
            "block_size": engine.scheduler.block_size,
            "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
            "long_prompt_tokens": 8192, "short_prompt_tokens": 128,
            "short_request_count": 32, "max_new_tokens": 16,
            "temperature": 0.0, "max_num_seqs": 64, **gpu_env_record()})
        segment_start = len(capture.events)
        ledgers: list[dict] = []
        ledgers_start = len(all_ledgers)
        attach_runner_spies(engine, all_ledgers)

        seq_to_index: dict[int, int] = {}
        rid = engine.add_request(long_tokens, SamplingParams(
            max_tokens=16, ignore_eos=True, temperature=0.0))
        long_seq_id = next(s.seq_id for s in engine.scheduler.requests.values()
                           if s.request_id == rid)
        seq_to_index[long_seq_id] = -1
        for i, st in enumerate(short_tokens_list):
            rid = engine.add_request(st, SamplingParams(
                max_tokens=16, ignore_eos=True, temperature=0.0))
            sid = next(s.seq_id for s in engine.scheduler.requests.values()
                       if s.request_id == rid)
            seq_to_index[sid] = i

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        first_token_time: dict[int, float] = {}
        finish_time: dict[int, float] = {}
        decode_rounds_of = {sid: [] for sid in seq_to_index}
        rounds = 0
        long_prefill_rounds = 0
        decode_active_rounds = 0
        decode_items_per_round = []
        outputs: dict[int, list[int]] = {}
        for _ in range(400):
            if engine.is_finished():
                break
            running_before = {s.seq_id for s in engine.scheduler.running}
            step_outputs, _ = engine.step()
            rounds += 1
            # 首 token / 完成时刻记在产出结果的 step() 之后（TTFT/TPOT 的
            # 观测点为 token 实际可用时刻，而不是本轮开始时刻）
            now = time.perf_counter()
            stats = engine.scheduler.last_schedule_stats or {}
            for sid in seq_to_index:
                seq = engine.scheduler.requests.get(sid)
                if sid not in first_token_time and (seq is None
                                                    or seq.num_completion_tokens >= 1):
                    first_token_time[sid] = now
                if sid not in finish_time and (seq is None or seq.is_terminal):
                    finish_time[sid] = now
            long_in_prefill = any(d["seq_id"] == long_seq_id and d["phase"] == "prefill"
                                  for d in stats.get("decisions", []))
            if long_in_prefill:
                long_prefill_rounds += 1
                decode_items_per_round.append(stats["decode_items"])
                if running_before:
                    if stats["decode_items"] == len(running_before):
                        decode_active_rounds += 1
                    else:
                        problems_msg = (f"[{tag}] round {stats['round_id']}: 长 prompt "
                                        f"分块期间 decode 产出 {stats['decode_items']} "
                                        f"!= decode 源 {len(running_before)}")
                        print(problems_msg)
                        capture.write_record({"event": "violation", "variant": tag,
                                              "round_id": stats["round_id"],
                                              "message": problems_msg})
                        ok = False
            for d in stats.get("decisions", []):
                if d["phase"] == "decode" and d["reason"] == "scheduled":
                    decode_rounds_of[d["seq_id"]].append(time.perf_counter())
            for sid, token_ids in step_outputs:
                outputs[seq_to_index[sid]] = token_ids
        wall = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        bm = engine.scheduler.block_manager
        kv_balanced = (len(bm.used_block_ids) == 0
                       and len(bm.free_block_ids) == len(bm.blocks))

        short_ids = [sid for sid, i in seq_to_index.items() if i >= 0]
        short_ttfts = [first_token_time[sid] - t0 for sid in short_ids
                       if sid in first_token_time]
        # TPOT 近似口径：同步 API 下 (完成时刻 - 首 token 时刻)/(completions-1)
        short_tpots = []
        for sid in short_ids:
            if sid in first_token_time and sid in finish_time:
                short_tpots.append((finish_time[sid] - first_token_time[sid]) / 15)
        decode_itls = []
        for sid in short_ids:
            times = decode_rounds_of[sid]
            decode_itls.extend(times[i + 1] - times[i] for i in range(len(times) - 1))
        long_ttft = first_token_time.get(long_seq_id, 0.0) - t0
        finished = engine.is_finished()
        capture.write_record({
            "event": "perf_matrix", "scenario": "main", "variant": tag,
            "chunk_size": chunk_size, "token_budget": budget,
            "wall_time": wall, "rounds": rounds, "finished": finished,
            "long_prefill_rounds": long_prefill_rounds,
            "decode_active_rounds": decode_active_rounds,
            "decode_items_per_long_prefill_round": decode_items_per_round,
            "long_ttft": long_ttft,
            "short_ttft": percentiles(short_ttfts),
            "short_tpot": percentiles(short_tpots),
            "short_decode_itl": percentiles(decode_itls),
            "short_ttft_all": short_ttfts,
            "kv_final_balanced": kv_balanced,
            "runner_input_observations": len(all_ledgers) - ledgers_start,
            "max_memory_allocated": peak_alloc,
            "max_memory_reserved": peak_reserved,
        })
        print(f"[gpu-main:{tag}] rounds={rounds} wall={wall:.2f}s "
              f"long_ttft={long_ttft:.2f}s "
              f"short_ttft_p50={percentiles(short_ttfts).get('p50', 0):.3f}s "
              f"short_ttft_p95={percentiles(short_ttfts).get('p95', 0):.3f}s "
              f"short_tpot_p50={percentiles(short_tpots).get('p50', 0) * 1000:.1f}ms "
              f"decode轮={decode_active_rounds}/{long_prefill_rounds} "
              f"peak_alloc={peak_alloc / 2 ** 20:.0f}MiB")
        if not finished:
            print(f"[gpu-main:{tag}] FAIL: 未在有界轮次内完成")
            ok = False
        if not kv_balanced:
            print(f"[gpu-main:{tag}] FAIL: KV 收尾不平衡")
            ok = False
        results[tag] = outputs    # 键为请求下标（-1 = 8K 长请求，0..31 = 短请求）
        segments.append((capture.events[segment_start:],
                         all_ledgers[ledgers_start:]))
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    # greedy 一致性：mixed vs one-shot 逐请求 token_ids 对比（差异如实记录，
    # 不伪造一致；缺失输出判失败）
    one_shot_out, mixed_out = results.get("one_shot", {}), results.get("mixed", {})
    records = []
    for i in [-1] + list(range(32)):
        a, b = one_shot_out.get(i), mixed_out.get(i)
        records.append({"request_index": i, "match": a == b,
                        "tokens_one_shot": a, "tokens_mixed": b})
    missing = [r["request_index"] for r in records
               if r["tokens_one_shot"] is None or r["tokens_mixed"] is None]
    if missing:
        print(f"[gpu-main] FAIL: 一致性对比缺少输出 request_index={missing}")
        ok = False
    match_count = sum(1 for r in records if r["match"])
    print(f"[gpu-main] greedy 一致性 mixed vs one-shot: {match_count}/{len(records)} "
          "（差异如实记录，不伪造一致）")
    return ok, records, segments


def run_gpu(args, capture: JsonlCapture) -> bool:
    """GPU 模式：§9.4 主负载 mixed vs one-shot 对照 + 事件独立校验 + 一致性对比。"""
    ok, records, segments = scenario_main(args, capture)
    # 事件独立校验按 variant 分段：混合预算/归因复算/runner_input_len 对账
    # （input_len 来自 prepare_* 实际组装长度，与调度计划独立交叉核对）
    for i, (seg_events, seg_ledgers) in enumerate(segments):
        seg_problems = validate_events(seg_events, seg_ledgers)
        if seg_problems:
            print(f"[gpu] variant{i} 事件校验 FAIL:")
            for p in seg_problems:
                print(f"  - {p}")
            ok = False
        else:
            print(f"[gpu] variant{i} 事件校验 PASS")
    capture.write_record({"event": "greedy_compare", "scenario": "main",
                          "results": records})
    return ok


def main():
    parser = argparse.ArgumentParser(description="Day9 mixed prefill/decode 验收脚本")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", default=None, help="GPU 模式的本地模型目录")
    parser.add_argument("--output", default="/tmp/day9-mixed.jsonl")
    args = parser.parse_args()

    capture = JsonlCapture(Path(args.output))
    SCHED_LOGGER.setLevel(logging.INFO)
    ENGINE_LOGGER.setLevel(logging.INFO)
    SCHED_LOGGER.addHandler(capture)
    ENGINE_LOGGER.addHandler(capture)
    SCHED_LOGGER.propagate = False
    ENGINE_LOGGER.propagate = False
    try:
        if args.mode == "gpu":
            if not args.model:
                parser.error("gpu 模式必须提供 --model（本地模型目录，不自动下载）")
            ok = run_gpu(args, capture)
        else:
            problems, ledgers = run_main_workload(capture)
            # 全事件独立校验（含 runner_input_len 对账）
            problems.extend(validate_events(capture.events, ledgers))
            if problems:
                print("[events] 校验 FAIL:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print("[events] 校验 PASS")
            ok = not problems
    finally:
        capture.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
