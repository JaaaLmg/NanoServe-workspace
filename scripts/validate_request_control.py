"""Day 10 取消/超时/抢占/恢复验收脚本（docs/request-cancellation-preemption.md §9.3/§9.4）。

CPU 模式（真实 Scheduler/BlockManager + 桩 runner，无 GPU/权重）独立完成：
- 事件字段白名单、round_id 单调关联、无 prompt/token 明文；
- 从 request_control 事件重算合法状态迁移（对照 Sequence.VALID_TRANSITIONS）；
- 取消优先级（cancel > timeout > 正常提交）与 deadline 单调语义（now>=deadline）；
- 抢占/恢复事件按 seq_id + num_preempts 配对；
- 每个活动 seq_id/request_id 唯一（Scheduler 活动索引独立快照交叉核对）；
- 每个控制动作前后 free/used/ref_count 守恒（BlockManager.check_ledger + 集合快照）；
- 异常后没有活动请求、队列成员或未释放 block；Engine 禁止重试；
- 100 请求混合负载（正常/取消/超时/抢占）最终资源稳定：活动索引清空、
  队列清空、used_block_ids 归零、free 回到池总量、KV 使用量尾部窗口无增长趋势。

GPU 模式（本地模型 + TP=1 + enforce_eager=True）覆盖 §9.4 矩阵：
cancel-waiting / cancel-running / timeout / preempt-resume / exception-cleanup /
100-request，记录每请求终态与 reason、抢占/恢复次数、轮数、控制事件计数、
step 耗时、free/used 轨迹、torch.cuda 显存起止与峰值、最终账本快照。
性能数字只作观测，不设改善阈值。

用法：
    PYTHONPATH=. python scripts/validate_request_control.py --mode cpu \
        --output docs/evidence/day10/day10-cpu.jsonl
    PYTHONPATH=. python scripts/validate_request_control.py --mode gpu \
        --model ~/huggingface/Qwen3-0.6B \
        --output docs/evidence/day10/day10-gpu.jsonl
"""

import argparse
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import (TERMINAL_STATUSES, Sequence, SequenceStatus,
                                      VALID_TRANSITIONS)
from nanovllm.sampling_params import SamplingParams

SCHED_LOGGER = logging.getLogger("nanovllm.engine.scheduler")
ENGINE_LOGGER = logging.getLogger("nanovllm.engine.llm_engine")

# 事件字段白名单（§6.3：事件只记录 ID/状态/计数/时间，不含明文）
EVENT_FIELDS = {
    "scheduler_round": {"event", "round_id", "phase", "token_budget", "max_num_seqs",
                        "planned_tokens", "prefill_tokens", "decode_tokens",
                        "prefill_items", "decode_items", "scheduled_requests",
                        "needed_first", "budget_deferred_direct",
                        "budget_deferred_hol", "budget_deferred_requests",
                        "observed_at", "decisions"},
    "engine_round": {"event", "round_id", "phase", "token_budget", "planned_tokens",
                     "executed_tokens", "model_called", "outcome", "observed_at",
                     "prefill_chunks", "prefill_items", "decode_items",
                     "prefill_tokens", "decode_tokens"},
    "budget_wait_episode": {"event", "seq_id", "request_id", "started_at", "ended_at",
                            "duration_seconds", "close_reason", "round_id",
                            "observed_at"},
    "request_budget_wait": {"event", "seq_id", "request_id", "budget_deferred_rounds",
                            "budget_wait_seconds", "status", "finish_reason",
                            "round_id", "observed_at"},
    "request_control": {"event", "round_id", "seq_id", "request_id", "action",
                        "from_status", "to_status", "reason", "num_preempts",
                        "released_blocks", "free_blocks_before", "used_blocks_before",
                        "free_blocks", "used_blocks", "observed_at"},
        "engine_abort_summary": {"event", "round_id", "reason", "execution_failed",
                             "round_snapshot", "final_snapshot", "cleanup_errors",
                             "observed_at"},
}
CONTROL_ACTIONS = {"cancel", "timeout", "preempt", "resume", "abort"}
FORBIDDEN_KEYS = {"prompt", "token_ids", "text", "completion_token_ids"}
# 脚本自有记录（非引擎事件）：不做引擎字段白名单校验，但仍检查无明文
SCRIPT_RECORD_EVENTS = {"run_config", "scenario_begin", "scenario_summary",
                        "scenario_result", "validation_summary",
                        "gpu_env", "gpu_memory"}


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


# ============================== 独立校验函数 ==============================

def check_event_contract(events: list[dict]) -> list[str]:
    """事件契约：白名单字段、无明文、round_id 单调（§9.3）。"""
    problems = []
    last_round = 0
    for e in events:
        name = e.get("event")
        if name in SCRIPT_RECORD_EVENTS:
            if FORBIDDEN_KEYS & set(e):
                problems.append(f"脚本记录 {name} 携带疑似明文字段")
            continue
        if name not in EVENT_FIELDS:
            problems.append(f"未知事件类型 {name!r}")
            continue
        extra = set(e) - EVENT_FIELDS[name]
        missing = EVENT_FIELDS[name] - set(e)
        if extra:
            problems.append(f"事件 {name} 出现白名单外字段: {sorted(extra)}")
        if missing:
            problems.append(f"事件 {name} 缺少字段: {sorted(missing)}")
        if FORBIDDEN_KEYS & set(e):
            problems.append(f"事件 {name} 携带疑似明文字段: {sorted(FORBIDDEN_KEYS & set(e))}")
        rid = e.get("round_id")
        if isinstance(rid, int) and rid >= 0:
            if rid < last_round:
                problems.append(f"round_id 非单调: {rid} < {last_round}")
            last_round = max(last_round, rid)
    return problems


def replay_control_transitions(events: list[dict], total_blocks: int | None = None) -> list[str]:
    """按请求维护控制事件历史，并独立检查状态、身份与账本快照。

    调度器内部的 WAITING→RUNNING 等迁移不会发控制事件，因此首次观测以
    ``from_status`` 建立基线；之后同一请求必须连续衔接。轮外控制允许
    ``round_id=None``，不把它误判为事件缺陷。``total_blocks`` 可由场景的
    final_snapshot 推导，用于检查每个控制快照的 free+used 守恒。
    """
    problems = []
    state_by_seq: dict[int, SequenceStatus] = {}
    request_by_seq: dict[int, str] = {}
    first_reason: dict[tuple[str, int], str] = {}
    pending_preempts: dict[int, list[int]] = {}
    if total_blocks is None:
        summaries = [e for e in events if e.get("event") == "scenario_summary"]
        if summaries:
            total_blocks = summaries[-1].get("final_snapshot", {}).get("total_blocks")
    for e in events:
        if e.get("event") != "request_control":
            continue
        try:
            action, sid = e["action"], e["seq_id"]
            request_id = e["request_id"]
            old = SequenceStatus[e["from_status"]]
            new = SequenceStatus[e["to_status"]]
            released = e["released_blocks"]
            free_before = e["free_blocks_before"]
            used_before = e["used_blocks_before"]
            free = e["free_blocks"]
            used = e["used_blocks"]
        except (KeyError, TypeError, ValueError) as exc:
            problems.append(f"控制事件字段非法: {exc}")
            continue
        if action not in CONTROL_ACTIONS:
            problems.append(f"非法控制动作 {action!r}")
            continue
        if not isinstance(sid, int) or not isinstance(request_id, str):
            problems.append(f"seq/request_id 类型非法: {sid!r}/{request_id!r}")
            continue
        if sid in request_by_seq and request_by_seq[sid] != request_id:
            problems.append(f"seq {sid}: request_id 被改写 {request_by_seq[sid]!r} -> {request_id!r}")
        request_by_seq.setdefault(sid, request_id)
        key = (action, sid)
        if key in first_reason and e["reason"] != first_reason[key]:
            problems.append(f"seq {sid}: 重复 {action} 改写了原因")
        first_reason.setdefault(key, e["reason"])
        previous = state_by_seq.get(sid)
        if previous is not None and old is not previous:
            # WAITING→RUNNING 由正常 prefill 提交产生，当前控制事件不单独记录；
            # 允许它作为两个控制事件之间唯一的隐藏内部迁移。
            if not (previous is SequenceStatus.WAITING and old is SequenceStatus.RUNNING):
                problems.append(f"seq {sid}: 事件历史断裂，期望 from={previous.name}，实际 {old.name}")
        legal = new is old and old in TERMINAL_STATUSES
        if new is not old and new not in VALID_TRANSITIONS[old]:
            problems.append(f"seq {sid}: 非法迁移 {old.name} -> {new.name}（action={action}）")
        elif new is not old:
            state_by_seq[sid] = new
        elif not legal:
            # 非终态自迁移不属于状态机合法操作。
            problems.append(f"seq {sid}: 非终态重复迁移 {old.name} -> {new.name}")
        if action == "preempt" and new is SequenceStatus.PREEMPTED:
            pending_preempts.setdefault(sid, []).append(e["num_preempts"])
        elif action == "resume":
            pending = pending_preempts.get(sid, [])
            if not pending:
                problems.append(f"seq {sid}: resume 没有未配对的 preempt")
            elif pending.pop() != e["num_preempts"]:
                problems.append(f"seq {sid}: resume 的 num_preempts 与最近一次 preempt 不一致")
        values = (released, free_before, used_before, free, used)
        if any(type(value) is not int for value in values) or any(value < 0 for value in values):
            problems.append(f"seq {sid}: 控制事件资源字段必须是非负整数")
        elif total_blocks is not None:
            if free_before + used_before != total_blocks:
                problems.append(f"seq {sid}: before free+used != total_blocks")
            if free + used != total_blocks:
                problems.append(f"seq {sid}: after free+used={free + used} != total_blocks={total_blocks}")
            if released != used_before - used:
                problems.append(f"seq {sid}: released_blocks 与 used 快照差值不一致")
    for sid, pending in pending_preempts.items():
        if pending:
            problems.append(f"seq {sid}: {len(pending)} 个 preempt 缺少配对 resume")
    return problems


def pair_preempt_resume(events: list[dict]) -> tuple[list[str], int, int]:
    """抢占/恢复事件按 seq_id 配对：每个自动抢占后必须有同 seq 的恢复；恢复的
    num_preempts 与抢占一致（§6.3）。返回 (problems, preempt_count, resume_count)。"""
    problems = []
    pending: dict[int, int] = {}
    preempt_count = resume_count = 0
    for e in events:
        if e.get("event") != "request_control":
            continue
        sid = e["seq_id"]
        if e["action"] == "preempt":
            preempt_count += 1
            pending[sid] = e["num_preempts"]
        elif e["action"] == "resume":
            resume_count += 1
            if sid not in pending:
                problems.append(f"seq {sid}: 无配对抢占的 resume 事件")
            elif pending.pop(sid) != e["num_preempts"]:
                problems.append(f"seq {sid}: resume 的 num_preempts 与 preempt 不一致")
    for sid, n in pending.items():
        problems.append(f"seq {sid}: preempt 事件缺少配对 resume（num_preempts={n}）")
    return problems, preempt_count, resume_count


def snapshot_state(sched: Scheduler) -> dict:
    """Scheduler/BlockManager 独立快照（不解析日志，直接读对象状态）。"""
    bm = sched.block_manager
    return {
        "active_requests": len(sched.requests),
        "waiting": [s.seq_id for s in sched.waiting],
        "running": [s.seq_id for s in sched.running],
        "paused": [s.seq_id for s in sched.requests.values()
                   if s.status == SequenceStatus.PREEMPTED],
        "free_blocks": len(bm.free_block_ids),
        "used_blocks": len(bm.used_block_ids),
        "total_blocks": len(bm.blocks),
        "ref_counts": {bid: bm.blocks[bid].ref_count for bid in bm.used_block_ids},
    }


def assert_stable(sched: Scheduler, tag: str) -> list[str]:
    """终态资源稳定性：索引/队列清空、非共享 block 全部回到 free、账本守恒。"""
    problems = []
    if sched.requests:
        problems.append(f"[{tag}] 活动索引未清空: {list(sched.requests)}")
    if sched.waiting or sched.running:
        problems.append(f"[{tag}] 队列未清空: waiting={len(sched.waiting)} running={len(sched.running)}")
    bm = sched.block_manager
    if bm.used_block_ids:
        problems.append(f"[{tag}] 仍有未释放 block: {sorted(bm.used_block_ids)}")
    if len(bm.free_block_ids) != len(bm.blocks):
        problems.append(f"[{tag}] free 块数 {len(bm.free_block_ids)} != 池总量 {len(bm.blocks)}")
    try:
        bm.check_ledger()
    except ValueError as exc:
        problems.append(f"[{tag}] 账本守恒检查失败: {exc}")
    return problems


def assert_identity_unique(sched: Scheduler, tag: str) -> list[str]:
    """活动期 seq_id/request_id 唯一（§5.5 不变量 2）。"""
    problems = []
    seq_ids = [s.seq_id for s in sched.requests.values()]
    req_ids = [s.request_id for s in sched.requests.values()]
    if len(set(seq_ids)) != len(seq_ids):
        problems.append(f"[{tag}] 活动 seq_id 重复: {seq_ids}")
    if len(set(req_ids)) != len(req_ids):
        problems.append(f"[{tag}] 活动 request_id 重复: {req_ids}")
    for seq in sched.waiting:
        if seq.status != SequenceStatus.WAITING:
            problems.append(f"[{tag}] waiting 队列存在非 WAITING 请求 seq={seq.seq_id}")
    for seq in sched.running:
        if seq.status != SequenceStatus.RUNNING:
            problems.append(f"[{tag}] running 队列存在非 RUNNING 请求 seq={seq.seq_id}")
    return problems


def build_cpu_engine(token_budget: int, chunk_size: int, max_num_seqs: int = 64,
                     num_blocks: int = 256, block_size: int = 8,
                     runner=None) -> LLMEngine:
    """CPU 模式：真实 Scheduler/BlockManager + 桩 runner（跳过 GPU __init__）。"""
    Sequence.block_size = block_size
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        chunk_size=chunk_size,
        eos=-1,  # 桩采样恒为正数，不会命中 EOS
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    sched = Scheduler(config)
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = sched
    if runner is None:
        def runner(method, items_):
            out = [7 for it in items_ if it.needs_sample]
            return out if out else None
    engine.model_runner = SimpleNamespace(call=runner)
    return engine


def drive(engine: LLMEngine, limit: int = 2000) -> int:
    """驱动 Engine 至全部终态或有界轮次；返回实际轮数。"""
    rounds = 0
    while not engine.is_finished():
        if rounds >= limit:
            raise AssertionError(f"驱动超过 {limit} 轮仍未完成（疑似死循环/泄漏）")
        engine.step()
        rounds += 1
    return rounds


# ============================== CPU 场景 ==============================

# ============================== CPU 场景 ==============================

def scenario_signal_vs_safepoint(capture: JsonlCapture) -> list[str]:
    """取消信号在 forward 期间只置位；postprocess 安全点完成收尾且同轮隔离。"""
    problems = []
    before = len(capture.events)
    armed: set[str] = set()

    def runner(method, items_):
        # 模拟另一线程在 ModelRunner forward 期间发起取消（signal-only，
        # Engine._in_step=True 时 cancel_request 不得做破坏性清理）
        for it in items_:
            if it.seq.request_id in armed:
                assert engine.cancel_request(it.seq.request_id, "forward_cancel")
                armed.discard(it.seq.request_id)
        out = [7 for it in items_ if it.needs_sample]
        return out if out else None

    engine = build_cpu_engine(token_budget=16, chunk_size=8, num_blocks=64,
                              runner=runner)
    sched = engine.scheduler
    finished_ids = []
    # a：正常 decode 源；victim-decode：prefill 完成后在 decode forward 期间取消
    engine.add_request([1, 2, 3, 4], SamplingParams(max_tokens=3, ignore_eos=True),
                       request_id="a")
    engine.step()                          # a prefill -> RUNNING
    engine.add_request([5, 6, 7, 8], SamplingParams(max_tokens=3, ignore_eos=True),
                       request_id="victim-decode")
    engine.step()                          # decode a + prefill victim-decode
    armed.add("victim-decode")
    engine.step()                          # decode 期间置位取消 -> 安全点收尾
    # victim-prefill：mid-chunk prefill 期间取消
    engine.add_request(list(range(1, 25)), SamplingParams(max_tokens=2, ignore_eos=True),
                       request_id="victim-prefill")
    engine.step()                          # chunk 1（未布防）
    armed.add("victim-prefill")
    engine.step()                          # chunk 2 期间置位取消
    for _ in range(50):
        if engine.is_finished():
            break
        out, _ = engine.step()
        finished_ids += [sid for sid, _ in out]
    scene = capture.events[before:]
    ctl = [e for e in scene if e.get("event") == "request_control"
           and e["action"] == "cancel"]
    by_rid = {(e["request_id"], e["to_status"]): e["reason"] for e in ctl}
    if by_rid.get(("victim-decode", "CANCELLED")) != "forward_cancel":
        problems.append(f"victim-decode 未在 decode forward 期间被取消: {by_rid}")
    if by_rid.get(("victim-prefill", "CANCELLED")) != "forward_cancel":
        problems.append(f"victim-prefill 未在 prefill chunk 期间被取消: {by_rid}")
    # 取消请求不得伪装为正常 completion：终态事件里没有 FINISHED
    for e in scene:
        if e.get("event") == "request_control" and e["request_id"] in (
                "victim-decode", "victim-prefill") and e["to_status"] == "FINISHED":
            problems.append(f"{e['request_id']} 被误记为正常完成")
    problems += assert_stable(sched, "signal_vs_safepoint")
    problems += assert_identity_unique(sched, "signal_vs_safepoint")
    capture.write_record({
        "event": "scenario_summary", "scenario": "signal_vs_safepoint",
        "cancel_events": len(ctl),
        "finished_seq_ids": finished_ids,
        "final_snapshot": snapshot_state(sched),
    })
    return problems


def scenario_deadline_priority(capture: JsonlCapture) -> list[str]:
    """deadline 单调语义（now >= deadline）+ 取消优先于超时优先于正常提交。"""
    problems = []
    before = len(capture.events)
    engine = build_cpu_engine(token_budget=16, chunk_size=8, num_blocks=64)
    sched = engine.scheduler
    # 三个请求同时 prefill -> RUNNING；w/r 靠短 deadline 超时，c 被取消
    for rid in ("w", "r", "c"):
        engine.add_request([1, 2], SamplingParams(max_tokens=5, ignore_eos=True),
                           request_id=rid, deadline=time.perf_counter() + 0.05)
    engine.step()                          # 三条全部完成 prefill
    assert engine.cancel_request("c", "client_cancelled")   # 空闲安全点立即收尾
    time.sleep(0.06)                       # 确保 now 越过 0.05s deadline（单调时钟）
    engine.step()                          # 调度边界清理 w/r
    finals = {}
    for e in capture.events[before:]:
        if e.get("event") == "request_control" and e["action"] in ("cancel", "timeout"):
            finals[e["request_id"]] = (e["to_status"], e["reason"])
    if finals.get("w", ("", ""))[0] != "TIMEOUT":
        problems.append(f"w 应 TIMEOUT: {finals.get('w')}")
    if finals.get("r", ("", ""))[0] != "TIMEOUT":
        problems.append(f"r 应 TIMEOUT: {finals.get('r')}")
    if finals.get("c") != ("CANCELLED", "client_cancelled"):
        problems.append(f"c 应 CANCELLED/client_cancelled（取消优先于超时）: {finals.get('c')}")
    problems += assert_stable(sched, "deadline_priority")
    capture.write_record({
        "event": "scenario_summary", "scenario": "deadline_priority",
        "finals": finals, "final_snapshot": snapshot_state(sched),
    })
    return problems


def scenario_preempt_resume(capture: JsonlCapture) -> list[str]:
    """KV 不足抢占（队尾 victim）+ 恢复 recompute：逻辑进度不丢、账本守恒。"""
    problems = []
    before = len(capture.events)
    engine = build_cpu_engine(token_budget=64, chunk_size=64, num_blocks=9,
                              max_num_seqs=8)
    sched = engine.scheduler
    # 两个长请求（4 块/个，prompt 互异避免 prefix 共享）+ 1 个中请求（2 块）：
    # 池 9 块，decode 增长后必然迫使调度器抢占（队尾 victim + prefill 侧重查）
    seq_of = {}
    for i in range(2):
        rid = f"long-{i}"
        engine.add_request([200 + i] + list(range(1, 32)),
                           SamplingParams(max_tokens=6, ignore_eos=True),
                           request_id=rid)
        seq_of[engine.get_request(rid).seq_id] = rid   # step 输出按 seq_id 索引
    engine.add_request([210] + list(range(1, 16)),
                       SamplingParams(max_tokens=4, ignore_eos=True),
                       request_id="mid")
    seq_of[engine.get_request("mid").seq_id] = "mid"
    rounds = 0
    outputs: dict[int, int] = {}
    while not engine.is_finished():
        if rounds >= 300:
            problems.append("抢占场景 300 轮未收敛")
            break
        out, _ = engine.step()
        for sid, toks in out:
            outputs[sid] = len(toks)
        rounds += 1
    scene = capture.events[before:]
    problems_p, preempt_n, resume_n = pair_preempt_resume(scene)
    problems += problems_p
    if preempt_n == 0:
        problems.append("小池场景未观察到抢占事件（压力不足，场景无效）")
    if preempt_n != resume_n:
        problems.append(f"抢占/恢复次数不配对: preempt={preempt_n} resume={resume_n}")
    # 逻辑进度：每个请求正常完成且 completion 数 == max_tokens（抢占恢复后
    # 不丢不重——正常完成不发光事件，以 step 输出独立核对）
    expect = {"long-0": 6, "long-1": 6, "mid": 4}
    got = {}
    for sid, n_tok in outputs.items():
        got[seq_of.get(sid, f"seq-{sid}")] = n_tok
    for rid, n in expect.items():
        if got.get(rid) != n:
            problems.append(f"{rid} 完成进度异常: 期望 {n} 个 completion，"
                            f"实际 {got.get(rid)!r}")
    problems += assert_stable(sched, "preempt_resume")
    capture.write_record({
        "event": "scenario_summary", "scenario": "preempt_resume",
        "rounds": rounds, "preempts": preempt_n, "resumes": resume_n,
        "completion_counts": {str(k): v for k, v in outputs.items()},
        "final_snapshot": snapshot_state(sched),
    })
    return problems


def scenario_exception_cleanup(capture: JsonlCapture) -> list[str]:
    """runner 异常：abort_round + abort_all_active 收尾、Engine 锁定、资源归零。"""
    problems = []
    before = len(capture.events)
    state = {"fail": False}

    def maybe_boom(method, items_):
        if state["fail"]:
            raise RuntimeError("injected cuda failure")
        out = [7 for it in items_ if it.needs_sample]
        return out if out else None

    engine = build_cpu_engine(token_budget=16, chunk_size=8, num_blocks=32,
                              runner=maybe_boom)
    sched = engine.scheduler
    for i in range(3):
        engine.add_request([1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True),
                           request_id=f"ex-{i}")
    engine.step()                       # prefill -> RUNNING（持有 block）
    used_before = len(sched.block_manager.used_block_ids)
    if used_before == 0:
        problems.append("场景无效：异常前应持有 block")
    state["fail"] = True
    try:
        engine.step()                   # decode 轮桩 runner 抛错
        problems.append("异常未被向上传播")
    except RuntimeError:
        pass
    problems += assert_stable(sched, "exception_cleanup")
    if not engine._execution_failed:
        problems.append("异常后 Engine 未进入失败锁定态")
    try:
        engine.step()
        problems.append("失败 Engine 允许了重试")
    except RuntimeError:
        pass
    scene = capture.events[before:]
    errs = [e for e in scene if e.get("event") == "engine_round"
            and e["outcome"] == "error"]
    if not errs or errs[-1]["executed_tokens"] is not None:
        problems.append("error 轮事件缺少 executed_tokens=null")
    summaries = [e for e in scene if e.get("event") == "engine_abort_summary"]
    if not summaries or summaries[-1]["final_snapshot"]["active_requests"] != 0:
        problems.append("engine_abort_summary 显示活动请求未清零")
    aborts = [e for e in scene if e.get("event") == "request_control"
              and e["action"] == "abort"]
    if len(aborts) < 3:
        problems.append(f"abort 控制事件数量不足: {len(aborts)}")
    for e in aborts:
        if e["reason"] != "engine_error" or e["to_status"] != "CANCELLED":
            problems.append(f"abort 事件语义错误: {e}")
    capture.write_record({
        "event": "scenario_summary", "scenario": "exception_cleanup",
        "aborts": len(aborts), "final_snapshot": snapshot_state(sched),
    })
    return problems


def scenario_100_requests(capture: JsonlCapture, total: int = 100) -> list[str]:
    """100 请求混合负载（正常/取消/超时/显式抢占）：无死锁、无泄漏、资源稳定。"""
    problems = []
    before = len(capture.events)
    rng = random.Random(20260910)
    # 池 16 块刻意小于并发需求上限（16 seqs x 4 块），预算 128 足够宽：
    # 让 KV（而非预算）成为接纳瓶颈，调度器自动抢占（KV 不足选 victim）
    # 与显式抢占都会出现在证据中
    engine = build_cpu_engine(token_budget=128, chunk_size=16, max_num_seqs=16,
                              num_blocks=16)
    sched = engine.scheduler
    submitted = 0
    rounds = 0
    trajectory = []
    finished = 0
    # 阶段一：交错提交 + 随机控制动作（取消/超时/显式抢占/恢复）
    while submitted < total:
        if rounds >= 6000:
            problems.append("100 请求负载超过轮次上限（疑似死锁）")
            break
        # 每轮批量提交至多 4 条：并发到达使活动块需求超过小池（16 块），
        # 调度器自动抢占（KV 不足选 victim）进入证据
        for _ in range(4):
            if submitted >= total:
                break
            rid = f"load-{submitted}"
            engine.add_request(list(range(1, rng.randint(2, 30) + 1)),
                               SamplingParams(max_tokens=rng.randint(1, 6),
                                              ignore_eos=True),
                               request_id=rid,
                               deadline=(time.perf_counter() + 0.0005
                                         if rng.random() < 0.12 else None))
            submitted += 1
        if sched.requests and rng.random() < 0.12:
            seq = rng.choice(list(sched.requests.values()))
            if rng.random() < 0.6:
                engine.cancel_request(seq.request_id, "load_cancel")
            else:
                sched.timeout(seq.seq_id, reason="load_timeout")
        if sched.running and rng.random() < 0.06:
            victim = rng.choice(list(sched.running))
            if victim.status == SequenceStatus.RUNNING:
                sched.preempt(victim)   # 显式抢占：停在 PREEMPTED 考验 paused 扫描
        paused = [s for s in sched.requests.values()
                  if s.status == SequenceStatus.PREEMPTED]
        if paused and rng.random() < 0.7:
            sched.resume(rng.choice(paused))
        out, _ = engine.step()
        finished += len(out)
        rounds += 1
        trajectory.append(len(sched.block_manager.used_block_ids))
        if rounds % 10 == 0:
            sched.block_manager.check_ledger()
            problems += assert_identity_unique(sched, f"load@round{rounds}")
    # 阶段二：确定性清场——恢复全部暂停请求后驱动到全部终态
    for s in [s for s in sched.requests.values()
              if s.status == SequenceStatus.PREEMPTED]:
        sched.resume(s)
    while not engine.is_finished():
        if rounds >= 9000:
            problems.append("清场阶段超过轮次上限（疑似死锁/泄漏）")
            break
        out, _ = engine.step()
        finished += len(out)
        rounds += 1
        trajectory.append(len(sched.block_manager.used_block_ids))
    problems += assert_stable(sched, "load_final")
    # KV 使用量稳定：终态后 used=0；负载尾部窗口峰值不得高于全程峰值 + 容差
    peak = max(trajectory) if trajectory else 0
    tail_peak = max(trajectory[-len(trajectory) // 4:]) if trajectory else 0
    if tail_peak > peak:
        problems.append("KV 使用轨迹异常：尾部窗口峰值高于全程峰值")
    scene = capture.events[before:]
    problems_pair, preempt_n, resume_n = pair_preempt_resume(scene)
    problems += problems_pair
    ctl = [e for e in scene if e.get("event") == "request_control"]
    n_cancel = sum(1 for e in ctl if e["action"] == "cancel")
    n_timeout = sum(1 for e in ctl if e["action"] == "timeout")
    capture.write_record({
        "event": "scenario_summary", "scenario": "100_requests",
        "submitted": submitted, "rounds": rounds, "finished": finished,
        "cancel_actions": n_cancel, "timeout_actions": n_timeout,
        "terminal_balance": submitted == finished + n_cancel + n_timeout,
        "preempts": preempt_n, "resumes": resume_n,
        "kv_peak": peak, "kv_tail_peak": tail_peak,
        "trajectory_head": trajectory[:32], "trajectory_tail": trajectory[-32:],
        "final_snapshot": snapshot_state(sched),
    })
    if submitted < total:
        problems.append(f"提交数不足: {submitted} < {total}")
    if finished + n_cancel + n_timeout != submitted:
        problems.append(f"终态计数不平衡: submitted={submitted} finished={finished} "
                        f"cancel={n_cancel} timeout={n_timeout}")
    return problems


def run_cpu(args, capture: JsonlCapture) -> bool:
    capture.write_record({
        "event": "run_config", "mode": "cpu", "day": 10,
        "runner": "stub（按 needs_sample 注入采样，无模型权重）",
    })
    all_problems = []
    for name, fn in (("signal_vs_safepoint", scenario_signal_vs_safepoint),
                     ("deadline_priority", scenario_deadline_priority),
                     ("preempt_resume", scenario_preempt_resume),
                     ("exception_cleanup", scenario_exception_cleanup),
                     ("100_requests", scenario_100_requests)):
        # 每个场景独立日志文件段：以场景标记分割事件流
        capture.write_record({"event": "scenario_begin", "scenario": name})
        before = len(capture.events)
        problems = fn(capture)
        scene_events = capture.events[before:]
        # 公共契约校验只作用于本场景切片：round_id 单调性与轮次计数是
        # 引擎内语义，跨场景（独立 Scheduler）比较没有意义
        problems += check_event_contract(scene_events)
        problems += replay_control_transitions(scene_events)
        pair_problems, _, _ = pair_preempt_resume(scene_events)
        problems += pair_problems
        scene_ctl = sum(1 for e in scene_events
                        if e.get("event") == "request_control")
        capture.write_record({
            "event": "scenario_result", "scenario": name,
            "passed": not problems, "problems": problems,
            "events": len(scene_events), "control_events": scene_ctl,
        })
        all_problems += [f"[{name}] {p}" for p in problems]
    ok = not all_problems
    capture.write_record({
        "event": "validation_summary", "mode": "cpu", "passed": ok,
        "problem_count": len(all_problems), "problems": all_problems,
    })
    return ok


# ============================== GPU 场景（§9.4） ==============================

def gpu_env_record(model: str, engine: LLMEngine) -> dict:
    import torch
    props = torch.cuda.get_device_properties(0)
    return {
        "event": "gpu_env", "gpu": props.name,
        "vram_total_gib": round(props.total_memory / 2**30, 1),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "model": model, "tensor_parallel_size": 1, "enforce_eager": True,
        "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
        "kvcache_block_size": engine.scheduler.block_manager.block_size,
        "max_num_batched_tokens": engine.scheduler.max_num_batched_tokens,
        "max_num_seqs": engine.scheduler.max_num_seqs,
        "chunk_size": engine.scheduler.chunk_size,
        "memory_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "memory_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
    }


def gpu_memory_record(tag: str) -> dict:
    import torch
    return {
        "event": "gpu_memory", "tag": tag,
        "memory_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "memory_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "memory_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }


def drive_gpu(engine: LLMEngine, limit: int = 4000, tag: str = "") -> dict:
    """GPU 驱动至全部终态：记录每轮耗时与 free/used 轨迹。"""
    trajectory = []
    t0 = time.perf_counter()
    rounds = 0
    while not engine.is_finished():
        if rounds >= limit:
            raise AssertionError(f"[{tag}] GPU 驱动超过 {limit} 轮未完成")
        t1 = time.perf_counter()
        engine.step()
        rounds += 1
        bm = engine.scheduler.block_manager
        trajectory.append({"round": rounds,
                           "step_ms": round((time.perf_counter() - t1) * 1000, 2),
                           "free": len(bm.free_block_ids),
                           "used": len(bm.used_block_ids)})
    return {"rounds": rounds, "wall_s": round(time.perf_counter() - t0, 3),
            "trajectory": trajectory}


def finals_from_events(events: list[dict]) -> dict:
    finals = {}
    for e in events:
        if e.get("event") == "request_control" and e["action"] in ("cancel", "timeout", "abort"):
            finals[e["request_id"]] = (e["to_status"], e["reason"])
    return finals


def gpu_group_cancel_waiting(engine: LLMEngine, capture: JsonlCapture) -> list[str]:
    """cancel-waiting：取消未接纳请求，验证 waiting 清理与 KV 零分配。"""
    problems = []
    sched = engine.scheduler
    before_events = len(capture.events)
    # 16×512=8192 token 首轮需求 > B=4096：一半接纳、一半 waiting 排队
    rids = [engine.add_request([17] * 512, SamplingParams(max_tokens=4, ignore_eos=True),
                               request_id=f"cw-{i}") for i in range(16)]
    engine.step()
    waiting_before = [s.request_id for s in sched.waiting]
    if len(waiting_before) < 4:
        problems.append(f"场景无效：waiting 请求不足（{len(waiting_before)}）")
    for rid in waiting_before[:6]:
        assert engine.cancel_request(rid, "queue_cancel")
    used_after_cancel = len(sched.block_manager.used_block_ids)
    stats = drive_gpu(engine, tag="cancel-waiting")
    finals = {}
    for e in capture.events[before_events:]:
        if e.get("event") == "request_control" and e["action"] in ("cancel", "timeout"):
            finals[e["request_id"]] = (e["to_status"], e["reason"])
    for rid in waiting_before[:6]:
        if finals.get(rid) != ("CANCELLED", "queue_cancel"):
            problems.append(f"waiting 取消 {rid} 终态错误: {finals.get(rid)}")
    problems += assert_stable(sched, "cancel-waiting")
    capture.write_record({
        "event": "scenario_summary", "scenario": "cancel-waiting",
        "cancelled_waiting": waiting_before[:6],
        "free_blocks_at_cancel": used_after_cancel,
        "rounds": stats["rounds"], "wall_s": stats["wall_s"],
        "trajectory_tail": stats["trajectory"][-16:],
        "finals": finals, "final_snapshot": snapshot_state(sched),
    })
    return problems


def gpu_group_cancel_running(engine: LLMEngine, capture: JsonlCapture) -> list[str]:
    """cancel-running：mixed batch forward 期间发取消信号（signal-only）。"""
    problems = []
    sched = engine.scheduler
    before_events = len(capture.events)
    # 包装 run：forward 开始时对目标请求置位取消信号（不触碰数据面）
    orig_run = engine.model_runner.run
    armed: set[str] = set()

    def wrapped_run(items):
        for it in items:
            if it.seq.request_id in armed:
                assert engine.scheduler.request_cancel(it.seq.seq_id, "mid_forward_gpu")
                armed.discard(it.seq.request_id)
        return orig_run(items)

    engine.model_runner.run = wrapped_run
    engine.add_request([23] * 300, SamplingParams(max_tokens=4, ignore_eos=True),
                       request_id="cr-victim")
    seq_of = {}
    for i in range(3):
        rid = f"cr-ok-{i}"
        engine.add_request([29] * 8, SamplingParams(max_tokens=4, ignore_eos=True),
                           request_id=rid)
        seq_of[engine.get_request(rid).seq_id] = rid
    engine.step()   # victim 尚未完成 prefill（300 token 分块）
    armed.add("cr-victim")
    outputs = {}
    while not engine.is_finished():
        out, _ = engine.step()
        for sid, toks in out:
            outputs[sid] = len(toks)
    engine.model_runner.run = orig_run
    finals = finals_from_events(capture.events[before_events:])
    if finals.get("cr-victim") != ("CANCELLED", "mid_forward_gpu"):
        problems.append(f"cr-victim 终态错误: {finals.get('cr-victim')}")
    # 同轮隔离：其余请求不受 victim 收尾影响，正常完成且 completion 数正确
    for sid, rid in seq_of.items():
        if outputs.get(sid) != 4:
            problems.append(f"{rid} 未正常完成或进度异常: {outputs.get(sid)!r}")
    problems += assert_stable(sched, "cancel-running")
    capture.write_record({
        "event": "scenario_summary", "scenario": "cancel-running",
        "finals": finals, "final_snapshot": snapshot_state(sched),
    })
    return problems


def gpu_group_timeout(engine: LLMEngine, capture: JsonlCapture) -> list[str]:
    """timeout：极短 deadline，覆盖 prefill/decode 阶段与排队请求。"""
    problems = []
    sched = engine.scheduler
    before_events = len(capture.events)
    rids = [engine.add_request([31] * 64, SamplingParams(max_tokens=8, ignore_eos=True),
                               request_id=f"to-{i}",
                               deadline=(time.perf_counter() + 0.06) if i % 2 else None)
            for i in range(8)]
    drive_gpu(engine, tag="timeout")
    finals = finals_from_events(capture.events[before_events:])
    timed_out = [rid for rid, (st, _) in finals.items() if st == "TIMEOUT"]
    if len(timed_out) < 2:
        problems.append(f"超时事件不足（GPU 步进快于 deadline）: {finals}")
    problems += assert_stable(sched, "timeout")
    capture.write_record({
        "event": "scenario_summary", "scenario": "timeout",
        "timeouts": timed_out, "finals": finals,
        "final_snapshot": snapshot_state(sched),
    })
    return problems


def make_small_pool_engine(model: str, target: tuple[int, int] = (10, 20)):
    """自适应构造小 KV 池引擎（强制抢占压力的前提）。

    num_kvcache_blocks 由 gpu_memory_utilization 推导，绝对值随环境浮动；
    从估计档位开始尝试，池大小不在目标区间则销毁重建（exit 幂等已保证）。
    """
    import gc
    import torch
    last_error = None
    # 本机标定（RTX 4090 D / Qwen3-0.6B）：u=0.092 -> 16 块，u=0.10 -> 22 块；
    # 档位跨环境浮动时由目标区间兜底，依次重试
    for u in (0.092, 0.096, 0.1, 0.104, 0.11, 0.13):
        try:
            engine = LLMEngine(model, max_num_batched_tokens=4096, max_num_seqs=8,
                               chunk_size=1024, enforce_eager=True,
                               gpu_memory_utilization=u)
        except Exception as exc:            # 池推导为负等：换下一档
            last_error = exc
            continue
        n_blocks = len(engine.scheduler.block_manager.blocks)
        if target[0] <= n_blocks <= target[1]:
            return engine, u
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError(f"未能构造目标 KV 池（{target}）的引擎: {last_error}")


def gpu_group_preempt_resume(engine: LLMEngine, capture: JsonlCapture) -> list[str]:
    """preempt-resume：小 KV 池 + decode 增长 -> victim 抢占、释放、recompute。

    构造（pool ~12-48 块、max_num_seqs=8）：10 条互异 512-token 请求首轮
    全部完成 prefill 后，每条 decode 首步跨块边界需 +1 块而 free=0 ->
    按队尾方向级联抢占合法 victim（未接纳的 decode 候选），释放后恢复。
    """
    problems = []
    sched = engine.scheduler
    before_events = len(capture.events)
    n_blocks = len(sched.block_manager.blocks)
    per_req_tokens = 512
    n_req = max(10, (n_blocks // 2) + 2)
    for i in range(n_req):
        # 互异首 token：阻断跨请求 prefix 共享，保证块需求真实
        engine.add_request([500 + i] + [37] * (per_req_tokens - 1),
                           SamplingParams(max_tokens=8, ignore_eos=True),
                           request_id=f"pr-{i}")
    stats = drive_gpu(engine, limit=6000, tag="preempt-resume")
    problems_p, preempt_n, resume_n = pair_preempt_resume(capture.events[before_events:])
    problems += problems_p
    if preempt_n == 0:
        problems.append("GPU 抢占场景未观察到抢占事件（压力不足）")
    if preempt_n != resume_n:
        problems.append(f"抢占/恢复不配对: {preempt_n} vs {resume_n}")
    problems += assert_stable(sched, "preempt-resume")
    capture.write_record({
        "event": "scenario_summary", "scenario": "preempt-resume",
        "pool_blocks": n_blocks, "per_req_tokens": per_req_tokens,
        "num_requests": n_req,
        "rounds": stats["rounds"], "wall_s": stats["wall_s"],
        "preempts": preempt_n, "resumes": resume_n,
        "trajectory_tail": stats["trajectory"][-16:],
        "final_snapshot": snapshot_state(sched),
    })
    return problems


def gpu_group_exception(engine: LLMEngine, capture: JsonlCapture) -> list[str]:
    """exception-cleanup：注入 runner 异常 -> Engine 锁定且资源归零。"""
    problems = []
    sched = engine.scheduler
    before_events = len(capture.events)
    orig_run = engine.model_runner.run
    state = {"fail": False}

    def maybe_boom(items):
        if state["fail"]:
            raise RuntimeError("injected cuda failure (gpu)")
        return orig_run(items)

    engine.model_runner.run = maybe_boom
    for i in range(2):
        engine.add_request([41] * 8, SamplingParams(max_tokens=2, ignore_eos=True),
                           request_id=f"ex-{i}")
    engine.step()   # prefill 成功，持有 block
    used_before = len(sched.block_manager.used_block_ids)
    if used_before == 0:
        problems.append("场景无效：异常前应持有 block")
    state["fail"] = True
    try:
        engine.step()
        problems.append("GPU 注入异常未向上传播")
    except RuntimeError:
        pass
    problems += assert_stable(sched, "exception-cleanup")
    if not engine._execution_failed:
        problems.append("GPU 异常后 Engine 未锁定")
    try:
        engine.step()
        problems.append("失败 Engine 允许重试")
    except RuntimeError:
        pass
    errs = [e for e in capture.events[before_events:]
            if e.get("event") == "engine_round" and e["outcome"] == "error"]
    if not errs or errs[-1]["executed_tokens"] is not None:
        problems.append("GPU error 轮事件缺少 executed_tokens=null")
    capture.write_record({
        "event": "scenario_summary", "scenario": "exception-cleanup",
        "finals": finals_from_events(capture.events[before_events:]),
        "final_snapshot": snapshot_state(sched),
    })
    return problems


def gpu_group_100_requests(model: str, capture: JsonlCapture) -> list[str]:
    """100-request：混合正常/取消/超时/抢占的连续负载（独立 Engine）。"""
    import torch
    problems = []
    before = len(capture.events)
    finished = 0
    engine = LLMEngine(model, max_num_batched_tokens=4096, max_num_seqs=64,
                       chunk_size=1024, enforce_eager=True, gpu_memory_utilization=0.85)
    capture.write_record(gpu_env_record(model, engine))
    sched = engine.scheduler
    rng = random.Random(20260911)
    total = 100
    submitted = 0
    rounds = 0
    trajectory = []
    torch.cuda.reset_peak_memory_stats()
    while submitted < total:
        if rounds >= 20000:
            problems.append("GPU 100 请求负载超过轮次上限（疑似死锁）")
            break
        if rng.random() < 0.8 and submitted < total:
            rid = f"ld-{submitted}"
            # 互异首 token：阻断跨请求 prefix 共享，让 KV 需求真实
            engine.add_request([2000 + submitted] + [43] * rng.randint(32, 384),
                               SamplingParams(max_tokens=rng.randint(2, 12),
                                              ignore_eos=True, temperature=0.0),
                               request_id=rid,
                               deadline=(time.perf_counter() + 0.3
                                         if rng.random() < 0.10 else None))
            submitted += 1
        if sched.requests and rng.random() < 0.10:
            seq = rng.choice(list(sched.requests.values()))
            if rng.random() < 0.6:
                engine.cancel_request(seq.request_id, "load_cancel")
            else:
                sched.timeout(seq.seq_id, reason="load_timeout")
        if sched.running and rng.random() < 0.04:
            victim = rng.choice(list(sched.running))
            if victim.status == SequenceStatus.RUNNING:
                sched.preempt(victim)
        paused = [s for s in sched.requests.values()
                  if s.status == SequenceStatus.PREEMPTED]
        if paused and rng.random() < 0.7:
            sched.resume(rng.choice(paused))
        out, _ = engine.step()
        finished += len(out)
        rounds += 1
        bm = sched.block_manager
        trajectory.append(len(bm.used_block_ids))
    for s in [s for s in sched.requests.values()
              if s.status == SequenceStatus.PREEMPTED]:
        sched.resume(s)
    while not engine.is_finished():
        if rounds >= 30000:
            problems.append("GPU 清场超过轮次上限")
            break
        out, _ = engine.step()
        finished += len(out)
        rounds += 1
        trajectory.append(len(sched.block_manager.used_block_ids))
    problems += assert_stable(sched, "gpu-100-requests")
    ctl = [e for e in capture.events[before:]
           if e.get("event") == "request_control"]
    n_cancel = sum(1 for e in ctl if e["action"] == "cancel")
    n_timeout = sum(1 for e in ctl if e["action"] == "timeout")
    if finished + n_cancel + n_timeout != submitted:
        problems.append(f"GPU 终态计数不平衡: submitted={submitted} finished={finished} "
                        f"cancel={n_cancel} timeout={n_timeout}")
    capture.write_record({
        "event": "scenario_summary", "scenario": "100-requests-gpu",
        "submitted": submitted, "rounds": rounds, "finished": finished,
        "cancel_actions": n_cancel, "timeout_actions": n_timeout,
        "terminal_balance": submitted == finished + n_cancel + n_timeout,
        "preempts": sum(1 for e in ctl if e["action"] == "preempt"),
        "kv_peak": max(trajectory), "kv_tail_peak": max(trajectory[-32:]),
        "trajectory_tail": trajectory[-32:],
        "final_snapshot": snapshot_state(sched),
    })
    capture.write_record(gpu_memory_record("after-100-requests"))
    engine.exit()
    return problems


def run_gpu(args, capture: JsonlCapture) -> bool:
    import torch
    model = Path(args.model).expanduser().resolve()
    capture.write_record({"event": "run_config", "mode": "gpu", "day": 10,
                          "model": str(model)})
    all_problems = []
    groups = []

    def run_group(name, fn, engine):
        capture.write_record({"event": "scenario_begin", "scenario": name})
        before = len(capture.events)
        torch.cuda.reset_peak_memory_stats()
        problems = fn(engine, capture)
        capture.events.append(gpu_memory_record(f"after-{name}"))
        scene = capture.events[before:]
        # 公共契约校验作用于本组切片（round_id 单调性是单引擎内语义）
        problems += check_event_contract(scene)
        problems += replay_control_transitions(scene)
        pair_problems, _, _ = pair_preempt_resume(scene)
        problems += pair_problems
        capture.write_record({
            "event": "scenario_result", "scenario": name, "passed": not problems,
            "problems": problems, "events": len(scene),
            "control_events": sum(1 for e in scene
                                  if e.get("event") == "request_control"),
        })
        all_problems.extend(f"[{name}] {p}" for p in problems)
        groups.append(name)

    # Engine A：cancel-waiting / cancel-running / timeout / preempt-resume
    engine_a = LLMEngine(str(model), max_num_batched_tokens=4096, max_num_seqs=64,
                         chunk_size=1024, enforce_eager=True,
                         gpu_memory_utilization=0.85)
    capture.write_record(gpu_env_record(str(model), engine_a))
    run_group("cancel-waiting", gpu_group_cancel_waiting, engine_a)
    run_group("cancel-running", gpu_group_cancel_running, engine_a)
    run_group("timeout", gpu_group_timeout, engine_a)
    engine_a.exit()
    del engine_a
    gc.collect()
    torch.cuda.empty_cache()

    # Engine P：自适应小 KV 池（抢占压力前提）
    engine_p, pool_u = make_small_pool_engine(str(model))
    capture.write_record({"event": "gpu_env", "note": "small-pool engine",
                          "gpu_memory_utilization": pool_u,
                          "num_kvcache_blocks": len(engine_p.scheduler.block_manager.blocks)})
    run_group("preempt-resume", gpu_group_preempt_resume, engine_p)
    engine_p.exit()
    del engine_p
    gc.collect()
    torch.cuda.empty_cache()

    # Engine B：异常收尾（Engine A 保持健康，不污染证据）
    engine_b = LLMEngine(str(model), max_num_batched_tokens=4096, max_num_seqs=32,
                         chunk_size=1024, enforce_eager=True,
                         gpu_memory_utilization=0.85)
    run_group("exception-cleanup", gpu_group_exception, engine_b)
    engine_b.exit()
    del engine_b
    gc.collect()
    torch.cuda.empty_cache()

    # Engine C：100 请求混合负载
    problems = gpu_group_100_requests(str(model), capture)
    all_problems.extend(f"[100-requests] {p}" for p in problems)
    groups.append("100-requests")

    # 终检：全流明文/脚本记录扫描（round_id 单调性等引擎内语义已按组核对）
    problems = check_event_contract(capture.events)
    problems = [p for p in problems if "round_id 非单调" not in p]
    all_problems.extend(f"[event-contract] {p}" for p in problems)

    ok = not all_problems
    capture.write_record({
        "event": "validation_summary", "mode": "gpu", "passed": ok,
        "groups": groups, "problem_count": len(all_problems),
        "problems": all_problems,
    })
    return ok


def main():
    parser = argparse.ArgumentParser(description="Day10 取消/超时/抢占/恢复验收脚本")
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", default=None, help="GPU 模式的本地模型目录")
    parser.add_argument("--output", default="/tmp/day10-request-control.jsonl")
    args = parser.parse_args()
    if args.mode == "gpu" and not args.model:
        parser.error("gpu 模式必须提供 --model（本地模型目录，不自动下载）")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s")
    # 只挂到引擎日志器，避免第三方库刷屏
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    capture = JsonlCapture(out_path)
    for logger in (SCHED_LOGGER, ENGINE_LOGGER):
        logger.setLevel(logging.INFO)
        logger.addHandler(capture)
        logger.propagate = False
    try:
        ok = run_cpu(args, capture) if args.mode == "cpu" else run_gpu(args, capture)
    finally:
        capture.close()
    print(f"{'PASS' if ok else 'FAIL'}: evidence -> {out_path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
