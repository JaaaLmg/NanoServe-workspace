# Day 9 实现审查记录

> 对照 [mixed-prefill-decode.md](./mixed-prefill-decode.md) 全文，对 `feature/mixed-prefill-decode`
> 分支工作区未提交改动（11 个修改文件 + 3 个新增文件）做的实现审查。结论：**CPU 侧语义达到
> 设计要求，GPU 交付被一个启动期阻断缺陷挡住（§3.1），另有一处与设计不变量 18 字面规则不符的
> 归因偏差需要决策（§4.1）**。§10 验收清单已按本记录证据逐项勾选（清单原文未改）。

## 1. 审查范围与实际命令

审查对象：`nanovllm/engine/{scheduler,llm_engine,model_runner,sequence}.py`、`scripts/validate_{mixed_batch,chunked_prefill,token_budget}.py`、`tests/` 5 个适配文件与新增 `tests/test_mixed_batch.py`、`docs/README.md`。审查方式：全文对照设计文档逐节核对 + 最小场景复现 + 实际执行下列命令。

| 命令 | 结果 |
| --- | --- |
| `python -m pytest -q` | **301 passed**（5.67s） |
| `python -O -m pytest -q` | **301 passed**（6.26s） |
| `PYTHONPATH=. python scripts/validate_mixed_batch.py --output /tmp/day9-mixed.jsonl` | **PASS**（17 轮；长 prompt 分块轮 8，其中 decode 产出 7/7；8K 请求首 token 轮 = 8） |
| `PYTHONPATH=. python -O scripts/validate_mixed_batch.py` | **PASS** |
| `PYTHONPATH=. python scripts/validate_token_budget.py` | **PASS**（7 轮存在 decode_priority 让路证据；KV free/used 收尾 64/0） |
| `PYTHONPATH=. python scripts/validate_chunked_prefill.py` | **PASS** |
| GPU 冒烟：`LLM(Qwen3-0.6B 快照, enforce_eager=True)` | **TypeError 崩溃**（§3.1） |

GPU 环境：RTX 4090 D（24 GB）+ Qwen3-0.6B 本地快照均可用，但引擎无法启动，GPU 验收全部受阻。

## 2. 已确认正确的实现

以下各项经代码核对 + 测试/脚本证据确认符合设计：

1. **混合批次表示与四值 phase**（§3.1/§3.2）：`BatchItem` 逐请求标注，items decode 在前、prefill 在后，`assert_mixed_round_invariants` 逐轮断言；纯阶段/idle 为退化特例，Day7/8 行为断言原样通过（301 项回归含全部既有测试）。
2. **decode-first 与混合预算公式**（§3.3/§3.4）：decode 阶段先占预算与名额；四条上限（正整数、q≤chunk_size、总量≤B、条数≤max_num_seqs）+ decode q==1 + seq_id 去重 + BatchItem/seq 计划一致性在 `_finalize_round` 独立显式校验；Engine 侧 §4.2 校验齐备且不依赖 assert。CPU 主负载实测：8K prompt 分块期间 decode **7/7 轮持续产出**（首个 prefill 轮无 decode 源属正常），decode 保底是结构性的。
3. **归因重构**（§3.5）：`REASON_PHASE_PRIORITY` 已删除，全仓库代码无残留（仅历史文档与"已删除"注释提及）；`decode_priority` 不计入预算等待 episode（`_record_budget_decisions` 显式排除）；HOL 的 `blocking_reason` 跟随直接原因；验收脚本可独立复算判定条件。
4. **同轮分组执行**（§4.1）：`run(items)` 先 decode 后 prefill 两个子批，各自 `reset_context`；`Context`/`Attention`/`ParallelLMHead` **零修改**（git 确认未触碰）；合并顺序 = items 中 `needs_sample` 出现顺序，与 postprocess 消费共享同一快照谓词，两端不允许隐式重排；`needs_sample` 与 `Sequence.is_last_chunk_scheduled` 谓词交叉校验（不一致显式抛错）。
5. **逐 item 提交**（§4.3）：round_id 关联校验（与 `last_schedule_stats` 比对）+ `num_scheduled_tokens` 清零检测双保险；token 数与 `sum(needs_sample)` 显式校验、不静默截断；先整批校验再逐个提交；同轮单 item 终态（cancel/timeout/外部终态）不影响其他 item（专测覆盖）；prefill/decode 统一 `hash_blocks` 显式区间，decode 区间 [len-1, len) 与 Day7 逐位一致。
6. **接口与协议**（§6.2/§6.3）：`is_last_chunk_scheduled` 谓词落地，scheduler 接纳/决策快照、model_runner 行选择、engine 快照（经 BatchItem 字段传递，值源自同一谓词）收敛；`Sequence` pickle v3 协议未动（diff 仅新增 property），`BatchItem`/`round_id` 不进 payload；`step()` 返回口径改恒非负、`generate()` 吞吐改读 `prefill_tokens/decode_tokens` 结构化字段；`prefill_chunks` 保留为 `prefill_items` 同值别名（§4.4 允许的二选一分支）。
7. **事件契约**（§6.4）：`scheduler_round`/`engine_round` 白名单版本化（3 个脚本同步），idle/error 分支字段穷举补齐；无 prompt/token 明文。
8. **一条值得记录的验证**：decode 阶段被抢占的 victim（preempt+resume 至 waiting 队首）**同轮不会被 prefill 阶段重接纳**——`can_allocate` 要求整序列块数全部可用，而 decode 候选恰好消耗 victim 腾出的边界块，经验证 victim 同轮只能记 `kv_capacity`。因此"一轮一请求一决策"在 decode-first 重排后依然成立，事件消费方无需按 seq 聚合同轮多条决策。

## 3. 阻断项（GPU 交付前必须修复）

### 3.1 P0：`warmup_model` 仍按 Day8 签名调用 `run()`，GPU 引擎启动即崩溃

`model_runner.py:101`：

```python
seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
for seq in seqs:
    seq.num_scheduled_tokens = seq_len
self.run(seqs, True)        # ← run() 签名已改为 run(items)，此处未适配
```

实际执行证据（RTX 4090 D，enforce_eager=True）：

```text
File "nanovllm/engine/model_runner.py", line 101, in warmup_model
    self.run(seqs, True)
TypeError: ModelRunner.run() takes 2 positional arguments but 3 were given
```

影响：`ModelRunner.__init__` → `warmup_model()` 在引擎构造路径上，**任何 GPU/真实模型启动必然崩溃**。CPU 测试与验收脚本全部用桩 runner（`engine.model_runner = SimpleNamespace(call=...)`），不会触达该路径，因此 301 项测试全绿也无法暴露。

建议修复（与 Day8 warmup 语义逐位一致：warmup 序列 `prefill_offset=0`、`num_scheduled_tokens=seq_len`、`prefill_target=seq_len`，谓词判定为最后 chunk，全部行采样）：

```python
from nanovllm.engine.scheduler import BatchItem   # scheduler 不反向依赖 model_runner，无环

items = [BatchItem(seq=seq, phase="prefill", scheduled_tokens=seq.num_scheduled_tokens,
                   offset_before=0, is_last_chunk=True, needs_sample=True, round_id=0)
         for seq in seqs]
self.run(items)
```

修复后需按设计 §9.4 重跑 GPU 验收（当前环境具备：RTX 4090 D + Qwen3-0.6B 快照）。

## 4. 设计偏差与建议（不阻断 CPU 语义，需决策）

### 4.1 `decode_priority` 判定用"当前候选需求"，与不变量 18 的 `needed_first` 字面规则不符

设计 §3.5/§5.2（不变量 18）规定的可检验规则是 **`D > 0 且 needed_first <= B`**（`needed_first` = 本轮首个 prefill 候选的需求）。实现的两处归因点（`scheduler.py` 的 `remaining == 0` 分支与 `q < needed and prefill_admitted` 分支）都用**当前被延后候选自己的 `needed`** 参与判定。

实际复现（B=8，chunk_size=8，2 条 decode 源，waiting=[p1(20 token), p2(5 token)]）：

```text
phase = mixed
  d1  decode   scheduled   sched=1
  d2  decode   scheduled   sched=1
  p1  prefill  scheduled   needed=20 sched=6   ← 首候选拆分 q=min(20,8,6)
  p2  prefill  decode_priority         needed=5
```

按设计的 `needed_first` 规则：`needed_first=20 > B=8` → p2 应记 `budget`（无 decode 时 p1 会整占 8 token，p2 同样放不下，延后并非 decode 造成）；实现记了 `decode_priority`。这正是 §8 风险表点名的"把纯预算不足记成优先级让路（污染吞吐归因）"的误判方向。

需要说明的是，两条规则各有一个反例方向：在"D 很大、首候选只分到很小的 chunk"的场景（如 D=2000、B=2048、首候选 8K 只得 48 token），后续小候选无 decode 时本可接纳，实现按当前候选 `needed<=B` 记 `decode_priority` 反而更符合"预算短缺纯由 decode 造成"的语义，而 `needed_first` 规则会误记 `budget`。即：**实现更准的场景与设计字面规则更准的场景互为镜像，但设计文档是权威**，且 §9.3 验收脚本的独立复算口径（`d.needed_tokens <= B`）是随实现写的，不构成对设计规则的核对。

建议二选一（倾向前者，改动面小且使清单第 4 项可勾选）：

1. **代码对齐设计**：schedule() 中记录首轮考察的首个 prefill 候选 `needed_first`（含 `remaining==0` 时未接纳即停止的情形），两处归因改用 `decode_used > 0 and needed_first is not None and needed_first <= budget`；同步调整 `test_mixed_batch.py` 三分支单测与 `validate_mixed_batch.py` 的复算口径，并在 `docs/day9-validation.md` 记录该取舍。
2. **修订设计规则**：经用户确认后把 §3.5 判定改为"当前候选 `needed <= B`"，并同步不变量 18 表述（验收清单原文仍不改）。注意这条规则在碎片化场景仍会高估 decode 责任，须在文档写明是"可检验的近似"。

### 4.2 可读性与一致性建议

1. `scripts/validate_mixed_batch.py` `run_main_workload` 声明 `-> list[str]`，实际返回 `(problems, ledgers)` 元组，注解应改为 `-> tuple[list[str], list[dict]]`。
2. `scheduler.postprocess`：round 校验处 `any(not s.is_terminal for s in seqs)` 与其后 `has_live = any(not seq.is_terminal for seq in seqs)` 是同一谓词算了两遍，可复用一个 `has_live`。
3. `llm_engine.step()`：7 元组快照在 5 处以 `_, _, ph, n, _, _, _` 位置解包，且 `prefill_items` 计数表达式在 idle/error/completed 三个分支重复出现 3 次。建议引入一个小 dataclass（或 ` namedtuple`）快照类型 + `sum(1 for s in snapshot if s.phase == "prefill")` 的局部变量，可读性和抗字段增删能力都更好。
4. `Scheduler.schedule()` 约 190 行，decode 阶段（含抢占循环）可提取为 `_schedule_decode_phase(...)` 私有方法，与 prefill 扫描对称，主流程只剩两阶段骨架。属可选重构，当前实现正确性无问题。
5. `_finalize_round` 的 chunk_size 校验注释说"显式只查 prefill item"，实现一致；但 `planned <= 0` 校验在"每个 scheduled_tokens 为正整数"校验之后实为冗余（非空批次各 item ≥ 1 则 planned ≥ 1），可留作防御性冗余，不算问题。

## 5. 验收结论与遗留

> **（2026-09-12 第二轮复核更新）下列遗留项已全部闭环**，修复验证与 GPU 实测见 §6 与
> [day9-validation.md](./day9-validation.md)；§10 验收清单 15/15 项均已勾选。

- **CPU 侧**：设计 §10 清单"功能与口径""执行正确性"两组及"CPU 全量回归"项全部有实证，已勾选（除 §4.1 涉及的归因项）。
- **GPU 侧**：主负载 benchmark、mixed/one-shot 一致性、`day9-validation.md`、未测项记录共 4 项**未完成**，直接原因是 §3.1 的 warmup 阻断缺陷；修复并跑完 §9.4 后补勾。
- 遗留待办（按优先级）：
  1. 修复 `warmup_model` 签名（§3.1），重跑 GPU 冒烟确认可启动；
  2. 决策 §4.1 归因规则并落地（代码或设计二选一）；
  3. 按设计 §9.4 执行 GPU 主负载 `1×8K + 32×128` 与 one-shot 对照，归档 `docs/evidence/day9/`；
  4. 撰写 `docs/day9-validation.md`（GPU/模型可用性、实际命令、结果、未覆盖边界），更新 `docs/README.md` 索引状态；
  5. 可选：§4.2 的可读性项随修复一并处理。

## 6. 修复复核（第二轮，2026-09-12）

第一轮两项发现已由实现侧修复，本轮独立复核全部通过：

1. **§3.1 P0（warmup 签名）——已修复并实测**：`warmup_model` 按 Day8 逐位一致语义构造
   `BatchItem`（`needs_sample=True`、谓词成立）调用 `run(items)`；GPU 冒烟引擎正常启动并
   产出连贯结果；`validate_mixed_batch.py --mode gpu` 完整重跑（one-shot + mixed 两个
   variant）事件校验 PASS，指标与归档证据在噪声范围内复现（mixed 短请求 TTFT P50
   0.196s vs 归档 0.198s、decode 产出 7/8 轮、greedy 一致性 26/33 同集合）。
2. **§4.1 归因偏差——已按建议一修复并复现验证**：`decode_priority` 判定改为不变量 18
   字面规则，`needed_first` 在 prefill 扫描首次成功估算时记录并新增入 `scheduler_round`
   事件（3 个脚本白名单同步版本化）。原反例场景（B=8、D=2、首候选 20 拆分、次候选 5）
   复跑：次候选拒绝记 `budget`（修复前误记 `decode_priority`），`needed_first=20` 入档可复算。
   新增反例单测 2 条；既有 episode 统计断言按新规则重推后逐项保持（303 passed 双模式）。
3. **§4.2 可读性建议**：全部采纳（返回注解、`has_live` 复用、`_ItemSnapshot` NamedTuple、
   `_schedule_decode_phase` 提取），复核确认重构未改变行为。
4. **交付物核对**：`docs/day9-validation.md` 记录完整（环境/命令/结果/取舍/未覆盖边界）；
   `docs/evidence/day9/` 三份 JSONL 与记录数字一致，引擎事件经白名单校验无 prompt/token
   明文（`greedy_compare` 携带输出 token 供核对，沿用 Day8 同类证据先例）；
   `docs/README.md` 索引已更新。

结论：**Day 9 达到提交条件**（设计 §10 清单 15/15；TP>1 / CUDA Graph / 真实并发取消按
设计 §9.5 口径如实记录未测）。改动仍在工作区未提交，提交由用户执行。
