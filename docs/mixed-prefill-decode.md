# Day 9 混合 Prefill 与 Decode 设计

> 本文对应 `plan.md` 阶段二 Day 9，目标是让一个调度轮次同时容纳 prefill chunk 与 decode token，优先保障已有 decode 请求不被长 prefill 独占 GPU，并处理新请求动态加入、请求完成和 KV block 变化。本文是开发设计文档，描述目标行为、接口约定、实现步骤和验收标准；其中"当前实现"指编写文档时仓库（基线 dev=`42e142d`，含 Day8）的实际代码，不代表 Day9 已经完成。

- 适用范围：`Scheduler` 的混合轮调度与归因、批次数据模型（`BatchItem`）、`ModelRunner` 的分阶段执行契约、`LLMEngine.step()` 返回协议与事件、`postprocess` 的逐请求阶段提交、CPU/GPU 测试与混合负载 benchmark。
- 不在本日范围：单次 fused mixed attention（见 §2.2 显式设计决策）、完整取消/超时/抢占恢复策略（Day10）、HTTP/SSE 与流式输出（Day11–13）、Prefix Cache 淘汰（LRU/LFU）、自适应 chunk size、请求优先级、量化/投机解码/PD 分离。
- 相关设计：[请求生命周期](./request-lifecycle.md)、[每轮 Token Budget](./token-budget.md)、[Chunked Prefill](./chunked-prefill.md)、[Prefill/Decode](./prefill-decode.md)、[架构](./architecture.md)。
- 涉及代码：`nanovllm/engine/scheduler.py`、`nanovllm/engine/llm_engine.py`、`nanovllm/engine/model_runner.py`、`nanovllm/engine/sequence.py`、`nanovllm/engine/block_manager.py`、`nanovllm/utils/context.py`（不改）、`nanovllm/layers/attention.py`（不改）、`nanovllm/layers/embed_head.py`（不改）。

## 1. 需求背景与当前差距

Day8 完成后，一轮调度仍然是**整批单阶段**的：`schedule()` 返回 `(list[Sequence], bool)`，只要有可接纳的 prefill 就整轮做 prefill 并直接返回，running 队列统一记 `phase_priority` 让路（`scheduler.py:673-684`）。`postprocess(seqs, token_ids, is_prefill)` 依赖全批统一的 `is_prefill` 判定采样与提交；`ModelRunner.run(seqs, is_prefill)` 要求整批同阶段；`Context.is_prefill` 是全局单值，`Attention` 与 `ParallelLMHead` 据此在 varlen prefill 与 paged decode 两条 kernel 路径间二选一。

这套"prefill 优先"策略有一个计划文档明确点名的缺陷：**长 prompt 持续到达时 decode 完全停滞**——只要 waiting 队首还有一个 8K 请求在逐块推进，所有 running 请求的 TPOT 就无上界（`architecture.md` §7）。Day7 的 token budget 限制了每轮总量，但 prefill 优先意味着预算总是先被 prefill 吃满，decode 只能等 prefill 排空。

此外 Day8 审查（`day8-review.md` §4）留下了三项与本日直接相关的遗留，Day9 应一并收口：

1. `runner_input_len`（实际 runner 输入长度 vs 调度计划的独立交叉核对）未闭环；
2. 迟到 postprocess 在"同一 seq 已被重新规划"后无法识别（仅靠 `num_scheduled_tokens` 清零检测），同步引擎内不可达，但混合轮引入后应按 round 关联加固；
3. `prefill_offset + num_scheduled_tokens == prefill_target`（本轮接纳即完成 prefill）在 scheduler/model_runner/engine/测试/脚本中重复 5 处，应收敛为 `Sequence` 上的具名谓词。

`plan.md` Day 9 的四项任务与两条验收：

> 1. 设计一轮调度中同时包含 prefill chunk 和 decode token 的 batch 表示。
> 2. 优先保障已有 decode 请求，避免长 prefill 独占 GPU。
> 3. 处理新请求动态加入、请求完成和 KV block 变化。
> 4. 构造"1 个 8K prompt + 32 个 128-token prompt"的混合负载。
>
> - 混合负载下 decode 请求可持续产出 token。
> - 相比原始一次性 Prefill，短请求 P95 TTFT/TPOT 有可解释的变化。

## 2. 目标与非目标

### 2.1 必须达到的目标

1. **混合批次表示**：一轮调度可同时包含 prefill item 与 decode item，批次从"单阶段列表 + 全局布尔"升级为逐请求阶段标注的 `BatchItem` 列表；纯 prefill、纯 decode、idle 是混合形态的退化特例，既有语义不变。
2. **decode 优先保障**：同一轮内 decode 请求先于 prefill 获得预算与序列名额（decode-first 固定策略），已有 decode 请求在长 prefill 持续存在时仍能每轮推进 1 token，不再被记为 `phase_priority` 整轮让路。
3. **混合预算公式**：每轮 `sum(prefill q_i) + count(decode items) <= B`，`len(items) <= max_num_seqs` 跨阶段共享；每个 prefill item `0 < q_i <= chunk_size`，每个 decode item `q_i == 1`；四条上限独立显式校验（沿用 `_finalize_round` 机制）。
4. **归因重构**：废除 `phase_priority`；新增 `decode_priority` 原因（prefill 候选因预算被 decode 占用而延后，且无 decode 时本可接纳）；`budget/head_of_line/sequence_cap/kv_capacity/paused` 口径延续；预算等待 episode 统计对 prefill 延后继续生效。
5. **动态请求正确性**：新请求加入 waiting、prefill 最后 chunk 完成（WAITING→RUNNING）、decode 完成（FINISHED）、chunk 中取消/超时、抢占恢复 recompute、ID 复用等既有生命周期路径在混合轮内全部保持 Day6/7/8 不变量。
6. **采样对齐**：prefill 中间 chunk 不采样、不消耗 RNG（Day8 契约沿用）；prefill 最后 chunk 与 decode item 各产出 1 个采样 token；token 结果与 item 一一映射，数量不匹配显式抛错，不使用会静默截断的 zip。
7. **证据链收口**：`round_id` 贯穿 schedule → run → postprocess；`runner_input_len`（decode/prefill 两个子批的实际展平输入长度）由验收脚本侧 runner 包装独立观测并与计划对账（收 Day8 遗留）；事件字段白名单化、无 prompt/token 明文。
8. **兼容回归**：Day6/7/8 全部既有测试经适配后语义保持（单阶段轮的行为断言原样通过）；`generate()` 输出格式、采样参数语义、TP pickle v3 协议不变。

### 2.2 明确不做的事情

- **不实现单次 fused mixed attention**（显式设计决策）：`Context`/`Attention`/`ParallelLMHead` 当前都以整批单阶段为前提（`context.is_prefill` 全局单值、varlen 与 paged decode 两条 kernel 二选一、LM head prefill 分支按请求聚合末 query 行）。Day9 采用**同一调度轮内按 phase 分组先后执行两次前向**（decode 子批 + prefill 子批），完整复用现有 kernel 与输入组装契约；把三处单阶段假设改为支持真正单次混合前向（合并 block table、混合 slot mapping、混合 logits 行映射、CUDA Graph 重捕获）留作后续演进，不在本日冒险改坏已验证的 attention 数值语义。
- 不实现 Day10 的完整取消/超时/抢占恢复增强；沿用 Day6/7/8 既有行为，仅保证其在混合轮内不被破坏。
- 不做 HTTP/SSE、流式输出、请求优先级调度、Prefix Cache 淘汰、自适应 chunk size。
- 不新增 `Config` 字段：decode-first 是固定调度策略，不做可配置的 decode 保留预算比例（预算公式天然保证 decode 先占）。
- 不改变 `generate()` 的最终输出格式与采样参数语义；`step()` 第二个返回值的符号语义变更属于接口演进（§4.4），不属于行为扩展。
- 不承诺吞吐/TTFT/TPOT 的改善方向：benchmark 只如实记录观测值，"可解释的变化"包括变差的可能及原因分析。
- TP>1、CUDA Graph 捕获路径不强制实测（单卡环境 TP>1 不可行；GPU 验证统一 `enforce_eager=True` 基线），按既有方法学如实记录未测。
- 不支持运行中热修改任何调度参数。

## 3. 术语、数据模型与调度语义

### 3.1 核心定义

| 概念 | 定义 | 性质 |
| --- | --- | --- |
| **mixed round** | 一个调度轮次，其批次内同时含至少一个 prefill item 和至少一个 decode item | 调度层概念；phase 标签记为 `"mixed"` |
| **prefill item** | 批次中按 prefill 语义执行的请求：消费 `[prefill_offset, prefill_offset+q)` 区间，`0 < q <= chunk_size`，仅最后 chunk 采样 | 由 schedule 决定 |
| **decode item** | 批次中按 decode 语义执行的请求：`q == 1`，消费 `last_token`，每轮采样 1 token | 由 schedule 决定 |
| **BatchItem** | 逐请求阶段标注的数据结构（§3.2），一个请求在一轮中至多出现一次 | 调度产物，单一权威 |

`phase`（轮级标签）取值：`"prefill"`（仅 prefill item）、`"decode"`（仅 decode item）、`"mixed"`（两者都有）、`"idle"`（空批次）。既有测试与事件中的 `is_prefill` 布尔由 phase 派生兼容。

### 3.2 `BatchItem`：逐请求阶段标注

```python
@dataclass
class BatchItem:
    """单个请求在本调度轮的执行计划（仅 rank 0 持有，不进 TP Sequence payload）。"""
    seq: Sequence            # 执行对象引用（TP 传输经 Sequence pickle v3，协议不变）
    phase: str               # "prefill" | "decode"
    scheduled_tokens: int    # prefill 为 q（0<q<=chunk_size）；decode 恒为 1
    offset_before: int       # prefill：接纳时刻的 prefill_offset；decode 为 None
    is_last_chunk: bool      # prefill：offset_before + q == prefill_target；decode 为 False
    needs_sample: bool       # decode 恒 True；prefill 等于 is_last_chunk（中间 chunk 不采样）
    round_id: int            # 调度轮次 ID，postprocess 关联校验用
```

- `schedule()` 返回 `(items: list[BatchItem], phase: str)`。items 顺序固定为 **decode item 在前、prefill item 在后**（与调度顺序一致，也是执行子批顺序），同 phase 内保持各自队列的 FCFS 顺序。
- `needs_sample` 在调度时按计划快照冻结，`postprocess` 不再从全批 `is_prefill` 推导——这是混合轮采样对齐的单一权威（Day8 的"快照先于状态变更"原则延伸到 phase 维度）。
- `round_id` 使 postprocess 能校验"本批次确属当前调度轮"，双保险加固迟到结果检测（`num_scheduled_tokens` 清零检测保留）。
- `BatchItem` 为纯 rank 0 侧结构：不进入 `Sequence.__getstate__` payload，不改变 v3 协议；TP>1 时经 `model_runner.call("run", items)` 广播（dataclass 默认 pickle，`seq` 字段走既有 Sequence 序列化）。

### 3.3 混合预算公式与三量区分

每轮预算口径（Day7 §11 衔接条款的落地）：

```text
planned_tokens = sum(prefill item 的 q_i) + count(decode items)   # ≤ B
batch_size     = count(prefill items) + count(decode items)       # ≤ max_num_seqs
```

三个"大小"严格区分（沿用 Day8 §3.1 表）：`chunk_size`（单请求单轮 prefill 上限）、`max_num_batched_tokens`（全轮总预算 B）、`kvcache_block_size`（物理块粒度）。混合轮不改变三者的独立性与校验入口。

### 3.4 调度策略：decode-first

`schedule()` 单轮流程（替换现有"prefill 优先、整轮二选一"）：

```text
阶段 0  调度边界扫描（不变）：终态兜底 → 取消 → 超时（顺序即优先级）

阶段 1  decode（先于 prefill，decode-first）：
  对 running 队列按 FCFS 逐条考察：
    - len(items) >= max_num_seqs 或 used >= budget → 其余 decode 候选延后
      （cap 记 sequence_cap，预算记 budget，口径与 Day7 一致），停止
    - can_append 不足 → 抢占规则沿用（队尾牺牲，preempt + resume，记 kv_capacity）
    - 接纳：q = 1，may_append，used += 1，记 scheduled
  已接纳的 decode 保持 running 队列相对顺序

阶段 2  prefill（用剩余预算，沿用 Day8 全部接纳规则）：
  remaining = B - used（used 已含 decode 占用量）
  对 waiting 按"扫描位置与队列分离"规则考察：
    - 序列数名额与预算耗尽 → 停止（归因见 §3.5）
    - 首候选允许拆分：q = min(needed, chunk_size, remaining)
    - 后续候选必须整段放下（q == needed），否则停止扫描（FCFS 不跳过）
    - 接纳：allocate（首次）/ 续 chunk；最后 chunk 接纳时 WAITING→RUNNING
```

设计要点：

1. **decode 保底是结构性的**：decode 先分配预算与名额，prefill 只能用剩余量。decode 接纳永远不被同轮 prefill 挤掉；这直接兑现"优先保障已有 decode 请求"。无需引入保留比例配置。
2. **KV 记账跨阶段统一**：decode 的 `may_append` 与 prefill 的 `allocate` 在同一轮内先后消耗同一 free 池，`can_allocate`/`can_append` 的实时查询天然保证不超卖；KV 不足时 decode 侧抢占、prefill 侧停止接纳，原因独立归因。
3. **每请求每轮至多一个 item**：同一 seq 不允许同时以 prefill 和 decode 出现（硬校验，seq_id 去重沿用 `_finalize_round`）。prefill 最后 chunk 接纳后该请求本轮**不再**作为 decode item（其首个 completion token 由最后 chunk 的采样产生，与 Day8 语义一致）。
4. **纯阶段退化**：无 running 时阶段 1 为空（纯 prefill 轮）；无 waiting 接纳或 remaining 不足以放下任何 prefill 时阶段 2 为空（纯 decode 轮）；两者皆空为 idle 轮。三种退化形态的日志、事件与行为必须与 Day8 完全一致。

### 3.5 归因重构：`phase_priority` → `decode_priority`

`phase_priority` 的语义是"本轮已被另一阶段整体占用"，mixed 轮下不再成立。重构规则：

| 场景 | 旧归因 | 新归因 |
| --- | --- | --- |
| prefill 候选延后，且本轮 decode 占用 D>0，且无 decode 时该候选本可接纳（首候选 `needed <= B`，即预算短缺纯由 decode 造成） | `phase_priority` | `decode_priority`（新增） |
| prefill 候选延后，且 `needed > B`（无 decode 也放不下）或 D==0 | `budget` | `budget`（口径不变） |
| 被 decode 挤掉候选的后续 waiting 请求 | `phase_priority`/HOL | `head_of_line` + `blocking_reason` 指向首候选的直接原因 |
| decode 候选因预算/名额延后（decode 数 > B 或 > max_num_seqs） | `budget`/`sequence_cap` | 不变（decode-first 下仍可能发生） |
| 独立 PREEMPTED | `paused` | 不变 |

- `decode_priority` 的判定是**可检验的显式规则**：`D > 0 且 needed_first <= B`。测试必须覆盖三种分支（decode 挤占、需求本身超 B、D==0）。
- `decode_priority` **不计入预算等待 episode**（与 `phase_priority` 同理：不是预算的锅，是优先级策略的让路）；episode 结算规则沿用 Day7（接纳/原因变化/终态/抢占时结算）。
- `REASON_PHASE_PRIORITY` 常量删除；事件白名单、验收脚本、文档索引同步版本化更新。

### 3.6 与现有字段的关系

- `Sequence.is_prefill` 保留：作为请求当前所处阶段的标记（schedule 接纳时设置，postprocess 提交路径与 TP payload 消费），但**不再是批次的阶段权威**——批次权威是 `BatchItem.phase`。
- `num_scheduled_tokens` 语义不变：本轮已接纳、待执行的 query 数；调度接纳时设置，postprocess 成功后清零。
- `prefill_offset` 仍是唯一进度事实源；`hash_blocks(seq, offset_before, offset_before + q)` 显式区间提交对 prefill item 与 decode item 统一适用（decode 每轮区间 `[len-1, len)`，与 Day7 逐 token 登记行为逐位一致）。

## 4. 端到端执行契约

### 4.1 `ModelRunner.run(items)`：同轮分组执行

`run()` 改为接收 `list[BatchItem]`，内部按 phase 拆成两个**有序子批**先后执行（decode 子批在前）：

```text
decode 子批（非空时）：
  prepare_decode(decode_seqs)        # 沿用：last_token / num_tokens 元数据 / may_append 已在调度期补块
  run_model(is_prefill=False)        # 沿用；enforce_eager=False 且 bs≤512 时可走 CUDA Graph 路径
  采样：全部 decode item 采样 1 token（prepare_sample 沿用）

prefill 子批（非空时）：
  prepare_prefill(prefill_seqs)      # 沿用 Day8 显式 offset 契约：绝对位置、cu_seqlens、slot_mapping
  run_model(is_prefill=True)
  采样：仅最后 chunk item 采样（_select_prefill_sample_rows 沿用，
        logits 行数 == prefill 子批请求数显式校验）
```

- 每个子批执行后 `reset_context()`，两个子批互不残留 `Context` 状态；`Context`/`Attention`/`ParallelLMHead` **零修改**（§2.2 设计决策）。
- 返回值：两个子批的采样 token 按 **items 顺序**合并为一个扁平列表——即 `decode 子批结果（按 items 中 decode 顺序）+ prefill 需采样子批结果（按 items 中 prefill 顺序）`；整轮无任何采样 item 时返回 `None`（沿用"无采样返回 None"契约）。
- 合并顺序是**显式契约**：`postprocess` 按同一 items 顺序与 `needs_sample` 快照对齐消费（§4.3），两端共享同一谓词，不允许任何一端隐式重排。
- `runner_input_len` 观测：decode 子批展平输入长度 = decode 条数；prefill 子批展平输入长度 = `sum(q_i)`。库内不新增事件字段；验收脚本以 runner 包装在两个子批边界注入观测（沿用 Day8 设计口径："不属于库内事件"），并与调度计划对账——这是 Day8 §4.1 遗留条款的收口位置。
- 异常边界：任一子批异常 → Engine 记 error 事件（`executed_tokens=null`）并锁定，禁止同一 Engine 重试；异常不区分发生在哪个子批（锁定语义一致）。

### 4.2 调用前快照与计划校验（Engine）

`LLMEngine.step()` 在模型调用前保存逐 item 快照（扩展 Day8 的 5 元组）：

```python
snapshot = [(item.seq.seq_id, item.seq.request_id, item.phase,
             item.scheduled_tokens, item.offset_before, item.is_last_chunk,
             item.round_id) for item in items]
```

显式校验（不依赖 assert，`python -O` 生效）：

1. 非空批次每个 `scheduled_tokens` 为正整数；
2. `sum(prefill q_i) + count(decode) <= B`（混合口径）；
3. seq_id 无重复、phase ∈ {"prefill", "decode"}、decode item 的 `scheduled_tokens == 1`。

### 4.3 `postprocess(items, token_ids, now=None)`：逐 item 提交

签名变更：移除全批 `is_prefill` 参数，新增隐式 round 关联（items 携带 `round_id`，与 `scheduler._current_round_id`/`last_schedule_stats` 比对，不匹配即拒绝——迟到结果双保险之一；`num_scheduled_tokens` 清零检测保留为二保险）。

```text
前置校验：
  1. round_id 与当前调度轮一致（不一致：重复/迟到旧批次，显式拒绝）
  2. token 数 == sum(item.needs_sample)（items 顺序对齐，不静默截断）
  3. 含活动请求的批次中，每个活动 item 的 num_scheduled_tokens > 0
     （全终态批次重放豁免，沿用 Day6/7 契约）

逐 item（按 items 顺序，ti 为 token 游标）：
  安全检查（沿用，顺序不变）：终态兜底 → 取消标记 → 跨 deadline
    命中：幂等清理，不推进 offset，token 占位游标仍前进
  提交（成功路径）：
    hash_blocks(seq, offset_before, offset_before + q)   # prefill 与 decode 统一
    prefill_offset += q；num_scheduled_tokens = 0；推进后校验单调有界
  阶段分支（由 item.phase 决定，替代全批 is_prefill）：
    prefill 且未完成（offset < target）：丢弃 token（中间 chunk），保持 WAITING
    prefill 最后 chunk / decode：append_token(ti 处采样值)；
      EOS → mark_finished("stop")；max_tokens → mark_finished("length")
```

原子性、幂等、所有权保护全部沿用 Day8 §4.5：先整批校验再逐个提交、失败不提交、重复/迟到收尾拒绝、ID 复用隔离。混合轮新增约束：**同一轮内单个 item 的终态不影响其他 item 的提交**（测试必须覆盖"同轮一个 decode 完成或取消、其余 item 正常落账"）。

### 4.4 `step()` 返回协议与事件

- 返回值 `(outputs, num_tokens)` 结构不变，但 `num_tokens` 语义变更为**本轮总 query token 数（planned，恒为正；idle 轮为 0）**，不再用符号编码阶段。`generate()` 中 prefill/decode 吞吐显示改由 `scheduler.last_schedule_stats` 的结构化字段（`prefill_tokens`/`decode_tokens`）计算，符号口径退役。
- `engine_round` 事件字段扩展（白名单版本化更新）：
  - `phase`: `"prefill" | "decode" | "mixed" | "idle"`；
  - `prefill_tokens` / `decode_tokens`：分阶段计划 token 数（decode 轮 = 条数）；
  - `prefill_items` / `decode_items`：分阶段 item 数（`prefill_chunks` 保留为 `prefill_items` 的同值字段以平滑脚本迁移，或随白名单升级一并改名——实现时二选一并同步验收脚本，不允许两套口径并存）；
  - `executed_tokens`：总实际输入（= planned，调用前快照口径沿用）。
- `scheduler_round` 事件：`phase` 字段同步扩展；`decisions[]` 每条新增 `phase`；原因白名单删除 `phase_priority`、新增 `decode_priority`。

### 4.5 采样契约（混合轮)

1. decode item 全部采样；prefill item 仅最后 chunk 采样；中间 chunk 不采样、不消耗 RNG（Day8 不变量 7 原样继承）。
2. 每请求每轮至多 1 个采样 token；一轮混合批次总采样数 = `count(decode items) + count(最后 chunk items)`。
3. RNG 跨子批顺序：decode 子批先采样，prefill 子批后采样。固定 seed 下的确定性由此顺序定义；测试以 generator 状态判据锁定（中间 chunk 前后 RNG 状态不变）。
4. token 注入与消费共享同一 `needs_sample` 谓词（`BatchItem.needs_sample`），测试注入助手按 items 构造，两端不可能错位（Day8 `tokens_for_batch` 方法的混合版）。

### 4.6 动态请求与生命周期组合

| 场景 | 混合轮内行为 | 继承来源 |
| --- | --- | --- |
| 新请求动态加入 | `add_request` 入 waiting 队尾，下一轮 prefill 扫描可见；本轮已定批次不受影响 | Day6/7 |
| prefill 最后 chunk 完成 | WAITING→RUNNING；本轮不再有该 seq 的 decode item；首 completion 即最后 chunk 采样值 | Day8 |
| decode item 完成 | FINISHED → `_finalize` 幂等清理；同轮其他 item 正常提交 | Day6/7 |
| chunk 中 cancel/timeout | 安全检查丢弃该 item 结果、不推进 offset；其余 item 不受影响 | Day8 §4.5 |
| KV 不足 | decode 侧：队尾抢占（preempt+resume，recompute）；prefill 侧：停止接纳记 kv_capacity；两原因独立 | Day6/7/8 |
| 抢占恢复 recompute | 恢复请求以 WAITING 进入 prefill 扫描，offset 按 prefix 命中重算 | Day8 |
| 空轮 idle | schedule 边界清理后批次为空：不调模型，idle 事件沿用（`prefill_items=0`） | Day7/8 |

## 5. 状态时序与不变量

### 5.1 混合轮时序示例

配置：B=8，max_num_seqs=4，chunk_size=8，block_size 足够。running 中有 A、B（decode），waiting 队首 C（prompt=20）。

```text
轮1  decode 阶段：A、B 各接纳 1（D=2，used=2）
     prefill 阶段：remaining=6；C 首候选拆分 q=min(20, 8, 6)=6
     items = [A(dec), B(dec), C(prefill 6)]   phase="mixed"，planned=2+6=8 ≤ B
     run：decode 子批 [A,B] 前向+采样 → 2 token；prefill 子批 [C[0,6)] 前向，中间 chunk 无采样
     postprocess：A、B 各追加 1 token；C offset 0→6，保持 WAITING
轮2  decode：A、B 接纳（D=2）；prefill：C q=min(14, 8, 6)=6 → offset 6→12
轮3  decode：D=2；prefill：C q=min(8, 8, 6)=6 → offset 12→18
轮4  decode：D=2；prefill：C q=min(2, 8, 6)=2 → offset 18→20，最后 chunk 接纳，
     C WAITING→RUNNING，采样首 completion
轮5+ C 加入 running，每轮 A、B、C 各 decode 1 token（纯 decode 轮）
```

C 的 chunk 区间 `[0,6) [6,12) [12,18) [18,20)` 连续、单调、无重叠，并集 `[0,20)`；A、B 的 decode 在轮 1–4 **从未中断**（对比 Day8：轮 1–4 整轮 prefill，A、B 记 `phase_priority` 停滞）。

### 5.2 不变量清单

继承 Day8 §5.3 全部 13 条（进度、区间、上限、对应、位置、attention、采样、资源、状态、预算延后零副作用、提交原子性、协议、日志），在此基础上新增/修订：

14. **混合预算**：每轮 `sum(prefill q_i) + count(decode items) <= B` 且 `len(items) <= max_num_seqs`，四条上限（正整数、q_i≤chunk_size、总量≤B、条数≤max_num_seqs）独立显式校验；decode item 恒 `q==1`。
15. **唯一性**：同一 seq 每轮至多一个 item；decode item 在前、prefill item 在后的顺序稳定；prefill 最后 chunk 接纳的请求本轮不得再以 decode item 出现。
16. **decode 保底**：decode 接纳先于 prefill 分配预算与名额；任何 prefill 接纳不得减少本轮已接纳 decode 的预算或名额。
17. **阶段-执行一致**：item.phase 与实际执行子批一一对应；`runner_input_len`（decode 条数 / prefill sum(q_i)）与计划可对账。
18. **归因正确**：`decode_priority` 仅当 `D>0 且 needed_first <= B`；该原因不计入预算等待 episode；`phase_priority` 全仓库无残留。
19. **round 关联**：`round_id` 唯一且单调；postprocess 校验 items 的 round_id 与调度轮一致，重复/迟到旧批次显式拒绝。
20. **同轮隔离**：单个 item 的终态/失败不影响同轮其他 item 的提交与记账；批次内 seq_id 去重。

## 6. 配置与接口设计

### 6.1 Config

无新增字段。`chunk_size`、`max_num_batched_tokens`、`max_num_seqs` 校验入口不变。

### 6.2 Sequence：具名谓词收敛（Day8 遗留）

```python
@property
def is_last_chunk_scheduled(self) -> bool:
    """本轮接纳即完成 prefill（offset + q == target）——Day9 起为该判定的单一权威。"""
    return self.prefill_offset + self.num_scheduled_tokens == self.prefill_target
```

scheduler（接纳判定/决策快照）、model_runner（`_select_prefill_sample_rows`）、llm_engine（调用前快照）、既有测试与验收脚本共 5 处重复判定全部改用该谓词；语义与现状逐位一致，仅消除漂移风险。

### 6.3 接口签名变更表

| 模块 | 现状 | Day9 |
| --- | --- | --- |
| `Scheduler.schedule()` | `-> (list[Sequence], bool)` | `-> (list[BatchItem], str)`，phase ∈ {prefill, decode, mixed, idle} |
| `Scheduler.postprocess()` | `(seqs, token_ids, is_prefill, now=None)` | `(items, token_ids, now=None)`；round 校验内置 |
| `ModelRunner.run()` | `(seqs, is_prefill)` | `(items)`；内部拆 decode/prefill 子批顺序执行 |
| `LLMEngine.step()` | `(outputs, num_tokens±)` | `(outputs, num_tokens>=0)`；吞吐口径改读结构化统计 |
| `RoundDecision` | 无 phase | 新增 `phase` 字段 |
| 原因常量 | 含 `REASON_PHASE_PRIORITY` | 删除；新增 `REASON_DECODE_PRIORITY` |
| `Config` / pickle v3 | — | 不变（`BatchItem`/`round_id` 仅 rank 0，不进 payload） |

既有测试适配面：`test_token_budget.py`、`test_kv_cache_lifecycle.py`、`test_chunked_prefill.py`、`test_request_lifecycle.py`、`test_block_manager.py`（间接）均消费 `schedule()` 返回值，需按新签名适配；**单阶段轮的行为断言（预算 8/8/4、chunk 边界、抢占、终态等）原样保持**，仅注入方式随新契约调整（Day8 适配方法学沿用）。

### 6.4 事件与日志字段

- `scheduler_round`：新增顶层 `phase`（四值）；`decisions[].phase`；原因白名单更新；`budget_deferred_*`、episode 事件不变。
- `engine_round`：§4.4 字段；idle/error 轮字段契约按"事件类型"穷举（Day8 教训：新增字段必须覆盖所有分支）。
- 验收脚本字段白名单版本化更新；无 prompt/token 明文；`observed_at` 全事件保留。

## 7. 实现步骤

1. **契约测试先行**：新增 `tests/test_mixed_batch.py` 骨架（SimpleNamespace Config 桩、固定时钟、token 注入助手），先定义 `BatchItem` 与混合预算公式契约。
2. **谓词收敛**：`Sequence.is_last_chunk_scheduled` 落地，替换 5 处重复判定，全量回归确认零语义变化。
3. **BatchItem 与 schedule() 重构**：decode-first 两阶段调度；`RoundDecision` 加 phase；归因重构（删 `phase_priority`、增 `decode_priority`）；`_finalize_round` 上限校验扩展为混合口径（含 seq_id 去重、decode q==1 校验）。
4. **ModelRunner 分组执行**：`run(items)` 拆子批；合并采样结果契约；`prepare_prefill/prepare_decode` 与 `Context`/`Attention` 零修改确认。
5. **postprocess 逐 item 提交**：签名变更、round 校验、按 item.phase 分支；安全检查与原子提交沿用。
6. **Engine 适配**：调用前快照扩展、计划显式校验、`step()` 返回口径、`generate()` 吞吐显示改读结构化统计、事件字段扩展（覆盖 idle/error 分支）。
7. **既有测试适配**：5 个测试文件按新签名调整注入方式；行为断言不变。
8. **新测试矩阵**：按 §9.1 完成 `test_mixed_batch.py` 全部类别。
9. **CPU 全量回归**：新测试 + 全量 + `python -O`。
10. **验收脚本**：扩展 `scripts/validate_chunked_prefill.py` 或新增 `scripts/validate_mixed_batch.py`（含 runner 包装 `runner_input_len` 观测与对账）。
11. **GPU 验证与 benchmark**：按 §9.4 执行，证据归档 `docs/evidence/day9/`。
12. **验收记录**：`docs/day9-validation.md`（实际命令/结果/未覆盖边界）、必要时 `docs/day9-review.md`；更新 `docs/README.md` 索引。

## 8. 风险与取舍

| 风险 | 说明 | 对策 |
| --- | --- | --- |
| 两次前向的固定开销 | 混合轮 decode+prefill 两次 kernel launch，相比单次 fused 前向多一次调度开销；小批次时占比更高 | 如实记录观测值；fused 方向留作后续演进（§2.2），不在本日以正确性风险换性能 |
| 预算碎片化伤害长请求 TTFT | decode-first 后 prefill 首候选拆分阈值变小（remaining 被 decode 占用），8K 请求完成轮数可能增加 | 属"可解释变化"的预期情形；benchmark 对照 one-shot 如实报告；不预设改善方向 |
| `decode_priority` 误判 | 判定条件写错会把纯预算不足记成优先级让路（污染吞吐归因） | 判定规则显式化（§3.5）+ 三分支单测（D>0 挤占 / needed>B / D==0） |
| 子批结果合并错位 | decode/prefill 采样结果合并顺序与 postprocess 消费顺序不一致会静默串 token | 合并顺序为显式契约（§4.1）；两端共享 `needs_sample` 谓词；token 数显式校验 + 注入助手钉死 |
| 迟到结果二次提交 | 同 seq 重新规划后旧批次 postprocess 以新计划提交（Day8 遗留） | round_id 校验 + `num_scheduled_tokens` 清零检测双保险（§4.3）；单测覆盖 |
| 接口变更波及面大 | `schedule()`/`postprocess()`/`run()` 签名同时变更，5 个测试文件适配 | 步骤 7 独立成步；单阶段行为断言原样保持作回归锚点 |
| CUDA Graph 兼容 | decode 子批在 `enforce_eager=False` 时走 graph 路径，本日不实测捕获正确性 | GPU 验证统一 `enforce_eager=True` 基线；如实记录未测 |
| TP 传输 | `BatchItem` 经 call 广播为新增 pickle 面 | 仅 rank 0 结构 + dataclass 默认 pickle；协议版本不变；TP>1 保持未测标注 |
| 同轮完成竞态语义 | 混合轮内一个 item 终态后其他 item 的提交路径 | §4.6 明确隔离规则；专测覆盖 |

## 9. 测试设计（先 CPU，后真实 GPU）

### 9.1 新增 `tests/test_mixed_batch.py`（无 GPU/模型依赖）

文件头注明覆盖范围、是否依赖 GPU、重复执行命令。沿用既有方法学：SimpleNamespace Config 桩、固定时钟注入、采样 token 直接注入、runner spy、轮次上限防挂。

| 类别 | 必须覆盖的场景与断言 |
| --- | --- |
| 混合表示 | items 含 decode+prefill；phase 四值（prefill/decode/mixed/idle）；decode 在前 prefill 在后；同 seq 唯一 item；最后 chunk 接纳的请求无同轮 decode item |
| 混合预算 | `sum(prefill q)+D <= B` 逐轮断言；D 先占预算（prefill 用剩余）；B 恰好/不足/B<max_num_seqs；`q_i<=chunk_size` 与总量两条独立；预置超预算计划在 Engine 校验处显式拒绝 |
| decode 保底与归因 | 长 prefill 存在时 decode 每轮推进；`decode_priority` 三分支（D>0 挤占 / needed>B 仍 budget / D==0）；`decode_priority` 不计 episode；HOL 跟随；`phase_priority` 无残留 |
| chunk 行为 | 8K 级 prompt 与 decode 同轮逐块推进；区间连续单调无重叠；首候选拆分按 remaining；扫描位置与队列分离不回归 |
| 采样对齐 | 中间 chunk 不采样不耗 RNG（generator 状态判据）；decode 与最后 chunk 同轮各 1 token；token 数不匹配显式抛错；注入助手与 `needs_sample` 共享谓词；RNG 顺序（decode 先）固定 |
| 执行契约 | runner spy 核对两个子批的成员与展平输入长度 == 计划；子批顺序（decode 先）；`Context` 每子批独立（无跨子批残留可由 spy 断言 prepare 调用参数） |
| KV 与资源 | decode may_append 与 prefill allocate 同轮不超卖；同轮完成释放幂等；free/used/ref_count 平衡；KV 不足时 decode 抢占与 prefill 停止归因独立 |
| 生命周期 | 混合轮中单 item cancel/timeout/外部终态：不推进、不影响其他 item；ID 复用隔离；抢占恢复 recompute 在混合轮正常 |
| 迟到与重复 | round_id 不匹配拒绝；同 seq 重新规划后旧批次拒绝；全终态重放安全空操作 |
| 日志证据 | 事件字段白名单、`observed_at`、无明文；`scheduler_round`/`engine_round` 按 round_id 关联可重算（phase 分布、分阶段 token 数、`runner_input_len` 对账） |
| 随机混合 | 固定 seed 交错 add/cancel/timeout/preempt/resume/旧结果；轮次上限；终态全平衡（队列/索引/ref/free-used）；无 `phase_priority` 残留 |

### 9.2 既有测试适配与全量回归

```bash
python -m pytest tests/test_mixed_batch.py -q
python -m pytest tests/test_chunked_prefill.py tests/test_token_budget.py tests/test_request_lifecycle.py -q
python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q
python -m pytest -q
python -O -m pytest tests/test_mixed_batch.py -q
python -O -m pytest -q
```

适配原则：单阶段轮的行为断言（Day7 预算 8/8/4、Day8 chunk 边界与 one-shot 一致性等）原样保持；仅 `schedule()`/`postprocess()` 的消费方式随新签名调整；文件头注明 Day9 适配说明。

### 9.3 CPU 验收脚本

新增（或扩展）`scripts/validate_mixed_batch.py`，CPU 模式独立重算：JSONL 逐行可解析、round_id 唯一单调、每轮混合预算 <= B、分阶段 token 数可由 decisions 重算、phase 分布正确、`decode_priority` 判定条件独立复算、无 `phase_priority` 残留、episode 秒数可重算、字段白名单。

### 9.4 GPU 验证与 benchmark

环境口径沿用 Day7/8（GPU 型号/显存/torch/transformers/模型快照/TP=1/`enforce_eager=True`/block_size/KV 池大小），KV 容量自检先行，不自动下载权重。

**主负载**（`plan.md` 指定）：`1 × 8192-token prompt + 32 × 128-token prompt`，greedy（temperature=0），固定 max_tokens，固定输入顺序。

| 组 | 配置 | 说明 |
| --- | --- | --- |
| one-shot 对照（"原始一次性 Prefill"） | chunk_size=8192（>= 8K prompt），B 充裕 | Day9 前的基线：长 prompt 单轮 prefill 期间 32 个短请求全部停滞，decode 零产出 |
| Day9 mixed | chunk_size 与 B 按可复现基线（如 1024 / 与对照一致） | decode-first 混合调度 |

记录与验收口径：

- **decode 持续产出**：长 prompt prefill 期间（前 N 轮）每轮 decode item 数 > 0 的事件证据；decode 最大连续停顿轮数 = 0（mixed 组）；one-shot 组预期 decode 停顿（对照组证据）。
- **短请求延迟**：P50/P95 TTFT（观测点：最后 chunk 完成轮 / 首 completion）；TPOT/ITL 在同步 API 下的观测口径显式说明（按轮时间戳近似，注明局限）。
- **整体**：wall time、总轮数、decode token 吞吐、peak allocated/reserved memory、完成状态、KV 收尾（free/used 平衡）。
- **正确性**：mixed 与 one-shot 的最终 token_ids 一致性对比（数值差异如实分析，沿用 Day8 greedy 对比规则，不偷改预期）。
- **独立校验**：每轮混合预算、分阶段 token 数、`runner_input_len` 对账（runner 包装注入，收 Day8 遗留）、`is_last_chunk` 恰好一次、offset 轨迹连续。
- 原始 JSONL 证据归档 `docs/evidence/day9/`；事件不含 prompt/token 明文；性能只报观测值。

### 9.5 明确不实测项（沿用如实记录分支）

TP>1 端到端、CUDA Graph 捕获路径、真实并发取消/超时。序列化协议、分组执行等 TP 相关语义由 CPU 单测覆盖。

## 10. 最终验收清单（实施时逐项核对）

> 本次任务仅交付设计；实施后再按实际证据逐项勾选，清单原文不变。
> 2026-09-12 实现审查后首次勾选（day9-review.md 第一轮）；同日修复 warmup 启动缺陷与
> `decode_priority` 归因偏差后复核更新：全部 15 项均有实证——CPU 303 passed（`python -O` 同过）、
> `scripts/validate_mixed_batch.py` CPU/GPU 双模式 PASS（GPU 长 prompt 分块期 decode 7/7 轮
> 持续产出、`runner_input_len` 对账收口）、greedy 一致性 26/33 差异分析与确定性复现归档
> （[day9-validation.md](./day9-validation.md)，证据 `docs/evidence/day9/`）。

### 功能与口径

- [x] 一轮 batch 可同时含 prefill chunk 与 decode item；`BatchItem` 逐请求标注 phase，纯阶段/ idle 语义与 Day8 一致。
- [x] 混合预算公式 `sum(prefill q_i) + count(decode) <= B`、`len(items) <= max_num_seqs`、`q_i <= chunk_size`、decode `q==1` 四条独立显式校验。
- [x] decode-first：decode 先占预算与名额；长 prefill 存在时已有 decode 每轮持续推进。
- [x] `phase_priority` 废除；`decode_priority` 判定条件显式且不误记 budget；episode 统计口径正确。→ 已勾：修复后按不变量 18 字面规则以 `needed_first` 判定（事件新增该字段供独立复算），审查反例场景（首候选 20>B=8 拆分后小候选延后）实测记 `budget`，新增 2 条反例单测 + 脚本复算闭合证据链（day9-validation §3/§5）。
- [x] 新请求动态加入、prefill 完成（WAITING→RUNNING）、decode 完成、chunk 中取消/超时、抢占恢复在混合轮内符合 Day6/7/8 不变量。

### 执行正确性

- [x] `ModelRunner.run(items)` 同轮分组执行：decode 子批先、prefill 子批后；`Context`/`Attention`/`ParallelLMHead` 零修改。
- [x] 采样契约：中间 chunk 不采样不耗 RNG；decode 与最后 chunk 各 1 token；合并顺序与消费顺序一致，数量不匹配显式抛错。
- [x] `postprocess` 逐 item 提交：round_id 校验、原子提交、失败不推进、同轮单 item 终态不影响其他 item。
- [x] `runner_input_len` 独立观测与计划对账（Day8 遗留收口）；`Sequence` 具名谓词收敛 5 处重复判定。
- [x] `Sequence` pickle v3 协议不变；`BatchItem`/`round_id` 不进 TP payload。

### 实测与交付

- [x] CPU 全量回归（新测试 + Day2–8 适配）与 `python -O` 通过，数字为实际执行结果。（均为 303 passed；三个验收脚本 CPU 模式 PASS）
- [x] 主负载 `1×8K + 32×128` GPU 实测：decode 持续产出有事件证据；短请求 P50/P95 TTFT、TPOT/ITL 与 one-shot 对照归档；变差也如实记录并解释。（mixed 短请求 TTFT P50 3.568s→0.198s、TPOT 39.1→43.5ms 略有折让有解释；decode 7/8 轮产出、7/7 全推进；day9-validation §6）
- [x] mixed 与 one-shot 最终输出一致性对比（差异须分析记录）。（26/33 逐 token 一致；7 条差异确定性复现，机制分析归档 day9-validation §6.1）
- [x] `docs/day9-validation.md` 记录 GPU/模型可用性、实际命令、结果与未覆盖边界；本设计加入 `docs/README.md` 索引；日志证据无 prompt/token 明文。（引擎事件经白名单校验无明文；greedy_compare 携带输出 token 供核对，沿用 Day8 同类证据先例）
- [x] TP>1 / CUDA Graph / 真实并发取消：实测或如实记录未测。（三项均如实记录未测，day9-validation §8）

## 11. 与 Day10–14 的衔接

- **Day10 取消/超时/抢占恢复**：混合轮的安全检查边界（§4.3）与 round 关联（§5.2 不变量 19）是异步取消/迟到结果处理的直接地基；抢占 victim 的选择（队尾）在 decode-first 下频率预期下降，策略本身留待 Day10 重新评估。
- **Day11+ 服务化**：`BatchItem` 的逐请求计划与分阶段 token 统计可直接映射 SSE 首 token 事件（最后 chunk 完成轮 = TTFT 观测点）与增量输出；`step()` 结构化统计取代符号口径后，事件循环可按 phase 精确计费。
- **Day14 Prometheus 指标**：`decode_priority/budget/sequence_cap/kv_capacity` 计数、decode 停顿轮数、分阶段吞吐已有原始事件，映射时不需再改库内口径。
- **fused mixed attention 演进**：§2.2 的显式非目标是后续性能优化的入口；届时需版本化 `Context` 契约、重捕获 CUDA Graph，并以本日的 `runner_input_len` 对账工具验证等价性。
