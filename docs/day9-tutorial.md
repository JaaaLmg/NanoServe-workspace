# Day 9 教程：混合 Prefill 与 Decode——一轮调度如何同时推进长 prompt 和 decode 请求

> 读者假设：你已经跟着本项目走完 Day 1–8（跑通了 Qwen3 推理、理解了 Paged KV Cache、Prefill/Decode 调度、请求生命周期状态机、每轮 token budget 和 chunked prefill），现在想搞清楚"一个 8K 的长 prompt 在逐块推进时，为什么已有 decode 请求还能每轮产出 token，以及一批请求里 prefill 和 decode 是怎么共存的"。
> 配套代码：`nanovllm/engine/scheduler.py`、`model_runner.py`、`llm_engine.py`、`sequence.py`；需求来源：[`mixed-prefill-decode.md`](./mixed-prefill-decode.md)；验收记录：[`day9-validation.md`](./day9-validation.md)；审查记录：[`day9-review.md`](./day9-review.md)；写法参考：[`day6-tutorial.md`](./day6-tutorial.md)～[`day8-tutorial.md`](./day8-tutorial.md)。
> 建议读法：先通读 §1–§4 建立"批次是逐请求标注的 item 列表、decode 先占预算"的全局观，再对照代码精读 §5–§8，然后看 §10 的缺陷复盘（都是本轮真实抓到的），最后用 §11 的命令亲手跑一遍。

## 1. 先看全局：Day 8 的"整批单阶段"为什么不够用

先回忆 Day 7/8 建立的调度模型。每一轮调度是**整批单阶段**的：`schedule()` 返回 `(list[Sequence], bool)`——一个序列列表加一个全批布尔 `is_prefill`，整轮要么全是 prefill，要么全是 decode。只要 waiting 队首还有可接纳的 prefill，就整轮做 prefill，running 里的 decode 请求统一记 `phase_priority`（"本轮被另一阶段占用"）**整轮让路**：

```text
Day 8 的世界：waiting 队首有一个 8K prompt 在逐块推进（chunk_size=1024）
轮1: 整轮 prefill [0,1024)      A、B 两个 decode 请求：让路，零产出
轮2: 整轮 prefill [1024,2048)   A、B：继续让路
...
轮8: 整轮 prefill [7168,8192)   A、B：还在让路
轮9: 终于轮到 decode            A、B 各产出 1 个 token
```

8 轮里 decode 颗粒无收。更糟的是这个停顿**没有上界**：只要长 prompt 持续到达，decode 的 TPOT（每 token 生成时间）就跟着 prefill 排空速度走。Day 7 的 token budget 解决的是"单轮别算太多"，但 **prefill 优先**意味着预算每轮总是先被 prefill 吃满——预算限制了总量，没有改变分配的优先级。

`plan.md` Day 9 点名要修的就是这个：**优先保障已有 decode 请求，避免长 prefill 独占 GPU**。核心思路说破很简单：一轮预算里，decode 每条只要 1 个 token，先把 decode 的份额扣掉，剩下的再给 prefill——decode 的推进就从"看 prefill 脸色"变成"结构性保底"。

但"说破很简单"背后有三个必须严谨处理的问题，它们构成了 Day 9 的全部代码：

1. **批次怎么表示？** 一个列表加一个全局布尔装不下"同轮混合"——批次必须升级为**逐请求标注阶段**的结构（§3）。
2. **模型怎么执行？** `Context.is_prefill` 是全局单值，attention 有 varlen prefill 和 paged decode 两条互斥的 kernel 路径——不能一次前向同时喂两种请求（§2 的显式设计决策）。
3. **结果怎么对上账？** 采样结果从"整批一个顺序"变成"两个子批拼起来"，任何一端隐式重排都会静默串 token（§6/§7）。

## 2. 显式设计决策：不做单次混合前向，而是"一轮两次前向"

第一直觉是：既然一轮里既有 prefill 又有 decode，能不能把它们拼成一个张量做**一次**前向？设计文档 §2.2 把这条路显式排除了，原因值得理解——当前有三处以"整批单阶段"为前提的代码：

```python
# utils/context.py：Context.is_prefill 是全局单值，一次前向只能选一条 kernel 路径
# layers/attention.py：varlen prefill 与 paged decode 两条 kernel 二选一
# layers/embed_head.py：prefill 分支按 cu_seqlens_q 聚合每个请求的末 query 行，
#                       decode 分支直接对 [bs, vocab] 采样——两种 logits 形状互斥
```

把这三处全部改成支持真正的单次混合前向，需要合并 block table、混合 slot mapping、混合 logits 行映射，还要重捕获 CUDA Graph——每一步都在动**已经用 GPU 数值一致性验证过的 attention 语义**。Day 9 的选择是：**同一调度轮内按 phase 分组，先执行 decode 子批、再执行 prefill 子批**，两次前向各自完整复用现有契约，`Context`/`Attention`/`ParallelLMHead` 零修改。

代价是混合轮多一次 kernel launch（小批次时占比更高），收益是正确性风险为零。这是典型的工程取舍：**先用结构正确的方式把语义做对，性能优化（fused mixed attention）留作后续演进**——而且演进时要验证的等价性工具（`runner_input_len` 对账）恰好也是本轮交付的。

## 3. 核心概念：BatchItem——批次从"列表 + 布尔"到"逐请求标注"

```python
# scheduler.py —— Day 9 新增
@dataclass
class BatchItem:
    seq: Sequence            # 执行对象引用（TP 传输仍走 Sequence pickle v3，协议不变）
    phase: str               # "prefill" | "decode"
    scheduled_tokens: int    # prefill 为 q（0 < q <= chunk_size）；decode 恒为 1
    offset_before: int | None = None   # prefill：接纳时刻的 prefill_offset；decode 为 None
    is_last_chunk: bool = False        # prefill：接纳即完成 prefill；decode 恒为 False
    needs_sample: bool = False         # decode 恒 True；prefill 等于 is_last_chunk
    round_id: int = 0                  # 接纳时的调度轮次 ID
```

`schedule()` 的返回从 `(list[Sequence], bool)` 变成 `(list[BatchItem], str)`，第二个返回值是轮级标签：`"prefill"` / `"decode"` / `"mixed"` / `"idle"`。**纯阶段和空轮是混合的退化特例**——只有 decode item 就是 decode 轮，只有 prefill item 就是 prefill 轮，两者都有才是 mixed。Day 7/8 的全部行为在这个表示下原样保留。

逐个看四个字段为什么存在：

**`needs_sample`：调度时刻冻结的采样快照。** 这是整个混合轮采样对齐的单一权威。decode item 恒为 True；prefill item 等于 `is_last_chunk`（中间 chunk 不采样、不碰 RNG，Day 8 的不变量原样继承）。关键在于它在**调度接纳时**就定格，postprocess 不再从任何"当前状态"推导——序列状态在执行后是会变的，计划快照不会。

**`round_id`：迟到结果的双保险之一。** postprocess 收到批次时，先核对每个 item 携带的 `round_id` 是否等于最近一次调度轮——不等说明这是重复/迟到的旧批次结果，显式拒绝。原来的"活动请求必须带着待执行计划（`num_scheduled_tokens > 0`）"检测保留为二保险：前者在重新规划**之前**拦截，后者兜底。

**`offset_before` / `is_last_chunk`**：Day 8 的 chunk 观测字段原样搬进 item，使计划、执行、日志三端共享同一份快照。

**`BatchItem` 是纯 rank 0 结构**：不进 `Sequence.__getstate__` payload，TP pickle v3 协议一个字节都没动；TP>1 时经 `model_runner.call("run", items)` 广播，dataclass 默认 pickle，`seq` 字段走既有 Sequence 序列化。

items 的顺序也有契约：**decode item 在前、prefill item 在后**，同 phase 内保持各自队列的 FCFS 顺序。这不是随口约定——它同时是调度顺序、执行子批顺序和采样结果合并顺序（§6），三端共用一个顺序，才有"对齐"可谈。

## 4. 混合预算公式与 decode-first

每轮预算口径从"所有请求 query 之和"升级为：

```text
planned_tokens = sum(prefill item 的 q_i) + count(decode items)   # ≤ B
batch_size     = count(prefill items) + count(decode items)       # ≤ max_num_seqs
```

decode 每条贡献恰好 1。四条上限（q_i 正整数、prefill q_i ≤ chunk_size、总量 ≤ B、条数 ≤ max_num_seqs）在轮末 `_finalize_round` 独立显式校验，互不替代——Day 8 的纪律原样延伸到混合口径。

**decode-first 是结构性的，不是配置出来的**。没有"decode 保留预算比例"这种旋钮（设计文档 §2.2 明确不加 Config 字段），保底来自调度顺序本身：

```text
阶段 1  decode：对 running 队列按 FCFS 逐条考察，先扣预算与名额
阶段 2  prefill：remaining = B - used，用剩余量按 Day 8 规则接纳
```

用设计文档 §5.1 的时序例子走一遍（B=8，max_num_seqs=4，chunk_size=8；running 中有 A、B 两个 decode 源，waiting 队首 C 的 prompt=20）：

```text
轮1  decode 阶段：A、B 各接纳 1（used=2）
     prefill 阶段：remaining=6；C 是首候选，允许拆分 q=min(20, 8, 6)=6
     items = [A(dec), B(dec), C(prefill 6)]   phase="mixed"，planned=2+6=8 ≤ B
     执行：decode 子批 [A,B] 前向+采样 → 2 token；prefill 子批 [C[0,6)] 前向，中间 chunk 无采样
     提交：A、B 各追加 1 token；C offset 0→6，保持 WAITING
轮2  A、B 接纳；C q=min(14, 8, 6)=6 → offset 6→12
轮3  C q=min(8, 8, 6)=6  → offset 12→18
轮4  C q=min(2, 8, 6)=2  → offset 18→20，最后 chunk，C WAITING→RUNNING，采样首 completion
轮5+ C 加入 running，A、B、C 每轮各 decode 1 token（纯 decode 轮）
```

对比 §1 的 Day 8 时间线：C 的 chunk 区间 `[0,6) [6,12) [12,18) [18,20)` 同样连续单调无重叠，但 **A、B 的 decode 在轮 1–4 从未中断**。还有一个容易忽略的规则：**prefill 最后 chunk 接纳的请求本轮不再作为 decode item**——它的首个 completion token 就由最后 chunk 的采样产生，和 Day 8 语义一致（所以轮 4 的 items 里 C 只有一个 prefill item）。

KV 记账也是跨阶段统一的：decode 的 `may_append` 与 prefill 的 `allocate` 在同一轮内先后消耗同一个 free 池，`can_allocate`/`can_append` 的实时查询天然保证不超卖；不足时 decode 侧抢占（队尾牺牲）、prefill 侧停止接纳，两个原因独立归因。

## 5. Scheduler 精读：两阶段调度与归因重构

### 5.1 decode 阶段：先检查、后出队

```python
# scheduler.py —— _schedule_decode_phase（节选）
while self.running:
    if len(items) >= self.max_num_seqs:
        for seq in self.running:               # 序列数上限先阻止：其余候选记 sequence_cap
            decisions.append(RoundDecision(..., REASON_SEQUENCE_CAP, phase="decode"))
        break
    if used >= budget:
        for seq in self.running:               # 预算耗尽：decode 单位需求恒为 1（已知），
            decisions.append(RoundDecision(..., REASON_BUDGET, needed_tokens=1, phase="decode"))
        break
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        # 块不足：抢占队尾牺牲（preempt + resume 到 waiting 队首），记 kv_capacity
        ...
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        used += 1
        items.append(BatchItem(seq=seq, phase="decode", scheduled_tokens=1, ...,
                               needs_sample=True, round_id=round_id))
self.running.extendleft(reversed([it.seq for it in items]))
```

和 Day 7 同款的纪律：**上限检查先于 `popleft()`**——达标前不动队列、不补块、不抢占，被延后的请求原地保留 RUNNING、KV 与相对顺序。循环结束后把已接纳的批次恢复到 running 队首，延后请求跟在后面。

### 5.2 prefill 阶段：remaining 里做 Day 8 的老事

prefill 扫描与 Day 8 几乎逐行相同，只有两处输入变了：`remaining = budget - used`（used 已含 decode 占用），以及 `len(items) < max_num_seqs` 的名额检查现在数的是混合批次。首候选拆分 / 后续整段放下 / FCFS 不跳过 / 扫描位置与队列分离（中间 chunk `scan += 1`，最后 chunk `waiting.remove` 后 scan 不动）——这些 Day 8 教程 §5 讲过的机制原样工作，不再重复。

### 5.3 归因重构：`phase_priority` 之死与 `decode_priority` 的诞生

`phase_priority` 的语义是"本轮已被另一阶段**整体**占用"——mixed 轮下这句话不再成立，常量删除。新增 `decode_priority` 表达"预算被同轮 decode 优先占用"。它的判定条件是设计文档不变量 18 的字面规则：

```text
decode_priority 仅当 D > 0 且 needed_first <= B
```

`D` 是本轮 decode 实际占用量，`needed_first` 是**本轮首个被考察的 prefill 候选**的需求（不是被延后候选自己的需求！）。为什么必须是 `needed_first`？看一个反例（B=8，chunk=8，D=2，waiting=[P1(20t), P2(5t)]）：

```text
P1 是首候选，拆分接纳 q=min(20,8,6)=6 → remaining=0
P2 放不下被延后。它该记什么？
- 无 decode 的反事实：P1 会整占 8，P2 同样放不下 → 延后并非 decode 造成 → 应记 budget
- 若按 P2 自己的需求判定（5 ≤ 8 且 D>0）→ 会误记 decode_priority ✗
```

这就是设计文档 §8 风险表点名的"把纯预算不足记成优先级让路（污染吞吐归因）"。实现里 `needed_first` 在 prefill 扫描**第一次成功估算**时记录，两处归因点统一用它，并作为新字段写进 `scheduler_round` 事件供验收方独立复算。这个偏差在第一轮实现里真实存在（当时按"当前候选需求"判定），是审查抓出来后修复的——完整复盘见 §10。

配套口径：`decode_priority` **不计入预算等待 episode**（和当年的 `phase_priority` 一样：不是预算的锅，是优先级策略的让路），episode 的开启/结算只认 `budget` 与预算 HOL。

## 6. ModelRunner 精读：分组执行与合并采样契约

```python
# model_runner.py —— run(items)（节选）
decode_items = [it for it in items if it.phase == "decode"]
prefill_items = [it for it in items if it.phase == "prefill"]

if decode_items:                       # ---------- decode 子批（在前） ----------
    seqs = [it.seq for it in decode_items]
    input_ids, positions = self.prepare_decode(seqs)
    logits = self.run_model(input_ids, positions, False)   # 可走 CUDA Graph 路径
    if self.rank == 0:
        temperatures, top_ps, generators = self.prepare_sample(seqs)
        token_ids.extend(self.sampler(logits, temperatures, top_ps, generators).tolist())
    reset_context()

if prefill_items:                      # ---------- prefill 子批（在后） ----------
    seqs = [it.seq for it in prefill_items]
    input_ids, positions = self.prepare_prefill(seqs)
    logits = self.run_model(input_ids, positions, True)
    if self.rank == 0:
        if logits.shape[0] != len(seqs):
            raise ValueError(...)      # 行数与请求数一一对应（Day 8 契约沿用）
        sample_idx = [i for i, it in enumerate(prefill_items) if it.needs_sample]
        predicate_idx = self._select_prefill_sample_rows(seqs)
        if sample_idx != predicate_idx:
            raise ValueError(...)      # 调度快照与执行侧谓词交叉校验
        if sample_idx:
            token_ids.extend(self.sampler(logits[sample_idx], ...).tolist())
    reset_context()
return token_ids if token_ids else None
```

四个要点：

**两个子批各自完整走一遍"组装 → 前向 → 采样 → reset_context"**，互不残留 `Context` 状态。这就是 §2 设计决策的落地：`Context`/`Attention`/`ParallelLMHead` 一行没改，混合轮只是把它们的单阶段契约调用了两次。

**合并顺序是显式契约**。返回列表 = decode 子批采样结果 + prefill 最后 chunk 采样结果，顺序与 items 中 `needs_sample` 的出现顺序严格一致。postprocess 按同一顺序消费——两端共享同一份 `needs_sample` 快照，任何一端隐式重排都会串 token，所以顺序写进了 docstring 和测试。

**调度快照与执行侧谓词交叉校验**。`sample_idx` 来自 `BatchItem.needs_sample`（调度时冻结），`predicate_idx` 来自 `Sequence.is_last_chunk_scheduled`（Day 8 落地的具名谓词，Day 9 起是该判定的单一权威）。两者同源，不一致说明调度与执行之间请求状态被破坏——与其产出错位的采样，不如当场爆炸。

**无任何采样 item 时返回 None**（整轮全是中间 chunk），沿用 Day 8 契约；TP worker（rank≠0）只负责前向与 KV 写入，不消费 logits。

## 7. postprocess 精读：逐 item 提交与 round 关联

签名从 `(seqs, token_ids, is_prefill)` 变成 `(items, token_ids)`——全批布尔退休，阶段分支由 `item.phase` 决定：

```python
# scheduler.py —— postprocess（节选）
seqs = [it.seq for it in items]
has_live = any(not s.is_terminal for s in seqs)
# ① round 关联校验（双保险之一）：含活动请求的批次，round_id 必须等于最近一次调度轮
if has_live:
    last_round = (self.last_schedule_stats or {}).get("round_id")
    if last_round is None or any(it.round_id != last_round for it in items):
        raise ValueError("postprocess 批次与最近调度轮不匹配（疑似重复或迟到的旧批次）")
needs_sample = [it.needs_sample for it in items]      # ② 调度快照，逐 item 对齐
if has_live and len(samples) != sum(needs_sample):
    raise ValueError(...)                              # ③ token 数显式校验，不静默截断
for it, need in zip(items, needs_sample):
    token_id = samples[ti] if need else None           # ④ 游标与安全检查解耦（Day 8 原则）
    if need: ti += 1
    seq = it.seq
    if seq.is_terminal:            self._finalize(seq, now); continue   # 终态兜底
    if seq.cancel_requested:       mark_cancelled → _finalize; continue # 取消优先
    if seq.deadline 跨过:          mark_timeout → _finalize; continue   # 超时第三
    # ---------- 原子提交（prefill/decode 统一） ----------
    self.block_manager.hash_blocks(seq, offset, offset + q)
    seq.prefill_offset = offset + q; seq.num_scheduled_tokens = 0
    if it.phase == "prefill" and seq.prefill_offset < seq.prefill_target:
        continue                     # 中间 chunk：丢弃 token，保持 WAITING
    seq.append_token(token_id)       # 最后 chunk / decode：追加并判 EOS/max_tokens
```

三个混合轮特有的点：

**hash_blocks 对两种 item 统一**。`[offset_before, offset_before + q)` 这个显式区间对 prefill chunk 和 decode 是同一个公式——decode 每轮区间恰好是 `[len-1, len)`，与 Day 7 的逐 token 登记逐位一致。一段代码伺候两种阶段，正是"逐 item 统一提交"的收益。

**同轮隔离**。批次里某个 item 执行期间被取消/超时/外部置终态：它自己走安全检查被清理，**不推进 offset，也不影响其他 item 的提交**——游标照常前进占位，其余 item 正常落账。专测覆盖"同轮一个 decode 完成或取消、其余 item 正常落账"。

**round 校验为什么对比的是 `last_schedule_stats` 的 round_id**：因为 `schedule()` 每轮（包括空轮）都会刷新这份统计。若 postprocess 与下一次 schedule 之间插了一个 idle 轮，旧批次的 `round_id` 就对不上最新的轮次——迟到结果依然会被拦住。全终态批次豁免该检查：其请求已被安全路径幂等清理，重放是安全空操作（Day 6/7 所有权测试依赖的契约）。

## 8. Engine 精读：step() 口径与快照

`step()` 的第二个返回值 `num_tokens` 语义变了：Day 7/8 用**符号**编码阶段（prefill 为正、decode 为负），混合轮里"一个数既当总量又当符号"装不下了，改为**恒非负的本轮总 query token 数**。分阶段工作量改由事件的结构化字段承担：

```python
# llm_engine.py —— 调用前快照（NamedTuple 替代裸元组，按名访问抗字段增删）
class _ItemSnapshot(NamedTuple):
    seq_id: int; request_id: str; phase: str
    scheduled_tokens: int; offset_before: int | None
    is_last_chunk: bool; round_id: int

snapshot = [_ItemSnapshot(...) for it in items]
planned = prefill_tokens + decode_count      # 混合口径
num_tokens = planned                          # 恒非负，不再用符号编码阶段
```

`generate()` 的吞吐显示同步改读 `scheduler.last_schedule_stats` 的 `prefill_tokens` / `decode_tokens`——符号口径退役。`engine_round` 事件增加 `phase`（四值）、`prefill_tokens`/`decode_tokens`、`prefill_items`/`decode_items`（`prefill_chunks` 保留为同值别名平滑迁移）；idle 和 error 分支按"事件类型"穷举补齐全部字段（Day 8 的教训）。调用前校验也升级为混合口径：正整数、phase 合法、decode 恒为 1、seq_id 去重、`planned <= B`，全部显式 raise，`python -O` 下不失效。

## 9. 不变量清单（评审 checklist）

设计文档 §5.2 在 Day 8 的 13 条之上新增 14–20 条，混合轮特有的几条当评审 checklist 用：

1. **混合预算**：每轮 `sum(prefill q_i) + count(decode) <= B` 且 `len(items) <= max_num_seqs`；四条上限独立校验；decode item 恒 `q==1`。
2. **唯一性**：同一 seq 每轮至多一个 item；decode 在前 prefill 在后；最后 chunk 接纳的请求本轮不得再以 decode 出现。
3. **decode 保底**：decode 先于 prefill 分配预算与名额；任何 prefill 接纳不得减少已接纳 decode 的份额。
4. **阶段-执行一致**：`item.phase` 与实际执行子批一一对应；`runner_input_len`（decode 条数 / prefill `sum(q_i)`）与计划可对账。
5. **归因正确**：`decode_priority` 仅当 `D>0 且 needed_first <= B`；不计入预算 episode；`phase_priority` 全仓库无残留。
6. **round 关联**：`round_id` 唯一单调；postprocess 校验不通过即拒绝迟到/重复批次。
7. **同轮隔离**：单个 item 的终态/失败不影响同轮其他 item 的提交与记账。

## 10. 缺陷复盘：实现与审查真实抓到的问题

| 问题 | 现象 | 根因模式 |
| --- | --- | --- |
| warmup 旧签名（P0） | `run()` 签名改为 `run(items)` 后，`warmup_model` 仍调 `run(seqs, True)`，GPU 引擎构造即 `TypeError`；301 项 CPU 测试全绿却完全没暴露——桩 runner 绕过了真实 ModelRunner | 接口签名变更时按"调用点清点"排查，而测试桩恰好把最真实的调用点（warmup）排除在覆盖外；**签名变更必须 grep 全仓库调用点，桩测试绿 ≠ 真实路径通** |
| `decode_priority` 归因偏差 | 初版按"被延后候选自己的需求 ≤ B"判定；碎片化场景（首候选拆分后小候选延后）会把纯预算竞争记成优先级让路，污染归因 | 反事实推断（"无 decode 时它本可接纳吗"）必须看**整轮的第一个候选**，不是当前候选；两口径各有一个更准的反例方向，设计文档的字面规则是唯一权威 |
| 相同 prompt 被前缀折叠 | CPU 主负载 32 条短请求最初用同一 prompt，prefix cache 把后 31 条折叠成 8-token prefill，"32×128 负载"名不副实 | prefix 命中会改变负载的真实计算量；benchmark 负载构造要绕开缓存效应（改用互不相同的 prompt） |
| 签名变更的连带适配面 | `schedule()/postprocess()/run()` 三处签名同时变更，5 个测试文件 + 2 个验收脚本需适配 | 设计文档 §6.3 预先列出变更表，适配独立成步；单阶段轮的行为断言原样保留作回归锚点 |

抽象出来是三条工程原则，和 Day 6–8 的一脉相承：

1. **权威下沉到数据结构**：批次阶段、采样计划、轮次归属全部冻结在 `BatchItem` 里，下游（执行、提交、日志）只读快照，不做二次推导。
2. **一条顺序契约管三端**：items 的 decode-在前顺序同时是调度、执行、合并消费的顺序；顺序一旦成为契约，就由测试和脚本两头钉死。
3. **反事实归因要显式规则化**：`decode_priority` 的判定写成可检验的公式并进事件，验收方独立复算——"为什么延后"这类语义问题靠自觉迟早漂移。

## 11. 测试：怎么跑、怎么读

```bash
python -m pytest tests/test_mixed_batch.py -q           # 42 项（纯 CPU，毫秒级）
python -m pytest -q                                     # 全量 303 项
python -O -m pytest -q                                  # 验证保护不依赖 assert
PYTHONPATH=. python scripts/validate_mixed_batch.py --output /tmp/day9.jsonl
```

测试组织上最有学习价值的三件事：

- **注入助手与提交端共享同一谓词**：`tokens_for(items, token)` 按 `needs_sample` 构造注入列表——中间 chunk 的错误注入会直接触发显式异常，契约两端都被钉死（Day 8 `tokens_for_batch` 的逐 item 版）。
- **runner spy 钉死子批契约**：spy runner 记录每次收到的 items，断言 decode 子批在前、成员与展平输入长度（decode 条数 / prefill `sum(q_i)`）等于调度计划——这正是"合并顺序"契约的 CPU 侧证据（GPU 侧由脚本包装 `prepare_*` 独立观测对账）。
- **11 类场景矩阵 + 随机混合**：混合表示、混合预算、decode 保底与三分支归因、chunk 行为（8K 级 prompt 与 decode 同轮逐块推进）、采样对齐（RNG 状态判据）、执行契约、KV 不超卖、生命周期（单 item cancel/timeout 不影响其他 item）、迟到与重复、日志证据、固定 seed 随机混合（终态全平衡）。

## 12. GPU 验证怎么读（证据在 docs/evidence/day9/）

主负载是 `plan.md` 指定的 `1 × 8192-token prompt + 32 × 128-token prompt`（greedy，max_tokens=16），对照组 one-shot（chunk_size=B=8192，即"原始一次性 Prefill"），实验组 mixed（chunk_size=1024、B=2048）：

| 指标 | one-shot | Day9 mixed |
| --- | --- | --- |
| 轮数 / wall time | 17 / 4.15 s | 23 / **0.98 s** |
| 短请求 TTFT P50 / P95 | 3.568 s / 3.568 s | **0.198 s / 0.352 s** |
| 短请求 TPOT P50 | 39.1 ms | 43.5 ms |
| 长 prompt 分块轮 decode 产出 | 0/1 | 7/8（首轮无 decode 源） |

三个数字背后的机制值得嚼透：

**短请求 TTFT 降约 18 倍**：one-shot 下 32 条短请求必须等 8192 整批 prefill 轮完成才被接纳；mixed 下它们从第 1 轮起就与长 prompt 分块交错接纳并立即产出首 token（decode 产出逐轮爬坡 `[0, 8, 15, 22, 29, 32, 32, 32]`）。这正是 §1 那个"decode 停滞"问题的镜像收益。

**TPOT 略有折让（39.1 → 43.5 ms）**：混合轮的 decode 与 1024-token prefill 子批共享调度轮，decode 轮时长被 prefill 子批拉长——这是 decode-first 的预期代价，方向和幅度都有解释。设计文档 §2.2 明确"不承诺改善方向，如实记录"。

**greedy 一致性 26/33**：7 条请求的输出有分歧，但完整重跑后差异请求集合、首个分歧位置、分歧两侧 token **逐位复现**——是确定性数值差异，不是竞态。机制与 Day 8 §6 完全同类：varlen attention kernel 的批次形状不同 → 浮点归约顺序不同 → 近似平局处翻转 greedy argmax。CPU 桩注入实验证明固定注入下两条路径输出完全一致，调度/进度语义无错。处理方式依然是**如实记录分析，不偷改预期**。

另外两条横切证据：`runner_input_len`（Day 8 遗留）由脚本侧包装 `prepare_decode/prepare_prefill` 捕获每个子批实际组装的展平输入长度，与调度计划逐轮对账（17/17、23/23 轮全对）；TP>1、CUDA Graph 捕获、真实并发取消按设计 §9.5 如实记录未测。

## 13. 思考题（衔接 Day 10–14）

1. `postprocess` 的 round_id 校验对比的是 `last_schedule_stats` 的 round_id。Day 10 要做异步取消：如果 HTTP 层在模型执行期间调用 `cancel_request()`，这个校验会拦住什么、拦不住什么？（提示：取消标记在安全检查路径处理，不需要绕过 round 校验。）
2. decode-first 下，running 队尾的 decode 请求如果一直排在预算之外会被无限延后（继承 Day 7 的无轮转策略）。Day 10 如果要加"最长让路轮数"保护，应该改 `_schedule_decode_phase` 的哪一步？会不会破坏"延后请求保持相对顺序"？
3. §2 的两次前向有固定开销。如果把 decode 子批和 prefill 子批合并成单次 fused mixed attention，`Context`/`Attention`/`ParallelLMHead` 各需要什么新契约？`runner_input_len` 对账工具怎么用来验证等价性？
4. `step()` 的 `num_tokens` 不再用符号编码阶段。假如有个下游脚本还在按 `num_tokens < 0` 判断 decode 轮，它会怎么错？这类"接口语义演进"在变更表（设计 §6.3）之外还需要什么配套？（提示：grep 调用点 + 旧口径的过渡别名策略——本轮 `prefill_chunks` 保留别名就是同款处理。）
5. Day 11 的 SSE 首 token 事件需要"这个请求本轮产出了第一个 completion"。`BatchItem` 的哪个字段恰好就是判定依据？为什么它必须在调度时刻冻结而不能在 postprocess 时现算？
