# Day 7 教程：每轮 Token Budget——推理引擎如何保证"每一轮都不超额"

> 读者假设：你已经跟着本项目走完 Day 1–6（跑通了 Qwen3 推理、理解了 Paged KV Cache、Prefill/Decode 调度和请求生命周期状态机），现在想搞清楚"调度器怎么保证每一轮干活的量不超过上限，以及怎么向外界证明这件事"。
> 配套代码：`nanovllm/engine/scheduler.py`、`config.py`、`llm_engine.py`；需求来源：[`token-budget.md`](./token-budget.md)；验收记录：[`day7-validation.md`](./day7-validation.md)；写法参考：[`day6-tutorial.md`](./day6-tutorial.md)。
> 建议读法：先通读 §1–§3 建立预算的"账本观"，再对照代码精读 §4–§7，然后看 §8 的缺陷复盘（都是实现中真实抓到的），最后用 §9 的命令亲手跑一遍。

## 1. 先看全局：一轮调度在 Day 7 之后的完整图景

回忆 Day 6 建立的执行模型：同步批处理引擎，一切由 `LLMEngine.generate()` 的 `while` 循环驱动，每轮 `step()` 走三步——`schedule()` 决定跑谁、`model_runner.call("run")` 真正执行、`postprocess()` 记账。

Day 7 给这个循环加了一条纪律：**每一轮实际送进模型的输入 token 总量，不得超过一个上限 B（`max_num_batched_tokens`）**。而且要能拿出证据——不是"我保证没超"，而是每一轮都有可解析的日志，记录"计划了多少、实际执行了多少、谁因为预算被延后了、等了多久"。

```text
step()（Day 7 之后）
 ├─ scheduler.schedule()                 # 决定跑谁 + 扣预算 + 逐请求归因 → scheduler_round 事件
 │    ├─ 生命周期体检（Day6 沿用：终态兜底/取消/超时）
 │    ├─ prefill：按 waiting 队首依次尝试，首请求可分块，后续必须整装放下
 │    ├─ decode：先看钱包（预算/序列数上限），再谈 KV
 │    └─ _finalize_round：以批次实际字段重算校验 + 记录本轮日志
 ├─ 保存快照 (seq_id, request_id, n) + 调用前显式校验    # postprocess 会清零计数！
 ├─ model_runner.call("run")              # 真正执行
 │    └─ 异常 → engine_round(outcome=error, executed=null)，不虚报成功
 └─ scheduler.postprocess()               # Day6 记账 + 终态时结算预算等待 episode
```

### Day 7 之前的世界：一条半的纪律

`max_num_batched_tokens` 这个配置在 Day 7 之前就存在，调度器的 prefill 循环也确实按剩余预算给请求切分 token。但仔细看会发现它只有"一条半"纪律：

1. **prefill 有预算，decode 没有**。decode 循环只按 `max_num_seqs` 挑请求——若 B=2 而有 8 条 RUNNING，一轮就执行 8 个 decode token，上限名存实亡。
2. **半个**：prefill 的"预算"只是循环里的局部变量，没有任何校验和记录。你无法从日志证明某一轮没超限。
3. **没有归因**。请求这一轮没被选中，可能因为预算不够、KV 不够、序列数上限、或者本轮优先做 prefill——全部混在一起，无法回答"这个请求为什么在等"。
4. **没有等待统计**。因预算延后了多少请求、等了多少秒，一个数字都没有。而请求终态后会从索引删除，统计若不及时结算就永远丢失。

Day 7 的全部工作，就是把"每轮工作量上限"从约定变成**有账本、有归因、有证据的硬约束**。

## 2. 预算口径：先搞清楚"一个 token 的账"记的是什么

这是整个 Day 7 最容易想错的地方。预算 B 计的是**本轮送进模型的输入 query token 数量**，不是别的：

| 容易混淆的东西 | 为什么不算 |
| --- | --- |
| 请求的总上下文长度 | decode 一轮只喂 1 个 token，哪怕上下文已 8192 长 |
| 本轮输出 token 数 | prefill 512 token 只采样出 1 个输出，不能反过来说"只花了 1" |
| 物理块数 | 块有对齐和碎片；prefix 命中的块本轮不再执行、不占预算 |
| attention 读的历史长度 | 历史只在读 KV，不进 forward 的输入 |

三个核心符号贯穿全篇：

- **B** = `max_num_batched_tokens`，每轮上限；
- **U** = 本轮已承诺的输入 token（循环里叫 `used`）；
- **remaining** = B − U，钱包里还剩多少钱。

以及每个请求的两个数：

- **`needed_tokens`**：这个请求本轮**想**推进多少输入（新请求 = prompt 减去 prefix 命中；分块请求 = 总长减已完成进度；decode = 恒 1）；
- **`scheduled_tokens`**：实际**被接纳**的数（`num_scheduled_tokens`），非空批次必须为正。

一个判断："prefix 命中 256、未缓存 256"的请求该记多少？——256。缓存块本轮不执行、不占预算；如果按总长 512 记，预算就虚耗了。这正是为什么计费必须发生在 `BlockManager.can_allocate()`（只读查询 prefix 命中）**之后**，而不能提前按总长估算。

## 3. Config：一道"进门安检"

预算约束要成立，首先配置本身得是个正经的正整数。Day 7 之前的代码风格是 `assert`：

```python
# 旧风格：python -O 会把 assert 整个移除，保护失效
assert self.max_num_batched_tokens > 0
```

Day 7 换成显式校验，并处理一个 Python 的经典坑——**`bool` 是 `int` 的子类**：

```python
def validate_positive_int(value, field_name: str) -> int:
    if type(value) is not int:      # 必须用 type 精确匹配！
        raise ValueError(f"配置项 {field_name} 必须为正整数，实际为 {value!r}...")
    if value <= 0:
        raise ValueError(...)
    return value
```

为什么 `type(value) is not int` 而不是 `isinstance`？`isinstance(True, int)` 为真——`Config(max_num_batched_tokens=True)` 会静默变成 B=1。用 `type()` 精确匹配才能把 `True/False` 拦在门外。这类"看起来等价实际不等价"的细节，正是配置校验的常见事故源。

两个设计决策：

- **校验先于资源创建**：`Scheduler.__init__` 先调同一个校验函数，通过之后才 `BlockManager(...)`——"先验证后落笔"原则（Day6 不变量 7）在配置层的延续。
- **单一权威**：校验函数只有一份，`Config`（正常路径）和 Scheduler（测试用 SimpleNamespace 直接构造的路径）都调它。两套口径必然漂移，一份定义才能保证"测试里的 B 和线上的 B 是同一种东西"。

注意 B **不需要** ≥ max_num_seqs。B=8、max_num_seqs=16 是 decode 分批的正常配置——decode 每条只要 1 token，序列数上限和 token 上限是两个独立的约束，各自生效。

## 4. Scheduler prefill：排队买票，不插队、不找零

### 4.1 主循环：四个停止理由

```python
while self.waiting and len(batch) < self.max_num_seqs:   # 队列空 / 序列数满 → 停
    seq = self.waiting[0]
    remaining = budget - used
    if remaining == 0:            # ① 钱花完了
        self._attribute_prefill_stop(..., reason=REASON_BUDGET, needed=None, kv_checked=False)
        break
    num_cached_blocks, needed = self._estimate_prefill_tokens(seq)   # 只读查询，不分配
    if needed is None:            # ② KV 容量不足（can_allocate 返回 -1）
        self._attribute_prefill_stop(..., reason=REASON_KV_CAPACITY, needed=None, kv_checked=True)
        break
    if remaining < needed and batch:   # ③ 后续候选整装放不下
        self._attribute_prefill_stop(..., reason=REASON_BUDGET, needed=needed, ...)
        break
    # —— 通过全部检查，才分配 block、扣预算（接纳原子性）——
    kv_checked = not seq.block_table          # allocate 会填 block_table，先取证！
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(needed, remaining)
    used += seq.num_scheduled_tokens
    ...
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.transition_to(SequenceStatus.RUNNING)   # prefill 完成才转 RUNNING
        self.waiting.popleft()
        self._enqueue_running(seq)
    batch.append(seq)
```

对照设计文档 §4.3 的例子走一遍（B=8，需求 [3, 5, 2]）：

- 请求 3：needed=3 ≤ 8，接纳，U=3；
- 请求 5：needed=5 ≤ 5（remaining），接纳，U=8；
- 请求 2：remaining=0 → 停。记 **direct budget**，且 `needed_tokens=None`——预算在检查它之前就花完了，我们连它的需求都没算，更没查它的 KV，**不能虚构**。

再看 [6, 4, 1]：

- 请求 6：接纳，U=6；
- 请求 4：remaining=2 < needed=4 且批次非空 → 停。记 budget，`needed=4`（这次算出来了，如实记录），尾部请求 1 记 **head_of_line**（blocked_by 指向请求 4）。
- 关键纪律：**余量 2 宁可浪费也不给请求 1**。这就是 FCFS——不让"短请求"插队饿死长请求。日志里的 `budget_deferred_hol` 和 `blocked_by_seq_id` 就是为了把"我自己不够预算"和"被队首连累"区分开。

请求 3（needed=10 > B=8）呢？批次为空时 `min(needed, remaining)` 允许首请求分块——本轮跑 8 个，下一轮接着跑 2 个。这是上游已有的 chunked prefill 行为，**必须保留**：否则需求大于 B 的请求永远无法入队（饿死）。

### 4.2 停止归因：一个函数，两个角色

`_attribute_prefill_stop` 做的事：**首个未处理请求记直接原因，其后所有未处理请求记 head_of_line**。

```python
first = None
for seq in self.waiting:
    if seq.seq_id not in decided:      # 跳过本轮已有决策的请求（见 §8 的 bug 复盘）
        first = seq
        break
decisions.append(RoundDecision(first.seq_id, ..., reason, needed_tokens=needed, ...))
for seq in list(self.waiting):
    if seq.seq_id in decided:
        continue
    decisions.append(RoundDecision(..., REASON_HEAD_OF_LINE,
                    blocking_reason=reason, blocked_by_seq_id=first.seq_id))
```

为什么尾部记 HOL 而不是逐个检查？因为**我们根本没检查过它们**——FCFS 在队首就停了。若给尾部也记 budget，等于声称"它们每个都放不下"，这是没有依据的结论。日志里 `blocked_by_seq_id` 让验收程序能重算出"直接不足 vs 连带阻塞"两种人数，口径才经得起审计。

cap（序列数上限）与 budget 同时命中时统一记 sequence_cap——优先级规则必须固定，否则"因预算等待的人数"会随实现细节漂移。

### 4.3 estimate：只读，不分配

`_estimate_prefill_tokens` 只调用 `can_allocate`（只读的容量/命中查询），返回 `(缓存块数, needed 或 None)`。**查询和分配必须分离**：查询是"问价"，分配是"付钱"。若在估算时就分配了块，后面任何停止路径都得回滚——Day 7 的接纳原子性（延后零副作用，不变量 3）靠的就是"想清楚了才付钱"。

## 5. Scheduler decode：先看钱包，再动手

decode 的老代码是这样的：

```python
# 旧代码：先 popleft、先 may_append、再抢占腾资源，最后才发现预算没了
while self.running and len(scheduled) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        ...抢占队尾...
```

预算检查插在哪？设计文档 §4.2 的顺序是：**预算/序列数上限 → 队首取出 → KV 容量/抢占**。反过来的话，"因预算延后"的请求已经被 pop、被分配了块，甚至可能触发抢占——延后就有了资源副作用。

```python
while self.running:
    if len(batch) >= self.max_num_seqs:      # cap 先命中：其余记 sequence_cap
        for seq in self.running: decisions.append(... REASON_SEQUENCE_CAP)
        break
    if used >= budget:                        # 钱花完了：其余记 budget
        for seq in self.running:
            decisions.append(... REASON_BUDGET, needed_tokens=1)   # decode 需求恒为 1
        break
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):   # 钱够才轮到 KV 检查
        ...抢占队尾（KV 原因，不混记为 budget）...
    else:
        seq.num_scheduled_tokens = 1
        ...
```

两个值得注意的细节：

- **延后的请求根本没被 pop**——它原封不动留在 running 队列里。被选中的批次最后用 `extendleft(reversed(batch))` 放回队首，延后的保持相对顺序跟在后面。下一轮、下下轮，队首还是它们，直到前序请求完成。
- **KV 抢占和预算延后是两码事**。显存不够去抢占队尾（Day6 的 preempt/resume），归因记 `kv_capacity`；如果把资源不足也记成 budget，"预算等待人数"这个指标就废了。同样，本轮有 prefill 批次时 running 根本没成为候选，记 `phase_priority`——都不是预算的锅。

为什么 decode 延后可以逐条记 budget（prefill 的尾部却记 HOL）？因为 decode 候选的需求**恒为 1 且已知**——remaining=0 时每个 remaining 候选都确定放不下，不需要检查就有结论。prefill 的尾部请求需求各不相同且未查询，只能记 HOL。归因的精度取决于你"确实知道什么"。

## 6. 等待统计：episode 与三本累计

### 6.1 数据结构：只为欠账的人开账本

```python
@dataclass
class BudgetWaitStats:
    seq_id: int
    request_id: str
    budget_deferred_rounds: int = 0        # 处于预算等待原因的轮数（请求-轮次）
    budget_wait_started_at: float | None = None   # 当前开放 episode 的起点（单调时钟）
    budget_wait_seconds: float = 0.0       # 已结算的等待秒数
```

Scheduler 里只有三样东西：`budget_wait: dict[seq_id → BudgetWaitStats]`（只为**有过**预算等待的请求创建，终态后删除）和三个标量累计——唯一请求数、请求-轮次数、已结算秒数。不做全历史驻留：请求终态后发一条摘要事件，记录就删，内存不随历史增长。

### 6.2 episode 的生命周期：开启、不重开、结算

```python
def _update_budget_wait(self, seq, *, deferred, now, round_id, close_reason=None):
    if deferred:                       # 本轮观察原因为 budget/预算 HOL
        ...创建记录（首次：唯一请求数 +1）...
        stats.budget_deferred_rounds += 1
        if stats.budget_wait_started_at is None:   # 已有 episode 不重开！
            stats.budget_wait_started_at = now
        return
    ...本轮原因不再是 budget（被调度/切到 KV/phase/paused/终态）...
    duration = max(0.0, now - stats.budget_wait_started_at)
    stats.budget_wait_seconds += duration          # 结算
    self.budget_wait_closed_seconds_total += duration
    stats.budget_wait_started_at = None            # 关闭；重复调用是安全空操作
```

用一个可验算的时间线理解（也是测试断言的精确值）：

```text
t=10   C 因预算延后        → episode 开启（started=10）
t=12   C 再次延后          → rounds=2，episode 不重开（起点还是 10）
t=15   新 prefill 轮，C 变 phase_priority → 结算 15−10=5 秒，episode 关闭
t=20   C 又延后            → 新 episode 开启（rounds=3；unique 不再+1，还是 1 个请求）
t=23   cancel(C)           → 结算 23−20=3 秒；总计 rounds=3、closed=8 秒
```

四条语义规则，每条都有明确的"为什么"：

- **连续等待不重开**（t12 那次）：t10→t12 之间请求确实在等， episode 必须跨轮连续，否则等待时长被腰斩。
- **原因一变就结算**（t15）：一旦本轮观察到它不再因预算等待，立刻按"最后一次确认为 budget 的时刻到 now"结算。两轮之间发生的事无法观测（采样归因），不声称能拆出精确反事实。
- **显式抢占也结算**：抢占后等待原因是 paused/KV，把暂停时长算进"预算等待"是错误归因。
- **终态结算 + 删除**：cancel/timeout/自然完成都会走到 `_finalize` → 结算开放 episode → 发 `request_budget_wait` 摘要事件 → 删记录。重复 `_finalize` 有所有权保护（Day6 的 `owned` 检查），不会重复累计或误删新对象。

### 6.3 两种"人数"，别混着报

设计文档 §5.1 要求两个互不混淆的数字：

```text
budget_deferred_direct = 主原因就是 budget 的请求数
budget_deferred_hol    = 主因是 head_of_line 且 blocking_reason=budget 的请求数
budget_deferred_requests = 两者之和（本轮天然去重）
```

再加两个不同量纲的累计：`unique_requests`（出现过 episode 的唯一请求数——一条请求连等三轮只算 1）和 `request_rounds`（请求-轮次——连等三轮算 3，多个请求并行等待时可以大于墙钟时间）。汇报"因预算等了 8 秒"时，必须说清楚是"一个请求等了 8 秒"还是"四个请求各等了 2 秒"。

## 7. LLMEngine：快照、校验、证据链

### 7.1 为什么调用前必须拍快照

`num_scheduled_tokens` 是调度临时计数，`postprocess` 一开始就把它清零。如果 Engine 在模型返回**之后**才去统计"这轮执行了多少"，读到的是 0——这就是设计文档反复强调的"不能在其后据此倒推执行工作量"。所以 `step()` 在调用模型前先落快照：

```python
snapshot = [(seq.seq_id, seq.request_id, seq.num_scheduled_tokens) for seq in seqs]
planned = sum(n for _, _, n in snapshot)
```

这个快照还是**调用前校验**的依据：非空批次里任何非正或非整数的 `n`（`type(n) is not int`——又是 bool/float 伪造计数的防御）、或 planned 超过预算，直接抛带 round_id 的 ValueError——防御性检查放在不可逆动作（模型执行）之前，且不用 assert。

### 7.2 三种结局，三种记录，一条锁定

```python
try:
    token_ids = self.model_runner.call("run", seqs, is_prefill)
except Exception:
    self._execution_failed = True   # 失败锁定：禁止在未完成的 KV 状态上直接重试
    _log_event({..., "executed_tokens": None, "model_called": True,
                "outcome": "error", "observed_at": perf_counter()})
    raise                        # 记完日志原样上抛，不吞异常
executed = sum(n for _, _, n in snapshot)   # 从快照取，不是从已被清零的字段取
_log_event({..., "executed_tokens": executed, "outcome": "completed",
            "observed_at": perf_counter()})
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

- **completed**：`executed == planned <= B`。即使 postprocess 因取消/超时把输出丢弃，这轮的输入已经真实执行过，账不能销——"取消丢输出不抹去已执行输入"。
- **error**：模型抛异常，`executed_tokens=null`（工作量未知），绝不记成成功。异常后 Engine 进入失败锁定，下一次 `step()` 直接拒绝（"请销毁并重新创建 Engine"）——模型中途失败意味着 KV/进度账本的状态不可信，设计明确 Day7 不建 FAILED 状态、不做恢复，宁可显式锁死也不假装可以重试。
- **idle**：空批次不调模型，`model_called=False, executed=0`，但 round_id 照样分配、事件照样发——空轮也是轮，证据链不能断。

### 7.3 结构化日志的三个纪律

- **JSON Lines + 字段白名单**：每轮一条 `scheduler_round`（含逐请求 decisions），`engine_round` 按 round_id 与之关联，每个事件都带单调时钟 `observed_at`。验收程序能从原始事件独立重算所有计数，而不是只相信最终汇总——"解析器应能从 episode 起止时间独立重算等待秒数"。
- **库代码不做 basicConfig**：`logging.getLogger(__name__)` + 调用方决定开关。日志关闭时用 `isEnabledFor` 短路，连 JSON 字符串都不构造。
- **无内容泄漏**：事件只有 IDs、计数、原因，没有 prompt 文本和 token 内容。

## 8. 缺陷复盘：四个实现中真实抓到的 bug

写教程不等同于代码一次写对。这四个 bug 都是 Day 7 实现与测试过程中真实发生、真实修复的，每个都对应一种典型的思维漏洞：

| # | 现象 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | 首请求分块的轮次里，同一请求在 decisions 里出现两次：一条 scheduled、一条 budget | 分块后请求仍留在 waiting 队首，下一轮循环迭代再次考察它，`remaining==0` 分支把它当"未处理请求"归因 | 归因函数泛化为"首个**未处理**请求"（跳过已决策的），budget 停止归因到分块请求的下一个 |
| 2 | 所有 scheduled 决策的 `kv_checked` 恒为 False | `can_allocate` 的结果存在局部变量里，但 `kv_checked` 是在 `allocate()` 填充 block_table **之后**用 `not seq.block_table` 计算的——永远为 False | 在分配前捕获 `kv_checked`。教训：依赖"对象当前状态"的取证必须发生在改变状态的动作之前 |
| 3 | prefill 轮里刚被接纳的请求又多了一条 phase_priority 决策 | prefill 完成的请求已 `transition_to(RUNNING)` 并入 running 队列，`for seq in self.running` 发 phase_priority 时没排除它 | 用 `decided` 集合去重。同一条规则：一轮一请求一决策 |
| 4 | Engine 观测测试里 spy 记录的 `num_scheduled_tokens` 全是 0 | 测试的假 runner 保存的是**对象引用**，`step()` 返回后 postprocess 已把字段清零 | spy 在调用时刻拍值快照 `(seq_id, request_id, n)`。这正是"为什么 Engine 必须在调用前拍快照"的同一原因——测试先踩了坑，才真正理解了设计 |

共同模式：**"读取状态"和"改变状态"的时序**。归因、取证、快照都是在和状态变更抢时间，谁后到谁读到脏数据。Day6 的"安全边界先于不可逆记账"讲的是检查要早；Day7 的四个 bug 讲的是同一件事的反面——取证也要早。

## 9. 测试：怎么跑、怎么读

```bash
python -m pytest tests/test_token_budget.py -q      # Day7 68 项（纯 CPU）
python -m pytest -q                                  # 全量 167 项
python -O -m pytest tests/test_token_budget.py -q    # 验证保护不依赖 assert
python scripts/validate_token_budget.py --mode cpu --output /tmp/day7-cpu.jsonl
python scripts/validate_token_budget.py --mode gpu --model $MODEL_DIR \
    --token-budget 8 --max-num-seqs 16 --max-new-tokens 4 \
    --greedy-compare --output /tmp/day7-gpu.jsonl
```

继承 Day6 的三个测试纪律，并新增三个 Day7 特有的手法：

- **不热改预算**：要造"decode 候选数 > B"的场景，用 B=2 从初始化就固定，靠多轮 prefill 自然积累 N 条 RUNNING——绝不为了测试中途改 B 绕过真实行为。
- **确定性时钟**：`schedule(now=10.0)` 逐轮注入，§6.2 的时间线（rounds=3、seconds=8）就是用注入值精确断言的，无 sleep、无真实等待。
- **runner 桩 + spy 快照**：CPU 测试用 `SimpleNamespace(call=...)` 替代 ModelRunner；spy 记录调用时刻的 `(seq_id, request_id, n)` 快照（§8 bug 4 的教训），使"计划 == 实际执行输入"可以从 runner 入参独立验证，而不是拿同一个变量自证。
- **混合随机场景**：固定 seed 交错 add/cancel/timeout/preempt/resume/迟到重放，120 轮上限防挂死，最后断言身份、队列、引用计数、free/used 全部平衡——单点正确 ≠ 组合正确。
- **验收脚本即第二道关**：脚本不看代码、只读日志，独立重算所有口径（planned 求和、人数去重、episode 时长），任何一轮成功执行超预算即非零退出。日志证据和单测断言互相独立，才算真正的证据。

## 10. 不变量清单（评审 checklist）

1. **统一上限**：prefill/decode 同口径计费；非空批次每条 `scheduled_tokens > 0`，总和 ≤ B。
2. **两项约束独立**：token 预算与 max_num_seqs 各自生效、各自归因，cap 优先于 budget。
3. **延后零副作用**：仅因预算未选中的请求，不分配、不释放、不抢占、不追加 token，队列相对顺序稳定。
4. **状态机单一权威**：预算是调度决策，不是新状态；一切状态迁移仍走 Day6 的 `transition_to`。
5. **计划与执行区分**：调用前快照，executed 取自快照；取消丢输出不减账，异常不记成功。
6. **时间可复现**：单调时钟，episode 连续不重开、原因变更即结算、累计不为负。
7. **证据可重算**：验收程序从原始事件能独立算出每个数字，不信任汇总。
8. **内存有界**：只留本轮统计 + 活动记录 + 标量累计，终态即结算删除。

## 11. 思考题（衔接 Day 8–10）

1. Day 8 要引入显式的 chunk 控制和进度接口。现在"首请求分块"的隐式规则是 `min(needed, remaining)`，如果要支持"按 chunk_size 切、不同请求不同粒度"，`_estimate_prefill_tokens` 和 prefill 循环该改哪几行？归因和预算口径需要变吗？（口径不变——预算仍按实际 query token 记。）
2. Day 9 要做 mixed prefill/decode batch：一轮里既有 prefill 又有 decode。现在 `schedule()` 返回 `(list, bool)`，`phase_priority` 归因将失去意义——你觉得 batch 的表示该改成什么样？一轮的总预算公式会变成什么？（`sum(prefill tokens) + decode 条数`，见设计 §11。）
3. `budget_wait` 统计目前由 Scheduler（rank 0）独占。若未来统计要跨进程聚合（TP>1），`RoundDecision` 和 `BudgetWaitStats` 哪些字段该进 pickle、哪些该留在 rank 0？
4. 当前 decode 延后用"原地保留 + 批次回队首"实现 FCFS。如果同一请求连续 20 轮延后（队首永远选不中它），算不算饿死？结合 §6 的统计，你会用什么指标发现它？（提示：连续 budget_deferred_rounds 的分布。）
