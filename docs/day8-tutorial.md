# Day 8 教程：Chunked Prefill——推理引擎如何把一个长 prompt 拆成多轮执行

> 读者假设：你已经跟着本项目走完 Day 1–7（跑通了 Qwen3 推理、理解了 Paged KV Cache、Prefill/Decode 调度、请求生命周期状态机和每轮 token budget），现在想搞清楚"一个 8K 的长 prompt 是怎么被拆成多轮执行的，以及为什么拆完之后输出和一次性 prefill 完全一样"。
> 配套代码：`nanovllm/engine/sequence.py`、`scheduler.py`、`block_manager.py`、`model_runner.py`、`llm_engine.py`；需求来源：[`chunked-prefill.md`](./chunked-prefill.md)；验收记录：[`day8-validation.md`](./day8-validation.md)；审查记录：[`day8-review.md`](./day8-review.md)；写法参考：[`day6-tutorial.md`](./day6-tutorial.md)、[`day7-tutorial.md`](./day7-tutorial.md)。
> 建议读法：先通读 §1–§3 建立"进度是一本账"的全局观，再对照代码精读 §4–§8，然后看 §10 的缺陷复盘（都是实现中真实抓到的），最后用 §11 的命令亲手跑一遍。

## 1. 先看全局：Day 7 的"隐式分块"为什么不够用

先回忆 Day 7 建立的执行模型。调度器每一轮有 token 预算 `B = max_num_batched_tokens`，prefill 队首候选的需求超过剩余预算时会被**拆分**——只接纳放得下的前缀。所以"分块"这件事在 Day 7 就已经发生了：

```text
prompt=20 token, B=8（chunk_size 还不存在）
轮1: 接纳 [0,8)   → 执行 → 记进度 8
轮2: 接纳 [8,16)  → 执行 → 记进度 16
轮3: 接纳 [16,20) → 执行 → prefill 完成，进 decode
```

那 Day 8 还做什么？设计文档 §1 指出这套"预算驱动的隐式分块"有三个本质局限：

1. **进度语义是隐式的、双重的**。进度记在 `num_cached_tokens` 一个字段里，它同时表示两件不同的事：prefix 命中了多少（`BlockManager.allocate` 写入）、已执行并写入 KV 多少（`postprocess` 累加）。两个语义共用一个字段，靠调用时序区分，没有任何校验。一旦出现部分执行、异常、抢占恢复，就没有独立的事实源可以核对。
2. **没有 per-chunk 控制面**。单轮 prefill 的 query 数只受预算 `B` 约束。想表达"每个请求每轮最多推进 256 个 token"这种**计算量**控制，没有入口。
3. **采样行为与分块互相耦合**。这是最隐蔽的一个：整个 prefill 批次的 logits 和采样参数的行序对应关系没有任何显式校验，而且如果中间 chunk 也走采样，`torch.multinomial` 会消耗 generator 的随机数状态——**chunk 拆得不同，随机采样流就不同**，"chunked 和 one-shot 输出一致"这条验收在随机采样下根本不可能成立。

Day 8 的全部工作，就是用三个显式概念把这三件事修掉。这也是真实 vLLM 的 chunked prefill 的核心思想：**把 prefill 进度从隐式字段升级为显式、单一事实源的 offset，把计算量上限从预算中独立出来，把采样收敛到"每个请求每轮至多一个 token"**。

### 先分清三个"大小"，别混

Day 8 里有三个名字都带"size/num"的量，层面完全不同，混用是常见的事故来源：

| 量 | 含义 | 层面 |
| --- | --- | --- |
| `chunk_size` | 单请求单轮 prefill 的 query 上限 | 调度**计算量**控制（Day 8 新增） |
| `max_num_batched_tokens`（B） | 全轮所有请求的 query 总上限 | 调度**预算**（Day 7） |
| `kvcache_block_size` | 一个物理 KV 块装多少 token | **显存分配**粒度（Day 3，默认 256） |

特别注意：`chunk_size` **不是显存上限**。物理 block 仍然按逻辑上下文一次性分配（首轮接纳时就把 prompt + max_tokens 需要的块全部分好），chunk_size 只限制每轮"算多少"。分块分的是计算，不是存储。

## 2. 三个核心概念：offset、chunk_size、complete

```python
# sequence.py —— Day 8 新增的进度语义

self.prefill_offset = 0          # 唯一进度事实源：已成功提交到 KV 的上下文 token 数

@property
def prefill_target(self) -> int:          # prefill 阶段的目标终点 = 当前 num_tokens
    return self.num_tokens

@property
def prefill_complete(self) -> bool:       # 派生只读标志，不可独立赋值
    return self.prefill_offset >= self.prefill_target

@property
def num_cached_tokens(self) -> int:       # Day 7 兼容只读视图
    return self.prefill_offset
```

三个设计决策值得停下来想：

**为什么 offset 是"唯一可写字段"，`num_cached_tokens` 降级成 property？** 双事实源漂移是多轮迭代项目最典型的腐烂方式：两条写路径各自演化，某天某个新代码路径只更新了其中一个，两本账就对不上了，而且对不上的时候没有任何报错。property 化之后，旧代码里所有 `seq.num_cached_tokens = x` 的写入会直接抛 `AttributeError`——**漂移从"静默错误"变成"当天就炸的显式错误"**。

**为什么 `prefill_complete` 在 decode 阶段反而是 False？** 因为 decode 每轮 `append_token` 使 `num_tokens` 增长，而 `prefill_offset` 只跟踪已提交 KV 的上下文（始终是 `num_tokens - 1`）。这不是 bug，是定义使然（`prefill_target` 就是"当前有效上下文长度"）。所以这个标志只在 prefill 阶段消费，调度器内部实际使用的是更精确的判定"本轮接纳是否恰好补完"：`prefill_offset + num_scheduled_tokens == prefill_target`。

**为什么空 prompt 要在构造入口显式拒绝？** 旧代码对空 prompt 会在 `token_ids[-1]` 处 IndexError，或者更糟——撑到调度/组装阶段才在别处崩溃。错误暴露得越晚越难查，原则是"非法输入在门口就拒绝"（§3.3 的入口校验同一思想）。

初始 offset 有三种来源，后两种是理解 prefix 和抢占恢复的钥匙：

| 来源 | 有效上下文 | 初始 `prefill_offset` |
| --- | --- | --- |
| 普通新请求 | prompt | 0 |
| prefix 命中的新请求 | prompt，但前部完整块可复用 | 命中完整块数 × block_size（`allocate` 写入） |
| 抢占恢复（recompute） | prompt + 已生成 token | 物理块已释放，offset 归零；恢复时重新做 prefix 查询重算 |

## 3. 一轮 chunk 的生命周期

把一次 chunk 从接纳到提交的全流程放在一起看（伪代码，真实代码见 §5/§7/§8）：

```text
schedule()    计算计划 q = min(prefill_target - offset, chunk_size, B - used)
              接纳：首次分配 block（一次性按逻辑上下文）→ num_scheduled_tokens = q
              若 offset + q == target：WAITING → RUNNING（最后一个 chunk 才迁移！）
run()         ModelRunner 按 [offset, offset+q) 组装输入并执行
postprocess() 安全检查（终态/取消/超时）→
              成功：hash_blocks(offset_before, offset_before+q) → offset += q → 计数清零
                    中间 chunk：丢弃 logits，不采样，保持 WAITING
                    最后 chunk：采样 1 个 token 追加，按 EOS/max_tokens 判定
              失败/取消/超时：不推进 offset（进度随 KV 一起在终态清理中作废）
```

用 Day7 时序的同款例子走一遍（prompt=20，chunk_size=8，B=8）：

```text
轮1  schedule: offset=0,  q=min(20,8,8)=8    首候选拆分；block 一次性分配 3 块
     postprocess: 写 KV [0,8)   → offset=8    中间 chunk：不采样，保持 WAITING
轮2  schedule: offset=8,  q=min(12,8,8)=8
     postprocess: 写 KV [8,16)  → offset=16   中间 chunk：保持 WAITING
轮3  schedule: offset=16, q=min(4,8,4)=4
     postprocess: 写 KV [16,20) → offset=20   最后 chunk：采样首 completion
     → prefill 完成；WAITING→RUNNING；之后进 decode 每轮 1 token
```

三个区间 `[0,8) [8,16) [16,20)` 连续、单调、无重叠、无遗漏，并集恰为 `[0,20)`。**区间连续性就是 chunked prefill 的核心不变量**，§11 的测试和 GPU 验收脚本都在独立重算它。

为什么中间 chunk 保持 WAITING 而不迁移到 RUNNING？因为 Day 6 状态机的迁移条件是"prefill 完成才算被调度接纳"——一个只提交了一半上下文的请求，被抢占或取消时的清理语义与等待中请求一致。状态机一条新规则都没加，chunk 进度完全建立在六态之上（这是 Day 6 把状态机和资源策略解耦的直接收益）。

## 4. Config：第三个独立旋钮

```python
# config.py
chunk_size: int = 1024   # Day8：单请求单轮 prefill query 上限

def __post_init__(self):
    ...
    validate_positive_int(self.chunk_size, "chunk_size")   # 复用 Day7 的显式校验入口
```

和 Day 7 一样走集中式校验（拒绝 0/负/float/str/bool，`python -O` 下仍生效），三个量之间**不要求**任何对齐或大小关系——`chunk_size` 可以大于 B（此时预算主导）、小于 B（此时 chunk 主导）、和块大小无关。`Scheduler` 直接构造路径（测试用 SimpleNamespace）缺省该字段时回退默认 1024。

## 5. Scheduler 精读：扫描位置与队列分离

Day 8 之前，prefill 接纳循环用 `self.waiting[0]` + `popleft()` 工作：请求要么被接纳完（弹出），要么原队首停下。分块之后多了一种中间态——**队首请求被接纳了一个中间 chunk，但它既没完成（不能弹出）也不能在本轮再被扫描（每请求每轮至多一个 chunk）**。

第一直觉是"接纳后把它移到队尾"，但那会破坏 FCFS 顺序。实现用的是**扫描位置与队列分离**：

```python
# scheduler.py —— prefill 接纳循环（节选）
scan = 0
while self.waiting and len(batch) < self.max_num_seqs and scan < len(self.waiting):
    seq = self.waiting[scan]
    ...
    q = min(needed, self.chunk_size, remaining)
    if q < needed and batch:
        # 非首候选放不下：整请求延后，停止扫描（FCFS 不跳过）
        self._attribute_prefill_stop(decisions, decided, reason=REASON_BUDGET, ...)
        break
    ...
    if offset_before + q == seq.prefill_target:
        seq.transition_to(SequenceStatus.RUNNING)
        self.waiting.remove(seq)      # 按对象移除：scan 不动，恰好指向后继
        self._enqueue_running(seq)
    else:
        scan += 1                     # 中间 chunk：原地保留，本轮不再看它
    batch.append(seq)
```

这个双机制值得画表 trace 一遍（waiting=[A,B,C]，A=20t、B=6t、C=5t，chunk=8）：

| 步骤 | scan | waiting | batch | 动作 |
| --- | --- | --- | --- | --- |
| 看 A | 0 | [A,B,C] | [] | A 拆分接纳 q=8（batch 空，允许 q<needed）；中间 chunk → scan=1 |
| 看 B | 1 | [A,B,C] | [A] | B 需要 6 ≤ min(8,余量)，整段接纳且是最后 chunk → remove(B) |
| 看 C | 1 | [A,C] | [A,B] | waiting[1] 现在是 C（B 已移除，后续左移）→ 接纳完成 |

关键在最后两行：**remove 把 scan 之后的元素整体左移一位，而 scan 不前进，下一轮迭代恰好落在被移除元素的后继上**。如果 remove 之后还 `scan += 1`，就会跳过一个请求——这类 off-by-one 会被 `assert list(sched.waiting) == [a]` 这样的队列内容断言抓住。

两个配套规则：

- **首候选才允许拆分**：`q < needed and batch` 里那个 `batch` 非空判断是全部关键。batch 非空说明本轮已经接纳过请求，当前候选是"后续候选"——后续候选必须整段放下，否则停止扫描（不绕过它去接纳更短的尾部，FCFS 纪律与 Day 7 一致）。
- **chunk_size 不产生新的等待原因**：部分推进的被拆候选记 `scheduled`（部分推进仍属被调度），不进预算等待统计。chunk_size 造成的"本轮只推进部分"是正常行为，不是资源不足。

决策快照也扩展了三个观测字段，供日志排查与验证脚本独立重算：

```python
RoundDecision(..., chunk_index=1, offset_before=0, is_last_chunk=False)
```

`chunk_index` 是 1-based 的"当前 prefill 阶段第 N 个 chunk"，由 Scheduler 侧 per-request 计数器维护；prefix 命中或抢占恢复开启新 prefill 阶段时从 1 重新计数，终态时随请求回收。

## 6. BlockManager 精读：hash_blocks 的显式区间

prefix cache 的登记函数从"内部推导"改成了"显式区间"：

```python
# block_manager.py
def hash_blocks(self, seq: Sequence, start: int, end: int):
    start_block = start // self.block_size
    end_block = end // self.block_size      # floor 除：未写满的尾块不进登记范围
    if start_block == end_block: return     # 空区间：安全空操作
    # 链式哈希从前一个已登记块延续
    h = self.blocks[seq.block_table[start_block - 1]].hash if start_block > 0 else -1
    for i in range(start_block, end_block):
        ...计算哈希、登记进 hash_to_block_id...
```

三个点用例子讲清楚（block_size=8）：

**只登记写满的块**。chunk 执行 `[0,12)`：`start_block=0`、`end_block=12//8=1`，只登记块 0。块 1 只写了 4 个 token，不登记——否则另一个请求会"命中"一个内容还没写完的块。

**跨 chunk 拼写满的块，由补完它的那个 chunk 登记**。chunk1 执行 `[0,12)`：`start_block=0`、`end_block=1`，登记块 0；块 1 只写了前 4 个 token，不登记。chunk2 执行 `[12,20)`：`start_block=1`、`end_block=2`，登记块 1——它的开头 4 个 token（位置 8–11）是 chunk1 写的。没关系：块的内容此刻已经完整，`seq.block(i)` 从 `token_ids`（prompt 本身）取内容算哈希，与"哪些 token 是哪轮写的"无关。

**链式哈希的前驱必然已登记**。区间起点 `start` 之前的内容要么来自 prefix 命中（命中块自带哈希），要么来自前序 chunk 的提交（提交它的那轮 hash_blocks 已登记）。唯一 `h=-1` 重启链的情况是 `start_block == 0`。这条不变量成立的前提正是 §3 的"区间连续、只成功提交"——失败不提交的 chunk 永远不会留下"写了一半但没登记"的中间态。

`deallocate` 同步改了一行：释放物理块时把 `prefill_offset` 归零。**进度跟着物理块走**：块没了，"已提交到 KV"这个事实就不成立了，恢复时重新按 prefix 查询重算（Day 6 的 recompute 方案不变）。

## 7. ModelRunner 精读：输入组装与采样契约

### 7.1 显式 offset 契约

组装函数拆成了 CPU 纯静态方法（`_build_prefill_inputs`），不碰 CUDA，可以被单元测试直接调用——这是 Day 8 测试方法学的关键改造（§11）：

```python
for seq in seqs:
    start = seq.prefill_offset          # 已提交 KV 的上下文长度
    seqlen_q = seq.num_scheduled_tokens # 本轮接纳的 chunk 大小
    end = start + seqlen_q
    # 显式范围校验：区间必须落在有效上下文内且非空
    if not (0 <= start < end <= seq.prefill_target):
        raise ValueError(...)
    input_ids.extend(seq[start:end])         # 跨 chunk 时是正确的后续片段
    positions.extend(range(start, end))      # 绝对位置，绝不从 0 重置！
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
    cu_seqlens_k.append(cu_seqlens_k[-1] + end)   # 历史有效 KV + 当前 query
```

跨 chunk 正确性由四个字段共同保证，缺一不可：

- `input_ids` 取 `seq[start:end]`——第二个 chunk 拿到的是 prompt 的**后续**片段；
- `positions` 是绝对位置 `range(start, end)`——RoPE 位置编码与一次性 prefill 逐位相同（如果从 0 重置，模型的全部位置语义就错了）；
- `cu_seqlens_k` 每段是 `end = offset + q`——每个当前 query 能看到**它的全部历史 KV**（前面 chunk 写进物理块的部分，通过 block_table 读取）加上当前 chunk 里更早的 query（causal）；
- `slot_mapping` 覆盖 `[start, end)` 的物理槽位，按 block_table 跨块切分——测试里用一份独立实现的期望值逐槽交叉验证。

末尾还有展平长度校验（input_ids/positions/slot_mapping/`cu_seqlens_q[-1]` 一一对应），全部用显式 raise——**组装契约的任何一环错位都当场爆炸，而不是产出错误注意力结果**。

### 7.2 采样：最后 chunk 才采样，中间 chunk 不碰 RNG

先说一个实现中真实踩到的坑（完整复盘见 §10）：设计文档假设 prefill logits 是 `[总query数, vocab]`，但实际代码里 `ParallelLMHead` 的 prefill 分支**已经**做了行聚合：

```python
# layers/embed_head.py —— 既有代码
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1     # 每个请求最后一个 query 的行号
    x = x[last_indices]
```

也就是说 logits 早就聚合为 `[len(seqs), vocab]`，第 i 行就是第 i 个请求的末 query。初次实现按文档假设在这个已聚合张量上再取扁平行号，GPU warmup（8192 token，首次跑到该规模）直接触发索引越界的 device-side assert。教训：**对"上游给我什么形状"的假设，必须用一行断言钉死**：

```python
if logits.shape[0] != len(seqs):
    raise ValueError("prefill logits 行数与请求数不一致...")
```

采样契约本身只有两条：

```python
sample_idx = self._select_prefill_sample_rows(seqs)   # 只有最后 chunk 的请求
if not sample_idx:
    return None            # 本轮全是中间 chunk：logits 直接丢弃，不采样
token_ids = self.sampler(
    logits[sample_idx], temperatures[sample_idx], top_ps[sample_idx],
    [generators[i] for i in sample_idx],
).tolist()
```

为什么中间 chunk 连采样都不采样（反正 greedy 不消耗 RNG）？因为 `torch.multinomial` 会**推进 generator 状态**。如果中间 chunk 也采样，两次随机采样请求的 generator 流就取决于 chunk 怎么拆——"chunked 与 one-shot 输出一致"在随机采样下不可达。统一成"最后 chunk 才采样"后，**RNG 流与 chunk 划分彻底无关**，greedy 和随机两种模式同时满足一致性。

### 7.3 decode 元数据修正（顺手修的既有隐患）

TP>1 时 decode 序列反序列化后 `token_ids` 是空的（payload 只保留 `last_token`），旧代码用 `len(seq)` 推 positions/context_lens 是碰巧正确。Day 8 改为直接用元数据：

```python
positions.append(seq.num_tokens - 1)     # 元数据，不依赖 token_ids 长度
context_lens.append(seq.num_tokens)
```

TP=1 下数值完全不变；TP>1 下修正了潜在错误。配套测试用"pickle 往返后的空 token_ids 对象"验证两种路径组装结果一致。

## 8. postprocess：原子提交

模型返回后的收尾是 Day 8 提交语义的核心，结构上分四步：

```python
# ① 需采样集合：在任何状态变更前按调度快照计算
needs_sample = [(not is_prefill)
                or (seq.prefill_offset + seq.num_scheduled_tokens == seq.prefill_target)
                for seq in seqs]
# ② token 数显式校验（run() 对全中间 chunk 轮返回 None，与空列表同价）
if has_live and len(samples) != sum(needs_sample):
    raise ValueError(...)        # 不使用会静默截断的 zip
# ③ 重复/迟到收尾拒绝：活动请求必须带着本轮的待执行计划
if not seq.is_terminal and seq.num_scheduled_tokens <= 0:
    raise ValueError("疑似重复或迟到的 postprocess")
# ④ 逐个请求：安全检查 → 原子提交
offset_before = seq.prefill_offset
self.block_manager.hash_blocks(seq, offset_before, offset_before + q)
seq.prefill_offset = offset_before + q
seq.num_scheduled_tokens = 0
if seq.prefill_offset > seq.prefill_target:
    raise ValueError(...)        # 推进后立即校验单调有界
if is_prefill and seq.prefill_offset < seq.prefill_target:
    continue                     # 中间 chunk：丢弃采样，保持 WAITING
seq.append_token(token_id)       # 最后 chunk（或 decode）才追加 token
```

三个容易忽略的细节：

**needs_sample 为什么在循环前算？** 它描述的是"调度时刻的计划"（offset + q == target），必须在任何状态变更前定格。如果边提交边判断，第一个请求提交后它自己的 offset 已变，后续请求的判定基础就被污染了。

**采样游标为什么与安全检查解耦？** `token_ids` 列表与 seqs 按位置对齐（run() 按同样顺序产出），某个请求因执行期间被取消而丢弃输出时，它的 token 在列表里**仍占一个位置**——游标照常前进，后续请求才能拿到自己的 token。把"列表对齐"和"是否采用"两件事分开，对齐关系才不会因丢弃而错位。

**为什么先整批校验再逐个提交？** 如果第 3 个请求的 token 数不对，前两个已经提交就形成了半提交状态。先检查全批（数量、计划快照），通过后才进入逐个提交循环——把"校验失败"挡在所有副作用之前（Day 6"先验证后落笔"原则在批次粒度的重演）。

## 9. 不变量清单（评审 checklist）

设计文档 §5.3 列了 13 条，Day 8 特有的几条建议当评审 checklist 用：

1. **进度**：`0 <= prefill_offset <= prefill_target` 恒成立；offset 只在 postprocess 成功路径推进，单调不减。
2. **区间**：同一请求的 chunk 区间连续无重叠；并集恰为有效上下文（验证脚本独立重算）。
3. **上限**：每轮 `q_i` 正整数、`q_i <= chunk_size`、`sum(q_i) <= B`、`len(batch) <= max_num_seqs`，四条独立校验、互不替代。
4. **位置**：query j 的绝对位置 == `offset + j`，跨 chunk 不重置。
5. **采样**：每请求每轮至多 1 个采样 token；中间 chunk 不采样、不消耗 RNG。
6. **提交原子性**：每 chunk 至多提交一次；取消/超时/异常不提交；迟到/重复 postprocess 被拒。
7. **资源**：block 首次接纳一次性分配，后续 chunk 不重复分配；释放后 offset 作废，不污染复用 ID。
8. **状态**：中间 chunk 保持 WAITING；WAITING→RUNNING 仅在最后 chunk 接纳时发生。

## 10. 缺陷复盘：实现与审查真实抓到的问题

| 问题 | 现象 | 根因模式 |
| --- | --- | --- |
| logits 形状假设 | 按"扁平 [T, vocab] logits"实现行选择，GPU warmup（首次 8192 token）触发 device-side assert | 对上游输出形状的假设没有断言钉死；文档假设与实际代码不符时，信代码不信文档，然后把差异写回文档 |
| idle 事件缺字段 | 空轮 `engine_round` 没有 `prefill_chunks`，而验收脚本白名单要求所有 engine_round 都含它——合法 idle 轮会被误判 FAIL（审查补强） | 新增字段只改了"正常路径"的事件构造，漏了同一事件的另一条分支；事件契约要按"事件类型"穷举，不按"代码路径" |
| 验收脚本文案 | prefix 场景打印"期望 5"，实际期望 3（审查修正） | 输出文案硬编码，期望值改了文案没跟——脚本输出也要随逻辑走，不手抄 |
| `runner_input_len` 未闭环 | 设计 §6.4 要求的脚本侧观测未实现且未声明偏差（审查发现，留待 Day9） | 交付清单核对按验收清单走，而验收清单没覆盖到的"接口设计"条款容易静默丢失 |
| 设计文档内部张力 | §3.4 mermaid 写"抢占后 offset 进度保留"，§3.2/§4.2 说"归零重算"——实现遵循后者（正确） | 同一事实在文档不同章节各说一遍，措辞漂移；需求文档的权威定义应该只有一处 |

抽象出来是三条工程原则，和 Day 6 的四条一脉相承：

1. **单一事实源**：进度只有一本账，旧字段降级为只读视图，写入点清零。
2. **假设钉死为断言**：形状、区间、长度、行数——所有"应该成立"的上游契约都显式校验，`python -O` 也不失效。
3. **纯函数拆分**：GPU 组装逻辑拆成无 CUDA 依赖的静态方法，毫秒级 CPU 测试就能覆盖跨块 slot、绝对位置这些最容易 off-by-one 的地方。

## 11. 测试：怎么跑、怎么读

```bash
python -m pytest tests/test_chunked_prefill.py -q        # 93 项（纯 CPU，毫秒级）
python -m pytest -q                                      # 全量 260 项
python -O -m pytest -q                                   # 验证保护不依赖 assert
PYTHONPATH=. python scripts/validate_chunked_prefill.py --mode cpu --output /tmp/day8.jsonl
```

测试组织上最有学习价值的三件事：

- **真实组装函数 + 独立期望实现**：输入组装测试直接调用真实的 `_build_prefill_inputs`，同时用一份独立写的 `_prefill_slot_expectation(seq, start, end, bs)`（按 `位置 → 块号 → 物理槽位` 三步推导）逐槽交叉验证。两份独立实现对不上，必有一错——这比"用被测代码的镜像逻辑自证"可靠得多。
- **提交契约的注入助手**：测试里 `tokens_for_batch(batch, is_prefill, token)` 按"需采样集合"构造注入列表，与 `postprocess` 的判定共享同一个谓词——中间 chunk 传 token 的错误注入会直接触发显式异常，契约两端都被测试钉死。
- **12 类场景矩阵**：配置校验、字段语义、chunk 边界（1/chunk±1/B±1/8K）、预算交互（chunk 与 B 的小/等/大）、调度规则、输入组装、prefix 与恢复、采样、生命周期、一致性（固定注入下 one-shot 与多 chunk 输出逐 token 相同）、TP 协议（v3 往返 + v1/v2 映射）、随机混合（固定 seed 交错 add/cancel/timeout/preempt/resume/旧结果重放，终态全平衡）。

## 12. GPU 验证怎么读（证据在 docs/evidence/day8/）

| 场景 | 看什么 | 结论 |
| --- | --- | --- |
| 8K 主场景 | 8192-token prompt 在 cs=256/512/1024 三档的轮数（33/17/9）与 offset 轨迹 | 轮数 = 8192/cs + 1 逐档吻合；轨迹连续覆盖 [0, 8192)；均无 OOM |
| greedy 一致性 | 6 个固定 prompt，one-shot vs chunked 的最终 token_ids | cs=16 时 5/6 一致；cs=256 时 6/6 |
| prefix 命中 | 共享 512-token 前缀的第二条请求 | 初始 offset=512，只执行 3 个新增 token |

其中 greedy 那个 1/6 的差异值得单独讲，它是理解**数值等价与位级等价**的好教材：该 prompt 在生成第 12 个 token 时模型处于复读循环，两个候选 token 的 logits 是近似平局。chunked prefill 把同一序列的 attention 拆到多次 varlen kernel 调用里执行，与一次性全长的浮点归约顺序不同，产生微小数值差——在平局处翻转了 argmax。证据链：同一位置同一分歧确定性复现（不是随机竞态）；chunk 加大后边界减少、差异消失。这属于设计 §4.6 预先声明的可接受类别（数学语义等价，浮点顺序可有差异），处理方式是**如实记录分析，不偷改预期**。

## 13. 思考题（衔接 Day 9–11）

1. Day 9 要在一轮里同时放 prefill chunk 和 decode token（mixed batch）。现在"整轮要么 prefill 要么 decode"的 `phase_priority` 归因、以及 postprocess 的 `is_prefill` 参数，各需要怎么改？（提示：批次从"单阶段列表"变成"逐请求阶段标注"。）
2. `chunk_size` 目前是全局常量。如果想让长请求用大 chunk、短请求用小 chunk（自适应），`Sequence` 和调度循环哪里要动？会不会破坏"每请求每轮至多一个 chunk"？
3. 中间 chunk 不采样保证了 RNG 流与划分无关。如果产品要求"每完成一个 chunk 就向客户端推送一次进度"，进度推送会破坏这个性质吗？（不会——推送不消耗采样 RNG，但注意别顺手把 logits 采样了。）
4. `prefill_complete` 在 decode 阶段恒为 False。如果要在指标里表达"这个请求的 prefill 是否曾完成"，应该加字段还是改定义？改定义会影响谁？
5. TTFT 的观测点是"最后 chunk 完成轮"。Day 11 做流式输出时，中间 chunk 能否作为更早的观测点向客户端返回部分结果？难点在哪？（难点：中间 chunk 的 logits 对应的 token 还不完整，不能作为输出 token，只能做进度信号。）
