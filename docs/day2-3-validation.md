# Day 2–3 验收记录

## 1. 验收范围

本记录对应 `plan.md` 的 Day 2（端到端调用链）与 Day 3（Paged KV Cache）：

- Day 2 产出：`docs/architecture.md`（模块职责表、请求路径时序图、单轮调度流程图、关键数据结构、四个关键问题的口头解释）。
- Day 3 产出：`docs/kv-cache.md`（三层映射关系、生命周期图、五条关键路径、碎片分析）+ `tests/` 下两个单元测试文件。
- 不包含：prefill/decode 与采样的对比实验（Day 4）、benchmark（Day 5）、服务化（Day 11+）。

- 验收日期：2026-09-07
- 基准 commit：`78474b6`（文档与测试为该 commit 之上的新增文件）
- 代码阅读范围：`nanovllm/engine/`（llm_engine、scheduler、sequence、block_manager、model_runner）、`nanovllm/layers/attention.py`、`nanovllm/utils/context.py`、`nanovllm/config.py`、`nanovllm/sampling_params.py`，共约 900 行核心代码

## 2. Day 2 验收：架构与调用链

产出文档：[architecture.md](./architecture.md)，包含：

| 验收项 | 状态 | 证据 |
| --- | --- | --- |
| 模块职责和调用时序图 | 通过 | architecture.md 第 1–4 节：组件关系图（mermaid）、模块职责表、`generate` 完整时序图（mermaid）、`schedule()` 分支流程图（mermaid），关键节点均附 `文件:行号` |
| 能用自己的话解释：谁创建请求 | 通过 | architecture.md 第 6 节问题 1：`LLMEngine.add_request` tokenize 后构造 `Sequence` 入 `waiting` 队尾 |
| 谁分配 KV block | 通过 | 第 6 节问题 2：Scheduler 经 BlockManager 分配（prefill 整段 `allocate`、decode `may_append` 补块、不足时抢占释放） |
| 谁执行模型 | 通过 | 第 6 节问题 3：ModelRunner 组张量并前向，Attention 层 Triton kernel 写 KV、FA2 读块，Sampler 采样；TP>1 经共享内存广播 |
| 谁决定下一轮 | 通过 | 第 6 节问题 4：Engine 的 `while not is_finished()` 决定是否继续，Scheduler.schedule 决定每轮内容（prefill 优先） |

阅读中额外记录的设计取舍与隐患（prefill 饿死 decode、抢占即全量重算、无 greedy 采样、超长 prompt 触发 assert 崩溃）见 architecture.md 第 7 节，作为 Day 7–10 改进的输入。

## 3. Day 3 验收：Paged KV Cache

### 3.1 文档产出

| 验收项 | 状态 | 证据 |
| --- | --- | --- |
| KV Cache 生命周期图 | 通过 | kv-cache.md 第 4 节（Sequence/块双视角状态图）+ 第 3 节（物理块状态组合表） |
| 逻辑 token / 逻辑 block / 物理 KV block 映射关系 | 通过 | kv-cache.md 第 1 节三层映射图与三条映射规则 |
| block 的申请、写入、复用、释放、preemption 路径 | 通过 | kv-cache.md 第 5 节五条路径，逐条附代码行号 |
| 能解释为什么分页 KV Cache 能减少碎片 | 通过 | kv-cache.md 第 7 节：内部碎片/外部碎片/无共享三点对比 + 本机实测块大小（28 MiB/块）的量化估算 |

### 3.2 单元测试

测试环境：本机 Anaconda Python 3.12.3，pytest 9.1.1（本次验收前安装）；测试不依赖 GPU 与模型权重，`BlockManager` 与 `Scheduler.schedule/postprocess` 为纯 CPU 逻辑，采样 token 由测试注入。

```bash
python -m pytest tests/ -v
```

结果：**17 passed（5.30s），0 failed**，覆盖点：

- block 不足时的返回值：`can_allocate` 返回 -1（空闲不足、被占用导致不足两种场景）；`can_append` 在 decode 需新块而池空时返回 False；
- 释放后可复用性：`deallocate` 后块回到空闲池可被更大请求重新分配；完成序列的前缀哈希保留，新请求复用同一物理块且只需 prefill 增量 token；
- 生命周期：prefill 分配 → 写入记账（`hash_blocks`）→ decode 增长 → FINISHED 释放；chunked prefill 的进度推进与中间 token 丢弃；
- 抢占路径：块不足 → 抢占 running 队尾 → 释放的块立即被其他请求复用（同一 block_id）→ 被抢占请求进度不丢失并可恢复至完成；
- 引用计数与共享：共享块最后一个引用负责释放；占用中的命中块不消耗空闲容量；
- 正确性防护：链式哈希从头连续匹配、首块不匹配即中断、哈希命中后逐块 token 校验。

测试清单与行为映射表见 kv-cache.md 第 6 节。

### 3.3 KV 池实测数据（用于文档量化分析）

在 RTX 4090 D + Qwen3-0.6B（bf16，`enforce_eager=True`，TP=1）上实际加载模型测得：

| 参数 | 实测值 |
| --- | --- |
| num_kvcache_blocks | 698 |
| kv_cache 形状 | `(2, 28, 698, 256, 8, 128)`，共 19.086 GiB |
| 单块字节数 | 29,360,128 B ≈ 28 MiB |
| 可缓存 token | 178,688 |

## 4. 验收结论

| 项 | 状态 |
| --- | --- |
| Day 2 产出与验收 | **通过** |
| Day 3 产出与验收 | **通过** |
| 后续（Day 4 prefill/decode 与采样对比） | 未验收，不属于本次范围 |
