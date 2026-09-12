# Day 8 Chunked Prefill 验收记录

> 对应设计文档 [chunked-prefill.md](./chunked-prefill.md)。本文记录实现范围、实际执行的命令与结果、设计偏差取舍及未覆盖边界；设计文档中的验收清单（§10）保持原文，完成情况在 §8 逐项核对。原始事件证据位于 `docs/evidence/day8/`。

## 1. 实现范围与分支状态

- 分支：`feature/chunked-prefill`（基线 dev=`5d56710`，含 Day7），全部改动留在工作区，未提交、未合并、未改动分支状态。
- 代码改动：
  - `nanovllm/config.py`：新增 `chunk_size: int = 1024`，复用 `validate_positive_int` 显式校验（拒绝 0/负/float/str/bool，`python -O` 仍生效），校验先于资源创建。
  - `nanovllm/engine/sequence.py`：`prefill_offset` 成为唯一进度事实源；`prefill_target`/`prefill_complete` 派生只读属性；`num_cached_tokens` 改为兼容只读 property（数值语义与 Day7 完全一致）；pickle `STATE_VERSION=3`（payload 字段改名 `prefill_offset`，单向兼容读取 v1/v2）；空 prompt 在构造入口显式拒绝。
  - `nanovllm/engine/block_manager.py`：`hash_blocks(seq, start, end)` 显式区间化（只登记本次执行写满的完整块，尾块永不登记，中间 chunk 写满的块同样登记）；`allocate` 将 prefix 命中量写入 `prefill_offset`；`deallocate` 归零 offset（进度随物理块作废）。
  - `nanovllm/engine/scheduler.py`：prefill 接纳循环接入 `q = min(剩余需求, chunk_size, B-used)`；扫描位置与队列分离（每请求每轮至多一个 chunk，中间 chunk 保持 WAITING 原地、本轮不再被扫描，后续请求按 FCFS 继续考察）；首候选拆分、后续候选必须整段放下；决策快照新增 `chunk_index`（1-based，per-request 计数器，恢复重算后从 1 重新计数）/`offset_before`/`is_last_chunk`；`_finalize_round` 四条独立校验（q_i 正整数、q_i<=chunk_size、sum<=B、len(batch)<=max_num_seqs，另含 seq_id 去重）；`postprocess` 按 §4.5 契约重写（批次快照校验拒绝重复/迟到收尾、成功才推进 offset、推进后校验单调有界、中间 chunk 丢弃采样、token 数与需采样集合显式对齐，不使用会静默截断的 zip）。
  - `nanovllm/engine/model_runner.py`：`_build_prefill_inputs`/`_build_decode_inputs` 拆为 CPU 纯组装函数（可被无 GPU 单测直接调用）；`prepare_prefill` 消费显式 `prefill_offset` 并显式校验 `0 <= start < end <= prefill_target`，含展平长度显式校验（input_ids/positions/slot_mapping/cu_seqlens_q 一一对应）；`_build_decode_inputs` 的 positions/context_lens 改用 `num_tokens` 元数据（TP worker 空 token_ids 不影响）；`run()` 支持"无采样返回 None"，prefill 只对最后 chunk 请求采样并显式校验 logits 行数与请求数一致（见 §4 偏差说明）。
  - `nanovllm/engine/llm_engine.py`：调用前快照扩展为 `(seq_id, request_id, n, offset, is_last_chunk)`（rank 0 统计，不进 TP payload）；`engine_round` 事件新增 `prefill_chunks` 字段。
  - `nanovllm/layers/attention.py`、`nanovllm/utils/context.py`：未修改（§4.4：跨 chunk causal 语义由 positions 绝对位置与 `cu_seqlens_k = offset + q` 两个上游契约保证）。

## 2. 环境与模型可用性

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D（24564 MiB，驱动 595.71.05） |
| torch / CUDA | 2.5.1+cu124 / 12.4 |
| transformers | 5.16.1 |
| 模型 | Qwen3-0.6B，本地快照 `c1899de289a04d12100db370d81485cdf75e47ca`（HF cache，未下载） |
| 引擎配置 | TP=1，enforce_eager=True，kvcache_block_size=256，max_model_len=12288，KV 池 706 块 |
| 结论 | GPU 与模型均可用，§9.3 GPU 最小验证已实际执行 |

## 3. CPU 测试（实际命令与结果）

```text
python -m pytest tests/test_chunked_prefill.py -q      # 92 passed
python -m pytest -q                                    # 259 passed（新增 92 + 既有 167）
python -O -m pytest -q                                 # 259 passed（显式校验在优化模式下仍生效）
PYTHONPATH=. python scripts/validate_chunked_prefill.py --mode cpu --output /tmp/day8-cpu.jsonl
# [cpu] 校验结果: PASS
```

新增 `tests/test_chunked_prefill.py` 覆盖设计文档 §9.1 表格全部 12 类：配置校验（chunk_size=1 合法、0/负/float/str/bool 拒绝、默认 1024、SimpleNamespace 缺省回退）、字段语义（初始 offset=0、派生只读、兼容读取、deallocate 归零、复用 ID 不继承）、chunk 边界（prompt 长度 0 显式拒绝/1、7/8/9、B-1/B/B+1、8K=8192 且区间 [0,8192) 连续覆盖）、预算交互（chunk_size 与 B 小/等/大、独立校验、部分推进记 scheduled、预算耗尽归因不变）、调度规则（每请求每轮 1 chunk、首候选拆分、后续整段、FCFS 不跳过、不重复入批、chunk_index 阶段计数与恢复重置）、输入组装（真实 `_build_prefill_inputs`/`_build_decode_inputs`：片段/绝对位置/cu_seqlens/跨块 slot/展平长度/越界抛错/TP 反序列化 decode 元数据）、prefix 与恢复（命中块计入初始 offset、尾块不命中、中间 chunk 写满块登记复用、显式区间 hash、抢占恢复重算）、采样（中间 chunk 无行选择/不耗 RNG（generator 状态判据+违规对照）/token 数不匹配显式拒绝/混合批次对齐）、生命周期（chunk 中 cancel/timeout/外部 mark_*、重复与迟到 postprocess 拒绝、失败不提交）、一致性（固定注入下 one-shot 与多 chunk completion 序列一致）、TP 协议（v3 往返、v1/v2 映射、未知版本拒绝、payload 无 rank0 统计字段）、随机混合（固定 seed 交错操作、双上限逐轮断言、终态全平衡、chunk 计数回收）。

既有测试适配（行为断言不变，注入方式随新契约调整）：
- `tests/test_block_manager.py`：`hash_blocks` 显式区间签名（2 处调用）。
- `tests/test_kv_cache_lifecycle.py`：分块 prefill 中间轮不再传入采样 token，并新增"中间 chunk 传入 token 显式抛错"断言。
- `tests/test_token_budget.py`：决策快照精确字典断言扩展 3 个新字段；分块中间轮 token 注入改为按需采样集合注入（含两个随机混合场景）；文件头注明 Day8 适配说明。20-token/B=8 的 8/8/4 等既有行为断言原样保持。
- `tests/test_request_lifecycle.py`：超时分块用例的中间轮注入改为 `[]`（真实 runner 对中间 chunk 返回 None）。

## 4. 设计偏差与取舍说明

1. **prefill logits 形状与"采样行选择"的落实方式**。设计文档 §1.3/§4.5 假设 prefill logits 为 `[T, vocab]`（描述旧代码 zip 静默截断的行错位 bug）。实际代码中 `ParallelLMHead.forward` 的 prefill 分支**已经**按 `cu_seqlens_q[1:]-1` 把每个请求的最后一个 query 行聚合为 `[len(seqs), vocab]`，批内第 i 行就是第 i 个请求的末 query logits。初次实现按文档假设在已聚合张量上再取扁平行号，GPU warmup（8192 token，首次跑到该规模）触发索引越界 device-side assert，据此确认实际形状并修正。最终落实为三件事，行为语义与 §4.5 要求一致：(a) `run()` 显式校验 `logits.shape[0] == len(seqs)`（形状错位不再静默）；(b) 只对最后 chunk 请求做子集行选择（`_select_prefill_sample_rows`，批内下标即聚合后行号）；(c) 中间 chunk 不采样、不消耗 RNG。该前提偏差已同步标注在代码注释与测试 docstring 中。
2. **空 prompt（长度 0）**。文档 §9.1 要求覆盖"prompt 长度 0"。现状在 `Sequence.__init__` 中 `last_token = token_ids[-1]` 直接 IndexError；且空序列在 decode 组装（`block_table[-1]`）同样崩溃。Day8 选择在构造入口显式 `ValueError("prompt 不能为空")`（非法操作尽早抛显式异常），以测试固化行为，而非支持空上下文请求（后者需要改动 KV 分配与 decode 语义，超出 Day8"分配语义不变"边界）。
3. **chunk_index 口径**。文档 §6.4 示例中 `chunk_index: 1` 与 `offset_before: 0` 同现，据此实现为 **1-based 的"当前 prefill 阶段第 N 个 chunk"**，由 Scheduler 侧 per-request 计数器维护；prefix 命中或抢占恢复重算开启新 prefill 阶段时从 1 重新计数（比 `offset // chunk_size` 推导更能表达分块进度，且在预算主导拆分下不会失真）。计数器随终态回收，随机混合测试断言无残留。
4. **chunk_size 挡住非首候选时的归因**。§4.1 明确"不因 chunk_size 产生新的等待原因"，该场景沿用 Day7 的 `budget` 直接归因 + 后续 `head_of_line`；其中"剩余预算足够、仅 chunk_size 不足"的子场景归因口径偏保守（计入 budget 延后人数），已在代码注释与本记录标注，未新造原因字符串。
5. **`Sequence.chunk_size` 冗余字段未实现**。§6.2 将其标注为"可选，便于 prepare 校验"；Day8 选择由 Scheduler 在轮末以自身配置做 q_i<=chunk_size 独立校验，避免向每个请求注入配置副本。

## 5. GPU 验证（实际命令与结果）

```text
PYTHONPATH=. python scripts/validate_chunked_prefill.py --mode gpu \
  --model <Qwen3-0.6B 本地快照目录> \
  --output docs/evidence/day8/day8-gpu.jsonl
```

脚本流程：每个场景独立建引擎（KV 容量自检先行：8194 token 需 33 块 < 池 706 块，不足时报配置错误）；`torch.cuda.reset_peak_memory_stats()` 后驱动；逐轮收集调度决策中的 chunk 快照构成 offset 轨迹并独立重算连续性；最后对全部事件做白名单/上限/连续性独立校验。

### 5.1 8K 主场景（8192-token prompt，greedy，max_tokens=2）

| chunk_size | B | 轮数（prefill+decode） | wall time | TTFT* | peak allocated | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 8192（one-shot 对照） | 8192 | 2（1+1） | 2.44 s | 1.13 s | 21157 MiB | 完成，offset 覆盖 [0,8192) |
| 1024 | 2048 | 9（8+1） | 1.14 s | 1.10 s | 21157 MiB | 完成，offset 覆盖 [0,8192) |
| 512 | 2048 | 17（16+1） | 0.64 s | 0.61 s | 21144 MiB | 完成，offset 覆盖 [0,8192) |
| 256 | 2048 | 33（32+1） | 0.93 s | 0.91 s | 21137 MiB | 完成，offset 覆盖 [0,8192) |

\* TTFT 观测点为最后 chunk 完成轮（请求产出首个 completion token）。轮数与理论值逐档吻合（8192/chunk_size + 1），offset 轨迹由脚本独立重算：区间连续、单调、无重叠遗漏、并集恰为 [0, 8192)。三档均无 OOM。peak memory 为观测值，不承诺改善方向。

### 5.2 greedy 一致性（6 个固定 prompt，temperature=0，max_tokens=16）

- one-shot（chunk_size=8192）vs chunked（chunk_size=16）：**5/6 逐 token 一致**；差异分析与补充实验见 §6。
- chunk_size=256 复测：**6/6 一致**（`docs/evidence/day8/day8-greedy-cs256.jsonl`）。
- chunk_size=16 复跑：仍 5/6，且同一 prompt 同一位置同样分歧（确定性数值差异，非随机）。

### 5.3 prefix 命中

共享前缀 2 个完整物理块（512 token）：第二条请求初始 `offset_before=512`（期望 512），实际执行 3 token（= 新增 token 数，期望 3）——命中块不重算、尾块不假命中。证据字段 `prefix_check`。

### 5.4 事件校验

字段白名单（含 Day8 新增 `prefill_chunks` 与决策字段 `chunk_index/offset_before/is_last_chunk`）、`0 < q_i <= chunk_size`、`sum(q_i) <= B`、`prefill_chunks` 与调度决策一致、per-request chunk 区间连续且 `is_last_chunk` 恰好一次、`chunk_index` 从 1 连续递增：全部 PASS。事件不含 prompt/token 明文。

## 6. greedy 差异分析（§4.6：如实记录，不偷改预期）

- 现象：prompt 4（"To be or not to be, that is the question"）在第 11 个生成 token 分歧——one-shot 产出 3405、chunked(16) 产出 4226，此前 11 个 token 逐位一致。
- 语境：该步模型处于复读循环（输出呈 `1112 323 429 374 279 4226` 周期），两个候选 token 的 logits 处于近似平局。
- 机制：chunked prefill 将同一序列的 attention 拆到多次 varlen kernel 调用中执行，与一次性全长的浮点归约顺序不同，产生微小数值差；在近似平局处累积误差翻转了 argmax。属于 §4.6 预先声明的"varlen kernel 数值差异视为可接受"类别，非调度/进度语义错误（CPU 桩一致性测试证明固定注入下两路径输出完全一致；GPU 上分歧确定性复现、位置不变）。
- 佐证：chunk_size 增大到 256（chunk 边界减少）后 6/6 一致——数值漂移随 chunk 边界数量收敛；复跑 chunk_size=16 分歧位置与数值完全相同——差异是确定性的，不是随机采样或竞态。

## 7. 性能矩阵（§9.4，只报告观测值）

见 §5.1 表格；原始数据（含 `max_memory_reserved`、每档 run_config、offset 轨迹）见 `docs/evidence/day8/day8-gpu.jsonl` 的 `perf_matrix` / `perf_matrix_summary` / `offset_trajectory` 记录。中短 prompt 控制组未单独建档（见 §9 未覆盖边界）。观测摘要：本环境单请求 8K 场景下 chunked 档位的 wall time 与 TTFT 均低于 one-shot 档（one-shot 单轮激活峰值受 8192-token 全批影响），三档 chunk_size 之间差异在重复执行波动量级内；不据此承诺任何方向性结论。

## 8. 验收清单逐项核对（对照设计文档 §10）

**功能与口径**

- [x] `Config.chunk_size` 显式正整数校验、默认 1024、不可热修改、`python -O` 仍生效 —— 校验走 `validate_positive_int`（显式 raise）；`python -O -m pytest -q` 259 passed；无任何运行期修改入口。
- [x] `prefill_offset` 唯一进度事实源、`num_cached_tokens` 兼容只读、`prefill_complete` 派生；终态/释放后进度作废不污染复用 ID —— 单测覆盖（字段语义类 + 生命周期类），`num_cached_tokens` 全仓库写点已清零（只剩 property）。
- [x] 长 prompt 分块推进：区间连续单调无重叠遗漏、每请求每轮至多一个 chunk、首候选拆分、FCFS 不跳过 —— CPU 单测（含 8K 与 §5.1 时序同款场景）+ GPU offset 轨迹独立重算。
- [x] 每轮 `q_i <= chunk_size`、`sum(q_i) <= B`、`len(batch) <= max_num_seqs` 独立校验；chunk 部分推进不误记 budget —— `_finalize_round` 四条独立 raise；单测 + GPU 事件校验逐轮核对；`budget_deferred_requests==0` 断言。
- [x] prefix 命中/抢占恢复按显式 offset 正确重算；尾块永不命中；`hash_blocks` 显式区间只登记写满块 —— 单测（prefix/恢复类）+ GPU prefix 场景（初始 offset=512、执行 3 token）。
- [x] 中间 chunk 不采样、不消耗 RNG、不追加 completion；最后 chunk 采样该请求最后 query 行并完成 prefill —— 单测（采样类，generator 状态判据）+ GPU greedy 5/6+6/6 一致性（RNG 流与 chunk 划分无关的实证）。

**执行正确性**

- [x] `prepare_prefill` 消费显式 offset 并通过范围/长度校验；input_ids/positions/slot_mapping/cu_seqlens 一一对应；positions 为绝对位置 —— CPU 单测直接调用真实组装函数（跨块 slot 逐槽核对、越界显式抛错）。
- [x] attention key 长度 = 历史有效 KV + 当前 query；causal 不变；多请求混合批次无串扰 —— `cu_seqlens_k = offset+q` 组装单测 + GPU 8K/greedy/prefix 全部走该路径输出正常；attention 层未修改（§4.4 上游契约）。
- [x] decode 位置/上下文长度使用序列元数据（TP worker 空 token_ids 不影响）—— `_build_decode_inputs` 改用 `num_tokens`；单测含 TP 反序列化（token_ids 为空）前后组装结果一致。
- [x] postprocess 原子提交：成功才推进 offset；取消/超时/异常/重复收尾不推进；批次快照校验拒绝迟到结果 —— 单测（生命周期类）；全终态旧批次重放保持 Day7 幂等语义。
- [x] `Sequence` pickle v3 + v1/v2 单向兼容，往返测试通过；rank 0 统计不进 TP payload —— 单测（TP 协议类：v3 往返、v2/v1 映射、未知版本拒绝、payload 白名单）。

**实测与交付**

- [x] 8K prompt 在 256/512/1024 三档下分块完成且不 OOM（KV 容量自检先行）—— §5.1，706 块池 vs 需 33 块。
- [x] 至少 5 个固定 prompt 的 chunked vs one-shot greedy 输出一致（差异须分析记录）—— 6 prompt：chunk_size=16 时 5/6（差异分析见 §6）、chunk_size=256 时 6/6；证据 JSONL 齐备。
- [x] 256/512/1024 执行时间与显存峰值原始数据归档，配置/命令/seed 可复现 —— `docs/evidence/day8/day8-gpu.jsonl`（perf_matrix + run_config + offset 轨迹）；greedy 补充实验 `day8-greedy-cs256.jsonl`、`day8-greedy-cs16-repro.jsonl`。
- [x] CPU 全量回归（新测试 + Day2–7）与 `python -O` 通过，数字为实际执行结果 —— 259 passed / 259 passed（§3）。
- [x] `docs/day8-validation.md` 记录 GPU/模型可用性、实际命令、结果与未覆盖边界；设计文档已加入 `docs/README.md` 索引；日志证据无 prompt/token 明文 —— 见 §2/§3/§5；事件白名单校验通过。
- [ ] TP>1 / CUDA Graph / 真实并发取消：**未实测**（如实记录）—— 本环境单卡无法运行 TP>1；为隔离变量与显存观测，GPU 验证统一 enforce_eager=True（CUDA Graph 捕获路径未跑）；取消/超时为调度边界注入（Day7 方法学），非真实并发取消。序列化协议、decode 元数据等 TP 相关语义已由 CPU 单测覆盖。

## 9. 未覆盖边界与遗留

- TP>1 端到端、CUDA Graph 路径、真实并发取消/超时（见 §8 末项）。
- §9.4 的"中短 prompt 控制组"未单独归档：8K 主场景四档对比已归档，中短 prompt 的分块收益观测留待 Day9 mixed batch 一并评估。
- 非首候选被 chunk_size 挡住时沿用 budget 归因的口径偏保守（§4 第 4 条），若 Day9 引入 mixed batch 归因重构时应一并细化。
- 性能矩阵为单次运行观测值，未做多轮取均值；Torch 配置（如预分配池、kernel autotune）未做额外固化。
