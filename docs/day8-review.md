# Day 8 实现审查记录

> 审查日期：2026-09-12。本文对照 [chunked-prefill.md](./chunked-prefill.md) 全文（重点 §3–§6 契约与 §10 验收清单）及 [day8-validation.md](./day8-validation.md)，检查 `feature/chunked-prefill` 分支的未提交工作区；不替代设计文档与验收记录，也不扩大 Day8 范围。

## 1. 审查范围与实际命令

逐行检查了 `nanovllm/config.py`、`nanovllm/engine/{sequence,scheduler,block_manager,model_runner,llm_engine}.py` 的全部改动 diff，`tests/test_chunked_prefill.py`（1351 行）与四个既有测试的适配改动，`scripts/validate_chunked_prefill.py`，以及 `docs/evidence/day8/` 三份 JSONL 原始证据（脚本独立解析复核，非仅对照验收记录转述）。

本次实际执行：

- `python -m pytest -q`：审查初轮 259 passed；补强（§3）后 260 passed。
- `python -O -m pytest -q`：259 → 260 passed（显式校验在优化模式下仍生效）。
- `PYTHONPATH=. python scripts/validate_chunked_prefill.py --mode cpu --output /tmp/day8-cpu-review.jsonl`：PASS（补强后复跑仍 PASS）。
- GPU 未重新执行；对 `docs/evidence/day8/` 既有 JSONL 做了独立解析核对：8K 四档 offset 轨迹连续且并集恰为 [0, 8192)、轮数与 8192/chunk_size+1 逐档吻合、greedy 5/6（cs=16，分歧为 prompt 4 第 12 个 token 的确定性数值差异）与 6/6（cs=256）、prefix 初始 offset=512/执行 3 token，均与验收记录 §5/§6 声明一致。

## 2. 已确认正确的实现

- **单一事实源**：`num_cached_tokens` 全仓库已无写点残留（仅剩兼容读取与序列化注释），`prefill_offset` 是唯一可写进度字段；pickle v3 往返、v1/v2 单向映射、未知版本拒绝均有测试；`deallocate` 归零防止复用 request_id 继承进度。
- **调度扫描正确性**：逐情形 trace 验证了"扫描位置与队列分离"实现——中间 chunk 接纳后 `scan += 1`（原地保留）、最后 chunk 接纳后 `waiting.remove`（scan 不动恰好指向后继），`scan` 之前的元素必然全部已决策，无跳过、无重复扫描；首候选拆分（`batch` 为空才允许 `q < needed`）、后续整段放下、FCFS 不跳过与 §4.1 一致；`_finalize_round` 四条上限（正整数 / `q_i<=chunk_size` / `sum<=B` / `len(batch)<=max_num_seqs`）加 seq_id 去重独立生效。
- **hash_blocks 显式区间**：数学上验证了 floor 除只登记本次执行写满的块；跨 chunk 拼写满的块由"补完它的那个 chunk"登记（如 [0,100)+[100,300) 在第二个 chunk 登记 Block 0，链基 h=-1 正确重启）；区间起点的前驱块必然已登记（前序 chunk 提交或 prefix 命中，二者均携带链式哈希）；decode 轮逐 token 推进时的登记时机与 Day7 旧行为逐位一致，无回归。
- **输入组装**：`_build_prefill_inputs` 显式消费 `prefill_offset/num_scheduled_tokens`，范围与展平长度校验均为显式 raise；测试用独立实现的 `_prefill_slot_expectation` 逐槽交叉验证跨块 slot_mapping；positions 绝对连续；`cu_seqlens_k = offset + q`，多 chunk 续传与 prefix 命中场景下 `cu_k > cu_q` 恒触发 block_tables 分支，attention 层无需改动（§4.4 上游契约成立）。
- **采样契约**：中间 chunk 不进采样器、不消耗 RNG（generator 状态判据测试 + GPU greedy 一致性实证）；行选择依赖 `ParallelLMHead` prefill 分支既有的"每请求末 query 行聚合"，验收记录 §4 偏差 1 属实——设计文档 §1.3 的 `[T, vocab]` 假设与实际代码不符，实现按实际形状落实为"行数显式校验 + 最后 chunk 子集选择"，语义与 §4.5 等价且比原设计更严格。
- **postprocess 原子提交**：`needs_sample` 在任何状态变更前按调度快照计算，采样游标推进与安全检查解耦（被丢弃输出的占位不致错位）；先整批校验再逐个提交；终态/取消/超时不推进；推进后立即校验单调有界。
- **生命周期**：chunk 中取消/超时/外部 mark_* 不推进、KV 释放；preempt→resume 走 offset 归零 + prefix 命中重算，与 §3.2/§4.2/§5.3 不变量 8 一致（§3.4 mermaid 中"offset 进度保留"的措辞张力见 §4.6）；chunk 计数随终态回收。
- **既有行为保持**：Day7 的 20-token/B=8 → 8/8/4 断言原样通过；预算耗尽归因、sequence_cap 优先、phase_priority、空批次/异常锁定语义未变。

## 3. 本次补强（审查中修正）

1. **idle 轮 `engine_round` 事件补 `prefill_chunks: 0`**（`llm_engine.py`）。空批次分支原本缺失该字段，而验收脚本白名单要求所有 `engine_round` 事件都含它——一旦出现合法 idle 轮（KV 不足、全部暂停等），GPU 验收会被误判 FAIL（假阴性）。已补字段 + 新增单测 `test_idle_round_event_has_zero_prefill_chunks` 固化。
2. **`scenario_prefix` 打印文案修正**（`scripts/validate_chunked_prefill.py`）：硬编码"期望 5"改为动态 `expected_executed`（=3）。已归档 JSONL 不受影响（其中记录的 `expected_executed` 本来就是 3，仅 stdout 文案错误）。

## 4. 审查发现的问题与边界（不阻断验收）

1. **§6.4 `runner_input_len` 未实现且未声明为偏差（唯一未闭环的设计条款，建议尽快收口）**。设计要求验收脚本侧以 runner 包装注入每请求 `runner_input_len` 观测（"不属于库内事件"）；实际脚本、JSONL 证据、验收记录 §4 偏差清单中均无此字段，属于静默偏差。信息面上 `decisions[].scheduled_tokens` 已记录每请求接纳量，缺的是"实际 runner 输入长度 vs 调度计划"的独立交叉核对（与 Day7 审查 §4.1 指出的同类遗留）。建议二选一：在脚本中补 runner 包装并重跑 GPU 证据，或在 `day8-validation.md` §4 增补偏差条目说明取舍。
2. **GPU 全场景终检未传 `chunk_size`**：`run_gpu` 末尾的 `validate_events` 未提供 chunk_size，"0 < q_i <= chunk_size" 的独立重算只在 CPU 模式生效。GPU 侧该约束由 `Scheduler._finalize_round` 的硬校验兜底（违者当场 raise），证据数据本身亦可见合规，但"证据独立重算"的口径不完整。建议脚本按各场景 `run_config` 的 chunk_size 分桶校验。
3. **全终态批次豁免 token 数校验的换气口**：`postprocess` 对纯终态重放不校验 token 数（保 Day6/Day7 "重放是安全空操作"契约）；若重放方给出的 token 列表长度与 `needs_sample` 不符，会以 IndexError 而非显式 ValueError 暴露。仅库外误用可达，风险低，记录在案。
4. **迟到 postprocess 在"同 seq 已被重新规划"后无法识别**（继承自 Day7 的同构隐患，非 Day8 引入）：重复收尾在"计数已清零"时被拒，但若两次收尾之间同 seq 已被重新接纳（`num_scheduled_tokens > 0`），旧批次重放会以当前计划二次提交。同步引擎流程（schedule→run→postprocess 严格串行）内不可达；Day9/TP 异步结果路径引入时应以 round_id 关联校验加固。
5. **可读性建议（不强制，可在 Day9 顺手处理）**：
   - `prefill_offset + num_scheduled_tokens == prefill_target`（"本轮接纳即完成 prefill"判定）在 scheduler/model_runner/engine/测试/脚本中重复 5 处，可在 `Sequence` 上收敛为具名只读方法，单一权威；
   - `LLMEngine.step` 调用前快照 5 元组中的 `offset/is_last_chunk` 当前无消费方（设计 §6.3 要求保留），建议改为具名结构或注明用途，避免被当作死代码清理；
   - `_build_prefill_inputs` 的 `has_block_table` 实际语义是"批内全部持有 block_table"（任一缺失即 False），命名可更准确。
6. **设计文档内部措辞张力（无需改代码）**：§3.4 mermaid "PREEMPTED 释放物理块，offset/token 进度保留" 与 §3.2/§4.2 "deallocate 归零、恢复重算" 表述不一致；实现遵循后者（正确），mermaid 该行宜理解为"token_ids 生成进度保留"。
7. **性能矩阵为单次观测、中短 prompt 控制组未归档**：验收记录 §9 已如实声明，维持"只报告观测值"口径，留待 Day9 mixed batch 一并评估。

## 5. 审查结论

达到 §10 验收标准：17 项中 16 项经代码审查 + 测试复跑（260/260，`python -O` 同）+ 证据独立解析确认通过；"TP>1 / CUDA Graph / 真实并发取消"保持如实记录未测（单卡环境无法 TP>1，GPU 验证统一 enforce_eager）。`chunked-prefill.md` §10 清单已同步勾选（末项保持未勾并注明按"如实记录"分支处理）。§4.1 的 `runner_input_len` 是唯一未闭环的设计条款，已给出处理建议，不阻断 Day8 收口；建议随 Day9 mixed batch 的证据工具一并处理。
