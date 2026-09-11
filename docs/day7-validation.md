# Day 7 验收记录：每轮 Token Budget 与 FCFS 调度

> 对应设计标准 [token-budget.md](./token-budget.md)（§10 最终验收清单逐项核对见本文第 6 节）。
> 实施日期：2026-09-11。本记录为实施后创建，所有数字均为实际执行结果。

## 1. 基线集成与实施范围

- 集成方式：`feature/scheduler-token-budget` 合并 `dev`（合并提交 `742f382`），使开发分支具备 Day6 基线（`c84e73f` 经 dev 的合并提交 `af79adb` 进入本分支）；冲突文件仅 `docs/README.md`（索引条目合并）。
- 集成后、实施前基线回归：`python -m pytest -q` → **99 passed**（79 lifecycle + 20 既有），保留 Day6 身份/暂停语义。
- 实施涉及文件（与设计 §0 一致，未改 BlockManager 分配语义、模型执行布局与 Sequence TP 协议）：
  - `nanovllm/config.py`：新增 `validate_positive_int` 显式校验入口；`Config.__post_init__` 校验 `max_num_batched_tokens` 与 `max_num_seqs`（正整数、拒绝 bool/float/str、允许 B < max_num_seqs），不用 assert。
  - `nanovllm/engine/scheduler.py`：统一预算记账（prefill/decode 同口径）、阶段内 FCFS 延后、七类原因归因、预算等待 episode 统计与标量累计、结构化事件日志（`scheduler_round`/`budget_wait_episode`/`request_budget_wait`）；新增 `_estimate_prefill_tokens`、`_attribute_prefill_stop`、`_record_budget_decisions`、`_record_paused_decisions`、`_finalize_round` 等内部函数；`schedule(now=)`/`preempt(now=)`/`_finalize(now=)` 支持注入确定时钟。
  - `nanovllm/engine/llm_engine.py`：`step()` 读取 `last_schedule_stats` 关联 round_id，调用前保存 `(seq_id, request_id, n)` 快照并显式校验（非正计数 / 超预算抛 ValueError），模型异常记录 `outcome=error, executed_tokens=null` 后原样上抛；空轮记录 idle 事件；`generate()` 无进展 RuntimeError 契约保留并补充"KV 不可容纳"提示分支。
  - `tests/test_token_budget.py`（新增，68 项）、`scripts/validate_token_budget.py`（新增验收工具）。
- 全部新增/修改代码含中文注释（原因分类、统计口径、所有权保护等处均标注对应设计章节）。

## 2. 单元测试结果（§8.1 / §8.2）

实际执行命令与结果（工作区根目录）：

| 命令 | 结果 |
| --- | --- |
| `python -m pytest tests/test_token_budget.py -q` | **68 passed** |
| `python -m pytest tests/test_request_lifecycle.py -q` | **79 passed**（Day6 全量） |
| `python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q` | **20 passed** |
| `python -m pytest -q`（全量） | **167 passed** |
| `python -O -m pytest tests/test_token_budget.py -q` | **68 passed**（附 pytest 标准提示：非测试模块断言被 -O 忽略；校验逻辑为显式 raise，不受影响） |
| `python -O -m pytest tests/test_request_lifecycle.py -q` | **79 passed** |

`tests/test_token_budget.py` 覆盖 §8.1 矩阵全部 14 行：配置边界（B=1、B<max_num_seqs 合法；0/负/float/str/bool 拒绝；SimpleNamespace 直接构造复用同一校验入口且先于 BlockManager 创建）、空与终止、单/多 prefill（[3,5,2]/B8、[6,4,1]/B8、prefix 命中与 resume 重算计费）、decode 硬上限（B=2、6 条请求多轮 prefill 造出 N>B，未热改预算）、decode 副作用（未选中请求 token/status/KV/ref_count 不变、B=1 不为预算抢占、KV 不足抢占记 kv_capacity）、FCFS 最终完成、相互限制（cap 先于 budget、phase_priority、KV 队首不足 HOL）、统计人数（direct/HOL 去重、唯一请求数、部分 chunk 不计）、统计时间（t10/t12/t15/t20/t23 时间线 rounds=3、seconds=8 精确；取消/超时/抢占/直接 mark_*/重复清理）、所有权回归（两种结束方式 + ID 复用 + 迟到 postprocess）、Engine 观测（计划=runner 输入、postprocess 清零不丢证据、取消丢输出不减 executed、runner 抛错 executed=null）、日志（JSON 解析、round_id 唯一、<=B、计数可重算、无 prompt 明文）、长序列混合（固定 seed、120 轮上限、身份/队列/引用计数/free-used 平衡）。

## 3. 验收脚本与 CPU 证据（§6.4 / §8）

```bash
python scripts/validate_token_budget.py --mode cpu --output /tmp/day7-budget-cpu.jsonl
# 证据归档：docs/evidence/day7/day7-budget-cpu.jsonl
```

输出（exit=0）：

```text
[cpu] 调度轮次: 15（非空 15）
[cpu] 成功执行轮: 15，最大 executed_tokens: 8
[cpu] 预算延后: 唯一请求 17 个 / 请求-轮次 107 次 / 已结算等待 53.500 秒
[cpu] decode 受预算分批证据: 5 轮存在 budget 延后
[cpu] 结束时 KV block free/used: 64/0
[cpu] 校验结果: PASS
```

CPU 模式构造：真实 Scheduler + 真实 `LLMEngine.step` 契约，ModelRunner 为桩（固定注入采样 token）、固定单调时钟、block_size=8、B=8、max_num_seqs=16、无模型初始化。请求集含 [3,5,2]（恰好耗尽）、[6,4,1]（不跳过）、20-token（跨多轮分块）与 12 条 1-token 短请求（decode 12 > B=8）。

脚本独立校验内容（两类模式共用）：JSONL 逐行可解析、round_id 唯一且单调、每轮 `0 <= planned <= B`、批序列数 <= max_num_seqs、`budget_deferred_requests == direct + hol`、decisions 的 scheduled_tokens 求和 == planned_tokens、成功执行轮 `executed == planned <= budget`、idle 轮 model_called=False、error 轮 executed=null、episode 秒数可由起止时间重算、事件字段白名单（无 prompt/token 内容字段）、且必须存在 decode 预算延后轮（缺失即非零退出）。

## 4. GPU 最小验证（§8.3）

环境：NVIDIA GeForce RTX 4090 D（24,564 MiB）、torch 2.5.1+cu124、transformers 5.16.1、模型 `/root/huggingface/Qwen3-0.6B`（本地权重，无下载）、TP=1、enforce_eager、物理 block_size=256（budget 只控制请求迭代）、KV 池 714 blocks。

实际命令与结果（均 exit=0）：

```bash
python scripts/validate_token_budget.py --mode gpu --model /root/huggingface/Qwen3-0.6B \
  --token-budget 8  --max-num-seqs 16 --max-new-tokens 4 --output /tmp/day7-budget-gpu-b8.jsonl
python scripts/validate_token_budget.py --mode gpu --model /root/huggingface/Qwen3-0.6B \
  --token-budget 16 --max-num-seqs 64 --max-new-tokens 4 --output /tmp/day7-budget-gpu-b16.jsonl
python scripts/validate_token_budget.py --mode gpu --model /root/huggingface/Qwen3-0.6B \
  --token-budget 32 --max-num-seqs 64 --max-new-tokens 4 --output /tmp/day7-budget-gpu-b32.jsonl
python scripts/validate_token_budget.py --mode gpu --model /root/huggingface/Qwen3-0.6B \
  --token-budget 8  --max-num-seqs 16 --max-new-tokens 4 --greedy-compare \
  --output /tmp/day7-budget-gpu-b8-compare.jsonl
```

| 运行 | 轮次（非空） | 最大 executed_tokens | 预算延后（唯一/轮次/秒） | decode budget 延后轮 | free/used | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| B=8, cap=16 | 112 | 8 | 17 / 1345 / 74.4s | 4 | 714/0 | PASS |
| B=16, cap=64 | 63 | 16 | 25 / 1175 / 115.1s | 3 | 714/0 | PASS |
| B=32, cap=64 | 39 | 32 | 41 / 1175 / 126.4s | 3 | 714/0 | PASS |
| B=8 + greedy 对比 | 112 | 8 | 17 / 1345 / 96.5s | 4 | 714/0 | PASS |

说明：B=16/32 的 sweep 采用 max_num_seqs=64（§8.3 要求"包含 B<max_num_seqs 的场景"；B=8 时 8<16 已覆盖）。首轮尝试（cap=16）中 decode 延后全部由 cap 先触发（按设计记 sequence_cap 而非 budget），因此无 decode 预算证据轮，属配置与场景不匹配，非实现缺陷。

专项证据摘录（JSONL 原始事件，路径见 §5）：

- **decode 受预算分批**（b8 round 106，phase=decode）：`planned_tokens=8`（8 条各 1 token），`budget_deferred_direct=10`，延后请求 `needed_tokens=1, scheduled_tokens=0, reason="budget"`，未抢占、未释放 KV。
- **分块继承**：B=8 下 9-token prompt 记录 `needed=9, scheduled=8`，下一轮 `needed=1`；20-token prompt 按 8/8/4 三轮推进；512-token 共享前缀请求按 64 轮分块。
- **prefix 只计未缓存 token**：两条相同 512-token（2 个物理块）请求，第一条 `needed=512`；第二条在首块命中缓存后 `needed=256`（= 512 − block_size×1，末块按协议不缓存），逐轮递减。
- **greedy 对比**（5 个固定 prompt，temperature=0，max_new_tokens=4）：受限 B=8 与充裕 B=4096 的最终 token_ids **5/5 一致**，记录于 `day7-budget-gpu-b8-compare.jsonl` 的 `greedy_compare` 事件；未出现数值差异。
- **KV 收尾**：各次运行结束时 `free/used = 714/0`，活动/等待统计索引清空。

## 5. 证据文件与复现方式

| 文件（仓库内归档） | 产生命令（--output 即该文件） |
| --- | --- |
| `docs/evidence/day7/day7-budget-cpu.jsonl` | `... --mode cpu --output <该文件>` |
| `docs/evidence/day7/day7-budget-gpu-b8.jsonl` | `... --mode gpu --token-budget 8 --max-num-seqs 16 ...` |
| `docs/evidence/day7/day7-budget-gpu-b16.jsonl` | `... --mode gpu --token-budget 16 --max-num-seqs 64 ...` |
| `docs/evidence/day7/day7-budget-gpu-b32.jsonl` | `... --mode gpu --token-budget 32 --max-num-seqs 64 ...` |
| `docs/evidence/day7/day7-budget-gpu-b8-compare.jsonl` | `... --mode gpu --token-budget 8 --max-num-seqs 16 --greedy-compare ...` |

复现：进入仓库根目录，按上表以 `python scripts/validate_token_budget.py` + 对应参数执行即可（GPU 命令需本地 Qwen3-0.6B 权重）。日志开启 INFO 由脚本负责（库代码不做 basicConfig）。

## 6. 最终验收清单逐项核对（token-budget.md §10）

### 功能与口径

| 清单项 | 状态 | 证据 |
| --- | --- | --- |
| 复用单一预算配置，正整数显式校验，允许 B<max_num_seqs，优化模式仍生效 | 通过 | 测试 TestBudgetConfigValidation；`python -O` 68 passed |
| Prefill、decode 每轮实际 query token 均不超过 B，非空请求本轮计数为正 | 通过 | 脚本独立校验全部轮 planned<=B（CPU/GPU 全部 PASS）；Scheduler/Engine 双重显式校验 |
| Prefix 命中、已缓存分块、抢占恢复按真实未缓存输入计费 | 通过 | 测试（needed=8/2 等）；GPU 摘录：第二条共享前缀请求 needed=256 |
| 阶段内 FCFS 不跳过放不下的前序请求；首请求既有 chunking 可持续推进 | 通过 | 测试 [6,4,1]/B8；GPU 20/512-token 分块证据 |
| 预算延后保留状态、KV、token 和顺序，不误抢占、不重复分配或重复入队 | 通过 | 测试 TestDecodeSideEffects / TestFCFSCompletion |
| 正确区分 budget/direct、预算 HOL、KV、sequence cap、阶段优先、暂停等原因 | 通过 | 测试 TestMutualLimits；JSONL decisions 原因字段（七类均在证据中出现） |
| 每轮预算延后人数、请求-轮次数、唯一请求数与预算等待秒数口径明确且可重算 | 通过 | 脚本口径重算校验；时间线测试 rounds=3/seconds=8 |
| 时间 episode 在接纳/原因变化/终态/抢占时正确结算，重复清理及 ID 复用不污染新对象或重复累计 | 通过 | 测试 TestBudgetWaitTiming / TestOwnershipAndIdReuse |

### 接口、观测与兼容

| 清单项 | 状态 | 证据 |
| --- | --- | --- |
| schedule tuple、step 正负工作量、generate 输出、Day6 状态机与 TP 协议保持兼容 | 通过 | 全量回归 167 passed；测试断言 step 返回正/负数；Sequence 序列化协议未改动 |
| 计划日志与实际执行日志按 round_id 关联，postprocess 清零不丢证据，取消丢输出不抹去已执行输入 | 通过 | 测试 TestEngineObservation；JSONL engine_round 与 scheduler_round 关联校验 |
| 空轮不执行模型；模型异常不虚报成功 token；健康暂停不引入忙循环 | 通过 | 测试（idle/抛错/暂停）；JSONL idle 事件 |
| 标准日志配置由调用方控制，日志不含 prompt 明文，统计内存随活动请求数有界 | 通过 | 测试 TestLogEvidence（sentinel 不出现、字段白名单、记录终态删除） |
| 新 CPU 测试覆盖空/单/刚好/不足/多请求、统计时间、所有权与随机混合场景，并有防挂上限 | 通过 | 68 项测试；混合场景 120 轮上限 + 有界清场 |
| Day6 及 Day2–5 全量回归通过，数量与命令均为实际执行结果 | 通过 | 本文 §2 表格（99 基线 → 167 全量） |

### 实测与交付

| 清单项 | 状态 | 证据 |
| --- | --- | --- |
| 可解析日志证明每轮成功实际执行的 query token<=B，且有 decode 预算真正生效的证据 | 通过 | §3/§4：全部 JSONL 校验 PASS 且含 decode budget 延后轮（b8 round 106 摘录） |
| GPU/模型可用时完成最小实测与至少 5 个 prompt 的 greedy 对比 | 通过 | 三档预算 sweep + 5/5 一致（§4） |
| 设计与验收记录加入文档索引，保存配置、原始日志位置、未覆盖边界及已知策略限制 | 通过 | 本文 + docs/README.md 索引更新；§7 记录限制 |
| 实施前已完成经授权的 Day6 基线集成，提交说明覆盖预算、等待统计、资源/协议影响和实际测试 | 通过 | 合并提交 `742f382`；集成后基线 99 passed；后续实现改动未提交（按任务要求保留工作区状态） |

## 7. 本轮审查收尾补充

2026-09-11 收尾复核实际执行：

- `python -m pytest tests/test_token_budget.py -q`：68 passed。
- `python -m pytest -q`：167 passed。
- `python scripts/validate_token_budget.py --mode cpu --output /tmp/day7-final-cpu.jsonl`：PASS，15 轮，最大 executed=8，decode 预算延后 5 轮，KV=64/0。
- `python scripts/validate_token_budget.py --mode gpu --model /root/huggingface/Qwen3-0.6B --token-budget 8 --max-num-seqs 16 --max-new-tokens 4 --output docs/evidence/day7/day7-final-gpu-b8.jsonl`：PASS，112 轮，最大 executed=8，decode 预算延后 4 轮，KV=714/0。
- `nvidia-smi` 确认 RTX 4090 D 可用，本地 Qwen3-0.6B 权重存在；新增 GPU 原始证据已保存至 `docs/evidence/day7/day7-final-gpu-b8.jsonl`。

本轮同时补充了事件 `observed_at`、round_id 计划/执行对账、正整数运行时校验、验收工具未完成失败和最终资源硬校验，以及模型异常后的 Engine 重试拒绝。详细问题边界和未覆盖项见 [day7-review.md](./day7-review.md)。

## 8. 未覆盖边界与已知限制

- **TP>1 / CUDA Graph 未实测**：按 §8.3 不强制扩展到本日；GPU 验证均为 TP=1 + eager。Sequence 序列化协议未改动，但多 rank 路径未在预算日志下运行。
- **benchmarks/run_baseline.py 未适配**：按设计 §4.4，Day5 历史 baseline 数据与脚本推断口径保持原样，不以旧脚本结果作为预算/TTFT 验收证据；benchmark 适配另行记录。
- **性能不承诺**：小预算下尾部 decode 进度下降是 FCFS 预期行为（如 B=8 sweep 中请求-轮次 1345 次、74.4s 等待），本日只证明上限与正确性，不宣称吞吐/TTFT 改善，不切换轮转策略。
- **GPU prefix 证据的块粒度**：物理 block_size=256 下，短于一个物理块的共享前缀不产生缓存命中（协议使然，非缺陷）；prefix 计费证据使用 2 个完整物理块的共享前缀。
- **Day6 已记录的非阻塞限制沿用**：健康暂停 RuntimeError 路径 tqdm 未显式 close、异常不返回部分结果；Day7 未触及 generate 异常原子性。
- **Day8/Day9 衔接**：未实现 chunk_size/offset 体系（保留既有首请求分块）；未做 mixed prefill/decode batch；两类演进的预算口径衔接见设计 §11。
