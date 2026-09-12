# Day 10 取消、超时、抢占与恢复 验收记录

> 对应设计文档 [request-cancellation-preemption.md](./request-cancellation-preemption.md)。本文记录实现范围、设计取舍、实际执行的命令与数字结果及未覆盖边界；设计文档 §10 已在实现审查后按证据更新，逐项完成状态在本记录 §6 对照证据注明，供后续验收人复核。原始事件证据位于 `docs/evidence/day10/`。

## 1. 实现范围与分支状态

- 分支：`feature/request-cancellation-preemption`（基线 dev=`36ae91c`，含 Day9），**全部改动留在工作区，未提交、未合并**。
- `nanovllm/engine/sequence.py`：单调时钟入口 `Sequence.clock`（staticmethod，默认 `perf_counter`，测试可注入固定时钟）；构造入口校验 `deadline >= created_at`（§4.1/§6.2）。TP pickle **保持 v3 不变**——`status`/`cancel_requested`/`is_prefill`/`prefill_offset`/生成进度已在 v3 payload 中，无新增字段故无需升版；v1/v2/v3 读取与往返兼容测试补齐于 `tests/test_request_control.py`。
- `nanovllm/engine/scheduler.py`：
  - 控制锁 `_control_lock`（`threading.RLock`）：保护控制面状态与账本变更的线性化；`request_cancel()` 为 signal-only 入口（可在 forward 期间调用，只置位标记），`cancel()`/`timeout()` 为安全点收尾入口；`schedule()`/`postprocess()` 全程持锁（纯 CPU 记账安全点），锁不跨越 GPU forward（§3.4）。
  - 抢占事务（§4.2.2）：`_pick_victim()` 按队尾优先反向搜索，排除已终态/非 running/本轮已接纳 item/有本轮计划（`num_scheduled_tokens > 0`）的对象，`tried` 集合保证有限尝试；`_preempt_victim()` 完成 RUNNING→PREEMPTED→（同轮自动路径）resume 至 waiting 队首，一次抢占一条事件记录。
  - decode 侧 KV 不足：抢占 victim 让块后重查 `can_append`；**无合法 victim 时延后候选**（保留 RUNNING/KV 回队首原位，记 `kv_capacity` + 队首 HOL 归因），取代 Day9 的"自抢占"（自抢占不释放新容量，只损失有效 KV 进度）。
  - prefill 侧 KV 不足（§5.3）：先抢占合法 victim、对同一候选重查 `can_allocate`；waiting 队首插入造成的扫描索引偏移以 `scan += 1` 补偿；释放后仍不足则按 `kv_capacity` 停止，不破坏队列。
  - 异常收尾（§4.2.3）：`abort_round(items, reason)` / `abort_all_active(reason)`，均带对象所有权保护（旧批次迟到收尾不影响复用 ID 的新对象）、幂等可重入，返回资源快照。
  - `request_control` 事件（§6.3）：cancel/timeout/preempt/resume/abort 五类动作，字段含 `round_id/seq_id/request_id/action/from_status/to_status/reason/num_preempts/released_blocks/free_blocks/used_blocks/observed_at`；`released_blocks` 为 used 账本实际差值；`postprocess` 的取消/超时分支改走统一 `cancel()`/`timeout()` 入口使事件同源。
- `nanovllm/engine/llm_engine.py`：`step()` 事务化（§4.3）——外层 `_in_step` 作用域（控制锁内翻转，与 `cancel_request` 的空闲判定共用锁序，消除竞窗）；内层 `try: run + postprocess / except BaseException: error 事件 + abort_round + abort_all_active + 锁定 + raise / finally: 清除本轮临时计划与 `_current_round_id``。`cancel_request()` 新语义：先 signal-only，Engine 空闲时在控制锁内复查 `_in_step` 后立即安全点收尾，step 期间仅保留信号交由 postprocess/边界扫描收尾。`exit()` 幂等化（多引擎验收脚本需要）。
- `nanovllm/engine/block_manager.py`：新增 `check_ledger()`（§4.5）——free/used 互斥、总数守恒、used 块 ref_count>0、free 块 ref_count==0，显式 raise（不依赖会被 `python -O` 剥离的 assert）。
- `nanovllm/engine/model_runner.py`：无需改动——执行期间不检查/修改 Sequence 的边界（§4.6）由 Engine/Scheduler 安全点保证；decode 成功 + prefill 失败按整轮异常处理由 `step()` 的统一 except 覆盖。
- 测试：新增 `tests/test_request_control.py`（58 项，§9.1 全矩阵）；适配既有测试（4 个测试文件注入固定时钟 fixture、随机交错不变量接受"无 victim 延后"合法结果、2 个 Day9 KV 不足用例按新语义重写并更名）。
- 脚本：新增 `scripts/validate_request_control.py`（CPU 5 场景 + GPU 6 组，事件白名单、状态迁移重放、抢占/恢复配对、账本快照交叉核对）。

## 2. 环境与模型可用性

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D（24 GB） |
| torch / CUDA | 2.5.1+cu124 / 12.4 |
| 模型 | Qwen3-0.6B，本地目录 `/root/huggingface/Qwen3-0.6B` |
| 引擎配置（GPU 常规组） | TP=1，enforce_eager=True，kvcache_block_size=256，B=4096，max_num_seqs=64/8，chunk_size=1024，KV 池 668 块（u=0.85） |
| 引擎配置（GPU 小池组） | 同上，KV 池 16 块（u=0.092，自适应档位构造） |
| 结论 | GPU 与模型均可用，§5 GPU 验证已实际执行 |

## 3. 设计取舍说明

1. **无合法 victim 时"延后"取代 Day9"自抢占"**（§4.2.2 规则 5）：decode 候选块不足且 running 中无合法 victim 时，候选保留 RUNNING 与已提交 KV 回到队首原位，记录 `kv_capacity` 决策；不再做"抢占自己再恢复"的无谓重算（不释放任何新容量）。原语义的两个 Day9 用例（`test_mixed_batch.py::test_kv_shortage_decode_defers_prefill_stops`、`test_token_budget.py::test_kv_shortage_preempts_with_kv_reason_not_budget`）按新语义重写：断言 `num_preempts == 0`、状态保持 RUNNING、block_table 保留。随机交错测试的 decode-first 不变量扩展为三种合法结果（decode 接纳 / 抢占让块 / 无 victim 延后且必须带 `kv_capacity` 归因）。
2. **prefill 侧抢占的扫描索引补偿**：victim 恢复插入 waiting 队首会使 FCFS 扫描索引整体后移一位，每次插入以 `scan += 1` 补偿，保证"同一候选重新查询容量并继续调度"（§5.3）与每请求每轮至多一个 chunk 的约束同时成立。
3. **时钟注入**：既有测试在虚拟时间轴上注入小数值 `now`，与"deadline 不早于 created_at"的构造校验冲突，故提供 `Sequence.clock` 注入点（默认 perf_counter）；4 个既有测试文件以 autouse fixture 注入固定时钟并在用例结束恢复，`test_request_control.py` 使用可推进的可控时钟。
4. **序列化不升版**：Day10 未新增进 payload 的字段，v3 协议原样满足"rank 0 控制面权威"要求；测试覆盖 v1 元组、v2 字段名映射（`num_cached_tokens`→`prefill_offset`）、v3 往返（含 `cancel_requested`/`status`/`is_prefill`）与未知版本拒绝。
5. **`postprocess` 取消/超时分支统一入口化**：直接调用 `Scheduler.cancel()`/`timeout()`，与调度边界清理共享终态迁移、收尾和 `request_control` 事件发射，避免双路径漂移。
6. **抢占 victim 决策的 phase 标签**：victim 均为 RUNNING（decode 候选）请求，无论在 decode 阶段还是 prefill 阶段被抢占，决策 `phase` 一律记 `"decode"`（与其队列身份一致）。
7. **验收脚本的抢占压力构造**：请求 prompt 首 token 互异以阻断跨请求 prefix 共享（相同 prompt 会使后续请求折叠为 0~1 块需求，压力失真——CPU/GPU 第一版证据均出现过该现象）；GPU 小池通过自适应 `gpu_memory_utilization` 档位（本机标定 u=0.092→16 块）构造，跨环境由目标区间兜底重试。

## 4. CPU 测试（实际命令与结果）

```text
python -m pytest -q                                    # 377 passed（含 74 项 Day10 控制测试）
python -O -m pytest -q                                 # 377 passed
python -m pytest tests/test_request_control.py -q      # 74 passed
PYTHONPATH=. python scripts/validate_request_control.py --mode cpu \
    --output docs/evidence/day10/day10-cpu-final.jsonl # PASS（exit 0）

HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/validate_request_control.py --mode gpu \
    --model /root/huggingface/Qwen3-0.6B \
    --output docs/evidence/day10/day10-gpu-final.jsonl   # PASS（exit 0）
```

CPU 验收脚本场景结果（事件流见 `day10-cpu-final.jsonl`，均通过独立快照与事件重放交叉核对）：

| 场景 | 关键数字 |
| --- | --- |
| signal_vs_safepoint | 2 次 forward 期间取消（decode 与 prefill chunk 各一），安全点收尾 CANCELLED、原因 `forward_cancel` 保留，正常请求不受影响 |
| deadline_priority | waiting/running 各 1 例 TIMEOUT；取消+过期 deadline 并存时 CANCELLED 优先（`client_cancelled` 未被改写） |
| preempt_resume | 11 轮；1 次抢占 + 1 次恢复配对；完成后 completion 计数 6/6/4 与 max_tokens 逐一相等（不丢不重） |
| exception_cleanup | 3 条 abort 事件（`engine_error`）；`executed_tokens=null`；活动索引/队列/block 全部清零；失败后 step 拒绝重试 |
| 100_requests | 100 条全部终态（finished 85 + cancel 3 + timeout 12，账目平衡）；33 轮收敛；**自动抢占 5 次 + 显式抢占 1 次，均与恢复配对**；KV 峰值 16/16 块（真压力）、终态 used=0、free=池总量；每 10 轮 `check_ledger` 通过 |

## 5. GPU 验证（§9.4 矩阵，实际命令与结果）

```text
PYTHONPATH=. python scripts/validate_request_control.py --mode gpu \
    --model /root/huggingface/Qwen3-0.6B \
    --output docs/evidence/day10/day10-gpu-final.jsonl # PASS（exit 0）
```

| 组 | 关键数字与结论 |
| --- | --- |
| cancel-waiting | 16×512-token 请求首轮只接纳一半（B=4096），取消 6 条 waiting 请求（`queue_cancel`）；被取消请求未产生正常 completion；最终全部终态、账本归零 |
| cancel-running | 300-token victim 在 forward 期间被置位信号（signal-only 包装 `model_runner.run`），postprocess 安全点收尾 CANCELLED；同轮 3 条短请求正常完成且 completion 计数逐一正确（同轮隔离） |
| timeout | 4 条 +0.06s deadline 请求在步进中跨点，按 TIMEOUT 终止并回收资源；其余正常完成 |
| preempt-resume | 小池 16 块、10×512-token 请求：decode 首步跨块边界触发**调度器自动抢占 3 次**（队尾方向级联 victim），3 次恢复一一配对；16 轮收敛；终态账本归零 |
| exception-cleanup | 注入 runner 异常：整轮 abort + 全活动收尾，`executed_tokens=null`，Engine 锁定后拒绝重试，block 全部释放 |
| 100-requests-gpu | 100 条互异 prompt（32–384 token，max_tokens 2–12，10% 短 deadline）+ 随机取消/超时/显式抢占：129 轮收敛；终态账目平衡（finished 84 + cancel 6 + timeout 10 = 100）；显式抢占 2 次均配对恢复；KV 峰值 12/668 块、尾窗口无增长、终态 used=0 |

显存观测（`gpu_memory` 事件）：常规组引擎分配 19,869 MiB / 峰值 19,883 MiB；各组结束均正常释放（引擎 exit 幂等）。性能数字仅作观测，不设改善阈值。

## 6. 验收清单逐项核对（对照证据，设计清单同步见 `request-cancellation-preemption.md` §10）

> 本轮修复后重新执行 CPU/GPU 验收；CPU 专项测试扩展为 74 项，完整回归以最终命令输出为准。事件重放器已增加 per-request 历史、身份、资源总量和抢占栈校验，并允许未记录的内部 WAITING→RUNNING 迁移。

| 设计 §10 条目（摘要） | 状态 | 证据 |
| --- | --- | --- |
| waiting/running/PREEMPTED 均可取消，重复取消幂等、首次原因保留 | 通过 | `test_control_signal_*`；CPU signal 场景；事件重放 first-reason 检查 |
| deadline 单调时钟；调度边界与 postprocess 均能终止 | 通过 | `test_deadline_*`；CPU deadline_priority；GPU timeout 组 |
| 取消 > 超时 > 正常提交；命中不追加 token、不推进 KV | 通过 | `test_priority_*`（提交前取消/超时均丢弃采样、offset 不前进） |
| KV 不足只选合法 running victim；释放后可恢复重新 prefill | 通过 | `test_victim_*`（队尾优先/排除规则/无 victim 延后/有限尝试）；CPU+GPU preempt-resume 组 |
| 抢占恢复后 prompt/completion/计数/序列不丢不重 | 通过 | `test_recompute_recovery_*`；CPU preempt_resume 场景 completion 计数逐请求核对 |
| 混合轮单 item 取消/超时/完成不影响其他 item 提交与采样对齐 | 通过 | `test_mixed_round_isolation_*`；CPU signal_vs_safepoint；GPU cancel-running |
| 模型/采样/postprocess 异常统一收尾、不半完成重试 | 通过 | `test_exception_cleanup_*`；CPU+GPU exception-cleanup 组（`executed_tokens=null`、Engine 锁定） |
| 异常后活动请求/索引/本轮计划清理 | 通过 | `assert_stable`（索引/队列/used 全零）+ `engine_abort_summary.final_snapshot` |
| 重复调用不 double free/重复入队/负 ref_count | 通过 | `test_finalize_repeat_no_double_free`、`test_abort_entries_idempotent`；100 请求负载每 10 轮 `check_ledger` |
| free/used 互斥且总数守恒；活动结束资源回稳定范围 | 通过 | `check_ledger` 全程（含 `python -O`，显式 raise）；`assert_stable` 终态 free=池总量 |
| 连续 ≥100 请求无死锁/泄漏/KV 持续增长 | 通过 | CPU 100 请求（33 轮、kv_peak=16=池、终态 used=0）；GPU 100 请求（129 轮、终态平衡） |
| Day9 BatchItem/混合预算/decode-first/round_id/逐 item 回归 | 通过 | 377 项全量回归含 `test_mixed_batch.py` 全部用例；随机交错不变量按 Day10 语义扩展 |
| pickle 旧版本读取 + 当前版本往返；rank 0 控制面语义 | 部分通过 | `test_serialization_*`（v1/v2/v3/未知版本/往返）验证 CPU 协议；TP>1 rank 0/worker 控制面端到端未测，见 `day10-review.md` |
| `python -m pytest -q` 与 `python -O -m pytest -q` 结果已记录 | 通过 | 本文 §4（377/377） |
| CPU 验收脚本与 GPU 原始 JSONL 已归档 | 通过 | `docs/evidence/day10/day10-cpu-final.jsonl`、`day10-gpu-final.jsonl`（修复后）及历史 `day10-gpu.jsonl` |
| 验收记录注明环境/命令/结果/未覆盖边界；索引已更新 | 通过 | 本文 + `docs/README.md` |
| 事件无明文；控制动作/round_id/资源快照可独立核对 | 通过（范围内） | 字段白名单、明文键扫描、per-request 历史/身份/抢占栈和资源前后差值均通过；轮外控制允许 `round_id=null`，未记录的内部迁移不由控制事件单独重建 |

## 7. 未覆盖边界

- **TP>1 端到端控制传播**：未实测（全部 GPU 验证 TP=1）；worker 侧控制面语义由 pickle 协议（v3 字段）与设计边界覆盖。CPU v1/v2/v3 兼容测试通过，但不替代 TP>1 端到端证据。
- **CUDA Graph 捕获路径**：未覆盖（GPU 验证统一 `enforce_eager=True`）；取消/超时不触碰执行期数据面的约束不依赖 eager 与否，但图捕获下的行为未验证。
- **真实 HTTP/SSE 客户端断连**：不在本日范围（Day11/13）；本日只提供 `request_cancel`/`cancel_request` 控制面。
- **真正并行服务线程压力**：未实测。"forward 期间取消"以单线程在执行边界内置位信号模拟（CPU 桩与 GPU `run()` 包装），`_in_step` 在控制锁内翻转/复查消除了"空闲立即收尾 vs 执行期仅置位"判定的竞窗；真实多线程下与 `step()` 启动竞争的线性化由锁保证，但未做并发压测。
- **事件完整状态重放**：当前脚本验证字段白名单、单事件迁移合法性、资源快照和抢占/恢复配对；轮外控制事件允许 `round_id=null`，尚未实现按 `seq_id` 维护完整历史状态的审计重放。
- **GPU 100 请求组的抢占水位**：该组负载相对 667 块池压力不足（KV 峰值 12 块，2 次抢占均为显式抢占）；高水位**自动**抢占证据由修复后独立小池 preempt-resume 组（16 块池、3 次自动抢占）与 CPU 100 请求（16 块池、5 次自动抢占）提供。
- **GPU timeout 组的时效依赖**：deadline 触发依赖真实步进耗时（0.06s > 单步耗时）；若环境显著加速，脚本会以"超时事件不足"报压力不足（设计允许的正常失败模式），本次运行实际触发 4 例。
- **CPU/NVMe swap、prefix 淘汰策略、抢占公平性/优先级**：设计明确不做。
- **小池引擎的自适应构造**：`gpu_memory_utilization` 档位基于本机标定（u=0.092→16 块），跨环境由目标区间（10–20 块）兜底逐档重试；未在其他 GPU 型号上验证。
