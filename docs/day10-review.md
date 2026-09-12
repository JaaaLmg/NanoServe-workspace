# Day 10 实现审查与验收复核

> 对照设计文档 [request-cancellation-preemption.md](./request-cancellation-preemption.md) 对当前未提交工作区进行代码、测试和证据复核。本文记录审查发现、修复结果和仍未覆盖的边界；不改变 Day10 设计目标或验收标准。

## 1. 审查范围与基线

- 分支：`feature/request-cancellation-preemption`
- 基线：`dev@36ae91c`（Day9 混合 Prefill/Decode 合并提交）
- 审查对象：`nanovllm/engine/sequence.py`、`scheduler.py`、`block_manager.py`、`llm_engine.py`、`model_runner.py`，Day10 测试、验收脚本、设计文档和验收记录。
- 当前状态：所有改动仍在工作区，未提交、未合并、未 push。

## 2. 首轮审查结论

Day10 主流程已实现并有 CPU/GPU 自验，但首轮审查发现以下问题不能被原有 361 项回归覆盖（修复后最终回归总数为 377 项）：

| 编号 | 严重性 | 问题 | 处理结果 |
| --- | --- | --- | --- |
| R1 | P1 | Engine 异常清理中某一步失败可能遮蔽原始异常，且阻断后续活动请求清理 | 已修复：清理步骤隔离、继续执行 `abort_all_active`，记录清理错误并保留原异常 |
| R2 | P1 | `preempt()`/`resume()` 对未注册对象缺少 Scheduler 所有权校验 | 已修复：副作用前校验活动索引；已注册但状态/队列非法仍抛 `InvalidStateTransition` |
| R3 | P1 | 注册、查询、取消之间存在控制面竞态和 request_id 查找—seq_id 操作 TOCTOU | 已修复：Scheduler 控制入口统一使用 `RLock`，新增按 `request_id` 原子查询；Engine 取消在同一锁内完成 |
| R4 | P1 | 直接调用 `timeout()` 可绕过已有取消信号，得到 TIMEOUT | 已修复：`timeout()` 检测 `cancel_requested` 后转入 cancel，统一保证 `CANCELLED > TIMEOUT` |
| R5 | P2 | `Sequence.mark_cancelled()` 直接调用时没有继承已记录的首次取消原因 | 已修复：优先使用 `cancel_reason` |
| R6 | P2 | `check_ledger()` 只检查 free ID 集合，漏检 free deque 重复/越界 ID | 已修复：增加重复、越界、覆盖完整性检查；关键 BlockManager 所有权检查改为显式异常，`python -O` 下仍生效 |
| R7 | P2 | `ModelRunner.run()` 子批异常时可能不执行 `reset_context()` | 已修复：decode/prefill 子批各自使用 `try/finally` |
| R8 | P2 | `LLMEngine.exit()` 的 runner 退出异常可能跳过 worker join 和引用释放 | 已修复：退出路径使用异常隔离和 `finally`，并尝试收尾 Scheduler 活动请求 |
| R9 | P2 | 默认生命周期时间入口存在注入时钟与裸 `perf_counter()` 分叉 | 已修复：`Sequence.transition_to()` 使用 `Sequence.clock()`，Scheduler 控制入口使用统一的真实单调时钟 provider；测试路径继续通过显式 `now` 注入，避免与外部绝对 deadline 冲突 |
| R10 | P2 | `engine_abort_summary` 新增字段未同步验收脚本白名单 | 已修复：`cleanup_errors` 纳入脚本字段白名单，CPU 验收重新通过 |
| R11 | P1 | `schedule()` 阶段异常可能在 allocate/may_append 后留下半完成资源 | 已修复：step 外层统一失败收尾；BlockManager allocate/deallocate 事务化；新增 schedule/may_append 半修改异常测试 |
| R12 | P1 | BlockManager 非法 free/block_table 在修改前未完整校验 | 已修复：校验 free/used、ID、重复项和 ref_count，失败恢复完整账本快照 |
| R13 | P2 | `resume()` 可能把同一请求放入 waiting/running 双队列 | 已修复：恢复前检查工作队列成员并拒绝重复 |
| R14 | P2 | 事件 replay 仅检查单事件，资源快照无法独立重算 | 已修复：按请求维护状态/身份/抢占栈，事件增加 before 账本快照并校验释放差值 |

## 3. 代码正确性复核

### 3.1 状态和控制面

- 六状态及合法迁移仍集中在 `Sequence`，没有新增旁路状态。
- `request_cancel()` 只设置取消信号，不触碰 block、队列和 token；`cancel()`/`timeout()` 在安全点执行终态收尾。
- `cancel()`、`timeout()`、`preempt()`、`resume()`、`add()`、`get_request()` 和调度/提交路径均受 Scheduler 控制锁保护。
- `timeout()` 对已有取消信号转入 `cancel()`，直接入口与 `schedule()`/`postprocess()` 的优先级一致。
- `_finalize()` 以活动索引中的对象身份作为所有权凭证；非 owner 直接返回，不再触碰队列或 block。
- 抢占和恢复对未注册对象在任何状态、队列或账本副作用前拒绝；自动 victim 仍只来自当前 running 队列。

### 3.2 抢占、恢复和进度

- victim 继续采用 running 队尾优先和有限尝试策略。
- 抢占只释放有效 KV block，保留 `token_ids`、prompt、completion 计数和采样参数；恢复通过 prefill recompute 建立有效 KV。
- 显式 `preempt()` 停留在 `PREEMPTED`；自动路径以相同的 `now` 完成 `resume()`，事件时间线一致。
- 取消/超时命中时不执行 `hash_blocks()`、不推进本轮 `prefill_offset`、不追加 token；同轮其它 item 继续提交。

### 3.3 异常与资源

- Engine 在 runner、采样或 postprocess 异常后设置失败锁，不允许在不完整 KV 上重试。
- 异常清理中的单项失败不会阻止其它对象清理；清理异常单独记录，原始执行异常保留。
- `ModelRunner` 每个子批均在 `finally` 中 reset context，避免 decode/prefill Context 泄漏。
- `BlockManager.check_ledger()` 现在检查 free deque 重复/越界、free/used 互斥、完整覆盖、ref_count 约束；分配/释放的关键前置检查不依赖 `assert`。
- 所有终态和异常路径仍通过统一释放入口，重复清理不会 double free。

## 4. 测试与实际结果

### 4.1 CPU 回归

实际执行命令：

```bash
python -m pytest tests/test_request_control.py -q
python -m pytest tests/test_request_control.py tests/test_request_lifecycle.py \
    tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_mixed_batch.py -q
python -m pytest -q
python -O -m pytest -q
PYTHONPATH=. python scripts/validate_request_control.py --mode cpu \
    --output docs/evidence/day10/day10-cpu-final.jsonl
```

结果：

- Day10 控制测试：`74 passed`
- Day10 + Day6–9 相关测试：`114 passed`
- 正常模式全量：`377 passed`
- `python -O` 全量：`377 passed`，1 个 pytest 关于优化模式断言的配置警告
- Day10 CPU 验收脚本：`PASS`
- 原始 CPU 证据：`docs/evidence/day10/day10-cpu-final.jsonl`

新增针对性测试覆盖：

- schedule/allocate/may_append 半修改异常统一触发 Engine 失败收尾；
- 非法 block_table、坏 free 队首和 resume 双队列被副作用前拒绝；
- 事件 replay 检查跨请求历史断裂、身份改写、资源总量和抢占栈；
- runner exit 异常仍完成 worker join 和 runner 引用释放；
- 直接 `timeout()` 在已有取消信号下保持 CANCELLED；
- 未注册对象的 `preempt()`/`resume()` 无副作用；
- 非 owner 迟到 `_finalize()` 不触碰新对象；
- free deque 重复和越界 block ID 被账本检查拒绝；
- 直接 `mark_cancelled()` 保留首次取消原因；
- 单对象异常清理失败时其余活动请求仍继续收尾；
- decode/prefill 子批异常均调用 `reset_context()`。

### 4.2 GPU 证据

修复后已重新运行 GPU 验收：

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/validate_request_control.py \
    --mode gpu --model /root/huggingface/Qwen3-0.6B \
    --output docs/evidence/day10/day10-gpu-final.jsonl
```

结果：`PASS`（exit 0）。环境为 RTX 4090 D、torch 2.5.1+cu124、Qwen3-0.6B、TP=1、`enforce_eager=True`。六组均通过：

- `cancel-waiting`：6 条 waiting 请求取消，终态 `active=0, used=0`；
- `cancel-running`：forward 期间 signal-only 取消，同轮其它请求正常提交；
- `timeout`：4 条请求按 deadline 进入 TIMEOUT，终态 `active=0, used=0`；
- `preempt-resume`：16 块小池观察到 3 次自动抢占和 3 次恢复，16 轮收敛，终态 `used=0`；
- `exception-cleanup`：异常事件 `executed_tokens=null`，失败 Engine 拒绝重试，资源释放；
- `100-requests-gpu`：100 条请求、129 轮收敛，`finished=84/cancel=6/timeout=10`，终态 `free=668, used=0`，`terminal_balance=true`。

原始修复后证据位于 `docs/evidence/day10/day10-gpu-final.jsonl`。TP>1、CUDA Graph、真实并发服务线程和 HTTP/SSE 仍未覆盖。

## 5. 验收清单结论

### 可以标记为通过

- waiting/running/PREEMPTED 取消、重复取消幂等和首次原因保留；
- 单调 deadline、调度边界和 postprocess 超时处理；
- `CANCELLED > TIMEOUT > 正常提交`；
- 合法 victim、释放 KV、恢复 recompute 和逻辑 token 进度保持；
- 混合轮逐 item 终态隔离和采样对齐；
- runner/采样/postprocess 异常的统一失败锁和清理路径；
- 正常/取消/超时/抢占/异常重复清理的资源账本安全；
- CPU 100 请求资源最终稳定；
- Day9 混合批次、预算、decode-first、round_id 回归；
- CPU pickle v1 legacy tuple、v2、v3 读取和当前版本往返；
- 正常与优化模式全量 CPU 回归；
- CPU 验收脚本及修复后 GPU 原始证据归档；
- 事件无 prompt/token 明文，字段白名单、per-request 历史、身份/抢占栈和资源前后差值检查通过。

### 只能部分通过或保留边界

1. **TP>1 控制面**：CPU pickle 协议和 rank 0/worker 采样职责契约通过；当前 v3 payload 传递状态、取消标记、`is_prefill`、`prefill_offset` 和生成进度，采样只在 rank 0 执行。但 TP>1 NCCL/shared-memory 端到端启动和取消传播未实测。
2. **真实并发取消**：forward 期间取消由 CPU/GPU 执行边界内的包装模拟，真实服务线程压力未测。
3. **CUDA Graph**：GPU 验证统一 `enforce_eager=True`，捕获路径未测。
4. **轮外控制事件**：外部 cancel/timeout/preempt/resume 的 `request_control.round_id` 可以为 `null`；`engine_abort_summary` 可关联失败 round，但单个 abort 事件不一定带 round。当前脚本已按 request 维护历史状态、身份、抢占栈和 before/after 账本差值；未记录的内部状态迁移仍不可能从控制事件单独重建。
5. **GPU 100 请求自动抢占**：已有 GPU 100 请求组证明整体终态和资源平衡，但其自动抢占压力不足；自动高水位抢占由独立 16 块小池组和 CPU 100 请求组覆盖。
6. **退出异常和 TP/CUDA 实际路径**：代码增加了防护和 CPU 桩测试，但未在真实多进程/真实 runner 退出异常环境验证。

## 6. 代码可读性与后续建议

- 当前控制入口通过 `RLock` 维持安全点线性化，保持了现有同步 API 兼容性；后续服务层应优先调用按 `request_id` 的原子控制方法，不直接暴露可写 `Sequence`。
- `LLMEngine.get_request()` 仍返回内部 `Sequence`，Day11 服务化时应转换为只读状态 DTO，避免上层绕过 Scheduler 修改 token、block 或 status。
- `Sequence.clock` 仍是类级注入点，适合当前 CPU 测试但多个 Engine 并行时共享；后续可迁移为实例级 clock provider。
- `validate_request_control.py` 已按 `seq_id` 维护事件状态历史，并使用栈配对多次抢占恢复；若未来要重建未记录的全部内部迁移，仍需补充 scheduler 事件。
- 底层资源保护已将 Day10 关键新增检查改为显式异常；其余历史 `Sequence.block()` 等断言属于既有输入边界，后续如要求全库 `-O` 防御性语义再单独收敛。

## 7. 结论

经修复后，Day10 的核心功能和 CPU 验收标准达到要求：控制面、抢占恢复、混合轮安全点、异常清理和 KV 账本均有代码与测试证据，正常/优化模式全量回归为 `377 passed`，CPU 验收脚本 PASS。

正式交付仍应保留 TP>1、CUDA Graph、真实并发服务线程、HTTP/SSE 断连和 GPU 100 请求自动抢占水位等未覆盖边界；设计清单已对 CPU/TP=1 可验证部分勾选，并在文字中保留端到端边界。