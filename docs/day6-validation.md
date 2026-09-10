# Day 6 验收记录

设计：[`request-lifecycle.md`](./request-lifecycle.md)；审查及历史证据：[`day6-review.md`](./day6-review.md)。当前分支：`feature/request-lifecycle-state-machine`，`HEAD=555ef43` 上的未提交工作区。

> **2026-09-10 Day6 收尾：技术验收与提交交付闭环，16/16 项通过。** 三轮独立审查发现的问题均已关闭；正式回归 **99 passed**、第三轮真实 GPU **11 组检查通过**。用户现已授权在当前 feature 分支提交，本记录及提交说明随 `feat(engine): complete Day6 request lifecycle state machine` 一并交付。收尾同步设计清单和文档索引，不修改功能实现；不合并、不 push。下文测试数字为实际第三轮结果，收尾提交前另复跑全量回归。

## 1. 实现与问题关闭状态

- `sequence.py`：六状态、集中合法迁移表、显式异常、生命周期字段和单调时钟、取消标记、幂等终态方法；v2 协议版本校验、v1 prefill/decode 单向读取。
- `scheduler.py`：统一队列与资源入口、活动 seq_id 索引及 request ID 唯一性集合；终态/取消/deadline 扫描覆盖暂停请求；模型返回后检查超时；非法 add 先验证；重复 `_finalize()` 按活动记录对象所有权清理身份。
- `llm_engine.py`：可追踪 ID、查询/取消代理、空批次返回 `([], 0)`、只收集 FINISHED completion；同步 generate 对仍有健康暂停请求的无进展状态显式拒绝。
- 正式测试：`tests/test_request_lifecycle.py` **79 项**，较第二轮新增 **7 项** ID 复用收尾/同步暂停组合回归；Day2–5 既有测试 **20 项**。KV 生命周期测试保留抢占历史断言。

| 问题 | 第三轮状态 | 关闭依据 |
| --- | --- | --- |
| R1 空批次调用 | 已关闭 | Engine 不把空批次交给模型，CPU 与 GPU 的到期/取消两种路径通过。 |
| R2 返回边界漏超时 | 已关闭 | 后处理先超时再记账/追加/正常完成；prefill/decode GPU 均丢弃 token 并回收 KV。 |
| R3 终态调度前清理 | 已关闭 | schedule 兜底清理 mark_* 终态，重复控制入口补清理；正式回归保持通过。 |
| R4 身份唯一性（含二轮 B） | 已关闭 | 直接碰撞被拒绝；旧对象仅在仍拥有活动记录时才能删除 ID 标记，复用新所有者不受重复收尾影响。 |
| R5 暂停生命周期（含二轮 C） | 已关闭 | 扫描与完成判断覆盖暂停请求；generate 无进展显式报错，不忙循环、不把暂停伪装成完成。 |
| R6 add 原子性 | 已关闭 | 状态/身份验证先于写入，显式 ValueError；独立优化模式检查无污染。 |
| N1 未知协议版本 | 已关闭 | 版本 999 显式拒绝，错误包含版本。 |
| N2 legacy decode 模式 | 已关闭 | 按 payload 类型恢复模式，旧对象再次 pickle 往返通过。 |

**二轮 B 的加固核对：** `scheduler.py:99-104` 使用 `requests.get(seq.seq_id) is seq` 判定所有权。正式两种结束方式测试、独立 CPU 6 组（旧取消/完成 × 新 WAITING/RUNNING/PREEMPTED）、GPU 新所有者已持有 KV 的两种结束方式均通过；重复旧后处理三次不改变新对象，第三次 ID 注册被拒绝，取消序列 `(True, False)`。

**二轮 C 的契约核对：** `llm_engine.py:105-116` 在第一个无进展 step 后拒绝，并列出暂停 request ID。已有终态/取消/过期暂停请求在 schedule 先被处理，不会误拒绝；普通和分块 prefill、decode 有非零工作量，也不会因尚未产生 completion 被误拒绝。健康暂停显式恢复后能继续完成。该显式拒绝方案属于第二轮认可的默认处理，并非正常生成回归。

## 2. 第三轮实际验证

### 环境

实际执行环境探测：

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

输出 NVIDIA GeForce RTX 4090 D、driver 595.71.05、24564 MiB、测试前空闲 24082 MiB。GPU 和 Qwen3-0.6B 权重均可用并实际加载。

权重目录：`/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`。

### 正式回归命令与结果

| 实际命令 | 结果 |
| --- | --- |
| `python -m pytest tests/test_request_lifecycle.py -q` | 79 passed in 5.10s |
| `python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q` | 20 passed in 5.16s |
| `python -m pytest -q`（第三轮） | 99 passed in 5.13s |
| `python -m pytest -q`（授权收尾、提交前重跑） | 99 passed in 6.09s |
| `python -O -m pytest tests/test_request_lifecycle.py -q` | 79 passed, 1 warning in 5.10s |

优化模式产生通用 `PytestConfigWarning`：非测试模块/插件中的 assert 不执行。这不是定位到某条 KV 容量断言的故障报告，也不能声称优化模式下所有生产断言仍有效。另实际执行独立 `python -O - <<'PY'`，不用 assert、使用显式 if/raise，验证终态 add 抛 ValueError 后 `requests/_active_request_ids/waiting/running` 仍为空，通过。

二轮新增路径已进入正式测试，可用以下命令选择性重复验证（本轮已通过整个生命周期文件执行到这些测试）：

```bash
python -m pytest tests/test_request_lifecycle.py::TestStaleCleanupIdOwnership tests/test_request_lifecycle.py::TestGeneratePausedContract -q
```

### 真实 GPU 检查：11 组通过

实际运行：

```bash
PYTHONPATH=/root/NanoServe-workspace HF_HUB_OFFLINE=1 python /tmp/day6_round3_gpu.py
```

参数：TP=1、`enforce_eager=True`、`gpu_memory_utilization=0.3`、`max_model_len=256`、`max_num_batched_tokens=256`、`max_num_seqs=4`；采样 `temperature=0`、`ignore_eos=True`。临时脚本不作为仓库交付物，以下保留完整场景与预期口径；对应 CPU 路径可通过正式测试复验。

| 场景 | 组数 | 实测结果 |
| --- | --- | --- |
| 添加/查询/直接 ID 冲突拒绝/取消/重复取消 | 1 | 通过，活动请求可唯一查询，重复取消 False。 |
| 普通 generate | 1 | 两条 prompt 的格式、顺序和 token 与前轮一致。 |
| R1 最后请求到期/取消 | 2 | `step()==([], 0)`，分别 TIMEOUT/CANCELLED，全部完成。 |
| R2 prefill/decode 返回前超时 | 2 | 本轮 token 不追加、TIMEOUT、KV 释放。 |
| 二轮 B 旧请求取消/完成后重复收尾 | 2 | B 已 RUNNING 并持有真实 KV；对旧 A 后处理三次，B 的 ID/block table 保持，C 被拒绝。 |
| 二轮 C 健康暂停拒绝再恢复、暂停到期、暂停取消 | 3 | 拒绝保持 PREEMPTED；恢复后完成，到期/取消路径正常返回其余请求结果。 |

普通 generate 使用 `max_tokens=4`，结果：

- 输入 `'Hello, my name is'`：text `' Lina. I'`，token_ids `[444, 2210, 13, 358]`。
- 输入 `'The capital of France is'`：text `' Paris. The capital'`，token_ids `[12095, 13, 576, 6722]`。

返回仍是按请求顺序的 `list[dict]`，字段 `text/token_ids` 不变。健康暂停恢复请求最终 completion 数为 4，与新请求各返回一项。脚本注销 atexit 后只调用一次 exit，避免基线重复退出问题；没有修改引擎实现。

### 补充 CPU 检查

使用真实 Scheduler/Engine、仅 runner 注入 token：

- 6 组旧对象收尾 × 新所有者状态组合通过，每组重复旧后处理三次。
- 9 组同步驱动边界通过，包括首个空 step 及时拒绝、显式恢复、暂停到期/取消、空 prompts、正常/分块 prefill；有界包装避免回归时测试挂死。
- 30 个固定随机 seed × 100 轮混合生命周期操作，**3,000 次不变量快照通过**。逐轮检查两个身份索引相符且唯一、队列无重复且符合状态、暂停无 block、真实持有数等于 block ref_count、free/used 集合互补；每个 seed 最后全部取消并回收 64 块。
- 全部 36 种状态迁移组合、12 种 v2 六状态×prefill/decode pickle 往返通过。

这些额外检查不计入 99 项正式 pytest，也不是对任意并发/对象字段篡改的穷尽证明。测试覆盖范围和混合检查参数见审查报告“三轮 B”。

## 3. 验收清单：16/16 项通过（含本次提交交付）

### 功能验收

- [x] `SequenceStatus` 包含六种计划要求的状态。
- [x] 状态迁移图、合法迁移表和代码规则一致。
- [x] 非法迁移有显式异常，错误信息可定位请求。
- [x] 取消、超时、抢占、恢复和正常完成均有统一入口。
- [x] 终态请求不再被调度，且 KV block 已释放。
- [x] 抢占恢复不丢失 prompt 进度和已生成 token。
- [x] 重复完成/取消/清理不会 double free 或重复入队。旧对象重复收尾不再破坏复用新对象的身份记录。

### 代码与兼容性验收

- [x] Scheduler 不再散落直接修改状态的代码，或每处都有明确理由。
- [x] `LLMEngine.add_request()` 能返回可追踪 request ID。直接碰撞和已验证的 ID 复用/旧后处理组合均保持活动 ID 唯一。
- [x] `Sequence` 的序列化/反序列化兼容张量并行路径。**按 Day6 要求的协议 CPU 验证范围通过**：v2 往返、版本拒绝、v1 两种模式读取；TP>1 真机未实测。
- [x] Day2–5 既有测试全部通过（20 passed）。
- [x] 文档中的状态语义与测试断言、日志字段一致。TIMEOUT 原因、取消优先、暂停非终态和同步无进展拒绝一致。

### 交付物验收

- [x] `docs/request-lifecycle.md` 已加入 `docs/README.md` 索引。
- [x] 新增状态机测试文件及可重复执行命令。
- [x] 测试记录注明 GPU/模型是否可用和未覆盖的边界。
- [x] 提交说明包含状态迁移、资源清理和兼容性变更。本次授权提交说明覆盖六态与统一入口、取消/超时安全边界、KV 与身份所有权、同步暂停拒绝、v2/v1 兼容、99 项回归和 GPU 验证边界。

**完结口径：** Day6 技术工作达到设计范围内的验收标准，代码、测试、规则和文档在 `feature/request-lifecycle-state-machine` 上一并提交。提交主题为 `feat(engine): complete Day6 request lifecycle state machine`；可在该分支用 `git log -1 --format=full` 核对，避免在提交内容中嵌入其自身哈希。未进行分支合并或远端发布。

本次收尾同步 `request-lifecycle.md` 的最终勾选及依据和 README 索引，验收标准原文不变。审查报告中的首轮 10/16、第二轮 13/16、第三轮提交前 15/16 均保留为历史快照；最终状态以本记录和设计清单为准。

## 4. 非阻塞建议与未覆盖边界

### 低优先级建议

- `llm_engine.py:112-116` 新增 RuntimeError 路径跳过显式 `pbar.close()`（正常路径在 128 行）。可用 try/finally 完善终端显示清理；这不是 KV 资源泄漏，不阻塞 Day6。
- generate 显式拒绝时，不返回本次调用此前已完成的局部结果，也不保证重试重放这些结果。当前没有部分结果恢复/失败原子性接口；同步调用前应先处理独立暂停请求。这是本次拒绝契约的限制，不再扩展成新需求。
- 正式健康暂停测试可加 step 次数上限，以便未来忙循环回归时快速失败。当前实现及时拒绝，本轮独立验证已有上限保护。

### 实测边界

- **TP>1 未实测**：单卡环境没有运行 NCCL/共享内存多进程端到端；CPU pickle 不能替代多卡实测。CUDA Graph 也未实测。
- **协议兼容**：支持新版 reader 读取旧六元组；不承诺任意新旧进程双向混部署，不把有限测试写成协议完备证明。
- **超时与并发**：deadline 在 schedule/postprocess 安全边界检查，不打断 kernel。`cancel_request()` 同步释放 block，不是线程安全异步接口；真实外部线程取消、HTTP/SSE 和流式部分结果未验证。
- **同步暂停**：健康暂停且无可执行工作时显式拒绝；外部可 resume/cancel/设置 deadline 后重新驱动。受控异步等待属于后续服务层。
- **历史记录**：终态后允许 ID 复用，旧对象清理已保护新所有者；服务层终态历史保留策略延期。
- **基线问题**：超 KV 容量/无可调度批次断言、seeded generator 存储生命周期、重复 exit 均在基线已存在，单独跟踪，不计为本次引入。

## 5. 审查记录真实性校正

- 二轮 D 原脚本用于断言缺陷存在；修复后原样执行会在 ID 集合相关断言失败，不会输出修复记录声称的“B 标记保持、C 被拒绝”。本轮没有将其“原样通过”列为证据，使用第三轮新脚本及正式测试。
- 三轮审查时两份记录尚未追踪，彼时旧版本不在 git 历史中。首轮、第二轮审查正文保存在 `day6-review.md` 历史部分；本次授权收尾将两份记录首次纳入版本控制，不追溯声称先前已提交。
- 优化模式警告是通用 assert 禁用提醒，不能专门归因为某一条基线断言。
- 移除“任何公开入口组合均成立”等无界承诺，结论限定在设计要求和本轮已审查/实测的同步生命周期范围。
