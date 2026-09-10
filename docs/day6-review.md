# Day 6 实现审查（2026-09-10）

对应设计：[request-lifecycle.md](./request-lifecycle.md)。测试汇总：[day6-validation.md](./day6-validation.md)。

## 最终收尾与提交记录

2026-09-10，用户授权结束 Day6 并在当前 `feature/request-lifecycle-state-machine` 分支提交。第三轮技术结论保持通过；本次将代码、79 项生命周期测试、既有测试调整、AGENTS.md 与设计/审查/验收文档一并纳入 `feat(engine): complete Day6 request lifecycle state machine` 提交。设计验收清单和索引已同步最终 **16/16** 状态，标准原文不变。提交说明覆盖状态迁移、资源与身份所有权、兼容性、99 项回归及 GPU 验证范围；没有合并分支或 push。

以下第三轮的 15/16、未提交，以及更早两轮的失败结论，均为当轮时间点的历史记录；不代表本次收尾仍有未解决技术阻塞。非阻塞建议和 TP>1/CUDA Graph/真实并发未实测边界继续保留，不因提交而扩展承诺。最终状态见 [验收记录](./day6-validation.md)；提交哈希可在 Day6 feature 分支的提交日志中查询。

## 第三轮独立复核（提交前历史）：技术验收通过，可进入 Day6 收尾

**2026-09-10 最新结论：第二轮 B/C 两项阻塞均已修复；本轮未发现新的 Day6 范围内阻塞性缺陷。** 正式回归 **99 passed**，真实 GPU **11 组检查通过**，补充混合操作 **3,000 次不变量快照检查通过**。按既定的同步调度、标记式取消、recompute 和协议 CPU 验证范围，Day6 技术验收通过；清单 **15/16 项通过**，唯一未交付项为正式提交说明。本轮不擅自提交，因此结论不是“16 项交付已全部完成”。

审查对象仍是 `feature/request-lifecycle-state-machine`、`HEAD=555ef43` 上的未提交工作区。新增修改集中在 `_finalize()` 的记录所有权保护、`generate()` 的无进展拒绝，以及对应 7 项正式回归。完整复跑此前正式测试和 GPU 边界，并补充跨状态组合检查；未修改功能代码或正式测试。第三轮只更新本报告和验收记录。

### 三轮 A. 第二轮阻塞的关闭依据

**二轮 B / R4：ID 复用后旧对象再次收尾——已关闭。**

- `nanovllm/engine/scheduler.py:99-104` 先判定 `requests.get(seq.seq_id) is seq`，仅当前记录所有者可删除索引及 ID 唯一性标记；旧对象重复后处理不再影响新所有者。
- 正式测试 `tests/test_request_lifecycle.py:1033-1068` 覆盖取消/正常完成后 ID 复用，再收到旧批次结果：B 身份保留、C 注册被拒绝、B 只能取消一次。
- 独立 CPU 额外检查 **6 组**：旧请求取消/完成 × 新请求 WAITING/RUNNING/PREEMPTED，每组重复旧后处理三次，身份、队列、token、block table、空闲块和引用计数不变。
- GPU 两种旧请求结束方式均实测：先让新所有者进入 RUNNING 并持有真实 KV，再重复旧后处理三次；新对象的 ID 与 block table 保持，第三次注册被拒绝，取消结果为 `(True, False)`。

**二轮 C / R5：健康暂停使同步 generate 忙循环——已关闭。**

- `nanovllm/engine/llm_engine.py:105-116` 在无 completion、零执行量且仍有活动请求时显式抛带暂停 ID 的 RuntimeError；不把 PREEMPTED 伪装为完成，不自动取消，也不无界空转。
- 这是二轮认可的“同步驱动显式拒绝无法推进状态”方案。调度入口先清理取消/超时/终态，因此最后请求正常清理产生的空 step 不会被误拒绝；正常 prefill/chunked prefill 有正工作量、decode 有负工作量，也不会因暂无 completion 被误拒绝。
- 正式测试 `tests/test_request_lifecycle.py:1116-1162` 覆盖健康暂停报错、暂停到期/取消标记正常化解、显式恢复后完成。
- 独立 CPU **9 组**检查包括普通/分块 prefill、空 prompts 下最后暂停请求到期/取消等；有界包装确认健康暂停在**第一个无进展 step**即拒绝，之后显式 resume 可重新推进。
- GPU 验证健康暂停立即拒绝且状态仍为 PREEMPTED，显式 resume 后跑满 max_tokens；暂停到期、取消两种场景可由 generate 清理并正常返回新请求输出。

R1/R2/R3/R6、N1/N2 的前轮关闭结论保持成立；本轮重新运行 79 项生命周期测试、20 项既有测试，GPU 复跑 R1 两种空批次及 R2 prefill/decode 返回前超时，未出现回归。

### 三轮 B. 实际执行结果

| 命令/检查 | 第三轮结果 |
| --- | --- |
| `python -m pytest tests/test_request_lifecycle.py -q` | 79 passed in 5.10s |
| `python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q` | 20 passed in 5.16s |
| `python -m pytest -q` | 99 passed in 5.13s |
| `python -O -m pytest tests/test_request_lifecycle.py -q` | 79 passed, 1 warning in 5.10s |
| `PYTHONPATH=/root/NanoServe-workspace HF_HUB_OFFLINE=1 python /tmp/day6_round3_gpu.py` | 11 组真实 GPU 检查通过，见下方明细。 |
| 独立内嵌 CPU 命令（真实 Scheduler/Engine，runner 桩） | 6 组 ID 代次/状态组合、9 组同步生成边界通过。 |
| 内嵌 CPU 混合操作检查 | 30 个固定随机 seed × 100 轮，共 3,000 次不变量快照通过；另 36 种状态迁移和 12 种 v2 往返通过。 |
| 独立 `python -O - <<'PY'`（显式 if/raise） | 非法终态 add 抛 ValueError，两个索引及队列无污染。 |

`python -O` 的警告为通用 `PytestConfigWarning`：非测试模块/插件中的 assert 被禁用。它不是定位到某个 KV 容量断言的运行故障；独立 if/raise 检查用于避免误以为优化模式下所有断言仍在执行。补充检查不计入 **99 项正式 pytest 数量**。

**GPU 环境与可复现参数：** `nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv` 实际输出 RTX 4090 D、driver 595.71.05、24564 MiB、测试前空闲 24082 MiB。模型与 GPU 均实际可用；权重目录仍为首轮 §4 所列本地 Qwen3-0.6B。TP=1、eager、上下文/批 token 上限 256、max_num_seqs=4、gpu_memory_utilization=0.3；采样 temperature=0、ignore_eos=True。退出时注销 atexit 后只退出一次，不修改基线 exit。

11 组 GPU 检查为：控制接口及直接 ID 冲突拒绝（1）、普通生成（1）、R1 deadline/取消（2）、R2 prefill/decode（2）、旧对象取消/完成后重复收尾且新所有者已 RUNNING（2）、健康暂停拒绝再 resume/暂停到期/暂停取消（3）。普通生成仍为：

- `'Hello, my name is'` → `[444, 2210, 13, 358]`，`' Lina. I'`。
- `'The capital of France is'` → `[12095, 13, 576, 6722]`，`' Paris. The capital'`。

输出顺序与 `list[dict(text, token_ids)]` 格式不变。R2 的本轮采样 token 丢弃、TIMEOUT 和资源回收正常；健康暂停恢复请求最终 completion 数为 4。

临时 GPU 脚本不是仓库交付物，CPU 关键关闭路径已经进入正式测试，可重复运行：

```bash
python -m pytest tests/test_request_lifecycle.py::TestStaleCleanupIdOwnership tests/test_request_lifecycle.py::TestGeneratePausedContract -q
```

该选择式命令作为后续复验入口；本轮实际已通过整个生命周期文件运行到这 7 项，不将选择式命令写成另一次已执行测试。

混合操作检查使用 30 个固定 seed、64 个物理块、最多 8 个活动请求、小 block size=8，每轮交错注册、取消、取消标记、到期、抢占、恢复、schedule/postprocess 和旧终态批次重复后处理；逐轮核对两个身份索引相等且唯一、队列无重复且与状态相符、暂停无 block、逐块引用计数等于实际持有数、空闲/占用集合互补。最终全部取消后每个 seed 均回收 64 块。该证据覆盖常规所有权不变量，不是对任意并发或恶意对象篡改的形式化证明。

### 三轮 C. 非阻塞建议及契约边界

1. **进度条异常收尾（低优先级）**：`llm_engine.py:95,112-116,128` 的 RuntimeError 路径跳过显式 `pbar.close()`。可用 try/finally 完善，属于终端进度显示清理，不是 KV 泄漏，不阻塞 Day6。
2. **失败调用不返回部分结果**：generate 可能先完成其他请求，之后才因健康暂停抛错；本次调用的局部 completion 不随异常返回。当前选择的是显式拒绝、无部分结果恢复接口，二轮未要求失败原子性。调用方应在同步 generate 前处理独立暂停请求；不能承诺异常后重试会自动重放已经完成的输出。
3. **防挂测试可加硬上限**：`tests/test_request_lifecycle.py:1116-1122` 等待 RuntimeError，但正式测试尚未限制 step 次数。后续可加计数上限，以便忙循环回归时快速失败；本轮独立检查已使用有界包装，当前实现会及时拒绝。

以上均为建议/已说明限制，不重新升级为本轮阻塞。TP>1、CUDA Graph、真实线程取消、混版本双向部署继续标为未实测；Day6 设计要求的协议 CPU 检查已完成，不能据此宣称多卡真机验证通过。基线超容量断言、随机数生成器历史存储和重复 exit 仍单独跟踪。

### 三轮 D. 交付与记录校正

- **验收：15/16 项通过，技术工作可收尾；正式提交说明未交付。** 本轮只审查和更新文档，不擅自提交、合并、push 或变更分支。最终交付闭环与技术通过须分开表述。
- 只更新用户指定的两份文档，不修改设计验收标准/清单；`request-lifecycle.md` 的 10/16 和 README 中的 67 项仍是首轮快照，当前结论以本节和最新验收记录为准。
- 修复记录所称“二轮 D 原样脚本重跑输出 B 标记保持、C 被拒绝”不准确：二轮脚本断言的是缺陷存在，修复后原样运行会在“ID 集合为空”的断言处失败，不会生成所声称的新输出。本轮证据为新的第三轮脚本及正式测试，不沿用该说法。
- `day6-review.md` 和 `day6-validation.md` 当前均为未追踪文件，不能声称其旧版本已经保存在 git 历史。本报告保留一、二轮审查正文作为历史证据；当前文件需要随正式交付一并纳入版本控制。
- 将“任何公开入口组合均正确”等无界承诺收窄为本轮已审查的同步生命周期与实际验证场景，不把有限测试写成穷尽证明。

---

## 第二轮独立复核（历史）：当时仍不满足 Day6 完结标准

> 以下为修复前第二轮快照，已被上方第三轮通过结论替代；其中两项 P2 和 13/16 计数仅指第二轮代码。

**最新结论：92 项正式测试全部通过，但仍有两项 P2 阻塞，不能确认 Day6 完结。** R1/R2/R3/R6、N1/N2 已修复；R4 的直接 ID 碰撞检查已修复，但“ID 复用 + 旧对象重复收尾”仍破坏唯一性；R5 的暂停扫描和完成判断已修复，但同步 `generate()` 无进展忙循环尚未解决。

本轮基于 `HEAD=555ef43` 之上的最新未提交工作区，分支仍为 `feature/request-lifecycle-state-machine`。只更新 `day6-review.md`、`day6-validation.md`，没有修改实现、正式测试或分支。以下“第二轮”代码行号指修复后的当前工作区；后文首轮记录的行号与复现结果均为历史证据，不能直接当作当前结果。

### 二轮 A. 原问题修复状态

| 编号 | 独立复核结果 | 当前依据 |
| --- | --- | --- |
| R1 | 已修复 | `llm_engine.py:70-76` 空批次返回 `([], 0)`；正式 CPU 测试与真实 GPU 的最后请求到期/取消两条路径通过。 |
| R2 | 已修复 | `scheduler.py:273-295` 在记账、追加 token 和完成判定前检查 deadline；CPU prefill/decode/chunked 测试通过，GPU prefill/decode 均丢弃过期 token 并释放资源。 |
| R3 | 已修复 | `scheduler.py:168-180,205-211` 调度前清理终态；控制入口对尚被持有的终态对象补清理。另补查 waiting/running/paused 的 7 种合法终态组合通过。 |
| R4 | **部分修复，仍阻塞** | `add()` 直接重复注册会失败，但旧对象重复 `_finalize()` 可误删另一代请求的 ID 标记，详见二轮 B。 |
| R5 | **部分修复，仍阻塞** | 活动索引覆盖暂停请求，取消/超时/完成判断正确；但健康暂停请求使同步 `generate()` 忙循环，详见二轮 C。 |
| R6 | 已修复 | `scheduler.py:35-51` 先验证状态/身份再写索引，非 WAITING 请求显式 ValueError；独立 `python -O` 检查失败后两个索引和队列均不变。 |
| N1 | 已修复 | `sequence.py:277-282` 拒绝版本 999，错误包含版本信息。 |
| N2 | 已修复 | `sequence.py:293-307` 根据旧 payload 类型恢复 decode 模式，再次 pickle 往返成功；12 种 v2 状态×模式检查也通过。 |

取消与超时同时命中的默认规则现为“已有终态保持不变；否则取消标记优先于 deadline”。调度入口与后处理一致；正式测试覆盖后处理优先级，二轮额外 CPU 检查覆盖调度入口优先级。该取舍符合单一状态机与终态不可改写原则。

### 二轮 B. P2：旧终态对象重复清理，破坏复用 ID 的新请求唯一性（R4 未闭环）

**位置：** `nanovllm/engine/scheduler.py:90-98`，尤其无条件 `_active_request_ids.discard(seq.request_id)`；公开触发入口为 `postprocess():273-287`。查重依赖该集合（`47-50`），Engine 查询/取消则扫描 `requests`（`llm_engine.py:56-68`）。

**触发与实际结果（两条路径均已 CPU 复现，不需要并发）：**

1. A 使用 ID=`same`，完成一次 `schedule()` 并保留批次。
2. 取消 A，或通过 `postprocess()` 使 A 正常完成；此时 A 已从活动索引移除。
3. 注册 B，按已公开策略复用 ID=`same`。
4. 对 A 的旧批次执行延迟后处理，或重复执行已完成批次的后处理。
5. `_finalize(A)` 清除 B 的活动 ID 标记；此时 `requests` 仍包含 B，但 `_active_request_ids` 为空。
6. 再注册 C（ID=`same`）居然成功；B/C 同时活动。连续两次 `cancel_request('same')` 返回 `True, True`，分别取消不同对象。

**为什么属于 Day6 阻塞：** 设计 §5 与验收清单要求重复清理幂等，不能影响其他活动请求；允许终态 ID 复用不等于允许旧对象删除新所有者的身份记录。这里并不是任意篡改对象，也没有要求实现线程安全或 HTTP，调用的是已有控制和后处理入口。原 R4 的直接查重测试不能覆盖此组合。

**完善标准：** 清理 ID 必须确认所有权/对象代次，不能只按字符串无条件删除；资源、seq_id 索引与外部 ID 记录的收尾应一致。补“取消后 ID 复用再收到旧结果”和“正常完成后 ID 复用再重复后处理”两条正式回归，断言 B 的记录不变、C 仍被拒绝、B 只能被取消一次。现有 `tests/test_request_lifecycle.py:835-843` 只测试 ID 可以复用，没有测试复用后的重复清理。

### 二轮 C. P2：健康暂停请求使同步 generate 永久忙循环（R5 未闭环）

**位置：** `scheduler.py:27-31,216-218` 与 `llm_engine.py:70-76,102-115`。

**CPU 实际复现：** A 通过真实 `Engine.step()`（仅 runner 注入 token）进入 RUNNING，再由公开 `preempt(A)` 暂停，不设置 deadline。随后调用 `generate([[3, 4]], max_tokens=1, use_tqdm=False)`。新请求正常完成，但 A 仍为 PREEMPTED；`is_finished()` 一直为 False，`step()` 一直返回 `([], 0)`，`generate()` 无等待、恢复、错误或退出分支。复现包装器计数到 **100 次连续空 step** 后主动抛出专用异常截断，避免测试真的挂住；期间没有模型工作能使 A 推进。

**为什么不能归为“预期语义”：** 暂停请求不是终态、单步返回空批次都正确，但同步循环持续消耗 CPU 且永不返回，并不是可用的暂停契约。首轮 R5 已明确提醒不要仅修改完成判断而引入空转。常规调度内部立即 resume 的抢占路径不受影响，但独立 preempt 是本日明确提供并测试的入口。

**完善标准：** 为同步驱动定义“存在活动请求但当前不能推进”的处理，可选择显式拒绝该状态或受控恢复/等待；不能把暂停请求伪装成已完成，也不能无界忙循环。补调用真实 `generate()` 的有界回归，覆盖健康暂停、暂停后到期/取消、显式恢复，不仅测试 Scheduler 单独方法。

### 二轮 D. 实际测试与证据

环境探测：`nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv` 输出 RTX 4090 D、driver 595.71.05、总显存 24564 MiB、测试前空闲 24082 MiB。GPU 和本地 Qwen3-0.6B 权重实际可用。

| 本轮实际命令 | 结果 |
| --- | --- |
| `python -m pytest tests/test_request_lifecycle.py -q` | 72 passed in 5.63s |
| `python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q` | 20 passed in 5.73s |
| `python -m pytest -q` | 92 passed in 6.02s |
| `python -O -m pytest tests/test_request_lifecycle.py -q` | 72 passed, 1 warning in 6.10s；警告说明非测试模块 assert 在优化模式下被禁用。 |
| `PYTHONPATH=/root/NanoServe-workspace HF_HUB_OFFLINE=1 python /tmp/day6_round2_gpu.py` | 6 组检查通过：控制/重复 ID 拒绝、普通生成、R1 两种空批次、R2 prefill/decode 超时。 |
| `PYTHONPATH=/root/NanoServe-workspace python /tmp/day6_round2_combinations.py` | 成功复现 R4 两条组合缺陷及 R5 的 100 次空循环；不是验收通过。 |

GPU 脚本使用首轮 §4 的同一权重目录，TP=1、eager、上下文/批 token 256、最多 4 条请求、显存比例 0.3。普通生成两条 prompt 输出 token_ids 仍分别为 `[444, 2210, 13, 358]` 与 `[12095, 13, 576, 6722]`，格式与顺序不变。R2 复测在实际模型调用前将已调度请求 deadline 置为过期，再执行真实 GPU 模型和后处理；prefill/decode 均进入 TIMEOUT/deadline_exceeded，token 未追加、block 已回收。退出时注销 atexit 注册再仅退出一次，未修改基线 exit。

额外 CPU 检查：7 种队列位置×合法终态清理、调度时取消优先级、已缓存部分 prompt 的 WAITING 取消与重复清理均通过；`python -O` 的独立检查使用显式 if/raise（不用 assert）验证 R6 拒绝与零污染；12 种 v2 pickle 往返通过。这些临时检查不计入 92 项正式测试。

**记录更正：** 修复后的旧验收记录声称“审查报告 §4 的 CPU 复现脚本原样重跑，R1–R6、N1/N2 全部输出 fixed: True”，无法由原脚本支持：首轮内嵌代码不含 N1/N2，也没有 `fixed: True` 输出，R4 的预期拒绝还会使其提前抛 ValueError。不能沿用该项作本轮证据。本轮采用以上新脚本和正式测试，以实际输出为准。

临时脚本不是仓库交付物；以下保留组合缺陷核心代码，放入仓库根目录 `python - <<'PY' ... PY` 即可执行：

```python
from types import SimpleNamespace
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus as S
from nanovllm.sampling_params import SamplingParams
Sequence.block_size = 8

def engine():
    e = LLMEngine.__new__(LLMEngine)
    e.scheduler = Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=64,
        eos=999999, kvcache_block_size=8, num_kvcache_blocks=8))
    e.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens))
    e.model_runner = SimpleNamespace(call=lambda method, batch, prefill: [10] * len(batch))
    return e

for ending in ('cancel', 'finish'):
    e = engine()
    rid = e.add_request([1, 2], SamplingParams(max_tokens=1, ignore_eos=True), request_id='same')
    batch, prefill = e.scheduler.schedule()
    if ending == 'cancel':
        e.cancel_request(rid)
    else:
        e.scheduler.postprocess(batch, [10], prefill)
    e.add_request([3, 4], SamplingParams(), request_id='same')
    e.scheduler.postprocess(batch, [10], prefill)
    assert e.scheduler.requests and not e.scheduler._active_request_ids
    e.add_request([5, 6], SamplingParams(), request_id='same')
    print('R4', ending, len(e.scheduler.requests),
          e.cancel_request('same'), e.cancel_request('same'))

e = engine()
rid = e.add_request([1, 2], SamplingParams(max_tokens=8, ignore_eos=True))
s = e.get_request(rid)
e.step()
e.scheduler.preempt(s)
real_step = e.step
empty_steps = 0
class StopProbe(Exception):
    pass

def bounded_step():
    global empty_steps
    result = real_step()
    if result == ([], 0):
        empty_steps += 1
        if empty_steps == 100:
            raise StopProbe()
    return result

e.step = bounded_step
try:
    e.generate([[3, 4]], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
except StopProbe:
    print('R5', empty_steps, s.status.name, e.is_finished())
```

### 二轮 E. 验收与完结判断

逐条核对设计第 9 节，当前可支持 **13/16 项通过**：未通过项为“重复完成/取消/清理”“可追踪 request ID”，以及未提交导致的“提交说明”。终态清理、TIMEOUT 原因、暂停非终态语义现已正确；序列化项按 Day6 要求的协议 CPU 验证范围通过，**不表示 TP>1 已实测**。

R5 的同步忙循环是第 4.3 节驱动契约和首轮修复要求中的独立阻塞，不能因粗粒度复选框没有单列此项而忽略。**即使暂不要求提交，当前代码也不满足 Day6 完结标准。** 两项 P2 修复并补组合回归后，才可重新判断技术验收；未提交本身应与代码缺陷分开记录。

本轮只更新用户指定的两份记录，不重新修改设计标准或设计清单。`request-lifecycle.md` 第 9 节的 10/16 与 `docs/README.md` 的 67 项是首轮快照，当前状态以本节和最新验收记录为准。

仍未实测：TP>1/NCCL/共享内存端到端、CUDA Graph、真实并发取消及新旧进程混部署。它们不是已发现的新增错误，也不把 TP=1 或 pickle 测试扩展解释成通过。基线容量断言、随机数生成器历史存储、重复 exit 问题仍单独跟踪。

---

## 以下为首轮历史记录（已被第二轮最新结论替代）

保留当时的 67 项测试、缺陷复现和 10/16 清单映射，供对照修复过程；其中“当前”“必须修复”等表述仅指首轮代码快照。

## 1. 结论与范围

**当前实现暂不通过整体验收。** 已有 67 项测试全部通过，普通 TP=1 GPU 生成与取消控制接口的冒烟验证通过，但补充审查复现了 6 项需要修复的生命周期缺陷、2 项低优先级序列化边界问题。不能把“已有测试全绿”等同于“状态、队列、资源不变量已全部满足”。

审查基于 `feature/request-lifecycle-state-machine` 分支、`HEAD=555ef43` 之上的未提交工作区。完整阅读设计、验收记录、状态机、Scheduler、BlockManager、Engine、ModelRunner 和相关测试，并对照基线 diff 区分新增问题与既有限制。本轮只修改文档，不修复实现，不提交、合并或切换分支。

本次按用户明确要求更新设计文档第 9 节的勾选状态及核对依据，**不删除、改写或降低验收标准原文**。此为对 AGENTS.md 中“完成情况只记验收记录”的本次任务特例。

严重性：P1 为应优先修复的正常调用路径故障；P2 为必须补齐的生命周期/接口正确性问题；P3 为协议健壮性改进。以下行号均指本次审查时的工作区。

## 2. 必须修复的发现

### R1 · P1：最后一个请求在调度边界结束后，Engine 仍执行空批次

- 位置：`nanovllm/engine/llm_engine.py:70-74`；下游异常点 `nanovllm/engine/model_runner.py:124-125`。
- 触发：只有一个活动请求，deadline 已过期，或通过 `Sequence.request_cancel()` 标记待取消。调用 `step()` 前 `is_finished()` 为 False；`schedule()` 清理最后一个请求后返回 `([], False)`。
- 实际：Engine 无条件调用 `ModelRunner.run([], False)`，decode 组装 block table 时抛 `ValueError: max() iterable argument is empty`。请求已清理，但驱动循环异常退出。
- 证据：CPU 使用真实 Engine/Scheduler/ModelRunner 方法、仅替换张量创建以避免 CUDA；两种触发均复现。TP=1 真实 GPU 路径也复现两种触发。
- 需求：设计 §4.3、§7.3–7.4、§8.2 的空队列不执行和引擎正常驱动至全部终态。
- 归属：旧 Engine 同样没有空批次保护，但本次 Scheduler 新增正常返回空批次的路径，属于新增集成回归。
- 修复要求：Engine 接受空批次契约，返回无输出、零工作量，不调用 ModelRunner；补 Engine 级测试，而非只断言 Scheduler 返回空列表。

### R2 · P1：模型执行返回后不检查 deadline，超时可被误报为正常完成

- 位置：`nanovllm/engine/scheduler.py:227-250`（对照入口检查 `162-165`）。
- 触发：调度时尚未超时，模型执行期间跨过 deadline；该轮采样 token 达到 `max_tokens` 或 EOS。
- 实际：`postprocess()` 只检查终态和取消标记，仍追加 token，迁移至 `FINISHED`，记录 `length/stop`。请求从活动索引删除后，下一轮 `check_deadlines()` 已无法纠正。
- 证据：CPU 固定时钟复现 `finished_at > deadline` 但状态为 `FINISHED`、原因为 `length`；GPU 在调度之后、模型返回之前让 deadline 过期，同样复现。
- 需求：设计 §6 第 5 步、§7.3–7.4 明确要求模型返回后的安全边界处理取消/超时。不要求中断 kernel，但不能把“安全边界检查”推迟到已输出正常 completion 之后。
- 归属：新增 deadline 功能不完整。
- 修复要求：后处理在追加 token、KV 记账、正常完成判定之前处理已过期 deadline，保留 TIMEOUT 原因并释放资源；补 prefill/decode、EOS/长度完成同轮超时测试，并固定取消与超时同时发生的优先规则。

### R3 · P2：Sequence 已进入终态时，调度入口没有兜底清理

- 位置：`nanovllm/engine/scheduler.py:91-97,103-107,162-198,203-222`。
- 触发：调用已有公开方法 `seq.mark_timeout()` / `mark_cancelled()` / `mark_finished()`，对象仍被 Scheduler 持有；不是测试中直接赋值伪造状态。设计要求调度边界检查终态。
- 实际：`cancel/timeout` 遇到终态立即返回，未清理队列和资源；`schedule()` 也未扫描终态。RUNNING 队列内的 TIMEOUT 请求会再次进入 decode。WAITING 队列内的 TIMEOUT 请求会先分配 block，随后在尝试迁移至 RUNNING 时抛 `InvalidStateTransition`，留下队列成员和已分配资源。
- 证据：CPU 分别复现上述 RUNNING 与 WAITING 两条路径。
- 需求：设计 §5 不变量 1、5、7 和 §7.3 的调度入口终态检查；仅有 `postprocess()` 清理不够，因为终态请求已经进入了模型批次，或在到达后处理前抛错。
- 归属：新增状态对象接口与 Scheduler 集成不完整。
- 修复要求：在调度/分配前识别终态并幂等收尾；重复终态控制入口可保持返回 False，但不能遗留尚未完成的清理。补所有终态在 waiting/running 中的边界测试。

### R4 · P2：自定义 request_id 可碰撞，取消不能唯一定位请求

- 位置：`nanovllm/engine/llm_engine.py:43-68`。
- 触发：两次 `add_request(..., request_id="same")`，或自定义 ID 与自动生成的 `req-N` 冲突。
- 实际：两次注册都成功，第二次返回的 ID 查询到第一条请求；连续两次 `cancel_request("same")` 都返回 True，并依次取消两个不同请求。不能满足外部 ID 可追踪和重复取消的接口语义。
- 证据：CPU 复现两个不同 `seq_id` 共用一个外部 ID，连续两次取消作用于不同对象。
- 需求：设计 §2.1 稳定身份、§4.3 与第 9 节可追踪 request ID。内部 `seq_id` 去重不能替代外部 ID 唯一定位。
- 归属：新增自定义 ID 功能。
- 修复要求：注册前保证活动 request_id 唯一，包括自动 ID 与自定义 ID 交叉碰撞；拒绝重复时不得污染索引或队列。终态之后是否允许复用 ID 可另定策略，不必在 Day6 引入历史存储。

### R5 · P2：独立 PREEMPTED 请求被完成判断和边界扫描遗漏

- 位置：`nanovllm/engine/scheduler.py:23-24,110-138,140-158`。
- 触发：通过公开 `preempt(seq)` 暂停请求，暂不调用 `resume()`。这是实现明确支持的观察点，且已有测试使用。
- 实际：请求仍在 `requests` 中、`is_active=True`，但不在两个队列内；`is_finished()` 返回 True。`check_deadlines()` 和取消标记扫描只遍历工作队列，因此暂停请求的到期和待取消标记均被遗漏。
- 证据：CPU 复现暂停请求已过期且有取消标记，`schedule()` 仍返回空批次，状态保持 PREEMPTED，活动索引残留 1 项。
- 需求：PREEMPTED 为非终态，允许暂停期间取消/超时；设计 §3.2、§4.3 要求全部请求终态后才视为完成。
- 归属：本次从 preempt 拆出 resume 后新增的中间态覆盖缺口。`schedule()` 内立即 resume 的常规抢占路径不会暴露这一问题。
- 修复要求：基于活动索引或显式暂停集合执行生命周期扫描，并区分“本轮无可执行请求”和“所有请求已终态”；不要简单改变完成判断后让同步驱动陷入空转。补暂停后 timeout、取消标记、resume、完成判断测试。

### R6 · P2：add() 拒绝非法初始状态之前已写入活动索引

- 位置：`nanovllm/engine/scheduler.py:28-38`。
- 触发：将已通过 `mark_cancelled()` 进入终态的 Sequence 传给 `Scheduler.add()`，或传入其他非 WAITING 对象。
- 实际：先写 `requests`，随后 `_enqueue_waiting()` 的断言失败；调用虽然报错，索引却残留对象。若残留对象已终态，cancel/timeout 还会直接返回，无法通过该控制入口清除。`python -O` 会去除状态断言，进一步削弱入口保护。
- 证据：CPU 复现 add 抛 `AssertionError` 后 `len(requests)==1`，队列为空且 `is_finished()==True`。
- 需求：设计 §5 不变量 7、§4.2 入队前置条件，以及非法操作尽早显式报错的规则。
- 归属：新增活动索引及入队校验顺序问题。
- 修复要求：先验证状态、身份及所有权，再写索引/队列；使用不依赖优化模式的显式异常。补失败前后状态、索引、队列和资源快照完全一致的测试。

## 3. 协议边界与不应误判的问题

### N1 · P3：版本号没有参与反序列化校验

`nanovllm/engine/sequence.py:272-284` 只识别 `(任意值, dict)` 的结构；将正常 payload 的版本改为 `999` 仍被接受。建议显式分派支持版本，未知版本报清晰异常，并加测试。这不是当前正常 v2 路径已发生故障的证据。

### N2 · P3：旧 decode tuple 恢复为错误的 is_prefill

`nanovllm/engine/sequence.py:285-298` 将旧六元组一律设为 `is_prefill=True`。旧 decode payload `(3, 2, 2, 1, [0], 77)` 恢复后 `token_ids=[]`；再次 pickle 往返会以空列表作为 prefill payload，在 `last_token = token_ids[-1]` 处抛 `IndexError`。建议按 last_state 类型恢复模式，补 legacy decode 单测。

**边界说明**：当前 ModelRunner 通过独立 `is_prefill` 参数选择执行路径，worker 不再转发 Sequence。因此不把 N2 声称为“当前 TP 首次反序列化必然失败”。旧实现也没有为恢复对象提供完整二次转发能力。v2 版本化和 v1 单向读取兼容不等于任意新旧进程混部署兼容；应收窄代码中相关兼容承诺。

### 已核实正确的部分

- 六状态、终态集合、合法迁移表与设计图一致；额外枚举全部 36 种状态组合，非法迁移均抛异常且不改变状态。
- `started_at` 首次赋值、终态时间和原因、抢占次数与时间记录使用单调时钟；token_ids、prompt 长度与 completion 进度在 recompute 路径保留。
- Scheduler 没有直接散落赋值 `status`；常规 EOS/长度完成、Scheduler 取消/超时入口均统一收尾。
- 现有重复完成/取消/超时测试通过；补查重复 `_finalize` 不会 double free。
- 补查共享 prefix block 在取消一个持有者后引用计数仍为 1，最后持有者取消后回收且缓存 hash 保留；分块 prefill 的 WAITING 请求取消可回收全部已分配块。
- 重复 `preempt/resume` 会在非法迁移处显式拒绝，并不重复释放或入队。可接受“非法迁移显式失败且无副作用”的解释，不应宣称它们像终态方法一样返回 False；现有正式测试尚未覆盖此重复调用场景。
- v2 六状态 × prefill/decode 共 12 种 pickle 往返检查通过。worker 不执行 rank 0 采样，缺少采样参数、占位 seq_id、decode 不传全量 token_ids 不能单独作为当前 TP 缺陷。

### 既有或范围外限制

- 单请求超过 KV 池容量、自抢占后无可调度请求时的 `assert scheduled_seqs` 在基线已存在，不计为本次新引入缺陷；完整容量拒绝/抢占策略应单独跟踪。
- 真正的多线程取消、GPU kernel 执行中途的资源所有权保护尚未实现。`cancel_request()` 当前直接释放资源，**不是线程安全的异步取消入口**；仅 `request_cancel()` 的标记模式与安全边界处理可用作后续基础。Day6 不要求 HTTP，但不能将“本轮仅同步使用”写成“外部线程无法调用”。
- TP>1 真机、CUDA Graph、共享内存容量边界和新旧进程混合部署未实测。
- `ModelRunner.generators` 中 seeded 请求的随机数生成器生命周期、`LLMEngine.exit()` 重复调用问题均来自基线，不在本次修复范围。

## 4. 实测证据与复现方式

### 回归与环境

本轮实际执行：

```bash
python -m pytest tests/test_request_lifecycle.py -q
# 47 passed in 5.54s
python -m pytest tests/test_kv_cache_lifecycle.py tests/test_block_manager.py tests/test_sampler.py -q
# 20 passed in 4.46s
python -m pytest -q
# 67 passed in 5.92s
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
# NVIDIA GeForce RTX 4090 D, 595.71.05, 24564 MiB, 24082 MiB（测试前）
```

GPU 和模型均可用，权重使用本地目录：
`/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`。

临时审查脚本实际运行命令：

```bash
PYTHONPATH=/root/NanoServe-workspace python /tmp/day6_review_cpu.py
PYTHONPATH=/root/NanoServe-workspace HF_HUB_OFFLINE=1 python /tmp/day6_review_gpu.py
```

临时脚本不作为仓库交付物；下方保留核心复现代码供后续移入正式测试。CPU 脚本确认 10 个缺陷场景（R1 两种触发、R3 两种队列、R2/R4/R5/R6/N1/N2 各一）；**脚本正常退出表示成功复现缺陷，不表示验收通过**。另用 CPU 临时命令验证了共享前缀取消、分块取消/重复清理、重复抢占/恢复无副作用三组正向场景。它们未加入 pytest 统计，正式测试仍为 67 项。

### CPU 核心复现

在仓库根目录用 `python - <<'PY'` 包裹以下代码运行，无需 GPU/权重：

```python
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus as S
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

Sequence.block_size = 8

def make(max_tokens=8):
    q = Scheduler(SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=64,
        eos=999999, kvcache_block_size=8, num_kvcache_blocks=8))
    s = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=max_tokens, ignore_eos=True))
    q.add(s)
    return q, s

def run(q):
    batch, prefill = q.schedule()
    q.postprocess(batch, [10] * len(batch), prefill)

# R1：证明 Engine 把已清空的批次交给了 runner。
q, s = make()
s.deadline = perf_counter() - 1
e = LLMEngine.__new__(LLMEngine)
e.scheduler = q
def reject_empty(method, batch, prefill):
    assert batch, 'R1: Engine still calls runner with an empty batch'
e.model_runner = SimpleNamespace(call=reject_empty)
try:
    e.step()
except AssertionError as exc:
    print(exc)

# R2：执行返回时超过 deadline，仍记录正常完成。
q, s = make(max_tokens=1)
s.deadline = perf_counter() + 3600
batch, prefill = q.schedule()
with patch('nanovllm.engine.scheduler.perf_counter', return_value=s.deadline + 1), \
     patch('nanovllm.engine.sequence.perf_counter', return_value=s.deadline + 1):
    q.postprocess(batch, [10], prefill)
print('R2', s.status.name, s.finish_reason, s.finished_at > s.deadline)

# R3：公开状态方法产生的终态未被调度边界清除。
q, s = make()
run(q)
s.mark_timeout()
print('R3', q.timeout(s.seq_id), q.schedule()[0] == [s], s.block_table)

# R4：同一 request_id 取消两个对象。
e = LLMEngine.__new__(LLMEngine)
e.scheduler, _ = make()
for tokens in ([1], [2]):
    e.add_request(tokens, SamplingParams(), request_id='same')
print('R4', e.cancel_request('same'), e.cancel_request('same'))

# R5：暂停不是终态，但完成判断与扫描遗漏。
q, s = make()
run(q)
q.preempt(s)
s.deadline = perf_counter() - 1
s.request_cancel()
print('R5', q.is_finished(), q.check_deadlines(), q.schedule(), s.status.name)

# R6：非法 add 报错后仍污染索引。
q, s = make()
q.cancel(s.seq_id)
try:
    q.add(s)
except AssertionError:
    print('R6', len(q.requests), q.is_finished())
```

### GPU 验证参数与结果

模型：上述本地 Qwen3-0.6B；`enforce_eager=True`、`tensor_parallel_size=1`、`max_model_len=256`、`max_num_batched_tokens=256`、`max_num_seqs=4`、`gpu_memory_utilization=0.3`。采样使用 `temperature=0`、`ignore_eos=True`。

1. 添加请求后查 ID、取消成功、重复取消 False、终态查询 None：通过。
2. `generate()` 输入 `['Hello, my name is', 'The capital of France is']`、`max_tokens=4`：返回两个含 `text/token_ids` 的字典，输出顺序保持输入顺序。
   - 第一条 token_ids：`[444, 2210, 13, 358]`，text：`' Lina. I'`。
   - 第二条 token_ids：`[12095, 13, 576, 6722]`，text：`' Paris. The capital'`。
3. 添加最后一个请求后将 deadline 设为 `perf_counter()-1`，调用 `step()`：复现 R1 的 ValueError；改为 `seq.request_cancel()` 再调用 `step()`，同样复现。
4. `max_tokens=1` 请求完成 `scheduler.schedule()` 后设过期 deadline，调用真实 `model_runner.call('run', batch, prefill)` 和 `postprocess()`：复现 R2，状态仍为 FINISHED/length。

GPU 脚本通过 `atexit.unregister(engine.exit)` 后在 `finally` 中只调用一次 `engine.exit()`，避免基线重复退出问题影响审查；未修改引擎实现。TP=1 冒烟不能替代 TP>1 的真实通信与执行验证。

## 5. 验收映射与待补覆盖

设计第 9 节共 16 项：本轮 **10 项勾选通过、6 项保留未勾选**。

- 功能：六状态、迁移表、显式异常、统一入口、抢占进度保留可通过；“终态不再调度且释放”受 R1/R2/R3/R5 阻塞；“重复完成/取消/清理”在普通 Scheduler 路径成立，但外部 ID 碰撞导致 R4，暂不整体勾选。
- 代码与兼容性：Scheduler 集中迁移、Day2–5 回归通过；可追踪 ID 受 R4 阻塞；序列化项仅有 CPU 支持，N1/N2 待完善且 TP>1 未实测，保留未勾选；状态语义/日志字段项受 R2/R5 的 TIMEOUT 与非终态完成判断错误阻塞。
- 交付：设计索引、状态机测试与命令、注明环境和未覆盖边界的记录已齐备；当前未提交，也没有正式提交说明，“提交说明包含变更”不勾选，不为满足清单擅自提交。

修复后至少将 R1–R6 的确定性复现移入正式 CPU 回归，补 legacy decode/未知版本、暂停态超时与取消、终态调度前清理、共享前缀取消、非法入队原子性测试，再执行三条设计指定回归命令和 TP=1 最小验证。TP>1、CUDA Graph、真实并发需按可用环境分别记录，不得用单机 pickle 或同步取消测试替代。
