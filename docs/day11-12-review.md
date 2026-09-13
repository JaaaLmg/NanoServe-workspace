# Day 11–12 OpenAI API 实现审查报告

> 审查对象：`feature/openai-api-day11-12` 工作区中的未提交实现。
>
> 审查基线：Day10 合并提交 `7211dc5`；本次审查前工作区已有 Day11–12 实现、测试、验收脚本和证据，但尚未提交。本报告记录代码审查发现、修复、验证结果和仍然存在的边界；不代表已经提交或合并。
>
> 设计依据：[Day11–12 设计文档](./openai-api-day11-12.md)、[请求生命周期](./request-lifecycle.md)、[Day10 取消/超时/抢占](./request-cancellation-preemption.md)。

## 1. 审查结论

**主链路已达到 CPU/ASGI 和 TP=1、`enforce_eager=True` 的最小 HTTP 验收标准，但完整生产级验收仍为部分通过。** 当前主链路为：

```text
completion/chat schema
  → tokenizer encode/chat template
  → InternalRequest
  → RequestManager
  → 单一 EngineWorker
  → LLMEngine.step()
  → CompletedRequest/AbortedRequest
  → OpenAI JSON response
```

本轮代码审查发现并修复了 10 类问题，最重要的是 chat 上下文长度误算、submit/shutdown 竞态、完成记录异常悬挂和 ID 复用串请求。修复后实际结果：

- Day11–12 服务与配置测试：`117 passed`，1 条 Starlette/anyio 依赖弃用警告；
- 全量 CPU 回归：`495 passed`；
- `python -O -m pytest -q`：`495 passed`，2 条依赖/pytest 警告；
- CPU ASGI 验收脚本：`18/18`，失败 0；
- GPU 严格 HTTP 验收脚本：`8/8`，失败 0，模型为本地 Qwen3-0.6B、TP=1、`enforce_eager=True`。

这些数字不等价于所有 GPU/生产边界均已验证。真实 TCP 断连取消、Day13 SSE、TP>1、CUDA Graph、GPU 504 deadline、长时压力和 GPU 真实 OpenAI Client 仍未完成或未实测。

## 2. 审查范围与方法

审查了：

- `nanoserve/` 全部服务代码：配置、schema、路由、lifespan、worker、RequestManager、CPU 测试桩；
- `nanovllm/engine/completed_request.py`、`llm_engine.py`、`scheduler.py` 的完成记录适配；
- `pyproject.toml` 依赖、包发现和启动入口；
- `tests/test_service_schemas.py`、`test_chat_template.py`、`test_engine_worker.py`、`test_service_api.py`、`test_server_config.py`；
- `scripts/validate_openai_api.py` 与 `docs/evidence/day11-12/` 的 CPU/GPU JSONL；
- 既有 `docs/day11-12-validation.md` 与设计文档清单。

审查先运行了原有定向测试和 CPU 脚本：两者均通过（原代码报告为 104 项/17 项），但静态审查证明其中有测试未覆盖的逻辑缺陷。因此“测试全绿”没有直接作为验收结论，而是逐条构造回归用例后重新验证。

## 3. 发现的问题、影响与修复

### P0：chat 上下文长度使用错误对象（已修复）

**原问题**：`nanoserve/service.py` 在 `_normalize_template_output()` 得到 `token_ids` 后，仍使用 `len(template)` 检查上下文。对 `BatchEncoding/dict`，长度通常是字段数；对字符串，长度是字符数，不是 token 数。超长 chat 可能绕过 `max_model_len`。

**修复**：统一使用 `len(token_ids)`；规范化 `None`、非 token 序列、非整数 token 为请求侧 `InvalidRequestError(400)`。completion 编码异常也转换为明确 400。

**回归证据**：`tests/test_chat_template.py` 覆盖 list/dict/string 三种返回形态的预算拒绝和 malformed template；GPU 脚本使用 `"hello " * 5000`，避免把字符数误当 token 数，实际得到 `400 context_length_exceeded`。

### P1：RequestManager submit 与 shutdown 竞态（已修复）

**原问题**：pending 句柄登记后才调用 `worker.submit()`，shutdown 可插入停止/哨兵，导致命令无人消费、Future 永久悬挂；pending 快照也没有统一锁。

**修复**：登记、接收状态检查和 worker 命令入队统一放在 Manager 锁内；`pending_request_ids()`、resolve、fail、fail_all 均使用锁保护；worker 的 submit 与 shutdown 哨兵在状态锁内线性化；worker 已停止时显式抛 `ServiceDrainingError` 并回滚句柄。

**保留边界**：这保证了单进程线程模型下的命令排序，不提供跨进程 graceful handoff。

### P1：完成记录解码异常导致 Future 悬挂（已修复）

**原问题**：`resolve_completed()` 先从 pending 删除句柄，再调用 tokenizer.decode；decode 异常会让句柄消失但 Future 未完成，后续 fail-all 找不到它。

**修复**：decode 放入异常边界，失败时给对应 Future 设置 `EngineError`；worker 对每条完成/中止记录隔离消费异常，并在 pop 本身失败时仍执行统一 fail/teardown。

**回归证据**：`test_decode_failure_closes_future_as_engine_error`；worker 异常和关闭测试通过。

### P1：迟到记录可能完成 ID 复用后的新请求（已修复）

**原问题**：Manager 只按 `request_id` 匹配记录。旧记录在同 ID 新句柄注册后到达时，可能错误完成新请求。

**修复**：`RequestHandle` 保存 Engine admission 后绑定的 `seq_id`；worker 对真实 Engine 的 `add_request()` 返回兼容旧 API 的 request ID 时，通过 `get_request()` 读取 admission 后的 `seq_id`；resolve completed/aborted 同时校验 request ID 和 seq ID。旧记录被记录为迟到并丢弃。

**回归证据**：`test_late_record_cannot_resolve_reused_id`；既有 ID 复用测试继续通过。

### P1：lifespan 关闭超时后跨线程调用 Engine（已修复）

**原问题**：worker `join()` 超时后，lifespan 线程仍可能直接调用 `engine.exit()`，与 worker 的 `step()`/`exit()` 并发，违反单 Engine 所有者边界。

**修复**：worker 仍存活时，lifespan 不再调用 `engine.exit()`，只标记失败并记录无法安全回收；正常 worker 路径负责 exit；只有 worker 未启动或已结束且没有完成 exit 时才由 lifespan 做兜底。`engine.exit()` 失败不会被后续 stopped 状态覆盖。

**未覆盖**：没有真实 GPU forward 阻塞超过 drain timeout 的破坏性测试；该场景目前选择“报告失败而不跨线程清理”，后续需要进程级 supervisor/强制隔离方案。

### P1：worker 失败与健康/路由状态不一致（已修复）

**原问题**：worker 异常只停止接收 Manager，ServiceState 仍可能为 ready；健康检查和路由看到的状态不一致，短窗口可能接纳请求。

**修复**：worker 进入异常收尾前立即置 `FAILED`；`health_snapshot()` 同时检查 ServiceState、worker 存活和 `RUNNING` 状态；路由复用健康事实源；启动等待 worker 进入运行函数；`/v1/models` 也执行 ready 检查。

**回归证据**：`test_health_not_ok_after_engine_failure`、启动失败、draining/not-ready 测试。

### P2：CLI 忽略环境变量（已修复）

**原问题**：argparse 常量默认值遮蔽了 `NANOSERVE_HOST`、`NANOSERVE_PORT`、`NANOSERVE_ENFORCE_EAGER`、`NANOSERVE_TENSOR_PARALLEL_SIZE`。

**修复**：未显式提供的 CLI 参数使用 `ServerConfig.from_env()`；显式 CLI 值通过 `dataclasses.replace()` 覆盖；CLI 提供 `--model` 时仍保留其它环境变量。`max_request_seconds` 增加有限正数校验，空白 model 被拒绝。

**回归证据**：`tests/test_server_config.py` 9 项通过。

### P2：验收测试/脚本存在假阳性或证据不足（已修复/校正）

发现包括：

- chat dict/string 模板没有超限回归；
- “拒绝发生在 Engine admission 前”测试创建了未传入 builder 的 FakeEngine，断言恒真；
- CPU 验收脚本所谓多请求排队实际串行；
- CPU 脚本没有真正执行 cancel；
- GPU 脚本只检查任意 model ID、completion stream、部分 usage，漏掉 chat stream 和完整响应字段；
- GPU 超长 prompt 用字符数构造，可能不产生超长 token；
- fake cancel 丢失传入 reason。

本轮修复：增加真实 API admission spy、并发线程提交、双路由 stream 检查、GPU 精确 model/response/usage/context 检查、FakeEngine reason 保留，并重新生成 CPU/GPU JSONL。取消/timeout 的脚本化覆盖仍不完整，见未覆盖边界。

### P2：完成记录离线增长（已缓解）

`LLMEngine.generate()` 原本不消费服务完成记录，长期离线调用会让 deque 增长。现已在离线循环中 drain completed/aborted 记录；离线结果仍以原有 `step()` 返回值为准，保持兼容。

## 4. 已确认正确的设计部分

1. `CompletedRequest`/`AbortedRequest` 在 Scheduler `_finalize()` 所有权检查后捕获，usage、finish reason、request ID 同源；`pop_*()` 在控制锁内幂等 drain。
2. HTTP 层不直接访问 Sequence、block table、Scheduler 队列或 ModelRunner；EngineWorker 是单一 `step()` 所有者。
3. completion 与 chat 共用 `SamplingParams` 构造、上下文预算和 RequestManager；chat 使用 tokenizer `apply_chat_template(tokenize=True, add_generation_prompt=True)`。
4. Day10 cancel signal-only 和安全点收尾语义保持不变；服务层不在 GPU forward 中途清理资源。
5. OpenAI 响应 ID、`x-request-id`、usage 和 finish reason 由同一完成记录映射，不依赖已清理的 Sequence 反查。
6. FastAPI/Pydantic 输入错误统一转换为 OpenAI 风格 400；错误响应不回显 prompt/token 明文和堆栈。

## 5. 实际验证命令与结果

```bash
python -m pytest tests/test_service_schemas.py tests/test_chat_template.py \
    tests/test_engine_worker.py tests/test_service_api.py tests/test_server_config.py -q
# 117 passed，1 warning（Starlette/anyio 依赖弃用）

python -m pytest -q
# 495 passed，1 warning（Starlette/anyio 依赖弃用）

python -O -m pytest -q
# 495 passed，2 warnings（pytest -O 提示 + Starlette/anyio）

python scripts/validate_openai_api.py --mode cpu \
    --output docs/evidence/day11-12/cpu-validation-review.jsonl
# 18 项，失败 0
```

GPU 条件：RTX 4090 D 24GB，本地 `/root/huggingface/Qwen3-0.6B`，TP=1，`enforce_eager=True`。启动命令：

```bash
python -m nanoserve.server \
  --model /root/huggingface/Qwen3-0.6B --model-id Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --enforce-eager
```

等待 `/health` 成功后执行：

```bash
python scripts/validate_openai_api.py --mode gpu \
    --base-url http://127.0.0.1:8000 --model-id Qwen3-0.6B \
    --output docs/evidence/day11-12/gpu-validation-review.jsonl
# 8 项，失败 0
```

8 项包括：health、精确 model ID、completion 完整响应、chat 完整响应、两路由 stream 501、非法 role 400、真实 tokenizer 超上下文 400。第一次启动后立即运行脚本曾出现一次 `Connection refused`，原因是脚本早于 uvicorn 监听；等待 health 后重跑通过。该启动竞态不计为 API 通过证据。

## 6. 仍未覆盖或只能部分通过的边界

- **Day13**：SSE 增量事件、首 chunk/finish chunk/`[DONE]`、真实 TCP 客户端断连取消均未实现。
- **Day14**：Prometheus `/metrics`、TTFT/TPOT 指标和结构化请求日志尚未实现。
- **GPU 并发**：本轮严格 GPU 脚本的 8 项为串行 HTTP 冒烟；CPU 脚本已改为真实并发，但 GPU 真实并发与 mixed batch 的结构化 case 仍未纳入脚本。
- **GPU shutdown**：服务日志证明启动和退出流程执行过，但未形成“SIGTERM 命令 + 退出码 + shutdown 期间 health 503”的完整结构化 case。
- **GPU deadline/504**：底层 Day10 CPU 语义有测试，真实服务端到端 `max_request_seconds`/504 未测。
- **OpenAI Client**：CPU fake Engine + 真实 uvicorn 端口已调用 completion/chat；GPU 真实模型路径未用 OpenAI Python Client 单独执行。
- **TP>1、CUDA Graph**：未实测；当前 GPU 证据仅覆盖 TP=1、`enforce_eager=True`。
- **长期压力**：未做 10⁴ 级服务请求、长时内存和进程级恢复压测。
- **Future 硬超时**：`RequestManager.wait()` 仍是无 timeout 的阻塞等待；正常 Engine deadline 依赖 worker 继续到达安全点，若底层 forward 永久阻塞，服务线程仍可能等待。后续应设计可取消的 supervisor/future timeout，不在本任务中伪造已解决。
- **completed_records 生命周期**：离线 `generate()` 已 drain；如果调用者直接反复调用 `step()` 而不消费 `pop_*()`，记录仍会积累，这是公开契约允许调用方承担的控制面责任。

## 7. 代码结构与可读性建议

- 当前 `app.py`、`api.py` 和 `worker.py` 的状态字符串/枚举存在重复映射；后续可定义只读 lifecycle adapter，避免路由直接比较字符串。
- `LLMEngine` 目前通过 `scheduler._control_lock` 实现完成记录 drain；后续可在 Scheduler 提供正式 `pop_completed/pop_aborted` 方法，避免 Engine 依赖 Scheduler 私有锁/字段。
- `RequestManager` 仍同时承担句柄登记、异常类型和结果解码；若 Day13 增加增量事件，建议将完成/事件归一化拆成独立 ResultSink。
- `EngineWorker` 的 shutdown timeout 目前只能安全失败，不能回收仍卡住的线程；生产部署应在进程边界提供 supervisor，而不是增加线程强杀。
- `RequestManager.wait()` 的无 timeout 是明确的后续可靠性任务，不能用 daemon thread 或无界重试掩盖。

## 8. 最终评级

| 范围 | 结论 |
| --- | --- |
| CPU schema/worker/API 单元测试 | 通过：117 项新增/服务测试 |
| CPU ASGI 验收 | 通过：18/18 |
| GPU TP=1 eager 非流式最小 HTTP | 通过：8/8 |
| 完整 Day11–12 生产级验收 | 部分通过：并发 GPU、shutdown 结构化证据、504、长时压力仍缺 |
| Day13 SSE/断连 | 未实现，按设计留待后续 |
| Day14 metrics/observability | 未实现，按设计留待后续 |

本轮审查不提交、不合并、不 push；所有改动继续留在工作区供审阅。
