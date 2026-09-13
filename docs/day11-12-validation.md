# Day 11–12 OpenAI 兼容 HTTP API 验收记录

> 对应设计文档 [openai-api-day11-12.md](./openai-api-day11-12.md)。本文记录本轮审查后的实现范围、修复、实际执行命令与数字结果及未覆盖边界；设计文档 §9.3 验收清单已同步更新，本文 §7 区分 CPU/ASGI、GPU 最小 HTTP、手工观察和未测边界，供后续验收人复核。原始 JSONL 与服务日志证据位于 `docs/evidence/day11-12/`。

## 1. 实现范围与分支状态

- 分支：`feature/openai-api-day11-12`（基线 dev=`7211dc5`，含 Day10），**全部改动留在工作区，未提交、未合并**。
- `nanovllm/engine/completed_request.py`（新增）：`CompletedRequest`（正常完成记录，字段语义全项目统一为**只存 completion token IDs**，`prompt_tokens/completion_tokens/finish_reason/finished_at` 与终态迁移同源）与 `AbortedRequest`（取消/超时/异常收尾记录，不含 token 明文）。二者为 frozen dataclass，仅属 rank 0 控制面，不进 TP payload。
- `nanovllm/engine/scheduler.py`：`__init__` 增加 `completed_records`/`aborted_records` 两个 deque；`_finalize()` 在所有权检查通过后、资源清理前按 `seq.status` 捕获记录——所有权检查保证同一 seq 恰好记录一次，幂等重复收尾在检查处即被拦截。终态迁移与记录捕获同处一把控制锁内，request_id/finish_reason/usage 同源（§4.4）。
- `nanovllm/engine/llm_engine.py`：新增只读接口 `has_active_requests()`、`pop_completed()`、`pop_aborted()`（控制锁内 drain，重复调用返回空列表）与 `max_model_len` 属性。`step()` 的 `[(seq_id, completion_token_ids), num_tokens]` 返回格式与 `generate()` 行为**完全不变**；模块重导出 `CompletedRequest/AbortedRequest`，服务层无需感知 Scheduler 内部结构。
- `nanoserve/` 服务包（新增，模块职责与设计文档 §5.1 一致）：
  - `config.py`：`ServerConfig`（frozen dataclass），env（`NANOSERVE_*`）与 CLI 合并解析，显式校验 port/max_request_seconds/TP；`public_model_id` 默认取模型目录最后一段；不替代 `engine.Config`。
  - `schemas.py`：completion/chat 请求模型（`extra="forbid"`、`max_tokens` strict int 默认 **64** 全服务唯一口径、role 白名单 Literal、空白 prompt 拒绝）、OpenAI 风格响应模型（completion 带 `logprobs: null`，chat 不带）与错误模型。
  - `service.py`：`InternalRequest`（frozen，tuple token ids）、`build_sampling_params`（completion/chat 唯一映射入口）、`build_completion_request`/`build_chat_request`（编码/模板/上下文预算检查，全部发生在 `add_request` 之前）、错误类型层级（400/404/501/503/504/500 全覆盖）、`RequestManager`（pending 表 pop-once 语义：每个句柄恰好 resolve/fail 一次；未知/迟到记录记日志丢弃）。
  - `worker.py`：`EngineWorker`——唯一直接调用 `engine.step()` 的专用线程。命令队列（submit/shutdown 哨兵/消费哨兵）线性化；空闲期阻塞等待不忙轮询；活动期连续驱动 step 并逐轮消费完成记录；`cancel()` 直接调用 Day10 signal 入口（线程安全）并入队消费哨兵保证空闲期中止记录被及时排出；step 异常→fail_all+exit；空批次+仍有活动请求→按 Day10 无进展错误语义失败全部句柄；关闭路径：停止接收→失败排队命令→取消活动请求→有限 drain（10 s）→`engine.exit()`（幂等，异常记录且状态保持 FAILED）→兜底 fail_all。
  - `api.py`：`/health`、`/v1/models`、两个非流式 POST 路由；就绪→模型→stream→构建→提交的检查顺序；`x-request-id` 固定规则：成功响应 == body.id；已生成 ID 后失败 == 请求 ID；校验阶段为随机 `req-<hex>`。
  - `app.py`：`create_app(engine_factory=None, server_config=None)`；lifespan 顺序与设计文档 §5.2 一致；`ServiceState` 为生命周期单一事实源，健康检查要求状态 ready **且 worker 线程存活**；422 统一转换为 OpenAI 400（只暴露字段路径与错误类型，不回显输入值）；兜底 500 不含堆栈。
  - `server.py`：`python -m nanoserve.server` 与 console script `nanoserve`；`--enforce-eager` 默认 True（BooleanOptionalAction，`--no-enforce-eager` 关闭）。
  - `testing.py`：`FakeTokenizer`（确定性映射，记录模板调用参数，支持 list/str/dict（BatchEncoding 形态）三种模板返回风格）、`FakeEngine`（add_request/step 脚本/pop/cancel/exit 全契约桩，含无进展与异常脚本、step 延迟模拟）、`make_fake_factory`。
- `pyproject.toml`：运行依赖增加 `fastapi>=0.110.0`、`uvicorn[standard]>=0.29.0`；新增 `test` extra（httpx/pytest/openai）与 `[project.scripts] nanoserve`；包发现扩展为 `["nanovllm*", "nanoserve*"]`。
- 测试：新增/修订 5 个文件共 **117 项**（`test_service_schemas.py` 38、`test_chat_template.py` 17、`test_engine_worker.py` 24、`test_service_api.py` 29、`test_server_config.py` 9），文件头均注明 CPU 无 GPU/模型依赖与重复执行命令；其中新增回归覆盖模板 token 长度、坏模板、迟到记录世代和 decode 异常收口。
- 脚本：`scripts/validate_openai_api.py` 的 `--mode cpu` 本轮覆盖 18 项（含真实并发 HTTP、双路由 stream、结果形状）；`--mode gpu` 覆盖 8 项（完整响应字段、双路由 stream、真实 tokenizer 超上下文）；JSONL 输出不记录 prompt 明文。

## 2. 环境与模型可用性

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D（24 GB） |
| 驱动 / CUDA | 595.71.05 / 13.2 |
| Python / torch / transformers / flash-attn | 3.12.3 / 2.5.1+cu124 / 5.16.1 / 2.8.3.post1 |
| fastapi / uvicorn / openai / httpx（测试） | 0.141.1 / 0.52.4 / 3.13.0 / 0.27.2（本环境 pip 实装，`pip install fastapi "uvicorn[standard]" httpx openai`） |
| 模型 | Qwen3-0.6B，本地目录 `/root/huggingface/Qwen3-0.6B` |
| 服务配置（GPU 验收） | TP=1，enforce_eager=True，其余 Engine 参数取 Config 默认（max_model_len=4096、B=16384、chunk_size=1024） |
| 结论 | GPU 与模型均可用，§5 GPU 最小验收已实际执行 |

## 3. 设计取舍说明

1. **完成记录只存 completion token IDs**（§4.4 允许的两种口径之一）：字段名直接标注语义避免歧义；usage 由 `prompt_tokens`（记录）+ `completion_tokens` 完整还原，服务层用同一 tokenizer 解码（CPU/GPU 验证均走此路径）。
2. **记录捕获点在 `_finalize()` 所有权检查之后**而非各终态迁移点：单点捕获天然满足"FINISHED 迁移与 `_finalize()` 之间"的时序要求，且 `cancel()`/`timeout()`/`_postprocess`/`abort_all_active`/`_purge_terminal` 全部收口到 `_finalize`，不会漏记；重复收尾被所有权检查拦截，不会重记。
3. **空闲期取消的消费哨兵**：Engine 空闲时 `cancel_request()` 在安全点立即收尾并产生中止记录，此时无 step 驱动；`EngineWorker.cancel()` 在 signal 成功后入队 `_ConsumeCommand` 唤醒 worker 排出记录（记录消费只发生在 worker 线程，保持单所有者），否则已取消请求的 Future 会悬挂到关闭才被兜底收口。
4. **无进展即失败**：`step()` 返回空批次且仍有活动请求时（等待队首 KV 永久无法容纳、或全部暂停），worker 失败所有等待句柄并退出，与 Day10 `generate()` 的显式报错契约一致；不引入"连续空轮重试"的模糊策略，宁可失败也不无界循环或悬挂。单请求不可调度会连带失败同批请求，属 Day11–12 接受的边界（设计文档 §10 处理策略即 fail-all + exit）。
5. **transformers 5.x 模板返回兼容**：本环境 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`（UserDict 子类，非 dict 子类）。`_normalize_template_output()` 按"str→同一 tokenizer 再编码一次 / BatchEncoding→取 input_ids 首条 / list[int]→直接使用"规范化（§3.3 "以实际 tokenizer 兼容性为准"）；三种形态均有测试（`template_style` 桩）。
6. **`InternalRequest.prompt_token_ids` 为 tuple、Engine 边界转 list**：设计要求服务层不可变表示；`Sequence.append_token()` 会原地 append，故 `EngineWorker._handle_command` 在调用 `add_request` 时转换为 list（该缺陷由 GPU 验收首次暴露：CPU 桩未覆盖真实 Sequence 的 append 行为）。
7. **`RequestManager` pending 锁与命令线性化（本轮修复）**：pending 的登记、快照、resolve/fail 和 fail_all 统一受锁保护；登记与 worker 命令入队在同一临界区，worker submit 与 shutdown 哨兵也在线性化状态锁内排序，避免关闭竞态和 Future 悬挂。
8. **chat 对 Qwen3 思考模型的行为**：`temperature=0` 下 chat 输出含 `<think>` 前缀属模型模板自身行为（`add_generation_prompt=True` 后模型先输出思考段），Day11–12 不做模板改写（§2.2 不手写 ChatML），原样返回模型文本。

## 4. 实际执行的测试命令与结果

```bash
python -m pytest tests/test_service_schemas.py tests/test_chat_template.py \
    tests/test_engine_worker.py tests/test_service_api.py tests/test_server_config.py -q
# → 117 passed（38 + 17 + 24 + 29 + 9），1 warning（starlette 库自身的 anyio 弃用警告，与本项目代码无关）

python -m pytest tests/test_request_control.py tests/test_mixed_batch.py \
    tests/test_kv_cache_lifecycle.py tests/test_block_manager.py -q
# → 133 passed（Day10 基线 377 项中的控制/批次/KV/块管理子集，无回归）

python -m pytest -q
# → 495 passed（包含本轮新增/修订服务测试）

python -O -m pytest -q
# → 495 passed
```

## 5. CPU/ASGI 验收脚本（§9.1）

```bash
python scripts/validate_openai_api.py --mode cpu --output docs/evidence/day11-12/cpu-validation-review.jsonl
# → 共 18 项检查，失败 0 项（exit=0）；修订版包含真实并发提交、双路由 stream 与完整结果形状检查
```

覆盖 §9.1 的 CPU 可验证部分：health ready、models 与请求校验一致、completion/chat 非流式 OpenAI 字段与 usage 守恒、模板调用参数（原始有序 messages/`tokenize=True`/`add_generation_prompt=True`）、两条路由 SamplingParams 映射一致、非法 role/空 messages/上下文超限/模型不匹配/两个路由 `stream=true` 的状态码与 error code、3 个并发 HTTP 请求由单一 worker 驱动且结果形状不串、Engine 异常 500 与关闭 503、记录 drain 幂等与 ID 复用隔离、`exit()` 可重复恰好一次。脚本仍未覆盖真实 GPU 并发、服务端 504 和 TCP 断连。原始 JSONL 见 `docs/evidence/day11-12/cpu-validation-review.jsonl`。

## 6. GPU 最小验收（§9.2，已实际执行）

启动命令（与设计文档 §9.2 一致）：

```bash
python -m nanoserve.server \
  --model /root/huggingface/Qwen3-0.6B \
  --model-id Qwen3-0.6B \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --enforce-eager
```

```bash
python scripts/validate_openai_api.py --mode gpu \
  --base-url http://127.0.0.1:8000 --model-id Qwen3-0.6B \
  --output docs/evidence/day11-12/gpu-validation-review.jsonl
# → 共 8 项检查，失败 0 项（严格响应字段、双路由 stream、非法 role、真实 tokenizer 超上下文）
```

实际观察记录（原始输出节选存档于 `docs/evidence/day11-12/gpu-server.log`）：

- **health/model 信息**：`GET /health` → `{"status":"ok","model":"Qwen3-0.6B","accepting_requests":true}`；`GET /v1/models` → 单模型列表，`owned_by=nanoserve`。
- **completion 非流式**：`temperature=0`、`max_tokens=16` → `finish_reason=length`、`usage={"prompt_tokens":2,"completion_tokens":16,"total_tokens":18}`，总数守恒。
- **chat 非流式**：`temperature=0`、`max_tokens=16` → assistant message、`finish_reason=length`、`usage` 守恒（Qwen3 输出 `<think>` 前缀属模型模板自身行为，见 §3.8）。
- **确定性**：`temperature=0` 同一 prompt 重复 2 次输出逐字节相同（"Paris. The capital of Italy is Rome"）；仅验证 greedy 单点，不推广到全部采样配置。
- **上下文拒绝**：使用 `"hello " * 5000` 构造足够长的真实 tokenizer 输入，`max_tokens=8` → `400 context_length_exceeded`，发生在 KV 分配之前；不再把字符数误称为 token 数。
- **`stream=true`**：两路由均 `501 stream_not_implemented`。
- **并发**：CPU ASGI 脚本已用 3 个并发 HTTP 请求验证单一 worker 和响应不串；本轮 GPU 严格脚本为串行 8 项冒烟，GPU 真实并发仍未纳入结构化 case。
- **优雅关闭**：已有服务日志证明曾执行过 Uvicorn shutdown 和 `NanoServe 已停止`，但本轮未保存“SIGTERM 命令 + 退出码 + shutdown 期间 health 503”的完整结构化证据，因此只记为手工观察，不将其作为完整脚本验收。
- **GPU 验收暴露并修复的缺陷**：tuple prompt 导致 `Sequence.append_token` 崩溃（§3.6）；`BatchEncoding` 模板返回未被规范化导致 `ValueError: too many dimensions 'str'`（§3.5）。两者修复后均有对应单元测试（`test_prompt_token_ids_passed_to_engine`、`test_batch_encoding_template_output_normalized`）。
- **OpenAI Python Client 集成**：CPU fake Engine 路径已实际执行（`tests/test_service_api.py::TestOpenAIClientIntegration`，真实 uvicorn 端口 + `OpenAI(base_url=...)` 的 `completions.create` 与 `chat.completions.create`，断言 object/message/usage/finish_reason）；GPU 真实模型路径的 OpenAI Client 调用未单独执行（curl 等价请求已覆盖同一协议面），如实注明。

## 7. 设计文档 §9.3 验收清单逐项对照

> 清单本体已在本轮审查后同步更新。以下为逐项完成状态与证据指针；“通过”仅表示对应证据范围，不代表所有 GPU/生产边界均已覆盖。

### API 功能

| 项 | 状态 | 证据 |
| --- | --- | --- |
| `/v1/completions` 有效字符串 prompt 非流式响应 | 通过 | §5 case `completion_non_stream`；§6 curl 输出 |
| `/v1/chat/completions` 有效 messages 返回 assistant message | 通过 | §5 case `chat_non_stream`；§6 chat 输出 |
| `temperature/top_p/max_tokens` 两路由映射并生效 | 通过 | §5 case `sampling_params_consistent`；FakeEngine 桩收到逐字段一致参数 |
| completion/chat 共享内部 Request/采样/上下文校验路径 | 通过 | `build_*` 共用 `build_sampling_params`/`_check_context_length`；`test_chat_template.py::TestSamplingParamsConsistency` |
| chat 使用 tokenizer chat template，不手写格式 | 通过 | §5 case `chat_template_call_args`；`test_batch_encoding_template_output_normalized` |
| response ID/model/created/finish_reason/usage 稳定有测试 | 通过 | `test_completion_non_stream_shape`/`test_chat_non_stream_shape` |
| `stream` 解析：false 可用、true 明确 501 | 通过 | §5 case `error_stream_501`；§6 case `gpu_stream_501` |

### 错误与安全

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 缺少/非法 prompt、messages、role、采样参数清晰 400 | 通过 | `test_invalid_requests_return_unified_400`（7 组参数化） |
| 非法 model 统一 model-not-found | 通过 | `test_model_mismatch_404`；§6 case `gpu_models` 反向校验不动态加载 |
| 超上下文在分配 KV 前拒绝 | 通过 | 本轮 builder/API admission 回归（fake engine 零登记）；§6 GPU 严格脚本 `gpu_context_length_400` 使用 `"hello " * 5000` 实测 `400 context_length_exceeded` |
| 未 ready/draining/Engine 异常返回 503/500 | 通过 | `test_health_not_ok_after_engine_failure`、`test_draining_rejects_new_requests`、§5 case `engine_error_500`/`draining_503` |
| HTTP 层不修改 Sequence/Scheduler/KV | 通过 | 代码结构：路由/manager/worker 仅调用 engine 公开接口与只读记录通道 |
| 错误响应与日志不含 prompt/token 明文或堆栈 | 通过 | `test_error_bodies_and_logs_contain_no_prompt`（caplog 断言）；`test_template_error_maps_to_400` |

### 生命周期与工程质量

| 项 | 状态 | 证据 |
| --- | --- | --- |
| 一个 Engine 只有一个 worker 串行调用 `step()` | 通过 | `test_only_worker_thread_calls_step`；CPU 脚本 `concurrent_requests_single_step_driver`；GPU 真实并发仍未脚本化 |
| 完成记录可靠关联 request ID，迟到/复用不串请求 | 通过 | `TestRecordConsumption`（4 项）；`TestEngineCompletionRecordChannel::test_request_id_reuse_isolated_in_records` |
| 正常完成/取消/超时/异常不悬挂 Future | 通过 | `TestCancelAndErrors`、`test_shutdown_cancels_active_requests_without_hang`、§5 case `engine_error_500` |
| startup/shutdown/lifespan 幂等，退出走 Day10 清理 | 通过（CPU/正常路径） | `test_lifespan_shutdown_cleans_up_once`（exit 恰好一次）；GPU shutdown 仅有手工日志观察，未形成完整结构化证据 |
| `/health` 与 `/v1/models` 可用，启动失败不虚假 ready | 通过 | `test_startup_failure_propagates`、`test_health_draining_and_not_ready`、`test_health_not_ok_after_engine_failure` |
| `python -m nanoserve.server` 可重复启动 | 通过 | §6 启动命令在验收期间启动 3 次均成功；console script `nanoserve` 已注册 |
| CPU 服务测试/底层回归/`python -O` 回归实际执行并记录 | 通过 | 本轮 §4：117 passed（服务/配置）/ 495 passed / 495 passed |
| OpenAI Python Client 实际调用证据（注明路径） | 通过（CPU stub 路径） | `TestOpenAIClientIntegration`（真实 uvicorn 端口）；GPU 真实模型路径未执行，已注明 |

### 明确留给后续天数

| 项 | 状态 |
| --- | --- |
| Day13 SSE 增量输出与 `[DONE]` | 未实现（`stream=true` 一律 501，无伪装） |
| Day13 客户端断连检测并触发 cancel | 未实现 |
| Day14 `/metrics`、TTFT/TPOT、结构化日志 | 未实现 |

## 8. 未覆盖边界与遗留事项

- **GPU 真实模型路径的 OpenAI Python Client 调用未执行**：GPU 验收使用 curl 等价请求覆盖同一 HTTP 协议面；OpenAI Client 仅在 CPU fake Engine 路径实测。
- **TP>1 服务端到端、CUDA Graph 路径（`--no-enforce-eager`）**：未测（与 Day10 边界一致）。
- **无进展失败策略的粒度**：单个永久不可调度请求会失败该批全部等待请求（§3.4 取舍）；按请求粒度驱逐/降级留给后续版本。
- **`max_request_seconds` 服务层 deadline 在 GPU 路径未实测**：底层 TIMEOUT 语义由 Day10 调度器测试覆盖（`test_timeout_produces_aborted_record` 为 Engine 层等价验证），服务端到端 504 未在真实模型上触发。
- **Future 硬超时**：`RequestManager.wait()` 仍无 timeout；若底层 forward 永久阻塞，HTTP 线程可能继续等待，需后续 supervisor/可取消执行方案。
- **真实断连（客户端中断 TCP）下的取消**：Day13 范围，未实现未测。
- **长时运行内存基线**：完成记录队列以"服务 worker 每轮消费"为前提，未做 10⁴ 量级请求的泄漏压测（Day15–18 benchmark 范围）。
- **`python -O` 回归中的 2 条 warning** 包含 pytest 对优化模式断言的提示和 Starlette/anyio 库自身弃用警告；普通回归为 1 条 Starlette/anyio 警告，均与本项目业务逻辑无关。
