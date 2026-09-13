# Day 11–12 教程：从同步推理引擎到 OpenAI 兼容服务

> 读者假设：你刚开始学习大模型推理引擎，已经了解本项目 Day 1–10 建立的基础概念：token、Prefill/Decode、Paged KV Cache、Continuous Batching 和请求状态机。
>
> 本教程对应 [Day 11–12 设计文档](./openai-api-day11-12.md)，代码主要位于 `nanoserve/`、`nanovllm/engine/llm_engine.py` 和 `nanovllm/engine/scheduler.py`。验收与审查记录见 [day11-12-validation.md](./day11-12-validation.md) 和 [day11-12-review.md](./day11-12-review.md)。
>
> 建议读法：先读 §1–§3 建立全局地图，再对照 §4–§8 阅读代码，最后按 §10 的命令运行 CPU 测试。不要一开始就从 FastAPI 路由逐行钻进去；服务层的正确理解必须建立在 Day10 的 Engine 生命周期之上。

## 1. 这一轮解决了什么问题？

Day 1–10 的引擎已经可以做一件重要的事：给它一组 prompt，它在 GPU 上批量生成文本。典型调用方式是：

```python
outputs = llm.generate(prompts, sampling_params)
```

但真正的推理服务还需要解决另一层问题：

- 用户如何通过 HTTP 提交请求？
- completion 和 chat 两种请求怎样进入同一套调度器？
- 多个 HTTP 请求怎样共享一个 Engine，而不是并发踩坏 Scheduler？
- 模型完成后，`Sequence` 已被清理，服务层如何仍然拿到 request ID、usage 和 finish reason？
- 输入错误、模型错误、Engine 异常和服务关闭应该返回什么？
- 服务启动、健康检查、停止接收新请求和释放 GPU 资源的顺序是什么？

Day 11–12 的核心工作可以概括为：

```text
把“离线同步推理程序”包成“单 Engine、可排队、可测试的 HTTP 服务”
```

本轮明确支持：

- `POST /v1/completions`：非流式文本 completion；
- `POST /v1/chat/completions`：非流式 chat completion；
- `GET /health`：服务健康状态；
- `GET /v1/models`：当前公开模型；
- `stream=false`；
- `stream=true` 明确返回 `501 stream_not_implemented`，留给 Day13。

本轮没有实现 SSE。若把 `stream=true` 忽略后返回完整文本，客户端会以为自己拿到了流式协议，实际上没有得到任何增量事件；因此明确返回 501 比静默退化更正确。

## 2. 全局架构：HTTP 请求怎样走到 GPU

### 2.1 总体数据流

```text
客户端（curl / OpenAI Python Client）
        │ JSON
        ▼
FastAPI 路由（nanoserve/api.py）
        │ Pydantic 校验、model/stream 检查
        ▼
请求构建器（nanoserve/service.py）
        │ prompt 编码或 chat template
        │ → tuple[int, ...] + SamplingParams
        ▼
InternalRequest
        │ RequestManager 登记 Future
        ▼
EngineWorker（专用单线程）
        │ 唯一调用 engine.add_request()/step()/exit()
        ▼
LLMEngine
        │ add_request → Scheduler.waiting
        │ step → schedule → ModelRunner → postprocess
        ▼
Scheduler / Sequence / BlockManager
        │ 状态迁移、KV 分配、token 追加、终态清理
        ▼
CompletedRequest / AbortedRequest
        │ pop_completed()/pop_aborted()
        ▼
RequestManager 解码并 resolve Future
        ▼
OpenAI 风格 JSON
```

可以把这条链路分为两个平面。

### 数据面

数据面是大模型真正计算的部分：prompt token、completion token、KV Cache block、Prefill/Decode batch、GPU forward 和采样。这些仍由 `LLMEngine`、`Scheduler`、`ModelRunner` 和 `BlockManager` 管理，HTTP 层不能直接碰它们。

### 控制面

控制面是请求管理部分：request ID、Future、是否还在接收、完成/中止记录、worker 状态、readiness 和 shutdown 命令。Day 11–12 的主要新增代码在控制面。好的服务层不会重新实现 KV Cache，而是把 HTTP 请求安全地接入已有的底层生命周期。

### 2.2 为什么需要专用 EngineWorker

最容易写出的错误实现是：

```python
@app.post("/v1/completions")
def completion(request):
    return engine.generate([request.prompt], params)[0]
```

或者：

```python
async def completion(request):
    return await asyncio.to_thread(engine.step)
```

它们的问题分别是：

1. `generate()` 把请求当作独立离线任务，HTTP 请求之间难以形成 continuous batching。
2. 多个 handler 可能同时调用同一个 `step()`，两个线程会竞争 Scheduler、Sequence 或 KV 账本。
3. `asyncio.to_thread()` 只是把阻塞工作移到线程池，并没有保证同一个 Engine 只有一个调用者。

本项目采用单一专用线程：

```text
HTTP 线程：校验、构建 InternalRequest、等待 Future
                         │
                         ▼
              一个 EngineWorker 线程
              add_request → step → 读完成记录
```

多个 HTTP 请求仍然可以同时到达，但它们只是在服务层排队；真正的 Engine 操作由一个 worker 线性化。这样既保留 Scheduler 的 continuous batching，也遵守 Day10 的 Engine 安全边界。

## 3. 两个统一层次：HTTP Schema 与 InternalRequest

### 3.1 HTTP Schema 是外部协议

`nanoserve/schemas.py` 定义 Pydantic 模型。completion 的核心字段是：

```python
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
```

chat 的核心字段是：

```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
```

没有实现的字段不能被静默忽略，因此请求模型使用 `extra="forbid"`：

```json
{
  "model": "Qwen3-0.6B",
  "prompt": "hello",
  "stop": "\\n"
}
```

当前 MVP 会拒绝它，而不是假装 `stop` 已经生效。用户才能区分“模型输出不符合预期”和“服务根本没有处理这个参数”。

### 3.2 为什么字段要严格校验

服务边界应该避免隐式类型转换：

```python
max_tokens: int = Field(default=64, ge=1, strict=True)
temperature: float = Field(default=1.0, ge=0, strict=True)
top_p: float = Field(default=1.0, gt=0, le=1, strict=True)
stream: bool = Field(default=False, strict=True)
```

这样字符串数字、布尔值等错误输入会在 HTTP 边界得到 400，而不是进入 Engine 后才出现难以解释的错误。FastAPI 默认 Pydantic 422 也会被统一转换成 OpenAI 风格的 400。

### 3.3 InternalRequest 是内部统一表示

completion 的输入是字符串，chat 的输入是消息列表。Engine 不需要知道这两种 HTTP 外形，只需要 token：

```python
@dataclass(frozen=True, slots=True)
class InternalRequest:
    request_id: str
    kind: Literal["completion", "chat"]
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    model_id: str
    created_at: float
    deadline: float | None
```

值得学习的设计点：

- `frozen=True`：提交后服务层不能偷偷修改请求；
- `tuple[int, ...]`：服务层使用不可变 token 表示；
- `kind`：只用于最终响应包装，不污染底层状态机；
- 时间字段使用 `perf_counter()` 单调时钟；
- 不保存 `Sequence`、`block_table` 或 Scheduler 引用。

Engine 的 `Sequence` 会在生成过程中原地追加 token，所以到 Engine 边界时 worker 把 tuple 转成 list：

```python
self._engine.add_request(
    list(request.prompt_token_ids),
    request.sampling_params,
    request.request_id,
    request.deadline,
)
```

上层不可变、底层按既有数据结构转换，是一种很实用的边界设计。

## 4. Completion 路径：从 prompt 到响应

### 4.1 路由层只负责协议编排

`nanoserve/api.py` 的 completion 路由按固定顺序工作：

```text
1. 读取 ServiceState
2. 检查服务是否 ready
3. 检查 body.model 是否匹配公开模型 ID
4. 检查 stream=false
5. 构建 InternalRequest
6. 提交 RequestManager 并等待 Future
7. 包装 CompletionResponse
```

伪代码如下：

```python
service = _service_state(request)
_ensure_ready(service)
_ensure_model(service, body.model)
_ensure_not_stream(body.stream)
internal = build_completion_request(
    body,
    service.tokenizer,
    model_id=service.model_id,
    max_model_len=service.max_model_len,
    max_request_seconds=service.max_request_seconds,
)
handle = service.manager.submit(internal)
result = service.manager.wait(handle)
return CompletionResponse(...)
```

路由不会直接做这些事情：

```python
sequence.status = ...          # 不做
scheduler.running.append(...)  # 不做
block_manager.deallocate(...)  # 不做
engine.step()                  # 不做
```

如果 HTTP 层开始直接操作这些对象，状态、队列和 KV 三本账就会出现两个管理员。

### 4.2 prompt 编码和上下文检查

`build_completion_request()` 使用 Engine 的同一个 tokenizer：

```python
token_ids = tokenizer.encode(request.prompt)
_check_context_length(
    len(token_ids), request.max_tokens, max_model_len)
```

公式是：

```text
prompt token 数 + 最大生成 token 数 <= max_model_len
```

检查位置更重要：它发生在 `engine.add_request()` 之前，也就是 KV 分配之前。正确顺序是：

```text
encode → 计算 token 数 → 检查上下文 → add_request
```

错误顺序是先分配 KV，运行时才发现超限。

### 4.3 完成记录解决了什么问题

底层 `Sequence` 的生命周期是：

```text
WAITING → RUNNING → FINISHED
                         │
                         └─ _finalize()：释放 block、删除活动索引
```

`_finalize()` 后，`Scheduler.requests` 已经没有这个请求。如果服务层此时再调用 `get_request(request_id)`，通常得到 `None`，但响应仍需要 request ID、completion token、usage 和 finish reason。

因此 Day11–12 增加只读完成记录：

```python
@dataclass(frozen=True, slots=True)
class CompletedRequest:
    seq_id: int
    request_id: str
    completion_token_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]
    finished_at: float
```

Scheduler 在 `_finalize()` 的所有权检查通过后捕获记录，Engine 用：

```python
engine.pop_completed()
engine.pop_aborted()
```

把记录交给 worker。`pop` 是 drain 操作，重复调用不会返回同一记录。

### 4.4 为什么只保存 completion token IDs

本项目选择记录生成部分，并单独记录 prompt/completion 数量：

```text
completion_token_ids = [生成 token...]
prompt_tokens = prompt 长度
completion_tokens = 生成长度
```

服务层用同一个 tokenizer 解码：

```python
text = tokenizer.decode(list(record.completion_token_ids))
total_tokens = record.prompt_tokens + record.completion_tokens
```

关键不是必须保存完整 token 序列，而是字段语义必须明确且全项目统一。

## 5. Chat 路径：消息如何变成 prompt

### 5.1 Chat template 解决什么问题

不同模型的聊天格式不同。如果服务层自己拼接 ChatML，就可能和模型训练格式不一致。

Day12 使用 tokenizer 自带模板：

```python
template = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

三个参数含义：

- `messages`：有序的 system/user/assistant 消息；
- `tokenize=True`：尽可能直接得到 token IDs；
- `add_generation_prompt=True`：提示模型现在开始生成 assistant。

服务层不手写角色分隔符，也不在模板外额外插入 ChatML。

### 5.2 模板结果为什么需要规范化

不同 Transformers 版本可能返回：

1. `list[int]`；
2. `list[list[int]]`；
3. `BatchEncoding` 或带 `input_ids` 的映射；
4. 字符串。

`_normalize_template_output()` 将这些形式转换成单条 `list[int]`：

```text
str                  → 同一个 tokenizer.encode(str)
BatchEncoding/dict   → 取 input_ids，再取第一条
list[list[int]]      → 取第一条
list[int]            → 直接使用
其他/None            → InvalidRequestError(400)
```

上下文检查必须使用规范化后的 token 数：

```python
token_ids = _normalize_template_output(template, tokenizer)
_check_context_length(len(token_ids), request.max_tokens, max_model_len)
```

不能使用：

```python
len(template)  # 可能是字符数或字段数
```

例如 BatchEncoding 有 `input_ids` 和 `attention_mask` 两个字段，`len(template)` 可能是 2，而真实 prompt 可能有几千个 token。本轮审查发现并修复了这个 bug。

### 5.3 role 和 content 校验

当前 MVP 只允许：

```python
Literal["system", "user", "assistant"]
```

每条消息还必须有非空字符串 content。tool/function calling 需要独立的 schema、模板和执行语义，不能在当前 MVP 中半支持。

### 5.4 chat 响应只是另一种包装

进入 Engine 后，chat 与 completion 没有两套调度逻辑。完成后只根据 `kind` 选择外形：

```text
completion → choices[0].text
chat       → choices[0].message = {role: assistant, content: text}
```

底层 token、finish reason 和 usage 都来自同一个完成记录。

## 6. EngineWorker：服务层最重要的控制边界

### 6.1 worker 的状态和命令

`EngineWorker` 的生命周期是：

```python
NOT_STARTED → RUNNING → STOPPING → STOPPED
                              └────→ FAILED
```

命令队列中主要有：

- `_SubmitCommand`：把 `InternalRequest` 交给 Engine；
- `_ConsumeCommand`：消费空闲取消产生的中止记录；
- `_ShutdownCommand`：唤醒空闲 worker 并开始关闭。

HTTP 线程不直接改 worker 内部状态，而是放入命令或调用线程安全的高层方法。

### 6.2 活动期和空闲期

worker 主循环有两个分支：

```text
有活动请求：
  排空 submit/cancel 命令
  engine.step()
  pop_completed/pop_aborted
  若空批次但仍有活动请求 → 明确失败，禁止忙循环

无活动请求：
  阻塞等待命令
  不调用 step()
```

空闲时不能持续调用 `step()`，否则会造成 CPU 忙循环、无意义日志，甚至让 ModelRunner 在空 batch 上崩溃。

如果请求因为 KV 容量或暂停状态完全无法推进，同步 Engine 没有等待外部事件的抽象。继续循环只会无限重试，所以当前选择显式失败，让 Future 收口。

### 6.3 worker 是批处理边界

多个请求先进入命令队列：

```text
HTTP A → submit(A)
HTTP B → submit(B)
HTTP C → submit(C)
```

worker 先排空命令，再调用：

```text
engine.add_request(A)
engine.add_request(B)
engine.add_request(C)
engine.step()
```

Scheduler 因而可以把它们放入同一轮。测试 `test_requests_batched_into_one_round` 验证了这一点。

### 6.4 cancel 的两段式语义

Day10 的取消不是网络线程立即删除请求，而是：

```text
HTTP/控制线程
   │ cancel(request_id)
   ▼
Engine.cancel_request()
   │
   ├─ step 正在执行：只设置 cancel_requested
   └─ Engine 空闲：安全点完成 CANCELLED 与资源释放
```

这就是 **signal now, cleanup at safe point**：先表达控制意图，等 GPU forward 返回后再修改 token、队列和 block。

### 6.5 shutdown 的顺序

正常关闭顺序是：

```text
1. ServiceState 不再 accepting
2. worker 收到 shutdown
3. 排队但尚未 admission 的请求失败收口
4. 对仍活动请求发送 server_shutdown cancel signal
5. worker 在有限时间内继续 step，让请求自然收口
6. worker 所在线程调用 engine.exit()
7. 剩余 Future 兜底失败
8. worker → STOPPED（exit 失败则 FAILED）
```

最容易犯的错误是 join 超时后，lifespan 线程直接调用 `engine.exit()`。如果 worker 仍在 `step()`，这就变成两个线程同时操作 Engine。本轮实现改为：worker 仍存活时，lifespan 不跨线程调用 `engine.exit()`，宁可报告关闭失败，也不破坏 GPU/调度资源安全边界。

## 7. RequestManager：Future、记录和错误映射

### 7.1 为什么服务层使用 Future

路由可以写成：

```python
handle = manager.submit(internal)
result = manager.wait(handle)
```

实际生成可能需要很多轮 `step()`。Future 让 HTTP 线程等待最终结果，worker 线程完成时设置结果：

```text
HTTP 线程                         worker 线程
   │                                  │
   ├─ submit → Future                 ├─ step
   ├─ wait   ────────────────┐        ├─ pop_completed
   │                          │        └─ future.set_result
   └─ 返回 JSON ◄──────────────┘
```

`RequestHandle` 只保存 request ID、kind、创建时间、Future 和 admission 后绑定的 `seq_id`，不保存可写 Sequence。

### 7.2 pending 锁和线性化

pending 表同时被 HTTP 线程和 worker/关闭路径访问，不能只依赖 CPython GIL 的单次 dict 原子操作。需要保护这个业务事务：

```text
检查 accepting
  → 创建 handle
  → 放入 pending
  → 放入 worker 命令队列
```

本轮修复后：

- Manager 锁保护 pending 登记、快照、resolve/fail；
- submit 和 worker 命令入队在同一临界区；
- worker submit 与 shutdown 哨兵在线性化状态锁内排序。

### 7.3 为什么完成记录还要校验 seq_id

理论上 request ID 可能复用：

```text
旧请求 A：request_id = "reuse", seq_id = 1
A 的记录迟到
新请求 B：request_id = "reuse", seq_id = 2
旧记录到达
```

如果 Manager 只看 ID，就可能把 A 的记录交给 B。当前 handle 在 admission 后绑定 seq_id，resolve 时同时检查 request ID 和 seq ID；旧记录会被丢弃。测试 `test_late_record_cannot_resolve_reused_id` 固定了这个边界。

### 7.4 错误映射和解码失败

`AbortedRequest.finish_reason` 不伪装成成功文本：

```text
engine_error / execution_error → 500 EngineError
timeout / deadline_exceeded   → 504 RequestTimeoutError
cancelled / shutdown          → 503 ServiceDrainingError
```

即便 tokenizer decode 异常，也必须让 Future 得到 `EngineError`；否则 HTTP 请求会永久等待。本轮 worker 对单条记录消费做了异常隔离。

## 8. FastAPI 生命周期与错误协议

### 8.1 app factory 为什么支持注入

`create_app(engine_factory=None, server_config=None)` 让生产和测试走同一套路由：

```text
生产：lifespan → default_engine_factory → 真实 LLMEngine
测试：lifespan → make_fake_factory → FakeEngine/FakeTokenizer
```

因此 CPU CI 可以测试 schema、chat template、错误响应、worker 和 shutdown，而不初始化 CUDA/NCCL。

### 8.2 ServiceState 是健康检查事实源

ServiceState 的生命周期是：

```text
not_ready → ready → draining → stopped
     └──────────────任何阶段都可能进入 failed
```

`/health` 返回 200 的条件是：

```text
ServiceState == ready
且 worker.alive
且 worker.status == RUNNING
```

Engine worker 意外崩溃后，健康检查会回落为 `503 not_ready`。`/v1/models` 同样要求服务 ready，不能成为绕过生命周期的后门。

### 8.3 统一错误结构

客户端得到类似结构：

```json
{
  "error": {
    "message": "messages must contain at least one item",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "empty_messages"
  }
}
```

常见错误码：

| 场景 | HTTP | code |
| --- | ---: | --- |
| 字段/类型/role 错误 | 400 | `invalid_request_error` |
| 上下文超长 | 400 | `context_length_exceeded` |
| 模型 ID 不匹配 | 404 | `model_not_found` |
| 暂不支持流式 | 501 | `stream_not_implemented` |
| 未就绪/排空 | 503 | `service_not_ready` / `service_draining` |
| deadline 超时 | 504 | `request_timeout` |
| Engine 异常 | 500 | `engine_error` |

### 8.4 错误为什么不能泄漏 prompt

错误日志和 body 不应包含完整 prompt、token 明文、block table、Python traceback 或 tokenizer/Jinja 内部细节。当前测试使用特殊字符串检查 body 和捕获日志都不会回显 prompt。

## 9. 配置与启动

### 9.1 ServerConfig 与 Engine Config 的边界

`nanoserve/config.py` 只负责模型路径、公开 ID、host/port、服务 deadline、eager/TP。`nanovllm/config.py` 继续负责 `max_model_len`、token budget、chunk size、KV 容量和显存比例。

不要在服务配置中复制 Engine 校验逻辑；两个配置对象职责不同，分开才能避免启动入口变成第二个 Engine 构造器。

### 9.2 环境变量与 CLI 优先级

支持：

```text
NANOSERVE_MODEL
NANOSERVE_MODEL_ID
NANOSERVE_HOST
NANOSERVE_PORT
NANOSERVE_MAX_REQUEST_SECONDS
NANOSERVE_ENFORCE_EAGER
NANOSERVE_TENSOR_PARALLEL_SIZE
```

未显式给出的 CLI 参数读取环境变量，显式 CLI 参数优先：

```bash
python -m nanoserve.server \
  --model /root/huggingface/Qwen3-0.6B \
  --model-id Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager
```

原先 argparse 固定默认值会遮蔽环境变量；当前实现通过区分“未提供 CLI 参数”和“显式提供 CLI 参数”解决该问题。

### 9.3 真实模型启动时发生什么

真实 `LLMEngine` 初始化可能执行：读取 HuggingFace 配置、初始化 CUDA/NCCL、创建模型、加载权重、warmup、计算/分配 KV Cache、创建 tokenizer 和 Scheduler。因此 CPU 环境只能使用 fake factory；GPU 验收要记录 GPU、模型、TP、eager、命令和原始结果。

## 10. 测试：怎样用 CPU 学习 GPU 服务代码

### 10.1 运行新增服务测试

```bash
python -m pytest \
  tests/test_service_schemas.py \
  tests/test_chat_template.py \
  tests/test_engine_worker.py \
  tests/test_service_api.py \
  tests/test_server_config.py -q
```

覆盖 schema、chat template、token 上下文边界、完成记录、worker、batch、cancel、timeout、异常、shutdown、FastAPI 错误、OpenAI Python Client 和 CLI/environment 合并。

### 10.2 Fake Engine 测试思路

`nanoserve/testing.py` 的 FakeEngine 不是为了模拟模型质量，而是模拟 Engine 契约：

```text
add_request(token_ids, params, request_id, deadline)
step() → (outputs, num_tokens)
pop_completed()
pop_aborted()
cancel_request()
exit()
```

可以脚本化：

```python
FakeEngine(step_script=[("pending",)] * 3)
FakeEngine(step_script=[RuntimeError("boom")])
FakeEngine(step_script=[("noop",)])
```

这样可以在毫秒级 CPU 测试正常完成、多轮 pending、无进展、Engine 异常、取消和关闭。

### 10.3 审查新增的回归测试

| 测试 | 防止的错误 |
| --- | --- |
| `test_chat_budget_uses_normalized_token_count` | dict/string 模板用错长度 |
| `test_bad_template_output_maps_to_invalid_request` | malformed template 变成 500 |
| `test_decode_failure_closes_future_as_engine_error` | decode 异常导致 Future 悬挂 |
| `test_late_record_cannot_resolve_reused_id` | 旧记录完成新请求 |
| `test_environment_values_are_used_when_cli_omits_them` | CLI 默认遮蔽环境变量 |
| `test_explicit_cli_values_override_environment` | CLI 优先级失效 |
| `test_deadline_must_be_finite_positive` | NaN/无穷 deadline 永不超时 |

好的回归测试应该满足：删除修复后测试必然失败。原先孤立的 Engine 断言做不到这一点，所以被改成完整 route → builder → worker admission 检查。

### 10.4 单元测试与验收脚本的区别

单元测试精确验证一个函数或并发边界；验收脚本模拟用户看到的完整路径。

CPU 验收：

```bash
python scripts/validate_openai_api.py \
  --mode cpu \
  --output docs/evidence/day11-12/cpu-validation-final.jsonl
```

当前 CPU 脚本验证两个路由、健康检查、错误协议、双路由 stream 501、真实并发 HTTP、request ID 和响应形状。

GPU 验收：

```bash
python scripts/validate_openai_api.py \
  --mode gpu \
  --base-url http://127.0.0.1:8000 \
  --model-id Qwen3-0.6B \
  --output docs/evidence/day11-12/gpu-validation-review.jsonl
```

当前 GPU 脚本验证 8 项 TP=1 eager HTTP 最小路径：health、精确 model ID、completion/chat 完整响应、两个路由的 stream 501、非法 role 和真实 tokenizer 超上下文 400。

不要把 CPU 单测写成 GPU 全部通过，也不要把串行 GPU 冒烟写成 GPU 并发压力测试。

## 11. 典型错误模式

### 错误模式一：使用错误的长度单位

```python
len(template)  # 字符数、字段数或其他容器长度
```

必须先得到模型真正使用的 token IDs，再计算长度：

```python
token_ids = normalize(template)
len(token_ids)
```

### 错误模式二：用 GIL 代替业务锁

“dict 单次操作原子”不等于下面这个事务是原子的：

```text
检查状态 → 登记句柄 → 入队 → shutdown 快照
```

并发代码要保护业务不变量，而不只是保护某一条 Python 字节码。

### 错误模式三：先从索引删除，再做可能失败的工作

```python
handle = pending.pop(id)
text = tokenizer.decode(tokens)  # 可能抛异常
```

如果 decode 失败，必须仍然给等待者设置 exception。

### 错误模式四：关闭线程越权清理 Engine

worker 还在 forward 时，另一个线程不能因为 join 超时就直接 exit。线程强杀也不是安全方案；无法中止 GPU kernel 时，应在进程边界设计 supervisor。

### 错误模式五：把弱脚本检查当完整验收

只检查 `status_code == 200` 不能证明 object、model、choices、finish_reason、request ID 和 usage 都正确。验收脚本应检查协议字段，同时诚实写出没有检查的并发、长时和故障范围。

## 12. 运行示例

### 12.1 CPU：不加载模型验证 HTTP 协议

```bash
python scripts/validate_openai_api.py --mode cpu
```

### 12.2 GPU：启动本地服务

```bash
python -m nanoserve.server \
  --model /root/huggingface/Qwen3-0.6B \
  --model-id Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --enforce-eager
```

另一个终端：

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

completion：

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "prompt": "什么是 Paged KV Cache？",
    "max_tokens": 16,
    "temperature": 0
  }'
```

chat：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "你是一个简洁的助手。"},
      {"role": "user", "content": "解释 continuous batching。"}
    ],
    "max_tokens": 16,
    "temperature": 0
  }'
```

流式边界：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-0.6B",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": true
  }'
```

预期为 HTTP 501 和 `stream_not_implemented`，不是 SSE。

### 12.3 OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="test",  # 当前服务不做鉴权，但 Client 要求提供
)

completion = client.completions.create(
    model="Qwen3-0.6B", prompt="hello", max_tokens=8, temperature=0,
)
print(completion.choices[0].text)

chat = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=8,
    temperature=0,
)
print(chat.choices[0].message.content)
```

当前已用 fake Engine + 本地 uvicorn 路径测试 OpenAI Client；真实 GPU 模型上的 Client 单独调用仍未覆盖。

## 13. 验收结果怎样解读

本轮实际结果：

```text
Day11–12 服务/配置测试：117 passed
全量 CPU 回归：495 passed
python -O 全量回归：495 passed
CPU ASGI 验收：18/18
GPU TP=1 eager 严格 HTTP：8/8
```

这些数字支持：API 主链路接通、completion/chat 统一进入 Engine、完成记录和错误收口可工作、CPU 控制面有较完整测试、真实 Qwen3-0.6B TP=1 eager 非流式 HTTP 最小路径可用。

它们不支持：已实现 SSE、已验证真实 TCP 断连、已验证 TP>1/CUDA Graph、已完成 GPU 并发压力、已完成 GPU 504、已完成 10⁴ 请求长期稳定性或已完成完整 OpenAI API。

## 14. 思考题

1. 如果 `CompletedRequest` 不保存 `prompt_tokens`，服务层还能可靠构造 usage 吗？这样做会不会重新依赖已清理的 Sequence？
2. 为什么 `stream=true` 当前返回 501 比返回完整 JSON 更正确？Day13 需要把完成记录队列扩展成什么形式？
3. 如果允许多个 EngineWorker 同时调用同一个 `LLMEngine.step()`，至少会竞争哪些对象？为什么不能简单让一把 Python 锁跨越 GPU forward？
4. 如果 chat template 返回 BatchEncoding，为什么 `len(template)` 可能是 2？请设计一个测试，让旧实现必然失败、新实现通过。
5. shutdown 时 worker 卡在很慢的 GPU forward，为什么不能从主线程强杀 worker？进程级 supervisor 和线程级取消有什么区别？
6. 如果服务要支持 `stop`，应该在哪一层解析？怎样传到当前 `SamplingParams` 和 `Sequence`，又不让未支持字段被静默忽略？
7. 当前异常路径把 Engine 标记为不可重试。什么情况下可以安全实现 Engine 重载？需要怎样处理旧请求、GPU 进程和 request ID？
8. Day14 要记录 TTFT，应该从哪些时间点开始记录？为什么不能把 HTTP 接收时间直接当成首 token 时间？

## 15. 一句话总结

Day11–12 最重要的不是“加了两个 HTTP 路由”，而是建立了一条安全边界：

```text
外部协议可以变化，内部推理状态只有 Engine/Scheduler 管；
多个请求可以并发到达，但同一个 Engine 只有一个 worker 驱动；
完成后 Sequence 可以消失，但只读完成记录必须可靠地把结果交回客户端。
```

掌握这三条原则，你就开始从“会调用模型”走向“理解推理服务引擎”。
