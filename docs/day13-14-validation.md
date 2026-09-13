# Day 13–14 SSE 与可观测性验收记录

> 本记录对应 [Day 13–14 设计基线](./sse-observability-day13-14.md)。设计文档的验收清单不在此修改；本文件只记录实际执行结果。

## 1. 分支与基线

- 分支：`feature/sse-observability-day13-14`
- Day12 基线：`1de08bc`（`merge: integrate Day11-12 OpenAI APIs`）
- 实现提交：待本记录完成后创建
- 工作区模型/权重：CPU FakeEngine 验收使用 `/tmp/fake`，不加载模型权重，不初始化 CUDA/NCCL。

## 2. 实现范围

已实现的代码范围包括：

- `TokenEvent` 独立增量事件 DTO，以及 Scheduler/LLMEngine 的公共 drain；
- mixed prefill/decode 下按 `needs_sample` 产生事件，事件先于终态记录；
- StreamHandle 有界非阻塞队列、request/seq 世代校验和终态收口；
- completion/chat SSE 首 chunk、增量 chunk、finish chunk、usage、`[DONE]`；
- 客户端断连的高层 cancel 入口；
- Prometheus 独立 `CollectorRegistry`、计划指标和 `/metrics`；
- 请求时间线、单调时钟计算和结构化生命周期日志白名单；
- KV 物理资源快照及 prefix hit/miss/capacity-failure 分类统计；
- FakeEngine/FakeTokenizer 的 CPU 流式测试能力。

## 3. 实际执行命令与结果

### 3.1 Day11–12 服务回归基线

```bash
python -m pytest tests/test_service_api.py tests/test_service_schemas.py \
  tests/test_chat_template.py tests/test_engine_worker.py -q
```

结果：`109 passed, 1 warning`（实现前基线）。

### 3.2 Day13–14 定向测试

```bash
python -m pytest tests/test_service_sse.py tests/test_stream_disconnect.py \
  tests/test_observability.py tests/test_observability_unit.py -q
```

结果：`16 passed, 1 warning`。

### 3.3 关键服务回归

```bash
python -m pytest tests/test_service_api.py tests/test_engine_worker.py \
  tests/test_observability_unit.py -q
```

结果：`61 passed, 1 warning`。

### 3.4 全量 CPU 回归

```bash
python -m pytest -q
```

结果：`511 passed, 1 warning`。

```bash
python -O -m pytest -q
```

结果：`511 passed, 2 warnings`。

### 3.5 CPU OpenAI/SSE 验收脚本

```bash
python scripts/validate_openai_api.py --mode cpu
```

结果：`18/18 pass`，包含 completion/chat 非流式、SSE smoke、并发单 worker、错误、关闭、记录 drain、ID 复用和退出幂等。

```bash
python scripts/validate_sse_observability.py --mode cpu \
  --output docs/evidence/day13-14/cpu-validation.jsonl
```

结果：`4/4 pass`（completion/chat SSE、metrics、单 worker）；原始 JSONL 见 `docs/evidence/day13-14/cpu-validation.jsonl`。

## 4. 验收清单对照

### 4.1 Day13 SSE 与断连

| 项目 | 状态 | 证据/说明 |
| --- | --- | --- |
| completion 流式 media type、headers、首/增量/finish/`[DONE]` | 通过（CPU/ASGI） | `tests/test_service_sse.py`；定向测试通过 |
| chat chunk object、assistant role、delta、finish、`[DONE]` | 通过（CPU/ASGI） | `tests/test_service_sse.py` |
| 流/非流最终文本和 usage 一致 | 通过（FakeTokenizer） | `test_stream_aggregates_same_as_non_stream` |
| `stop`/`length` 与底层终态一致 | 通过（CPU fake） | SSE finish assertions |
| mixed prefill/decode 事件过滤与映射 | 通过（底层 CPU 回归） | Scheduler/mixed-batch tests |
| request ID + seq ID 事件世代隔离 | 部分通过 | 既有完成记录隔离通过；异常/迟到 TokenEvent 的扩展覆盖需继续补充 |
| 首 token 前、token 间、完成竞争断连 | 部分通过 | worker-level disconnect 通过；ASGI/TestClient 不等价于真实 TCP 断连 |
| 断连触发 `client_disconnected` 并释放 active/pending/KV | 部分通过 | CPU fake worker 路径通过；真实 TCP/GPU 未完成 |
| Engine error/timeout/shutdown 不伪造成功 finish | 部分通过 | 非流式和 worker 异常回归通过；流式异常专门证据需补充 |
| `step()`/`generate()`/非流式 API 兼容 | 通过（CPU） | 全量 511 passed |

### 4.2 Day14 指标与结构化日志

| 项目 | 状态 | 证据/说明 |
| --- | --- | --- |
| `/metrics` 可抓取并包含 9 个计划指标 | 通过（CPU/ASGI） | `tests/test_observability.py` |
| queue wait、TTFT、TPOT/ITL、latency 口径 | 通过（primitive） | `tests/test_observability*.py`；统一 `perf_counter` |
| 无首 token/少于两个 token 不伪造 TTFT/TPOT | 通过（primitive） | timeline 测试 |
| prompt/generation token 计数 | 通过（单请求 happy path） | metrics smoke；中止/异常口径仍需扩展 |
| running、KV used/total/utilization | 部分通过 | 底层 resource snapshot 通过；HTTP metrics 当前主要暴露 running/utilization |
| prefix hit/miss/capacity failure 统计 | 部分通过 | 底层分类通过；服务抓取后的累计 counter 需最终脚本复核 |
| lifecycle JSON envelope 与隐私白名单 | 通过（CPU primitive） | `test_structured_log_drops_sensitive_fields` |
| received→submitted→admitted→first token→finished/aborted 可串联 | 部分通过 | happy path 已接线；异常/断连全链需补真实事件证据 |
| 观测异常不阻塞 Engine | 部分通过 | 抛异常隔离有测试；慢 handler、永久阻塞未实测 |
| 低基数 labels、registry 隔离 | 通过（CPU primitive） | observability unit tests |

## 5. 模型/GPU/真实 TCP 可用性

- 本轮 CPU/ASGI 测试不依赖模型权重或 GPU。
- RTX 4090 D、Qwen3-0.6B、TP=1、`enforce_eager=True` 曾在 Day11–12 做过最小 HTTP 验收，但本轮 Day13–14 尚未重新执行真实 GPU 流式验证。
- 真实 TCP 断连、GPU 流式 completion/chat、GPU `/metrics` 和断连后 KV 稳定性尚未实测，不能标记为通过。
- TP>1、CUDA Graph、GPU 高并发、长时间压力、Engine 永久阻塞和进程级 supervisor 回收仍未覆盖。

## 6. 未覆盖边界与后续修复

1. 需要补真实 Uvicorn + `httpx.Client.stream()` 主动关闭连接的断连证据，确认 `client_disconnected`、pending/active 清零以及后续请求可继续服务。
2. 需要补流式首帧前 admission/step error 的真实 ASGI 错误响应测试，以及首帧后的错误关闭语义。
3. 需要补 stream queue size=1 的背压测试，确保不丢 token、不发送伪造成功 finish，并只影响单个流。
4. 需要补 TokenEvent 重复/乱序/completion_index 跳跃和无 `get_request()` Engine 兼容测试。
5. 需要补 abort/engine_error/server_shutdown 的指标状态与生命周期日志数值断言。
6. 需要补慢日志 handler 和观测异常旁路测试；当前只验证“抛异常”而未验证“阻塞很久”。
7. 需要在 GPU/模型可用时执行最小真实流式和资源验证；不将历史 Day11–12 GPU 结果替代本轮证据。

## 7. 结论

CPU 侧 Day13–14 核心 happy path、底层事件/账本、服务协议和全量回归已通过；真实 TCP/GPU 与若干高风险竞态仍需后续补证。当前验收结论应表述为“CPU/ASGI 核心功能通过，生产边界部分覆盖”，不能宣称全部最终验收项已完成。
