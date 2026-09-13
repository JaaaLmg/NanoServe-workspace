# Day 13–14 SSE 与可观测性验收记录

> 本记录对应 [Day 13–14 设计基线](./sse-observability-day13-14.md)。设计文档的验收清单不在此修改；本文件只记录实际执行结果。

## 1. 分支与基线

- 分支：`feature/sse-observability-day13-14`
- Day12 基线：`1de08bc`（`merge: integrate Day11-12 OpenAI APIs`）
- 基础实现提交：`35ceac1`
- 本轮加固与教程提交：待本记录完成后创建
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

### 3.3 验收加固测试

```bash
python -m pytest tests/test_day13_14_hardening.py -q
```

结果：`4 passed`，覆盖旧世代 token 不误取消、未绑定 seq 的记录不误收口、有限等待超时清理和 admission 时 prompt 计数。

### 3.4 当前工作区修复后的服务回归

```bash
python -m pytest tests/test_day13_14_hardening.py tests/test_engine_worker.py \
  tests/test_service_api.py tests/test_service_sse.py tests/test_stream_disconnect.py -q
```

结果：`64 passed, 1 warning`。

### 3.5 关键服务回归

```bash
python -m pytest tests/test_service_api.py tests/test_engine_worker.py \
  tests/test_observability_unit.py -q
```

结果：`61 passed, 1 warning`（基础实现阶段）。

### 3.6 全量 CPU 回归

```bash
python -m pytest -q
```

结果：`520 passed, 1 warning`。

```bash
python -O -m pytest -q
```

结果：`520 passed, 2 warnings`。

### 3.7 CPU OpenAI/SSE 验收脚本

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
| request ID + seq ID 事件世代隔离 | 通过（CPU） | `test_day13_14_hardening.py` 覆盖旧世代事件不误取消新请求 |
| 首 token 前、token 间、完成竞争断连 | 部分通过 | CPU worker/ASGI 覆盖；真实 TCP 只验证主动关闭后的恢复，不可观测服务内 cancel 原因 |
| 断连触发 `client_disconnected` 并释放 active/pending/KV | 部分通过 | CPU worker 路径通过；真实 TCP 后续请求恢复通过，但内部 cancel/KV 快照未从外部进程直接读取 |
| Engine error/timeout/shutdown 不伪造成功 finish | 通过（CPU/ASGI） | 首帧前 admission error、终态错误和有限等待由加固测试覆盖 |
| `step()`/`generate()`/非流式 API 兼容 | 通过（CPU） | 全量 520 passed |

### 4.2 Day14 指标与结构化日志

| 项目 | 状态 | 证据/说明 |
| --- | --- | --- |
| `/metrics` 可抓取并包含 9 个计划指标 | 通过（CPU/ASGI） | `tests/test_observability.py` |
| queue wait、TTFT、TPOT/ITL、latency 口径 | 通过（primitive） | `tests/test_observability*.py`；统一 `perf_counter` |
| 无首 token/少于两个 token 不伪造 TTFT/TPOT | 通过（primitive） | timeline 测试 |
| prompt/generation token 计数 | 部分通过 | admission prompt 与正常生成已验证；中止/Engine error 的完整 token usage 仍按实际 TokenEvent 口径统计 |
| running、KV used/total/utilization | 部分通过 | 底层 resource snapshot 通过；HTTP metrics 当前主要暴露 running/utilization |
| prefix hit/miss/capacity failure 统计 | 通过（CPU） | 底层分类与 snapshot counter 差分/重复抓取幂等已覆盖 |
| lifecycle JSON envelope 与隐私白名单 | 通过（CPU primitive） | `test_structured_log_drops_sensitive_fields` |
| received→submitted→admitted→first token→finished/aborted 可串联 | 部分通过 | CPU happy path 与 service rejection 已接线；真实 TCP 进程内日志链未单独导出 |
| 观测异常不阻塞 Engine | 部分通过 | 抛异常隔离有测试；慢 handler、永久阻塞未实测 |
| 低基数 labels、registry 隔离 | 通过（CPU primitive） | observability unit tests |

## 5. 模型/GPU/真实 TCP 可用性

- 本轮 CPU/ASGI 测试不依赖模型权重或 GPU。
- RTX 4090 D、Qwen3-0.6B、TP=1、`enforce_eager=True` 在本轮已重新执行最小真实 HTTP 验收。
- GPU completion/chat SSE、GPU `/metrics` 和真实 TCP 主动关闭后的后续请求恢复已通过；内部 cancel reason 与断连后 KV 快照未从外部进程直接读取。
- TP>1、CUDA Graph、GPU 高并发、长时间压力、Engine 永久阻塞和进程级 supervisor 回收仍未覆盖。

## 6. 未覆盖边界与后续修复

1. 真实 TCP 主动关闭后的后续请求恢复已验证，但服务内 `client_disconnected` cancel reason、pending/active 和 KV 快照尚未通过独立管理端点导出，仍需进程内证据。
2. TP>1、CUDA Graph、GPU 高并发和长时间压力尚未实测。
3. 慢日志 handler 的阻塞隔离、Engine 永久阻塞和进程级 supervisor 回收仍未覆盖。
4. Aborted 请求的完整 token usage 只按实际 TokenEvent 统计，未扩展底层 AbortedRequest 的 token 字段。

## 7. 结论

CPU 侧 Day13–14 核心 happy path、底层事件/账本、服务协议和全量回归已通过；真实 TCP/GPU 与若干高风险竞态仍需后续补证。当前验收结论应表述为“CPU/ASGI 核心功能通过，生产边界部分覆盖”，不能宣称全部最终验收项已完成。
