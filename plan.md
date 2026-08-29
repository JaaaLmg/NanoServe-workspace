# NanoServe：基于 nano-vLLM 的 20 天实践计划

## 1. 项目定位

### 项目名称

**NanoServe：基于 nano-vLLM 的长上下文、高并发 LLM 推理服务引擎**

### 项目目标

在理解 nano-vLLM 核心实现的基础上，扩展一个可运行、可测试、可测量的轻量推理服务：

```text
HTTP 请求
   ↓
Tokenizer / Chat Template
   ↓
Request Queue
   ↓
Continuous Batching Scheduler
   ↓
Chunked Prefill / Decode
   ↓
Paged KV Cache
   ↓
Streaming Response
```

20 天结束时，应能展示一个 GitHub 项目、可复现的 benchmark 和一份性能分析报告，而不只是源码阅读笔记。

### MVP 范围（必须完成）

- 理解并能运行 [GeeeekExplorer/nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- 请求生命周期：排队、运行、完成、取消、超时、抢占与恢复
- Continuous Batching（连续批处理）
- Chunked Prefill（分块预填充）
- OpenAI 兼容接口：`/v1/completions` 与 `/v1/chat/completions`
- SSE 流式输出
- Prometheus 指标和结构化日志
- 正确性测试、压力测试和统一 benchmark

### Stretch Goal（有余力再做）

- Prefix Cache 的 LRU/LFU 淘汰
- 请求优先级或自适应 chunk size
- 简单投机解码（参考 [nanovllm-dspark](https://github.com/babyinsunshine/nanovllm-dspark)）
- INT8/FP8 权重加载

投机解码、量化和 PD 分离不要放进 MVP，以免影响 20 天交付。

## 2. 环境与基线

### 推荐环境

- Linux 或 WSL2（nano-vLLM 依赖 CUDA、Triton、FlashAttention；Windows 下 Tensor Parallelism 可能受 NCCL 限制）
- Python 3.10–3.12
- PyTorch ≥ 2.4、Triton ≥ 3.0、Transformers ≥ 4.51
- NVIDIA GPU；没有本地 GPU 时使用云 GPU 或 Colab

### 模型选择

| 显存 | 建议模型 | 用途 |
| --- | --- | --- |
| 8 GB | Qwen3-0.6B | 日常开发、单元测试、快速 benchmark |
| 12–16 GB | Qwen2.5-1.5B | 更接近真实服务负载的实验 |
| 更高 | Qwen3-8B 等 | 性能展示（不是完成 MVP 的前提） |

第 1 天先固定一个模型和采样配置；后续 benchmark 不要随意改变硬件、模型、prompt 集和采样参数。

## 3. 20 天详细计划

建议每天投入 4–6 小时。每一天都保留代码、实验数据或笔记，晚上做一次可运行验收。

### 阶段一：环境、源码与基线（Day 1–5）

#### Day 1：搭建环境并跑通最小闭环

**任务**

1. 准备 Linux/WSL2、CUDA、Python 虚拟环境。
2. 安装 PyTorch、Transformers、Triton、FlashAttention、xxhash。
3. 克隆并安装 nano-vLLM，确认 GPU、CUDA 和 Triton 版本可用。
4. 下载 Qwen3-0.6B（或显存允许的模型），运行官方示例生成文本。
5. 记录显卡型号、驱动、CUDA、PyTorch、Triton、模型版本。

**产出与验收**

- `env.md`：环境版本和安装命令。
- 一条可重复执行的启动命令。
- 至少 3 个 prompt 能稳定生成结果；显存不足时记录降级方案。

#### Day 2：画出端到端调用链

**任务**

1. 从 `LLM.generate` 入口开始跟踪调用。
2. 阅读 Engine、Scheduler、ModelRunner、Tokenizer、SamplingParams 的职责。
3. 标出一次请求从进入队列到返回 token 的关键数据结构。
4. 对照代码画出“请求路径”和“单轮调度路径”。

**产出与验收**

- `docs/architecture.md`：模块职责和调用时序图。
- 能用自己的话解释：谁创建请求、谁分配 KV block、谁执行模型、谁决定下一轮。

#### Day 3：理解 Paged KV Cache

**任务**

1. 阅读 Sequence、Block、BlockManager 和 prefix cache 相关实现。
2. 跟踪 block 的申请、写入、复用、释放和 preemption 路径。
3. 画出逻辑 token、逻辑 block、物理 KV block 的映射关系。
4. 写一个小测试，验证 block 不足时的返回值和释放后可复用性。

**产出与验收**

- `docs/kv-cache.md`：KV Cache 生命周期图。
- block 分配/释放单元测试通过；能解释为什么分页 KV Cache 能减少碎片。

#### Day 4：区分 Prefill、Decode 与采样

**任务**

1. 阅读模型执行入口，定位 Prefill 和 Decode 的分支。
2. 记录两阶段的输入 shape、KV 写入方式和计算瓶颈。
3. 阅读 greedy、temperature、top-p、max tokens 的处理逻辑。
4. 用固定 seed 对比 greedy 与采样输出，确认参数确实生效。

**产出与验收**

- `docs/prefill-decode.md`：阶段对比表和数据流图。
- 至少 5 个 prompt 的 greedy 输出可复现；采样参数测试通过。

#### Day 5：建立原版 baseline benchmark

**任务**

1. 编写 `benchmarks/run_baseline.py`，统一记录请求时间、首 token 时间、每 token 时间和总吞吐。
2. 测试并发数 1、8、32（显存不足时降低上限）。
3. 测试输入长度 128、512、2K，输出长度 32、128。
4. 保存原始数据和运行配置，不只保留平均值。

**产出与验收**

- `benchmarks/results/baseline.json` 或 CSV。
- 得到原版 nano-vLLM 的 TTFT、TPOT/ITL、请求吞吐、输出吞吐和显存占用基线。

### 阶段二：调度器与 Chunked Prefill（Day 6–10）

#### Day 6：定义请求状态机

**任务**

1. 设计 Request 对象：请求 ID、prompt token、已生成 token、最大新 token 数、时间戳和取消标记。
2. 定义 `WAITING`、`RUNNING`、`PREEMPTED`、`FINISHED`、`CANCELLED`、`TIMEOUT` 状态。
3. 明确每种状态允许的迁移和资源释放动作。
4. 为非法迁移增加显式异常或保护。

**产出与验收**

- `docs/request-lifecycle.md`：状态迁移图。
- 状态迁移、重复完成、取消运行中请求等测试通过。

#### Day 7：加入每轮 token budget

**任务**

1. 在 scheduler 中增加 `max_num_batched_tokens` 或等价配置。
2. 统计每个 waiting/running 请求本轮所需 token 数。
3. 实现 FCFS 选择逻辑：预算不足时延后请求，不突破上限。
4. 记录因预算不足而等待的请求数量和等待时间。

**产出与验收**

- scheduler 单测覆盖空队列、单请求、预算刚好、不足和多请求场景。
- 日志中能证明每一轮实际 token 数不超过预算。

#### Day 8：实现 Chunked Prefill

**任务**

1. 为长 prompt 增加 `prefill_offset`、`chunk_size` 和完成标志。
2. 将一个长 prompt 拆成多个 iteration，逐块写入 KV Cache。
3. 确保同一请求的 token 顺序、位置编码和 attention mask 正确。
4. 对比 chunk size 为 256、512、1024 的执行时间和显存峰值。

**产出与验收**

- 8K prompt 能被切块完成，且不会 OOM。
- Chunked Prefill 与一次性 Prefill 在 greedy 模式下输出一致。

#### Day 9：混合 Prefill 与 Decode

**任务**

1. 设计一轮调度中同时包含 prefill chunk 和 decode token 的 batch 表示。
2. 优先保障已有 decode 请求，避免长 prefill 独占 GPU。
3. 处理新请求动态加入、请求完成和 KV block 变化。
4. 构造“1 个 8K prompt + 32 个 128-token prompt”的混合负载。

**产出与验收**

- 混合负载下 decode 请求可持续产出 token。
- 相比原始一次性 Prefill，短请求 P95 TTFT/TPOT 有可解释的变化。

#### Day 10：取消、超时、抢占与恢复

**任务**

1. 为请求增加 deadline 和取消信号。
2. 实现等待队列取消、运行中取消和客户端断开后的清理。
3. 显存不足时选择一个 running 请求进行抢占，释放其可回收 block。
4. 恢复被抢占请求，验证 prompt 进度和已生成 token 不丢失。
5. 对异常路径加 finally 清理，避免 engine 卡死或资源泄漏。

**产出与验收**

- 取消、超时、抢占、恢复测试通过。
- 连续运行至少 100 个请求后，KV Cache 使用率回到稳定范围，没有持续增长。

### 阶段三：服务化与可观测性（Day 11–14）

#### Day 11：实现 `/v1/completions`

**任务**

1. 选择 FastAPI 或 Starlette，封装 engine 生命周期。
2. 定义 OpenAI 风格请求与响应模型。
3. 支持 `prompt`、`max_tokens`、`temperature`、`top_p`、`stream` 等核心字段。
4. 增加健康检查、模型信息和优雅关闭。

**产出与验收**

- `curl` 可以完成一次非流式 completion。
- 参数错误返回清晰的 4xx；服务停止时正在运行的请求能安全结束或取消。

#### Day 12：实现 `/v1/chat/completions`

**任务**

1. 处理 `system`、`user`、`assistant` 消息列表。
2. 使用 Transformers tokenizer 的 chat template，避免手写格式和模型不一致。
3. 统一 completion 与 chat 请求在内部的 Request 表示。
4. 用 OpenAI Python Client 指向本地服务进行调用。

**产出与验收**

- 普通 chat 请求可用 OpenAI Python Client 调通。
- 缺少 messages、非法 role、超出上下文长度时返回可理解的错误。

#### Day 13：SSE 流式输出与断连处理

**任务**

1. 将每个新 token 封装为 SSE `data:` 事件。
2. 发送首个 chunk、增量文本、finish reason 和 `[DONE]`。
3. 检测客户端断开，触发 scheduler 取消并释放 KV Cache。
4. 增加非流式与流式最终文本一致性测试。

**产出与验收**

- `curl -N` 或浏览器能实时看到 token。
- 客户端中途断开后，请求不会继续占用资源。

#### Day 14：Prometheus 指标与结构化日志

**任务**

1. 增加 `/metrics` 接口和请求级 trace/request ID。
2. 记录队列等待、TTFT、TPOT/ITL、端到端延迟。
3. 记录 prompt/generation token 数、running requests、KV Cache 使用率。
4. 记录 prefix cache 命中/未命中（若沿用 nano-vLLM prefix cache）。
5. 输出 JSON 或 key-value 结构化日志，便于后续分析。

**产出与验收**

- `/metrics` 能看到以下指标或等价指标：

  - `request_queue_time_seconds`
  - `time_to_first_token_seconds`
  - `time_per_output_token_seconds`
  - `request_latency_seconds`
  - `prompt_tokens_total`
  - `generation_tokens_total`
  - `kv_cache_utilization`
  - `prefix_cache_hit_rate`
  - `running_requests`

- 至少一条请求能在日志中串起“接收—调度—首 token—完成”的全过程。

### 阶段四：Benchmark、测试与交付（Day 15–20）

#### Day 15：统一对比 HuggingFace、nano-vLLM 与 vLLM

**任务**

1. 统一模型、tokenizer、prompt 集、采样参数和停止条件。
2. 为 HuggingFace Transformers、原版 nano-vLLM、NanoServe、官方 vLLM 编写相同的适配器。
3. 测量 TTFT P50/P95、TPOT/ITL P50/P95、请求吞吐、输出吞吐、显存占用。
4. 保留启动参数、commit、硬件和原始结果。

**产出与验收**

- `benchmarks/run_all.py` 一键运行。
- `benchmarks/results/` 有 JSON/CSV 原始数据和生成时间戳。

#### Day 16：Profiling 与瓶颈定位

**任务**

1. 用 PyTorch Profiler 观察 CPU 调度、GPU kernel、同步和数据拷贝。
2. 在代表性负载下分别分析 prefill、decode 和 HTTP 层。
3. 如条件允许，使用 Nsight Systems/Compute 检查 kernel 时间线和显存传输。
4. 将每个瓶颈写成“现象—原因假设—验证方法”。

**产出与验收**

- `docs/profiling.md`：至少一张时间线/火焰图和 3 个瓶颈结论。
- 每个优化项都能说明它影响的是 TTFT、TPOT、吞吐还是显存。

#### Day 17：针对性优化并复测

**任务**

1. 根据 Day 16 结果选择 1–3 个低风险优化：scheduler 批组织、CPU-GPU 拷贝、chunk size、日志开销或 batch 构造。
2. 每次只改一个变量，保留优化前后的 commit/配置。
3. 重跑相同测试矩阵，检查是否出现正确性回归。
4. 汇总优化前后对比，不预先承诺性能数字。

**产出与验收**

- `benchmarks/results/optimization.csv`。
- README 中有真实测得的提升或退化，以及适用条件和限制。

#### Day 18：正确性与压力测试

**任务**

1. 准备至少 100 个 prompt，覆盖短文本、长文本、中文、英文、重复前缀和边界长度。
2. 对比 HuggingFace 或可靠参考实现的 greedy 输出。
3. 检查流式/非流式最终结果一致、sampling 参数生效、取消后 KV Cache 释放。
4. 做长时间压力测试，随机加入、完成、取消和超时请求。

**产出与验收**

- `tests/` 包含单元、集成和压力测试入口。
- Greedy 结果与参考实现一致（或对已知差异作出说明）。
- 连续压力测试无死锁、请求泄漏或显存持续上涨。

#### Day 19：完善文档与一键启动

**任务**

1. 完善 README：项目动机、架构、安装、API 示例、benchmark 方法和限制。
2. 添加架构图、请求时序图和关键设计取舍。
3. 提供 `scripts/start_server.*`、示例请求和测试命令。
4. 增加 Dockerfile 或明确的 WSL2/云 GPU 安装脚本。
5. 从全新环境验证“30 分钟内跑通”。

**产出与验收**

- 新用户按 README 能启动服务、发起请求并运行最小 benchmark。
- 所有链接、命令、模型路径和环境变量经过实际验证。

#### Day 20：发布、演示与简历化

**任务**

1. 冻结一个可展示的 release/commit，清理临时文件和无关实验。
2. 录制 2–3 分钟演示：启动服务、chat 请求、流式输出、指标页面。
3. 整理性能表、profiling 结论和已知限制。
4. 写一页技术总结：问题、设计、实现、实验、下一步。
5. 根据真实数据编写简历 bullet，不填未经测量的提升数字。

**产出与验收**

- 可公开展示的 GitHub 仓库、README、benchmark 结果和演示视频。
- 一段能在面试中讲清楚的 3 分钟项目介绍。

## 4. Benchmark 设计

### 推荐测试矩阵

| 变量 | 取值 |
| --- | --- |
| 并发数 | 1、8、32、64（按显存调整） |
| 输入长度 | 128、512、2K、8K |
| 输出长度 | 32、128、512 |
| 请求模式 | 固定并发、Poisson 到达 |
| Prefix | 无共享、50% 共享、80% 共享 |
| Prefill | 一次性、chunk size 256/512/1024 |

### 必报指标

- TTFT（Time to First Token）P50/P95/P99
- TPOT/ITL（Time Per Output Token / Inter-Token Latency）P50/P95
- Decode throughput、request throughput
- 端到端延迟
- GPU 显存峰值与 KV Cache 使用率
- Prefix Cache 命中率
- 取消、抢占和恢复的成功率

结果必须同时保存原始数据、运行配置和硬件信息；简历只填写实际测得的数据。

## 5. 最终验收清单

### 正确性

- [ ] Greedy decoding 与参考实现一致，差异有说明。
- [ ] `temperature`、`top_p`、`max_tokens` 生效。
- [ ] 流式和非流式最终结果一致。
- [ ] 请求取消后 KV Cache 正确释放。
- [ ] 抢占后的请求可以恢复并继续生成。

### 服务与工程质量

- [ ] `/v1/completions` 可用。
- [ ] `/v1/chat/completions` 可用。
- [ ] SSE 流式输出可用，客户端断开能清理请求。
- [ ] `/metrics` 可抓取，日志可关联单个请求。
- [ ] 单元测试、集成测试、压力测试可重复运行。
- [ ] Docker 或一键安装/启动脚本可用。

### 性能与文档

- [ ] 有 HuggingFace、nano-vLLM、NanoServe、vLLM 的统一对比。
- [ ] 有 profiling 证据和至少一轮优化前后对比。
- [ ] README、架构图、benchmark 说明和限制说明完整。
- [ ] 新用户能在约 30 分钟内跑通最小示例。

## 6. 学习资源

### 必读顺序

1. [vLLM GitHub README](https://github.com/vllm-project/vllm)：了解完整项目定位和组件边界。
2. [vLLM Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)：建立 Engine、Scheduler、Worker/ModelRunner 的整体心智模型。
3. [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：理解分页 KV Cache 的动机、映射和性能收益。
4. [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：主线实现，按 Day 2–4 的调用链阅读。
5. [MinivLLM 教程](https://github.com/Wenyueh/MinivLLM/blob/main/HowToApproachvLLM.md)：用更小的实现补足 Paged Attention、Flash Attention 和推理流程。
6. [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)：理解 prefix block 哈希、命中和淘汰思路。
7. [vLLM V1 Scheduler 源码](https://github.com/vllm-project/vllm)：完成 MVP 后再深入调度策略和抢占细节。

### 按主题补充

| 主题 | 资源 | 建议阅读日 |
| --- | --- | --- |
| Paged KV Cache | [PagedAttention 论文](https://arxiv.org/abs/2309.06180)、nano-vLLM `block_manager` | Day 3 |
| Attention Kernel | [MinivLLM](https://github.com/Wenyueh/MinivLLM)、[Flex-nano-vLLM](https://github.com/changjonathanc/flex-nano-vllm) | Day 3–4、选做 |
| 调度与 Chunked Prefill | [nano-vLLM-v1](https://github.com/slwang-ustc/nano-vllm-v1)、vLLM scheduler 源码 | Day 6–10 |
| 服务 API | [OpenAI API Reference](https://platform.openai.com/docs/api-reference)、[FastAPI 文档](https://fastapi.tiangolo.com/) | Day 11–13 |
| SSE | [MDN Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) | Day 13 |
| 监控 | [Prometheus Python Client](https://github.com/prometheus/client_python) | Day 14 |
| Profiling | [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)、[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | Day 16 |
| 投机解码（选做） | [nanovllm-dspark](https://github.com/babyinsunshine/nanovllm-dspark) | Day 18–20 之后 |

### 可选对比项目

- [nano-vLLM-v1](https://github.com/slwang-ustc/nano-vllm-v1)：偏调度器、Chunked Prefill 和抢占研究。
- [MinivLLM](https://github.com/Wenyueh/MinivLLM)：偏 Attention、Triton 和教学实现。
- [Flex-nano-vLLM](https://github.com/changjonathanc/flex-nano-vllm)：硬件环境不稳定时，使用 PyTorch FlexAttention 的较轻量替代方案。

## 7. 简历表述模板

完成并测出真实数据后，可按下面格式改写：

> **NanoServe — 基于 nano-vLLM 的高并发 LLM 推理服务引擎**：基于 PyTorch/Triton 扩展 nano-vLLM，实现 Continuous Batching、Chunked Prefill、Paged KV Cache 和 OpenAI-Compatible Streaming API；设计请求生命周期、抢占恢复与 KV Cache 管理机制；构建统一 benchmark，对比 HuggingFace、nano-vLLM 与 vLLM 的 TTFT、TPOT、吞吐和显存占用。

带数据时使用真实测量结果：

> 在 **[模型]、[GPU]、并发 [N]** 的混合长短请求负载下，相比原版 nano-vLLM 将 P95 TTFT 从 **[A] ms** 变为 **[B] ms**，输出吞吐达到 **[C] tok/s**，并将 KV Cache 使用率稳定在 **[D]%**。

不要填写尚未实测的性能提升；如果结果在某些负载下退化，也应在 README 中诚实说明原因和适用范围。
