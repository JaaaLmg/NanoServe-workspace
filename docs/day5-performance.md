# Day 5 性能测试详细报告

## 目的与口径

`benchmarks/run_baseline.py` 针对原版 nano-vLLM，以固定 token-id prompt 运行并发 1/8/32、输入 128/512/2048、输出 32/128 的矩阵。每个 case 保留逐请求时间戳、配置、环境和异常状态；聚合值不是原始数据的替代品。

- TTFT：提交后到首次 decode token 被观察到的时间。
- ITL/TPOT：相邻 decode token 间隔；当前同步 `LLMEngine` API 不暴露中间 token，因此 runner 保留字段并将聚合值标为 `null`，避免把整批耗时冒充 TPOT。
- 请求吞吐：完成请求数 / case 总时间。
- 输出吞吐：完成输出 token 数 / case 总时间。
- 显存：CUDA peak allocated/reserved；CPU 环境明确写 `available:false`。

## 测试组织逻辑（通俗解释）

可以把 baseline 想成一张“实验网格”：并发数、输入长度、输出长度分别是三条轴。脚本用三重循环把每个组合都跑一遍，例如“8 个请求、每个 prompt 512 token、每个最多生成 128 token”就是一个独立 case。这样做的好处是：改变一个维度后，可以比较它对延迟、吞吐和显存的影响，而不是把不同负载混在一起。

每个 case 的执行顺序如下：

1. **创建 engine**：用同一模型和 `max_num_seqs=concurrency` 启动原版 `LLM`。启动失败会记录为 error case。
2. **造固定输入**：用确定性的 token id 列表构造指定长度的 prompt；同一个 case 的所有请求长度一致，避免 tokenizer 随机性干扰比较。
3. **一次性提交请求**：把 concurrency 个请求加入 waiting 队列，并记录共同提交时间。
4. **逐轮驱动 engine**：反复调用 `llm.step()`。Scheduler 会先做 Prefill，再做 Decode；脚本观察每轮正在运行的请求 id，在第一次 Decode 轮记下首 token 时间，在请求完成时记下完成时间。
5. **计算指标**：由原始时间戳计算每个请求 TTFT 和总延迟，再计算 P50/P95、请求吞吐和输出吞吐。
6. **采集显存并保存**：case 结束后读取 CUDA peak allocated/reserved，把原始请求记录和聚合结果一起写入 JSON。

换句话说，脚本不是只按“开始前打一个时间、结束后打一个时间”来估算平均速度，而是尽量保留每个请求发生了什么。这样以后即使要换统计方法，也能从 `requests[]` 重新计算。

## 为什么需要这些维度

- **并发 1/8/32**：看单请求延迟如何随着批量增大变化，以及 GPU 是否能通过批处理提高吞吐。
- **输入 128/512/2048**：输入越长，Prefill 要处理的 token 越多，通常主要影响 TTFT 和显存。
- **输出 32/128**：输出越长，Decode 循环次数越多，通常更能暴露每 token 成本和总吞吐。
- **enforce-eager**：关闭 CUDA Graph 后便于做稳定、易解释的对照；如果比较 CUDA Graph，必须单独记录配置。

## 时间点为什么这样定义

`submit_time` 是请求进入 waiting 队列的时间；`first_token_time` 是第一次 Decode step 被观察到的时间；`finish_time` 是 engine 报告请求完成的时间。TTFT 就是前两者之差，端到端延迟就是提交到完成之差。当前 `LLMEngine` 是同步批式接口，不会把每个中间 token 回调给 benchmark，因此无法诚实地得到每个 token 的精确时间戳；脚本把 ITL/TPOT 保留为 `null`，而不是把整批时间错误地当成 TPOT。若未来加入流式 token 事件，只需在每个事件处追加时间戳，即可计算相邻 token 间隔。

## 如何读结果

先看 `cases[].status`，确认不是 error；再按 `concurrency/prompt_tokens/max_tokens` 找到目标组合。`summary.ttft_p95` 反映慢请求的首 token 体验，`request_throughput` 反映服务每秒完成多少请求，`output_throughput` 反映每秒生成多少 token，`memory` 反映峰值显存。不要只比较一个数字：例如长 prompt 可能让 TTFT 变差，但更高并发可能让输出吞吐提高。


```bash
python benchmarks/run_baseline.py --model /path/to/Qwen3-0.6B \
  --output benchmarks/results/baseline.json --enforce-eager
```

显存不足时显式覆盖 `--concurrency 1 8`，并在结果配置中保留实际矩阵。运行前建议确认模型目录、CUDA、PyTorch 与 tokenizer 可加载；脚本将 OOM/加载错误写入对应 case 的 `status:error` 和 `error` 字段。

## 结果结构

顶层包含 `schema_version`、`created_at`、`config`、`environment`、`cases`。每个 case 包含维度、`status`、`requests[]` 和 `summary`；请求记录含 `submit_time`、`first_token_time`、`finish_time`、`ttft`、`latency`、`output_tokens`。因此可以直接用 jq/pandas 重算 P50/P95 和吞吐。

## 解释与限制

先执行 warmup 是推荐操作，但当前脚本的每个 case 都新建 engine，首次加载时间包含在 case 总时间内，适合端到端基线但不适合纯 steady-state 对比。原版调度器 Prefill 优先，长 prompt 可能推高其他请求 TTFT；`enforce_eager` 与 CUDA Graph 会改变 Decode 延迟，必须固定配置比较。由于本仓库当前环境未提供可确认的模型权重/GPU实测结果，不能填写虚构数字；生成 JSON 后应把文件作为本报告的唯一数据源，并记录 git commit、硬件和命令行。
