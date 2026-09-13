# NanoServe 文档索引

按 `plan.md` 的 20 天计划组织，当前实现与验收覆盖 **Day 1–10、Day 11–12 与 Day 13–14**。Day13–14 已完成 SSE 流式输出、断连安全边界、TokenEvent、Prometheus 指标与结构化日志，并通过 CPU/ASGI 及 TP=1 eager 最小 GPU/TCP 验证；TP>1、CUDA Graph、高并发和长期压力等生产边界仍部分覆盖，详见 `day13-14-validation.md`。Day11–12 OpenAI 兼容 HTTP API 及其完整生产边界见 `day11-12-validation.md` 与 `day11-12-review.md`。Day 9 已完成实现与验收；Day 10 已完成实现、代码审查与 CPU 验收，GPU 主流程证据沿用已有记录，详见 `day10-validation.md` 与 `day10-review.md`。Day7 已完成实现与验收，记录见 `day7-validation.md`；Day8 已在 `feature/chunked-prefill` 分支完成实现，验收记录见 `day8-validation.md`。

## 按天索引

- [Day 1 环境记录](./env.md)：已验证的软件、硬件、模型、安装命令、启动命令和显存不足降级方案。
- [Day 1 最小闭环验收](./day1-validation.md)：三条 prompt 的实际运行结果、基础检查和验收结论。
- [Day 2 架构与调用链](./architecture.md)：模块职责表、`LLM.generate` 请求路径时序图、单轮调度流程图、关键数据结构，以及"谁创建请求 / 谁分配 KV block / 谁执行模型 / 谁决定下一轮"的解释。
- [Day 3 Paged KV Cache](./kv-cache.md)：逻辑 token → 逻辑块 → 物理 KV 块的三层映射、块生命周期图、申请/写入/复用/释放/抢占五条路径、prefix cache 机制与碎片分析（含本机实测 KV 池数据）。
- [Day 2–3 验收记录](./day2-3-validation.md)：两天的产出清单、单元测试运行结果（17 passed）和验收结论。

- [Day 4 Prefill/Decode 与采样](./prefill-decode.md)：阶段 shape、KV 数据流、瓶颈与固定 seed 采样实验。
- [采样器代码详解](./sampler.md)：SamplingParams、Sampler、seed 与推理链路的详细说明。
- [Day 5 性能测试报告](./day5-performance.md)：baseline runner 的指标口径、矩阵、原始结果 schema 与复现步骤。
- [Day 4–5 验收记录](./day4-5-validation.md)：任务逐项验收范围、命令和实测边界。
- [Day 6 请求生命周期与状态机设计](./request-lifecycle.md)：六态模型、迁移图、Scheduler/Sequence/Engine 调整方案、资源不变量、风险、测试与最终验收清单。
- [Day 6 验收记录](./day6-validation.md)：Day6 收尾与提交交付（16/16 项），99 项回归、GPU 11 组实测及未覆盖边界。
- [Day 6 实现审查](./day6-review.md)：三轮独立审查、缺陷修复关闭依据、CPU/GPU 证据及最终收尾记录。
- [Day 6 请求生命周期教程](./day6-tutorial.md)：面向推理引擎初学者的代码导读——引擎执行模型、六状态机、Sequence/Scheduler/Engine 精读、三本账一致性不变量、审查缺陷复盘与测试方法。
- [Day 7 每轮 Token Budget 与 FCFS 调度设计](./token-budget.md)：两阶段统一预算、FCFS 与既有分块边界、预算等待人数/时间、计划与执行日志、CPU/GPU 测试及验收方案。
- [Day 7 验收记录](./day7-validation.md)：预算/统计/日志实现与 68 项新测试、三档 GPU 预算 sweep、greedy 5/5 一致对比及验收清单逐项核对；原始 JSONL 证据位于 `docs/evidence/day7/`。
- [Day 7 实现审查](./day7-review.md)：对照设计复核调度逻辑、事件契约、验收工具和证据完整性；记录已补强项与尚未达标边界。
- [Day 7 Token Budget 教程](./day7-tutorial.md)：面向推理引擎初学者的代码导读——预算口径与账本观、Config 显式校验、prefill/decode 预算调度精读、等待 episode 统计与时间线验算、计划/执行证据链、四个实现缺陷复盘与测试方法。
- [Day 8 Chunked Prefill 设计](./chunked-prefill.md)：显式 `prefill_offset/chunk_size/prefill_complete` 体系、跨 chunk 位置/attention/KV 提交契约、采样行选择修正、8K 与 greedy 一致性验收及 256/512/1024 性能矩阵。
- [Day 8 验收记录](./day8-validation.md)：chunked prefill 实现范围、CPU 92 项新测试与全量 259 项回归、GPU 8K 三档分块/greedy 一致性/prefix 命中实测、greedy 数值差异分析与验收清单逐项核对；原始 JSONL 证据位于 `docs/evidence/day8/`。
- [Day 8 实现审查](./day8-review.md)：对照设计复核 chunk 调度/KV 提交/采样契约与证据完整性；记录审查补强（idle 事件字段、脚本文案）、未闭环的 `runner_input_len` 条款与可读性建议。
- [Day 8 Chunked Prefill 教程](./day8-tutorial.md)：面向推理引擎初学者的代码导读——从 Day7 隐式分块到显式进度、三个核心概念、扫描位置与队列分离、hash_blocks 显式区间、输入组装与采样契约、原子提交、GPU 一致性证据解读与思考题。
- [Day 9 混合 Prefill 与 Decode 设计](./mixed-prefill-decode.md)：混合轮 `BatchItem` 逐请求阶段标注、decode-first 调度与混合预算公式、`decode_priority` 归因重构（不变量 18 的 `needed_first` 判定）、同轮分组执行与逐 item 提交契约、`1×8K + 32×128` 混合负载 benchmark 与验收方案。
- [Day 9 实现审查](./day9-review.md)：对照设计复核混合调度/归因/采样与执行契约；记录 `warmup_model` 旧签名调用 `run()` 的 GPU 启动阻断缺陷（附修复建议）、`decode_priority` 判定与不变量 18 的偏差分析及可读性建议。
- [Day 9 验收记录](./day9-validation.md)：审查问题修复（warmup 签名 P0、归因对齐不变量 18）、CPU 303 项回归、GPU 主负载 mixed vs one-shot 对照（短请求 TTFT P50 3.57s→0.20s、decode 7/7 轮持续产出、`runner_input_len` 对账收口）、greedy 26/33 差异分析与确定性复现；原始 JSONL 证据位于 `docs/evidence/day9/`。
- [Day 9 混合 Prefill 与 Decode 教程](./day9-tutorial.md)：面向推理引擎初学者的代码导读——从"整批单阶段"的 decode 停滞到 decode-first 混合调度、`BatchItem` 逐请求标注、一轮两次前向的显式取舍、归因重构与 `needed_first` 规则、逐 item 提交与 round 关联、GPU 对照证据解读与思考题。
- [Day 10 取消、超时、抢占与恢复设计](./request-cancellation-preemption.md)：取消信号与 deadline 安全边界、KV 不足时的抢占/recompute 恢复、混合轮逐 item 隔离、异常 `finally` 清理、100 请求资源稳定性测试矩阵与验收清单。
- [Day 10 验收记录](./day10-validation.md)：实现范围与取舍（signal-only 取消/安全点收尾、victim 队尾优先与无 victim 延后、abort 事务、账本守恒检查）、修复后 CPU 377 项回归与 5 场景验收、GPU 六组矩阵（含 16 块小池 3 次自动抢占与 100 请求终态平衡），并记录本轮审查后的未覆盖边界；原始 JSONL 证据位于 `docs/evidence/day10/`。
- [Day 10 实现审查](./day10-review.md)：逐项复核状态/控制面、抢占恢复、异常清理、KV 账本、序列化与事件契约；记录最终修复项、377 项 CPU 回归、TP=1 GPU 六组证据和 TP>1/CUDA Graph/真实并发等未覆盖边界。
- [Day 10 教程](./day10-tutorial.md)：面向初学者讲解三种进度、六态生命周期、取消与 deadline 安全点、KV 抢占恢复、异常事务、账本所有权、事件审计、测试证据与后续服务化衔接。
- [Day 11–12 OpenAI 兼容 HTTP API 设计](./openai-api-day11-12.md)：合并规划 `/v1/completions` 与 `/v1/chat/completions`、统一内部请求、单 Engine worker、完成记录、错误/生命周期契约、CPU/GPU 测试和验收方案；SSE 明确留给 Day13。
- [Day 11–12 验收记录](./day11-12-validation.md)：实现范围、修复后的 CPU 117 项服务测试与全量 495 项回归（含 `python -O`）、CPU ASGI 18 项与 GPU RTX 4090 D TP=1 eager 严格 HTTP 8 项验收；区分脚本证据、手工观察和未覆盖生产边界，原始 JSONL 位于 `docs/evidence/day11-12/`。
- [Day 11–12 实现审查](./day11-12-review.md)：对照设计文档复核服务、worker、完成记录、上下文校验、并发/关闭生命周期和验收脚本；记录修复项、代码简化建议及 TP>1、CUDA Graph、SSE/断连、GPU 504 和长期压力等未覆盖边界。
- [Day 11–12 教程](./day11-12-tutorial.md)：面向初学者讲解 HTTP 到 Engine 的完整请求链路、统一 InternalRequest、chat template、完成记录、单 worker 批处理、Future/取消/关闭、测试方法、典型缺陷和后续 Day13/14 衔接。
- [Day 13–14 SSE 与可观测性设计](./sse-observability-day13-14.md)：合并规划 SSE 首/增量/finish/[DONE]、客户端断连与 KV 清理、独立 TokenEvent、Prometheus 指标、请求级结构化日志、测试矩阵和 CPU/TCP/GPU 验收方案；设计基线与实现结果分别以本文和 [验收记录](./day13-14-validation.md) 为准。
- [Day 13–14 验收记录](./day13-14-validation.md)：记录 SSE、断连、Prometheus 指标、结构化日志的实际测试命令、结果、CPU/模型/GPU 可用性与未覆盖边界。
- [Day 13–14 教程](./day13-14-tutorial.md)：面向初学者讲解从非流式 API 到 TokenEvent/SSE、断连安全点、KV/prefix 账本、Prometheus 指标、结构化日志、测试分层和常见并发问题。

## 其他

- [开发待办清单](./TODO.md)：记录后续阅读与实现事项。
- [上游 nano-vLLM README 存档](./nanovllm-README.md)：上游项目说明，不代表 NanoServe 后续功能已完成。
