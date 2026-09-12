# NanoServe 文档索引

按 `plan.md` 的 20 天计划组织，当前覆盖 **Day 1–9**（Day 9 已完成实现与验收，记录见 `day9-validation.md`）。本分支从 `dev=555ef43` 创建，并已合并 dev 集成 Day6 实现（提交 `c84e73f`）。Day7 已完成实现与验收，记录见 `day7-validation.md`；Day8 已在 `feature/chunked-prefill` 分支完成实现，验收记录见 `day8-validation.md`。

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

## 其他

- [开发待办清单](./TODO.md)：记录后续阅读与实现事项。
- [上游 nano-vLLM README 存档](./nanovllm-README.md)：上游项目说明，不代表 NanoServe 后续功能已完成。
