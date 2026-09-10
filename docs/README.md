# NanoServe 文档索引

按 `plan.md` 的 20 天计划组织，当前覆盖 **Day 1–6**。

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

## 其他

- [开发待办清单](./TODO.md)：记录后续阅读与实现事项。
- [上游 nano-vLLM README 存档](./nanovllm-README.md)：上游项目说明，不代表 NanoServe 后续功能已完成。
