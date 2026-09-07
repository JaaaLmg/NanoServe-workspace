# NanoServe 文档索引

按 `plan.md` 的 20 天计划组织，当前覆盖 **Day 1–3**。

## 按天索引

- [Day 1 环境记录](./env.md)：已验证的软件、硬件、模型、安装命令、启动命令和显存不足降级方案。
- [Day 1 最小闭环验收](./day1-validation.md)：三条 prompt 的实际运行结果、基础检查和验收结论。
- [Day 2 架构与调用链](./architecture.md)：模块职责表、`LLM.generate` 请求路径时序图、单轮调度流程图、关键数据结构，以及"谁创建请求 / 谁分配 KV block / 谁执行模型 / 谁决定下一轮"的解释。
- [Day 3 Paged KV Cache](./kv-cache.md)：逻辑 token → 逻辑块 → 物理 KV 块的三层映射、块生命周期图、申请/写入/复用/释放/抢占五条路径、prefix cache 机制与碎片分析（含本机实测 KV 池数据）。
- [Day 2–3 验收记录](./day2-3-validation.md)：两天的产出清单、单元测试运行结果（17 passed）和验收结论。

## 其他

- [上游 nano-vLLM README 存档](./nanovllm-README.md)：上游项目说明，不代表 NanoServe 后续功能已完成。

Day 4 及之后的 prefill/decode 对比、服务接口、benchmark 等文档将在对应任务完成后再补充。
