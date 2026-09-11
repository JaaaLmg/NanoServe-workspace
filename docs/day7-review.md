# Day 7 实现审查记录

> 审查日期：2026-09-11。本文对照 [token-budget.md](./token-budget.md) §3–§10 检查当前未提交工作区；不替代设计文档，也不扩大 Day7 范围。

## 1. 审查范围与实际命令

检查了 `nanovllm/config.py`、`nanovllm/engine/scheduler.py`、`nanovllm/engine/llm_engine.py`、`tests/test_token_budget.py`、`scripts/validate_token_budget.py`、已有生命周期/KV 测试及 `docs/evidence/day7/` JSONL。

本次实际执行：

- `python -m pytest tests/test_token_budget.py -q`：68 passed。
- `python scripts/validate_token_budget.py --mode cpu --output /tmp/day7-review-cpu.jsonl`：exit 0，15 轮、最大 executed 8、decode 预算延后 5 轮、KV free/used=64/0、PASS。
- GPU、本次全量回归和 `python -O` 回归尚未在本轮重新执行；`day7-validation.md` 中的历史数字是实施记录，不作为本次重新执行结果。

## 2. 已确认正确的实现

- `max_num_batched_tokens` 与 `max_num_seqs` 使用集中式正整数校验，明确拒绝 bool/float/str/非正值，允许 `B < max_num_seqs`。
- Prefill 首候选保留已有 chunking；后续候选严格阶段内 FCFS，不跳过队首；prefix 命中、分块和 resume 按未缓存输入估算。
- Decode 在出队、`may_append` 和抢占前检查预算/sequence cap；预算延后不释放 KV、不改 token、不改变相对顺序。
- 七类调度原因、预算等待 episode、终态所有权和 request_id 复用保护的主路径逻辑清晰，正常路径测试覆盖较充分。
- `schedule()` tuple、`step()` 正/负工作量、空批次和模型异常的既有接口语义保持。

## 3. 本次补强

本轮进一步完成了异常重试保护和验收资源硬校验；模型异常后同一 Engine 的后续 `step()` 会明确拒绝，验收脚本会将最终活动索引、KV used/ref_count、free 列表不一致判为失败。GPU/CPU 收尾命令均已重新执行并生成新证据。

- 为 `budget_wait_episode`、`request_budget_wait` 和三种 `engine_round` 事件补充 `observed_at`，并同步更新验收脚本字段白名单。
- Scheduler/Engine 轮次验证增加 `round_id` 对应关系以及 phase/planned token 对账。
- Scheduler 和 Engine 的非空批次检查补充“必须为真正的正整数”约束，避免 bool、浮点数等伪造计数绕过防御校验。
- 验收驱动达到轮次上限时改为失败，不再取消剩余请求后继续报告 PASS；greedy 对比缺失任一输出时显式失败。

## 4. 尚未达到完整验收证据标准的问题

以下项目不应在当前状态下被表述为“已完全证明”：

1. 计划/执行的 query token 仍主要来自同一份 `num_scheduled_tokens` 快照；尚未在验收脚本中直接核对真实 `prepare_prefill` 展平 input_ids 长度及 decode runner 成员。
2. 当前历史日志验收工具缺少对所有摘要/episode 的完整交叉重算（例如从 decisions 重建 direct/HOL、从 episode 重建全局累计），本次仅补了轮次配对，仍需进一步增强。
3. GPU 执行期间取消/超时、动态到达等 §8.3 专项场景本轮未重新实测；TP>1/CUDA Graph 仍未测，符合设计中的非强制边界但必须保持未覆盖标注。
4. 原有日志测试中 sentinel 未真正作为请求输入，且没有可靠构造非零 episode 与 idle 事件；该测试保护能力偏弱，后续应补真实输入和非空断言。
5. 模型异常后直接再次调用 `step()` 的恢复/故障锁定未实现。设计明确不要求 Day7 新建 FAILED 状态，因此本轮只保留 error 事件，不声称支持异常后安全重试。
6. 长序列随机测试的资源收尾断言仍应增加 `used == 0`、ref_count 全清零和 free 列表无重复等硬条件。

## 5. 当前风险与取舍

小预算下持续 prefill 可能压住 decode，running 队列仍是阶段内 FCFS，Day7 不承诺公平延迟；token budget 也不是 KV 显存硬上限。队列的对象查找/移除仍存在高请求量下的线性重复扫描风险，但不影响当前功能正确性，建议后续单独优化。空 prompt、超上下文长度和非整数 `max_tokens` 等既有输入边界也未在本轮扩展。

## 6. 审查结论

核心调度正常路径基本符合设计，但在补齐独立输入证据、完整日志交叉校验和 GPU 边界实测前，不能把 §10 全部项目认定为无条件通过。`token-budget.md` §10 已按“实现/证据均充分”原则更新；未勾选项目代表待补证据或明确未覆盖，不代表已发现必然的正常路径越界。
