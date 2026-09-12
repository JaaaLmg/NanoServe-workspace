# Day 9 混合 Prefill/Decode 验收记录

> 对应设计文档 [mixed-prefill-decode.md](./mixed-prefill-decode.md) 与审查报告 [day9-review.md](./day9-review.md)。本文记录实现与修复范围、实际执行的命令与结果、设计取舍及未覆盖边界；设计文档中的验收清单（§10）保持原文，完成情况在 §7 逐项核对。原始事件证据位于 `docs/evidence/day9/`。

## 1. 实现范围与分支状态

- 分支：`feature/mixed-prefill-decode`（基线 dev=`42e142d`，含 Day8），**全部改动留在工作区，未提交、未合并**。
- 按设计文档完成的实现（第一轮，详见审查报告 §2 确认项）：`BatchItem` 混合批次表示、decode-first 调度与混合预算公式、`decode_priority` 归因重构、`ModelRunner.run(items)` 同轮分组执行、`postprocess` 逐 item 提交（round_id 关联校验）、Engine 接口与事件契约、`Sequence.is_last_chunk_scheduled` 具名谓词。
- 本轮按审查报告修复：
  1. **§3.1 P0**：`ModelRunner.warmup_model` 适配 `run(items)` 签名——warmup 语义与 Day8 逐位一致（`prefill_offset=0`、`num_scheduled_tokens=seq_len`、`prefill_target=seq_len`，每条序列构造为最后 chunk 的 `BatchItem`，全部行采样；`round_id=0` 占位不消费）。
  2. **§4.1 归因对齐设计（采纳审查建议一）**：`decode_priority` 判定从"当前被延后候选自己的需求"改为**不变量 18 的字面规则 `D > 0 且 needed_first <= B`**——`needed_first` 是本轮首个被考察的 prefill 候选的需求（含未被接纳即停止的情形），在 prefill 扫描第一次成功估算时记录；`remaining==0` 分支若该候选即首候选则就地补一次只读估算（KV 不足仍记 `kv_capacity`），非首候选路径照旧估算。`scheduler_round` 事件新增 `needed_first` 字段供验收方独立复算；被延后候选自身的需求在两条路径上分别如实入档（首候选路径）或记 `None` 不虚构（非首候选路径，见 §4 取舍 2）。
  3. **§4.2 可读性**：`run_main_workload` 返回注解修正；`postprocess` 的 `has_live` 谓词复用；`llm_engine.step()` 引入 `_ItemSnapshot` NamedTuple 快照 + 分阶段计数局部变量（消除 5 处位置解包与 3 处重复计数表达式）；decode 阶段（含抢占循环）提取为 `_schedule_decode_phase` 私有方法，主流程只剩两阶段骨架。
- 配套：`validate_mixed_batch.py` 复算口径同步（`decode_priority` 判定按 `needed_first` 复算）、3 个脚本事件白名单版本化（`needed_first`）；新增审查反例单测 2 条（§3.4）；GPU 模式（§9.4 主负载对照 + runner 包装 `runner_input_len` 独立观测）落地；CPU 主负载短 prompt 改为互不相同（保持每条 128 token 的真实 prefill 工作量）。

## 2. 环境与模型可用性

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 D（24 GB） |
| torch / CUDA | 2.5.1+cu124 / 12.4 |
| 模型 | Qwen3-0.6B，本地目录 `/root/huggingface/Qwen3-0.6B`（未下载） |
| 引擎配置 | TP=1，enforce_eager=True，kvcache_block_size=256，max_model_len=12288，KV 池 706 块 |
| 结论 | GPU 与模型均可用，§5 GPU 验证已实际执行 |

## 3. 审查阻断项修复验证（§3.1）

- 修复前（审查复现）：`warmup_model` 以旧签名调用 `run(seqs, True)` → GPU 引擎构造即 `TypeError`。
- 修复后 GPU 冒烟：引擎正常完成 warmup/KV 分配并输出连贯结果（`"The capital of France is"` → `" Paris. The capital of Italy is Rome"`），随后执行 §5 全部 GPU 验证——阻断解除。

## 4. 设计取舍说明

1. **归因规则采纳"代码对齐设计"**（审查 §4.1 建议一）：`decode_priority` 按不变量 18 的 `needed_first` 字面规则判定。已知该规则在"decode 占用极大、首候选只分到很小的 chunk"的碎片化场景会高估 decode 责任（记 `budget`），审查已指出两种口径各有一个更准的反例方向，设计文档是权威，故按设计落地并在此记录。
2. **被延后候选的 `needed_tokens` 记录口径**：`remaining==0` 路径上，若被延后候选本身就是本轮首个被考察的候选（就地估算），其需求如实入档；若 `needed_first` 已由更早的候选提供，该候选未做估算，`needed_tokens` 记 `None`（不虚构需求，沿用 Day7"未成为候选不虚构需求"口径）。
3. **CPU/GPU 主负载的短 prompt 互不相同**：32 条短请求若共用同一 prompt，prefix 命中会把后续短请求折叠成 8-token prefill（CPU 第一版证据即出现该现象），偏离 `plan.md` "32 个 128-token prompt" 的负载意图，故统一改为互不相同的词表内 prompt。
4. **GPU 短请求 `max_tokens=16`**：保证 32 条短请求贯穿 8K prompt 的整个分块期，使"decode 持续产出"证据覆盖每一轮混合调度。
5. **TTFT/TPOT 观测点**：首 token / 完成时刻记在产出结果的 `step()` 之后；TPOT 为同步 API 近似口径 `(完成时刻 - 首 token 时刻)/(completions-1)`，ITL 为该请求相邻 decode 轮的实际输入间隔，均注明局限。

## 5. CPU 测试（实际命令与结果）

```text
python -m pytest -q                                   # 303 passed（新增 2 条审查反例用例）
python -O -m pytest -q                                # 303 passed
PYTHONPATH=. python scripts/validate_mixed_batch.py \
    --output docs/evidence/day9/day9-cpu.jsonl        # PASS
PYTHONPATH=. python scripts/validate_token_budget.py  # PASS
PYTHONPATH=. python scripts/validate_chunked_prefill.py --mode cpu  # PASS
```

- 新增审查反例用例：`test_budget_when_needed_first_exceeds_budget`（`remaining==0` 路径：D=2、首候选 needed=20>B=8 拆分接纳后，needed=5 的小候选延后记 `budget`，事件 `needed_first=20` 供复算）与 `test_budget_when_needed_first_exceeds_budget_nonfirst_scan_stop`（非首候选整段放不下路径：B=12 同场景记 `budget` 且自身需求 5 入档）。
- 既有场景按 needed_first 规则重推：`test_unique_counted_once_across_rounds`（分块请求 6/5/4 三轮需求均 >B=2，c 连续记 budget）与 `test_timeline_exact_rounds_and_seconds`（`c_tokens=8` 使 e 的 4 轮延后均 >B），episode 统计断言（unique=1/rounds=3、unique=2/rounds=7/closed=9.5s）逐项保持。
- CPU 主负载（互不相同短 prompt）：25 轮，长 prompt 分块 8 轮，其中 decode 产出 **7/7** 轮（首轮无 decode 源属正常），8K 请求首 token 轮 = 8；混合预算、归因复算、`runner_input_len` 对账（CPU 桩口径）全部 PASS。

## 6. GPU 验证（§9.4，实际命令与结果）

```text
PYTHONPATH=. HF_HUB_OFFLINE=1 python scripts/validate_mixed_batch.py --mode gpu \
    --model /root/huggingface/Qwen3-0.6B \
    --output docs/evidence/day9/day9-gpu.jsonl
# [gpu-main:one_shot] ... [gpu-main:mixed] ... [gpu-main] greedy 一致性 26/33
# [gpu] variant0/variant1 事件校验 PASS（exit 0）
```

主负载：1 × 8192-token prompt + 32 × 128-token prompt（互不相同），greedy（temperature=0），`max_tokens=16`，`max_num_seqs=64`；KV 容量自检先行（8208+32 token << 706 块）。对照组为"原始一次性 Prefill"（chunk_size=B=8192，长 prompt 单轮 prefill），实验组为 Day9 mixed（chunk_size=1024、B=2048）。性能为单次运行观测值，不承诺改善方向。

| 指标 | one-shot（原始一次性 Prefill） | Day9 mixed |
| --- | --- | --- |
| 轮数 / wall time | 17 / 4.15 s | 23 / 0.98 s |
| 8K 请求 TTFT | 1.11 s | 0.57 s |
| 短请求 TTFT P50 / P95 | 3.568 s / 3.568 s | **0.198 s / 0.352 s** |
| 短请求 TPOT P50 / P95 | 39.1 / 39.1 ms | 43.5 / 50.1 ms |
| 短请求 decode ITL P50 / P95 | 36.6 / 40.1 ms | 28.2 / 77.0 ms |
| 长 prompt 分块轮中 decode 产出 | 0/1（唯一分块轮无 decode 源） | **7/8**（首轮无 decode 源属正常） |
| peak allocated / reserved | 21157 / 21290 MiB | 21188 / 21268 MiB |
| KV 收尾 | free/used 平衡 | free/used 平衡 |

对照解读（只报观测值，"可解释的变化"含变差方向）：

- **短请求 TTFT**：mixed 组 P50 从 3.568 s 降至 0.198 s（约 18 倍）。机制：one-shot 下 32 条短请求必须等 8192 整批 prefill 轮完成后才被接纳；mixed 下短请求从第 1 轮起与长 prompt 分块交错接纳并立即产出首 token（decode 产出轮的 decode 条数爬坡 `[0, 8, 15, 22, 29, 32, 32, 32]` 见证据 JSONL `perf_matrix`）。
- **短请求 TPOT/ITL**：mixed 组 P50 略有折让（39.1 → 43.5 ms），ITL P95 波动更大（40.1 → 77.0 ms）——混合轮的 decode 与 1024-token 分块共享调度轮，decode 轮时长被 prefill 子批拉长；属 decode-first 的预期代价，方向与幅度均有解释。
- **8K 请求 TTFT**：mixed 组反而更低（1.11 → 0.57 s）：1024-token 分块前向的激活峰值低于 8192 整批（与 Day8 §5.1 one-shot 档观测一致）。
- **吞吐**：mixed 组总 wall time 更短（0.98 s vs 4.15 s），来自分块前向的更高 GPU 利用率与短请求的流水化。
- **decode 持续产出**（plan.md 验收条目 1）：mixed 组长 prompt 分块期 8 轮中 7 轮有 decode 源、**7/7 轮 decode 全部按 decode-first 推进**（逐轮 `decode_items == 轮前 RUNNING 数` 独立校验）；唯一无 decode 产出的是首轮（尚无任何 RUNNING 请求，属结构性正常）。
- **`runner_input_len` 对账**（Day8 遗留收口）：`prepare_decode/prepare_prefill` 包装捕获每子批实际组装的展平 `input_ids` 长度（与调度计划同源字段无关），两个 variant 分别 17/17、23/23 轮与调度计划一致，事件校验 PASS。
- **事件独立校验**：混合预算公式、phase 与构成一致性、`decode_priority` 判定条件（`D>0 且 needed_first<=B`）复算、episode 秒数重算、字段白名单（含 `needed_first`）——两 variant 全部 PASS；事件不含 prompt/token 明文。

### 6.1 greedy 一致性差异分析（如实记录，不伪造一致）

mixed vs one-shot 最终 `token_ids` 逐请求对比：**26/33 逐 token 一致，7 条存在分歧**（请求下标 2、5、10、16、21、23、31）。

- 差异形态：3 条在位置 0 分歧（其 128-token prefill 在 mixed 中与长 prompt 分块同批执行，one-shot 中与其他短请求同批执行），4 条在位置 5–14 分歧（前缀逐位一致后翻转）。
- 确定性：完整重跑一次 GPU 验证（`day9-gpu-repro.jsonl`），**差异请求集合、首个分歧位置、分歧两侧 token 逐位完全复现**——是确定性数值差异，非随机采样或竞态。
-机制：one-shot 与 mixed 的 varlen attention kernel 批次形状不同（批内序列长度组合不同 → kernel 归约顺序不同），产生微小数值差，在近似平局处翻转 greedy argmax。与 Day8 §6 记录的 chunked vs one-shot 差异同类（Day8 中 chunk_size=16 时 5/6 一致、256 时 6/6 一致）；CPU 桩注入测试证明固定注入下两路径输出完全一致，调度/进度语义无错。

## 7. 对照设计文档 §10 验收清单的核对依据（清单原文未改）

- **功能与口径**（§10 第 1–5 项）：审查报告 §2 已确认并勾选；本轮 §4.1 修复后，第 4 项（归因）的实现与不变量 18 字面规则一致，新增反例单测 + 脚本 `needed_first` 复算闭合该项的证据链。
- **执行正确性**（§10 第 6–10 项）：审查报告 §2 确认；本轮 GPU 实测进一步覆盖第 7 项（`Context/Attention/LMHead` 零修改下的真实模型分组执行）与第 9 项（`runner_input_len` 对账，GPU 侧以 `prepare_*` 实际组装长度独立观测）。
- **实测与交付**（§10 第 11–15 项）：
  - CPU 全量回归 + `python -O`：303 passed（§5）。
  - GPU 主负载：§6 对照表，decode 持续产出有逐轮事件证据；短请求 P50/P95 TTFT、TPOT/ITL 与 one-shot 对照归档。
  - mixed vs one-shot 一致性：26/33，差异分析与确定性复现归档（§6.1）。
  - 本记录 + `docs/README.md` 索引更新；证据无 prompt/token 明文。
  - TP>1 / CUDA Graph / 真实并发取消：**未实测**（如实记录，见 §8）。

## 8. 未覆盖边界与遗留

- TP>1 端到端、CUDA Graph 捕获路径（GPU 验证统一 `enforce_eager=True` 基线）、真实并发取消/超时：未实测。
- 性能矩阵为单次运行观测值（同一 variant 多次运行 wall time 在 3.3–4.4 s / 0.8–1.0 s 间波动），未做多轮取均值；Torch 配置未额外固化。
- 短请求 TPOT 为同步 API 近似口径；ITL 的 P95 受混合轮 prefill 子批时长影响波动较大（77 ms），如需服务化口径应基于 Day11+ 的流式事件重测。
- `decode_priority` 规则在碎片化场景（decode 占用极大、首候选只得极小 chunk）会高估 decode 责任（记 `budget`），属设计字面规则的已知近似，已在 §4 取舍 1 记录。
- `docs/README.md` 中 Day8 索引行引用的 `day8-validation.md` §9 "中短 prompt 控制组留待 Day9"事项：由本记录 §6 的 mixed 主负载对照覆盖。
