# Day 4–5 验收记录

## Day 4

- 产出：[`prefill-decode.md`](./prefill-decode.md)、[`day4_prefill_decode.py`](../experiments/day4_prefill_decode.py)、[`test_sampler.py`](../tests/test_sampler.py)。
- 验收：覆盖 Prefill/Decode shape、KV 写入/读取、瓶颈、greedy/temperature/top-p/max_tokens；5 个固定 prompt 的 greedy 可复现由脚本记录。
- 命令：`python experiments/day4_prefill_decode.py`、`python -m pytest tests/test_sampler.py -q`。

## Day 5

- 产出：[`run_baseline.py`](../benchmarks/run_baseline.py)、[`day5-performance.md`](./day5-performance.md)、`benchmarks/results/baseline.json`（运行 benchmark 后生成）。
- 验收：runner 默认覆盖并发 1/8/32、输入 128/512/2048、输出 32/128，并保留逐请求原始记录、配置、环境、TTFT、吞吐和显存状态；ITL/TPOT 在同步 API 无中间 token 时明确为 null。
- 当前环境若无 GPU/模型权重，仅能验收脚本与 CPU 采样测试，不能宣称 GPU 基线已实测。
