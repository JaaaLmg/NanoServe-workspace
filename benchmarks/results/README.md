# Baseline 运行记录

- 日期：2026-09-10
- git commit：`51e420a`（main）
- 模型：Qwen/Qwen3-0.6B（HF 快照 `c1899de289a04d12100db370d81485cdf75e47ca`，经 hf-mirror.com 下载）
- 硬件：NVIDIA GeForce RTX 4090 D（24GB），torch 2.5.1+cu124
- 命令：

  ```bash
  python benchmarks/run_baseline.py \
    --model <Qwen3-0.6B 快照路径> \
    --output benchmarks/results/baseline.json \
    --enforce-eager
  ```

- 矩阵：并发 1/8/32 × 输入 128/512/2048 × 输出 32/128，共 18 个 case，全部 `status:ok`。
- 说明：本次运行对 `run_baseline.py` 做了一处修改——每个 case 结束后显式调用引擎的
  `exit()`（销毁 NCCL 进程组并释放 KV cache），否则第二个 case 起
  `dist.init_process_group` 会报 "trying to initialize the default process group twice!"。
  引擎本身未改动。第一个 case（并发 1 / 128 / 32）的 TTFT（约 3.04s）包含首个 case
  的 CUDA 内核冷启动开销，跨 case 比较时应注意。
