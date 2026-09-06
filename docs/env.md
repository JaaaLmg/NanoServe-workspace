# Day 1 环境记录

> 本文档只记录 Day 1 的环境准备和最小离线推理闭环，不代表后续服务化功能已经实现。

## 1. 已验证环境

| 项目 | 实际值 |
| --- | --- |
| 操作系统 | Linux 5.15.0-94-generic，x86_64，glibc 2.35 |
| Python | 3.12.3（Anaconda） |
| GPU | NVIDIA GeForce RTX 4090 D，24,564 MiB |
| NVIDIA 驱动 | 580.76.05 |
| PyTorch | 2.5.1+cu124 |
| PyTorch CUDA | 12.4 |
| Triton | 3.1.0 |
| Transformers | 5.16.1 |
| xxhash | 4.0.1 |
| 模型 | Qwen/Qwen3-0.6B |
| 本地模型目录 | `~/huggingface/Qwen3-0.6B/`（已存在，约 1.5G） |
| NanoServe 仓库 commit | `352e16e` |

环境探测命令：

```bash
python --version
python -c "import torch, triton, transformers, xxhash; print(torch.__version__, torch.version.cuda, triton.__version__, transformers.__version__, xxhash.VERSION)"
nvidia-smi
```

## 2. 安装与模型准备

建议使用 Python 3.10–3.12 的独立环境。依赖版本应以实际可用的 CUDA/PyTorch 组合为准：

```bash
# 在仓库根目录执行
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch transformers triton xxhash
```

如果使用 Anaconda，可用：

```bash
conda create -n nanoserve-day1 python=3.12 -y
conda activate nanoserve-day1
python -m pip install torch transformers triton xxhash
```

下载 Qwen3-0.6B：

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \\
  --local-dir ~/huggingface/Qwen3-0.6B/ \\
  --local-dir-use-symlinks False
```

下载完成后确认目录中存在模型配置、tokenizer 和权重文件。若模型放在其他位置，需要同步修改 `example.py` 中的 `path`。

## 3. 可重复启动命令

在仓库根目录、模型目录已经准备好的前提下：

```bash
python example.py
```

当前 `example.py` 使用以下配置：

- 模型：`~/huggingface/Qwen3-0.6B/`
- `enforce_eager=True`
- `tensor_parallel_size=1`
- `temperature=0.6`
- `max_tokens=256`
- 使用 Transformers chat template 构造输入

该命令是当前 Day 1 的最小闭环入口：加载 tokenizer 和模型，执行两条 chat prompt 的批量生成，并打印 completion。

## 4. 基础检查

不加载模型的语法检查：

```bash
python -m compileall -q nanovllm example.py bench.py
```

本次检查结果：通过。

## 5. 显存不足降级方案

按以下顺序降低资源需求：

1. 将 `SamplingParams(max_tokens=256)` 调小，例如改为 `max_tokens=64`。
2. 减少一次传入的 prompt 数量，改为逐条生成。
3. 使用更小的模型或量化模型，并将 `path` 指向新的本地目录。
4. 保持 `tensor_parallel_size=1`，避免在单卡环境误启用多卡。
5. 保持 `enforce_eager=True`，避免 CUDA Graph 在显存紧张或环境不兼容时增加额外开销。

CPU-only 环境不属于当前 nano-vLLM Day 1 的完整运行路径；此时可以完成文档、导入和编译检查，但不能把离线生成验收标记为通过。
