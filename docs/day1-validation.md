# Day 1 最小闭环验收

## 1. 验收范围

本记录对应 `plan.md` 的 Day 1：环境可用、模型可加载、命令可重复执行，并使用至少 3 个 prompt 产生输出。这里不验收 HTTP 服务、流式输出、调度器或性能 benchmark。

- 验收日期：2026-09-06
- 仓库 commit：`352e16e`
- 模型：Qwen/Qwen3-0.6B
- 模型路径：`~/huggingface/Qwen3-0.6B/`
- 采样配置：`temperature=0.6`，`max_tokens=256`（第三条临时验证使用 `max_tokens=64`）
- 运行设备：NVIDIA GeForce RTX 4090 D

## 2. 可重复命令验收

### 2.1 官方示例

```bash
python example.py
```

结果：**通过**，退出码为 `0`。模型成功加载并完成 2 条 prompt 的批量生成；终端输出包含 `Prompt:` 和 `Completion:`。

### 2.2 第三条 prompt

为不修改仓库源码，使用一次性 Python 脚本复用同一套 `LLM`、tokenizer 和采样配置：

```bash
python - <<'PY'
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser('~/huggingface/Qwen3-0.6B/')
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
prompt = '用一句话解释什么是机器学习。'
formatted = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
output = llm.generate(
    [formatted], SamplingParams(temperature=0.6, max_tokens=64)
)[0]
print('Prompt:', prompt)
print('Completion:', repr(output['text']))
PY
```

结果：**通过**，退出码为 `0`，产生中文 completion 输出。

## 3. 三条 prompt 结果

| # | Prompt | 验收结果 | 实际输出摘要 |
| --- | --- | --- | --- |
| 1 | `introduce yourself` | 通过 | 产生英文自我介绍 completion，输出以 `<think>` 开始并继续生成英文回答。 |
| 2 | `list all prime numbers within 100` | 通过 | 产生英文回答，输出列出 100 以内的质数并继续解释。 |
| 3 | `用一句话解释什么是机器学习。` | 通过 | 产生中文解释文本，输出以 `<think>` 开始并继续生成中文回答。 |

以上摘要只用于证明生成链路有输出，不对回答事实正确性、格式质量或模型能力做额外承诺。由于采样温度为 `0.6`，没有把本次输出声明为逐字可复现的 deterministic 结果。

## 4. 基础检查结果

```bash
python -m compileall -q nanovllm example.py bench.py
```

结果：**通过**，没有语法错误。

## 5. Day 1 验收结论

| 验收项 | 状态 | 证据 |
| --- | --- | --- |
| 环境信息已记录 | 通过 | [`env.md`](./env.md) |
| 有可重复启动命令 | 通过 | `python example.py` |
| 至少 3 个 prompt 产生输出 | 通过 | 本文第 3 节 |
| 显存不足降级方案已记录 | 通过 | [`env.md`](./env.md) 第 5 节 |
| 后续服务化能力 | 未验收 | 不属于 Day 1 范围 |
