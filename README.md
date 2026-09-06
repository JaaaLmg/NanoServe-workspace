# NanoServe

基于 nano-vLLM 的长上下文、高并发 LLM 推理服务引擎。开发计划见 [plan.md](./plan.md)。

## 项目结构

- `nanovllm/` — 推理引擎核心（vendor 自 nano-vLLM，作为后续开发的基础）
- `example.py` — nano-vLLM 官方推理示例
- `bench.py` — nano-vLLM 官方 benchmark 脚本
- `docs/nanovllm-README.md` — 上游 README 存档

## 出处声明

`nanovllm/` 目录、`example.py`、`bench.py` 来源于
[GeeeekExplorer/nano-vLLM](https://github.com/GeeeekExplorer/nano-vLLM)
（commit `bb823b3`），按 [MIT License](./LICENSE.nanovllm) 授权，版权归上游作者所有。
本项目在此基础上进行的扩展与修改，归本仓库作者所有。
