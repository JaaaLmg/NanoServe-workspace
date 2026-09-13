"""python -m nanoserve.server：生产启动入口（Day11–12 §5.6）。

可重复启动命令：
    python -m nanoserve.server --model /path/to/Qwen3-0.6B \
        --model-id Qwen3-0.6B --host 127.0.0.1 --port 8000 \
        --tensor-parallel-size 1 --enforce-eager

CLI 参数只覆盖服务层配置（模型、监听、deadline、eager/TP）；
max_model_len、chunk_size 等 Engine 参数仍由 nanovllm.config.Config 负责，
不在 CLI 中复刻 Config 逻辑。所有参数都有对应环境变量默认值（见 config.py），
CLI 显式传入时优先生效。
"""

import argparse
import logging
import os
import sys
from dataclasses import replace

import uvicorn

from nanoserve.app import create_app
from nanoserve.config import (DEFAULT_ENFORCE_EAGER, DEFAULT_HOST, DEFAULT_PORT,
                              DEFAULT_TENSOR_PARALLEL_SIZE, ENV_MODEL, ServerConfig)

logger = logging.getLogger("nanoserve.server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nanoserve.server",
        description="NanoServe OpenAI 兼容 HTTP 服务（Day11–12：非流式）")
    parser.add_argument("--model", default=None,
                        help="HuggingFace 本地模型目录（默认读 NANOSERVE_MODEL）")
    parser.add_argument("--model-id", default=None,
                        help="对外公开模型 ID（默认取模型目录最后一段）")
    # 使用 SUPPRESS 区分“未传 CLI”与“显式传入默认值”，否则 argparse 的
    # 常量默认会遮蔽对应 NANOSERVE_* 环境变量。
    parser.add_argument("--host", default=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-request-seconds", type=float,
                        default=argparse.SUPPRESS,
                        help="服务层 deadline 秒数（默认无 deadline）")
    # 默认值由 ServerConfig.from_env 提供；显式 --enforce-eager/--no-enforce-eager
    # 才覆盖环境变量。
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction,
                        default=argparse.SUPPRESS)
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=argparse.SUPPRESS)
    return parser


def build_server_config(args: argparse.Namespace) -> ServerConfig:
    """CLI 参数 + 环境变量合并为 ServerConfig（CLI 显式值优先）。"""
    try:
        config = ServerConfig.from_env()
    except ValueError as exc:
        # CLI 的 --model 可以补足环境变量中的必填项；补足后重新走同一
        # 环境解析路径，不能丢失 host/port/eager/TP 等其它环境配置。
        model = getattr(args, "model", None)
        if not model:
            raise SystemExit(str(exc)) from exc
        environ = dict(os.environ)
        environ["NANOSERVE_MODEL"] = model
        config = ServerConfig.from_env(environ)
    overrides = {}
    for name in ("model", "model_id", "host", "port", "max_request_seconds",
                 "enforce_eager", "tensor_parallel_size"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return replace(config, **overrides)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        server_config = build_server_config(args)
    except (ValueError, SystemExit) as exc:
        # 配置错误尽早失败：打印可理解信息并以非零码退出
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    app = create_app(server_config=server_config)
    uvicorn.run(app, host=server_config.host, port=server_config.port,
                log_level="info")


if __name__ == "__main__":
    main()
