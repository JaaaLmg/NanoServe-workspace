"""Day11–12 服务配置与 CLI 优先级测试。

覆盖环境变量解析、CLI 显式覆盖、有限 deadline 校验和空白模型拒绝；
不加载 GPU/模型。重复执行命令：
    python -m pytest tests/test_server_config.py -q
"""

import argparse
import math

import pytest

from nanoserve.config import ServerConfig
from nanoserve.server import build_server_config, build_parser


def test_environment_values_are_used_when_cli_omits_them(monkeypatch):
    monkeypatch.setenv("NANOSERVE_MODEL", "/tmp/model")
    monkeypatch.setenv("NANOSERVE_MODEL_ID", "env-model")
    monkeypatch.setenv("NANOSERVE_HOST", "0.0.0.0")
    monkeypatch.setenv("NANOSERVE_PORT", "9001")
    monkeypatch.setenv("NANOSERVE_MAX_REQUEST_SECONDS", "2.5")
    monkeypatch.setenv("NANOSERVE_ENFORCE_EAGER", "false")
    monkeypatch.setenv("NANOSERVE_TENSOR_PARALLEL_SIZE", "2")

    config = build_server_config(build_parser().parse_args([]))
    assert (config.model, config.model_id, config.host, config.port) == (
        "/tmp/model", "env-model", "0.0.0.0", 9001)
    assert config.max_request_seconds == 2.5
    assert config.enforce_eager is False
    assert config.tensor_parallel_size == 2


def test_explicit_cli_values_override_environment(monkeypatch):
    monkeypatch.setenv("NANOSERVE_MODEL", "/tmp/env-model")
    monkeypatch.setenv("NANOSERVE_HOST", "0.0.0.0")
    monkeypatch.setenv("NANOSERVE_PORT", "9001")
    monkeypatch.setenv("NANOSERVE_ENFORCE_EAGER", "false")
    monkeypatch.setenv("NANOSERVE_TENSOR_PARALLEL_SIZE", "2")

    args = build_parser().parse_args([
        "--model", "/tmp/cli-model", "--host", "127.0.0.1",
        "--port", "8001", "--enforce-eager", "--tensor-parallel-size", "1",
    ])
    config = build_server_config(args)
    assert (config.model, config.host, config.port) == (
        "/tmp/cli-model", "127.0.0.1", 8001)
    assert config.enforce_eager is True
    assert config.tensor_parallel_size == 1


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, "x"])
def test_deadline_must_be_finite_positive(value):
    with pytest.raises(ValueError):
        ServerConfig(model="/tmp/model", max_request_seconds=value)


def test_blank_model_is_rejected():
    with pytest.raises(ValueError):
        ServerConfig(model="   ")
