"""ServerConfig：服务层配置解析（环境变量/CLI），不替代 engine.Config。

职责边界：
- 本模块只负责服务生命周期配置（模型路径、公开 ID、监听地址、deadline、
  eager/TP 开关），并给出对外公开的模型 ID；
- Engine 侧参数（max_model_len、max_num_batched_tokens、chunk_size、KV 容量
  等）仍由 nanovllm.config.Config 负责显式校验，服务配置不绕过、不复刻。

环境变量契约（docs/openai-api-day11-12.md §3.1）：
    NANOSERVE_MODEL                  必填，HuggingFace 本地模型目录或可加载标识
    NANOSERVE_MODEL_ID               可选，默认取模型目录最后一段
    NANOSERVE_HOST                   默认 127.0.0.1
    NANOSERVE_PORT                   默认 8000
    NANOSERVE_MAX_REQUEST_SECONDS    可选，单调时钟 deadline；默认无服务层 deadline
    NANOSERVE_ENFORCE_EAGER          默认 true，便于最小服务验收
    NANOSERVE_TENSOR_PARALLEL_SIZE   默认 1
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_ENFORCE_EAGER = True
DEFAULT_TENSOR_PARALLEL_SIZE = 1

ENV_MODEL = "NANOSERVE_MODEL"
ENV_MODEL_ID = "NANOSERVE_MODEL_ID"
ENV_HOST = "NANOSERVE_HOST"
ENV_PORT = "NANOSERVE_PORT"
ENV_MAX_REQUEST_SECONDS = "NANOSERVE_MAX_REQUEST_SECONDS"
ENV_ENFORCE_EAGER = "NANOSERVE_ENFORCE_EAGER"
ENV_TENSOR_PARALLEL_SIZE = "NANOSERVE_TENSOR_PARALLEL_SIZE"


@dataclass(frozen=True)
class ServerConfig:
    """服务生命周期配置（不可变）；Engine 参数由调用方另行传给 Config。"""

    model: str
    model_id: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # 服务层 deadline（秒）：以 perf_counter()+seconds 传给 add_request；
    # None 表示不设服务层 deadline（Day10 底层超时语义不受影响）
    max_request_seconds: float | None = None
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE

    def __post_init__(self):
        # 显式校验（不用 assert，python -O 下仍生效）：
        # - port 必须落在合法监听范围；
        # - max_request_seconds 必须为正数或 None；
        # - tensor_parallel_size 必须为正整数（bool 是 int 子类，显式排除）。
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("服务配置 model 必须为非空字符串")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError(f"port 必须为 1-65535 的整数，实际为 {self.port!r}")
        if self.max_request_seconds is not None and (
                isinstance(self.max_request_seconds, bool)
                or not isinstance(self.max_request_seconds, (int, float))
                or not math.isfinite(self.max_request_seconds)
                or self.max_request_seconds <= 0):
            raise ValueError(
                f"max_request_seconds 必须为有限正数或 None，实际为 {self.max_request_seconds!r}")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size 必须为正整数，实际为 {self.tensor_parallel_size!r}")
        if not 1 <= self.tensor_parallel_size <= 8:
            # 与 nanovllm.config.Config 的 TP 上限一致，尽早失败
            raise ValueError(
                f"tensor_parallel_size 必须在 1-8 之间，实际为 {self.tensor_parallel_size!r}")

    @property
    def public_model_id(self) -> str:
        """对外公开模型 ID：显式配置优先，否则取模型路径最后一段。

        对外 ``model`` 字段必须与该 ID 完全一致；不支持请求内动态切换模型。
        """
        if self.model_id:
            return self.model_id
        name = Path(self.model).name
        return name or self.model

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "ServerConfig":
        """从环境变量构造配置；NANOSERVE_MODEL 缺失时抛出可理解错误。

        environ 参数供测试注入，生产路径直接读 os.environ。
        """
        env = os.environ if environ is None else environ
        model = env.get(ENV_MODEL)
        if not model:
            raise ValueError(
                f"缺少模型配置：请设置环境变量 {ENV_MODEL} 或通过 CLI --model 指定")
        port = env.get(ENV_PORT)
        max_request_seconds = env.get(ENV_MAX_REQUEST_SECONDS)
        enforce_eager_raw = env.get(ENV_ENFORCE_EAGER)
        return cls(
            model=model,
            model_id=env.get(ENV_MODEL_ID) or None,
            host=env.get(ENV_HOST) or DEFAULT_HOST,
            port=int(port) if port else DEFAULT_PORT,
            max_request_seconds=float(max_request_seconds)
            if max_request_seconds else None,
            # 仅接受显式 "0/false/no/off" 关闭 eager，其余按默认开启处理
            enforce_eager=DEFAULT_ENFORCE_EAGER
            if enforce_eager_raw is None
            else enforce_eager_raw.strip().lower()
            not in ("0", "false", "no", "off"),
            tensor_parallel_size=int(env.get(ENV_TENSOR_PARALLEL_SIZE)
                                     or DEFAULT_TENSOR_PARALLEL_SIZE),
        )
