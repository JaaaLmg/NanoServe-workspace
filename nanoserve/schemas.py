"""Pydantic 请求/响应与错误模型（Day11–12 OpenAI 外部契约）。

设计要点（docs/openai-api-day11-12.md §3/§5.3）：
- 请求模型 extra="forbid"：未支持字段（数组 prompt、n>1、stop 等）直接被
  拒绝并经统一异常处理器转换为 400，而不是被静默忽略让用户误以为生效；
- max_tokens 默认 64（全服务唯一口径，与 SamplingParams 默认值一致）；
  temperature/top_p 边界与 SamplingParams 相同——schema 层只提供字段，
  SamplingParams 仍是最终校验入口；
- 响应模型保持 OpenAI 风格：completion 带 logprobs: null 兼容字段，
  chat 不添加 completion 专属字段；
- 错误模型永不序列化 Python 异常堆栈或 prompt/token 明文。

注意：FastAPI 默认的 Pydantic 422 由 app.py 的统一异常处理器转换成
OpenAI 风格 400，避免同类输入错误出现两种协议。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 全服务唯一的 max_tokens 默认值：文档、schema、测试共用同一口径
DEFAULT_MAX_TOKENS = 64

# chat role 白名单：非法 role 返回 400
ALLOWED_ROLES = ("system", "user", "assistant")


class CompletionRequest(BaseModel):
    """POST /v1/completions 请求体（Day11 MVP：只支持单个字符串 prompt）。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="必须与服务公开模型 ID 完全一致")
    prompt: str = Field(min_length=1, description="单个非空字符串 prompt")
    # strict=True：bool/float 不被静默整编为 int（python -O 与 lax 模式下口径一致）
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, strict=True)
    temperature: float = Field(default=1.0, ge=0, strict=True)
    top_p: float = Field(default=1.0, gt=0, le=1, strict=True)
    # stream 只做解析与校验：false 走非流式；true 在路由层返回 501（Day13）
    stream: bool = Field(default=False, strict=True)

    @field_validator("prompt")
    @classmethod
    def _reject_blank_prompt(cls, value: str) -> str:
        # 空白 prompt 明确拒绝（§3.2），不推迟到 tokenizer 结果才失败
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class ChatMessage(BaseModel):
    """单条对话消息：role 白名单 + 非空字符串 content，额外字段拒绝。"""

    model_config = ConfigDict(extra="forbid")

    # role 白名单严格生效（§3.3）：非法 role 在 schema 层即被拒绝
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions 请求体（Day12 MVP）。"""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="必须与服务公开模型 ID 完全一致")
    messages: list[ChatMessage] = Field(min_length=1,
                                        description="至少一条消息")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, strict=True)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False


# ---------- 非流式成功响应模型 ----------

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    # OpenAI completion 兼容字段：本阶段不返回 logprobs，恒为 null
    logprobs: None = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ChatAssistantMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatAssistantMessage
    # chat 响应不携带 completion 专属字段（logprobs）
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# ---------- 统一错误响应模型 ----------

class ErrorBody(BaseModel):
    message: str
    type: str
    # param 指向出错的请求字段，便于定位；无对应字段时为 null
    param: str | None = None
    code: str


class ErrorResponse(BaseModel):
    error: ErrorBody
