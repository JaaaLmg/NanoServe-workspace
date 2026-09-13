"""Day11–12 schema 与参数测试（docs/openai-api-day11-12.md §8.2）。

覆盖范围：
- completion/chat 最小有效请求与默认采样值（max_tokens 默认 64 全服务唯一口径）；
- system/user/assistant 混合消息与多个 user turn；
- 缺失/空 prompt、缺失/空 messages、缺失 content、非法 role、非字符串 content；
- max_tokens=0/负数/bool、temperature 负数、top_p 为 0/大于 1；
- stream 默认 false；
- 未支持字段（extra="forbid"）：数组 prompt、n>1、未知字段一律拒绝。

无 GPU/模型依赖：纯 Pydantic 校验逻辑，不加载权重、不初始化 CUDA。

重复执行命令：
    python -m pytest tests/test_service_schemas.py -q
    python -O -m pytest tests/test_service_schemas.py -q
"""

import pytest
from pydantic import ValidationError

from nanoserve.schemas import (ChatCompletionRequest, ChatMessage,
                               CompletionRequest, DEFAULT_MAX_TOKENS)


class TestCompletionSchema:
    """POST /v1/completions 请求模型契约。"""

    def test_minimal_valid_request_uses_defaults(self):
        """最小有效请求：max_tokens=64、temperature=1.0、top_p=1.0、stream=False。"""
        req = CompletionRequest.model_validate(
            {"model": "Qwen3-0.6B", "prompt": "介绍一下分页 KV Cache。"})
        assert req.max_tokens == DEFAULT_MAX_TOKENS == 64
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.stream is False

    def test_explicit_fields_roundtrip(self):
        req = CompletionRequest.model_validate({
            "model": "Qwen3-0.6B", "prompt": "hello",
            "max_tokens": 16, "temperature": 0.0, "top_p": 0.5,
            "stream": False})
        assert (req.max_tokens, req.temperature, req.top_p) == (16, 0.0, 0.5)

    @pytest.mark.parametrize("payload", [
        {"model": "m"},                                   # 缺失 prompt
        {"model": "m", "prompt": ""},                     # 空 prompt
        {"model": "m", "prompt": "   "},                  # 空白 prompt
        {"model": "m", "prompt": ["a", "b"]},             # 数组 prompt 未支持
        {"prompt": "x"},                                  # 缺失 model
    ])
    def test_invalid_prompt_or_model_rejected(self, payload):
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(payload)

    @pytest.mark.parametrize("max_tokens", [0, -1, -100, True, False])
    def test_invalid_max_tokens_rejected(self, max_tokens):
        """max_tokens 必须为正整数；bool 显式拒绝（pydantic 不接受 bool→int）。"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(
                {"model": "m", "prompt": "x", "max_tokens": max_tokens})

    def test_negative_temperature_rejected(self):
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(
                {"model": "m", "prompt": "x", "temperature": -0.1})

    @pytest.mark.parametrize("top_p", [0, 0.0, 1.5, -0.5])
    def test_invalid_top_p_rejected(self, top_p):
        """top_p 必须落在 (0, 1]，与 SamplingParams 边界一致。"""
        with pytest.raises(ValidationError):
            CompletionRequest.model_validate(
                {"model": "m", "prompt": "x", "top_p": top_p})

    @pytest.mark.parametrize("extra", [
        {"n": 2},                       # n>1 未支持
        {"best_of": 2},                 # best_of 未支持
        {"stop": ["\n"]},               # stop 未支持
        {"seed": 42},                   # 第一版不暴露扩展字段
        {"totally_unknown": 1},         # 未知字段
    ])
    def test_unsupported_fields_rejected(self, extra):
        """extra="forbid"：未支持字段返回校验错误（路由层转 400），不被忽略。"""
        payload = {"model": "m", "prompt": "x", **extra}
        with pytest.raises(ValidationError) as exc_info:
            CompletionRequest.model_validate(payload)
        assert any(e["type"] == "extra_forbidden" for e in exc_info.value.errors())


class TestChatSchema:
    """POST /v1/chat/completions 请求模型契约。"""

    def test_minimal_valid_request(self):
        req = ChatCompletionRequest.model_validate({
            "model": "Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}]})
        assert req.max_tokens == DEFAULT_MAX_TOKENS
        assert req.stream is False
        assert len(req.messages) == 1

    def test_mixed_roles_and_multiple_user_turns(self):
        """system/user/assistant 混合与多个 user turn 都合法且保序。"""
        messages = [
            {"role": "system", "content": "你是一个简洁的助手。"},
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "问题二"},
        ]
        req = ChatCompletionRequest.model_validate(
            {"model": "m", "messages": messages})
        assert [m.role for m in req.messages] == \
            ["system", "user", "assistant", "user"]

    @pytest.mark.parametrize("payload", [
        {"model": "m"},                                    # 缺失 messages
        {"model": "m", "messages": []},                    # 空 messages
        {"model": "m", "messages": [{"role": "user"}]},    # 缺失 content
        {"model": "m", "messages": [{"content": "hi"}]},   # 缺失 role
        {"model": "m", "messages": [{"role": "user", "content": ""}]},  # 空 content
        {"model": "m", "messages": "hello"},               # messages 不是数组
        {"model": "m", "messages": [{"role": "tool", "content": "x"}]},  # 非法 role
        {"model": "m", "messages": [{"role": "function", "content": "x"}]},
        {"model": "m", "messages": [{"role": "user", "content": 42}]},   # 非字符串 content
        {"model": "m", "messages": [{"role": "user", "content": "x",
                                     "name": "bob"}]},     # 未支持消息字段
    ])
    def test_invalid_messages_rejected(self, payload):
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate(payload)

    def test_chat_model_missing_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate(
                {"messages": [{"role": "user", "content": "x"}]})

    def test_chat_invalid_sampling_params_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate({
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 0})
        with pytest.raises(ValidationError):
            ChatCompletionRequest.model_validate({
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "top_p": 1.01})


class TestErrorMessageContract:
    """错误体形状：param 定位字段、code 稳定，且永不序列化堆栈。"""

    def test_validation_error_locates_field(self):
        with pytest.raises(ValidationError) as exc_info:
            CompletionRequest.model_validate({"model": "m", "prompt": ""})
        errors = exc_info.value.errors()
        assert errors[0]["loc"][-1] == "prompt"

    def test_error_payload_shape_via_api_error(self):
        """APIError.to_payload 输出 OpenAI 风格四字段错误体。"""
        from nanoserve.service import InvalidRequestError

        err = InvalidRequestError("messages must contain at least one item",
                                  param="messages")
        payload = err.to_payload()
        assert set(payload["error"]) == {"message", "type", "param", "code"}
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["param"] == "messages"
        # 异常堆栈不出现在错误体中
        assert "Traceback" not in str(payload)
