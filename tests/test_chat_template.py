"""Day11–12 chat template 与上下文长度测试（docs/openai-api-day11-12.md §8.3）。

覆盖范围：
- apply_chat_template 收到原始有序 messages、tokenize=True、add_generation_prompt=True；
- 模板返回 token 列表时不重复编码；返回字符串时只调用同一 tokenizer 编码一次；
- 模板抛出异常时返回可理解的 400（InvalidRequestError），不泄漏 traceback；
- completion 与 chat 的 SamplingParams 对相同字段产生相同值（同一构造路径）；
- prompt token 数 + max_tokens == max_model_len 通过；大于上限被拒绝；
- 拒绝发生在 engine.add_request 之前（fake engine 无任何请求登记）。

无 GPU/模型依赖：使用 FakeTokenizer 桩，不加载权重、不初始化 CUDA。

重复执行命令：
    python -m pytest tests/test_chat_template.py -q
    python -O -m pytest tests/test_chat_template.py -q
"""

import pytest
from pydantic import ValidationError

from nanoserve.schemas import ChatCompletionRequest, ChatMessage, CompletionRequest
from nanoserve.service import (ContextLengthExceededError,
                               InvalidRequestError, build_chat_request,
                               build_completion_request,
                               build_sampling_params)
from nanoserve.testing import FakeTokenizer

MODEL_ID = "Qwen3-0.6B"
MAX_MODEL_LEN = 4096

MIXED_MESSAGES = [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "解释什么是 continuous batching。"},
    {"role": "assistant", "content": "每轮调度合并不同请求。"},
    {"role": "user", "content": "再举一个例子。"},
]


def _chat_request(messages: list[dict], **kwargs) -> ChatCompletionRequest:
    payload = {"model": MODEL_ID, "messages": messages}
    payload.update(kwargs)
    return ChatCompletionRequest.model_validate(payload)


def _build_chat(tokenizer: FakeTokenizer, messages: list[dict], **kwargs):
    req = _chat_request(messages, **kwargs)
    return build_chat_request(req, tokenizer, model_id=MODEL_ID,
                              max_model_len=MAX_MODEL_LEN)


class TestChatTemplateCalls:
    """模板调用契约：参数、顺序与编码次数。"""

    def test_template_receives_ordered_messages_and_flags(self):
        """apply_chat_template 收到原始有序 messages 与固定调用参数。"""
        tokenizer = FakeTokenizer()
        internal = _build_chat(tokenizer, MIXED_MESSAGES, max_tokens=8)
        assert len(tokenizer.calls) == 1
        call = tokenizer.calls[0]
        assert call["messages"] == MIXED_MESSAGES  # 原样、保序
        assert call["tokenize"] is True
        assert call["add_generation_prompt"] is True

    def test_token_list_used_directly_without_reencoding(self):
        """tokenize=True 的 token 列表直接作为 prompt_token_ids，不再编码。"""
        tokenizer = FakeTokenizer()
        internal = _build_chat(tokenizer, MIXED_MESSAGES)
        # FakeTokenizer 的确定性模板序列：头 [11,12] + 每消息一个标记 + 生成提示
        assert internal.prompt_token_ids == (11, 12, 20, 21, 22, 23, 99)
        # 全程未调用 encode（模板结果未经历字符串再编码）
        assert tokenizer.encode_calls == []

    def test_string_template_encoded_once_with_same_tokenizer(self):
        """tokenizer 只能返回字符串时：用同一 tokenizer 再编码恰好一次。"""
        tokenizer = FakeTokenizer(template_style="str")
        internal = _build_chat(tokenizer, MIXED_MESSAGES)
        assert tokenizer.encode_calls == ["<template-string>"]
        assert internal.prompt_token_ids == \
            tuple(tokenizer.encode("<template-string>"))

    def test_template_exception_maps_to_400_without_traceback(self):
        """模板失败 → 可理解的 400，异常消息不含原始 traceback。"""
        tokenizer = FakeTokenizer(
            template_error=RuntimeError("jinja bug at line 42\nboom"))
        with pytest.raises(InvalidRequestError) as exc_info:
            _build_chat(tokenizer, MIXED_MESSAGES)
        assert exc_info.value.status_code == 400
        assert exc_info.value.code == "invalid_request_error"
        assert exc_info.value.param == "messages"
        # 不泄漏原始异常内容与堆栈关键词
        assert "jinja" not in str(exc_info.value)
        assert "Traceback" not in str(exc_info.value)

    def test_batch_encoding_template_output_normalized(self):
        """transformers 5.x 形态（BatchEncoding/dict 含 input_ids）：取首条序列。"""
        tokenizer = FakeTokenizer(template_style="dict")
        internal = _build_chat(tokenizer, MIXED_MESSAGES)
        assert internal.prompt_token_ids == (11, 12, 20, 21, 22, 23, 99)
        assert tokenizer.encode_calls == []

    def test_role_whitelist_enforced_by_schema(self):
        """role 白名单在 schema 层严格生效，非法 role 到不了模板调用。"""
        tokenizer = FakeTokenizer()
        with pytest.raises(ValidationError):
            _build_chat(tokenizer, [{"role": "tool", "content": "x"}])
        # 模板从未被调用
        assert tokenizer.calls == []


class TestContextLengthBudget:
    """上下文预算：检查发生在 engine.add_request（KV 分配）之前。"""

    def test_exact_budget_passes(self):
        """prompt tokens + max_tokens == max_model_len 恰好通过。"""
        tokenizer = FakeTokenizer()
        prompt = "a" * (MAX_MODEL_LEN - 8)
        internal = build_completion_request(
            CompletionRequest.model_validate(
                {"model": MODEL_ID, "prompt": prompt, "max_tokens": 8}),
            tokenizer, model_id=MODEL_ID, max_model_len=MAX_MODEL_LEN)
        assert len(internal.prompt_token_ids) == MAX_MODEL_LEN - 8

    def test_over_budget_completion_rejected(self):
        tokenizer = FakeTokenizer()
        with pytest.raises(ContextLengthExceededError) as exc_info:
            build_completion_request(
                CompletionRequest.model_validate(
                    {"model": MODEL_ID, "prompt": "a" * (MAX_MODEL_LEN - 7),
                     "max_tokens": 8}),
                tokenizer, model_id=MODEL_ID, max_model_len=MAX_MODEL_LEN)
        assert exc_info.value.code == "context_length_exceeded"
        # 错误消息只含计数，不含 prompt 明文
        assert "aaaa" not in str(exc_info.value)

    @pytest.mark.parametrize("template_style", ["list", "dict", "str"])
    def test_chat_budget_uses_normalized_token_count(self, template_style):
        """所有模板返回形态都按规范化后的 token 数检查预算。"""
        tokenizer = FakeTokenizer(template_style=template_style)
        req = _chat_request(MIXED_MESSAGES, max_tokens=1)
        # 先在足够大的上下文中生成规范化 token，再用其精确长度构造边界。
        unconstrained = build_chat_request(
            req, tokenizer, model_id=MODEL_ID, max_model_len=4096)
        token_count = len(unconstrained.prompt_token_ids)
        with pytest.raises(ContextLengthExceededError):
            build_chat_request(req, FakeTokenizer(template_style=template_style),
                               model_id=MODEL_ID, max_model_len=token_count)

    def test_bad_template_output_maps_to_invalid_request(self):
        class BadTokenizer(FakeTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                return None

        with pytest.raises(InvalidRequestError) as exc_info:
            _build_chat(BadTokenizer(), MIXED_MESSAGES)
        assert exc_info.value.status_code == 400
        assert exc_info.value.param == "messages"

    def test_rejection_happens_before_engine_admission(self):
        """HTTP builder 在 Engine admission 前拒绝超限请求。"""
        from nanoserve.testing import FakeEngine

        # 通过实际路由/worker admission spy 验证，而不是断言一个从未传入
        # builder 的孤立 FakeEngine 状态。
        from fastapi.testclient import TestClient
        from nanoserve.app import create_app
        from nanoserve.config import ServerConfig
        from nanoserve.testing import make_fake_factory

        engine = FakeEngine()
        app = create_app(
            engine_factory=make_fake_factory(engine=engine, max_model_len=8),
            server_config=ServerConfig(model="/tmp/fake-model",
                                       model_id=MODEL_ID))
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 8,
            })
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert engine.added_requests == []
        assert not engine.has_active_requests()


class TestSamplingParamsConsistency:
    """completion/chat 共享同一 SamplingParams 构造路径（不变量 6）。"""

    def test_same_fields_same_params(self):
        params = build_sampling_params(max_tokens=16, temperature=0.0,
                                       top_p=0.9)
        assert (params.max_tokens, params.temperature, params.top_p) == \
            (16, 0.0, 0.9)
        # 两条路由的构建函数产出逐字段一致
        tokenizer = FakeTokenizer()
        completion = build_completion_request(
            CompletionRequest.model_validate(
                {"model": MODEL_ID, "prompt": "hi", "max_tokens": 16,
                 "temperature": 0.0, "top_p": 0.9}),
            tokenizer, model_id=MODEL_ID, max_model_len=MAX_MODEL_LEN)
        chat = _build_chat(tokenizer, [{"role": "user", "content": "hi"}],
                           max_tokens=16, temperature=0.0, top_p=0.9)
        assert completion.sampling_params == chat.sampling_params

    def test_sampling_params_final_validation_entry(self):
        """SamplingParams 仍是最终校验入口：越界值在这里显式失败。"""
        with pytest.raises(ValueError):
            build_sampling_params(max_tokens=1, temperature=-1.0, top_p=1.0)
        with pytest.raises(ValueError):
            build_sampling_params(max_tokens=1, temperature=1.0, top_p=0.0)

    def test_request_ids_prefixed_by_kind(self):
        """request_id 前缀区分来源：cmpl- / chatcmpl-（§4.1）。"""
        tokenizer = FakeTokenizer()
        completion = build_completion_request(
            CompletionRequest.model_validate(
                {"model": MODEL_ID, "prompt": "hi"}),
            tokenizer, model_id=MODEL_ID, max_model_len=MAX_MODEL_LEN)
        chat = _build_chat(tokenizer, [{"role": "user", "content": "hi"}])
        assert completion.request_id.startswith("cmpl-")
        assert chat.request_id.startswith("chatcmpl-")

    def test_chat_messages_reject_non_string_content_via_builder(self):
        """ChatMessage content 类型由 schema 保证；builder 收到的必为字符串。"""
        with pytest.raises(ValidationError):
            ChatMessage(role="user", content=1.5)
