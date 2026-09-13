"""Day13 SSE 协议与流/非流一致性 CPU/ASGI 测试。

不加载 GPU/模型；使用 FakeEngine/FakeTokenizer。重复执行：
    python -m pytest tests/test_service_sse.py -q
"""
import json

from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

MODEL = "Qwen3-0.6B"


def client(engine):
    app = create_app(
        engine_factory=make_fake_factory(engine=engine),
        server_config=ServerConfig(model="/tmp/fake", model_id=MODEL),
    )
    return TestClient(app)


def frames(text):
    result = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        data = block.removeprefix("data: ")
        result.append(data if data == "[DONE]" else json.loads(data))
    return result


def test_completion_sse_order_and_usage():
    tok = FakeTokenizer(token_text={21: "A", 22: "B"})
    with client(FakeEngine(tokenizer=tok, completion_tokens=(21, 22))) as c:
        response = c.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "max_tokens": 2, "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = frames(response.text)
    assert events[-1] == "[DONE]"
    json_events = events[:-1]
    assert json_events[0]["choices"][0]["text"] == ""
    assert [e["choices"][0]["text"] for e in json_events[1:-1]] == ["A", "B"]
    assert json_events[-1]["choices"][0]["finish_reason"] == "length"
    assert json_events[-1]["usage"]["completion_tokens"] == 2
    assert response.headers["x-request-id"] == json_events[0]["id"]


def test_chat_sse_first_role_and_done_once():
    tok = FakeTokenizer(token_text={21: "A"})
    with client(FakeEngine(tokenizer=tok, completion_tokens=(21,))) as c:
        response = c.post("/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1, "stream": True})
    assert response.status_code == 200
    events = frames(response.text)
    assert events[-1] == "[DONE]"
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert events[1]["choices"][0]["delta"] == {"content": "A"}
    assert events[-2]["choices"][0]["finish_reason"] == "length"
    assert response.text.count("data: [DONE]") == 1


def test_stream_aggregates_same_as_non_stream():
    tok = FakeTokenizer(token_text={21: "A", 22: "B", 23: "C"})
    with client(FakeEngine(tokenizer=tok, completion_tokens=(21, 22, 23))) as c:
        streamed = c.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "max_tokens": 3, "stream": True})
    tok2 = FakeTokenizer(token_text={21: "A", 22: "B", 23: "C"})
    with client(FakeEngine(tokenizer=tok2, completion_tokens=(21, 22, 23))) as c:
        regular = c.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "max_tokens": 3})
    events = frames(streamed.text)[:-1]
    aggregate = "".join(e["choices"][0]["text"] for e in events[1:-1])
    assert aggregate == regular.json()["choices"][0]["text"]
    assert events[-1]["usage"] == regular.json()["usage"]


def test_stream_schema_error_is_json_before_first_frame():
    with client(FakeEngine()) as c:
        response = c.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "stream": "yes"})
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "invalid_request_error"
