"""Day11–12 OpenAI 兼容 API 验收脚本（docs/openai-api-day11-12.md §9.1/§9.2）。

两种模式：
- --mode cpu（默认）：fake Engine + FastAPI TestClient（ASGI transport），
  覆盖 §9.1 的 10 组检查：health 状态迁移、models 与请求校验一致性、
  非流式成功响应、模板调用参数、采样映射一致性、错误码、单 step 驱动者、
  异常/关闭无悬挂、drain 幂等与 ID 复用隔离、退出可重复。
  不加载 GPU/权重。
- --mode gpu：对已启动的真实服务（python -m nanoserve.server）执行
  §9.2 的 curl 等价检查（health/models/completion/chat/错误路径）。
  GPU 环境准备与启动命令见设计文档 §9.2，脚本只做 HTTP 验证。

输出 JSONL（stdout 或 --output 指定文件），字段：case/status/http_status/
request_id/error_code/observed_at；不记录 prompt 明文。

用法：
    python scripts/validate_openai_api.py --mode cpu \
        [--output docs/evidence/day11-12/cpu-validation.jsonl]
    python scripts/validate_openai_api.py --mode gpu \
        --base-url http://127.0.0.1:8000 --model-id Qwen3-0.6B
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

# 脚本可直接以 `python scripts/validate_openai_api.py` 运行：
# 把仓库根目录加入 sys.path，保证 nanoserve/nanovllm 可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_ID = "Qwen3-0.6B"


def emit(records: list[dict], case: str, status: str, *,
         http_status: int | None = None, request_id: str | None = None,
         error_code: str | None = None) -> bool:
    """追加一条 JSONL 记录并返回是否通过。"""
    records.append({
        "case": case,
        "status": status,
        "http_status": http_status,
        "request_id": request_id,
        "error_code": error_code,
        "observed_at": perf_counter(),
    })
    return status == "pass"


def error_code_of(resp) -> str | None:
    try:
        return resp.json().get("error", {}).get("code")
    except Exception:
        return None


def error_id_of(resp) -> str | None:
    return resp.headers.get("x-request-id")


# ============================== CPU stub/ASGI 模式 ==============================

def run_cpu_mode(records: list[dict]) -> int:
    from fastapi.testclient import TestClient

    from nanoserve.app import create_app
    from nanoserve.config import ServerConfig
    from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

    server_config = ServerConfig(model="/tmp/fake-model", model_id=MODEL_ID)
    engine = FakeEngine()
    tokenizer = engine.tokenizer
    app = create_app(engine_factory=make_fake_factory(
        engine=engine, max_model_len=64), server_config=server_config)

    failures = 0

    def check(case: str, condition: bool, *, resp=None, request_id=None,
              error_code=None):
        nonlocal failures
        ok = emit(records, case, "pass" if condition else "fail",
                  http_status=resp.status_code if resp is not None else None,
                  request_id=request_id, error_code=error_code)
        if not ok:
            failures += 1

    # 1) health 状态迁移（本模式 app 由 with 直接进入 ready；draining 在末段覆盖）
    with TestClient(app) as client:
        resp = client.get("/health")
        check("health_ready", resp.status_code == 200
              and resp.json()["status"] == "ok", resp=resp)

        # 2) /v1/models 与请求校验一致
        resp = client.get("/v1/models")
        models_ok = resp.status_code == 200 and \
            [m["id"] for m in resp.json()["data"]] == [MODEL_ID]
        resp_bad_model = client.post("/v1/completions",
                                     json={"model": "other", "prompt": "x"})
        models_ok = models_ok and resp_bad_model.status_code == 404
        check("models_and_validation_consistent", models_ok,
              resp=resp, error_code=error_code_of(resp_bad_model))

        # 3) 非流式成功响应（completion 与 chat 的 OpenAI 字段/usage/finish）
        resp = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": "hello", "max_tokens": 2,
            "temperature": 0.0})
        body = resp.json() if resp.status_code == 200 else {}
        usage = body.get("usage", {})
        completion_ok = (
            resp.status_code == 200
            and body.get("object") == "text_completion"
            and body.get("id", "").startswith("cmpl-")
            and resp.headers.get("x-request-id") == body.get("id")
            and body.get("choices", [{}])[0].get("finish_reason") in
            ("stop", "length")
            and usage.get("total_tokens")
            == usage.get("prompt_tokens", -1) + usage.get("completion_tokens", -1))
        check("completion_non_stream", completion_ok, resp=resp,
              request_id=body.get("id"))

        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2})
        body = resp.json() if resp.status_code == 200 else {}
        choices = body.get("choices", [])
        usage = body.get("usage", {})
        chat_ok = (
            resp.status_code == 200
            and body.get("object") == "chat.completion"
            and body.get("model") == MODEL_ID
            and body.get("id", "").startswith("chatcmpl-")
            and resp.headers.get("x-request-id") == body.get("id")
            and len(choices) == 1
            and choices[0].get("index") == 0
            and choices[0].get("message", {}).get("role") == "assistant"
            and isinstance(choices[0].get("message", {}).get("content"), str)
            and choices[0].get("finish_reason") in ("stop", "length")
            and usage.get("total_tokens") == usage.get("prompt_tokens", -1)
            + usage.get("completion_tokens", -1))
        check("chat_non_stream", chat_ok, resp=resp,
              request_id=body.get("id"))

        # 4) chat template 调用参数（原始 messages / tokenize / generation prompt）
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
            "max_tokens": 8})
        call = tokenizer.calls[-1] if tokenizer.calls else {}
        template_ok = (
            resp.status_code == 200
            and call.get("tokenize") is True
            and call.get("add_generation_prompt") is True
            and [m["role"] for m in call.get("messages", [])]
            == ["system", "user"])
        check("chat_template_call_args", template_ok, resp=resp)

        # 5) 相同采样参数在两条路由的 SamplingParams 映射一致
        engine.added_requests.clear()
        resp_c = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": "abc", "max_tokens": 5,
            "temperature": 0.3, "top_p": 0.8})
        resp_ch = client.post("/v1/chat/completions", json={
            "model": MODEL_ID, "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 5, "temperature": 0.3, "top_p": 0.8})
        if len(engine.added_requests) == 2:
            params_c = engine.added_requests[0][1]
            params_ch = engine.added_requests[1][1]
            params_ok = (params_c.temperature, params_c.top_p,
                         params_c.max_tokens) == \
                        (params_ch.temperature, params_ch.top_p,
                         params_ch.max_tokens) == (0.3, 0.8, 5)
        else:
            params_ok = False
        check("sampling_params_consistent", params_ok, resp=resp_c)

        # 6) 错误路径：非法 role / 空 messages / 上下文超限 / 模型不匹配 /
        #    stream=true
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID,
            "messages": [{"role": "tool", "content": "x"}]})
        check("error_invalid_role", resp.status_code == 400
              and error_code_of(resp) == "invalid_request_error", resp=resp,
              error_code=error_code_of(resp))
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID, "messages": []})
        check("error_empty_messages", resp.status_code == 400, resp=resp,
              error_code=error_code_of(resp))
        resp = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": "a" * 60, "max_tokens": 16})
        check("error_context_length", resp.status_code == 400
              and error_code_of(resp) == "context_length_exceeded",
              resp=resp, error_code=error_code_of(resp))
        resp = client.post("/v1/completions",
                           json={"model": "nope", "prompt": "x"})
        check("error_model_not_found", resp.status_code == 404
              and error_code_of(resp) == "model_not_found", resp=resp,
              error_code=error_code_of(resp))
        for route, payload in [
            ("/v1/completions", {"model": MODEL_ID, "prompt": "x",
                                 "stream": True}),
            ("/v1/chat/completions", {"model": MODEL_ID,
                                       "messages": [{"role": "user",
                                                     "content": "x"}],
                                       "stream": True}),
        ]:
            resp = client.post(route, json=payload)
            check("error_stream_501", resp.status_code == 501
                  and error_code_of(resp) == "stream_not_implemented",
                  resp=resp, error_code=error_code_of(resp))

        # 7) 并发请求：多个 HTTP 等待者同时提交，结果必须按各自 response
        # body 关联；FakeEngine 的 step 仍只能由一个 worker 线程驱动。
        engine.step_thread_ids.clear()
        barrier = threading.Barrier(3)

        def issue(i):
            barrier.wait()
            return client.post("/v1/completions", json={
                "model": MODEL_ID, "prompt": f"req{i}", "max_tokens": 2})

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(issue, range(3)))
        concurrent_ok = len(engine.step_thread_ids) == 1
        concurrent_ok = concurrent_ok and all(
            response.status_code == 200
            and response.json().get("object") == "text_completion"
            and response.json().get("model") == MODEL_ID
            and response.json().get("id", "").startswith("cmpl-")
            and response.headers.get("x-request-id") == response.json().get("id")
            and len(response.json().get("choices", [])) == 1
            and response.json()["choices"][0]["finish_reason"] in ("stop", "length")
            for response in responses)
        check("concurrent_requests_single_step_driver", concurrent_ok,
              request_id=str([r.json().get("id") for r in responses]))

        # 8) Engine 异常与关闭后无悬挂 handle（脚本抛错 → 500 engine_error）
        engine.step_script.append(RuntimeError("boom"))
        resp = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": "x", "max_tokens": 8})
        check("engine_error_500", resp.status_code == 500
              and error_code_of(resp) == "engine_error", resp=resp,
              error_code=error_code_of(resp))
        # 关闭阶段：停止接收后新请求 503，不悬挂
        service = client.app.state.service
        service.mark_draining()
        service.manager.stop_accepting()
        resp = client.post("/v1/completions",
                           json={"model": MODEL_ID, "prompt": "x"})
        check("draining_503", resp.status_code == 503
              and error_code_of(resp) == "service_draining", resp=resp,
              error_code=error_code_of(resp))

    # 9/10) 完成记录 drain 幂等、ID 复用隔离、退出可重复（Engine 层直接验证）
    from types import SimpleNamespace

    from nanovllm.sampling_params import SamplingParams

    from nanoserve.service import InternalRequest, RequestManager

    engine2 = FakeEngine()
    manager = RequestManager(engine2.tokenizer)
    manager.attach_worker(SimpleNamespace(submit=lambda r: None))
    request = InternalRequest(
        request_id="dup", kind="completion", prompt_token_ids=(1, 2),
        sampling_params=SamplingParams(max_tokens=2), model_id=MODEL_ID,
        created_at=perf_counter(), deadline=None)
    handle = manager.submit(request)
    engine2.add_request((1, 2), request.sampling_params, "dup")
    engine2.step()
    for record in engine2.pop_completed():
        manager.resolve_completed(record)
    result = manager.wait(handle)
    drain_ok = engine2.pop_completed() == [] and \
        result.completion_tokens == 2
    # ID 复用：旧记录（迟到重复 drain 后不再存在）不会完成复用 ID 的新句柄
    engine2.add_request((1, 2), request.sampling_params, "dup")
    engine2.step()
    records2 = engine2.pop_completed()
    id_reuse_ok = len(records2) == 1 and records2[0].request_id == "dup"
    check("record_drain_idempotent", drain_ok)
    check("request_id_reuse_isolated", id_reuse_ok)

    exit_ok = (engine2.exit_calls == 0)
    engine2.exit()
    engine2.exit()
    exit_ok = exit_ok and engine2.exit_calls == 1
    check("exit_idempotent", exit_ok)

    return failures


# ============================== GPU 真实服务模式 ==============================

def run_gpu_mode(records: list[dict], base_url: str) -> int:
    import httpx

    failures = 0

    def check(case: str, condition: bool, *, resp=None, request_id=None,
              error_code=None):
        nonlocal failures
        ok = emit(records, case, "pass" if condition else "fail",
                  http_status=resp.status_code if resp is not None else None,
                  request_id=request_id, error_code=error_code)
        if not ok:
            failures += 1

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        resp = client.get("/health")
        check("gpu_health", resp.status_code == 200
              and resp.json().get("status") == "ok", resp=resp)

        resp = client.get("/v1/models")
        models = resp.json().get("data", []) if resp.status_code == 200 else []
        check("gpu_models", resp.status_code == 200
              and len(models) == 1 and models[0].get("id") == MODEL_ID
              and models[0].get("object") == "model",
              resp=resp)

        resp = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": "介绍你自己", "max_tokens": 16,
            "temperature": 0})
        body = resp.json() if resp.status_code == 200 else {}
        usage = body.get("usage", {})
        choices = body.get("choices", [])
        check("gpu_completion", resp.status_code == 200
              and body.get("object") == "text_completion"
              and body.get("model") == MODEL_ID
              and body.get("id", "").startswith("cmpl-")
              and resp.headers.get("x-request-id") == body.get("id")
              and len(choices) == 1 and choices[0].get("index") == 0
              and isinstance(choices[0].get("text"), str)
              and choices[0].get("finish_reason") in ("stop", "length")
              and all(isinstance(usage.get(key), int) and usage[key] >= 0
                      for key in ("prompt_tokens", "completion_tokens",
                                  "total_tokens"))
              and usage.get("total_tokens")
              == usage.get("prompt_tokens", -1)
              + usage.get("completion_tokens", -1),
              resp=resp, request_id=body.get("id"))

        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "什么是 continuous batching？"}],
            "max_tokens": 16, "temperature": 0})
        body = resp.json() if resp.status_code == 200 else {}
        usage = body.get("usage", {})
        choices = body.get("choices", [])
        check("gpu_chat", resp.status_code == 200
              and body.get("object") == "chat.completion"
              and body.get("model") == MODEL_ID
              and body.get("id", "").startswith("chatcmpl-")
              and resp.headers.get("x-request-id") == body.get("id")
              and len(choices) == 1 and choices[0].get("index") == 0
              and choices[0].get("message", {}).get("role") == "assistant"
              and isinstance(choices[0].get("message", {}).get("content"), str)
              and choices[0].get("finish_reason") in ("stop", "length")
              and all(isinstance(usage.get(key), int) and usage[key] >= 0
                      for key in ("prompt_tokens", "completion_tokens",
                                  "total_tokens"))
              and usage.get("total_tokens")
              == usage.get("prompt_tokens", -1)
              + usage.get("completion_tokens", -1),
              resp=resp, request_id=body.get("id"))

        for route, payload in [
            ("/v1/completions", {"model": MODEL_ID, "prompt": "x",
                                 "stream": True}),
            ("/v1/chat/completions", {"model": MODEL_ID,
                                       "messages": [{"role": "user",
                                                     "content": "x"}],
                                       "stream": True}),
        ]:
            resp = client.post(route, json=payload)
            check("gpu_stream_501", resp.status_code == 501
                  and error_code_of(resp) == "stream_not_implemented",
                  resp=resp, error_code=error_code_of(resp))

        resp = client.post("/v1/chat/completions", json={
            "model": MODEL_ID, "messages": [{"role": "tool", "content": "x"}]})
        check("gpu_invalid_role_400", resp.status_code == 400
              and error_code_of(resp) == "invalid_request_error", resp=resp,
              error_code=error_code_of(resp))

        # 不记录 prompt 明文，只校验上下文预算错误码。
        # 字符数不等于 token 数（BPE 可能合并重复字符）；使用足够长的
        # 空格分隔词序列，确保真实 tokenizer 编码后超过 4096 tokens。
        oversized_prompt = "hello " * 5000
        resp = client.post("/v1/completions", json={
            "model": MODEL_ID, "prompt": oversized_prompt, "max_tokens": 8})
        check("gpu_context_length_400", resp.status_code == 400
              and error_code_of(resp) == "context_length_exceeded", resp=resp,
              error_code=error_code_of(resp))

    return failures


def main(argv=None) -> int:
    global MODEL_ID
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="gpu 模式的服务地址")
    parser.add_argument("--model-id", default=MODEL_ID,
                        help="gpu 模式下请求使用的 model ID")
    parser.add_argument("--output", default=None,
                        help="JSONL 输出文件路径（默认 stdout）")
    args = parser.parse_args(argv)
    MODEL_ID = args.model_id

    records: list[dict] = []
    started = time.time()
    if args.mode == "cpu":
        failures = run_cpu_mode(records)
    else:
        failures = run_gpu_mode(records, args.base_url)

    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    summary = json.dumps({
        "case": "summary", "mode": args.mode,
        "status": "pass" if failures == 0 else "fail",
        "total": len(records), "failures": failures,
        "wall_seconds": round(time.time() - started, 3),
        "observed_at": perf_counter(),
    }, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines + [summary]) + "\n")
        print(f"共 {len(records)} 项检查，失败 {failures} 项；"
              f"JSONL 已写入 {args.output}")
    else:
        print("\n".join(lines + [summary]))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
