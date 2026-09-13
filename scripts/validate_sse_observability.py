"""Day13–14 CPU 验收入口：SSE、指标和生命周期日志。

不加载 GPU/模型，使用 FakeEngine；真实 TCP/GPU 验收需使用独立服务地址。
重复执行：python scripts/validate_sse_observability.py --mode cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 直接以 `python scripts/...` 运行时，将仓库根目录加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from nanoserve.app import create_app
from nanoserve.config import ServerConfig
from nanoserve.testing import FakeEngine, FakeTokenizer, make_fake_factory

MODEL = "Qwen3-0.6B"


def check(case: str, ok: bool, *, details: dict | None = None, output=None):
    row = {"case": case, "status": "pass" if ok else "fail",
           "observed_at": time.perf_counter()}
    if details:
        row.update(details)
    print(json.dumps(row, ensure_ascii=False))
    if output is not None:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return ok


def run_cpu(output=None) -> bool:
    tokenizer = FakeTokenizer(token_text={21: "A", 22: "B", 23: "C"})
    engine = FakeEngine(tokenizer=tokenizer, completion_tokens=(21, 22, 23))
    app = create_app(
        engine_factory=make_fake_factory(engine=engine),
        server_config=ServerConfig(model="/tmp/fake", model_id=MODEL),
    )
    passed = True
    with TestClient(app) as client:
        response = client.post("/v1/completions", json={
            "model": MODEL, "prompt": "x", "max_tokens": 3, "stream": True})
        frames = [part for part in response.text.split("\n\n") if part]
        passed &= check(
            "completion_sse_contract",
            response.status_code == 200
            and response.headers.get("content-type", "").startswith("text/event-stream")
            and frames[-1] == "data: [DONE]"
            and sum(frame.startswith("data: {") for frame in frames) >= 5,
            details={"http_status": response.status_code}, output=output)

        response = client.post("/v1/chat/completions", json={
            "model": MODEL, "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 2, "stream": True})
        passed &= check(
            "chat_sse_contract",
            response.status_code == 200 and "chat.completion.chunk" in response.text
            and response.text.count("data: [DONE]") == 1,
            details={"http_status": response.status_code}, output=output)

        metrics = client.get("/metrics")
        required = (
            "request_queue_time_seconds", "time_to_first_token_seconds",
            "time_per_output_token_seconds", "request_latency_seconds",
            "prompt_tokens_total", "generation_tokens_total",
            "kv_cache_utilization", "prefix_cache_hit_rate", "running_requests",
        )
        passed &= check("metrics_contract", metrics.status_code == 200
                        and all(f"# HELP {name}" in metrics.text for name in required),
                        details={"http_status": metrics.status_code}, output=output)

    passed &= check("worker_single_owner", len(engine.step_thread_ids) == 1,
                    details={"step_threads": len(engine.step_thread_ids)}, output=output)
    return bool(passed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output.open("w", encoding="utf-8") if args.output else None
    try:
        ok = run_cpu(output)
    finally:
        if output is not None:
            output.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
