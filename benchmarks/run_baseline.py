"""Baseline benchmark runner with request-level timing.

GPU/model runs are intentionally explicit: unavailable or OOM cases are kept
in the JSON rather than silently omitted.  The engine API is synchronous, so
TTFT is measured at the first completed decode step.
"""
import argparse, atexit, json, os, platform, subprocess, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


def percentile(values, q):
    if not values: return None
    values = sorted(values); return values[min(len(values)-1, int((len(values)-1)*q))]

def run_case(model, concurrency, prompt_len, output_len, enforce_eager):
    started = time.perf_counter(); record = {"status":"ok", "concurrency":concurrency, "prompt_tokens":prompt_len, "max_tokens":output_len, "requests":[]}
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(model, enforce_eager=enforce_eager, max_num_seqs=concurrency)
        prompts = [list(range(100, 100 + prompt_len)) for _ in range(concurrency)]
        params = SamplingParams(temperature=0, max_tokens=output_len, seed=1234)
        request_ids = []
        for prompt in prompts:
            llm.add_request(prompt, params)
            request_ids.append(max(seq.seq_id for seq in llm.scheduler.waiting))
        submit = time.perf_counter()
        first = {}; done = {}
        while not llm.is_finished():
            active_ids = [seq.seq_id for seq in llm.scheduler.running]
            is_prefill = not bool(active_ids) or any(seq.is_prefill for seq in llm.scheduler.running)
            before = time.perf_counter(); outputs, _ = llm.step(); after = time.perf_counter()
            if not is_prefill:
                for seq_id in active_ids:
                    first.setdefault(seq_id, after)
            for seq_id, tokens in outputs: done[seq_id] = after
        total = time.perf_counter() - started
        for i, seq_id in enumerate(request_ids):
            ft = first.get(seq_id); dt = done.get(seq_id, time.perf_counter())
            record["requests"].append({"request_index":i, "submit_time":submit, "first_token_time":ft, "finish_time":dt, "ttft":None if ft is None else ft-submit, "latency":dt-submit, "output_tokens":output_len})
        ttfts = [r["ttft"] for r in record["requests"] if r["ttft"] is not None]
        record["summary"] = {"ttft_p50":percentile(ttfts,.5), "ttft_p95":percentile(ttfts,.95), "request_throughput":concurrency/total, "output_throughput":concurrency*output_len/total, "itl_p50":None, "itl_p95":None}
        if torch.cuda.is_available(): record["memory"]={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        else: record["memory"]={"available":False}
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Each case builds its own engine; without tearing it down the next
        # case hits "default process group twice" and old KV cache stays resident.
        if llm is not None:
            try: atexit.unregister(llm.exit); llm.exit()
            except Exception as cleanup_exc: print(f"cleanup warning: {cleanup_exc}", file=sys.stderr)
            llm = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return record

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--output", default="benchmarks/results/baseline.json"); ap.add_argument("--concurrency", nargs="+", type=int, default=[1,8,32]); ap.add_argument("--prompt-lens", nargs="+", type=int, default=[128,512,2048]); ap.add_argument("--output-lens", nargs="+", type=int, default=[32,128]); ap.add_argument("--enforce-eager", action="store_true")
    a=ap.parse_args(); cases=[]
    for c in a.concurrency:
      for p in a.prompt_lens:
       for o in a.output_lens: cases.append(run_case(a.model,c,p,o,a.enforce_eager))
    out={"schema_version":1,"created_at":time.time(),"config":vars(a),"environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"cases":cases}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2,default=str)+"\n"); print(path)
if __name__=="__main__": main()
