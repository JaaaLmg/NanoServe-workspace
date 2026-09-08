"""Day 4 deterministic sampling experiment.

Run with ``python experiments/day4_prefill_decode.py``.  The CPU portion is
model-independent; pass --model to additionally run five prompts through LLM.
"""
import argparse, json, random, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams

PROMPTS = ["Explain caching in one sentence.", "What is a tensor?", "Give three colors.", "Why batch requests?", "Define latency."]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="experiments/day4_results.json"); p.add_argument("--model")
    args = p.parse_args(); torch.manual_seed(2026); random.seed(2026)
    logits = torch.tensor([[3.0, 1.0, .2, -.5]] * len(PROMPTS))
    greedy = Sampler.sample(logits, torch.zeros(len(PROMPTS))).tolist()
    greedy2 = Sampler.sample(logits, torch.zeros(len(PROMPTS))).tolist()
    g = [torch.Generator().manual_seed(2026+i) for i in range(len(PROMPTS))]
    sampled = Sampler.sample(logits, torch.ones(len(PROMPTS)), torch.full((len(PROMPTS),), .8), g).tolist()
    result = {"seed": 2026, "prompts": PROMPTS, "greedy_run_1": greedy, "greedy_run_2": greedy2,
              "greedy_reproducible": greedy == greedy2, "sampled_top_p_0.8": sampled,
              "max_tokens": SamplingParams(max_tokens=8).max_tokens,
              "environment": {"python": platform.python_version(), "torch": torch.__version__}}
    if args.model:
        try:
            from nanovllm import LLM
            llm = LLM(args.model, enforce_eager=True)
            result["model_outputs"] = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=8, seed=2026), use_tqdm=False)
        except Exception as exc:
            result["model_error"] = f"{type(exc).__name__}: {exc}"
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == "__main__": main()
