"""Score RL prompts by the policy's pass rate and keep the ones GRPO can learn from.

GRPO's advantage is (reward - group mean): a prompt the policy always solves or never solves gives
every completion in its group the same reward, so zero gradient. Keeping prompts whose pass rate
is strictly between 0 and 1 (solved by some but not all of k samples) makes every training group
informative (DAPO's "dynamic sampling", done offline).

Sampling matches GRPO rollouts, not the eval protocol: T=1.0, top_p=1.0, no top_k, no presence
penalty, max_tokens = the RL completion budget.

Outputs:
  <out_dir>/<name>-scored.parquet   every scored prompt with `pass_rate`, `mean_tokens`, `truncated`
  <out_dir>/<name>.parquet          only prompts with pass_rate strictly between 0 and 1 (the GRPO training set)

Usage:
  uv run scripts/filter_by_passrate.py --model Qwen/Qwen3.5-0.8B-Base --mode zero --name rlzero-0.8b
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from posttrain.prompts import STOP_STRINGS, render_prompt
from posttrain.rewards import is_correct


def _grade_chunk(args: tuple[list[str], list[str], bool]) -> list[bool]:
    # Runs in worker processes: each has its own main thread, so math_verify's signal-based timeouts work.
    texts, golds, think = args
    return [bool(is_correct(t, g, think=think)) for t, g in zip(texts, golds, strict=True)]


def grade_parallel(texts: list[str], golds: list[str], think: bool, workers: int = 16, chunk: int = 500) -> list[bool]:
    jobs = [(texts[i : i + chunk], golds[i : i + chunk], think) for i in range(0, len(texts), chunk)]
    with ProcessPoolExecutor(workers) as pool:
        return [ok for part in pool.map(_grade_chunk, jobs) for ok in part]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, default=Path("data/rl/pool.parquet"))
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["think", "zero"], required=True)
    p.add_argument("--name", required=True, help="output file stem, e.g. rlzero-0.8b")
    p.add_argument("--out-dir", type=Path, default=Path("data/rl"))
    p.add_argument("--n-prompts", type=int, default=20000, help="random subset of the pool to score")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096, help="match GRPO max_completion_length")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    pool = pd.read_parquet(args.pool)
    df = pool.sample(n=min(args.n_prompts, len(pool)), random_state=args.seed).reset_index(drop=True)
    print(f"[data] scoring {len(df)} of {len(pool)} prompts, k={args.k}, max_tokens={args.max_tokens}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=1024 + args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    prompts = [render_prompt(problem, args.mode, tok) for problem in df.problem]
    # GRPO rollout sampling (TRL defaults), not the eval protocol
    params = SamplingParams(
        n=args.k, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens, stop=STOP_STRINGS[args.mode], seed=args.seed
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    print(f"[gen] {len(prompts) * args.k} completions in {time.time() - t0:.0f}s")

    texts = [s.text for out in outputs for s in out.outputs]
    golds = [gold for gold in df.solution for _ in range(args.k)]
    t0 = time.time()
    correct = grade_parallel(texts, golds, think=args.mode == "think")
    print(f"[grade] {len(texts)} completions in {time.time() - t0:.0f}s")

    k = args.k
    df["pass_rate"] = [sum(correct[i * k : (i + 1) * k]) / k for i in range(len(df))]
    df["mean_tokens"] = [sum(len(s.token_ids) for s in out.outputs) / k for out in outputs]
    df["truncated"] = [sum(s.finish_reason == "length" for s in out.outputs) / k for out in outputs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / f"{args.name}-scored.parquet")
    keep = df[(df.pass_rate > 0) & (df.pass_rate < 1)].reset_index(drop=True)
    keep.to_parquet(args.out_dir / f"{args.name}.parquet")

    print("\npass rate   all    dapo  deepmath")
    for lo, hi, label in [(0, 0, "0"), (1 / k, 3 / k, f"1-3/{k}"), (4 / k, 7 / k, f"4-7/{k}"), (1, 1, f"{k}/{k}")]:
        m = (df.pass_rate >= lo) & (df.pass_rate <= hi)
        print(f"{label:9s} {m.mean():6.1%} {m[df.source == 'dapo'].mean():6.1%} {m[df.source == 'deepmath'].mean():6.1%}")
    print(f"\nmean pass rate {df.pass_rate.mean():.1%}  truncated {df.truncated.mean():.1%}  mean tokens {df.mean_tokens.mean():.0f}")
    print(f"[done] kept {len(keep)} / {len(df)} prompts with pass rate strictly between 0 and 1 -> {args.out_dir / (args.name + '.parquet')}")
    print(f"       by source: {keep.source.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
