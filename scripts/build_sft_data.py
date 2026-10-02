"""Build the SFT cold-start set from nvidia/Nemotron-SFT-Math-v3 (PLAN.md, Stage 1).

The dataset is one 154 GB JSONL file, so it is sampled instead of downloaded: the file is cut into
equal slices, and the start of each slice is read with an HTTP range request, in random order,
until enough traces are kept. Sampling across the whole file matters because it is laid out in
blocks: chain-of-thought rows fill roughly the first 65%, alternating between AoPS and
StackExchange blocks, and Python tool-use (TIR) rows the rest. Taking the first N rows would give
StackExchange problems only.

A trace is kept if it
  - is a plain chain-of-thought row (no TIR) with one user and one assistant turn,
  - has reasoning and a \\boxed{} answer, and no think tags inside either,
  - fits the token budgets (completion incl. <|im_end|> <= --max-completion-tokens, prompt <= 1024),
  - ends in an answer our own grader accepts against Nemotron's `expected_answer`, so every trace
    would earn the RL correctness reward (disable with --keep-unverified),
  - does not overlap an eval problem (same 13-gram rule as the RL pool),
  - is not a further trace for a problem that already has --max-per-problem.

Outputs (mode-neutral rows: id, problem, reasoning, answer, solution, source, prompt_tokens,
completion_tokens; scripts/sft.py renders them):
  <out-dir>/<name>-train.parquet
  <out-dir>/<name>-val.parquet     held-out problems for the eval loss

Usage:
  uv run scripts/build_sft_data.py
  uv run scripts/build_sft_data.py --target 200 --val 20 --out-dir /tmp/sft-check   # quick check
"""

import argparse
import json
import multiprocessing
import os
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # workers are forked processes

import pandas as pd
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

from posttrain.data import OVERLAP_THRESHOLD, eval_index, nemotron_to_trace, problem_key, sft_completion
from posttrain.prompts import render_prompt
from posttrain.rewards import is_correct

NEMOTRON_FILE = "datasets/nvidia/Nemotron-SFT-Math-v3/data/train.jsonl"
MAX_PROMPT_TOKENS = 1024  # eval.py's --max-prompt-tokens: longer prompts would not fit at eval time
# A slice whose first rows are all tool-use rows lies in the TIR part of the file: stop reading it.
TIR_PROBE_ROWS = 8
# Traces longer than this many characters per budget token are over budget for certain (they
# average ~3.3); lets us skip tokenizing the many 50k+ token traces.
MAX_CHARS_PER_TOKEN = 8

# Per-worker state. `_index` is built in the parent and inherited through fork; the tokenizer and
# file system are created in each worker.
_args: argparse.Namespace
_index = None
_tokenizer = None
_fs = None


def _init_worker() -> None:
    global _tokenizer, _fs
    _tokenizer = AutoTokenizer.from_pretrained(_args.tokenizer)
    _fs = HfFileSystem(skip_instance_cache=True)


def check_trace(row: dict) -> dict | str:
    """The trace to keep, with token counts, or the reason it is dropped."""
    trace = nemotron_to_trace(row)
    if isinstance(trace, str):
        return trace
    completion = sft_completion(trace["reasoning"], trace["answer"])
    if len(completion) > MAX_CHARS_PER_TOKEN * _args.max_completion_tokens:
        return "too_long"
    completion_tokens = len(_tokenizer(completion)["input_ids"]) + 1  # + <|im_end|>
    if completion_tokens > _args.max_completion_tokens:
        return "too_long"
    prompt_tokens = len(_tokenizer(render_prompt(trace["problem"], "think", _tokenizer))["input_ids"])
    if prompt_tokens > MAX_PROMPT_TOKENS:
        return "long_prompt"
    # Workers are processes, so this runs in a main thread and math_verify's timeouts work.
    if not _args.keep_unverified and is_correct(completion, trace["solution"], think=True) is not True:
        return "unverified"
    if _index.max_overlap(trace["problem"]) >= OVERLAP_THRESHOLD:
        return "eval_overlap"
    return {**trace, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def read_slice(start: int) -> tuple[list[dict], Counter, int]:
    """Check the rows in the first --slice-mb of the slice at byte offset `start`."""
    traces, drops = [], Counter()
    budget = _args.slice_mb * 2**20
    with _fs.open(NEMOTRON_FILE, "rb", block_size=8 * 2**20) as f:
        if start:
            f.seek(start)
            f.readline()  # we landed mid-row; start at the next one
        seen = 0
        while f.tell() - start < budget:
            line = f.readline()
            if not line:
                break
            result = check_trace(json.loads(line))
            seen += 1
            if isinstance(result, str):
                drops[result] += 1
                if seen == TIR_PROBE_ROWS and drops["tir"] == seen:
                    break
            else:
                traces.append(result)
        return traces, drops, f.tell() - start


def main() -> None:
    global _args, _index
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=100_000, help="training traces to keep")
    p.add_argument("--val", type=int, default=500, help="held-out problems for the eval loss")
    p.add_argument("--max-completion-tokens", type=int, default=16384, help="reasoning + answer + <|im_end|>")
    p.add_argument("--max-per-problem", type=int, default=1, help="traces kept per distinct problem")
    p.add_argument("--keep-unverified", action="store_true", help="keep traces our grader cannot confirm as correct")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B-Base", help="shared by all Qwen3.5 sizes")
    p.add_argument("--slices", type=int, default=2048, help="equal slices the file is cut into")
    p.add_argument("--slice-mb", type=int, default=24, help="MB read from the start of each slice")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    p.add_argument("--name", default="nemotron")
    p.add_argument("--seed", type=int, default=0)
    _args = p.parse_args()
    args = _args

    _index = eval_index()
    size = HfFileSystem().info(NEMOTRON_FILE)["size"]
    starts = [i * (size // args.slices) for i in range(args.slices)]
    random.Random(args.seed).shuffle(starts)
    max_gb = args.slices * args.slice_mb / 1024
    print(f"[data] {NEMOTRON_FILE}: {size / 1e9:.0f} GB; reading up to {max_gb:.0f} GB in {args.slices} slices")

    want = args.target + args.val
    kept, per_problem, drops, read_bytes, t0 = [], Counter(), Counter(), 0, time.time()
    # Ordered imap + a seeded slice order: the slices used are a prefix of that order, so a rebuild
    # gives the same set whatever the workers' timing.
    with multiprocessing.get_context("fork").Pool(args.workers, initializer=_init_worker) as pool:
        for n, (traces, slice_drops, nbytes) in enumerate(pool.imap(read_slice, starts), 1):
            drops.update(slice_drops)
            read_bytes += nbytes
            for trace in traces:
                key = problem_key(trace["problem"])
                if per_problem[key] >= args.max_per_problem:
                    drops["duplicate_problem"] += 1
                    continue
                per_problem[key] += 1
                kept.append(trace)
            if n % 20 == 0:
                print(f"[read] {n} slices, {read_bytes / 1e9:.1f} GB, kept {len(kept)}/{want}, {time.time() - t0:.0f}s", flush=True)
            if len(kept) >= want:
                break
    if len(kept) < want:
        print(f"[warn] only {len(kept)} of {want} traces after reading every slice: raise --slice-mb")

    # Hold out whole problems, so no validation problem has a second trace in the training set.
    df = pd.DataFrame(kept[:want]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    keys = df.problem.map(problem_key)
    is_val = keys.isin(set(keys.drop_duplicates()[: args.val]))
    train, val = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out_dir / f"{args.name}-train.parquet")
    val.to_parquet(args.out_dir / f"{args.name}-val.parquet")

    checked = len(kept) + sum(drops.values())
    print(f"\n[done] read {read_bytes / 1e9:.1f} GB, {checked} rows -> {len(train)} train + {len(val)} val in {args.out_dir}")
    print("dropped: " + ", ".join(f"{reason} {n} ({n / checked:.1%})" for reason, n in drops.most_common()))
    print(f"by source: {train.source.value_counts().to_dict()}")
    tokens = train.completion_tokens
    print(
        "completion tokens: "
        + "  ".join(f"p{q}={tokens.quantile(q / 100):.0f}" for q in (10, 50, 90))
        + f"  mean={tokens.mean():.0f}  total={tokens.sum() / 1e6:.0f}M"
    )
    for budget in (4096, 8192, 16384):
        if budget <= args.max_completion_tokens:
            print(f"  <= {budget}: {(tokens <= budget).sum()} traces ({(tokens <= budget).mean():.0%})")


if __name__ == "__main__":
    main()
