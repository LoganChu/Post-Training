"""Evaluate a model on math benchmarks (and GPQA-Diamond for out-of-domain generalization).

Generates k samples per problem with vLLM and grades them with the same `is_correct` used as
the RL reward, so eval and training can never disagree about what counts as correct.

Metrics per benchmark:
  avg@k      mean accuracy over all k samples (the headline number; low variance)
  pass@k     fraction of problems solved by at least one sample (headroom RL can exploit)
  format     fraction of samples with a well-formed, boxed final answer
  truncated  fraction of samples that hit max_tokens (too many => raise max_tokens or the model rambles)

Usage:
  uv run -m posttrain.eval --model Qwen/Qwen3.5-0.8B-Base --mode zero
  uv run -m posttrain.eval --model outputs/sft-0.8b --mode think --benchmarks math500,aime25 --max-tokens 16384
"""

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from posttrain.prompts import STOP_STRINGS, Mode, render_prompt
from posttrain.rewards import answer_region, is_correct


@dataclass(frozen=True)
class Benchmark:
    dataset: str
    split: str
    problem_col: str
    answer_col: str
    default_k: int  # small benchmarks get more samples to reduce variance
    config: str | None = None
    kind: str = "math"  # "math" -> math_verify, "mcq" -> letter match


BENCHMARKS: dict[str, Benchmark] = {
    "math500": Benchmark("HuggingFaceH4/MATH-500", "test", "problem", "answer", default_k=4),
    "aime24": Benchmark("HuggingFaceH4/aime_2024", "train", "problem", "answer", default_k=16),
    "aime25": Benchmark("MathArena/aime_2025", "train", "problem", "answer", default_k=16),
    "aime26": Benchmark("MathArena/aime_2026", "train", "problem", "answer", default_k=16),
    "amc23": Benchmark("math-ai/amc23", "test", "question", "answer", default_k=8),
    # Gated: accept the terms on the Hub and set HF_TOKEN. Skipped with a warning otherwise.
    "gpqa": Benchmark("Idavidrein/gpqa", "train", "Question", "Correct Answer", default_k=4, config="gpqa_diamond", kind="mcq"),
}
DEFAULT_BENCHMARKS = "math500,aime24,aime25,aime26,amc23,gpqa"


def load_problems(bench: Benchmark, seed: int) -> list[dict]:
    """Rows as {"problem": str, "answer": str}. MCQ options are shuffled deterministically."""
    ds = load_dataset(bench.dataset, bench.config, split=bench.split)
    rows = []
    for i, row in enumerate(ds):
        problem, answer = str(row[bench.problem_col]), str(row[bench.answer_col])
        if bench.kind == "mcq":
            options = [answer] + [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
            random.Random(seed + i).shuffle(options)
            letters = "ABCD"
            problem += "\n\n" + "\n".join(f"{letters[j]}. {str(o).strip()}" for j, o in enumerate(options))
            # Must agree with the prompt templates' \boxed{} instruction, or models answer with a bare letter.
            problem += "\n\nGive the letter of the correct option (A, B, C, or D) as your final answer within \\boxed{}."
            answer = letters[options.index(answer)]
        rows.append({"problem": problem, "answer": answer})
    return rows


def grade(text: str, gold: str, kind: str, mode: Mode) -> bool:
    think = mode == "think"
    if kind == "mcq":
        region = answer_region(text, think)
        found = re.findall(r"\\boxed\{\s*(?:\\text\{)?\s*([A-D])\b", region or "")
        return len(set(found)) == 1 and found[0] == gold
    return bool(is_correct(text, gold, think=think))


def sampling_info(args: argparse.Namespace) -> dict:
    return {k: getattr(args, k) for k in ["temperature", "top_p", "top_k", "presence_penalty", "seed"]}


def grade_records(records: list[dict], kind: str, mode: Mode) -> None:
    """(Re)compute the `correct` and `well_formed` fields of saved samples in place."""
    for r in records:
        r["correct"] = grade(r["completion"], r["gold"], kind, mode)
        region = answer_region(r["completion"], mode == "think")
        r["well_formed"] = region is not None and "\\boxed{" in region


def summarize(records: list[dict], k: int, gen_seconds: float | None = None) -> dict:
    """Benchmark metrics from graded samples (stored as k consecutive samples per problem)."""
    n = len(records)
    per_problem = [records[i : i + k] for i in range(0, n, k)]
    return {
        "num_problems": len(per_problem),
        "k": k,
        "avg@k": sum(r["correct"] for r in records) / n,
        "pass@k": sum(any(r["correct"] for r in group) for group in per_problem) / len(per_problem),
        "format": sum(r["well_formed"] for r in records) / n,
        "truncated": sum(r["truncated"] for r in records) / n,
        "mean_tokens": sum(r["num_tokens"] for r in records) / n,
        "gen_seconds": gen_seconds,
    }


def regrade(args: argparse.Namespace) -> dict:
    """Re-grade saved samples with the current grader and rebuild summary.json; no generation."""
    out_dir = Path(args.output_dir)
    summary_path = out_dir / "summary.json"
    old = json.loads(summary_path.read_text())["benchmarks"] if summary_path.exists() else {}
    summary = {"model": args.model, "mode": args.mode, "max_tokens": args.max_tokens, "sampling": sampling_info(args), "benchmarks": {}}
    for path in sorted(out_dir.glob("*.jsonl")):
        name = path.stem
        records = [json.loads(line) for line in path.open()]
        num_problems = len(dict.fromkeys(r["problem"] for r in records))
        grade_records(records, BENCHMARKS[name].kind, args.mode)
        result = summarize(records, len(records) // num_problems, old.get(name, {}).get("gen_seconds"))
        summary["benchmarks"][name] = result
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2))
    print_table(summary)
    return summary


def evaluate(args: argparse.Namespace) -> dict:
    from vllm import LLM, SamplingParams  # imported lazily: heavy, and unit tests don't need it

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_prompt_tokens + args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        limit_mm_per_prompt={"image": 0, "video": 0},  # text-only eval; skip the vision encoder
    )
    tokenizer = llm.get_tokenizer()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge into an existing summary so a subset of benchmarks can be re-run without losing the rest.
    summary_path = out_dir / "summary.json"
    summary = {"model": args.model, "mode": args.mode, "max_tokens": args.max_tokens, "sampling": sampling_info(args), "benchmarks": {}}
    if summary_path.exists():
        summary["benchmarks"] = json.loads(summary_path.read_text())["benchmarks"]
    for name in args.benchmarks.split(","):
        bench = BENCHMARKS[name]
        try:
            problems = load_problems(bench, args.seed)
        except Exception as e:  # noqa: BLE001 - gated/unavailable datasets should not kill the run
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if args.limit:
            problems = problems[: args.limit]
        k = args.k or bench.default_k
        params = SamplingParams(
            n=k,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            stop=STOP_STRINGS[args.mode],
            seed=args.seed,
        )
        prompts = [render_prompt(p["problem"], args.mode, tokenizer) for p in problems]
        t0 = time.time()
        outputs = llm.generate(prompts, params)

        records = []
        for prob, prompt, out in zip(problems, prompts, outputs, strict=True):
            for s in out.outputs:
                records.append(
                    {
                        "problem": prob["problem"],
                        "gold": prob["answer"],
                        "prompt": prompt,
                        "completion": s.text,
                        "num_tokens": len(s.token_ids),
                        "truncated": s.finish_reason == "length",
                    }
                )
        grade_records(records, bench.kind, args.mode)
        result = summarize(records, k, gen_seconds=round(time.time() - t0, 1))
        summary["benchmarks"][name] = result
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[{name}] " + "  ".join(f"{key}={v:.3f}" if isinstance(v, float) else f"{key}={v}" for key, v in result.items()))

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print_table(summary)
    return summary


def print_table(summary: dict) -> None:
    print(f"\nmodel={summary['model']}  mode={summary['mode']}  max_tokens={summary['max_tokens']}  sampling={summary.get('sampling')}")
    print(f"{'benchmark':10s} {'k':>3s} {'avg@k':>7s} {'pass@k':>7s} {'format':>7s} {'trunc':>7s} {'tokens':>7s}")
    for name, r in summary["benchmarks"].items():
        print(
            f"{name:10s} {r['k']:3d} {r['avg@k']:7.1%} {r['pass@k']:7.1%} {r['format']:7.1%} "
            f"{r['truncated']:7.1%} {r['mean_tokens']:7.0f}"
        )


def _exit_now() -> None:
    """vLLM's engine-core subprocess sometimes never exits at interpreter shutdown, hanging the run
    after every result is already on disk. Kill it and leave instead of waiting forever."""
    import os
    import sys

    import psutil

    sys.stdout.flush()
    sys.stderr.flush()
    for child in psutil.Process().children(recursive=True):
        child.kill()
    os._exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["think", "zero"], required=True)
    p.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    p.add_argument("--k", type=int, default=None, help="samples per problem (default: per-benchmark)")
    p.add_argument("--limit", type=int, default=None, help="only the first N problems (quick checks)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    # One protocol for every model. T=0.6/top_p=0.95/top_k=20 is the usual math-eval recipe
    # (DeepSeek-R1, Qwen3 reports). presence_penalty=1.5 (Qwen3.5's thinking-mode recommendation)
    # is needed because without it Qwen3.5 models fall into repetition loops (all 28 of the
    # post-trained 2B's truncated MATH-500 samples in a 50-problem check; 42% -> 70% accuracy with
    # the penalty). Ablation on MATH-500 subsets: at T=0.6 the penalty costs the base model nothing
    # (40.3% vs 39.8%), while raising T to 1.0 drops it to 29% with or without the penalty.
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    # vLLM claims this fraction of the *whole* GPU. The desktop's own GPU use varies (3.1 GB on
    # 2026-10-01, 4.3 GB on 2026-10-03, when 0.85 no longer fit and engine startup OOMed); 0.75
    # leaves headroom and is still far more KV cache than 0.8B-2B models need at 8-32k tokens.
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="default: outputs/eval/<model>-<mode>")
    p.add_argument("--regrade", action="store_true", help="re-grade saved samples in --output-dir; no generation")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/eval/{args.model.rstrip('/').replace('/', '--')}-{args.mode}"
    if args.regrade:
        regrade(args)
    else:
        evaluate(args)
        _exit_now()


if __name__ == "__main__":
    main()
