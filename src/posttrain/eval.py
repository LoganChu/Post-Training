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
            problem += "\n\nAnswer with the letter of the correct option."
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

    summary = {"model": args.model, "mode": args.mode, "max_tokens": args.max_tokens, "benchmarks": {}}
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
            max_tokens=args.max_tokens,
            stop=STOP_STRINGS[args.mode],
            seed=args.seed,
        )
        prompts = [render_prompt(p["problem"], args.mode, tokenizer) for p in problems]
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        gen_time = time.time() - t0

        records, per_problem = [], []
        for prob, prompt, out in zip(problems, prompts, outputs, strict=True):
            correct = []
            for s in out.outputs:
                ok = grade(s.text, prob["answer"], bench.kind, args.mode)
                correct.append(ok)
                records.append(
                    {
                        "problem": prob["problem"],
                        "gold": prob["answer"],
                        "prompt": prompt,
                        "completion": s.text,
                        "correct": ok,
                        "num_tokens": len(s.token_ids),
                        "truncated": s.finish_reason == "length",
                        "well_formed": (r := answer_region(s.text, args.mode == "think")) is not None and "\\boxed{" in r,
                    }
                )
            per_problem.append(correct)

        n = len(records)
        result = {
            "num_problems": len(problems),
            "k": k,
            "avg@k": sum(r["correct"] for r in records) / n,
            "pass@k": sum(any(c) for c in per_problem) / len(per_problem),
            "format": sum(r["well_formed"] for r in records) / n,
            "truncated": sum(r["truncated"] for r in records) / n,
            "mean_tokens": sum(r["num_tokens"] for r in records) / n,
            "gen_seconds": round(gen_time, 1),
        }
        summary["benchmarks"][name] = result
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[{name}] " + "  ".join(f"{key}={v:.3f}" if isinstance(v, float) else f"{key}={v}" for key, v in result.items()))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_table(summary)
    return summary


def print_table(summary: dict) -> None:
    print(f"\nmodel={summary['model']}  mode={summary['mode']}  max_tokens={summary['max_tokens']}")
    print(f"{'benchmark':10s} {'k':>3s} {'avg@k':>7s} {'pass@k':>7s} {'format':>7s} {'trunc':>7s} {'tokens':>7s}")
    for name, r in summary["benchmarks"].items():
        print(
            f"{name:10s} {r['k']:3d} {r['avg@k']:7.1%} {r['pass@k']:7.1%} {r['format']:7.1%} "
            f"{r['truncated']:7.1%} {r['mean_tokens']:7.0f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=["think", "zero"], required=True)
    p.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    p.add_argument("--k", type=int, default=None, help="samples per problem (default: per-benchmark)")
    p.add_argument("--limit", type=int, default=None, help="only the first N problems (quick checks)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    # Common math-eval sampling (DeepSeek-R1 / Qwen3 reports): T=0.6, top_p=0.95, top_k=20.
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="default: outputs/eval/<model>-<mode>")
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"outputs/eval/{args.model.rstrip('/').replace('/', '--')}-{args.mode}"
    evaluate(args)


if __name__ == "__main__":
    main()
