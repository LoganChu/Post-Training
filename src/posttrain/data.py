"""Training datasets: loading, cleaning, decontamination, and conversion to TRL's formats.

RL prompts are stored mode-neutral as {"problem", "solution", "source", "id"}; `to_grpo_dataset`
renders the prompt for a given mode ("zero" or "think") at training time, so both tracks share one
file. SFT traces (built by scripts/build_sft_data.py) add the teacher's {"reasoning", "answer"};
`to_sft_dataset` renders them as prompt/completion pairs with the same think-mode prompt as eval.

Usage (build the cleaned RL pool once):
  uv run -m posttrain.data build --out data/rl/pool.parquet
"""

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from posttrain.eval import BENCHMARKS, load_problems
from posttrain.prompts import ZERO_TEMPLATE, Mode, conversational_prompt, render_prompt
from posttrain.rewards import THINK_CLOSE, THINK_OPEN, is_correct

# DAPO-Math-17k wraps every problem in a fixed instruction asking for "Answer: $Answer"; strip it so
# our own \boxed{} prompt is the only answer-format instruction the model sees.
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def load_dapo() -> Dataset:
    """DAPO-Math-17k: ~1.79M rows on the Hub, each problem repeated ~100x; deduplicated by index."""
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    seen, rows, bad = set(), [], 0
    for r in ds:
        idx = r["extra_info"]["index"]
        if idx in seen:
            continue
        seen.add(idx)
        text = r["prompt"][0]["content"]
        if not (text.startswith(DAPO_PREFIX) and text.endswith(DAPO_SUFFIX)):
            bad += 1
            continue
        problem = text[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)].strip()
        rows.append({"problem": problem, "solution": r["reward_model"]["ground_truth"], "source": "dapo", "id": f"dapo-{idx}"})
    print(f"[dapo] {len(ds)} rows -> {len(rows)} unique problems ({bad} with unexpected wrapper, dropped)")
    return Dataset.from_list(rows)


def load_deepmath() -> Dataset:
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    rows = [
        {"problem": r["question"].strip(), "solution": r["final_answer"].strip(), "source": "deepmath", "id": f"deepmath-{i}"}
        for i, r in enumerate(ds)
    ]
    print(f"[deepmath] {len(rows)} problems")
    return Dataset.from_list(rows)


def gold_is_gradable(solution: str) -> bool:
    """Keep only rows whose answer our grader can match against itself. Otherwise the reward is
    None for every completion and the row silently contributes nothing to training."""
    return is_correct(f"\\boxed{{{solution}}}", solution, think=False) is True


BINARY_ANSWERS = {"yes", "no", "true", "false"}


def is_binary_answer(solution: str) -> bool:
    """Yes/no and true/false answers. Dropped from the RL pool: they are ~18% of DeepMath (15.6% of the pool) and
    skewed ~4:1 towards "yes", so always answering "Yes" would earn reward without solving anything."""
    text = re.sub(r"\\text\{(.*)\}", r"\1", solution.strip())
    return text.strip().lower().rstrip(".") in BINARY_ANSWERS


# --- decontamination ---
# A training problem is dropped if it contains >= 30% of some eval problem's word 13-grams.
# Any single shared 13-gram is far too aggressive: AIME's boilerplate ("m/n where m and n are
# relatively prime positive integers. Find m+n") alone flagged 902 problems, almost all unrelated.
# Inspected on the 2026-10-01 pool: below 0.3 matches are boilerplate; 0.3-0.4 are mostly MATH
# template siblings (same stem, different numbers/question); >= 0.4 are copies. 0.3 errs toward
# removing siblings too, which costs ~100 of 120k prompts.
NGRAM = 13
OVERLAP_THRESHOLD = 0.3


def _ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class EvalIndex:
    """Maps each eval-problem 13-gram to the eval problems containing it."""

    def __init__(self, problems: Iterable[str]):
        self.sizes: list[int] = []
        self.index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for j, text in enumerate(problems):
            grams = _ngrams(text)
            self.sizes.append(len(grams))
            for g in grams:
                self.index[g].append(j)

    def max_overlap(self, text: str) -> float:
        """Largest fraction of any single eval problem's 13-grams that appear in `text`."""
        hits = Counter(j for g in _ngrams(text) for j in self.index.get(g, ()))
        return max((n / self.sizes[j] for j, n in hits.items()), default=0.0)


def eval_index(benchmarks: Iterable[str] = BENCHMARKS) -> EvalIndex:
    # GPQA problems include our appended options; the question text alone is what matters
    texts = [p["problem"].split("\n\nA. ")[0] for name in benchmarks for p in load_problems(BENCHMARKS[name], seed=0)]
    return EvalIndex(texts)


def decontaminate(ds: Dataset, index: EvalIndex, threshold: float = OVERLAP_THRESHOLD) -> Dataset:
    keep = ds.filter(lambda r: index.max_overlap(r["problem"]) < threshold, num_proc=8)
    print(f"[decontam] {len(ds)} -> {len(keep)} ({len(ds) - len(keep)} overlapping eval problems removed)")
    return keep


def build_pool(out: Path) -> Dataset:
    ds = concatenate_datasets([load_dapo(), load_deepmath()])
    n = len(ds)
    ds = ds.filter(lambda r: gold_is_gradable(r["solution"]), num_proc=8)
    print(f"[gradable] {n} -> {len(ds)} ({n - len(ds)} answers our grader cannot verify, dropped)")
    n = len(ds)
    ds = ds.filter(lambda r: not is_binary_answer(r["solution"]), num_proc=8)
    print(f"[binary] {n} -> {len(ds)} ({n - len(ds)} yes/no/true/false answers dropped)")
    ds = decontaminate(ds, eval_index())
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out)
    print(f"[done] {len(ds)} rows -> {out}  by source: {dict(Counter(ds['source']))}")
    return ds


def to_grpo_dataset(ds: Dataset, mode: Mode) -> Dataset:
    """TRL GRPO format: a `prompt` column (plain string for zero mode, chat messages for think
    mode, which TRL renders with the chat template) plus `solution` for the reward functions."""

    def fmt(r):
        prompt = ZERO_TEMPLATE.format(problem=r["problem"]) if mode == "zero" else conversational_prompt(r["problem"])
        return {"prompt": prompt}

    return ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in ("solution", "source", "id")])


# --- SFT traces (nvidia/Nemotron-SFT-Math-v3) ---
NEMOTRON_NO_TIR = "without Python TIR"  # `tool_usage` of plain chain-of-thought rows (no Python tool calls)


def problem_key(problem: str) -> str:
    """Whitespace- and case-insensitive key for counting traces per problem."""
    return " ".join(problem.lower().split())


def nemotron_to_trace(row: dict) -> dict | str:
    """One Nemotron row as a mode-neutral trace, or the reason (a short string) it is unusable.

    Nemotron keeps the teacher's reasoning in the assistant message's `reasoning_content` and the
    final write-up in `content`. Its user turn carries its own instruction ("Solve the following
    math problem. Make sure to put the answer (and only answer) inside \\boxed{}."), so we take the
    bare `problem` field and add our own prompt at training time.
    """
    if row["tool_usage"] != NEMOTRON_NO_TIR:
        return "tir"
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        return "not_single_turn"
    reasoning = (messages[1].get("reasoning_content") or "").strip()
    answer = (messages[1].get("content") or "").strip()
    if not reasoning or not answer:
        return "empty"
    # The rewards give credit only for exactly one </think> (ours) and no <think> in the completion.
    if any(tag in text for tag in (THINK_OPEN, THINK_CLOSE) for text in (reasoning, answer)):
        return "think_tags"
    if "\\boxed{" not in answer:
        return "no_boxed"
    return {
        "id": f"nemotron-{row['uuid']}",
        "problem": row["problem"].strip(),
        "reasoning": reasoning,
        "answer": answer,
        "solution": row["expected_answer"].strip(),
        "source": row["data_source"],
    }


def sft_completion(reasoning: str, answer: str) -> str:
    """The text the model should generate after the think-mode prompt (which ends in "<think>\\n").

    Identical to what Qwen3.5's chat template renders for an assistant turn with `reasoning_content`,
    up to but excluding the closing <|im_end|>: SFTTrainer appends the EOS token itself.
    """
    return f"{reasoning.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"


def to_sft_dataset(ds: Dataset, tokenizer) -> Dataset:
    """TRL SFT prompt-completion format: loss on the completion only, and the prompt is rendered by
    the same `render_prompt` eval and RL use, so the model trains on exactly the prompt it is later
    run with."""

    def fmt(r):
        return {
            "prompt": render_prompt(r["problem"], "think", tokenizer),
            "completion": sft_completion(r["reasoning"], r["answer"]),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download, clean, decontaminate -> parquet")
    b.add_argument("--out", type=Path, default=Path("data/rl/pool.parquet"))
    args = p.parse_args()
    if args.cmd == "build":
        build_pool(args.out)


if __name__ == "__main__":
    main()
