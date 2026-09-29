"""Reward functions for math RLVR, shared by GRPO training and evaluation.

Two completion shapes, selected explicitly with `think`:
- think=True (main track, chat template with thinking on): Qwen3.5's generation prompt already
  ends with "<think>\\n", so a completion looks like "reasoning ... </think>\\n\\nanswer \\boxed{..}".
  A completion without </think> never finished reasoning (e.g. truncated) and gets no credit,
  even if a \\boxed{} appears mid-reasoning.
- think=False (RL-Zero track, plain-text prompt on the Base model): the answer is the \\boxed{} in
  the text, after the think block if the model chose to write one.
Several boxed answers parse as a set and never match (anti-hedging).

GRPOTrainer calls reward functions with `completions` (strings, or lists of one message dict for
conversational datasets), `completion_ids`, and every extra dataset column as a keyword argument;
our datasets provide the gold answer in a `solution` column.
"""

import logging
import threading

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from trl.rewards import get_soft_overlong_punishment

__all__ = [
    "answer_region",
    "is_correct",
    "correctness_reward",
    "correctness_reward_zero",
    "format_reward",
    "format_reward_zero",
    "get_soft_overlong_punishment",
]

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

# Same answer extraction as TRL's accuracy_reward: prefer \boxed{}, require a LaTeX anchor so stray
# numbers in the text are not picked up as the answer.
_ANSWER_EXTRACTION = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(units=True),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]


def _text(completion: str | list[dict]) -> str:
    return completion if isinstance(completion, str) else completion[-1]["content"]


def answer_region(text: str, think: bool) -> str | None:
    """The part of a completion that should hold the final answer, or None if it has none.

    think=True:  text after exactly one </think>; None if reasoning never closed or tags repeat.
    think=False: the whole text, or, if the model opened a think block on its own (Qwen3.5-Base
                 does this ~10% of the time), the text after a single well-formed
                 <think>...</think> pair. Unclosed/repeated tags -> None.
    """
    n_open, n_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    if think:
        if n_close != 1 or n_open != 0:
            return None
        return text.split(THINK_CLOSE, 1)[1]
    if n_open == 0 and n_close == 0:
        return text
    if n_open == 1 and n_close == 1 and text.index(THINK_OPEN) < text.index(THINK_CLOSE):
        return text.split(THINK_CLOSE, 1)[1]
    return None


def _timeouts() -> tuple[int | None, int | None]:
    # math_verify enforces timeouts with signal.alarm(), which only works in the main thread.
    if threading.current_thread() is threading.main_thread():
        return 10, 5
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    return None, None


def is_correct(text: str, gold: str, *, think: bool) -> bool | None:
    """True/False if the completion's final answer matches `gold`; None if `gold` is unparseable."""
    parse_timeout, verify_timeout = _timeouts()
    # Gold answers are usually bare LaTeX ("34", "(x+1)^2", "\\sqrt{8}"), which math_verify cannot
    # anchor on; wrapping in \boxed{} lets it parse them exactly like model answers.
    gold_parsed = parse(f"\\boxed{{{gold}}}", extraction_config=_ANSWER_EXTRACTION, parsing_timeout=parse_timeout)
    if not gold_parsed:
        return None
    region = answer_region(text, think)
    if region is None:
        return False
    answer_parsed = parse(
        region, extraction_config=_ANSWER_EXTRACTION, extraction_mode="first_match", parsing_timeout=parse_timeout
    )
    return bool(verify(gold_parsed, answer_parsed, timeout_seconds=verify_timeout))


def _correctness(completions: list, solution: list[str], think: bool) -> list[float | None]:
    results = [is_correct(_text(c), gold, think=think) for c, gold in zip(completions, solution, strict=True)]
    # None -> GRPOTrainer ignores this reward for rows whose gold answer is unparseable.
    return [None if r is None else float(r) for r in results]


def _format(completions: list, think: bool) -> list[float]:
    regions = [answer_region(_text(c), think) for c in completions]
    return [1.0 if r is not None and "\\boxed{" in r else 0.0 for r in regions]


# Named module-level functions (not closures/partials): GRPOTrainer logs each reward under the
# function's __name__, and AsyncGRPO needs reward functions to be picklable.
def correctness_reward(completions: list, solution: list[str], **kwargs) -> list[float | None]:
    """Main track: 1.0 if the answer after </think> is equivalent to the gold solution."""
    return _correctness(completions, solution, think=True)


def correctness_reward_zero(completions: list, solution: list[str], **kwargs) -> list[float | None]:
    """RL-Zero track: 1.0 if the \\boxed{} answer is equivalent to the gold solution."""
    return _correctness(completions, solution, think=False)


def format_reward(completions: list, **kwargs) -> list[float]:
    """Main track: 1.0 if reasoning is closed exactly once and a \\boxed{} answer follows."""
    return _format(completions, think=True)


def format_reward_zero(completions: list, **kwargs) -> list[float]:
    """RL-Zero track: 1.0 if there is a \\boxed{} answer and any think block is well-formed."""
    return _format(completions, think=False)
