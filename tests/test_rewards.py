import pytest

from posttrain.rewards import (
    correctness_reward,
    correctness_reward_zero,
    format_reward,
    format_reward_zero,
    get_soft_overlong_punishment,
    is_correct,
)


def think(reasoning: str, answer: str) -> str:
    """A main-track completion: the prompt already ends with '<think>\\n'."""
    return f"{reasoning}\n</think>\n\n{answer}"


# --- equivalence: math_verify should accept different surface forms of the same answer ---
@pytest.mark.parametrize(
    "answer, gold",
    [
        (r"\boxed{34}", "34"),
        (r"\boxed{\frac{1}{2}}", "0.5"),
        (r"\boxed{0.5}", r"\frac{1}{2}"),
        (r"\boxed{\dfrac{3}{4}}", r"\frac{3}{4}"),
        (r"\boxed{2\sqrt{2}}", r"\sqrt{8}"),
        (r"\boxed{x^2 + 2x + 1}", "(x+1)^2"),
        (r"\boxed{1,000}", "1000"),
    ],
)
def test_equivalent_forms_are_correct(answer, gold):
    assert is_correct(think("work", f"The answer is {answer}."), gold, think=True) is True


@pytest.mark.parametrize("answer, gold", [(r"\boxed{35}", "34"), (r"\boxed{\frac{1}{3}}", "0.5")])
def test_wrong_answers(answer, gold):
    assert is_correct(think("work", answer), gold, think=True) is False


def test_no_boxed_answer_is_wrong():
    # try_extract_without_anchor=False: a bare number in prose is not taken as the answer
    assert is_correct(think("work", "so it is 34"), "34", think=True) is False


def test_unparseable_gold_returns_none():
    assert correctness_reward([think("w", r"\boxed{1}")], solution=[""]) == [None]


# --- main track (think=True): reasoning must be closed exactly once ---
def test_unclosed_reasoning_gets_no_credit():
    # truncated generation with a boxed guess mid-reasoning must not be rewarded
    truncated = r"Let me try \boxed{34} ... hmm, let me check again"
    assert correctness_reward([truncated], solution=["34"]) == [0.0]
    assert format_reward([truncated]) == [0.0]


def test_boxed_only_inside_reasoning_is_wrong():
    c = think(r"maybe \boxed{34}", "I am not sure.")
    assert correctness_reward([c], solution=["34"]) == [0.0]
    assert format_reward([c]) == [0.0]


def test_repeated_think_blocks_rejected():
    c = think("a", "b") + "\n<think>\nmore\n</think>\n" + r"\boxed{34}"
    assert correctness_reward([c], solution=["34"]) == [0.0]
    assert format_reward([c]) == [0.0]


def test_hedging_with_multiple_boxed_answers_is_wrong():
    # math_verify merges several boxes into a set {34, 35}, which never equals the gold answer
    c = think("work", r"\boxed{34}, or maybe \boxed{35}")
    assert correctness_reward([c], solution=["34"]) == [0.0]


# --- RL-Zero track (think=False) ---
def test_zero_track_plain_text():
    assert correctness_reward_zero([r"Step 1... so \boxed{34}"], solution=["34"]) == [1.0]
    assert format_reward_zero([r"so \boxed{34}"]) == [1.0]
    assert format_reward_zero(["so 34"]) == [0.0]


def test_zero_track_accepts_self_opened_think_block():
    # Qwen3.5-Base sometimes opens a think block on its own; the answer follows it
    c = "<think>\n" + think("w", r"\boxed{34}")
    assert correctness_reward_zero([c], solution=["34"]) == [1.0]
    assert format_reward_zero([c]) == [1.0]


def test_zero_track_rejects_malformed_think_tags():
    unclosed = r"<think> maybe \boxed{34}"
    close_without_open = think("w", r"\boxed{34}")
    reversed_tags = r"</think> a <think> \boxed{34}"
    for c in [unclosed, close_without_open, reversed_tags]:
        assert correctness_reward_zero([c], solution=["34"]) == [0.0]


# --- trainer-facing shapes ---
def test_conversational_completions():
    completions = [[{"role": "assistant", "content": think("w", r"\boxed{34}")}]]
    assert correctness_reward(completions, solution=["34"]) == [1.0]
    assert format_reward(completions) == [1.0]


def test_batch_and_extra_kwargs():
    completions = [think("w", r"\boxed{34}"), think("w", r"\boxed{7}")]
    rewards = correctness_reward(completions, solution=["34", "34"], prompts=["p", "p"], completion_ids=[[1], [2]])
    assert rewards == [1.0, 0.0]


def test_soft_overlong_punishment_reexport():
    punish = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
    assert punish(completion_ids=[[0] * 50, [0] * 90, [0] * 101]) == [0.0, -0.5, -1.0]
