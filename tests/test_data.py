from datasets import Dataset
from transformers import AutoTokenizer

from posttrain.data import (
    DAPO_PREFIX,
    DAPO_SUFFIX,
    EvalIndex,
    decontaminate,
    gold_is_gradable,
    is_binary_answer,
    nemotron_to_trace,
    sft_completion,
    to_grpo_dataset,
    to_sft_dataset,
)
from posttrain.prompts import math_user_message
from posttrain.rewards import is_correct


def test_dapo_wrapper_constants_round_trip():
    problem = "Compute $1+1$."
    wrapped = DAPO_PREFIX + problem + DAPO_SUFFIX
    assert wrapped[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)] == problem


def test_gold_gradability():
    for ok in ["34", "-3", "\\frac{1}{2}", "2\\sqrt{2}", "(x+1)^2"]:
        assert gold_is_gradable(ok), ok
    assert not gold_is_gradable("")


def test_binary_answers():
    for a in ["Yes", "no", "\\text{True}", "False.", " yes "]:
        assert is_binary_answer(a), a
    for a in ["34", "\\frac{1}{2}", "Yesterday", "p < 1", "\\text{No solution}"]:
        assert not is_binary_answer(a), a


BOILERPLATE = "can be written as m n where m and n are relatively prime positive integers find m n"


def test_decontaminate_drops_copies_not_boilerplate():
    eval_problem = (
        "When rolling a certain unfair six sided die with faces numbered 1 2 3 4 5 and 6 the probability "
        "of obtaining face F is greater than one sixth and the probability of the opposite face is less. " + BOILERPLATE
    )
    index = EvalIndex([eval_problem])
    ds = Dataset.from_list(
        [
            {"problem": "Restated: " + eval_problem.replace("six sided", "six-sided"), "solution": "copy"},
            {"problem": "A circle is inscribed in a right triangle with legs 6 and 8. Its area " + BOILERPLATE, "solution": "boilerplate"},
            {"problem": "Unrelated problem about counting lattice points inside a circle of radius ten", "solution": "unrelated"},
        ]
    )
    assert decontaminate(ds, index)["solution"] == ["boilerplate", "unrelated"]


def test_grpo_formats():
    ds = Dataset.from_list([{"problem": "What is 1+1?", "solution": "2", "source": "t", "id": "t-0"}])
    zero = to_grpo_dataset(ds, "zero")[0]
    assert isinstance(zero["prompt"], str) and zero["prompt"].endswith("Assistant:") and zero["solution"] == "2"
    assert "problem" not in zero
    think = to_grpo_dataset(ds, "think")[0]
    assert think["prompt"][0]["role"] == "user" and "\\boxed{}" in think["prompt"][0]["content"]


def nemotron_row(**overrides):
    row = {
        "uuid": "u1",
        "problem": " What is 1+1? ",
        "expected_answer": "2",
        "data_source": "AoPS",
        "tool_usage": "without Python TIR",
        "messages": [
            {"role": "user", "content": "Solve the following math problem.\n\nWhat is 1+1?"},
            {"role": "assistant", "reasoning_content": "\nOne plus one is two.\n", "content": "The sum is \\boxed{2}.\n"},
        ],
    }
    return {**row, **overrides}


def test_nemotron_trace_and_drop_reasons():
    trace = nemotron_to_trace(nemotron_row())
    assert trace == {
        "id": "nemotron-u1",
        "problem": "What is 1+1?",
        "reasoning": "One plus one is two.",
        "answer": "The sum is \\boxed{2}.",
        "solution": "2",
        "source": "AoPS",
    }
    assert nemotron_to_trace(nemotron_row(tool_usage="with Python TIR")) == "tir"

    def with_assistant(**fields):
        row = nemotron_row()
        return {**row, "messages": [row["messages"][0], {**row["messages"][1], **fields}]}

    assert nemotron_to_trace(nemotron_row(messages=nemotron_row()["messages"] * 2)) == "not_single_turn"
    assert nemotron_to_trace(with_assistant(reasoning_content=None)) == "empty"
    assert nemotron_to_trace(with_assistant(reasoning_content="done </think> more")) == "think_tags"
    assert nemotron_to_trace(with_assistant(content="The sum is 2.")) == "no_boxed"


def test_sft_format_matches_chat_template_and_grader():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    trace = nemotron_to_trace(nemotron_row())
    ex = to_sft_dataset(Dataset.from_list([trace]), tokenizer)[0]
    assert set(ex) == {"prompt", "completion"}
    # prompt + completion + EOS is exactly the chat template's rendering of the conversation
    messages = [
        {"role": "user", "content": math_user_message(trace["problem"])},
        {"role": "assistant", "reasoning_content": trace["reasoning"], "content": trace["answer"]},
    ]
    assert ex["prompt"] + ex["completion"] + "<|im_end|>\n" == tokenizer.apply_chat_template(messages, tokenize=False)
    # the prompt's tokens are a prefix of the full sequence's, so the completion mask lines up
    prompt_ids = tokenizer(ex["prompt"])["input_ids"]
    assert tokenizer(ex["prompt"] + ex["completion"])["input_ids"][: len(prompt_ids)] == prompt_ids
    # and the completion is what the think-mode reward accepts
    assert is_correct(ex["completion"], trace["solution"], think=True)
    assert ex["completion"] == sft_completion(trace["reasoning"], trace["answer"])
