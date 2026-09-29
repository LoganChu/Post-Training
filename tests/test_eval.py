from posttrain.eval import grade
from posttrain.prompts import STOP_STRINGS, render_prompt


def test_mcq_grading():
    assert grade("reasoning</think>\n\n\\boxed{C}", "C", "mcq", "think")
    assert grade("reasoning</think>\n\n\\boxed{\\text{C}}", "C", "mcq", "think")
    assert not grade("reasoning</think>\n\n\\boxed{B}", "C", "mcq", "think")
    assert not grade("reasoning</think>\n\n\\boxed{B} or \\boxed{C}", "C", "mcq", "think")  # hedging
    assert not grade("\\boxed{C} reasoning never closed", "C", "mcq", "think")
    assert grade("so the answer is \\boxed{C}", "C", "mcq", "zero")


def test_math_grading_uses_reward_logic():
    assert grade("work</think>\n\n\\boxed{\\frac{1}{2}}", "0.5", "math", "think")
    assert not grade("\\boxed{0.5} never closed", "0.5", "math", "think")


def test_zero_prompt_is_plain_text():
    p = render_prompt("What is 1+1?", "zero")
    assert p.endswith("Assistant:") and "What is 1+1?" in p and "<think>" not in p
    assert "\nUser:" in STOP_STRINGS["zero"]


def test_think_prompt_opens_reasoning():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
    p = render_prompt("What is 1+1?", "think", tok)
    assert p.endswith("<|im_start|>assistant\n<think>\n")
    assert "\\boxed{}" in p
