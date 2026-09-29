"""Prompt templates shared by data building, GRPO training and evaluation.

Two prompt modes, matching the two reward/grading modes in `rewards.py`:
- "think": Qwen chat template with thinking enabled. The rendered prompt ends with
  "<|im_start|>assistant\\n<think>\\n", so the model continues inside the reasoning block.
- "zero": plain-text R1-Zero-style template for RL directly on a Base model (no chat template,
  no think tags). Generation must stop before the model invents the next "User:" turn.
"""

from typing import Literal

Mode = Literal["think", "zero"]

MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

ZERO_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a math question, and the Assistant solves it. "
    "The Assistant first reasons through the problem step by step, then gives the final answer within \\boxed{{}}.\n"
    "User: {problem}\n"
    "Assistant:"
)

# Chat turns end with <|im_end|>; Base tokenizers use <|endoftext|> as EOS, so pass both as stops.
STOP_STRINGS: dict[Mode, list[str]] = {
    "think": ["<|im_end|>", "<|endoftext|>"],
    "zero": ["\nUser:", "<|endoftext|>", "<|im_end|>"],
}


def math_user_message(problem: str) -> str:
    return f"{problem.strip()}\n\n{MATH_INSTRUCTION}"


def conversational_prompt(problem: str) -> list[dict[str, str]]:
    """Prompt in TRL's conversational format (TRL applies the chat template itself)."""
    return [{"role": "user", "content": math_user_message(problem)}]


def render_prompt(problem: str, mode: Mode, tokenizer=None) -> str:
    """Fully rendered prompt string, for direct generation with vLLM."""
    if mode == "zero":
        return ZERO_TEMPLATE.format(problem=problem.strip())
    return tokenizer.apply_chat_template(
        conversational_prompt(problem), tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
