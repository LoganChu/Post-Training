"""On-policy distillation (PLAN.md, Stage 2): the SFT student generates think-mode completions, a
larger teacher with the same vocabulary scores every token of them, and the student is trained to
match the teacher's next-token distributions (reverse KL at beta=1.0). No answers or rewards are
used, only prompts.

All hyperparameters live in a YAML config (TrlParser format: script options + DistillationConfig
fields, which include the teacher). Prompts are the think-mode prompts eval and RL use.

Usage:
  uv run scripts/distill.py --config configs/distill_0.8b.yaml
  uv run scripts/distill.py --config configs/distill_0.8b.yaml --max_steps 5 --output_dir outputs/smoke-distill   # smoke test
"""

from dataclasses import dataclass, field

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DistillationConfig, DistillationTrainer, TrlParser

# Qwen3.5 workarounds shared with the other stages. Importing grpo also selects vLLM's V1 model
# runner, which has to happen before vLLM reads its config.
from grpo import text_only_vllm
from sft import CHAT_EOS, copy_processor_files

from posttrain.data import to_grpo_dataset


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "student: the stage-1 SFT checkpoint"})
    train_data: str = field(default="data/rl/pool.parquet", metadata={"help": "decontaminated prompt pool (posttrain.data build)"})
    source: str | None = field(default="deepmath", metadata={"help": "keep only this `source` of the pool; None keeps all"})
    max_prompt_tokens: int = field(
        default=1024, metadata={"help": "drop longer prompts; vllm_max_model_length must cover this + max_completion_length"}
    )


def trim_padding(inputs: dict) -> dict:
    """Cut a micro-batch back to its own longest prompt (left-padded) and completion (right-padded)."""
    prompt_len = max(int(inputs["prompt_mask"].sum(dim=1).max()), 1)
    completion_len = max(int(inputs["completion_mask"].sum(dim=1).max()), 1)
    return {
        **inputs,
        "prompt_ids": inputs["prompt_ids"][:, -prompt_len:],
        "prompt_mask": inputs["prompt_mask"][:, -prompt_len:],
        "completion_ids": inputs["completion_ids"][:, :completion_len],
        "completion_mask": inputs["completion_mask"][:, :completion_len],
    }


class TrimmedDistillationTrainer(DistillationTrainer):
    """Two changes to how TRL 1.14 runs the student and teacher backbones for the loss.

    1. No `logits_to_keep`. TRL passes it to the backbone whenever the LM-head model's forward accepts
       it. Qwen3.5's backbone does not declare it and hands unknown keywords down to every layer's
       DeltaNet and attention kernels. TRL slices out the completion positions itself afterwards, so
       nothing is lost (GRPO's chunked path, which works here, never passes it either).
    2. No padding. TRL pads the whole generation batch to its longest prompt and completion before
       splitting it into micro-batches, so every forward pass would run at the length of the longest
       completion of the rollout (often the full budget). Each micro-batch is cut back to its own
       longest row; at batch size 1 that is no padding at all, as in SFT.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_kwarg_keys = set(self.model_kwarg_keys) - {"logits_to_keep"}

    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        return super()._compute_loss(unwrapped_student, trim_padding(inputs), num_items_in_batch)


def main() -> None:
    text_only_vllm()
    parser = TrlParser((ScriptArguments, DistillationConfig))
    script_args, training_args = parser.parse_args_and_config()
    assert training_args.teacher_model_name_or_path, "set teacher_model_name_or_path in the config"

    # Qwen3.5 checkpoints are multimodal: left to itself, TRL loads Qwen3VLProcessor and takes its
    # vision-model code paths. We train text-only, so pass the plain tokenizer (TRL's padding settings).
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, padding_side="left", truncation_side="left"
    )
    # Rollouts end at the tokenizer's EOS. SFT checkpoints save <|im_end|> as EOS; with a Base
    # tokenizer (<|endoftext|>) every rollout would run to the length limit.
    assert tokenizer.eos_token == CHAT_EOS, f"student EOS is {tokenizer.eos_token!r}: not an SFT checkpoint?"

    pool = pd.read_parquet(script_args.train_data)
    if script_args.source is not None:
        pool = pool[pool.source == script_args.source]
    train = to_grpo_dataset(Dataset.from_pandas(pool, preserve_index=False), "think")

    # A few DeepMath prompts are very long (up to ~1.8k tokens); prompt + max_completion_length must
    # fit vLLM's context window. Token counts come from the same rendering TRL uses for the rollouts.
    def within_prompt_budget(batch):
        rendered = tokenizer.apply_chat_template(
            batch["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True,
            **(training_args.chat_template_kwargs or {}),
        )
        return [len(ids) <= script_args.max_prompt_tokens for ids in rendered["input_ids"]]

    n = len(train)
    train = train.filter(within_prompt_budget, batched=True)
    print(
        f"[data] {len(train)} prompts from {script_args.train_data} (source={script_args.source}; "
        f"{n - len(train)} over {script_args.max_prompt_tokens} prompt tokens dropped)"
    )

    trainer = TrimmedDistillationTrainer(
        model=script_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    copy_processor_files(script_args.model_name_or_path, training_args.output_dir)
    if torch.cuda.is_available():
        print(f"[mem] peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
