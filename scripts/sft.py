"""SFT cold-start (PLAN.md, Stage 1): teach a Base model the think format, when to stop, and
reasoning habits, from the teacher traces built by scripts/build_sft_data.py.

All hyperparameters live in a YAML config (TrlParser format: script options + SFTConfig fields).
The loss covers only the completion (reasoning, </think>, answer, <|im_end|>); the prompt is the
same think-mode prompt eval and RL use.

Usage:
  uv run scripts/sft.py --config configs/sft_0.8b.yaml
  uv run scripts/sft.py --config configs/sft_0.8b.yaml --max_steps 20 --eval_strategy no --output_dir outputs/smoke-sft   # smoke test
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer, TrlParser

from posttrain.data import to_sft_dataset

# Chat turns end with <|im_end|>, but the Base tokenizer's EOS is <|endoftext|>. SFTTrainer appends
# the EOS token to every completion, so with the Base default the model would never learn to stop.
CHAT_EOS = "<|im_end|>"

# Checkpoints keep the multimodal layout but not these files, which vLLM reads for this architecture.
PROCESSOR_FILES = ["preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "HF id or local checkpoint"})
    train_data: str = field(metadata={"help": "<name>-train.parquet from scripts/build_sft_data.py"})
    eval_data: str | None = field(default=None, metadata={"help": "<name>-val.parquet, for the eval loss"})
    max_completion_tokens: int = field(
        default=8192, metadata={"help": "train only on traces whose completion fits this budget (the eval/RL budget)"}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "random subset of the training traces"})


def load_traces(path: str, max_completion_tokens: int, tokenizer, max_samples: int | None = None, seed: int = 0) -> Dataset:
    df = pd.read_parquet(path)
    keep = df[df.completion_tokens <= max_completion_tokens]
    if max_samples is not None and max_samples < len(keep):
        keep = keep.sample(n=max_samples, random_state=seed)
    print(
        f"[data] {path}: {len(keep)} of {len(df)} traces within {max_completion_tokens} completion tokens "
        f"(mean {keep.completion_tokens.mean():.0f}, total {keep.completion_tokens.sum() / 1e6:.1f}M)"
    )
    return to_sft_dataset(Dataset.from_pandas(keep, preserve_index=False), tokenizer)


def copy_processor_files(model_name_or_path: str, output_dir: str) -> None:
    """Make the saved checkpoint loadable by vLLM (eval.py, GRPO) without further steps."""
    for name in PROCESSOR_FILES:
        local = Path(model_name_or_path) / name
        shutil.copy(local if local.exists() else hf_hub_download(model_name_or_path, name), output_dir)


def main() -> None:
    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()
    if training_args.eos_token is None:
        training_args.eos_token = CHAT_EOS

    # Qwen3.5 checkpoints are multimodal: left to itself, TRL loads Qwen3VLProcessor. We train
    # text-only, so pass the plain tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    train = load_traces(
        script_args.train_data, script_args.max_completion_tokens, tokenizer, script_args.max_train_samples, training_args.seed
    )
    eval_ds = None
    if script_args.eval_data and training_args.eval_strategy != "no":
        eval_ds = load_traces(script_args.eval_data, script_args.max_completion_tokens, tokenizer)

    trainer = SFTTrainer(
        model=script_args.model_name_or_path,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    copy_processor_files(script_args.model_name_or_path, training_args.output_dir)
    if torch.cuda.is_available():
        print(f"[mem] peak VRAM allocated: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
