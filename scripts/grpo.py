"""GRPO training for both tracks: RL-Zero (Base model, plain-text "zero" prompt) and the main
track's RL stage (post-SFT/distillation checkpoint, chat template with thinking, "think" prompt).

All hyperparameters live in a YAML config (TrlParser format: script options + GRPOConfig fields).
Reward functions and the prompt format are picked from `mode`, so the two tracks share one script.

Usage:
  uv run scripts/grpo.py --config configs/grpo_rlzero_0.8b.yaml
  uv run scripts/grpo.py --config configs/grpo_rlzero_0.8b.yaml --max_steps 20 --output_dir outputs/smoke-grpo   # smoke test
"""

import functools
import os
from dataclasses import dataclass, field

# vLLM 0.30's default Model Runner V2 crashes in its startup profiling pass on Qwen3.5's DeltaNet
# layers when run in-process the way TRL colocates it ("'NoneType' object has no attribute 'size'"
# in qwen3_next.forward). The V1 runner works. Must be set before vLLM reads its config.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# Reuse freed GPU memory instead of stranding it (distillation OOMed on 3.5 GiB of fragmented
# reserve, 2026-10-03). vLLM's sleep-mode pool switches this off around its own allocations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
import trl.generation.vllm_generation as trl_vllm
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, TrlParser

from accelerate.utils import is_peft_model
from peft import LoraConfig
from peft.tuners.lora import LoraLayer

from posttrain.data import to_grpo_dataset
from posttrain.prompts import STOP_STRINGS
from posttrain.rewards import (
    correctness_reward,
    correctness_reward_zero,
    format_reward,
    format_reward_zero,
    get_soft_overlong_punishment,
)


# Every linear layer of the Qwen3.5 language model (attention, Gated DeltaNet, MLP); not the vision
# tower, not lm_head (tied to the embeddings). LoRA on all layers incl. the MLPs is what lets it match
# full fine-tuning in RL ("LoRA Without Regret", Thinking Machines, 2025).
LORA_TARGETS = (
    r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj"
    r"|gate_proj|up_proj|down_proj)$"
)


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "HF id or local checkpoint"})
    mode: str = field(metadata={"help": "'zero' (RL-Zero on Base) or 'think' (main track)"})
    train_data: str = field(metadata={"help": "parquet from scripts/filter_by_passrate.py"})
    soft_punish_cache: int = field(
        default=1024, metadata={"help": "DAPO overlong penalty ramps over the last N tokens of max_completion_length"}
    )
    format_reward_weight: float = field(default=0.1, metadata={"help": "small: format must not outweigh correctness"})
    use_lora: bool = field(default=False, metadata={"help": "train LoRA adapters on a frozen base (memory for >= 2B)"})
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=32)
    lora_target_modules: str = field(
        default=LORA_TARGETS, metadata={"help": "regex over module names: all language-model linear layers"}
    )


def text_only_vllm() -> None:
    """Make TRL's colocated vLLM engine text-only.

    TRL builds `vllm.LLM(...)` without `limit_mm_per_prompt`, so for Qwen3.5's multimodal checkpoint
    vLLM profiles the vision encoder with dummy max-size images/videos at startup. That used up the
    whole `vllm_gpu_memory_utilization` budget ("No available memory for the cache blocks"). We never
    send images, so disable them, as eval.py does.
    """
    trl_vllm.LLM = functools.partial(trl_vllm.LLM, limit_mm_per_prompt={"image": 0, "video": 0})


class RowBatchedGRPOTrainer(GRPOTrainer):
    """Bound the rows per forward pass when scoring rollouts on TRL's chunked (Liger) log-prob path.

    After generation TRL scores the whole rollout batch (512 completions) in one call, passing
    `batch_size` to cap rows per forward. The full-logits path honors it, but the chunked path drops it
    and runs the backbone on all rows at once (8.7 GiB in one RMSNorm, OOM). Split such calls into
    `batch_size`-row slices. Text-only: per-row multimodal inputs are not supported here.
    """

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **kwargs):
        if not self.use_liger_kernel or batch_size is None or input_ids.size(0) <= batch_size:
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size, **kwargs
            )
        assert all(v is None for k, v in kwargs.items() if k not in ("compute_entropy", "compute_aux_loss")), (
            "row batching is text-only"
        )
        parts = [
            super(RowBatchedGRPOTrainer, self)._get_per_token_logps_and_entropies(
                model, input_ids[i : i + batch_size], attention_mask[i : i + batch_size], logits_to_keep, **kwargs
            )
            for i in range(0, input_ids.size(0), batch_size)
        ]
        logps = torch.cat([p[0] for p in parts])
        entropies = None if parts[0][1] is None else torch.cat([p[1] for p in parts])
        aux = None if parts[0][2] is None else sum(p[2] for p in parts) / len(parts)
        return logps, entropies, aux


def lora_safe_weight_sync() -> None:
    """Sync LoRA weights to vLLM without modifying the base weights.

    TRL merges the adapters into the base weights in place, pushes them, and unmerges (`merge_adapter`
    / `unmerge_adapter`). With a frozen bf16 base, every merge and unmerge rounds: the base weights
    drift a little on every sync (once per rollout, 100 times per run). Instead, compute W + delta-W per layer in
    fp32, round once for vLLM, and leave the base untouched. Single-GPU only; otherwise TRL's path.
    """
    original = trl_vllm.VLLMGeneration._iter_named_params

    def _iter_named_params(self):
        model = self.model
        if not is_peft_model(model) or self._dist.is_fsdp or self.accelerator.num_processes > 1:
            yield from original(self)
            return
        lora_layers = {name: m for name, m in model.named_modules() if isinstance(m, LoraLayer)}
        for name, param in model.named_parameters():
            if model.prefix in name or "original_module" in name:  # adapter weights / saved originals
                continue
            data = param.data
            if name.endswith(".base_layer.weight"):
                layer = lora_layers[name.removesuffix(".base_layer.weight")]
                delta = sum(layer.get_delta_weight(a).float() for a in layer.active_adapters)
                data = (data.float() + delta).to(data.dtype)
            vllm_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            yield self._fix_param_name_to_vllm(vllm_name, extra_prefixes=["modules_to_save.default."]), data

    trl_vllm.VLLMGeneration._iter_named_params = _iter_named_params


def main() -> None:
    text_only_vllm()
    lora_safe_weight_sync()
    parser = TrlParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_and_config()
    assert script_args.mode in ("zero", "think"), script_args.mode

    train = to_grpo_dataset(Dataset.from_pandas(pd.read_parquet(script_args.train_data)), script_args.mode)
    print(f"[data] {len(train)} prompts from {script_args.train_data} (mode={script_args.mode})")

    if script_args.mode == "zero":
        rewards = [correctness_reward_zero, format_reward_zero]
    else:
        rewards = [correctness_reward, format_reward]
    rewards.append(get_soft_overlong_punishment(training_args.max_completion_length, script_args.soft_punish_cache))
    training_args.reward_weights = [1.0, script_args.format_reward_weight, 1.0]

    # Stop where eval stops: zero-mode Base models would otherwise write the next "User:" turn.
    training_args.generation_kwargs = {**(training_args.generation_kwargs or {}), "stop": STOP_STRINGS[script_args.mode]}

    # Qwen3.5 checkpoints are multimodal: left to itself, TRL loads Qwen3VLProcessor and takes its
    # vision-model code paths. We train text-only, so pass the plain tokenizer (TRL's padding settings).
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, padding_side="left", truncation_side="left"
    )

    trainer = RowBatchedGRPOTrainer(
        model=script_args.model_name_or_path,
        reward_funcs=rewards,
        args=training_args,
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=0.0,
            target_modules=script_args.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        if script_args.use_lora
        else None,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if script_args.use_lora:
        # Adapter in <output_dir>/adapter; the merged model at <output_dir> for eval / vLLM / later stages.
        trainer.save_model(f"{training_args.output_dir}/adapter")
        trainer.model.merge_and_unload().save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
