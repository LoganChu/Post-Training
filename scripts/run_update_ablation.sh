#!/usr/bin/env bash
# Updates-per-rollout ablation for RL-Zero on Qwen3.5-0.8B-Base (PLAN.md, step 8).
# Same rollout budget for every arm (100 rollouts x 32 prompts x 16 samples); only how each rollout
# batch is used changes:
#   A  1 update/rollout (GRPO paper)       = the main run outputs/rlzero-0.8b, compared at step 100
#   B  4 updates/rollout, clip 0.2/0.28    (DAPO-style clip-higher) -> outputs/rlzero-0.8b-4upd-cliphigher
#   (arm C, symmetric clip, dropped: see the end of this script)
# 4 updates/rollout: the 512-completion rollout (steps_per_generation=64 micro-batches of 8) feeds
# 4 optimizer steps of 16 micro-batches each, so 100 rollouts = 400 optimizer steps; warmup and
# checkpointing are scaled x4 to stay matched per rollout.
# Each arm then gets the standard eval (zero mode, 8k, eval.py protocol).
#
# Started right after the main run's launch; waits for it. Resumable: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

MAIN=outputs/rlzero-0.8b
A100=outputs/rlzero-0.8b-step100

# Checkpoints hold the language model + untouched vision tower in the original layout, but not the
# image/video preprocessor configs that vLLM may read for this multimodal architecture.
add_processor_files() {
  uv run python - "$1" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
for f in ["preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]:
    shutil.copy(hf_hub_download("Qwen/Qwen3.5-0.8B-Base", f), sys.argv[1])
EOF
}

evaluate() {  # evaluate <model_dir>: standard zero-mode eval, skipped if already done
  local model=$1 out="outputs/eval/$(echo "$1" | tr '/' '-')-zero"
  if [ -f "$out/summary.json" ]; then log "skip eval $model (done)"; return; fi
  add_processor_files "$model"
  log "eval $model"
  uv run -m posttrain.eval --model "$model" --mode zero --max-tokens 8192 --output-dir "$out" 2>&1 \
    | tee -a outputs/ablation-full.log | grep -E "^\[|^model=|^benchmark|^(math500|aime|amc23|gpqa)" \
    || log "EVAL FAILED for $model (see outputs/ablation-full.log)"
}

train_arm() {  # train_arm <output_dir> <extra overrides...>
  local out=$1; shift
  if [ -f "$out/model.safetensors" ]; then log "skip training $out (done)"; return; fi
  log "train $out"
  mkdir -p "$out"
  uv run scripts/grpo.py --config configs/grpo_rlzero_0.8b.yaml --output_dir "$out" \
    --gradient_accumulation_steps 16 --steps_per_generation 64 \
    --max_steps 400 --warmup_steps 40 --save_steps 100 "$@" > "$out/train.log" 2>&1 \
    || log "TRAINING FAILED for $out (see $out/train.log)"
}

# 1. Rescue the main run's step-100 checkpoint: save_total_limit=4 deletes it at step 200.
if [ ! -f "$A100/model.safetensors" ]; then
  until [ -f "$MAIN/checkpoint-100/trainer_state.json" ]; do sleep 60; done
  sleep 30  # let the save finish writing
  mkdir -p "$A100"
  cp "$MAIN"/checkpoint-100/{model.safetensors,config.json,generation_config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja} "$A100"/
  log "copied step-100 checkpoint to $A100"
fi

# 2. Wait for the main run to finish (GPU must be free).
until grep -q "train_runtime" "$MAIN/train.log" 2>/dev/null && ! pgrep -f "scripts/grpo[.]py" > /dev/null; do sleep 60; done
log "main run finished"

# 3. Arm A evals (step 100 = ablation comparison point; step 200 = the run's final result).
evaluate "$A100"
evaluate "$MAIN"

# 4. Arm B.
train_arm outputs/rlzero-0.8b-4upd-cliphigher --epsilon_high 0.28
evaluate outputs/rlzero-0.8b-4upd-cliphigher
# Arm C (4 updates, symmetric clip 0.2/0.2) was dropped 2026-10-02: in arm B only ~0.008% of tokens
# were clipped (lr 1e-6 moves the policy too little in 4 updates to reach the bounds), so the
# clip-higher vs symmetric comparison could not differ. To run it anyway:
#   train_arm outputs/rlzero-0.8b-4upd-symclip --epsilon_high 0.2
#   evaluate outputs/rlzero-0.8b-4upd-symclip
log "ablation done"
