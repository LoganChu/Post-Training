#!/usr/bin/env bash
# Step 7: main-track RL (GRPO, think mode, 8k budget), then the standard think-mode eval at 8k.
#   scripts/run_rl_main.sh                                        # 0.8B: configs/grpo_main_0.8b.yaml -> outputs/rl-0.8b
#   CONFIG=configs/grpo_main_2b.yaml NAME=rl-2b BASE_REPO=Qwen/Qwen3.5-2B-Base scripts/run_rl_main.sh   # 2B (LoRA)
# Resumable: if a checkpoint exists, training continues from the latest one; finished stages are
# skipped. Progress: outputs/<NAME>.log via the caller, metrics in outputs/<NAME>/train.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

CONFIG=${CONFIG:-configs/grpo_main_0.8b.yaml}
NAME=${NAME:-rl-0.8b}
BASE_REPO=${BASE_REPO:-Qwen/Qwen3.5-0.8B-Base}   # source of the processor files vLLM needs
OUT=outputs/$NAME
EVAL=outputs/eval/outputs-$NAME-think

if [ ! -f "$OUT/model.safetensors" ] && [ ! -f "$OUT/model.safetensors.index.json" ]; then
  mkdir -p "$OUT"
  resume=()
  last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [ -n "$last" ]; then resume=(--resume_from_checkpoint "$last"); log "resuming from $last"; fi
  log "train $NAME ($CONFIG)"
  uv run scripts/grpo.py --config "$CONFIG" --output_dir "$OUT" "${resume[@]}" >> "$OUT/train.log" 2>&1
  # Checkpoints keep the multimodal layout but not these files, which vLLM reads for this architecture.
  uv run python - "$OUT" "$BASE_REPO" <<'PYEOF'
import shutil, sys
from huggingface_hub import hf_hub_download
for f in ["preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]:
    shutil.copy(hf_hub_download(sys.argv[2], f), sys.argv[1])
PYEOF
fi

if [ ! -f "$EVAL/summary.json" ]; then
  log "eval $OUT (think, 8k)"
  uv run -m posttrain.eval --model "$OUT" --mode think --max-tokens 8192 --output-dir "$EVAL" 2>&1 \
    | grep -E "^\[|^model=|^benchmark|^(math500|aime2[456]|amc23|gpqa) "
fi
log "done"
