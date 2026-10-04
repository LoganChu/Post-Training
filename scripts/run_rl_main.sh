#!/usr/bin/env bash
# Step 7: main-track RL on the SFT'd Qwen3.5-0.8B (configs/grpo_main_0.8b.yaml: GRPO, think mode,
# 8k budget, 100 rollouts x 256 completions, ~8.7 min per rollout ≈ 14.5 h), then the standard
# think-mode eval at 8k. Resumable: if a checkpoint exists, training continues from the latest one;
# finished stages are skipped. Progress: outputs/rl-main.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

OUT=outputs/rl-0.8b
EVAL=outputs/eval/outputs-rl-0.8b-think

if [ ! -f "$OUT/model.safetensors" ]; then
  mkdir -p "$OUT"
  resume=()
  last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [ -n "$last" ]; then resume=(--resume_from_checkpoint "$last"); log "resuming from $last"; fi
  log "train main-track RL"
  uv run scripts/grpo.py --config configs/grpo_main_0.8b.yaml --output_dir "$OUT" "${resume[@]}" >> "$OUT/train.log" 2>&1
  uv run python - "$OUT" <<'PYEOF'
import shutil, sys
from huggingface_hub import hf_hub_download
for f in ["preprocessor_config.json", "video_preprocessor_config.json", "vocab.json"]:
    shutil.copy(hf_hub_download("Qwen/Qwen3.5-0.8B-Base", f), sys.argv[1])
PYEOF
fi

if [ ! -f "$EVAL/summary.json" ]; then
  log "eval $OUT (think, 8k)"
  uv run -m posttrain.eval --model "$OUT" --mode think --max-tokens 8192 --output-dir "$EVAL" 2>&1 \
    | grep -E "^\[|^model=|^benchmark|^(math500|aime2[456]|amc23|gpqa) "
fi
log "done"
