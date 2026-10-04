#!/usr/bin/env bash
# Step 6: on-policy distillation into the SFT'd Qwen3.5-0.8B (PLAN.md, Stage 2), end to end:
#   1. distill $STUDENT (default outputs/sft-0.8b-8k) against Qwen3.5-4B with configs/distill_0.8b.yaml
#   2. standard think-mode eval at 8k (compare: the SFT row)
# Resumable: finished stages are skipped. Run the smoke test in scripts/distill.py first.
# Status (2026-10-03): stopped. At an 8k budget the student drifted toward the teacher's long
# reasoning (97% of rollouts truncated) and the run OOMed at step 34; see PLAN.md, Stage 2.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

STUDENT=${STUDENT:-outputs/sft-0.8b-8k}
OUT=outputs/distill-0.8b
EVAL=outputs/eval/outputs-distill-0.8b-think

if [ ! -f "$OUT/model.safetensors" ]; then
  log "train distillation (student $STUDENT)"
  mkdir -p "$OUT"
  uv run scripts/distill.py --config configs/distill_0.8b.yaml --model_name_or_path "$STUDENT" --output_dir "$OUT" > "$OUT/train.log" 2>&1
  grep -E "\[data\]|peak VRAM|train_runtime" "$OUT/train.log" | tr '\r' '\n' | grep -E "\[data\]|peak VRAM|train_runtime"
fi

if [ ! -f "$EVAL/summary.json" ]; then
  log "eval $OUT (think, 8k)"
  uv run -m posttrain.eval --model "$OUT" --mode think --max-tokens 8192 --output-dir "$EVAL" 2>&1 \
    | grep -E "^\[|^model=|^benchmark|^(math500|aime|amc23|gpqa)"
fi
log "done"
