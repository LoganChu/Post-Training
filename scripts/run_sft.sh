#!/usr/bin/env bash
# Step 5: SFT cold-start on Qwen3.5-0.8B-Base (PLAN.md, Stage 1), end to end:
#   1. build the trace set from Nemotron-SFT-Math-v3 (50k traces up to 16k tokens; sft.py trains on
#      the ~27k that fit the 8k budget)
#   2. SFT with configs/sft_0.8b.yaml
#   3. standard think-mode eval at 8k (compare: 0.8B-Base think row in README.md)
# Resumable: finished stages are skipped. Progress: outputs/sft.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

DATA=data/sft
OUT=outputs/sft-0.8b
EVAL=outputs/eval/outputs-sft-0.8b-think

if [ ! -f "$DATA/nemotron-train.parquet" ]; then
  log "build SFT data"
  uv run scripts/build_sft_data.py --target 50000 --val 500 --out-dir "$DATA" 2>&1 | grep --line-buffered -v -E "Timeout during comparison|^Exception ignored|^Traceback|^  File |^    |TimeoutException|^\s*$"
fi

if [ ! -f "$OUT/model.safetensors" ]; then
  log "train SFT"
  mkdir -p "$OUT"
  uv run scripts/sft.py --config configs/sft_0.8b.yaml --output_dir "$OUT" > "$OUT/train.log" 2>&1
  grep -E "\[data\]|peak VRAM|train_runtime" "$OUT/train.log" | tr '\r' '\n' | grep -E "\[data\]|peak VRAM|train_runtime"
fi

if [ ! -f "$EVAL/summary.json" ]; then
  log "eval $OUT (think, 8k)"
  uv run -m posttrain.eval --model "$OUT" --mode think --max-tokens 8192 --output-dir "$EVAL" 2>&1 \
    | grep -E "^\[|^model=|^benchmark|^(math500|aime|amc23|gpqa)"
fi
log "done"
