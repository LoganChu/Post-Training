#!/usr/bin/env bash
# SFT v2 (2026-10-03): rerun SFT from Qwen3.5-0.8B-Base with fp32 master weights (the first run's
# bf16 weights froze most parameters; see PLAN.md), on traces whose completion fits the 8k budget
# all stages are evaluated at (25.7k traces). Then the standard think-mode eval, and the
# distillation chain (smoke test -> gate -> full run) on this checkpoint.
# A 4k-trace arm (12.2k traces) was dropped before training, at the user's request; to run it
# later, add `arm 4k 4096` below.
# Resumable: finished stages are skipped. Progress: outputs/sft-v2.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

arm() {  # arm <name> <max_completion_tokens>
  local out=outputs/sft-0.8b-$1 eval=outputs/eval/outputs-sft-0.8b-$1-think
  if [ ! -f "$out/model.safetensors" ]; then
    log "train SFT $1 (completion <= $2 tokens, fp32 weights)"
    mkdir -p "$out"
    uv run scripts/sft.py --config configs/sft_0.8b.yaml --max_completion_tokens "$2" --output_dir "$out" > "$out/train.log" 2>&1
    tr '\r' '\n' < "$out/train.log" | grep -aE "\[data\]|peak VRAM|train_runtime|'eval_loss'"
  fi
  if [ ! -f "$eval/summary.json" ]; then
    log "eval $out (think, 8k)"
    uv run -m posttrain.eval --model "$out" --mode think --max-tokens 8192 --output-dir "$eval" 2>&1 \
      | grep -E "^\[|^model=|^benchmark|^(math500|aime2[456]|amc23|gpqa) "
  fi
}

arm 8k 8192

STUDENT=outputs/sft-0.8b-8k
log "distillation student: $STUDENT"
STUDENT="$STUDENT" scripts/run_distill_chain.sh
log "done"
