#!/usr/bin/env bash
# Stage 0: baselines every later stage is compared against.
#   - Base models in both prompt modes: "zero" (plain text, what RL-Zero starts from) and
#     "think" (chat template, what SFT will teach; shows how much format alone is worth).
#   - The official post-trained Qwen3.5-2B in thinking mode: the "ceiling" our 2B pipeline chases.
# Results: outputs/eval/<model>-<mode>/summary.json (+ per-sample jsonl).
set -euo pipefail
cd "$(dirname "$0")/.."

run() { uv run -m posttrain.eval "$@" 2>&1 | grep -v -E "it/s\]|^\s*$" | grep -E "^\[|model=|^benchmark|^(math500|aime|amc23|gpqa)"; }

for model in Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-2B-Base; do
  for mode in zero think; do
    run --model "$model" --mode "$mode" --max-tokens 8192
  done
done
run --model Qwen/Qwen3.5-2B --mode think --max-tokens 32768
