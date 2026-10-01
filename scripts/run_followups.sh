#!/usr/bin/env bash
# Follow-ups to the stage-0 baselines (run after scripts/run_baselines.sh):
#   1. Matched-budget ceiling: post-trained Qwen3.5-2B at 8,192 tokens (the budget our own models
#      are trained and evaluated at), same reduced k as the 32k ceiling run.
#   2. Output-budget sweep: do the Base models (think mode) gain accuracy with more tokens, or
#      just truncate less? 2B then 0.8B; AIME 2025 (k=16) + first 100 MATH-500 problems (k=4) at
#      16k and 32k. The 8k point is the matching subset of each model's main think baseline.
# Same eval protocol as the baselines (eval.py defaults). Resumable: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

# done_already <output_dir> <benchmarks>: true if summary.json has those benchmarks under this protocol.
done_already() {
  uv run python - "$1/summary.json" "$2" <<'EOF'
import json, sys
path, benches = sys.argv[1], sys.argv[2].split(",")
try:
    s = json.load(open(path))
except FileNotFoundError:
    sys.exit(1)
sp = s.get("sampling") or {}
ok = sp.get("temperature") == 0.6 and sp.get("presence_penalty") == 1.5 and all(b in s["benchmarks"] for b in benches)
sys.exit(0 if ok else 1)
EOF
}

run() {
  local out=$1 benches=$2; shift 2
  if done_already "$out" "$benches"; then echo "[skip] $out ($benches): already done"; return; fi
  uv run -m posttrain.eval --output-dir "$out" --benchmarks "$benches" "$@" 2>&1 \
    | tee -a outputs/followups-full.log | grep -v -E "it/s\]|^\s*$" | grep -E "^\[|model=|^benchmark|^(math500|aime|amc23|gpqa)"
}

CEIL8K=outputs/eval/Qwen--Qwen3.5-2B-think-8k
run $CEIL8K math500,gpqa --model Qwen/Qwen3.5-2B --mode think --max-tokens 8192 --k 1
run $CEIL8K aime24,aime25,aime26,amc23 --model Qwen/Qwen3.5-2B --mode think --max-tokens 8192 --k 4

for size in 2B 0.8B; do
  for budget in 16384 32768; do
    run outputs/eval/budget-sweep/Qwen3.5-$size-Base-think-$budget aime25,math500 \
      --model Qwen/Qwen3.5-$size-Base --mode think --max-tokens $budget --limit 100
  done
done
