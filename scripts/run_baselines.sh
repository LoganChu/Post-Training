#!/usr/bin/env bash
# Stage 0: baselines every later stage is compared against.
#   - Base models in both prompt modes: "zero" (plain text, what RL-Zero starts from) and
#     "think" (chat template, what SFT will teach; shows how much format alone is worth).
#   - The official post-trained Qwen3.5-2B in thinking mode: the "ceiling" our 2B pipeline chases.
#     Reduced k (it writes 10k+ token traces; full k would take 5-7 h); it is a reference point.
# Sampling: eval.py defaults (T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5) for every model.
# Results: outputs/eval/<model>-<mode>/summary.json (+ per-sample jsonl); full vLLM output in
# outputs/baselines-full.log.
#
# Resumable: a run is skipped if its summary.json already has all requested benchmarks under the
# current protocol, so after an interruption just run the script again.
set -euo pipefail
cd "$(dirname "$0")/.."

ALL=math500,aime24,aime25,aime26,amc23,gpqa

# done <model> <mode> <benchmarks>: true if those benchmarks were already run with this protocol.
done_already() {
  local dir="outputs/eval/${1//\//--}-$2"
  uv run python - "$dir/summary.json" "$3" <<'EOF'
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
  local model=$1 mode=$2 benches=$3; shift 3
  if done_already "$model" "$mode" "$benches"; then echo "[skip] $model $mode ($benches): already done"; return; fi
  uv run -m posttrain.eval --model "$model" --mode "$mode" --benchmarks "$benches" "$@" 2>&1 \
    | tee -a outputs/baselines-full.log | grep -v -E "it/s\]|^\s*$" | grep -E "^\[|model=|^benchmark|^(math500|aime|amc23|gpqa)"
}

for model in Qwen/Qwen3.5-0.8B-Base Qwen/Qwen3.5-2B-Base; do
  for mode in zero think; do
    run "$model" "$mode" "$ALL" --max-tokens 8192
  done
done
run Qwen/Qwen3.5-2B think math500,gpqa --max-tokens 32768 --k 1
run Qwen/Qwen3.5-2B think aime24,aime25,aime26,amc23 --max-tokens 32768 --k 4
