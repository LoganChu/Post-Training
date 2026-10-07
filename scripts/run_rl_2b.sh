#!/usr/bin/env bash
# Step 9: main-track RL on Qwen3.5-2B-Base (LoRA), end to end (PLAN.md, "Main-track RL on 2B"):
#   1. score prompts with 2B (think mode, 8k, k=8, 3,000 prompts) -> data/rl/rl-2b.parquet
#   2. 2-rollout smoke test (16 optimizer steps), gated: clean exit, 16 finite losses, merged model saved
#   3. only if it passed: the full run + think-mode eval (scripts/run_rl_main.sh)
# Stops (does not retry) on failure, leaving logs for debugging. Resumable: finished stages are
# skipped. Progress: outputs/rl-2b-chain.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

PROMPTS=data/rl/rl-2b.parquet
SMOKE=outputs/smoke-rl-2b
MIN_PROMPTS=400

if [ ! -f "$PROMPTS" ]; then
  log "score prompts with 2B"
  uv run scripts/filter_by_passrate.py --model Qwen/Qwen3.5-2B-Base --mode think --name rl-2b \
    --n-prompts 3000 --max-tokens 8192 --gpu-memory-utilization 0.75 > outputs/passrate-rl-2b.log 2>&1
  grep -aE "^\[|pass rate|^[0-9k/-]+ |^mean|by source" outputs/passrate-rl-2b.log
fi
n=$(uv run python -c "import pandas as pd; print(len(pd.read_parquet('$PROMPTS')))")
if [ "$n" -lt "$MIN_PROMPTS" ]; then log "STOP: only $n usable prompts (< $MIN_PROMPTS)"; exit 1; fi
log "$n usable prompts"

if [ ! -f "$SMOKE/PASSED" ]; then
  rm -rf "$SMOKE"; mkdir -p "$SMOKE"
  log "smoke test: 2 rollouts"
  status=0
  uv run scripts/grpo.py --config configs/grpo_main_2b.yaml --max_steps 16 --save_steps 100000 \
    --output_dir "$SMOKE" > "$SMOKE/train.log" 2>&1 || status=$?
  if uv run python - "$SMOKE" "$status" <<'PYEOF'
import ast, math, re, sys
from pathlib import Path
out, status = Path(sys.argv[1]), int(sys.argv[2])
txt = (out / "train.log").read_text(errors="ignore").replace("\r", "\n")
rows = [ast.literal_eval(m) for m in re.findall(r"\{'loss'.*?\}", txt)]
problems = []
if status != 0:
    problems.append(f"exit status {status}")
if not ((out / "model.safetensors").exists() or (out / "model.safetensors.index.json").exists()):
    problems.append("no merged model saved")
if len(rows) < 16 or not all(math.isfinite(float(r["loss"])) for r in rows):
    problems.append(f"{len(rows)} logged steps / non-finite loss")
for r in rows:
    if "rewards/correctness_reward/mean" in r:
        print("[smoke rollout]", {k.split("/")[-2] if k.startswith("rewards") else k: r[k] for k in r if k in (
            "rewards/correctness_reward/mean", "completions/clipped_ratio", "completions/mean_length",
            "frac_reward_zero_std", "entropy", "step_time")})
print("[smoke] step times:", [r.get("step_time") for r in rows])
if problems:
    print("[smoke] FAIL:", "; ".join(problems)); (out / "FAILED").write_text("\n".join(problems) + "\n"); sys.exit(1)
print("[smoke] PASS"); (out / "PASSED").write_text("ok\n")
PYEOF
  then
    rm -rf "$SMOKE"/model*.safetensors* "$SMOKE"/adapter "$SMOKE"/checkpoint-*
  else
    log "STOP: smoke test failed; see $SMOKE/train.log and $SMOKE/FAILED"; exit 1
  fi
fi

log "full run"
CONFIG=configs/grpo_main_2b.yaml NAME=rl-2b BASE_REPO=Qwen/Qwen3.5-2B-Base scripts/run_rl_main.sh
log "chain done"
