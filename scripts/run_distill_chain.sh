#!/usr/bin/env bash
# Step 6 chain (run by scripts/run_sft_v2.sh once SFT is done; the GPU must be free):
#   1. 3-step smoke test of scripts/distill.py on $STUDENT, gated on explicit pass criteria
#   2. only if it passed: the full run (scripts/run_distill.sh: distillation + think-mode eval)
# A failed smoke test stops the chain and leaves outputs/smoke-distill/ for debugging; nothing is
# retried automatically. Resumable: a recorded PASS is not re-run. Progress: outputs/distill-chain.log.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

export STUDENT=${STUDENT:-outputs/sft-0.8b-8k}
SMOKE=outputs/smoke-distill
if [ ! -f "$STUDENT/model.safetensors" ]; then
  log "STOP: no student checkpoint at $STUDENT"; exit 1
fi

if [ ! -f "$SMOKE/PASSED" ]; then
  rm -rf "$SMOKE"; mkdir -p "$SMOKE"
  log "smoke test: 3 steps (student $STUDENT)"
  status=0
  uv run scripts/distill.py --config configs/distill_0.8b.yaml --model_name_or_path "$STUDENT" --max_steps 3 --save_steps 1000 \
    --output_dir "$SMOKE" > "$SMOKE/train.log" 2>&1 || status=$?
  # Pass criteria: clean exit, saved model, 3 finite losses, and some rollouts ending on <|im_end|>
  # in every step (clipped_ratio = share cut at max_completion_length). Not "most": the SFT student
  # genuinely truncates 80-90% of T=1.0 rollouts at 8k (2026-10-03 smoke test); <= 0.95 checks that
  # EOS works at all, which is what would make a full run pointless.
  if uv run python - "$SMOKE" "$status" <<'EOF'
import ast, math, re, sys
from pathlib import Path
out, status = Path(sys.argv[1]), int(sys.argv[2])
txt = (out / "train.log").read_text(errors="ignore").replace("\r", "\n")
rows = [ast.literal_eval(m) for m in re.findall(r"\{'loss'.*?\}", txt)]
clipped = [float(r["completions/clipped_ratio"]) for r in rows if "completions/clipped_ratio" in r]
problems = []
if status != 0:
    problems.append(f"exit status {status}")
if not (out / "model.safetensors").exists():
    problems.append("no model saved")
if len(rows) < 3 or not all(math.isfinite(float(r["loss"])) for r in rows):
    problems.append(f"{len(rows)} logged steps / non-finite loss")
if not clipped or max(clipped) > 0.95:
    problems.append(f"clipped_ratio {clipped} (almost no rollouts end on <|im_end|>)")
for r in rows:
    keep = {k: r[k] for k in r if k in ("loss", "completions/mean_length", "completions/clipped_ratio", "step_time")}
    print("[smoke step]", keep)
mem = re.findall(r"peak VRAM allocated: ([\d.]+) GiB", txt)
print("[smoke] peak VRAM:", mem[-1] if mem else "n/a", "GiB")
if problems:
    print("[smoke] FAIL:", "; ".join(problems))
    (out / "FAILED").write_text("\n".join(problems) + "\n")
    sys.exit(1)
print("[smoke] PASS")
(out / "PASSED").write_text("ok\n")
EOF
  then
    rm -f "$SMOKE/model.safetensors"; rm -rf "$SMOKE"/checkpoint-*
  else
    log "STOP: smoke test failed; see $SMOKE/train.log and $SMOKE/FAILED"; exit 1
  fi
fi

log "full distillation run"
scripts/run_distill.sh
log "chain done"
