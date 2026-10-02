# Plan: SOTA math-reasoning post-training with TRL on one RTX 5090

Status as of 2026-10-01. Steps 1-3 done (environment, rewards, eval harness, all stage-0 baselines
and the output-budget sweep; results in README.md). Step 4 (RL-Zero) is next.

## Context and decisions

Goal: learn modern LLM post-training hands-on, with the strongest small open models and current
RL methods, on one RTX 5090 (32 GB, Blackwell sm_120) using TRL.

| decision | choice |
|---|---|
| Domain | Math reasoning with verifiable rewards (RLVR) |
| Models | Qwen3.5 small series. Iterate on **0.8B-Base**, main target **2B-Base**, stretch 4B-Base (LoRA) |
| Starting point | **Base** model, run every stage ourselves so each stage's contribution is measurable |
| Main track | Base → SFT → on-policy distillation → RL |
| Ablations | RL-Zero (RL directly on Base), +DPO before RL, SFT-only vs SFT+distill, RL loss grid |
| SFT data | `nvidia/Nemotron-SFT-Math-v3` (most current open traces; DeepSeek-V3.2-Speciale teacher) |
| Reference ceiling | Official post-trained `Qwen/Qwen3.5-2B` (thinking mode) |

Stack (verified working): torch 2.13 (CUDA 13.0, sm_120) · transformers 5.17 · TRL 1.14 ·
vLLM 0.30 · peft 0.21 · flash-linear-attention 0.5.2 · HF Hub `kernels` 0.16.

## Why the stages are in this order

**SFT** is off-policy distillation: the model learns the `<think>` format, when to stop, and
reasoning habits from a frontier teacher's traces. **On-policy distillation** then has the
student generate and a same-vocabulary teacher score every token, fixing the train/inference
mismatch of SFT cheaply. **RL** last pushes beyond the teachers with verifiable rewards.
This follows DeepSeek-R1 (distill first, then RL, beats RL alone on small models) and Qwen3 (small
models via distillation). DPO would sit between distillation and RL (Tülu 3 / Olmo 3 order) but is
only an ablation here: for a math-only pipeline it is the most optional stage.

## Steps and status

| # | step | status |
|---|---|---|
| 1 | Environment: `pyproject.toml`, `scripts/check_env.py` | done |
| 2 | Rewards: `src/posttrain/rewards.py` + tests | done |
| 3 | Eval harness: `src/posttrain/eval.py`, `prompts.py` + tests; stage-0 baselines | done |
| 4 | **RL-Zero** on 0.8B-Base (GRPO, zero prompt): fastest way to validate the GRPO setup | running: started 2026-10-01 19:35, 200 steps x ~230 s ≈ 13 h, `outputs/rlzero-0.8b/` |
| 5 | SFT data build (Nemotron) + SFT on 0.8B | todo |
| 6 | On-policy distillation on 0.8B | todo |
| 7 | RL (main track) on 0.8B | todo |
| 8 | Ablations on 0.8B: +DPO, SFT-only vs SFT+distill, RL loss grid | todo |
| 9 | Promote winning configs to 2B (then 4B LoRA if time) | todo |

Every step: small increments, a smoke test (20 steps, peak VRAM logged) before any long run, and
an eval with the same protocol as the baselines afterwards.

## Stage details

### Stage 1: SFT cold-start (`scripts/sft.py`, `scripts/build_sft_data.py`)
- Data: Nemotron-SFT-Math-v3, **no-TIR CoT subset only**, streamed (full set is 144 GB).
  Filter to traces of 16k tokens or fewer, dedupe to 1-2 traces per problem, sample 50-100k,
  decontaminate against all eval sets. Check how reasoning is stored in `messages` (inline
  `<think>` vs a separate field) and render it into Qwen3.5's `<think>\n...\n</think>\n\n` format.
- **Set EOS to `<|im_end|>`.** The Base tokenizer's EOS is `<|endoftext|>`, but chat turns end in
  `<|im_end|>`; without this the model never learns to stop.
- 2B full fine-tune: bf16, Liger / chunked cross-entropy (see memory note below), 8-bit AdamW,
  gradient checkpointing, packing, max length ~16k, `use_kernels=True`.
- Success: think-mode truncation on AIME drops well below the Base model's 84-90%, MATH-500 up.
- Risk: frontier-teacher traces may be hard for a 0.8-2B student to imitate (capacity gap).
  Fallback: compare against a small set of self-generated Qwen3.5-9B traces.

### Stage 2: On-policy distillation (`scripts/distill.py`)
- `trl.DistillationTrainer`, `beta≈1.0` (reverse KL; mode-seeking, standard for on-policy KD).
- Student: stage-1 checkpoint. Teacher: Qwen3.5-4B thinking (must share the student's
  vocabulary, so the Nemotron teacher cannot be used here).
- Prompt-only math prompts (DeepMath-103K), decontaminated.
- vLLM colocate with `vllm_enable_sleep_mode`. If 4B teacher + 2B student + vLLM do not fit:
  LoRA student, quantized teacher, or `AsyncDistillationTrainer`.

### Stage 3 (ablation only): DPO, delta-learning style (`scripts/dpo.py`, `scripts/make_dpo_pairs.py`)
- Chosen = a correct Qwen3.5-9B answer; rejected = the current policy's wrong answer on the
  same prompt. LoRA, so the frozen base doubles as the reference model.
- Compare RL curves and final scores with and without this step.

### Stage 4: RL with verifiable rewards (`scripts/grpo.py`)
- Data: DAPO-Math-17k + DeepMath-103K, **pre-filtered by the current policy's pass rate**
  (`scripts/filter_by_passrate.py`, k=8, keep 0 < p < 1), so every group has a learning signal.
  Pool (`uv run -m posttrain.data build` → `data/rl/pool.parquet`, 101,957 prompts: 17,853 DAPO,
  84,104 DeepMath). DAPO's "Answer: $Answer" wrapper is stripped; DAPO's 1.79M Hub rows are ~100x
  duplicates. Dropped: 2 empty answers; **18,876 yes/no/true/false answers** (skewed ~4:1 to
  "yes", so always guessing "Yes" would be rewarded); 104 eval overlaps (≥30% of an eval problem's
  13-grams; a single shared 13-gram flagged 902, mostly AIME boilerplate). Known grader limit:
  some equivalent notations are missed (e.g. `x < -5` vs `(-\infty,-5)`), giving occasional false
  negatives on DeepMath's non-integer answers; DAPO's answers are all integers.
- Rewards (`rewards.py`): `correctness_reward` (math-verify, 1/0), `format_reward` (small
  weight), `get_soft_overlong_punishment` (DAPO length penalty).
- Recipe (SOTA defaults in TRL 1.14):
  - `loss_type="dapo"`, `epsilon=0.2`, `epsilon_high=0.28` (clip-higher), `beta=0.0`
  - `num_generations=16`, rollout `temperature=1.0`, `max_completion_length` 4k then 8k.
    Staged lengthening follows DeepScaleR (1.5B: 8k → 16k → 24k), not DAPO: DAPO (Qwen2.5-32B)
    used a fixed 20,480 tokens (16,384 expected + 4,096 soft-punish cache), 16 responses per
    prompt, 512 prompts per batch, lr 1e-6. A short start suits 0.8B, which gained little from
    longer budgets in the sweep; 2B may justify 16k (step 9).
  - `scale_rewards="batch"`, `mask_truncated_completions=True`
  - default vLLM importance-sampling correction (`sequence_mask`)
  - vLLM colocate, `vllm_gpu_memory_utilization≈0.35`, sleep mode; full fine-tune
- Monitor: reward mean, entropy (collapse), clip ratio, fraction of groups where every sample
  gets the same reward, completion length, truncation.
- Loss ablation grid (0.8B, fixed step budget): `dapo` vs `dr_grpo` vs `cispo`
  (`epsilon_high=5.0`) vs GSPO (`importance_sampling_level="sequence"`) vs `sapo`.

### GRPO on Qwen3.5 in TRL 1.14: what the smoke test needed (`scripts/grpo.py`)
1. **Text-only vLLM.** TRL builds its colocated `vllm.LLM` without `limit_mm_per_prompt`, so vLLM
   profiled the vision encoder and had no memory left for the KV cache. Patched in `grpo.py`.
2. **vLLM V1 model runner** (`VLLM_USE_V2_MODEL_RUNNER=0`, set in `grpo.py`). The default V2
   runner crashes in its startup profiling pass on Qwen3.5's DeltaNet layers in TRL's in-process mode.
3. **Plain tokenizer.** Otherwise TRL loads `Qwen3VLProcessor` and takes its vision-model paths.
4. **Logits memory.** `use_liger_kernel: true` selects TRL's chunked log-prob path (full logits:
   30 GiB OOM). That path ignores `batch_size` and scored all 512 rollouts in one forward pass
   (OOM again); `RowBatchedGRPOTrainer` in `grpo.py` splits those calls into 8-row slices.
5. **8-bit AdamW** (`optim: adamw_bnb_8bit`). fp32 Adam states (~6 GB) left no room for vLLM to
   wake for step 2. With it, vLLM at `gpu_memory_utilization: 0.30`.
6. **Keep the GPU to ourselves.** An Ollama service on this machine loaded a 15.6 GB model mid-way
   through testing. Long runs checkpoint every 25 steps so a crash can be resumed.

Smoke test (3 steps): ~230 s/step; correctness reward ~17%, format ~64%, ~37% of rollouts hit the
4,096-token limit, 0% all-equal-reward groups (the pass-rate filter works), entropy ~1.0.

### RL-Zero track (step 4, also an ablation)
- Same stage-4 recipe on the Base model with the plain-text `zero` prompt, and
  `correctness_reward_zero` / `format_reward_zero`. The zero rewards accept a well-formed,
  self-opened `<think>` block (the Base model writes one ~10% of the time) and reject malformed
  tags.
- Answers: what do SFT and distillation add on top of pure RL?

## Evaluation

### Protocol (settled; do not change without re-running baselines)
T=0.6, top_p=0.95, top_k=20, **presence_penalty=1.5**, seed 0: the `eval.py` defaults, recorded
in every `summary.json`. Why: without the penalty, the post-trained 2B loops in ~90% of its
truncated samples (38% on MATH-500); Qwen's recommended T=1.0 fixes that but drops the Base
models (0.8B zero: 34% → 22%), while T=0.6 + penalty costs them nothing. RL still *samples*
rollouts at T=1.0; only eval uses this protocol.

Evidence (MATH-500 subsets, run 2026-09-29; the raw outputs of these and of all old-protocol
baselines were deleted 2026-09-30, only these numbers are kept):

| model | mode | T | presence penalty | problems | avg@4 | truncated |
|---|---|---|---|---|---|---|
| 0.8B-Base | zero | 0.6 | 0.5 | 100 | 36.2% | 20% |
| 0.8B-Base | zero | **0.6** | **1.5** | 100 | **40.2%** | 14% |
| 0.8B-Base | zero | 1.0 | 0.0 | 100 | 29.0% | 10% |
| 0.8B-Base | zero | 1.0 | 0.5 | 100 | 31.0% | 7% |
| post-trained 2B | think (32k) | 0.6 | 0.5 | 50 | 44.0% | 56% |
| post-trained 2B | think (32k) | **0.6** | **1.5** | 50 | **70.0%** | 28% |
| post-trained 2B | think (32k) | 1.0 | 0.0 | 50 | 58.0% | 40% |

Full-benchmark runs under superseded protocols: 0.8B-Base zero MATH-500 was 34.1% at T=0.6 with
no penalty, and 22.0% at T=1.0 + penalty 1.5; post-trained 2B MATH-500 was 38.3% at T=0.6 with no
penalty (88% of its truncations were repetition loops).

Benchmarks: MATH-500 (k=4), AIME 2024/2025/2026 (k=16), AMC23 (k=8), GPQA-Diamond (k=4,
out-of-domain). Metrics: avg@k (headline), pass@k (RL headroom), format rate, truncation rate,
mean tokens. Grading uses the same `is_correct` as the RL reward.

### Stage-0 baselines and budget sweep (done 2026-10-01)
Results tables: README.md. Raw samples: `outputs/eval/` (Base runs: default k; post-trained 2B:
k=1 for MATH-500/GPQA, k=4 for AIME/AMC, at 32k and at 8k; budget sweep in
`outputs/eval/budget-sweep/`). Scripts: `scripts/run_baselines.sh`, `scripts/run_followups.sh`
(both resumable: runs already complete under the current protocol are skipped).

Why the sweep: max output length is an eval choice, but the right value depends on how long a
model needs to finish. The Base baselines stay at 8k to match the SFT/RL training budget. The
model card recommends 32,768 output tokens in general and 81,920 for competition-math
benchmarking, but does not say which length produced its own benchmark tables.

If an eval process hangs at the end (vLLM shutdown bug), kill `VLLM::EngineCore`. To re-score
saved samples after a grader change: `uv run -m posttrain.eval --model M --mode X --regrade`.

### Open ceiling follow-ups (optional)
- **Official sampling.** The card's T=1.0 scored 72% vs 70% at our protocol on a 50-problem
  check (within noise); rerun at T=1.0 into `outputs/eval/Qwen--Qwen3.5-2B-think-official` only
  if reporting against Qwen's published numbers.
- **81,920-token budget.** At 32k, 82-88% of the post-trained 2B's AIME samples still truncate.
  Only worth it if the ceiling is the number being reported against.

### Per-stage evals (same protocol and benchmarks, compared line-for-line with the baselines)

| stage | checkpoint | mode | max_tokens | compare against |
|---|---|---|---|---|
| RL-Zero | `outputs/rlzero-0.8b`, `outputs/rlzero-2b` | zero | 8192 | Base zero (rows 1, 3) |
| SFT | `outputs/sft-0.8b`, `outputs/sft-2b` | think | 8192 (raise if truncation > 30%) | Base think (rows 2, 4) |
| SFT + distill | `outputs/distill-0.8b`, `outputs/distill-2b` | think | 8192-16384 | SFT rows |
| SFT + distill + RL | `outputs/rl-0.8b`, `outputs/rl-2b` | think | 8192-16384 | distill rows and post-trained 2B at the same budget |
| +DPO ablation | `outputs/dpo-rl-0.8b` | think | same as RL | RL row |

```
uv run -m posttrain.eval --model outputs/<checkpoint> --mode <zero|think> --max-tokens 8192
uv run -m posttrain.eval --model outputs/<checkpoint> --mode think --benchmarks math500 --limit 100   # ~2 min sanity check
```

## Findings so far (things that shaped the code)
- **Qwen3.5 is a hybrid DeltaNet/attention model.** Fast training needs `fla` (delta-rule
  kernel) and the HF Hub `kernels` package (`use_kernels=True`, prebuilt causal-conv1d; the
  PyPI package needs `nvcc` and hung while building). With both: 0.48 s/step at 8k tokens (0.8B).
- **Logits dominate memory.** 0.8B at 8k tokens peaks at 24.7 GiB, mostly the 248k-vocab
  logits and their gradient. Use chunked / Liger cross-entropy in every trainer.
- **The chat template opens the think block.** Qwen3.5's generation prompt ends with
  `<think>\n`, so completions contain only `...</think>\n\nanswer`. TRL's `think_format_reward`
  (expects a leading `<think>`) would always score 0; we use our own `format_reward`.
- **Reward hardening.** Main-track credit requires exactly one `</think>` before the answer
  (truncated guesses get 0); multiple `\boxed{}` answers parse as a set and never match
  (anti-hedging); gold answers are wrapped in `\boxed{}` before parsing (bare LaTeX like
  `(x+1)^2` otherwise fails silently and the row is dropped).
- **The Base models already reason, but do not stop.** 2B-Base: 61% MATH-500 in zero mode, 72%
  in think mode, with no post-training. In think mode both sizes truncate on 80-90% of AIME at 8k
  (zero mode: ~40-47%): the `<think>` block starts long reasoning they never learned to end.
  Learning to finish is a large part of what SFT must teach.
- **Extra tokens help 2B, not 0.8B.** 2B-Base think on AIME 2025: 12.3% → 16.2% → 19.6% at
  8k/16k/32k (pass@16 30% → 53%); MATH-500 saturates at 16k. 0.8B-Base stays flat (AIME 1.0% →
  1.7%) while truncation falls: its long outputs wander. Implication: 8k is fine for 0.8B (its
  problem is reasoning quality); for 2B, consider a 16k completion budget in later RL (step 9).
- **The post-trained 2B is very verbose.** ~12.5k tokens on MATH-500 even when it finishes
  (97% correct when it does). At 8k it scores 0% on all AIME years and trails its own Base model
  on every benchmark; at 32k it is roughly level with 2B-Base at 8k. Beating it at our 8k budget
  is a concrete target for SFT + RL.

## Repo layout

```
pyproject.toml                 # uv project (done)
PLAN.md                        # this file
src/posttrain/
  prompts.py                   # think / zero prompt templates + stop strings (done)
  rewards.py                   # correctness, format, overlong penalty (done)
  eval.py                      # vLLM eval, --regrade (done)
  data.py                      # dataset loaders, filters, decontamination (todo, step 4-5)
configs/                       # one YAML per stage x model size, TrlParser format (todo)
scripts/
  check_env.py                 # environment + kernel check (done)
  run_baselines.sh             # stage-0 evals, resumable (done)
  run_followups.sh             # 8k ceiling + output-budget sweep (done)
  grpo.py                      # RL-Zero and main-track RL (todo, step 4)
  filter_by_passrate.py        # keep prompts with 0 < pass rate < 1 (todo, step 4)
  build_sft_data.py, sft.py    # (todo, step 5)
  distill.py                   # (todo, step 6)
  make_dpo_pairs.py, dpo.py    # (todo, step 8)
  smoke_test.sh                # 20 steps of each stage on 0.8B (todo)
tests/                         # test_rewards.py, test_eval.py (25 passing)
```

## Risks / open points
- Qwen3.5 in vLLM colocate with weight sync and sleep mode during GRPO (hybrid architecture,
  multimodal checkpoint). Fallbacks: `vllm_mode="server"` on the same GPU, or Qwen3 dense models.
- Memory for distillation (4B teacher + 2B student + vLLM).
- Nemotron trace length vs student capacity (see stage 1).
- Background jobs started from a Claude session die when the session ends; long runs are
  launched with `setsid nohup` instead.

## Housekeeping log
- 2026-09-30: old-protocol outputs, protocol-ablation scratch runs and `scripts/rerun_gpqa.sh`
  deleted; their numbers are kept in the protocol section above.
- 2026-10-01: stage-0 complete; eval changes and scripts committed (`a568fa3`); results added to
  README.md.
