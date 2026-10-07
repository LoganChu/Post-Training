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
| 4 | **RL-Zero** on 0.8B-Base (GRPO, zero prompt): fastest way to validate the GRPO setup | done 2026-10-02 08:15 (`outputs/rlzero-0.8b/`); results below |
| 5 | SFT data build (Nemotron) + SFT on 0.8B | done 2026-10-03 (`scripts/run_sft.sh`): fp32 SFT on <=8k traces behaves like the Base model; the first (bf16) run learned almost nothing |
| 6 | On-policy distillation on 0.8B | stopped: student grew longer (97% truncated) and the run OOMed at step 34 (2026-10-03 17:36); no checkpoint |
| 7 | RL (main track) on 0.8B | done 2026-10-04 08:31 (`outputs/rl-0.8b`): truncation 33% → 5%, half the tokens, accuracy ~unchanged |
| 8 | Ablations on 0.8B: **updates per rollout / clip-higher** (done; launcher removed, overrides in the `configs/grpo_rlzero_0.8b.yaml` header), +DPO, SFT-only vs SFT+distill, RL loss grid | updates ablation done 2026-10-02 (arm C dropped); rest todo |
| 9 | Main-track RL on 2B (LoRA) | prompts scored (1,565 kept) and smoke test passed 2026-10-04; full run stopped at step 8 (19:30) to switch to token-level IS; not restarted |

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

Draft notes (2026-10-02; nothing run on the GPU yet, the update ablation had it):
- **Data layout.** One 154 GB JSONL. Reasoning is in the assistant message's `reasoning_content`,
  the write-up in `content`; Qwen3.5's chat template renders that pair as
  `<think>\n...\n</think>\n\n...`. CoT rows fill about the first 65% of the file in alternating
  AoPS / StackExchange blocks, TIR rows the rest, so `build_sft_data.py` reads the start of
  randomly ordered slices (HTTP range requests) instead of the first N rows.
- **Traces are long.** No-TIR completions: median ~12.7k tokens, 37% within 8k, 60% within 16k
  (156-row sample). The build keeps traces up to 16k; `configs/sft_0.8b.yaml` trains on those
  within 8,192 (`max_completion_tokens`), the budget 0.8B is evaluated and RL-trained at.
- **Grader-verified traces only** (default; `--keep-unverified` disables): the trace's boxed
  answer must match `expected_answer` under `is_correct`. This drops ~57% of the length-OK rows
  (about 65% of StackExchange, 33% of AoPS), mostly free-form answers the grader cannot compare
  (text, several-part answers), plus some real mismatches. Overall yield ~23% of CoT rows, so
  100k traces need ~18 GB read.
- **Prompt/completion format.** The prompt is `render_prompt(problem, "think")`, as in eval and
  RL; loss on the completion only; TRL appends `<|im_end|>` (`eos_token`).
- **No packing.** TRL separates packed traces by `position_ids`, which only its Flash Attention
  path honors; the DeltaNet layers would carry state across traces. Batch size 1 (no padding)
  with gradient accumulation 16 instead. Loss is `chunked_nll`, not Liger.

- **Review + smoke test (2026-10-02).** Verified on real tokenized examples: no loss on the prompt,
  loss on reasoning + `</think>` + answer + final `<|im_end|>`; prompt/completion token boundary
  clean (0/50 mismatches); `chunked_nll` is TRL 1.14's default. 20-step smoke: 6.9 s/step (16
  traces, mean 4.2k tokens), peak 7.8 GiB allocated, loss ~0.55 / token accuracy ~82% from the
  start (0.8B-Base already predicts the teacher well; the job is mostly format and finishing).
- **Small build stats** (400 traces): 54% of kept traces fit the 8k budget (median 7.5k tokens);
  34% of rows dropped as "unverified". Sampled: mostly open-ended answers our grader cannot check
  (multi-line derivations, constructions, statements), plus some equivalent notations; keeping the
  filter aligns SFT with what RL rewards and eval measures. Grader limit found: `math_verify`
  reads `\log` as base 10, so `\log 2` != `\ln 2`.
- **Full run:** 50k-trace build (~27k within 8k) chosen over 100k: ~2-2.5 h build + ~3.3 h SFT.
- **Crash 2026-10-02 16:09 (first full attempt).** One build worker grew to 20 GB of RAM while
  checking traces; the 30 GB machine ran out of memory, VS Code's terminal host was killed, and the
  pipeline (inside VS Code's cgroup despite `setsid`) died with it. Culprit row not identified;
  likely pathological sympy arithmetic in the grader (sympy uses Python ints here, which the
  grader's signal timeouts cannot interrupt mid-operation). Fix: each worker caps its heap at its
  starting VmData + 2.5 GB (`RLIMIT_DATA`; a fixed cap fails because forked workers already hold
  ~2.7 GB of virtual data), and a `MemoryError` drops the row as `grader_memory` with its uuid
  logged. Long jobs now run in their own systemd scope (`systemd-run --user --scope`), so a VS
  Code cleanup cannot take them down.
- **Stall 2026-10-02 16:27-21:06 (second attempt).** The build froze at slice 120 (~10 min in)
  for 4.5 h: 6 workers at 100% CPU in endless sympy computations, one blocked on a lock. Root
  cause: math_verify's SIGALRM timeouts are lost when the alarm fires inside an object's `__del__`
  (Python swallows it: "Exception ignored in ... __del__"); this probably also explains the 20 GB
  worker. Results are consumed in order, so one stuck slice blocked everything while finished
  results piled up. Fixes in `build_sft_data.py`: a watchdog thread per worker re-raises a
  `RowTimeout` via SIGUSR1 every second once a row exceeds 30 s (SIGINT/`interrupt_main()` does not
  work: background jobs start with SIGINT ignored, and Python drops simulated SIGINTs); garbage
  collection is paused while a row is graded; at most 2 slices per worker are in flight. Tested:
  a never-ending computation and one that keeps swallowing interrupts are both cut at 30 s, and
  the 400-trace build is unchanged. `run_sft.sh` now line-buffers the build log so stalls show.
  The GRPO reward path uses the same math_verify timeouts; the same failure there would hang a
  training step rather than crash it.

### Precision bug: bf16 weights froze most parameters (found 2026-10-03)
All training configs loaded models with `model_init_kwargs: {dtype: bfloat16}`, so weights were
stored in bf16 and AdamW wrote updates straight into them. bf16 has 7 mantissa bits (~1/128
relative): a weight of 0.02 can only move in steps of ~1.6e-4, while an Adam step is ~lr (1e-5 for
SFT, 1e-6 for GRPO), so the update rounds away. In the first SFT checkpoint (layer-10 down_proj):
|w| < 1e-3 → 99.5% of elements changed; 1e-3-3e-3 → 54%; 3e-3-1e-2 → 2%; >= 1e-2 → 0%. All norms,
q/k norms, `A_log` and `dt_bias` (values ~1) were bit-identical. Training loss stayed ~0.596 for
1,600 steps; eval matched the Base model. Fix: `dtype: float32` (fp32 master weights) with
`bf16: true` (bf16 compute) in all three configs; the frozen distillation teacher stays bf16.
Checks: with fp32 every element moves (~3.4e-5 each, all magnitudes); overfitting 32 traces,
loss after 10 steps is 0.24 (fp32) vs 0.38 (bf16); SFT peak memory 12.7 GiB (was 7.8).
**Affected earlier results:** RL-Zero and the updates-per-rollout ablation ran with the bf16
setting (lr 1e-6), so they learned through only the smallest weights; their gains and conclusions
may understate RL. Rerun in fp32 later (GRPO memory will be tighter: fp32 weights + grads add ~3 GB).
Separately from the bug, the SFT data leaves little to learn: even in fp32 the real-data loss stays
~0.60 early on (0.8B-Base already predicts the teacher at 81% token accuracy), and the teacher's
"reason as long as needed" style runs past 8k on hard problems. Hence SFT v2's 4k-trace arm.

### SFT v2: fp32, <=8k traces (started 2026-10-03 11:01; now `scripts/run_sft.sh`)
Starts again from `Qwen/Qwen3.5-0.8B-Base` (not from RL-Zero: SFT is the main track's first
stage), on the 25.7k traces within 8,192 completion tokens, the eval budget every stage shares.
Then the think-mode eval, then the distillation chain (a one-off script, since removed) on `outputs/sft-0.8b-8k`. A 4k-trace arm
(12.2k traces) was started and stopped after a few minutes at the user's request; not run.
- **Training done 14:11 (3.1 h, peak 12.7 GiB).** Loss with fp32 weights: 0.604 (steps 1-100),
  *rising* to 0.634 around the peak lr (steps 200-300), then 0.595 at the end; eval loss 0.6198
  (step 500) → 0.5916 (final), vs 0.5929 for the bf16 run. So once updates actually apply, lr 1e-5
  first disrupts the model and the cosine decay recovers it, and the net token-level gain over
  the Base model is ~nil: these traces sit near the loss floor for 0.8B. Whether SFT changed the
  *behavior* that matters (finishing within 8k, format) is for the eval to show.
- **Eval OOM at startup (14:12), not related to the model.** The desktop's GPU use had grown to
  4.3 GB (one gnome-shell at 2.15 GB; 3.1 GB on 2026-10-01), and vLLM's 85% share no longer fit.
  `eval.py` now defaults to `--gpu-memory-utilization 0.75`; verified on the fp32 checkpoint
  (and the fp32 save itself is fine for vLLM). The same squeeze threatens distillation: its smoke
  test peaked at 26.1 GiB with a bf16 student, and the fp32 student adds ~3 GB.
- **Eval (think, 8k): unchanged from Base.** MATH-500 48.4 (Base 47.5, bf16 SFT 49.9), AIME
  2.7/1.2/1.2, AMC 21.9 (24.7), GPQA 10.7 (12.0); truncation MATH 33%, AIME 82-86% (Base 88-90%);
  MATH format 67%. All within noise. **Conclusion: SFT on these teacher traces does not change
  0.8B's behavior**, with or without the precision fix: the Base model already writes long
  chain-of-thought in this style (token accuracy 81% from the start), and the teacher's "reason as
  long as needed" habit teaches nothing about finishing within 8k. The main track continues with
  distillation from this checkpoint (effectively the Base model).
- **Distillation smoke test with the fp32 student OOMed** in its first backward pass (29.9 GiB
  allocated, 2.1 GiB more needed). Fix: the frozen teacher loads in 8-bit (bitsandbytes,
  `teacher_load_in_8bit: true`, ~4.8 GB instead of 9.3). Chain relaunched 15:17.
- That run completed step 1 and OOMed in step 2's backward (2.0 GiB needed, 3.5 GiB reserved but
  fragmented). Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set at the top of
  `distill.py` (vLLM's sleep-mode pool switches it off around its own allocations). **Smoke test
  passed 15:36:** losses 0.337/0.340/0.335, ~185 s/step, rollouts 6,959-7,671 tokens with 67-89%
  truncated at 8k; peak allocated reported 33.5 GiB (includes vLLM's pool), i.e. very little
  headroom. Full run (200 steps x 64 prompts, ~10 h) started 15:36 → `outputs/distill-0.8b/`,
  then think-mode eval. Watch rollout length / truncation: falling = learning the teacher's
  reasoning; rising = the teacher's long-thinking style winning.
- **Full run: student grew longer, then OOMed at step 34 (17:36), no checkpoint saved.** Steps
  1-20 / 21-33: loss 0.228 / 0.179, rollout length 7,921 / 8,076, truncated 94% / 97%, length of
  the few finished rollouts ~2.0-2.3k, entropy 0.68 / 0.62. The student matched the teacher
  better while drifting toward its long-thinking style: at nearly every position of an 8k rollout
  Qwen3.5-4B (trained to think 10k+ tokens) is still mid-reasoning, so reverse KL teaches "keep
  going", and with ~3% of rollouts finishing there is almost no signal about concluding. The OOM
  was a genuine shortage (2.51 GiB needed, 2.35 free, only 0.48 fragmented): full-length 8k
  sequences need more backward-pass activation memory than early, shorter rollouts. A planned
  stop-at-step-50 eval could not run. Distillation at an 8k budget from this teacher looks
  counterproductive for 0.8B; making it work needs both less memory (shorter rollouts / smaller
  teacher) and a fix for the length drift. The distillation gate now only checks that EOS works
(`clipped_ratio` <= 0.95 every step): the bf16 SFT student truncated 80-91% of T=1.0 rollouts,
which is genuine length, not an EOS bug. Memory risk: the earlier smoke test peaked at 26.1 GiB
with a bf16 student; an fp32 student adds ~3 GB. First (bf16) SFT result for reference, think 8k:
MATH-500 49.9, AIME 2.5/0.8/0.8, AMC 24.7, GPQA 11.5; truncation MATH 32%, AIME 84-87%
(`outputs/sft-0.8b-bf16`, `outputs/eval/outputs-sft-0.8b-bf16-think`).

### Stage 2: On-policy distillation (`scripts/distill.py`)
- `trl.DistillationTrainer`, `beta≈1.0` (reverse KL; mode-seeking, standard for on-policy KD).
- Student: stage-1 checkpoint. Teacher: Qwen3.5-4B thinking (must share the student's
  vocabulary, so the Nemotron teacher cannot be used here).
- Prompt-only math prompts (DeepMath-103K), decontaminated.
- vLLM colocate with `vllm_enable_sleep_mode`. If 4B teacher + 2B student + vLLM do not fit:
  LoRA student, quantized teacher, or `AsyncDistillationTrainer`.

Draft notes (2026-10-02; nothing run on the GPU or with model weights: the SFT pipeline had the
machine, with ~6 GB of RAM free):
- **Setup.** Student `outputs/sft-0.8b`, teacher `Qwen/Qwen3.5-4B` (same 248,320-token vocabulary;
  9.3 GB of weights, not downloaded yet). Prompts: the 84,104 DeepMath rows of `data/rl/pool.parquet`
  (already decontaminated), minus 8 whose prompt exceeds 1,024 tokens (max 1,782), which would not
  fit vLLM's 9,216-token window with an 8,192-token completion.
- **`enable_thinking: true` is required** (`chat_template_kwargs`). TRL renders the chat template
  itself, and the Base template then writes an empty `<think>\n\n</think>\n\n`. With the flag the
  prompt tokens equal `render_prompt(..., "think")` (2,000/2,000 checked), and the teacher's own
  template renders the same prompt. **The think-mode GRPO config (step 7) needs the same setting.**
- **One rollout = one optimizer step.** TRL's distillation has no `steps_per_generation`: it
  generates `per_device_train_batch_size x gradient_accumulation_steps` completions and takes one
  step on them. Config: 1 x 64, lr 5e-6 cosine, 200 steps (12,800 prompts); lr and steps are
  first guesses.
- **`TrimmedDistillationTrainer`** (`distill.py`): (1) TRL passes `logits_to_keep` to the backbone,
  which Qwen3.5's backbone forwards as an unknown keyword to its DeltaNet/attention kernels;
  dropped, as in GRPO's chunked path. Whether it actually fails with the Hub kernels was not tested.
  (2) TRL pads the whole rollout to its longest completion before splitting into micro-batches;
  each micro-batch is trimmed back, so batch size 1 runs unpadded.
- **To check in the smoke test:** memory (estimate: teacher 9.3 GB + vLLM ~9.8 GB + student during
  rollouts), time per step, and that rollouts end in `<|im_end|>` (`completions/clipped_ratio`).
- **Off-GPU review (2026-10-02 23:50).** Checked: config parses; TRL's distillation trainer uses
  the same `VLLMGeneration` as GRPO, so the text-only and V1-runner workarounds apply (the V1
  runner env var is set by importing `grpo`); SFT checkpoints save `<|im_end|>` as EOS (the
  script's assert holds); TRL renders the prompts identically to eval's think prompt (500/500);
  the subclass's overrides (`model_kwarg_keys`, `_compute_loss`, input keys) match TRL 1.14; the
  loss is computed per micro-batch with chunked JSD and there is no whole-rollout scoring pass (so
  GRPO's scoring OOM does not apply). Teacher downloaded. Not checkable without the GPU: memory,
  the `logits_to_keep` workaround with Hub kernels, step time, rollouts ending on `<|im_end|>`.
- **Queued chain** (a one-off script `run_distill_chain.sh`, removed 2026-10-04; own systemd scope): waits for the SFT
  pipeline, runs 3 steps into `outputs/smoke-distill/`, and starts the full run only if the smoke
  test exits cleanly, saves a model, logs 3 finite losses and has `clipped_ratio` <= 0.5 in every
  step. Otherwise it stops and writes the reasons to `outputs/smoke-distill/FAILED`.
- **Risk: teacher verbosity.** The post-trained Qwen3.5 models think far past 8k tokens (2B: ~12.5k
  on MATH-500). Matching the 4B's token distributions may pull the student toward longer reasoning
  and undo part of what SFT taught about finishing within 8k. Watch `completions/mean_length` and
  `clipped_ratio` during training, and truncation in the eval.

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

### Updates-per-rollout ablation (step 8, done; launcher removed 2026-10-04, arm-B overrides in the `configs/grpo_rlzero_0.8b.yaml` header)
With one optimizer step per rollout (our default, and the GRPO paper's: "the policy model only has
a single update following each exploration stage"), the probability ratio is exactly 1, so PPO-style
clipping, and with it DAPO's clip-higher, never triggers (clip ratio 0 in every logged step). DAPO
took 16 updates per rollout (512 prompts x 16 responses, mini-batch 512). Arms, all at the same
rollout budget of 100 rollouts x 512 samples:

| arm | updates/rollout | clip (low/high) | output |
|---|---|---|---|
| A | 1 | inactive | main run, step-100 checkpoint copied to `outputs/rlzero-0.8b-step100` |
| B | 4 | 0.2 / 0.28 | `outputs/rlzero-0.8b-4upd-cliphigher` |
| ~~C~~ | 4 | 0.2 / 0.2 | dropped (see below) |

4 updates/rollout = `gradient_accumulation_steps: 16`, `steps_per_generation: 64` (TRL then keeps
generation-time log-probs, so ratios move off 1 on updates 2-4); `max_steps: 400`, warmup 40 and
saves every 100 to stay matched per rollout. Same learning rate, so B/C also take 4x as many
optimizer steps per rollout (inherent to the comparison, as in DAPO). Questions: learning per
rollout, stability (entropy, reward), and whether clip-higher keeps entropy up. Each arm is
evaluated with the standard zero-mode eval; arm A also at step 200. ~13 h for B + C after the
main run; queue log `outputs/ablation.log`.

**Arm C dropped (2026-10-02).** Even with 4 updates per rollout, arm B clips only ~0.008% of
tokens (max 0.09% in any step): at lr 1e-6 four updates move the 0.8B policy too little for ratios
to reach 0.8 / 1.28, so clip-higher vs symmetric clipping cannot differ here. A more informative
follow-up, if wanted: 1 update/rollout at lr 4e-6, to separate "reusing samples" from "moving the
weights 4x further per rollout". Early signal: arm B's training reward at rollouts 21-24 was 21.5%
vs ~18.5% for arm A at the same point.

**Result (2026-10-02).** Zero-mode eval, avg@k %:

| | rollouts | MATH-500 | AIME 24/25/26 | AMC23 | GPQA | MATH format | MATH truncated | MATH tokens |
|---|---|---|---|---|---|---|---|---|
| A, 1 update/rollout | 100 | 34.8 | 0.8 / 0.2 / 0.4 | 16.9 | 22.3 | 92% | 7% | 1,163 |
| B, 4 updates/rollout | 100 | 37.5 | 1.9 / 0.0 / 0.6 | 15.0 | 25.9 | 96% | 4% | 941 |
| A, 1 update/rollout | 200 | 37.9 | 1.5 / 0.4 / 0.2 | 17.5 | 22.2 | 96% | 4% | 887 |

B after 100 rollouts matches A after 200 (MATH-500, format, truncation, length; GPQA/AMC
differences are within noise), in ~6.5 h instead of ~12.7 h. Training curves show the same
trajectory traversed ~2x faster per rollout, entropy included, so B is faster, not better at the
end; without the 4x-lr control, "more weight movement per rollout" and "sample reuse" are not
separated. Clipping stayed negligible (~0.009% of tokens). **Decision: use 4 updates per rollout
(`gradient_accumulation_steps: 16`, `steps_per_generation: 64`) for later RL runs.**

### RL-Zero result: Qwen3.5-0.8B-Base, 200 steps (2026-10-02)
Eval (zero mode, 8k, standard protocol, avg@k %):

| | MATH-500 | AIME 24/25/26 | AMC23 | GPQA | MATH format | MATH truncated | MATH tokens |
|---|---|---|---|---|---|---|---|
| 0.8B-Base | 35.5 | 1.2 / 0.8 / 0.2 | 16.9 | 17.8 | 79% | 17% | 2,124 |
| step 100 | 34.8 | 0.8 / 0.2 / 0.4 | 16.9 | 22.3 | 92% | 7% | 1,163 |
| step 200 | 37.9 | 1.5 / 0.4 / 0.2 | 17.5 | 22.2 | 96% | 4% | 887 |

Training (25-step blocks): reward 17.7% → 31.1%, truncation 34% → 9%, length 1,386 → 1,027,
entropy 1.02 → 0.58. Reading: RL-Zero mainly taught finishing cleanly and concisely; the large
training-reward gain (at T=1.0, where the Base model rambles and loops) transfers only modestly to
the eval protocol (T=0.6 + presence penalty already suppresses much of that), and the 1,913
prompts were seen ~3.3 times. MATH-500 +2.4 and GPQA +4.4 are ~1.5x their noise; AIME/AMC flat.
pass@k did not rise (AIME pass@16 fell), consistent with RLVR sharpening existing ability more
than adding new solutions; entropy is declining steadily and needs watching in longer runs.
Clipping never triggered (1 update/rollout: ratio exactly 1).

### Main-track RL (step 7, `configs/grpo_main_0.8b.yaml`, started 2026-10-03)
Distillation was skipped (see Stage 2). GRPO in think mode from the SFT checkpoint (behaves like
the Base model), with these changes from RL-Zero:
- **8k completion budget** (RL-Zero: 4k): the student truncates ~33% of MATH-500 answers at 8k
  and 80%+ of T=1.0 rollouts; at 4k almost nothing would earn reward. Overlong penalty ramps over
  the last 2,048 tokens.
- **256 completions per rollout** (16 prompts x 16; RL-Zero: 512): rollouts are ~5x longer.
- **Micro-batches of 2** (RL-Zero: 8): activation memory at 9k tokens per sequence.
- **Truncated completions not masked** (RL-Zero: masked): with 80%+ truncated, masking would drop
  most samples and never penalize running past the budget; unmasked, a truncated rollout gets 0
  correctness and format and -1 overlong penalty, i.e. a negative advantage.
- **Same 1,913 prompts as RL-Zero** (pass rate strictly between 0 and 1 for 0.8B-Base): re-scoring
  at 8k would cost hours, and the comparison with RL-Zero stays direct.
- fp32 master weights, 4 updates per rollout, `enable_thinking: true`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (now also set in `grpo.py`).
- **Smoke test passed (2026-10-03 20:37-20:56):** 2 rollouts = 17.3 min, i.e. **~8.7 min per
  rollout** (~5.5 min generation + 4 updates x ~53 s); peak ~23 GB on the GPU. Rollout 1 / 2:
  correctness 25.8% / 21.9% (RL-Zero started at 17%), format 28.9% / 26.2%, truncated 71% / 74%,
  mean length 6,736 / 6,629, overlong penalty mean -0.73 / -0.75, 0% all-equal groups; clipping
  ~0.009% of tokens on updates 2-4.

- **Full run (2026-10-03 23:21 → 10-04 08:31, 100 rollouts, ~5.5 min/rollout on average as
  rollouts shortened).** Training, rollouts 1-5 → 96-100: correctness reward 21.8% → 60.9%,
  format 30% → 97%, truncated 70% → 3%, mean length 6,631 → 2,101; entropy steady ~0.75 (no
  collapse); 0% all-equal groups throughout; 1,600 prompts used, under one pass over 1,913.
- **Eval (think, 8k):** MATH-500 49.3 (SFT 48.4, Base 47.5), AIME 2.1/0.8/0.0, AMC 23.8, GPQA 26.0
  (Base 12.0); MATH truncation 5% (33%), format 95% (67%), mean tokens 1,872 (3,842); pass@k
  unchanged (MATH-500 67.4). Beats the post-trained Qwen3.5-2B at 8k on every benchmark (that model
  truncates 60% of MATH answers at 8k). **Reading:** RL taught finishing concisely within budget,
  with no loss of accuracy, but did not raise math accuracy: the problems the model used to
  truncate on were mostly ones it could not solve. GPQA's jump is mostly from finishing: 26% is at
  chance for 4-option questions, while the Base model scored below chance because 70% of its
  answers truncated before a letter. Across stages, only RL changed 0.8B's behavior, and none
  raised its reasoning accuracy, which points to model capacity (consistent with the budget sweep:
  extra tokens help 2B, not 0.8B). Next: scale to 2B, whose truncated reasoning is productive.

### Main-track RL on 2B (step 9, prepared 2026-10-04, `configs/grpo_main_2b.yaml`)
Same recipe as 0.8B (think mode, 8k, 16 prompts x 16, truncation penalized, DAPO loss), with these
changes:
- **8 updates per rollout of 32 completions** (0.8B: 4 of 64): LoRA is less tolerant of large
  batches (TRL's LoRA Without Regret guide: effective batch < 32). 100 rollouts = 800 optimizer
  steps; warmup 80 and checkpoints every 80 steps (10 rollouts each).
- **From Qwen3.5-2B-Base**, no SFT (SFT left 0.8B unchanged).
- **LoRA r=32, alpha=32 on all 186 language-model linear layers** (attention q/k/v/o, DeltaNet
  in_proj_qkv/z/a/b and out_proj, MLP gate/up/down; not vision, not lm_head): 33.6M trainable
  params in fp32 on a frozen bf16 base. Full fine-tuning with fp32 master weights (~20 GB of
  weights/grads/Adam) does not fit next to vLLM. LoRA on all layers matches full fine-tuning in RL
  ("LoRA Without Regret", Thinking Machines 2025); lr 1e-5 = 10x the full-fine-tuning lr, per the
  same work. fp32 Adam (`adamw_torch_fused`) since only the adapters are optimized.
- **Non-destructive LoRA weight sync** (`lora_safe_weight_sync` in `grpo.py`): TRL merges adapters
  into the base weights in place and unmerges after each vLLM sync, which with a bf16 base rounds
  the base on every sync (once per rollout, 100 times per run). The override yields W + delta-W per layer computed in
  fp32 and rounded once, never touching the base. Tested on the real 2B (CPU): 186 LoRA layers,
  none in vision/lm_head; after a sync with non-zero adapters the base is bit-identical and each
  synced weight equals W + delta-W. At the end, the adapter is saved to `outputs/rl-2b/adapter` and
  the merged model to `outputs/rl-2b`.
- **Prompts re-scored for 2B** (think mode, 8k, k=8, 3,000 prompts → `data/rl/rl-2b.parquet`): the
  0.8B set would be largely too easy for 2B (no gradient from all-correct groups).
- Not yet verified on the GPU: memory, the LoRA sync with vLLM, step time (rough guess ~2x 0.8B).

- **Prompt scoring (15:09-18:43):** 2B-Base think mode at 8k on 3,000 prompts: pass rate 0 for
  38.8%, 1-3/8 26.4%, 4-7/8 25.7%, 8/8 9.1%; mean 33.1%, 53.9% truncated, mean 6,007 tokens. Kept
  1,565 (1,366 DeepMath, 199 DAPO) → `data/rl/rl-2b.parquet`.
- **Smoke test passed (18:43-19:08):** rollouts 1/2: correctness 52.7% / 39.5%, truncated 30% /
  48%, length 4,775 / 6,002, entropy 0.75 / 0.86, 0% all-equal groups. ~387-401 s generation + 7
  updates x ~39.5 s ≈ **11 min per rollout**. LoRA weight sync verified: the vLLM-vs-trainer log-prob
  gap is the same before and after the 8 updates (mean 0.0117 at both rollouts), so vLLM generates
  from the updated weights.
- **Importance-sampling finding → switched to `token_truncate`.** TRL's default `sequence_mask`
  weights each completion by exp(sum of per-token log-prob gaps vLLM vs trainer) and drops weights
  above 3; there is no lower bound. Over 5-7k tokens the bf16 noise sums to weights far above 3
  (dropped) or far below 1 (near zero): mean weight 0.24-0.33 in the 0.8B main-track run and
  0.30-0.37 in the 2B smoke test, vs 0.86-0.92 for RL-Zero's ~1.4k-token completions. So the 0.8B
  main track trained on roughly a quarter to a third of its effective sample weight (it still
  worked). `token_truncate` caps each token's ratio at 3 instead (truncated IS, Yao et al. 2025;
  standard for long CoT in verl). The first full 2B run was stopped at step 8 and moved to
  `outputs/rl-2b-aborted-seqmask/`; the config now sets `vllm_importance_sampling_mode:
  token_truncate`. Restart: `scripts/run_rl_2b.sh` (skips the finished scoring and smoke test).

### TRL truncation check vs stop strings: RL-Zero masked finished answers (found 2026-10-03)
TRL counts a completion as truncated if its last token is not EOS/pad (`grpo_trainer.py` ~2353,
2540). vLLM drops a matched stop *string* from the output, so a zero-mode completion that finished
by reaching `"\nUser:"` ends on an ordinary token and is counted as truncated. Test: 64/64
zero-mode samples stopped on `"\nUser:"`, none ending in EOS. With RL-Zero's
`mask_truncated_completions: true`, every such finished answer got no gradient: true truncation
on its prompts was 16.4% vs TRL's `clipped_ratio` 36-38% at the start, so ~20% of all RL-Zero
completions (complete answers, many correct) were silently excluded. Think mode is unaffected:
completions end on `<|im_end|>`, which is the SFT tokenizer's EOS and kept in the output (64/64
counted correctly), so the main track's `clipped_ratio` is real truncation. Fix needed before
re-running RL-Zero: e.g. count `finish_reason == "stop"` as finished, or drop `"\nUser:"` as a
stop string and handle extra turns in the reward.

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
pyproject.toml, uv.lock        # uv project
PLAN.md, README.md             # plan/findings, results
src/posttrain/
  prompts.py                   # think / zero prompt templates + stop strings
  rewards.py                   # correctness, format, overlong penalty (training and eval)
  eval.py                      # vLLM eval, --regrade
  data.py                      # RL pool, decontamination, GRPO / SFT dataset formats
configs/                       # TrlParser YAMLs (script options + trainer config)
  sft_0.8b.yaml                # SFT, fp32 weights, traces <= 8k
  grpo_rlzero_0.8b.yaml        # RL-Zero (header notes how the recorded run differed)
  grpo_main_0.8b.yaml          # main-track RL, think mode, 8k
  grpo_main_2b.yaml            # main-track RL on 2B with LoRA
  distill_0.8b.yaml            # on-policy distillation (tried, stopped)
scripts/
  check_env.py                 # environment + kernel check
  run_baselines.sh             # stage-0 evals (resumable)
  run_followups.sh             # 8k ceiling + output-budget sweep
  build_sft_data.py, sft.py    # SFT data + trainer
  run_sft.sh                   # build -> SFT -> eval
  filter_by_passrate.py        # RL prompts with pass rate strictly between 0 and 1
  grpo.py                      # GRPO for both tracks (Qwen3.5 / TRL workarounds)
  run_rl_main.sh               # main-track RL -> eval (resumable; CONFIG/NAME/BASE_REPO select the size)
  run_rl_2b.sh                 # 2B: score prompts -> gated smoke test -> run_rl_main.sh
  distill.py, run_distill.sh   # on-policy distillation (tried, stopped)
tests/                         # test_rewards.py, test_eval.py, test_data.py
data/rl/                       # pool.parquet, rlzero-0.8b(-scored).parquet
data/sft/                      # nemotron-{train,val}.parquet
outputs/                       # final models, evals (outputs/eval/), logs; not in git
```

## Risks / open points
- Qwen3.5 in vLLM colocate with weight sync and sleep mode during GRPO (hybrid architecture,
  multimodal checkpoint). Fallbacks: `vllm_mode="server"` on the same GPU, or Qwen3 dense models.
- Memory for distillation (4B teacher + 2B student + vLLM).
- Nemotron trace length vs student capacity (see stage 1).
- Background jobs started from a Claude session die when the session ends; long runs are
  launched with `setsid nohup` inside their own `systemd-run --user --scope`, since a VS Code
  memory-pressure cleanup killed a `setsid` job on 2026-10-02.
- **The grader can blow up memory** (see the SFT crash). The build is now capped, but GRPO computes
  rewards in the trainer process: a model output that triggers the same pathology could OOM a run.
  Not seen in ~300 GRPO steps so far; if it happens, run reward grading in a capped subprocess.

## Housekeeping log
- 2026-09-30: old-protocol outputs, protocol-ablation scratch runs and `scripts/rerun_gpqa.sh`
  deleted; their numbers are kept in the protocol section above.
- 2026-10-01: stage-0 complete; eval changes and scripts committed (`a568fa3`); results added to
  README.md.
- 2026-10-04: cleanup. Removed one-off launchers (`run_sft_v2.sh`, merged into `run_sft.sh`;
  `run_distill_chain.sh`, `run_distill_step50_eval.sh`, `run_update_ablation.sh`) and
  `data/sft-smoke/`. Deleted ~102 GB of intermediate checkpoints (optimizer states) and
  smoke/debug model files; kept every run's final model, all evals and all logs. The bf16 SFT
  model (effectively Base) was deleted; its log and eval remain.
