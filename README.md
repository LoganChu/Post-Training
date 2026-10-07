# posttrain

Math-reasoning post-training experiments on Qwen3.5 small models with TRL, on a single RTX 5090
(32 GB). Main track: Base → SFT → on-policy distillation → RL with verifiable rewards; ablations
include RL-Zero (RL directly on Base) and DPO before RL. See [PLAN.md](PLAN.md) for the full plan,
status and findings.

## Setup

```
uv sync
uv run scripts/check_env.py --use-kernels   # GPU, versions, fast DeltaNet kernels, fwd/bwd test
uv run --group dev pytest tests              # reward + eval unit tests
```

## Reproduce

Each launcher is resumable (finished stages are skipped) and logs to `outputs/`:

```
scripts/run_baselines.sh                     # stage-0 evals (Base models, post-trained 2B)
scripts/run_followups.sh                     # 8k ceiling + output-budget sweep
uv run -m posttrain.data build               # RL prompt pool -> data/rl/pool.parquet
uv run scripts/filter_by_passrate.py --model Qwen/Qwen3.5-0.8B-Base --mode zero --name rlzero-0.8b --n-prompts 6000
uv run scripts/grpo.py --config configs/grpo_rlzero_0.8b.yaml   # RL-Zero (see the config header)
scripts/run_sft.sh                           # SFT data build -> SFT -> eval
scripts/run_rl_main.sh                       # main-track RL from the SFT model -> eval
scripts/run_rl_2b.sh                         # 2B (LoRA): score prompts -> smoke test -> RL -> eval
scripts/run_distill.sh                       # on-policy distillation (tried, stopped; see PLAN.md)
```

## Evaluation

```
uv run -m posttrain.eval --model <hf-id-or-path> --mode <think|zero> [--max-tokens 8192]
uv run -m posttrain.eval --model <...> --mode <...> --regrade   # re-score saved samples, no generation
```

- `think`: Qwen chat template with thinking on (prompt ends in `<think>\n`). `zero`: plain-text
  R1-Zero-style prompt for Base models.
- Protocol for every model: T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, seed 0 (why: PLAN.md,
  Evaluation → Protocol). Samples per problem: MATH-500 4, AIME 16, AMC23 8, GPQA-Diamond 4.
- Grading uses the same `is_correct` as the RL reward (`src/posttrain/rewards.py`).

## Results

### Stage 0: baselines (avg@k, %)

| Model | Mode | Budget | MATH-500 | AIME24 | AIME25 | AIME26 | AMC23 | GPQA-D |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | zero | 8k | 35.5 | 1.2 | 0.8 | 0.2 | 16.9 | 17.8 |
| Qwen3.5-0.8B-Base | think | 8k | 47.5 | 1.7 | 1.0 | 0.8 | 24.7 | 12.0 |
| Qwen3.5-2B-Base | zero | 8k | 61.1 | 6.7 | 7.1 | 7.3 | 37.5 | 23.9 |
| Qwen3.5-2B-Base | think | 8k | 72.2 | 13.5 | 12.3 | 10.6 | 46.2 | 24.5 |
| Qwen3.5-2B (post-trained) | think | 8k | 39.0 | 0.0 | 0.0 | 0.0 | 16.2 | 0.5 |
| Qwen3.5-2B (post-trained) | think | 32k | 66.8 | 13.3 | 18.3 | 12.5 | 41.9 | 24.2 |

The post-trained rows use reduced k (1 for MATH-500 and GPQA, 4 for AIME and AMC). At 32k, 82-88%
of its AIME samples still truncate (the model card recommends 81,920 tokens for competition math),
so this row understates Qwen's reported numbers. At our 8k budget it trails its own Base model on
every benchmark: it was trained to think far longer than 8k allows.

### Output-budget sweep (Base, think mode)

MATH-500 = first 100 problems (k=4); AIME 2025 k=16.

| Model | Budget | MATH-500 | AIME25 avg@16 | AIME25 pass@16 | AIME25 truncated |
|---|---|---|---|---|---|
| 2B-Base | 8k | 76.2 | 12.3 | 30.0 | 80% |
| 2B-Base | 16k | 83.0 | 16.2 | 40.0 | 70% |
| 2B-Base | 32k | 80.8 | 19.6 | 53.3 | 64% |
| 0.8B-Base | 8k | 55.5 | 1.0 | 6.7 | 89% |
| 0.8B-Base | 16k | 57.8 | 1.2 | 6.7 | 81% |
| 0.8B-Base | 32k | 56.8 | 1.7 | 10.0 | 71% |

2B-Base turns extra tokens into AIME accuracy; 0.8B-Base mostly does not (truncation falls,
accuracy stays flat).

### Main track on Qwen3.5-0.8B (think mode, 8k, avg@k %)

| Stage | MATH-500 | AIME24 | AIME25 | AIME26 | AMC23 | GPQA-D | MATH truncated | MATH tokens |
|---|---|---|---|---|---|---|---|---|
| Base | 47.5 | 1.7 | 1.0 | 0.8 | 24.7 | 12.0 | 33% | 3,810 |
| SFT (Nemotron traces <= 8k, fp32) | 48.4 | 2.7 | 1.2 | 1.2 | 21.9 | 10.7 | 33% | 3,842 |
| SFT + RL (GRPO, 100 rollouts) | 49.3 | 2.1 | 0.8 | 0.0 | 23.8 | 26.0 | 5% | 1,872 |

On-policy distillation from Qwen3.5-4B was tried and stopped (the student drifted toward the
teacher's long reasoning; see PLAN.md). RL taught the model to finish within budget (truncation
33% → 5%, half the tokens) without changing math accuracy; GPQA's rise is mostly from answers now
finishing (26% is chance level for 4 options).

