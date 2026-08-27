# ArmBench-LLM full-run submission: Gemma 4 31B IT

This is a standalone result submission for a complete ArmBench-LLM run. It is intentionally separate from proposed changes to the ArmBench task definition and the LiteLLM/OpenRouter runner.

## Submitted result

| Field | Value |
|---|---|
| Model | `google/gemma-4-31b-it` |
| OpenRouter model | `openrouter/google/gemma-4-31b-it` |
| Provider | DeepInfra, pinned; fallbacks disabled |
| Reasoning | Disabled |
| Temperature | 0 |
| Seed | 0 |
| ArmBench examples | 2,928 / 2,928 with nonblank outputs |
| Overall ArmBench score | **0.6141561129879723** (display: **0.6142**) |
| Runtime | 5,662.62 seconds |
| Recorded cost | approximately $0.3869 |

The cost is marked approximate because the auxiliary request ledger used request-derived keys; identical requests can overwrite earlier ledger entries. This does not affect the saved 2,928 sample outputs or benchmark score.

## Category scores

| Category | Score |
|---|---:|
| NER | 0.7168503431 |
| POS | 0.8000000000 |
| Reading | 0.5842618319 |
| Classification | 0.6428571429 |
| MCQA | 0.8733333333 |
| Generation | 0.1086606355 |
| Translation | 0.2838959010 |
| Exams | 0.5000000000 |
| Text processing | 0.8369071472 |
| MMLU-Pro | 0.7947947948 |

The overall score is the unweighted mean of the ten ArmBench category scores.

## Reproducibility

- Metric-AI LightEval commit: `2e22157f66c25a476fd2e5ebf6f75d84dd6efd5e`
- ArmBench dataset revision: `c1a7672da0b615de6d8437c74e4aa4c2f1a681f3`
- Configuration: `armbench_gemma4_31b_nothink.yaml`
- Dependency snapshot: `armbench_requirements_lock.txt`
- The exact uncommitted safety/reproducibility changes used by the run are included in `runner.patch.gz`.
- Standard LightEval aggregate JSON and all 24 per-task Parquet detail files are included under `artifacts/`.

No API key or other credential is included.
