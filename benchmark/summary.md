# Benchmark Results — Enterprise Migration Copilot

Generated: 2026-06-29 02:56
Test set: 480 scripts (stratified — 120 per language, all 4 difficulties)
Metrics: `syntax_valid` AND `has_pyspark_ops` AND `semantic_sim` (60% table coverage)

## Leaderboard

| Model | SQL | HiveQL | PL/SQL | SP | Overall |
|---|---|---|---|---|---|
| DeepSeek-1.3B-base | 0% | 0% | 0% | 0% | 0% |
| DeepSeek-1.3B-finetuned | 29% | 32% | 27% | 20% | 27% |
| Qwen2.5-1.5B-base | 0% | 0% | 0% | 0% | 0% |
| Qwen2.5-1.5B-finetuned | 48% | 61% | 50% | 22% | 45% |
| Phi-3.5-mini-base | 0% | 0% | 0% | 0% | 0% |
| Phi-3.5-mini-finetuned ⭐ | 64% | 74% | 57% | 32% | 57% |

## Fine-tuning Improvement

| Family | Base | Fine-tuned | Delta |
|---|---|---|---|
| Deepseek | 0.0% | 27.1% | +27.1% |
| Qwen | 0.0% | 45.2% | +45.2% |
| Phi | 0.0% | 56.9% | +56.9% |

## Per-Difficulty Breakdown (fine-tuned models only)

| Model | Easy | Medium | Hard | Expert |
|---|---|---|---|---|
| DeepSeek-1.3B-finetuned | 32% | 28% | 29% | 11% |
| Qwen2.5-1.5B-finetuned | 47% | 52% | 36% | 45% |
| Phi-3.5-mini-finetuned | 72% | 57% | 46% | 47% |

## Per-Metric Breakdown (fine-tuned models only)

| Model | Syntax Valid | DF Ops | Semantic Align |
|---|---|---|---|
| DeepSeek-1.3B-finetuned | 33% | 87% | 88% |
| Qwen2.5-1.5B-finetuned | 51% | 94% | 91% |
| Phi-3.5-mini-finetuned | 66% | 88% | 88% |

## Best Model

**Phi-3.5-mini-finetuned** achieves the highest overall pass rate: **56.9%** on 480 held-out scripts.

## Metric Definitions

- `syntax_valid` — generated PySpark passes `ast.parse()`, no syntax errors
- `has_pyspark_ops` — at least 2 DataFrame operations (`.select`, `.filter`, `.groupBy`, etc.)
- `semantic_sim` — at least 60% of source table names appear in generated output
- `overall` — all 3 metrics must pass simultaneously
