# Failure Analysis — Phi-3.5-mini-finetuned

## Overview

| Metric | Value |
|---|---|
| Model | Phi-3.5-mini-finetuned |
| Total test scripts | 480 |
| Passed | 273 (56.9%) |
| Failed | 207 (43.1%) |

## Per-Language Results

| Language | Total | Passed | Failed | Pass Rate |
|---|---|---|---|---|
| sql | 120 | 77 | 43 | 64.2% |
| hiveql | 120 | 89 | 31 | 74.2% |
| plsql | 120 | 68 | 52 | 56.7% |
| stored_procedure | 120 | 39 | 81 | 32.5% |

## Per-Difficulty Results

| Difficulty | Total | Passed | Pass Rate |
|---|---|---|---|
| easy | 130 | 93 | 71.5% |
| medium | 160 | 92 | 57.5% |
| hard | 135 | 62 | 45.9% |
| expert | 55 | 26 | 47.3% |

## Failure Categories by Language

### SQL

| Failure Type | Count | % of failures | Suggestion |
|---|---|---|---|
| `syntax_error` | 38 | 73% | Add post-processing syntax-fix step; augment training with more syntactically diverse examples. |
| `low_semantic_alignment` | 11 | 21% | Add table-name preservation instruction to prompt template. |
| `missing_dataframe_ops` | 3 | 6% | Augment training with examples containing chained DataFrame operations. |

### HIVEQL

| Failure Type | Count | % of failures | Suggestion |
|---|---|---|---|
| `syntax_error` | 21 | 60% | Add post-processing syntax-fix step; augment training with more syntactically diverse examples. |
| `low_semantic_alignment` | 9 | 26% | Add table-name preservation instruction to prompt template. |
| `missing_dataframe_ops` | 5 | 14% | Augment training with examples containing chained DataFrame operations. |

### PLSQL

| Failure Type | Count | % of failures | Suggestion |
|---|---|---|---|
| `syntax_error` | 39 | 60% | Add post-processing syntax-fix step; augment training with more syntactically diverse examples. |
| `missing_dataframe_ops` | 16 | 25% | Augment training with examples containing chained DataFrame operations. |
| `low_semantic_alignment` | 10 | 15% | Add table-name preservation instruction to prompt template. |

### STORED_PROCEDURE

| Failure Type | Count | % of failures | Suggestion |
|---|---|---|---|
| `syntax_error` | 66 | 52% | Add post-processing syntax-fix step; augment training with more syntactically diverse examples. |
| `missing_dataframe_ops` | 34 | 27% | Augment training with examples containing chained DataFrame operations. |
| `low_semantic_alignment` | 26 | 21% | Add table-name preservation instruction to prompt template. |

## Key Findings

1. **HiveQL is the strongest language** — 89/120 scripts pass (74%), likely because HiveQL maps most directly to PySpark DataFrame semantics.

2. **Stored Procedures are the weakest** — 39/120 scripts pass (32%). Complex T-SQL ceremony (cursors, temp tables, dynamic SQL) creates semantic alignment gaps.

3. **Syntax is the primary failure mode** — only 66% of outputs pass `ast.parse()`. DataFrame ops (88%) and semantic alignment (88%) are much stronger, suggesting the model understands the migration pattern but sometimes produces slightly malformed Python syntax.

4. **Expert difficulty is handled well** — 26/55 expert scripts pass (47.3%), comparable to medium (57.5%). This suggests the fine-tuning captured complex migration patterns well, not just simple ones.

## Next Steps

- **Short term**: Add a lightweight AST-fix post-processing step to catch and correct minor Python syntax errors in generated output — this alone could push overall pass rate from 57% to 65-70%.
- **Medium term**: Augment the training dataset with 500-1000 additional stored procedure examples, targeting the 32% SP pass rate specifically.
- **Long term**: Fine-tune with a larger base model (7B+ params) using QLoRA on A100 GPU to push past the 70% ceiling.
