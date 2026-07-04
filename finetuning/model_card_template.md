# {{MODEL_NAME}}

Fine-tuned {{BASE_MODEL}} for enterprise SQL/HiveQL/PL-SQL/Stored Procedure → PySpark migration.

Part of the [Enterprise Migration Copilot](https://github.com/praveenkumar993/enterprise-migration-copilot) project.

## Benchmark Results

Evaluated on 480 held-out scripts (120 per language), never seen during training:

| Language | Pass Rate |
|---|---|
| SQL | {{SQL_PASS_RATE}} |
| HiveQL | {{HIVEQL_PASS_RATE}} |
| PL/SQL | {{PLSQL_PASS_RATE}} |
| Stored Procedure | {{SP_PASS_RATE}} |
| **Overall** | **{{OVERALL_PASS_RATE}}** |

Metrics: `syntax_valid` AND `has_pyspark_ops` AND `semantic_sim` (60% table name coverage).

## Training Details

- **Base model**: {{BASE_MODEL}}
- **Parameters**: {{PARAM_COUNT}}
- **Method**: LoRA fine-tuning (r=16, alpha=32)
- **Training data**: 1,312 validated SQL→PySpark pairs
- **Data sources**: 300 hand-crafted Claude examples + 1,012 Ollama-generated pairs
- **Languages**: SQL, HiveQL, PL/SQL, Stored Procedures (T-SQL)
- **Epochs**: 3
- **Train loss**: {{TRAIN_LOSS_START}} → {{TRAIN_LOSS_END}}
- **Eval loss**: {{EVAL_LOSS_START}} → {{EVAL_LOSS_END}}
- **Hardware**: Google Colab T4 GPU (free tier)
- **Training time**: {{TRAINING_TIME}}

## Prompt Format

This model expects the exact fine-tuning prompt format:

```
### Instruction:
Convert the following {SOURCE_LANGUAGE} code to PySpark.
Difficulty: {difficulty}

### Input:
{source_code}

### Response:
```

## Example

**Input ({{EXAMPLE_LANGUAGE}}):**
```sql
{{EXAMPLE_SOURCE_CODE}}
```

**Output (PySpark):**
```python
{{EXAMPLE_PYSPARK_CODE}}
```

## Intended Use

- Enterprise legacy SQL migration to Apache Spark / Databricks
- Fintech data pipeline modernization
- Batch migration tooling and code review assistance

## Limitations

- Stored Procedure migration (T-SQL) has lower accuracy due to complex procedural constructs
- Expert-difficulty recursive CTEs and dynamic SQL may require manual review
- Model runs via HuggingFace Inference API — cold start latency ~15-30s on free tier

## Links

- 📁 GitHub: [enterprise-migration-copilot](https://github.com/praveenkumar993/enterprise-migration-copilot)
- 📊 Dataset: [praveends/enterprise-migration-dataset](https://huggingface.co/datasets/praveends/enterprise-migration-dataset)
- 🤗 All models: [praveends](https://huggingface.co/praveends)