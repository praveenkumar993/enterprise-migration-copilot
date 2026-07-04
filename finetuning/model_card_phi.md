---
language:
- en
license: mit
tags:
- pyspark
- sql
- code-generation
- migration
- fintech
- fine-tuned
base_model: microsoft/Phi-3.5-mini-instruct
---

# migration-copilot-phi-3-5-mini-instruct

Fine-tuned [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) for enterprise SQL/HiveQL/PL-SQL/Stored Procedure → PySpark migration.

Part of the [Enterprise Migration Copilot](https://github.com/praveenkumar993/enterprise-migration-copilot) project.

## Benchmark Results

Evaluated on 480 held-out scripts (120 per language), never seen during training:

| Language | Pass Rate |
|---|---|
| SQL | 64% |
| HiveQL | 74% |
| PL/SQL | 57% |
| Stored Procedure | 32% |
| **Overall** | **57%** |

Metrics: `syntax_valid` AND `has_pyspark_ops` AND `semantic_sim` (60% table name coverage).

**Best fine-tuned model** across all 3 models trained in this project.

## Training Details

- **Base model**: microsoft/Phi-3.5-mini-instruct (3.8B parameters)
- **Method**: QLoRA / LoRA fine-tuning (r=16, alpha=32)
- **Training data**: 1,312 validated SQL→PySpark pairs
- **Data sources**: 300 hand-crafted Claude examples (99.67% validation pass rate) + 1,012 Ollama-generated pairs (79.58% pass rate)
- **Languages**: SQL, HiveQL, PL/SQL, Stored Procedures (T-SQL)
- **Epochs**: 3
- **Train loss**: 2.133 → (high due to chat template format mismatch, but eval loss converged well)
- **Eval loss**: 0.294 (best across all 3 models)
- **Hardware**: Google Colab T4 GPU (free tier)
- **Training time**: ~54 minutes

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

**Input (PL/SQL):**
```sql
DECLARE
  CURSOR c_emp IS
    SELECT emp_id, salary FROM employees WHERE dept_id = 10;
BEGIN
  FOR rec IN c_emp LOOP
    DBMS_OUTPUT.PUT_LINE(rec.emp_id || ': ' || rec.salary);
  END LOOP;
END;
```

**Output (PySpark):**
```python
from pyspark.sql import functions as F

df = spark.table('employees')
emp_df = df.filter(F.col('dept_id') == 10).select('emp_id', 'salary')
for row in emp_df.collect():
    print(f"{row['emp_id']}: {row['salary']}")
```

## Intended Use

- Enterprise legacy SQL migration to Apache Spark / Databricks
- Fintech data pipeline modernization
- Batch migration tooling and code review assistance

## Limitations

- Stored Procedure migration (T-SQL) has lower accuracy (32%) due to complex procedural constructs
- Expert-difficulty recursive CTEs and dynamic SQL may require manual review
- Model runs via HuggingFace Inference API — cold start latency ~15-30s on free tier

## Links

- 📁 GitHub: [enterprise-migration-copilot](https://github.com/praveenkumar993/enterprise-migration-copilot)
- 📊 Dataset: [praveends/enterprise-migration-dataset](https://huggingface.co/datasets/praveends/enterprise-migration-dataset)
- 🤗 All models: [praveends](https://huggingface.co/praveends)