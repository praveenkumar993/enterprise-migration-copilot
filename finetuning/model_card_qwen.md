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
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
---

# migration-copilot-qwen2-5-coder-1-5b-instruct

Fine-tuned [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) for enterprise SQL/HiveQL/PL-SQL/Stored Procedure → PySpark migration.

Part of the [Enterprise Migration Copilot](https://github.com/praveenkumar993/enterprise-migration-copilot) project.

## Benchmark Results

Evaluated on 480 held-out scripts (120 per language), never seen during training:

| Language | Pass Rate |
|---|---|
| SQL | 48% |
| HiveQL | 61% |
| PL/SQL | 50% |
| Stored Procedure | 22% |
| **Overall** | **45%** |

Metrics: `syntax_valid` AND `has_pyspark_ops` AND `semantic_sim` (60% table name coverage).

## Training Details

- **Base model**: Qwen/Qwen2.5-Coder-1.5B-Instruct (1.5B parameters)
- **Method**: LoRA fine-tuning (r=16, alpha=32)
- **Training data**: 1,312 validated SQL→PySpark pairs
- **Data sources**: 300 hand-crafted Claude examples (99.67% validation pass rate) + 1,012 Ollama-generated pairs (79.58% pass rate)
- **Languages**: SQL, HiveQL, PL/SQL, Stored Procedures (T-SQL)
- **Epochs**: 3
- **Train loss**: 0.387 → 0.307
- **Eval loss**: 0.390 → 0.344
- **Hardware**: Google Colab T4 GPU (free tier)
- **Training time**: ~21 minutes

## Prompt Format

```
### Instruction:
Convert the following {SOURCE_LANGUAGE} code to PySpark.
Difficulty: {difficulty}

### Input:
{source_code}

### Response:
```

## Example

**Input (SQL):**
```sql
SELECT customer_id, SUM(amount) AS total
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total DESC;
```

**Output (PySpark):**
```python
from pyspark.sql import functions as F

df = spark.table('orders')
result = (
    df.filter(F.col('status') == 'completed')
      .groupBy('customer_id')
      .agg(F.sum('amount').alias('total'))
      .orderBy(F.col('total').desc())
)
result.show()
```

## Intended Use

- Enterprise legacy SQL migration to Apache Spark / Databricks
- Fintech data pipeline modernization
- Lightweight deployment (1.5B params, lower memory footprint)

## Limitations

- Lower overall accuracy (45%) vs Phi-3.5-mini (57%) due to smaller base model
- Stored Procedure migration accuracy is 22% — complex T-SQL constructs require manual review

## Links

- 📁 GitHub: [enterprise-migration-copilot](https://github.com/praveenkumar993/enterprise-migration-copilot)
- 📊 Dataset: [praveends/enterprise-migration-dataset](https://huggingface.co/datasets/praveends/enterprise-migration-dataset)
- 🤗 All models: [praveends](https://huggingface.co/praveends)