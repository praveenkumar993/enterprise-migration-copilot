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
base_model: deepseek-ai/deepseek-coder-1.3b-instruct
---

# migration-copilot-deepseek-coder-1-3b-instruct

Fine-tuned [deepseek-coder-1.3b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) for enterprise SQL/HiveQL/PL-SQL/Stored Procedure → PySpark migration.

Part of the [Enterprise Migration Copilot](https://github.com/praveenkumar993/enterprise-migration-copilot) project.

## Benchmark Results

Evaluated on 480 held-out scripts (120 per language), never seen during training:

| Language | Pass Rate |
|---|---|
| SQL | 29% |
| HiveQL | 32% |
| PL/SQL | 27% |
| Stored Procedure | 20% |
| **Overall** | **27%** |

Metrics: `syntax_valid` AND `has_pyspark_ops` AND `semantic_sim` (60% table name coverage).

## Training Details

- **Base model**: deepseek-ai/deepseek-coder-1.3b-instruct (1.3B parameters)
- **Method**: LoRA fine-tuning (r=16, alpha=32)
- **Training data**: 1,312 validated SQL→PySpark pairs
- **Data sources**: 300 hand-crafted Claude examples (99.67% validation pass rate) + 1,012 Ollama-generated pairs (79.58% pass rate)
- **Languages**: SQL, HiveQL, PL/SQL, Stored Procedures (T-SQL)
- **Epochs**: 3
- **Train loss**: 0.347 → 0.258 (best training convergence across all 3 models)
- **Eval loss**: 0.363 → 0.329
- **Hardware**: Google Colab T4 GPU (free tier)
- **Training time**: ~19 minutes (fastest training due to smallest model size)

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

**Input (HiveQL):**
```sql
SELECT user_id, tag
FROM user_tags
LATERAL VIEW EXPLODE(tags) t AS tag
WHERE size(tags) > 0
DISTRIBUTE BY user_id
SORT BY tag;
```

**Output (PySpark):**
```python
from pyspark.sql import functions as F

result = (
    user_tags
    .filter(F.size('tags') > 0)
    .select('user_id', F.explode('tags').alias('tag'))
    .repartition('user_id')
    .sortWithinPartitions('tag')
)
result.show()
```

## Intended Use

- Enterprise legacy SQL migration to Apache Spark / Databricks
- Lightweight/edge deployment (1.3B params, smallest memory footprint of the 3 models)
- Fast prototyping and experimentation

## Limitations

- Lowest overall accuracy (27%) due to smallest base model size
- Best training loss convergence (0.258) but smallest base model capacity limits output quality
- Stored Procedure migration accuracy is 20% — complex T-SQL constructs require manual review

## Model Comparison

| Model | Params | Overall | Train Loss | Eval Loss |
|---|---|---|---|---|
| **deepseek-coder-1.3b** (this) | 1.3B | 27% | 0.258 | 0.329 |
| qwen2.5-coder-1.5b | 1.5B | 45% | 0.307 | 0.344 |
| phi-3.5-mini | 3.8B | 57% | 2.133 | 0.294 |

## Links

- 📁 GitHub: [enterprise-migration-copilot](https://github.com/praveenkumar993/enterprise-migration-copilot)
- 📊 Dataset: [praveends/enterprise-migration-dataset](https://huggingface.co/datasets/praveends/enterprise-migration-dataset)
- 🤗 All models: [praveends](https://huggingface.co/praveends)