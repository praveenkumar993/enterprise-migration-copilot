"""
Few-shot prompt templates for HiveQL bulk generation via Ollama.
Uses delimiter-based output format instead of raw JSON.
"""

HIVEQL_GENERATION_PROMPT = """You are generating training data for a HiveQL to PySpark migration model.
Study this example conversion carefully, then generate exactly 1 NEW, DIFFERENT example
in the exact same style and difficulty level.

EXAMPLE:
{few_shot_examples}

Now generate exactly 1 NEW HiveQL to PySpark migration example at {difficulty} difficulty.
Use HiveQL-specific features where appropriate: LATERAL VIEW EXPLODE, DISTRIBUTE BY, SORT BY, CLUSTER BY, partitioned tables.
pyspark_code must use SparkSession-based modern syntax (spark.table, spark.read), NOT SparkContext or SQLContext.
Map DISTRIBUTE BY to repartition(), SORT BY to sortWithinPartitions(), LATERAL VIEW EXPLODE to F.explode().
Use a realistic business table name like orders, customers, products, transactions, or user_events.

Respond using EXACTLY this format with these exact markers. Do not use JSON. Do not use markdown code fences.
Write plain HiveQL and plain Python code between the markers, nothing else.

===FEATURES===
comma separated list of HiveQL concepts used, like: LATERAL VIEW EXPLODE, DISTRIBUTE BY

===SOURCE_CODE===
the raw HiveQL query here, multiple lines allowed

===PYSPARK_CODE===
the raw PySpark Python code here, multiple lines allowed

===END===
"""

HIVEQL_TOPICS_BY_DIFFICULTY = {
    "easy": [
        "basic SELECT with WHERE", "simple GROUP BY", "ORDER BY with LIMIT",
        "DISTINCT values", "basic table scan",
    ],
    "medium": [
        "INNER JOIN", "LEFT OUTER JOIN", "GROUP BY with HAVING",
        "partitioned table query", "date functions",
    ],
    "hard": [
        "LATERAL VIEW EXPLODE", "DISTRIBUTE BY and SORT BY", "CLUSTER BY",
        "window functions with partitioning", "bucketed table joins",
    ],
    "expert": [
        "multi-level partitioning", "nested LATERAL VIEW EXPLODE", "dynamic partition insert",
        "multi-insert from single source", "skew data handling",
    ],
}