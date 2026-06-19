"""
Few-shot prompt templates for SQL bulk generation via Ollama.
Uses delimiter-based output format instead of raw JSON to avoid
small-model JSON-escaping failures with multi-line code strings.
"""

SQL_GENERATION_PROMPT = """You are generating training data for a SQL to PySpark migration model.
Study this example conversion carefully, then generate exactly 1 NEW, DIFFERENT example
in the exact same style and difficulty level.

EXAMPLE:
{few_shot_examples}

Now generate exactly 1 NEW SQL to PySpark migration example at {difficulty} difficulty.
Use a realistic business table name like orders, customers, products, transactions, or employees.
pyspark_code must use SparkSession-based modern syntax (spark.table, spark.read), NOT SparkContext or SQLContext.
pyspark_code must use pyspark.sql.functions imported as F for transformations.

Respond using EXACTLY this format with these exact markers. Do not use JSON. Do not use markdown code fences.
Write plain SQL and plain Python code between the markers, nothing else.

===FEATURES===
comma separated list of SQL concepts used, like: SELECT, WHERE clause, aggregation

===SOURCE_CODE===
the raw SQL query here, multiple lines allowed

===PYSPARK_CODE===
the raw PySpark Python code here, multiple lines allowed

===END===
"""

SQL_TOPICS_BY_DIFFICULTY = {
    "easy": [
        "basic SELECT with WHERE", "COUNT and SUM aggregation", "ORDER BY and LIMIT",
        "DISTINCT values", "simple column projection", "basic string functions",
    ],
    "medium": [
        "INNER JOIN with WHERE", "LEFT JOIN with NULL check", "GROUP BY with HAVING",
        "CASE WHEN expressions", "date functions and filtering", "multiple aggregations",
    ],
    "hard": [
        "window functions with PARTITION BY", "CTE with multiple joins", "subqueries in WHERE",
        "self-joins", "running totals", "ranking functions",
    ],
    "expert": [
        "recursive CTEs", "multiple window functions combined", "PIVOT operations",
        "correlated subqueries", "GROUPING SETS", "complex date/time analytics",
    ],
}