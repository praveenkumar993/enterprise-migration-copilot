"""
Few-shot prompt templates for Stored Procedure (T-SQL) bulk generation via Ollama.
Uses delimiter-based output format instead of raw JSON.
"""

SP_GENERATION_PROMPT = """You are generating training data for a T-SQL Stored Procedure to PySpark migration model.
Study this example conversion carefully, then generate exactly 1 NEW, DIFFERENT example
in the exact same style and difficulty level.

EXAMPLE:
{few_shot_examples}

Now generate exactly 1 NEW T-SQL Stored Procedure to PySpark migration example at {difficulty} difficulty.
Use realistic T-SQL constructs where appropriate: CREATE PROCEDURE, temp tables (#table), MERGE, TRY/CATCH, cursors, OUTPUT parameters.
pyspark_code must use SparkSession-based modern syntax (spark.table, spark.read), NOT SparkContext or SQLContext.
Map temp tables to createOrReplaceTempView, MERGE to Delta Lake merge, TRY/CATCH to try/except.
Use a realistic business table name like Orders, Customers, Inventory, or Employees.

Respond using EXACTLY this format with these exact markers. Do not use JSON. Do not use markdown code fences.
Write plain T-SQL and plain Python code between the markers, nothing else.

===FEATURES===
comma separated list of T-SQL concepts used, like: temp tables, MERGE statement

===SOURCE_CODE===
the raw T-SQL stored procedure here, multiple lines allowed

===PYSPARK_CODE===
the raw PySpark Python code here, multiple lines allowed

===END===
"""

SP_TOPICS_BY_DIFFICULTY = {
    "easy": [
        "basic SELECT procedure", "procedure with input parameter", "TOP N query",
        "simple aggregation", "ORDER BY procedure",
    ],
    "medium": [
        "JOIN with GROUP BY", "OUTPUT parameters", "IF EXISTS UPDATE INSERT pattern",
        "window functions in procedure", "multiple OUTPUT parameters",
    ],
    "hard": [
        "temp tables with MERGE", "TRY CATCH with transactions", "cursor with WHILE loop",
        "dynamic SQL with sp_executesql", "batched DELETE with @@ROWCOUNT",
    ],
    "expert": [
        "recursive CTE hierarchy", "dynamic PIVOT", "deadlock-safe retry logic",
        "SCD Type 2 pattern", "graph traversal with temp tables",
    ],
}