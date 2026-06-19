"""
Few-shot prompt templates for PL/SQL bulk generation via Ollama.
Uses delimiter-based output format instead of raw JSON.
"""

PLSQL_GENERATION_PROMPT = """You are generating training data for a PL/SQL to PySpark migration model.
Study this example conversion carefully, then generate exactly 1 NEW, DIFFERENT example
in the exact same style and difficulty level.

EXAMPLE:
{few_shot_examples}

Now generate exactly 1 NEW PL/SQL to PySpark migration example at {difficulty} difficulty.
Use realistic Oracle PL/SQL constructs where appropriate: CURSOR, BULK COLLECT, EXCEPTION handlers, DBMS_OUTPUT, stored procedures/functions.
pyspark_code must use SparkSession-based modern syntax (spark.table, spark.read), NOT SparkContext or SQLContext.
Map cursors to DataFrame iteration or vectorized operations, EXCEPTION to try/except, DBMS_OUTPUT to Python logging.
Use a realistic business table name like employees, orders, accounts, or departments.

Respond using EXACTLY this format with these exact markers. Do not use JSON. Do not use markdown code fences.
Write plain PL/SQL and plain Python code between the markers, nothing else.

===FEATURES===
comma separated list of PL/SQL concepts used, like: CURSOR, EXCEPTION handling

===SOURCE_CODE===
the raw PL/SQL block here, multiple lines allowed

===PYSPARK_CODE===
the raw PySpark Python code here, multiple lines allowed

===END===
"""

PLSQL_TOPICS_BY_DIFFICULTY = {
    "easy": [
        "basic SELECT in PL/SQL block", "simple WHERE filter", "basic aggregation",
        "ORDER BY", "column projection",
    ],
    "medium": [
        "INNER JOIN in SELECT", "stored function with RETURN", "trigger AFTER INSERT",
        "CASE expression", "date arithmetic functions",
    ],
    "hard": [
        "explicit CURSOR with FETCH loop", "BULK COLLECT INTO", "EXCEPTION handling with WHEN OTHERS",
        "stored procedure with OUT parameter", "nested PL/SQL blocks",
    ],
    "expert": [
        "nested cursors", "FORALL with bulk DML", "package specification and body",
        "dynamic SQL with EXECUTE IMMEDIATE", "PRAGMA AUTONOMOUS_TRANSACTION",
    ],
}