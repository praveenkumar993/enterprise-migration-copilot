"""
Optimizer Agent — Pure Python, no LLM.
Detects anti-patterns in generated PySpark code and suggests fixes.
"""

import re
from typing import Any


ANTI_PATTERNS = [
    {
        "pattern": r"\.collect\(\)",
        "name": "collect() on potentially large DataFrame",
        "suggestion": "Use df.write.parquet() or df.show() instead of collect() to avoid OOM on large datasets",
        "severity": "high",
    },
    {
        "pattern": r"crossJoin\(",
        "name": "Cartesian join detected",
        "suggestion": "Add a join condition to avoid cartesian product explosion",
        "severity": "high",
    },
    {
        "pattern": r"\.repartition\(\d{3,}\)",
        "name": "Very large repartition",
        "suggestion": "Repartition to a number close to your cluster cores, not an arbitrary large number",
        "severity": "medium",
    },
    {
        "pattern": r"for\s+\w+\s+in\s+.+:\s*\n.*\.collect\(\)",
        "name": "Python for loop with collect()",
        "suggestion": "Replace loop + collect() with DataFrame transformation — use df.withColumn() or df.agg()",
        "severity": "high",
    },
    {
        "pattern": r"udf\(",
        "name": "UDF usage detected",
        "suggestion": "Check if a built-in PySpark function (F.upper, F.regexp_replace, etc.) can replace this UDF",
        "severity": "medium",
    },
    {
        "pattern": r"select\(\s*[\"\']\*[\"\']\s*\)",
        "name": "SELECT * usage",
        "suggestion": "Explicitly select required columns to avoid schema drift and improve performance",
        "severity": "low",
    },
    {
        "pattern": r"for\s+\w+\s+in\s+\w+",
        "name": "Python for loop detected",
        "suggestion": "Replace Python for loop with DataFrame transformation — use df.withColumn() or df.agg() instead of iterating rows",
        "severity": "medium",
    },
]

OPTIMIZATION_SUGGESTIONS = [
    {
        "trigger_pattern": r"\.join\(.+\).*\.join\(.+\).*\.join\(.+\)",
        "suggestion": "Multiple JOINs detected — consider broadcasting smaller lookup tables with F.broadcast(df)",
    },
    {
        "trigger_pattern": r"spark\.read\.table\(.+\)[\s\S]{0,200}spark\.read\.table\(.+\)[\s\S]{0,200}spark\.read\.table\(.+\)",
        "suggestion": "Multiple table reads — cache frequently accessed DataFrames with .cache()",
    },
    {
        "trigger_pattern": r"\.groupBy\(",
        "suggestion": "After groupBy().agg(), consider .cache() if this result is used multiple times downstream",
    },
]


def optimize(pyspark_code: str) -> dict[str, Any]:
    """
    Detect anti-patterns in PySpark code and suggest optimizations.

    Pure Python — no LLM. Rule-based detection.

    Args:
        pyspark_code: Generated PySpark code string

    Returns:
        Optimization result with anti_patterns, suggestions, and score
    """
    if not pyspark_code or not pyspark_code.strip():
        return {
            "anti_patterns": [],
            "suggestions": [],
            "optimization_score": 1.0,
            "issues_found": 0,
            "high_severity_count": 0,
        }

    detected_anti_patterns = []
    suggestions = []

    # Check anti-patterns
    for ap in ANTI_PATTERNS:
        if re.search(ap["pattern"], pyspark_code, re.MULTILINE | re.IGNORECASE):
            detected_anti_patterns.append({
                "name": ap["name"],
                "suggestion": ap["suggestion"],
                "severity": ap["severity"],
            })

    # Check optimization suggestions
    for opt in OPTIMIZATION_SUGGESTIONS:
        if re.search(opt["trigger_pattern"], pyspark_code, re.DOTALL):
            if opt["suggestion"] not in suggestions:
                suggestions.append(opt["suggestion"])

    # Calculate optimization score
    high_count = sum(1 for ap in detected_anti_patterns if ap["severity"] == "high")
    medium_count = sum(1 for ap in detected_anti_patterns if ap["severity"] == "medium")
    low_count = sum(1 for ap in detected_anti_patterns if ap["severity"] == "low")

    penalty = (high_count * 0.20) + (medium_count * 0.10) + (low_count * 0.05)
    optimization_score = max(0.0, round(1.0 - penalty, 3))

    return {
        "anti_patterns": detected_anti_patterns,
        "suggestions": suggestions,
        "optimization_score": optimization_score,
        "issues_found": len(detected_anti_patterns),
        "high_severity_count": high_count,
    }