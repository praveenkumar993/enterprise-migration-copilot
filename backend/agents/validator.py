"""
Validator Agent — Pure Python, no LLM.
Validates generated PySpark code against IR expectations.
"""

import ast
import re
from typing import Any


# DataFrame operations that must appear in valid PySpark output
DATAFRAME_OPS = [
    r"\.select\(",
    r"\.filter\(",
    r"\.where\(",
    r"\.groupBy\(",
    r"\.join\(",
    r"\.agg\(",
    r"\.withColumn\(",
    r"\.orderBy\(",
    r"\.union\(",
    r"\.distinct\(",
    r"\.drop\(",
    r"\.limit\(",
    r"\.repartition\(",
    r"\.cache\(",
    r"spark\.read",
    r"spark\.sql\(",
]

# Procedural review markers the model should include
REVIEW_MARKERS = [
    r"#.*manual",
    r"#.*review",
    r"#.*TODO",
    r"#.*NOTE",
    r"#.*FLAG",
]


def check_syntax(code: str) -> tuple[bool, str]:
    """
    Check Python syntax using ast.parse.
    Returns (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_sparksession(code: str) -> bool:
    """Check if code references spark (SparkSession) or df."""
    return bool(
        re.search(r"\bspark\b", code) or
        re.search(r"\bdf\b", code) or
        re.search(r"SparkSession", code)
    )


def check_dataframe_ops(code: str) -> bool:
    """Check if code contains at least one DataFrame operation."""
    for pattern in DATAFRAME_OPS:
        if re.search(pattern, code):
            return True
    return False


def check_tables_match(ir: dict[str, Any], code: str) -> bool:
    """
    Check if tables from IR appear in the PySpark code.
    At least 50% of detected tables should be referenced.
    """
    tables = ir.get("tables", [])
    if not tables:
        return True  # No tables to check

    code_lower = code.lower()
    found = sum(1 for t in tables if t.lower() in code_lower)
    return found >= max(1, len(tables) * 0.5)


def check_procedural_flags_surfaced(ir: dict[str, Any], code: str) -> bool:
    """
    If IR has procedural flags, check that PySpark code contains
    at least one review comment surfacing them.
    """
    flags = ir.get("procedural_flags", [])
    if not flags:
        return True  # No flags to surface — auto-pass

    code_lower = code.lower()
    for pattern in REVIEW_MARKERS:
        if re.search(pattern, code_lower, re.IGNORECASE):
            return True

    return False


def validate(ir: dict[str, Any], pyspark_code: str) -> dict[str, Any]:
    """
    Validate generated PySpark code against IR expectations.

    Pure Python — deterministic checks only.

    Args:
        ir: Unified IR dict from parser
        pyspark_code: Generated PySpark code string

    Returns:
        Validation result dict with score and issues list
    """
    issues = []

    if not pyspark_code or not pyspark_code.strip():
        return {
            "syntax_valid": False,
            "has_sparksession": False,
            "has_dataframe_ops": False,
            "detected_tables_match": False,
            "procedural_flags_surfaced": False,
            "validation_score": 0.0,
            "issues": ["Empty PySpark code returned"],
            "passed": False,
        }

    # Check 1 — Syntax
    syntax_valid, syntax_error = check_syntax(pyspark_code)
    if not syntax_valid:
        issues.append(f"Syntax error: {syntax_error}")

    # Check 2 — SparkSession reference
    has_sparksession = check_sparksession(pyspark_code)
    if not has_sparksession:
        issues.append("No SparkSession (spark) or DataFrame (df) reference found")

    # Check 3 — DataFrame operations
    has_dataframe_ops = check_dataframe_ops(pyspark_code)
    if not has_dataframe_ops:
        issues.append("No DataFrame operations found (.select, .filter, .groupBy, etc.)")

    # Check 4 — Tables match
    detected_tables_match = check_tables_match(ir, pyspark_code)
    if not detected_tables_match:
        tables = ir.get("tables", [])
        issues.append(f"Tables from IR not found in PySpark code: {', '.join(tables)}")

    # Check 5 — Procedural flags surfaced
    procedural_flags_surfaced = check_procedural_flags_surfaced(ir, pyspark_code)
    has_procedural = len(ir.get("procedural_flags", [])) > 0
    if has_procedural and not procedural_flags_surfaced:
        issues.append(
            "Procedural flags not surfaced as comments in PySpark code — "
            "add # NOTE or # TODO comments for manual review items"
        )

    # --- Validation score ---
    score = 0.0
    score += 0.30 if syntax_valid else 0.0
    score += 0.20 if has_sparksession else 0.0
    score += 0.25 if has_dataframe_ops else 0.0
    score += 0.15 if detected_tables_match else 0.0

    if has_procedural:
        score += 0.10 if procedural_flags_surfaced else 0.0
    else:
        # No procedural flags — redistribute that 0.10 to tables check
        score += 0.10 if detected_tables_match else 0.0

    passed = score >= 0.60

    return {
        "syntax_valid": syntax_valid,
        "has_sparksession": has_sparksession,
        "has_dataframe_ops": has_dataframe_ops,
        "detected_tables_match": detected_tables_match,
        "procedural_flags_surfaced": procedural_flags_surfaced,
        "validation_score": round(score, 3),
        "issues": issues,
        "passed": passed,
    }