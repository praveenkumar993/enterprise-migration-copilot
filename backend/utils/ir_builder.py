"""
IR Builder — Unified Intermediate Representation
Every source language produces the same IR structure before hitting the agents.
Adding a new source language = new parser only, not a pipeline change.
"""

from typing import Any


def build_empty_ir() -> dict[str, Any]:
    """Return a blank IR with all fields initialized to safe defaults."""
    return {
        "source_language": "",
        "dialect": "",
        "tables": [],
        "columns": [],
        "joins": [],
        "aggregations": [],
        "window_functions": [],
        "ctes": [],
        "subquery_depth": 0,
        "has_udf": False,
        "complexity_score": 1,
        "complexity_label": "simple",
        "procedural_flags": [],
        "raw_source": "",
        "parse_errors": [],
    }


def compute_complexity(ir: dict[str, Any]) -> dict[str, Any]:
    """
    Score complexity 1-4 based on features detected in the IR.

    1 = simple     — basic SELECT, no joins, no aggregations
    2 = moderate   — joins OR aggregations, no window functions
    3 = complex    — window functions OR subqueries OR CTEs
    4 = expert     — procedural flags OR UDFs OR subquery_depth > 2
    """
    score = 1

    has_joins = len(ir.get("joins", [])) > 0
    has_aggs = len(ir.get("aggregations", [])) > 0
    has_windows = len(ir.get("window_functions", [])) > 0
    has_ctes = len(ir.get("ctes", [])) > 0
    has_subqueries = ir.get("subquery_depth", 0) > 0
    has_udf = ir.get("has_udf", False)
    has_procedural = len(ir.get("procedural_flags", [])) > 0

    if has_joins or has_aggs:
        score = 2
    if has_windows or has_ctes or has_subqueries:
        score = 3
    if has_udf or has_procedural or ir.get("subquery_depth", 0) > 2:
        score = 4

    labels = {1: "simple", 2: "moderate", 3: "complex", 4: "expert"}
    ir["complexity_score"] = score
    ir["complexity_label"] = labels[score]
    return ir


def merge_ir_fields(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two IRs together — used by PL/SQL and SP parsers
    when multiple SQL blocks are extracted from one procedure.
    Lists are combined and deduplicated.
    Numeric fields take the max value.
    """
    list_fields = [
        "tables", "columns", "joins", "aggregations",
        "window_functions", "ctes", "procedural_flags", "parse_errors"
    ]
    for field in list_fields:
        combined = base.get(field, []) + extra.get(field, [])
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for item in combined:
            key = str(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        base[field] = deduped

    base["subquery_depth"] = max(
        base.get("subquery_depth", 0),
        extra.get("subquery_depth", 0)
    )
    base["has_udf"] = base.get("has_udf", False) or extra.get("has_udf", False)
    return base


def validate_ir(ir: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all required IR fields exist and are the correct type.
    Adds a parse_error if a required field is missing.
    """
    required_fields = {
        "source_language": str,
        "dialect": str,
        "tables": list,
        "columns": list,
        "joins": list,
        "aggregations": list,
        "window_functions": list,
        "ctes": list,
        "subquery_depth": int,
        "has_udf": bool,
        "complexity_score": int,
        "complexity_label": str,
        "procedural_flags": list,
        "raw_source": str,
        "parse_errors": list,
    }

    for field, expected_type in required_fields.items():
        if field not in ir:
            ir.setdefault("parse_errors", []).append(
                f"Missing required IR field: {field}"
            )
            # Set safe default
            ir[field] = expected_type()
        elif not isinstance(ir[field], expected_type):
            ir.setdefault("parse_errors", []).append(
                f"Field {field} expected {expected_type.__name__}, "
                f"got {type(ir[field]).__name__}"
            )

    return ir