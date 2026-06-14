"""
HiveQL Parser — extends SQL parser with Hive-specific feature detection.
Uses sqlglot hive dialect as base, then overlays custom detection.
"""

import re
from typing import Any

from parsers.sql_parser import parse_sql
from utils.ir_builder import compute_complexity, validate_ir


# HiveQL-specific keywords that sqlglot may not fully capture
HIVEQL_KEYWORDS = {
    "DISTRIBUTE BY": "DISTRIBUTE BY detected — maps to df.repartition()",
    "SORT BY": "SORT BY detected — maps to df.sortWithinPartitions()",
    "CLUSTER BY": "CLUSTER BY detected — maps to df.repartition().sortWithinPartitions()",
    "LATERAL VIEW EXPLODE": "LATERAL VIEW EXPLODE detected — maps to df.withColumn(F.explode())",
    "LATERAL VIEW OUTER": "LATERAL VIEW OUTER detected — maps to df.withColumn(F.explode()) with nulls",
    "TABLESAMPLE": "TABLESAMPLE detected — maps to df.sample()",
    "TRANSFORM": "HIVE TRANSFORM detected — use pandas UDF or mapPartitions",
}


def parse_hiveql(source: str) -> dict[str, Any]:
    """
    Parse a HiveQL string and return a unified IR dict.

    First runs sqlglot with hive dialect, then adds
    HiveQL-specific keyword detection on top.

    Args:
        source: Raw HiveQL string

    Returns:
        IR dict with source_language set to hiveql
    """
    # Run base SQL parsing with hive dialect
    ir = parse_sql(source, dialect="hive")

    # Override source language
    ir["source_language"] = "hiveql"
    ir["dialect"] = "hive"

    if not source or not source.strip():
        return validate_ir(ir)

    source_upper = source.upper()

    # Detect HiveQL-specific keywords and add to parse notes
    # We store them in parse_errors with INFO prefix to surface in the IR
    hive_notes = []
    for keyword, note in HIVEQL_KEYWORDS.items():
        if keyword in source_upper:
            hive_notes.append(note)

    # LATERAL VIEW EXPLODE maps to a special column extraction
    if "LATERAL VIEW" in source_upper and "EXPLODE" in source_upper:
        # Try to extract the exploded column name
        match = re.search(
            r"LATERAL\s+VIEW\s+(?:OUTER\s+)?EXPLODE\s*\(\s*(\w+)\s*\)\s+\w+\s+AS\s+(\w+)",
            source_upper
        )
        if match:
            array_col = match.group(1).lower()
            alias_col = match.group(2).lower()
            if array_col not in ir["columns"]:
                ir["columns"].append(array_col)
            if alias_col not in ir["columns"]:
                ir["columns"].append(alias_col)

    # DISTRIBUTE BY columns
    dist_match = re.search(r"DISTRIBUTE\s+BY\s+([\w\s,]+?)(?:SORT|ORDER|LIMIT|$)", source_upper)
    if dist_match:
        dist_cols = [c.strip().lower() for c in dist_match.group(1).split(",")]
        for col in dist_cols:
            if col and col not in ir["columns"]:
                ir["columns"].append(col)

    # Add hive notes to parse_errors with INFO prefix
    for note in hive_notes:
        ir["parse_errors"].append(f"INFO: {note}")

    ir = compute_complexity(ir)
    ir = validate_ir(ir)
    return ir