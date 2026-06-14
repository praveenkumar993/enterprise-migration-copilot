"""
PL/SQL Parser — extracts SQL blocks and procedural metadata from Oracle PL/SQL.
Uses sqlglot for embedded SQL blocks, regex for procedural constructs.
"""

import re
from typing import Any

from parsers.sql_parser import parse_sql
from utils.ir_builder import (
    build_empty_ir, compute_complexity,
    merge_ir_fields, validate_ir
)


# Procedural flag patterns — things that cannot be auto-converted
PROCEDURAL_FLAGS = {
    r"\bCURSOR\s+\w+\s+IS\b":
        "CURSOR loop detected — manual review required",
    r"\bFOR\s+\w+\s+IN\s+\(":
        "FOR loop detected — consider replacing with DataFrame transformation",
    r"\bWHILE\b.*\bLOOP\b":
        "WHILE loop detected — consider replacing with DataFrame transformation",
    r"\bEXCEPTION\b.*\bWHEN\b":
        "EXCEPTION handler detected — add try/except in PySpark job",
    r"\bBULK\s+COLLECT\b":
        "BULK COLLECT detected — maps to df.collect() but check memory usage",
    r"\bFORALL\b":
        "FORALL detected — maps to DataFrame batch write",
    r"\bEXECUTE\s+IMMEDIATE\b":
        "EXECUTE IMMEDIATE detected — dynamic SQL, manual review required",
    r"\bDBMS_OUTPUT\b":
        "DBMS_OUTPUT detected — replace with Python logging",
    r"\bDBMS_\w+\b":
        "DBMS package detected — no direct PySpark equivalent, manual review required",
    r"\bUTL_\w+\b":
        "UTL package detected — no direct PySpark equivalent, manual review required",
    r"\bPIPELINED\b":
        "PIPELINED function detected — consider mapPartitions in PySpark",
    r"\bOBJECT\s+TYPE\b":
        "Oracle OBJECT TYPE detected — map to PySpark StructType",
    r"\bVARRAY\b":
        "VARRAY detected — map to PySpark ArrayType",
    r"\bNESTED\s+TABLE\b":
        "NESTED TABLE detected — map to PySpark ArrayType",
}

# Patterns to extract embedded SQL blocks from PL/SQL
SQL_BLOCK_PATTERNS = [
    # SELECT INTO
    r"(SELECT\s+.+?\s+INTO\s+.+?(?:FROM\s+.+?)(?:WHERE\s+.+?)?(?:GROUP\s+BY\s+.+?)?(?:ORDER\s+BY\s+.+?)?);",
    # Plain SELECT
    r"(SELECT\s+.+?FROM\s+.+?(?:WHERE\s+.+?)?(?:GROUP\s+BY\s+.+?)?(?:ORDER\s+BY\s+.+?)?);",
    # INSERT
    r"(INSERT\s+INTO\s+.+?(?:VALUES\s*\(.+?\)|SELECT\s+.+?FROM\s+.+?));",
    # UPDATE
    r"(UPDATE\s+\w+\s+SET\s+.+?(?:WHERE\s+.+?)?);",
    # DELETE
    r"(DELETE\s+FROM\s+\w+(?:\s+WHERE\s+.+?)?);",
    # MERGE
    r"(MERGE\s+INTO\s+.+?USING\s+.+?ON\s+.+?(?:WHEN\s+.+?)+);",
]

# PL/SQL variable type patterns
PLSQL_TYPES = r"\b(\w+)\s+(?:VARCHAR2|NUMBER|DATE|BOOLEAN|INTEGER|PLS_INTEGER|BINARY_INTEGER|CLOB|BLOB|ROWTYPE|%TYPE|%ROWTYPE)\b"


def extract_sql_blocks(source: str) -> list[str]:
    """
    Extract embedded SQL statements from a PL/SQL block.
    Returns list of raw SQL strings for further parsing.
    """
    blocks = []
    source_upper = source.upper()

    for pattern in SQL_BLOCK_PATTERNS:
        matches = re.findall(pattern, source_upper, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match and len(match.strip()) > 10:
                blocks.append(match.strip())

    # Deduplicate while preserving order
    seen = set()
    unique_blocks = []
    for block in blocks:
        key = block[:50]
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)

    return unique_blocks


def detect_procedural_flags(source: str) -> list[str]:
    """
    Scan PL/SQL source for procedural constructs that need manual review.
    Returns list of human-readable flag strings.
    """
    flags = []
    source_upper = source.upper()
    seen_flags = set()

    for pattern, message in PROCEDURAL_FLAGS.items():
        if re.search(pattern, source_upper, re.DOTALL):
            if message not in seen_flags:
                flags.append(message)
                seen_flags.add(message)

    return flags


def extract_package_name(source: str) -> str:
    """Extract package or procedure name from CREATE statement."""
    match = re.search(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:PACKAGE|PROCEDURE|FUNCTION|TRIGGER)\s+(\w+)",
        source,
        re.IGNORECASE
    )
    return match.group(1).lower() if match else ""


def parse_plsql(source: str) -> dict[str, Any]:
    """
    Parse a PL/SQL block and return a unified IR dict.

    Strategy:
    1. Detect procedural flags (cursors, loops, exceptions, DBMS packages)
    2. Extract embedded SQL blocks
    3. Parse each SQL block with sqlglot
    4. Merge all SQL IRs into one unified IR
    5. Add procedural flags and set complexity to at least 3

    Args:
        source: Raw PL/SQL string

    Returns:
        IR dict with source_language set to plsql
    """
    ir = build_empty_ir()
    ir["source_language"] = "plsql"
    ir["dialect"] = "oracle"
    ir["raw_source"] = source

    if not source or not source.strip():
        ir["parse_errors"].append("Empty source provided")
        return validate_ir(ir)

    # Step 1 — Detect procedural flags
    flags = detect_procedural_flags(source)
    ir["procedural_flags"] = flags

    # Step 2 — Extract package/procedure name and add to tables context
    pkg_name = extract_package_name(source)
    if pkg_name:
        ir["parse_errors"].append(f"INFO: Parsing PL/SQL object: {pkg_name}")

    # Step 3 — Extract and parse embedded SQL blocks
    sql_blocks = extract_sql_blocks(source)

    if sql_blocks:
        for block in sql_blocks:
            try:
                block_ir = parse_sql(block, dialect="oracle")
                ir = merge_ir_fields(ir, block_ir)
            except Exception as e:
                ir["parse_errors"].append(f"SQL block parse error: {str(e)}")
    else:
        ir["parse_errors"].append(
            "INFO: No embedded SQL blocks extracted — procedural-only block"
        )

    # Step 4 — PL/SQL is always at least complexity 3
    # (procedural code with flags pushes to 4)
    ir = compute_complexity(ir)
    if ir["complexity_score"] < 3:
        ir["complexity_score"] = 3
        ir["complexity_label"] = "complex"
    if flags:
        ir["complexity_score"] = 4
        ir["complexity_label"] = "expert"

    ir = validate_ir(ir)
    return ir