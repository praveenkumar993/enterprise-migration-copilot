"""
Language Router — auto-detects source language and routes to the correct parser.
Detection is keyword-based and order-sensitive.
More specific patterns are checked before generic ones.
"""

import re
from typing import Any

from parsers.sql_parser import parse_sql
from parsers.hiveql_parser import parse_hiveql


# Stored Procedure indicators (checked first — most specific)
SP_PATTERNS = [
    r"\bCREATE\s+(OR\s+REPLACE\s+)?PROCEDURE\b",
    r"\bCREATE\s+PROC\b",
    r"\bALTER\s+PROCEDURE\b",
    r"\bEXEC\s+\w+\b",
    r"\bSET\s+NOCOUNT\b",
    r"\bRAISERROR\b",
    r"@\w+\s+(INT|VARCHAR|NVARCHAR|BIT|DATETIME)",  # SQL Server params
]

# PL/SQL indicators
PLSQL_PATTERNS = [
    r"\bCREATE\s+(OR\s+REPLACE\s+)?(FUNCTION|PROCEDURE|PACKAGE|TRIGGER)\b",
    r"\bDECLARE\b.*\bBEGIN\b",
    r"\bBEGIN\b.*\bEND\s*;",
    r"\bCURSOR\s+\w+\s+IS\b",
    r"\bEXCEPTION\b.*\bWHEN\b",
    r"\bBULK\s+COLLECT\b",
    r"\bEXECUTE\s+IMMEDIATE\b",
    r"\bDBMS_\w+\b",
    r"\b:=\b",  # PL/SQL assignment operator
]

# HiveQL indicators
HIVEQL_PATTERNS = [
    r"\bDISTRIBUTE\s+BY\b",
    r"\bSORT\s+BY\b",
    r"\bCLUSTER\s+BY\b",
    r"\bLATERAL\s+VIEW\b",
    r"\bTABLESAMPLE\b",
    r"\bSTORED\s+AS\s+(ORC|PARQUET|AVRO|TEXTFILE|RCFILE)\b",
    r"\bROW\s+FORMAT\s+DELIMITED\b",
    r"\bHIVE\b",
]


def detect_language(source: str) -> str:
    """
    Detect source language from raw input string.

    Returns one of: sql, hiveql, plsql, stored_procedure

    Detection order matters:
    1. Stored Procedure (most specific procedural)
    2. PL/SQL (Oracle procedural)
    3. HiveQL (Hive-specific SQL)
    4. SQL (default fallback)
    """
    if not source or not source.strip():
        return "sql"

    source_upper = source.upper()

    # 1. Check Stored Procedure first
    for pattern in SP_PATTERNS:
        if re.search(pattern, source_upper, re.DOTALL):
            return "stored_procedure"

    # 2. Check PL/SQL
    for pattern in PLSQL_PATTERNS:
        if re.search(pattern, source_upper, re.DOTALL):
            return "plsql"

    # 3. Check HiveQL
    for pattern in HIVEQL_PATTERNS:
        if re.search(pattern, source_upper, re.DOTALL):
            return "hiveql"

    # 4. Default to SQL
    return "sql"


def route(source: str, dialect: str = "ansi") -> dict[str, Any]:
    """
    Auto-detect language and parse into IR.

    Args:
        source: Raw source code string
        dialect: SQL dialect hint (used only for SQL routing)

    Returns:
        IR dict with source_language set correctly
    """
    language = detect_language(source)

    if language == "hiveql":
        return parse_hiveql(source)

    if language == "plsql":
        # Stub — will be replaced in Phase 1.5
        from utils.ir_builder import build_empty_ir, validate_ir
        ir = build_empty_ir()
        ir["source_language"] = "plsql"
        ir["raw_source"] = source
        ir["parse_errors"].append(
            "INFO: PL/SQL parser not yet implemented — coming in Phase 1.5"
        )
        return validate_ir(ir)

    if language == "stored_procedure":
        # Stub — will be replaced in Phase 1.5
        from utils.ir_builder import build_empty_ir, validate_ir
        ir = build_empty_ir()
        ir["source_language"] = "stored_procedure"
        ir["raw_source"] = source
        ir["parse_errors"].append(
            "INFO: Stored Procedure parser not yet implemented — coming in Phase 1.5"
        )
        return validate_ir(ir)

    # Default: SQL
    return parse_sql(source, dialect=dialect)