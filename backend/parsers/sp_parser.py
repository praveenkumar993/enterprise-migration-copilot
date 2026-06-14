"""
Stored Procedure Parser — handles T-SQL / SQL Server stored procedures.
Uses regex-based extraction since sqlglot has limited T-SQL procedural support.
"""

import re
from typing import Any

from parsers.sql_parser import parse_sql
from utils.ir_builder import (
    build_empty_ir, compute_complexity,
    merge_ir_fields, validate_ir
)


# T-SQL specific procedural flags
SP_PROCEDURAL_FLAGS = {
    r"\bCURSOR\b.*\bFOR\b":
        "CURSOR detected — manual review required, map to DataFrame iteration",
    r"\bFETCH\s+NEXT\b":
        "FETCH NEXT detected — cursor loop, map to df.collect() + Python iteration",
    r"\bWHILE\b.*\bBEGIN\b":
        "WHILE loop detected — consider replacing with DataFrame transformation",
    r"\bIF\s+EXISTS\b":
        "IF EXISTS detected — map to df.isEmpty() check in PySpark",
    r"\bTRY\b.*\bCATCH\b":
        "TRY/CATCH detected — add try/except in PySpark job",
    r"\bRAISERROR\b":
        "RAISERROR detected — map to Python raise Exception()",
    r"\bTHROW\b":
        "THROW detected — map to Python raise Exception()",
    r"\bEXEC\s+\w+\b":
        "EXEC call detected — nested procedure call, manual review required",
    r"\bSET\s+NOCOUNT\s+ON\b":
        "SET NOCOUNT ON detected — no PySpark equivalent, safe to remove",
    r"\bPRINT\b":
        "PRINT detected — replace with Python logging",
    r"\bXACT_ABORT\b":
        "XACT_ABORT detected — handle with PySpark transaction semantics",
    r"\bBEGIN\s+TRAN(?:SACTION)?\b":
        "BEGIN TRANSACTION detected — map to Delta Lake transaction if available",
    r"\bCOMMIT\s+TRAN(?:SACTION)?\b":
        "COMMIT TRANSACTION detected — map to Delta Lake commit",
    r"\bROLLBACK\b":
        "ROLLBACK detected — map to Delta Lake rollback",
    r"#\w+":
        "Temp table (#table) detected — map to df.createOrReplaceTempView()",
    r"\bINSERT\s+INTO\s+#":
        "Insert into temp table detected — map to createOrReplaceTempView",
    r"@\w+\s+TABLE":
        "Table variable (@var TABLE) detected — map to PySpark DataFrame",
    r"\bMERGE\b.*\bUSING\b":
        "MERGE statement detected — map to Delta Lake MERGE or manual upsert logic",
    r"\bOPENROWSET\b":
        "OPENROWSET detected — map to spark.read with appropriate connector",
    r"\bLINKED\s+SERVER\b":
        "Linked server detected — map to spark.read with JDBC connector",
}

# Patterns to extract SQL blocks from T-SQL procedures
SP_SQL_PATTERNS = [
    r"(SELECT\s+.+?FROM\s+.+?(?:WHERE\s+.+?)?(?:GROUP\s+BY\s+.+?)?(?:ORDER\s+BY\s+.+?)?(?:HAVING\s+.+?)?);",
    r"(INSERT\s+INTO\s+\w+.+?(?:SELECT\s+.+?FROM\s+.+?|VALUES\s*\(.+?\)));",
    r"(UPDATE\s+\w+\s+SET\s+.+?(?:WHERE\s+.+?)?);",
    r"(DELETE\s+(?:FROM\s+)?\w+(?:\s+WHERE\s+.+?)?);",
    r"(MERGE\s+.+?USING\s+.+?ON\s+.+?(?:WHEN\s+.+?)+);",
    r"(CREATE\s+(?:OR\s+ALTER\s+)?(?:VIEW|TABLE)\s+.+?(?:AS\s+SELECT\s+.+?)?);",
]


def extract_sp_parameters(source: str) -> list[dict]:
    """
    Extract stored procedure input/output parameters.
    Returns list of dicts with name, type, direction.
    """
    params = []
    # Match @param_name data_type [= default] [OUTPUT]
    pattern = r"(@\w+)\s+([\w\(\),]+)(?:\s*=\s*[\w']+)?(\s+OUTPUT|\s+OUT)?"
    matches = re.findall(pattern, source, re.IGNORECASE)
    for match in matches:
        name, dtype, direction = match
        params.append({
            "name": name.lower(),
            "type": dtype.upper(),
            "direction": "OUTPUT" if direction.strip().upper() in ("OUTPUT", "OUT") else "INPUT",
        })
    return params


def extract_temp_tables(source: str) -> list[str]:
    """Extract temp table names (#tablename) from the procedure."""
    matches = re.findall(r"#(\w+)", source, re.IGNORECASE)
    return list(set(m.lower() for m in matches))


def extract_sp_sql_blocks(source: str) -> list[str]:
    """Extract embedded SQL statements from T-SQL stored procedure."""
    blocks = []

    for pattern in SP_SQL_PATTERNS:
        matches = re.findall(pattern, source, re.DOTALL | re.IGNORECASE)
        for match in matches:
            if match and len(match.strip()) > 10:
                blocks.append(match.strip())

    # Deduplicate
    seen = set()
    unique = []
    for b in blocks:
        key = b[:50]
        if key not in seen:
            seen.add(key)
            unique.append(b)

    return unique


def detect_sp_flags(source: str) -> list[str]:
    """Detect T-SQL procedural constructs that need manual review."""
    flags = []
    seen = set()

    for pattern, message in SP_PROCEDURAL_FLAGS.items():
        if re.search(pattern, source, re.DOTALL | re.IGNORECASE):
            if message not in seen:
                flags.append(message)
                seen.add(message)

    return flags


def parse_stored_procedure(source: str) -> dict[str, Any]:
    """
    Parse a T-SQL stored procedure and return a unified IR dict.

    Strategy:
    1. Extract parameters and temp tables
    2. Detect T-SQL procedural flags
    3. Extract and parse embedded SQL blocks
    4. Merge all SQL IRs
    5. Add SP-specific metadata to IR

    Args:
        source: Raw T-SQL stored procedure string

    Returns:
        IR dict with source_language set to stored_procedure
    """
    ir = build_empty_ir()
    ir["source_language"] = "stored_procedure"
    ir["dialect"] = "tsql"
    ir["raw_source"] = source

    if not source or not source.strip():
        ir["parse_errors"].append("Empty source provided")
        return validate_ir(ir)

    # Step 1 — Extract parameters
    params = extract_sp_parameters(source)
    if params:
        param_names = [p["name"] for p in params]
        ir["parse_errors"].append(
            f"INFO: SP parameters detected: {', '.join(param_names)}"
        )

    # Step 2 — Extract temp tables
    temp_tables = extract_temp_tables(source)
    for t in temp_tables:
        if t not in ir["tables"]:
            ir["tables"].append(t)
        ir["parse_errors"].append(
            f"INFO: Temp table #{t} detected — map to createOrReplaceTempView('{t}')"
        )

    # Step 3 — Detect procedural flags
    flags = detect_sp_flags(source)
    ir["procedural_flags"] = flags

    # Step 4 — Extract and parse embedded SQL blocks
    sql_blocks = extract_sp_sql_blocks(source)

    if sql_blocks:
        for block in sql_blocks:
            try:
                # Use mssql dialect for T-SQL
                block_ir = parse_sql(block, dialect="tsql")
                ir = merge_ir_fields(ir, block_ir)
            except Exception as e:
                ir["parse_errors"].append(f"SQL block parse error: {str(e)}")
    else:
        ir["parse_errors"].append(
            "INFO: No embedded SQL blocks extracted — procedural-only block"
        )

    # Step 5 — SP is always at least complexity 3
    ir = compute_complexity(ir)
    if ir["complexity_score"] < 3:
        ir["complexity_score"] = 3
        ir["complexity_label"] = "complex"
    if flags:
        ir["complexity_score"] = 4
        ir["complexity_label"] = "expert"

    ir = validate_ir(ir)
    return ir