"""
SQL Parser — converts SQL (all dialects) to unified IR.
Uses sqlglot for robust multi-dialect parsing.
"""

import sqlglot
import sqlglot.expressions as exp
from typing import Any

from utils.ir_builder import build_empty_ir, compute_complexity, validate_ir


SUPPORTED_DIALECTS = [
    "", "bigquery", "clickhouse", "duckdb",
    "mysql", "postgres", "snowflake", "spark", "hive",
]

KNOWN_BUILTINS = {
    "count", "sum", "avg", "min", "max", "coalesce", "nullif",
    "upper", "lower", "trim", "length", "substr", "substring",
    "cast", "convert", "round", "floor", "ceil", "abs",
    "rank", "row_number", "dense_rank", "lag", "lead",
    "ntile", "first_value", "last_value", "date_trunc",
    "datediff", "dateadd", "to_date", "now", "current_date",
    "ifnull", "isnull", "nvl", "decode", "case",
}

WINDOW_FUNC_NAMES = {
    "rank", "row_number", "dense_rank", "lag", "lead",
    "ntile", "first_value", "last_value", "percent_rank", "cume_dist",
}


def parse_sql(source: str, dialect: str = "") -> dict[str, Any]:
    """
    Parse a SQL string and return a unified IR dict.

    Args:
        source: Raw SQL string
        dialect: sqlglot dialect name. Use empty string for generic SQL.

    Returns:
        IR dict with all fields populated
    """
    ir = build_empty_ir()
    ir["source_language"] = "sql"
    ir["dialect"] = dialect
    ir["raw_source"] = source

    if not source or not source.strip():
        ir["parse_errors"].append("Empty source provided")
        return validate_ir(ir)

    # sqlglot 23.12.2 does not support "ansi" — use empty string for generic
    safe_dialect = dialect if dialect in SUPPORTED_DIALECTS else ""

    try:
        statements = sqlglot.parse(
            source,
            dialect=safe_dialect,
            error_level=sqlglot.ErrorLevel.WARN,
        )
    except Exception as e:
        ir["parse_errors"].append(f"sqlglot parse failed: {str(e)}")
        return validate_ir(ir)

    if not statements:
        ir["parse_errors"].append("No statements parsed")
        return validate_ir(ir)

    tables = set()
    columns = set()
    joins = []
    aggregations = set()
    window_functions = set()
    ctes = set()
    max_subquery_depth = 0
    has_udf = False

    for statement in statements:
        if statement is None:
            continue

        # --- Tables ---
        for node in statement.find_all(exp.Table):
            if node.name:
                tables.add(node.name.lower())

        # --- Columns ---
        for node in statement.find_all(exp.Column):
            if node.name:
                columns.add(node.name.lower())

        # --- JOINs ---
        # In sqlglot 23.12.2: join.args has 'kind' (INNER/LEFT/etc), 'on', 'this'
        for node in statement.find_all(exp.Join):
            side = node.args.get("side") or ""
            kind = node.args.get("kind") or ""

            if side and kind:
                join_type = f"{side} {kind}".strip().upper()
            elif kind:
                join_type = kind.upper()
            elif side:
                join_type = side.upper()
            else:
                join_type = "JOIN"

            join_table = ""
            if isinstance(node.this, exp.Table) and node.this.name:
                join_table = node.this.name.lower()

            on_clause = ""
            if node.args.get("on"):
                on_clause = node.args["on"].sql()

            joins.append({
                "type": join_type,
                "table": join_table,
                "on": on_clause,
            })

        # --- Aggregations (typed nodes) ---
        for node in statement.find_all(exp.Count):
            aggregations.add("COUNT")
        for node in statement.find_all(exp.Sum):
            aggregations.add("SUM")
        for node in statement.find_all(exp.Avg):
            aggregations.add("AVG")
        for node in statement.find_all(exp.Min):
            aggregations.add("MIN")
        for node in statement.find_all(exp.Max):
            aggregations.add("MAX")

        # --- Window Functions ---
        # In sqlglot 23.12.2: RANK() inside Window is exp.Anonymous
        for node in statement.find_all(exp.Window):
            func = node.this
            if func is None:
                continue
            if isinstance(func, exp.Anonymous):
                fname = (func.name or "").lower()
                window_functions.add(fname.upper())
            else:
                window_functions.add(type(func).__name__.upper())

        # --- CTEs ---
        for node in statement.find_all(exp.CTE):
            if node.alias:
                ctes.add(node.alias.lower())

        # --- UDF Detection via Anonymous functions ---
        for node in statement.find_all(exp.Anonymous):
            fname = (node.name or "").lower()
            # Skip if it's a known window function used inside OVER()
            if fname and fname not in KNOWN_BUILTINS and fname not in WINDOW_FUNC_NAMES:
                has_udf = True

        # --- Subquery Depth ---
        def _depth(node: exp.Expression, d: int = 0) -> int:
            best = d
            for child in node.walk():
                if child is node:
                    continue
                if isinstance(child, exp.Subquery):
                    best = max(best, _depth(child, d + 1))
            return best

        max_subquery_depth = max(max_subquery_depth, _depth(statement))

    ir["tables"] = sorted(tables)
    ir["columns"] = sorted(columns)
    ir["joins"] = joins
    ir["aggregations"] = sorted(aggregations)
    ir["window_functions"] = sorted(window_functions)
    ir["ctes"] = sorted(ctes)
    ir["subquery_depth"] = max_subquery_depth
    ir["has_udf"] = has_udf

    ir = compute_complexity(ir)
    ir = validate_ir(ir)
    return ir