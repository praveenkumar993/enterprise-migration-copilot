"""
Analyzer Agent — Pure Python, no LLM.
Reads IR and produces analysis output for the Migrator agent.
"""

from typing import Any


REVIEW_TIME = {
    "simple": "approximately 5 min",
    "moderate": "approximately 15 min",
    "complex": "approximately 30 min",
    "expert": "approximately 60 min",
}

MIGRATION_STRATEGY = {
    "sql": "Direct DataFrame API translation — map each SQL clause to its PySpark equivalent using spark.read.table() and DataFrame transformations.",
    "hiveql": "Direct DataFrame API translation — map HiveQL clauses to PySpark. Replace DISTRIBUTE BY with repartition(), SORT BY with sortWithinPartitions(), LATERAL VIEW EXPLODE with withColumn(F.explode()).",
    "plsql": "Extract SQL transformation logic from procedural shell — convert embedded SELECT/INSERT/UPDATE to DataFrame operations. Flag CURSOR loops, EXCEPTION handlers, and DBMS packages for manual review.",
    "stored_procedure": "Convert to parameterized PySpark function — map input parameters to function arguments, temp tables to intermediate DataFrames or createOrReplaceTempView(), flag TRY/CATCH and MERGE for manual review.",
}


def analyze(ir: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze IR and produce structured output for downstream agents.

    Pure Python — no LLM call. Deterministic.

    Args:
        ir: Unified IR dict from any parser

    Returns:
        Analysis dict with complexity, strategy, risk flags, and RAG query
    """
    source_language = ir.get("source_language", "sql")
    complexity_label = ir.get("complexity_label", "simple")
    complexity_score = ir.get("complexity_score", 1)

    # --- Detected features ---
    detected_features = []

    joins = ir.get("joins", [])
    if joins:
        join_types = list({j.get("type", "JOIN") for j in joins})
        detected_features.append(f"{len(joins)} JOIN(s): {', '.join(join_types)}")

    aggs = ir.get("aggregations", [])
    if aggs:
        detected_features.append(f"Aggregations: {', '.join(aggs)}")

    windows = ir.get("window_functions", [])
    if windows:
        detected_features.append(f"Window functions: {', '.join(windows)}")

    ctes = ir.get("ctes", [])
    if ctes:
        detected_features.append(f"CTEs: {', '.join(ctes)}")

    subquery_depth = ir.get("subquery_depth", 0)
    if subquery_depth > 0:
        detected_features.append(f"Subquery depth: {subquery_depth}")

    if ir.get("has_udf"):
        detected_features.append("UDF detected")

    tables = ir.get("tables", [])
    if tables:
        detected_features.append(f"Tables: {', '.join(tables)}")

    procedural_flags = ir.get("procedural_flags", [])
    if procedural_flags:
        detected_features.append(f"{len(procedural_flags)} procedural construct(s) flagged")

    # --- Risk flags ---
    risk_flags = []

    if windows:
        risk_flags.append("Window functions detected — verify PARTITION BY logic")

    if subquery_depth > 1:
        risk_flags.append("Deep subquery nesting — review restructuring")

    if ir.get("has_udf"):
        risk_flags.append("UDF detected — may need custom PySpark implementation")

    if len(joins) > 3:
        risk_flags.append("Multiple JOINs — verify join order and broadcast hints")

    if ctes:
        risk_flags.append("CTE detected — converts to intermediate DataFrames")

    # Pass through each procedural flag as a risk flag
    for flag in procedural_flags:
        risk_flags.append(flag)

    # --- Migration strategy ---
    strategy = MIGRATION_STRATEGY.get(
        source_language,
        MIGRATION_STRATEGY["sql"]
    )

    # --- Needs procedural context ---
    needs_procedural_context = source_language in ("plsql", "stored_procedure")

    # --- RAG query ---
    rag_parts = [f"migrate {source_language} to pyspark"]
    rag_parts.extend(detected_features)
    if complexity_label:
        rag_parts.append(f"complexity {complexity_label}")
    rag_query = " ".join(rag_parts)

    # --- Estimated review time ---
    review_time = REVIEW_TIME.get(complexity_label, "approximately 15 min")
    if procedural_flags:
        review_time += " plus manual review of flagged constructs"

    return {
        "complexity": complexity_label,
        "complexity_score": complexity_score,
        "source_language": source_language,
        "migration_strategy": strategy,
        "risk_flags": risk_flags,
        "detected_features": detected_features,
        "procedural_flags": procedural_flags,
        "rag_query": rag_query,
        "needs_procedural_context": needs_procedural_context,
        "estimated_review_time": review_time,
    }