"""
Risk and Compliance Agent — LLM-powered PII detection and cost flagging.
Fintech-specific: detects PAN, Aadhaar, account numbers, and other sensitive fields.
"""

import re
from typing import Any


# PII column name patterns — case insensitive
PII_COLUMN_PATTERNS = [
    r"\bpan\b", r"\bpan_number\b", r"\bpan_no\b",
    r"\baadhaar\b", r"\baadhaar_number\b", r"\buid\b",
    r"\baccount_number\b", r"\baccount_no\b", r"\bacct_no\b",
    r"\bcard_number\b", r"\bcredit_card\b", r"\bdebit_card\b", r"\bcard_no\b",
    r"\bcvv\b", r"\bcvv2\b", r"\bcvc\b",
    r"\bssn\b", r"\bsocial_security\b",
    r"\bpassport\b", r"\bpassport_number\b",
    r"\bdriver_license\b", r"\bdl_number\b",
    r"\bdob\b", r"\bdate_of_birth\b", r"\bbirth_date\b",
    r"\bemail\b", r"\bemail_address\b", r"\bemail_id\b",
    r"\bphone\b", r"\bphone_number\b", r"\bmobile\b", r"\bmobile_number\b",
    r"\bifsc\b", r"\bifsc_code\b",
    r"\bswift_code\b", r"\bbic_code\b",
    r"\brouting_number\b", r"\baba_routing\b",
    r"\bsalary\b", r"\bincome\b", r"\bannual_income\b", r"\bctc\b",
    r"\btax_id\b", r"\btin\b", r"\bgstin\b",
    r"\bvoter_id\b", r"\bnational_id\b",
    r"\bbiometric_id\b", r"\bfingerprint_id\b",
]

# Cost flag patterns in generated PySpark code
COST_PATTERNS = [
    {
        "pattern": r"\.collect\(\)",
        "flag": "collect() call detected — pulls entire DataFrame to driver, high memory cost",
        "weight": 2.0,
    },
    {
        "pattern": r"crossJoin\(",
        "flag": "crossJoin() detected — cartesian product, extreme compute cost",
        "weight": 3.0,
    },
    {
        "pattern": r"\.repartition\(\s*[5-9]\d{2,}\s*\)",
        "flag": "Very large repartition (500+) — unnecessary shuffle cost",
        "weight": 1.5,
    },
    {
        "pattern": r"udf\(",
        "flag": "UDF detected — Python UDFs disable Catalyst optimization",
        "weight": 1.0,
    },
    {
        "pattern": r"\.cache\(\)[\s\S]{0,500}\.cache\(\)",
        "flag": "Multiple cache() calls — verify memory usage won't cause OOM",
        "weight": 0.5,
    },
]

# Null safety patterns — joins on nullable columns
NULL_SAFETY_PATTERN = r"\.join\([^)]+\)"


def detect_pii_columns(ir: dict[str, Any], pyspark_code: str) -> list[str]:
    """
    Detect PII column names from IR columns list and in PySpark code.

    Returns list of detected PII column names.
    """
    pii_found = []
    columns = ir.get("columns", [])
    code_lower = pyspark_code.lower()

    # Check IR columns
    for col in columns:
        col_lower = col.lower()
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col_lower):
                if col not in pii_found:
                    pii_found.append(col)
                break

    # Also check raw source for column names not caught by parser
    raw_source = ir.get("raw_source", "").lower()
    for pattern in PII_COLUMN_PATTERNS:
        matches = re.findall(pattern, raw_source)
        for match in matches:
            if match and match not in pii_found:
                pii_found.append(match)

    return pii_found


def detect_cost_flags(pyspark_code: str) -> list[dict]:
    """
    Detect cost-heavy patterns in generated PySpark code.

    Returns list of cost flag dicts with flag message and weight.
    """
    flags = []
    for cp in COST_PATTERNS:
        if re.search(cp["pattern"], pyspark_code, re.DOTALL):
            flags.append({
                "flag": cp["flag"],
                "weight": cp["weight"],
            })
    return flags


def check_null_safety(ir: dict[str, Any], pyspark_code: str) -> list[str]:
    """
    Check if join keys are validated for NULL safety.

    Returns list of null safety warnings.
    """
    warnings = []
    joins = ir.get("joins", [])

    if not joins:
        return warnings

    if re.search(NULL_SAFETY_PATTERN, pyspark_code):
        # Check if there's a null filter before the join
        has_null_check = bool(
            re.search(r"isNotNull\(\)", pyspark_code) or
            re.search(r"isNull\(\)", pyspark_code) or
            re.search(r"\.na\.drop\(", pyspark_code) or
            re.search(r"\.dropna\(", pyspark_code)
        )
        if not has_null_check:
            for join in joins:
                join_table = join.get("table", "unknown")
                warnings.append(
                    f"JOIN on {join_table} — no null safety check detected. "
                    f"Add .filter(F.col('join_key').isNotNull()) before join"
                )

    return warnings


def check_risk(ir: dict[str, Any], pyspark_code: str) -> dict[str, Any]:
    """
    Perform risk and compliance checks on IR and generated PySpark code.

    Fintech-specific: PII detection, cost flags, null safety, procedural risk.

    Args:
        ir: Unified IR dict
        pyspark_code: Generated PySpark code from Migrator

    Returns:
        Risk result dict with pii_columns, compliance_flags, risk_score
    """
    compliance_flags = []
    pii_columns = detect_pii_columns(ir, pyspark_code)
    cost_flags = detect_cost_flags(pyspark_code)
    null_warnings = check_null_safety(ir, pyspark_code)

    # Risk score starts at 0 — higher is worse
    risk_score = 0.0

    # PII risk
    if pii_columns:
        compliance_flags.append(
            f"PII columns detected: {', '.join(pii_columns)} — "
            f"apply masking before writing to non-secure storage"
        )
        risk_score += len(pii_columns) * 1.5

    # Cost flags
    for cf in cost_flags:
        compliance_flags.append(cf["flag"])
        risk_score += cf["weight"]

    # Null safety
    for warning in null_warnings:
        compliance_flags.append(warning)
        risk_score += 0.5

    # Procedural migration risk
    procedural_flags = ir.get("procedural_flags", [])
    source_language = ir.get("source_language", "sql")
    if source_language in ("plsql", "stored_procedure") and procedural_flags:
        compliance_flags.append(
            "Procedural migration — manual review of flagged constructs "
            "required before production deployment"
        )
        risk_score += len(procedural_flags) * 0.5

    # Risk level label
    if risk_score == 0:
        risk_level = "low"
    elif risk_score <= 3:
        risk_level = "medium"
    elif risk_score <= 7:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "pii_columns": pii_columns,
        "compliance_flags": compliance_flags,
        "cost_flags": [cf["flag"] for cf in cost_flags],
        "null_safety_warnings": null_warnings,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "requires_manual_review": len(procedural_flags) > 0 or len(pii_columns) > 0,
    }