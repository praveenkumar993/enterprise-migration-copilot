"""
Reviewer Agent — LLM-powered code review and correction.
Reviews generated PySpark code for semantic correctness.
"""

import os
import re
import requests
import time
from typing import Any
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveenkumar993")
REVIEW_MODEL = f"{HF_USERNAME}/phi-enterprise-migration"
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
HF_API_BASE = "https://api-inference.huggingface.co/models"


def count_sql_blocks(ir: dict[str, Any]) -> int:
    """Estimate number of SQL blocks from IR — used for procedural review."""
    source_language = ir.get("source_language", "sql")
    if source_language not in ("plsql", "stored_procedure"):
        return 1

    # Estimate from tables accessed — each table likely has one SQL block
    tables = ir.get("tables", [])
    flags = ir.get("procedural_flags", [])

    # If procedural flags mention UPDATE or INSERT, count those too
    block_count = max(1, len(tables))
    for flag in flags:
        if any(kw in flag.upper() for kw in ["INSERT", "UPDATE", "DELETE", "MERGE"]):
            block_count += 1

    return block_count


def count_dataframe_chains(code: str) -> int:
    """Count distinct DataFrame transformation chains in PySpark code."""
    # Count spark.read calls and df variable assignments
    read_count = len(re.findall(r"spark\.read\.", code))
    df_assign = len(re.findall(r"\bdf\w*\s*=", code))
    result_assign = len(re.findall(r"\bresult\w*\s*=", code))
    return max(read_count, df_assign + result_assign, 1)


def semantic_review(ir: dict[str, Any], pyspark_code: str) -> dict[str, Any]:
    """
    Pure Python semantic checks on the generated PySpark code.

    Returns semantic review result without LLM.
    """
    semantic_issues = []
    score = 1.0

    source_language = ir.get("source_language", "sql")
    tables = ir.get("tables", [])
    joins = ir.get("joins", [])
    windows = ir.get("window_functions", [])
    procedural_flags = ir.get("procedural_flags", [])

    # Check 1 — Tables present
    code_lower = pyspark_code.lower()
    missing_tables = [t for t in tables if t.lower() not in code_lower]
    if missing_tables:
        semantic_issues.append(f"Tables not referenced: {', '.join(missing_tables)}")
        score -= 0.15 * len(missing_tables)

    # Check 2 — Joins reflected
    if joins and ".join(" not in pyspark_code:
        semantic_issues.append("IR has JOINs but no .join() found in PySpark code")
        score -= 0.20

    # Check 3 — Window functions reflected
    if windows and "Window" not in pyspark_code and ".over(" not in pyspark_code:
        semantic_issues.append(
            "IR has window functions but no Window spec found in PySpark code"
        )
        score -= 0.20

    # Check 4 — Procedural SQL blocks reflected
    if source_language in ("plsql", "stored_procedure") and procedural_flags:
        expected_blocks = count_sql_blocks(ir)
        actual_chains = count_dataframe_chains(pyspark_code)
        if expected_blocks > 2 and actual_chains < 2:
            semantic_issues.append(
                f"Missing transformation blocks — parser found ~{expected_blocks} "
                f"SQL blocks but PySpark only has {actual_chains} DataFrame chain(s)"
            )
            score -= 0.15

    # Check 5 — Import present
    if "import" not in pyspark_code:
        semantic_issues.append("No imports found — add 'import pyspark.sql.functions as F'")
        score -= 0.05

    semantic_score = max(0.0, round(score, 3))
    return {
        "semantic_issues": semantic_issues,
        "semantic_score": semantic_score,
        "tables_covered": len(tables) - len(missing_tables) if tables else 0,
        "total_tables": len(tables),
    }


def review(ir: dict[str, Any], pyspark_code: str) -> dict[str, Any]:
    """
    Review generated PySpark code for semantic correctness.

    Combines pure Python semantic checks with a lightweight
    LLM review pass when HF_TOKEN is available.

    Args:
        ir: Unified IR dict
        pyspark_code: Generated PySpark code from Migrator

    Returns:
        Review result dict with score and issues
    """
    if not pyspark_code or not pyspark_code.strip():
        return {
            "semantic_score": 0.0,
            "semantic_issues": ["Empty PySpark code — nothing to review"],
            "tables_covered": 0,
            "total_tables": len(ir.get("tables", [])),
            "llm_feedback": "",
            "reviewed_code": pyspark_code,
        }

    # Pure Python semantic review
    semantic_result = semantic_review(ir, pyspark_code)

    # LLM review pass (lightweight — just asks for issues)
    llm_feedback = ""
    if HF_TOKEN and semantic_result["semantic_score"] < 0.8:
        source_language = ir.get("source_language", "sql")
        review_prompt = (
            f"Review this PySpark code converted from {source_language}. "
            f"List any issues in one sentence each. "
            f"Code:\n{pyspark_code[:800]}"
        )
        try:
            url = f"{HF_API_BASE}/{REVIEW_MODEL}"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            payload = {
                "inputs": review_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.1,
                    "return_full_text": False,
                },
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    llm_feedback = data[0].get("generated_text", "").strip()
        except Exception:
            pass  # LLM review is best-effort

    return {
        "semantic_score": semantic_result["semantic_score"],
        "semantic_issues": semantic_result["semantic_issues"],
        "tables_covered": semantic_result["tables_covered"],
        "total_tables": semantic_result["total_tables"],
        "llm_feedback": llm_feedback,
        "reviewed_code": pyspark_code,
    }