"""
Dataset Validation Pipeline — 7-check validator for migration pairs.
Validates both Claude seed pairs and Ollama bulk-generated pairs.
Filters raw pairs down to clean, trainable examples.
"""

import ast
import json
import re
from pathlib import Path
from typing import Any


VALID_SOURCE_LANGUAGES = {"sql", "hiveql", "plsql", "stored_procedure"}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "expert"}
VALID_GENERATED_BY = {"claude-seed", "ollama-bulk"}

# Minimum lengths to filter out junk/empty generations
MIN_SOURCE_LENGTH = 15
MIN_PYSPARK_LENGTH = 20

# DataFrame operations that must appear in valid PySpark output
DATAFRAME_OP_PATTERNS = [
    r"\.select\(", r"\.filter\(", r"\.where\(", r"\.groupBy\(",
    r"\.join\(", r"\.agg\(", r"\.withColumn\(", r"\.orderBy\(",
    r"\.union\(", r"\.distinct\(", r"\.drop\(", r"\.limit\(",
    r"spark\.read", r"spark\.sql\(", r"\.write\.", r"spark\.table\(",
    r"\.show\(", r"\.count\(", r"\.collect\(", r"\.toPandas\(",
    r"\.createDataFrame\(", r"\.repartition\(", r"\.cache\(",
    r"\.persist\(", r"DeltaTable\.",
]


def check_required_fields(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 1 — All required fields exist."""
    required = {
        "source_language", "difficulty", "source_code",
        "pyspark_code", "features", "generated_by"
    }
    missing = required - set(pair.keys())
    if missing:
        return False, f"Missing fields: {', '.join(sorted(missing))}"
    return True, ""


def check_field_values(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 2 — Field values are valid enums and correct types."""
    if pair.get("source_language") not in VALID_SOURCE_LANGUAGES:
        return False, f"Invalid source_language: {pair.get('source_language')}"

    if pair.get("difficulty") not in VALID_DIFFICULTIES:
        return False, f"Invalid difficulty: {pair.get('difficulty')}"

    if pair.get("generated_by") not in VALID_GENERATED_BY:
        return False, f"Invalid generated_by: {pair.get('generated_by')}"

    if not isinstance(pair.get("features"), list):
        return False, "features must be a list"

    if not isinstance(pair.get("source_code"), str):
        return False, "source_code must be a string"

    if not isinstance(pair.get("pyspark_code"), str):
        return False, "pyspark_code must be a string"

    return True, ""


def check_minimum_length(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 3 — Source and PySpark code meet minimum length."""
    source = pair.get("source_code", "")
    pyspark = pair.get("pyspark_code", "")

    if len(source.strip()) < MIN_SOURCE_LENGTH:
        return False, f"source_code too short ({len(source.strip())} chars)"

    if len(pyspark.strip()) < MIN_PYSPARK_LENGTH:
        return False, f"pyspark_code too short ({len(pyspark.strip())} chars)"

    return True, ""


def check_pyspark_syntax(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 4 — PySpark code is valid Python syntax."""
    pyspark = pair.get("pyspark_code", "")
    try:
        ast.parse(pyspark)
        return True, ""
    except SyntaxError as e:
        return False, f"PySpark syntax error: line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Parse error: {str(e)}"


def check_dataframe_operations(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 5 — PySpark code contains at least one real DataFrame operation."""
    pyspark = pair.get("pyspark_code", "")
    for pattern in DATAFRAME_OP_PATTERNS:
        if re.search(pattern, pyspark):
            return True, ""
    return False, "No DataFrame operations found in pyspark_code"


def check_no_placeholder_text(pair: dict[str, Any]) -> tuple[bool, str]:
    """Check 6 — No placeholder or incomplete generation artifacts."""
    pyspark = pair.get("pyspark_code", "")
    source = pair.get("source_code", "")

    placeholder_markers = [
        "TODO: implement", "...", "<your code here>",
        "[INSERT", "PLACEHOLDER", "FIXME: add",
        "# Add your code", "lorem ipsum",
    ]

    combined = (pyspark + " " + source).lower()
    for marker in placeholder_markers:
        if marker.lower() in combined:
            return False, f"Placeholder text found: '{marker}'"

    # Check code didn't get cut off mid-generation
    if pyspark.strip().endswith((",", "(", "+", "-", "=", "and", "or")):
        return False, "PySpark code appears truncated (ends mid-expression)"

    return True, ""


def check_source_pyspark_alignment(pair: dict[str, Any]) -> tuple[bool, str]:
    """
    Check 7 — Loose semantic alignment between source and PySpark.
    At least one identifier-like token from source should appear in PySpark.
    """
    source = pair.get("source_code", "")
    pyspark = pair.get("pyspark_code", "")

    # Extract identifier-like tokens from source (table/column name candidates)
    source_tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", source.lower()))
    pyspark_lower = pyspark.lower()

    # Filter out common SQL keywords that don't count as alignment signal
    sql_keywords = {
        "select", "from", "where", "group", "order", "join", "inner",
        "left", "right", "outer", "on", "and", "or", "not", "null",
        "as", "by", "having", "with", "case", "when", "then", "else",
        "end", "begin", "declare", "exception", "cursor", "loop", "for",
        "into", "values", "insert", "update", "delete", "create",
        "table", "procedure", "function", "is", "in", "exists",
    }
    candidate_tokens = source_tokens - sql_keywords

    if not candidate_tokens:
        return True, ""  # Nothing to check against — pass

    matches = sum(1 for t in candidate_tokens if t in pyspark_lower)
    match_ratio = matches / len(candidate_tokens)

    if match_ratio < 0.15:
        return False, (
            f"Low alignment between source and pyspark "
            f"({match_ratio:.0%} of source tokens found in output)"
        )

    return True, ""


# All checks in order — run cheapest/fastest checks first
ALL_CHECKS = [
    ("required_fields", check_required_fields),
    ("field_values", check_field_values),
    ("minimum_length", check_minimum_length),
    ("pyspark_syntax", check_pyspark_syntax),
    ("dataframe_operations", check_dataframe_operations),
    ("no_placeholder_text", check_no_placeholder_text),
    ("source_pyspark_alignment", check_source_pyspark_alignment),
]


def validate_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """
    Run all 7 checks on a single pair.

    Returns dict with: valid (bool), failed_check (str or None), reason (str)
    Stops at first failure for speed.
    """
    for check_name, check_fn in ALL_CHECKS:
        try:
            passed, reason = check_fn(pair)
        except Exception as e:
            passed, reason = False, f"Check raised exception: {str(e)}"

        if not passed:
            return {
                "valid": False,
                "failed_check": check_name,
                "reason": reason,
            }

    return {"valid": True, "failed_check": None, "reason": ""}


def validate_dataset(
    input_path: str,
    valid_output_path: str,
    skipped_output_path: str,
) -> dict[str, Any]:
    """
    Validate an entire JSONL dataset file.

    Reads raw pairs, runs 7-check validation, writes valid pairs and
    skipped pairs (with reasons) to separate output files.

    Args:
        input_path: Path to raw JSONL pairs (one JSON object per line)
        valid_output_path: Path to write validated pairs
        skipped_output_path: Path to write skipped pairs with reasons

    Returns:
        Summary stats dict
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    total = 0
    valid_count = 0
    skipped_count = 0
    failure_breakdown: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_generated_by: dict[str, dict[str, int]] = {
        "claude-seed": {"total": 0, "valid": 0},
        "ollama-bulk": {"total": 0, "valid": 0},
    }

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(valid_output_path, "w", encoding="utf-8") as valid_out, \
         open(skipped_output_path, "w", encoding="utf-8") as skipped_out:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                skipped_count += 1
                failure_breakdown["invalid_json"] = failure_breakdown.get("invalid_json", 0) + 1
                skipped_out.write(json.dumps({
                    "line_number": line_num,
                    "failed_check": "invalid_json",
                    "reason": str(e),
                    "raw_line": line[:200],
                }) + "\n")
                continue

            gen_by = pair.get("generated_by", "unknown")
            if gen_by in by_generated_by:
                by_generated_by[gen_by]["total"] += 1

            result = validate_pair(pair)

            if result["valid"]:
                valid_count += 1
                source_lang = pair.get("source_language", "unknown")
                by_source[source_lang] = by_source.get(source_lang, 0) + 1
                if gen_by in by_generated_by:
                    by_generated_by[gen_by]["valid"] += 1
                valid_out.write(json.dumps(pair) + "\n")
            else:
                skipped_count += 1
                check_name = result["failed_check"]
                failure_breakdown[check_name] = failure_breakdown.get(check_name, 0) + 1
                skipped_out.write(json.dumps({
                    "line_number": line_num,
                    "failed_check": check_name,
                    "reason": result["reason"],
                    "source_language": pair.get("source_language", "unknown"),
                    "generated_by": gen_by,
                }) + "\n")

    pass_rate = round((valid_count / total) * 100, 2) if total > 0 else 0.0

    # Calculate pass rates per generation source
    for gen_by, stats in by_generated_by.items():
        stats["pass_rate"] = (
            round((stats["valid"] / stats["total"]) * 100, 2)
            if stats["total"] > 0 else 0.0
        )

    summary = {
        "total_pairs": total,
        "valid_pairs": valid_count,
        "skipped_pairs": skipped_count,
        "pass_rate_percent": pass_rate,
        "failure_breakdown": failure_breakdown,
        "valid_by_source_language": by_source,
        "by_generation_source": by_generated_by,
    }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print validation summary to console."""
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs processed:  {summary['total_pairs']}")
    print(f"Valid pairs:            {summary['valid_pairs']}")
    print(f"Skipped pairs:          {summary['skipped_pairs']}")
    print(f"Pass rate:              {summary['pass_rate_percent']}%")

    print("\n--- Valid pairs by source language ---")
    for lang, count in summary["valid_by_source_language"].items():
        print(f"  {lang:20s} {count}")

    print("\n--- Pass rate by generation source ---")
    for gen_by, stats in summary["by_generation_source"].items():
        print(f"  {gen_by:15s} total={stats['total']:5d}  valid={stats['valid']:5d}  pass_rate={stats['pass_rate']}%")

    print("\n--- Failure breakdown ---")
    for check, count in sorted(summary["failure_breakdown"].items(), key=lambda x: -x[1]):
        print(f"  {check:30s} {count}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_gen/raw_pairs.jsonl"
    valid_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_gen/valid_pairs.jsonl"
    skipped_path = sys.argv[3] if len(sys.argv) > 3 else "dataset_gen/skipped.jsonl"

    summary = validate_dataset(input_path, valid_path, skipped_path)
    print_summary(summary)