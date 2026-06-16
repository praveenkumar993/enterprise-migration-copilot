"""
Dataset Validator Tests — 8 unit tests for the 7-check validation pipeline.
Run with: pytest dataset_gen/test_validate.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from validate import (
    validate_pair,
    check_required_fields,
    check_field_values,
    check_minimum_length,
    check_pyspark_syntax,
    check_dataframe_operations,
    check_no_placeholder_text,
    check_source_pyspark_alignment,
)


GOOD_PAIR = {
    "source_language": "sql",
    "difficulty": "easy",
    "source_code": "SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id",
    "pyspark_code": "df = spark.read.table('orders')\nresult = df.groupBy('customer_id').agg(F.sum('amount').alias('total'))",
    "features": ["aggregation", "groupby"],
    "generated_by": "claude-seed",
}


def test_good_pair_passes_all_checks():
    """A well-formed pair passes validation."""
    result = validate_pair(GOOD_PAIR)
    assert result["valid"] is True
    assert result["failed_check"] is None


def test_missing_field_fails():
    """Pair missing a required field fails check 1."""
    bad_pair = {k: v for k, v in GOOD_PAIR.items() if k != "pyspark_code"}
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "required_fields"


def test_invalid_source_language_fails():
    """Pair with invalid source_language fails check 2."""
    bad_pair = {**GOOD_PAIR, "source_language": "cobol"}
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "field_values"


def test_too_short_code_fails():
    """Pair with too-short pyspark_code fails check 3."""
    bad_pair = {**GOOD_PAIR, "pyspark_code": "df.show()"}
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "minimum_length"


def test_invalid_python_syntax_fails():
    """Pair with broken Python syntax fails check 4."""
    bad_pair = {**GOOD_PAIR, "pyspark_code": "df = spark.read.table('orders'\nresult = df.groupBy((((("}
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "pyspark_syntax"


def test_no_dataframe_ops_fails():
    """Pair with valid Python but no DataFrame ops fails check 5."""
    bad_pair = {**GOOD_PAIR, "pyspark_code": "x = 1 + 1\nprint('hello world this is just text')"}
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "dataframe_operations"


def test_placeholder_text_fails():
    """Pair with placeholder text fails check 6."""
    bad_pair = {
        **GOOD_PAIR,
        "pyspark_code": "df = spark.read.table('orders')\n# TODO: implement the rest\nresult = df.select('*')"
    }
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "no_placeholder_text"


def test_completely_unrelated_pyspark_fails_alignment():
    """Pair where pyspark code shares no tokens with source fails check 7."""
    bad_pair = {
        **GOOD_PAIR,
        "source_code": "SELECT zyxqpr_unique_table_name_xyz FROM weird_table_abc123",
        "pyspark_code": "df = spark.read.table('completely_different')\nresult = df.select('nothing_matching').filter(df.col1 > 5).groupBy('col2').agg(F.count('col3'))",
    }
    result = validate_pair(bad_pair)
    assert result["valid"] is False
    assert result["failed_check"] == "source_pyspark_alignment"