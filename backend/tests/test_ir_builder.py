"""
IR Builder Tests — 8 unit tests.
Run with: pytest tests/test_ir_builder.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.ir_builder import (
    build_empty_ir,
    compute_complexity,
    merge_ir_fields,
    validate_ir,
)


def test_build_empty_ir_has_all_fields():
    """Empty IR has all required fields with correct types."""
    ir = build_empty_ir()
    assert isinstance(ir["tables"], list)
    assert isinstance(ir["joins"], list)
    assert isinstance(ir["aggregations"], list)
    assert isinstance(ir["window_functions"], list)
    assert isinstance(ir["ctes"], list)
    assert isinstance(ir["procedural_flags"], list)
    assert isinstance(ir["parse_errors"], list)
    assert isinstance(ir["has_udf"], bool)
    assert isinstance(ir["subquery_depth"], int)
    assert isinstance(ir["complexity_score"], int)
    assert isinstance(ir["complexity_label"], str)


def test_complexity_1_simple_select():
    """No joins, no aggs, no windows = complexity 1 simple."""
    ir = build_empty_ir()
    ir = compute_complexity(ir)
    assert ir["complexity_score"] == 1
    assert ir["complexity_label"] == "simple"


def test_complexity_2_with_joins():
    """Joins present = complexity 2 moderate."""
    ir = build_empty_ir()
    ir["joins"] = [{"type": "INNER", "table": "customers", "on": "a.id = b.id"}]
    ir = compute_complexity(ir)
    assert ir["complexity_score"] == 2
    assert ir["complexity_label"] == "moderate"


def test_complexity_3_with_window_functions():
    """Window functions = complexity 3 complex."""
    ir = build_empty_ir()
    ir["window_functions"] = ["RANK"]
    ir = compute_complexity(ir)
    assert ir["complexity_score"] == 3
    assert ir["complexity_label"] == "complex"


def test_complexity_4_with_udf():
    """UDF present = complexity 4 expert."""
    ir = build_empty_ir()
    ir["has_udf"] = True
    ir = compute_complexity(ir)
    assert ir["complexity_score"] == 4
    assert ir["complexity_label"] == "expert"


def test_complexity_4_with_procedural_flags():
    """Procedural flags = complexity 4 expert."""
    ir = build_empty_ir()
    ir["procedural_flags"] = ["CURSOR loop detected — manual review required"]
    ir = compute_complexity(ir)
    assert ir["complexity_score"] == 4
    assert ir["complexity_label"] == "expert"


def test_merge_ir_fields_combines_lists():
    """Merge combines tables and columns from both IRs without duplicates."""
    base = build_empty_ir()
    base["tables"] = ["orders", "customers"]
    base["columns"] = ["id", "name"]

    extra = build_empty_ir()
    extra["tables"] = ["customers", "products"]
    extra["columns"] = ["name", "price"]

    merged = merge_ir_fields(base, extra)
    assert "orders" in merged["tables"]
    assert "products" in merged["tables"]
    assert merged["tables"].count("customers") == 1  # no duplicates
    assert merged["columns"].count("name") == 1


def test_validate_ir_catches_missing_field():
    """validate_ir adds a parse_error when a required field is missing."""
    ir = build_empty_ir()
    del ir["tables"]
    ir = validate_ir(ir)
    assert any("tables" in e for e in ir["parse_errors"])
    assert isinstance(ir["tables"], list)  # restored to safe default