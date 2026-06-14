"""
Phase 1.0 Parser Tests — 10 unit tests.
Run with: pytest tests/test_parsers.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from parsers.sql_parser import parse_sql
from parsers.hiveql_parser import parse_hiveql
from parsers.language_router import detect_language, route


# ---------- Test SQL Parser ----------

def test_simple_select_complexity_1():
    """Simple SELECT with no joins or aggregations = complexity 1."""
    sql = "SELECT id, name FROM customers"
    ir = parse_sql(sql)
    assert ir["complexity_score"] == 1
    assert ir["complexity_label"] == "simple"
    assert ir["source_language"] == "sql"


def test_sql_join_detected():
    """INNER JOIN is detected with correct type."""
    sql = """
        SELECT o.id, c.name
        FROM orders o
        INNER JOIN customers c ON o.customer_id = c.id
    """
    ir = parse_sql(sql)
    assert len(ir["joins"]) == 1
    assert "INNER" in ir["joins"][0]["type"]
    assert ir["joins"][0]["table"] == "customers"


def test_sql_window_function_complexity_3():
    """Window function (RANK) raises complexity to 3."""
    sql = """
        SELECT c.name, COUNT(o.id) as order_count,
               SUM(o.amount) as total_spent,
               RANK() OVER (PARTITION BY c.region ORDER BY SUM(o.amount) DESC) as rnk
        FROM orders o
        INNER JOIN customers c ON o.customer_id = c.id
        WHERE o.created_at >= '2024-01-01'
        GROUP BY c.name, c.region
        ORDER BY total_spent DESC
    """
    ir = parse_sql(sql)
    assert ir["complexity_score"] == 3
    assert len(ir["window_functions"]) >= 1
    assert "RANK" in ir["window_functions"]


def test_sql_aggregations_detected():
    """COUNT and SUM aggregations are both detected."""
    sql = """
        SELECT c.name, COUNT(o.id) as order_count, SUM(o.amount) as total_spent
        FROM orders o
        INNER JOIN customers c ON o.customer_id = c.id
        GROUP BY c.name
    """
    ir = parse_sql(sql)
    assert "COUNT" in ir["aggregations"]
    assert "SUM" in ir["aggregations"]


def test_sql_tables_extracted():
    """Table names are correctly extracted."""
    sql = "SELECT o.id FROM orders o JOIN customers c ON o.customer_id = c.id"
    ir = parse_sql(sql)
    assert "orders" in ir["tables"]
    assert "customers" in ir["tables"]


def test_sql_invalid_returns_parse_errors_not_exception():
    """Invalid SQL returns parse_errors list, does not raise exception."""
    sql = "THIS IS NOT VALID SQL @@##"
    try:
        ir = parse_sql(sql)
        # Should not raise — parse_errors may be populated
        assert isinstance(ir["parse_errors"], list)
    except Exception:
        pytest.fail("parse_sql raised an exception on invalid SQL — should return IR with parse_errors")


def test_all_dialects_parse_without_crashing():
    """All 6 SQL dialects parse the same query without crashing."""
    sql = "SELECT id, name, amount FROM transactions WHERE amount > 1000"
    dialects = ["ansi", "mysql", "postgres", "bigquery", "snowflake", "spark"]
    for dialect in dialects:
        try:
            ir = parse_sql(sql, dialect=dialect)
            assert ir["source_language"] == "sql"
        except Exception as e:
            pytest.fail(f"parse_sql crashed on dialect {dialect}: {e}")


# ---------- Test HiveQL Parser ----------

def test_hiveql_source_language_set():
    """HiveQL parser sets source_language to hiveql."""
    hiveql = """
        SELECT customer_id, product_id, spend
        FROM transactions
        LATERAL VIEW EXPLODE(product_list) tmp AS product_id
        DISTRIBUTE BY customer_id
        SORT BY spend DESC
    """
    ir = parse_hiveql(hiveql)
    assert ir["source_language"] == "hiveql"


def test_hiveql_lateral_view_detected():
    """LATERAL VIEW EXPLODE is detected and noted."""
    hiveql = """
        SELECT customer_id, product_id
        FROM transactions
        LATERAL VIEW EXPLODE(product_list) tmp AS product_id
    """
    ir = parse_hiveql(hiveql)
    notes = " ".join(ir["parse_errors"])
    assert "LATERAL VIEW" in notes.upper() or "EXPLODE" in notes.upper()


# ---------- Test Language Router ----------

def test_router_detects_hiveql_from_distribute_by():
    """Language router detects HiveQL from DISTRIBUTE BY keyword."""
    source = "SELECT id FROM events DISTRIBUTE BY customer_id"
    assert detect_language(source) == "hiveql"


def test_router_detects_stored_procedure():
    """Language router detects Stored Procedure from CREATE PROCEDURE."""
    source = "CREATE PROCEDURE UpdateStatus AS BEGIN SELECT 1 END"
    assert detect_language(source) == "stored_procedure"


def test_router_defaults_to_sql_for_plain_select():
    """Language router defaults to sql for a plain SELECT."""
    source = "SELECT id, name FROM customers WHERE active = 1"
    assert detect_language(source) == "sql"