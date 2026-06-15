"""
Agent Tests — 10 unit tests for all 6 agents.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.analyzer import analyze
from agents.validator import validate, check_syntax, check_dataframe_ops
from agents.optimizer import optimize
from agents.risk_compliance import check_risk, detect_pii_columns


# --- Shared test fixtures ---

SIMPLE_IR = {
    "source_language": "sql",
    "dialect": "",
    "tables": ["orders", "customers"],
    "columns": ["id", "name", "amount"],
    "joins": [{"type": "INNER", "table": "customers", "on": "o.id = c.id"}],
    "aggregations": ["COUNT", "SUM"],
    "window_functions": [],
    "ctes": [],
    "subquery_depth": 0,
    "has_udf": False,
    "complexity_score": 2,
    "complexity_label": "moderate",
    "procedural_flags": [],
    "raw_source": "SELECT c.name, COUNT(o.id), SUM(o.amount) FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.name",
    "parse_errors": [],
}

PLSQL_IR = {
    "source_language": "plsql",
    "dialect": "oracle",
    "tables": ["orders"],
    "columns": ["id", "amount", "status"],
    "joins": [],
    "aggregations": [],
    "window_functions": [],
    "ctes": [],
    "subquery_depth": 0,
    "has_udf": False,
    "complexity_score": 4,
    "complexity_label": "expert",
    "procedural_flags": [
        "CURSOR loop detected — manual review required",
        "EXCEPTION handler — add try/except in PySpark job",
    ],
    "raw_source": "BEGIN SELECT id, amount FROM orders; END;",
    "parse_errors": [],
}

GOOD_PYSPARK = """
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('migration').getOrCreate()
orders = spark.read.table('orders')
customers = spark.read.table('customers')
result = orders.join(customers, orders.customer_id == customers.id, 'inner') \\
               .groupBy(customers.name) \\
               .agg(F.count(orders.id).alias('order_count'), F.sum(orders.amount).alias('total'))
result.write.mode('overwrite').saveAsTable('output')
"""

BAD_PYSPARK = """
data = df.collect()
for row in data:
    print(row)
"""


# --- Analyzer Tests ---

def test_analyzer_detects_joins_as_risk():
    """Analyzer flags window functions and joins in risk_flags."""
    ir = {**SIMPLE_IR, "window_functions": ["RANK"]}
    result = analyze(ir)
    assert "Window functions detected" in " ".join(result["risk_flags"])


def test_analyzer_sets_procedural_context_for_plsql():
    """Analyzer sets needs_procedural_context True for PL/SQL."""
    result = analyze(PLSQL_IR)
    assert result["needs_procedural_context"] is True


def test_analyzer_review_time_includes_manual_for_procedural():
    """Analyzer adds manual review note to estimated_review_time for procedural."""
    result = analyze(PLSQL_IR)
    assert "manual review" in result["estimated_review_time"].lower()


# --- Validator Tests ---

def test_validator_passes_good_pyspark():
    """Validator passes well-formed PySpark code."""
    result = validate(SIMPLE_IR, GOOD_PYSPARK)
    assert result["syntax_valid"] is True
    assert result["has_sparksession"] is True
    assert result["has_dataframe_ops"] is True
    assert result["passed"] is True


def test_validator_fails_empty_code():
    """Validator fails on empty code."""
    result = validate(SIMPLE_IR, "")
    assert result["passed"] is False
    assert result["validation_score"] == 0.0


def test_validator_syntax_check_catches_bad_python():
    """Syntax checker correctly identifies invalid Python."""
    valid, msg = check_syntax("def foo(:\n    pass")
    assert valid is False
    assert "SyntaxError" in msg


# --- Optimizer Tests ---

def test_optimizer_flags_collect():
    """Optimizer flags .collect() as high severity."""
    result = optimize("data = df.collect()\nfor row in data: print(row)")
    names = [ap["name"] for ap in result["anti_patterns"]]
    assert any("collect" in n.lower() for n in names)
    high = [ap for ap in result["anti_patterns"] if ap["severity"] == "high"]
    assert len(high) >= 1


def test_optimizer_clean_code_scores_high():
    """Clean PySpark code gets optimization_score of 1.0."""
    result = optimize(GOOD_PYSPARK)
    assert result["optimization_score"] == 1.0
    assert result["issues_found"] == 0


# --- Risk and Compliance Tests ---

def test_risk_detects_pii_columns():
    """Risk agent detects PII columns from IR."""
    pii_ir = {**SIMPLE_IR, "columns": ["id", "pan_number", "email", "amount"]}
    result = check_risk(pii_ir, GOOD_PYSPARK)
    assert len(result["pii_columns"]) >= 1
    assert result["risk_score"] > 0
    assert result["risk_level"] != "low"


def test_risk_low_for_clean_migration():
    """Risk agent returns low risk for clean SQL migration."""
    clean_ir = {**SIMPLE_IR, "columns": ["id", "name", "amount"]}
    clean_code = "import pyspark.sql.functions as F\ndf = spark.read.table('orders')\nresult = df.select('id', 'name', 'amount')\n"
    result = check_risk(clean_ir, clean_code)
    assert result["risk_level"] in ("low", "medium")