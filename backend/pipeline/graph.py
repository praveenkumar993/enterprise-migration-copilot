"""
LangGraph Pipeline — Enterprise Migration Copilot
9-node linear pipeline: route → analyze → retrieve → migrate →
validate → review → optimize → risk → finalize

Each node has try/except — pipeline never crashes, always returns state.
LangSmith tracing via @traceable on each node.
Prometheus metrics recorded per-agent and at finalize.
"""

import time
import os
from typing import Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langsmith import traceable
from utils.metrics import record_migration, record_agent
from parsers.language_router import route as language_router_route
from agents.analyzer import analyze
from agents.migrator import migrate
from agents.validator import validate
from agents.reviewer import review
from agents.optimizer import optimize
from agents.risk_compliance import check_risk
from rag.hybrid_retriever import HybridRetriever

load_dotenv()

# Initialize RAG retriever once at module level
_retriever = None


def get_retriever() -> HybridRetriever:
    """Lazy-initialize the hybrid retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------- State Definition ----------

class PipelineState(TypedDict):
    # Input
    source: str
    source_language: str
    dialect: str

    # Stage outputs
    ir: dict
    analyzer_output: dict
    rag_context: list
    pyspark_code: str
    validation: dict
    review_result: dict
    optimization: dict
    risk_result: dict

    # Meta
    status: str
    error: str
    processing_time_ms: float
    start_time: float


# ---------- Pipeline Nodes ----------

@traceable(name="node_route")
def node_route(state: PipelineState) -> PipelineState:
    """Parse source code into unified IR using language router."""
    node_start = time.time()
    try:
        source = state.get("source", "")
        dialect = state.get("dialect", "")

        ir = language_router_route(source, dialect=dialect)
        state["ir"] = ir
        state["source_language"] = ir.get("source_language", "sql")

        record_agent("route", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_route failed: {str(e)}"
        state["status"] = "error"
        record_agent("route", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_analyze")
def node_analyze(state: PipelineState) -> PipelineState:
    """Analyze IR and produce migration strategy."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        ir = state.get("ir", {})
        analyzer_output = analyze(ir)
        state["analyzer_output"] = analyzer_output

        record_agent("analyzer", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_analyze failed: {str(e)}"
        state["status"] = "error"
        record_agent("analyzer", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_retrieve")
def node_retrieve(state: PipelineState) -> PipelineState:
    """Retrieve relevant PySpark patterns from hybrid RAG."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        retriever = get_retriever()
        analyzer_output = state.get("analyzer_output", {})
        ir = state.get("ir", {})

        # Primary retrieval using RAG query from analyzer
        rag_query = analyzer_output.get("rag_query", "migrate sql to pyspark")
        rag_context = retriever.retrieve(rag_query, top_k=5)

        # If procedural — also retrieve language-specific patterns
        if analyzer_output.get("needs_procedural_context"):
            source_language = ir.get("source_language", "sql")
            procedural_query = (
                f"{source_language} cursor loop exception handler "
                f"procedural migration pyspark"
            )
            procedural_context = retriever.retrieve(procedural_query, top_k=3)

            # Merge — deduplicate by text prefix
            seen = {r["text"][:80] for r in rag_context}
            for item in procedural_context:
                key = item["text"][:80]
                if key not in seen:
                    rag_context.append(item)
                    seen.add(key)

        state["rag_context"] = rag_context

        record_agent("retrieve", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_retrieve failed: {str(e)}"
        state["status"] = "error"
        record_agent("retrieve", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_migrate")
def node_migrate(state: PipelineState) -> PipelineState:
    """Generate PySpark code using fine-tuned LLM."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        ir = state.get("ir", {})
        analyzer_output = state.get("analyzer_output", {})
        rag_context = state.get("rag_context", [])

        result = migrate(ir, analyzer_output, rag_context)
        state["pyspark_code"] = result.get("pyspark_code", "")

        record_agent("migrator", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_migrate failed: {str(e)}"
        state["status"] = "error"
        record_agent("migrator", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_validate")
def node_validate(state: PipelineState) -> PipelineState:
    """Validate generated PySpark code against IR."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        ir = state.get("ir", {})
        pyspark_code = state.get("pyspark_code", "")

        validation = validate(ir, pyspark_code)
        state["validation"] = validation

        record_agent("validator", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_validate failed: {str(e)}"
        state["status"] = "error"
        record_agent("validator", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_review")
def node_review(state: PipelineState) -> PipelineState:
    """Semantic review of generated PySpark code."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        ir = state.get("ir", {})
        pyspark_code = state.get("pyspark_code", "")

        review_result = review(ir, pyspark_code)
        state["review_result"] = review_result

        record_agent("reviewer", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_review failed: {str(e)}"
        state["status"] = "error"
        record_agent("reviewer", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_optimize")
def node_optimize(state: PipelineState) -> PipelineState:
    """Detect anti-patterns and suggest optimizations."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        pyspark_code = state.get("pyspark_code", "")
        optimization = optimize(pyspark_code)
        state["optimization"] = optimization

        record_agent("optimizer", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_optimize failed: {str(e)}"
        state["status"] = "error"
        record_agent("optimizer", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_risk")
def node_risk(state: PipelineState) -> PipelineState:
    """Run risk and compliance checks."""
    if state.get("status") == "error":
        return state
    node_start = time.time()
    try:
        ir = state.get("ir", {})
        pyspark_code = state.get("pyspark_code", "")

        risk = check_risk(ir, pyspark_code)
        state["risk_result"] = risk

        # Record each procedural flag for the Grafana panel
        from utils.metrics import record_procedural_flag
        for flag in risk.get("compliance_flags", []):
            record_procedural_flag(flag)

        record_agent("risk_compliance", (time.time() - node_start) * 1000)
        return state

    except Exception as e:
        state["error"] = f"node_risk failed: {str(e)}"
        state["status"] = "error"
        record_agent("risk_compliance", (time.time() - node_start) * 1000)
        return state


@traceable(name="node_finalize")
def node_finalize(state: PipelineState) -> PipelineState:
    """Set final status, processing time, and record overall migration metrics."""
    try:
        validation = state.get("validation", {})
        review_result = state.get("review_result", {})
        risk_result = state.get("risk_result", {})

        validation_passed = validation.get("passed", False)
        semantic_score = review_result.get("semantic_score", 0.0)

        if validation_passed and semantic_score >= 0.5:
            status = "success"
        elif validation_passed:
            status = "low_confidence"
        else:
            status = "failed"

        # Only set status if not already in error
        if state.get("status") != "error":
            state["status"] = status

        # Processing time
        start = state.get("start_time", time.time())
        processing_time_ms = round((time.time() - start) * 1000, 2)
        state["processing_time_ms"] = processing_time_ms

        # Record overall migration metrics
        record_migration(
            status=state.get("status", "unknown"),
            source_language=state.get("source_language", "unknown"),
            duration_ms=processing_time_ms,
            risk_score=risk_result.get("risk_score", 0.0),
            validation_score=validation.get("validation_score", 0.0),
        )

        return state

    except Exception as e:
        state["error"] = f"node_finalize failed: {str(e)}"
        state["status"] = "error"
        return state


# ---------- Build Graph ----------

def build_graph() -> Any:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("route", node_route)
    graph.add_node("analyze", node_analyze)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("migrate", node_migrate)
    graph.add_node("validate", node_validate)
    graph.add_node("review", node_review)
    graph.add_node("optimize", node_optimize)
    graph.add_node("risk", node_risk)
    graph.add_node("finalize", node_finalize)

    # Linear edges
    graph.set_entry_point("route")
    graph.add_edge("route", "analyze")
    graph.add_edge("analyze", "retrieve")
    graph.add_edge("retrieve", "migrate")
    graph.add_edge("migrate", "validate")
    graph.add_edge("validate", "review")
    graph.add_edge("review", "optimize")
    graph.add_edge("optimize", "risk")
    graph.add_edge("risk", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# Compile once at module level
# Lazy initialization — compile on first request, not at import time
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_graph()
    return _pipeline


# ---------- Public API ----------

def run_pipeline(
    source: str,
    source_language: str = "auto",
    dialect: str = "",
) -> dict[str, Any]:
    """
    Run the full 9-node migration pipeline.

    Args:
        source: Raw source code (SQL, HiveQL, PL/SQL, or Stored Procedure)
        source_language: Language hint or "auto" for auto-detection
        dialect: SQL dialect hint (optional)

    Returns:
        Full pipeline state dict with all agent outputs
    """
    initial_state: PipelineState = {
        "source": source,
        "source_language": source_language,
        "dialect": dialect,
        "ir": {},
        "analyzer_output": {},
        "rag_context": [],
        "pyspark_code": "",
        "validation": {},
        "review_result": {},
        "optimization": {},
        "risk_result": {},
        "status": "running",
        "error": "",
        "processing_time_ms": 0.0,
        "start_time": time.time(),
    }

    try:
        final_state = get_pipeline().invoke(initial_state)
        # Remove internal fields before returning
        final_state.pop("start_time", None)
        result = dict(final_state)
        # Normalize keys for API consistency
        result["risk"] = result.get("risk_result", {})
        result["review"] = result.get("review_result", {})
        return result

    except Exception as e:
        return {
            "source": source,
            "status": "error",
            "error": f"Pipeline invocation failed: {str(e)}",
            "processing_time_ms": 0.0,
            "pyspark_code": "",
        }