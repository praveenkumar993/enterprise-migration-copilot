"""
Enterprise Migration Copilot — FastAPI Backend
Entry point. Exposes /parse, /detect, /retrieve, /migrate endpoints.
Full agent pipeline added in Phase 5.
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Enterprise Migration Copilot",
    description="AI-powered multi-source enterprise data transformation platform",
    version="0.1.0",
)

# CORS — locked to known origins in production, open in development
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*",  # overridden in production via env var to your Vercel URL
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

from pipeline.graph import run_pipeline
from parsers.language_router import route, detect_language


# ---------- Request / Response Models ----------

class ParseRequest(BaseModel):
    source: str
    dialect: str = "ansi"


class ParseResponse(BaseModel):
    source_language: str
    dialect: str
    tables: list[str]
    columns: list[str]
    joins: list[dict]
    aggregations: list[str]
    window_functions: list[str]
    ctes: list[str]
    subquery_depth: int
    has_udf: bool
    complexity_score: int
    complexity_label: str
    procedural_flags: list[str]
    parse_errors: list[str]


class DetectResponse(BaseModel):
    detected_language: str
    source_preview: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrieveResult(BaseModel):
    text: str
    source: str
    score: float


class RetrieveResponse(BaseModel):
    results: list[RetrieveResult]
    query: str
    total: int


class MigrateRequest(BaseModel):
    source: str
    source_language: str = "auto"
    dialect: str = ""


class MigrateResponse(BaseModel):
    # Core output
    status: str
    source_language: str
    detected_language: str
    pyspark_code: str
    model_used: str

    # Validation card
    validation_score: float
    validation_issues: list[str]
    complexity: str

    # Risk card
    risk_score: float
    risk_level: str
    pii_columns: list[str]
    compliance_flags: list[str]

    # Migration notes card
    procedural_flags: list[str]
    anti_patterns: list[dict]
    performance_notes: list[str]
    estimated_review_time: str

    # Meta
    processing_time_ms: float
    error: str


# ---------- Routes ----------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Enterprise Migration Copilot",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/parse", response_model=ParseResponse)
def parse_source(request: ParseRequest) -> Any:
    """Parse source code into unified IR."""
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")
    try:
        ir = route(request.source, dialect=request.dialect)
        return ir
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse failed: {str(e)}")


@app.post("/detect", response_model=DetectResponse)
def detect_source_language(request: ParseRequest) -> Any:
    """Detect source language without full parsing."""
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")
    language = detect_language(request.source)
    preview = request.source.strip()[:100]
    return {"detected_language": language, "source_preview": preview}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(request: RetrieveRequest) -> Any:
    """Retrieve relevant PySpark migration patterns using hybrid RAG."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        from rag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        results = retriever.retrieve(request.query, top_k=request.top_k)
        return {"results": results, "query": request.query, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@app.post("/migrate", response_model=MigrateResponse)
def migrate_source(request: MigrateRequest) -> Any:
    """
    Full 9-node migration pipeline.
    Returns flattened response matching frontend card structure.
    """
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")

    try:
        result = run_pipeline(
            source=request.source,
            source_language=request.source_language,
            dialect=request.dialect,
        )

        # --- Extract nested agent outputs ---
        validation = result.get("validation", {})
        risk = result.get("risk", result.get("risk_result", {}))
        optimization = result.get("optimization", {})
        analyzer = result.get("analyzer_output", {})
        review = result.get("review", result.get("review_result", {}))
        migrator_out = result.get("migrator_output", {})

        # --- Validation card ---
        raw_score = validation.get("validation_score", 0)
        # normalize to 0-100 if returned as 0.0-1.0
        if isinstance(raw_score, float) and raw_score <= 1.0:
            validation_score = round(raw_score * 100)
        else:
            validation_score = int(raw_score)

        validation_issues = validation.get("issues", [])
        if not validation_issues:
            validation_issues = validation.get("errors", [])

        # --- Risk card ---
        risk_score = float(risk.get("risk_score", 0.0))
        risk_level = risk.get("risk_level", "LOW").upper()
        pii_columns = risk.get("pii_columns", [])
        compliance_flags = risk.get("compliance_flags", [])

        # --- Migration notes card ---
        procedural_flags = (
            analyzer.get("procedural_flags", [])
            or result.get("ir", {}).get("procedural_flags", [])
        )

        anti_patterns_raw = optimization.get("anti_patterns", [])
        # Normalize — some agents return strings, others return dicts
        anti_patterns = []
        for ap in anti_patterns_raw:
            if isinstance(ap, str):
                anti_patterns.append({"pattern": ap, "suggestion": ""})
            elif isinstance(ap, dict):
                anti_patterns.append({
                    "pattern": ap.get("pattern", ap.get("name", str(ap))),
                    "suggestion": ap.get("suggestion", ap.get("fix", "")),
                })

        performance_notes = optimization.get("performance_notes", [])
        if not performance_notes:
            performance_notes = optimization.get("suggestions", [])

        estimated_review_time = analyzer.get("estimated_review_time", "")

        # --- Model used ---
        model_used = result.get("model_used", "unknown")

        return {
            "status": result.get("status", "error"),
            "source_language": result.get("source_language", "unknown"),
            "detected_language": result.get("source_language", "unknown"),
            "pyspark_code": result.get("pyspark_code", ""),
            "model_used": model_used,

            "validation_score": validation_score,
            "validation_issues": validation_issues,
            "complexity": analyzer.get("complexity", "medium"),

            "risk_score": risk_score,
            "risk_level": risk_level,
            "pii_columns": pii_columns,
            "compliance_flags": compliance_flags,

            "procedural_flags": procedural_flags,
            "anti_patterns": anti_patterns,
            "performance_notes": performance_notes,
            "estimated_review_time": estimated_review_time,

            "processing_time_ms": result.get("processing_time_ms", 0.0),
            "error": result.get("error", ""),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")