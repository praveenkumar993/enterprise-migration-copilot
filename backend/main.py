"""
Enterprise Migration Copilot — FastAPI Backend
Entry point. Exposes /parse endpoint for Phase 1.0.
Full agent pipeline added in Phase 5.
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from parsers.language_router import route, detect_language

load_dotenv()

app = FastAPI(
    title="Enterprise Migration Copilot",
    description="AI-powered multi-source enterprise data transformation platform",
    version="0.1.0",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------- Routes ----------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Enterprise Migration Copilot",
        "version": "0.1.0",
        "status": "running",
        "phase": "1.0 — Parsers and IR",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/parse", response_model=ParseResponse)
def parse_source(request: ParseRequest) -> Any:
    """
    Parse source code into unified IR.
    Auto-detects language (SQL, HiveQL, PL/SQL, Stored Procedure).
    """
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")

    try:
        ir = route(request.source, dialect=request.dialect)
        return ir
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse failed: {str(e)}")


@app.post("/detect", response_model=DetectResponse)
def detect_source_language(request: ParseRequest) -> Any:
    """
    Detect source language without full parsing.
    Fast endpoint for frontend language indicator.
    """
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")

    language = detect_language(request.source)
    preview = request.source.strip()[:100]
    return {
        "detected_language": language,
        "source_preview": preview,
    }
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


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_context(request: RetrieveRequest) -> Any:
    """
    Retrieve relevant PySpark migration patterns using hybrid RAG.
    Used internally by the Analyzer agent.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        from rag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        results = retriever.retrieve(request.query, top_k=request.top_k)
        return {
            "results": results,
            "query": request.query,
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")