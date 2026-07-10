"""
Enterprise Migration Copilot — Lightweight API for free tier deployment.
Skips ChromaDB, LangGraph, and embedding model.
Runs parser + agents directly without the full pipeline orchestration.
Uses HF Inference API for LLM generation only.
"""

import os
import time
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

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)


# ---------- Request Models ----------

class MigrateRequest(BaseModel):
    source: str
    source_language: str = "auto"
    dialect: str = ""


class MigrateResponse(BaseModel):
    status: str
    source_language: str
    detected_language: str
    pyspark_code: str
    model_used: str
    validation_score: float
    validation_issues: list[str]
    complexity: str
    risk_score: float
    risk_level: str
    pii_columns: list[str]
    compliance_flags: list[str]
    procedural_flags: list[str]
    anti_patterns: list[dict]
    performance_notes: list[str]
    estimated_review_time: str
    processing_time_ms: float
    error: str


class ParseRequest(BaseModel):
    source: str
    dialect: str = "ansi"


class DetectResponse(BaseModel):
    detected_language: str
    source_preview: str


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "service": "Enterprise Migration Copilot",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect", response_model=DetectResponse)
def detect_source_language(request: ParseRequest):
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")
    from parsers.language_router import detect_language
    language = detect_language(request.source)
    return {"detected_language": language, "source_preview": request.source.strip()[:100]}


@app.post("/migrate", response_model=MigrateResponse)
def migrate_source(request: MigrateRequest):
    """
    Lightweight migration pipeline for free tier deployment.
    Runs parser → analyzer → migrator → validator → optimizer → risk
    without LangGraph orchestration or ChromaDB RAG.
    """
    if not request.source or not request.source.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty")

    start_time = time.time()

    try:
        # Step 1 — Parse
        from parsers.language_router import route
        ir = route(request.source, dialect=request.dialect)
        source_language = ir.get("source_language", "sql")

        # Step 2 — Analyze
        from agents.analyzer import analyze
        analyzer_output = analyze(ir)

        # Step 3 — Migrate (no RAG context — empty list)
        from agents.migrator import migrate
        migrator_output = migrate(ir, analyzer_output, rag_context=[])
        pyspark_code = migrator_output.get("pyspark_code", "")
        model_used = migrator_output.get("model_used", "unknown")

        # Step 4 — Validate
        from agents.validator import validate
        validation = validate(ir, pyspark_code)

        # Step 5 — Optimize
        from agents.optimizer import optimize
        optimization = optimize(pyspark_code)

        # Step 6 — Risk
        from agents.risk_compliance import check_risk
        risk = check_risk(ir, pyspark_code)

        # Determine status
        validation_passed = validation.get("passed", False)
        status = "success" if validation_passed else "low_confidence"

        # Flatten response
        raw_score = validation.get("validation_score", 0)
        if isinstance(raw_score, float) and raw_score <= 1.0:
            validation_score = round(raw_score * 100)
        else:
            validation_score = int(raw_score)

        validation_issues = validation.get("issues", validation.get("errors", []))
        risk_score = float(risk.get("risk_score", 0.0))
        risk_level = risk.get("risk_level", "LOW").upper()
        pii_columns = risk.get("pii_columns", [])
        compliance_flags = risk.get("compliance_flags", [])
        procedural_flags = analyzer_output.get("procedural_flags", [])

        anti_patterns_raw = optimization.get("anti_patterns", [])
        anti_patterns = []
        for ap in anti_patterns_raw:
            if isinstance(ap, str):
                # Try to parse if it looks like a dict string
                try:
                    import ast
                    ap = ast.literal_eval(ap)
                except Exception:
                    anti_patterns.append({"pattern": ap, "suggestion": ""})
                    continue
            if isinstance(ap, dict):
                anti_patterns.append({
                    "pattern": ap.get("pattern", ap.get("name", str(ap))),
                    "suggestion": ap.get("suggestion", ap.get("fix", "")),
                })

        performance_notes = optimization.get("performance_notes",
                           optimization.get("suggestions", []))
        estimated_review_time = analyzer_output.get("estimated_review_time", "")
        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": status,
            "source_language": source_language,
            "detected_language": source_language,
            "pyspark_code": pyspark_code,
            "model_used": model_used,
            "validation_score": validation_score,
            "validation_issues": validation_issues,
            "complexity": analyzer_output.get("complexity", "moderate"),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "pii_columns": pii_columns,
            "compliance_flags": compliance_flags,
            "procedural_flags": procedural_flags,
            "anti_patterns": anti_patterns,
            "performance_notes": performance_notes,
            "estimated_review_time": estimated_review_time,
            "processing_time_ms": processing_time_ms,
            "error": "",
        }

    except Exception as e:
        return {
            "status": "error",
            "source_language": request.source_language,
            "detected_language": request.source_language,
            "pyspark_code": "",
            "model_used": "none",
            "validation_score": 0,
            "validation_issues": [],
            "complexity": "unknown",
            "risk_score": 0.0,
            "risk_level": "LOW",
            "pii_columns": [],
            "compliance_flags": [],
            "procedural_flags": [],
            "anti_patterns": [],
            "performance_notes": [],
            "estimated_review_time": "",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e),
        }
@app.get("/debug-hf")
def debug_hf():
    """Temporary debug endpoint — remove after fixing HF API issue."""
    import requests
    import os
    
    token = os.getenv("HF_TOKEN", "")
    username = os.getenv("HF_USERNAME", "praveends")
    results = {}
    results["token_present"] = bool(token)
    results["token_prefix"] = token[:8] + "..." if token else "MISSING"
    results["username"] = username
    
    try:
        r = requests.get("https://huggingface.co", timeout=10)
        results["hf_main_site"] = r.status_code
    except Exception as e:
        results["hf_main_site"] = str(e)
    
    try:
        r = requests.get(
            "https://api-inference.huggingface.co/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        results["hf_inference_api"] = r.status_code
        results["hf_inference_response"] = r.text[:200]
    except Exception as e:
        results["hf_inference_api"] = str(e)
    
    model = f"{username}/migration-copilot-phi-3-5-mini-instruct"
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={"inputs": "SELECT 1", "parameters": {"max_new_tokens": 10}},
            timeout=30
        )
        results["model_status"] = r.status_code
        results["model_response"] = r.text[:300]
    except Exception as e:
        results["model_endpoint"] = str(e)

    # Test newer router endpoint
    try:
        r = requests.get(
            "https://router.huggingface.co",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        results["hf_router"] = r.status_code
        results["hf_router_response"] = r.text[:200]
    except Exception as e:
        results["hf_router"] = str(e)
    
    return results