"""
Migrator Agent — LLM-powered PySpark code generation.
Calls HuggingFace Inference API with fine-tuned Phi-3.5-mini model.
Falls back to Qwen2.5-Coder if fine-tuned model is cold-starting.
"""

import os
import time
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveends")

# Primary: our best fine-tuned model (Phi-3.5-mini, 57% benchmark pass rate)
PRIMARY_MODEL = f"{HF_USERNAME}/migration-copilot-phi-3-5-mini-instruct"

# Fallback 1: our second-best fine-tuned model (Qwen2.5, 45% pass rate)
FALLBACK_MODEL_1 = f"{HF_USERNAME}/migration-copilot-qwen2-5-coder-1-5b-instruct"

# Fallback 2: public base model — always available, no fine-tuning
FALLBACK_MODEL_2 = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

HF_API_BASE = "https://api-inference.huggingface.co/models"

MAX_RETRIES = 3
RETRY_WAIT = 10  # seconds


def build_prompt(
    ir: dict[str, Any],
    analyzer_output: dict[str, Any],
    rag_context: list[dict],
) -> str:
    """
    Build the instruction prompt matching the exact format used during fine-tuning.
    Using the same ### Instruction / ### Input / ### Response format is critical —
    the fine-tuned model was trained on this exact structure.
    """
    source_language = ir.get("source_language", "sql").upper()
    difficulty = analyzer_output.get("complexity", "medium")
    raw_source = ir.get("raw_source", "")
    procedural_flags = analyzer_output.get("procedural_flags", [])

    # Format RAG context as inline hints
    rag_text = ""
    if rag_context:
        rag_snippets = [r.get("text", "") for r in rag_context[:3]]
        rag_text = "\n\n".join(rag_snippets)

    # Procedural warning for cursors, exceptions, temp tables etc.
    procedural_note = ""
    if procedural_flags:
        flags_str = "\n".join(f"- {f}" for f in procedural_flags[:5])
        procedural_note = (
            f"\nNote: The following procedural constructs require manual review:\n"
            f"{flags_str}\n"
            f"Add a # NOTE comment in your PySpark code for each flagged construct.\n"
        )

    # RAG context note
    rag_note = f"\nReference PySpark patterns:\n{rag_text}\n" if rag_text else ""

    # This prompt format matches EXACTLY what was used during fine-tuning
    prompt = f"""### Instruction:
Convert the following {source_language} code to PySpark.
Difficulty: {difficulty}
{procedural_note}{rag_note}
### Input:
{raw_source}

### Response:
"""
    return prompt


def call_hf_api(model: str, prompt: str) -> tuple[str, bool]:
    """
    Call HuggingFace Inference API with retry on cold start (503).

    Returns:
        (generated_text, success) tuple
    """
    url = f"{HF_API_BASE}/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.1,
            "return_full_text": False,
            "do_sample": False,
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            # Model cold start — wait and retry
            if response.status_code == 503:
                try:
                    body = response.json()
                    wait = min(body.get("estimated_time", RETRY_WAIT), 30)
                except Exception:
                    wait = RETRY_WAIT
                time.sleep(wait)
                continue

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "")
                    return text, bool(text.strip())
                elif isinstance(data, dict):
                    text = data.get("generated_text", "")
                    return text, bool(text.strip())

            # Non-retryable error (404 model not found, 401 auth etc.)
            if response.status_code in (401, 404):
                return "", False

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)

        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)

    return "", False


def clean_output(raw: str) -> str:
    """
    Strip any repeated prompt artifacts from model output.
    Some models echo the ### Response: marker or repeat the input.
    """
    # Cut off at second ### if model starts repeating
    if "### Instruction:" in raw:
        raw = raw[:raw.index("### Instruction:")].strip()
    if "### Input:" in raw:
        raw = raw[:raw.index("### Input:")].strip()
    # Strip leading/trailing whitespace and markdown fences
    raw = raw.strip()
    if raw.startswith("```python"):
        raw = raw[9:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def migrate(
    ir: dict[str, Any],
    analyzer_output: dict[str, Any],
    rag_context: list[dict],
) -> dict[str, Any]:
    """
    Generate PySpark code from IR using fine-tuned LLM.

    Model priority:
    1. Phi-3.5-mini fine-tuned (best, 57% benchmark pass rate)
    2. Qwen2.5-Coder fine-tuned (fallback, 45% pass rate)
    3. Qwen2.5-Coder base (public, always available)
    4. Stub with TODO comment (last resort)

    Args:
        ir: Unified IR dict from parser
        analyzer_output: Output from Analyzer agent
        rag_context: Retrieved PySpark patterns from RAG

    Returns:
        Dict with pyspark_code, model_used, raw_response
    """
    prompt = build_prompt(ir, analyzer_output, rag_context)

    if not HF_TOKEN:
        return _stub_response(prompt, reason="HF_TOKEN not set")

    # Try each model in priority order
    for model, label in [
        (PRIMARY_MODEL, "phi-3.5-mini-finetuned"),
        (FALLBACK_MODEL_1, "qwen2.5-finetuned"),
        (FALLBACK_MODEL_2, "qwen2.5-base"),
    ]:
        code, success = call_hf_api(model, prompt)
        if success and code.strip():
            cleaned = clean_output(code)
            if len(cleaned) > 20:  # sanity check — reject empty/trivial outputs
                return {
                    "pyspark_code": cleaned,
                    "raw_response": code,
                    "model_used": model,
                    "prompt_used": prompt,
                }

    return _stub_response(prompt, reason="All models failed or returned empty output")


def _stub_response(prompt: str, reason: str = "") -> dict[str, Any]:
    """Return a clearly-marked stub when all model calls fail."""
    stub = (
        f"# NOTE: LLM generation failed — {reason}\n"
        "# TODO: Manual migration required\n"
        "import pyspark.sql.functions as F\n"
        "from pyspark.sql import SparkSession\n\n"
        "spark = SparkSession.builder.appName('migration').getOrCreate()\n"
        "# Add PySpark migration code here\n"
        "df = spark.read.table('source_table')\n"
        "result = df.select('*')\n"
        "result.show()\n"
    )
    return {
        "pyspark_code": stub,
        "raw_response": "",
        "model_used": "stub",
        "prompt_used": prompt,
    }