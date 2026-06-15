"""
Migrator Agent — LLM-powered PySpark code generation.
Calls HuggingFace Inference API with fine-tuned Phi-3.5-mini model.
Falls back to HF free Inference API if fine-tuned model is not ready.
"""

import os
import time
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveenkumar993")

# Primary: fine-tuned model (available after Day 12)
PRIMARY_MODEL = f"{HF_USERNAME}/phi-enterprise-migration"

# Fallback: public HF model for development
FALLBACK_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

HF_API_BASE = "https://api-inference.huggingface.co/models"

MAX_RETRIES = 3
RETRY_WAIT = 10  # seconds


def build_prompt(
    ir: dict[str, Any],
    analyzer_output: dict[str, Any],
    rag_context: list[dict],
) -> str:
    """
    Build the instruction prompt for the fine-tuned model.

    Args:
        ir: Unified IR dict
        analyzer_output: Output from Analyzer agent
        rag_context: Retrieved PySpark patterns from RAG

    Returns:
        Formatted prompt string
    """
    source_language = ir.get("source_language", "sql")
    raw_source = ir.get("raw_source", "")
    procedural_flags = analyzer_output.get("procedural_flags", [])
    migration_strategy = analyzer_output.get("migration_strategy", "")

    # Format RAG context
    rag_text = ""
    if rag_context:
        rag_snippets = [r.get("text", "") for r in rag_context[:3]]
        rag_text = "\n\n".join(rag_snippets)

    # Procedural warning
    procedural_note = ""
    if procedural_flags:
        flags_str = "\n".join(f"- {f}" for f in procedural_flags[:5])
        procedural_note = (
            f"\nNote: The following procedural constructs cannot be auto-converted "
            f"and require manual review:\n{flags_str}\n"
            f"Add a # NOTE comment in your PySpark code for each flagged construct.\n"
        )

    prompt = f"""### Instruction:
Convert the following {source_language} code to PySpark DataFrame API code.
Use spark as the SparkSession variable name.
Import pyspark.sql.functions as F at the top.
{procedural_note}
Context — PySpark patterns to follow:
{rag_text}

Migration strategy: {migration_strategy}

### Input:
{raw_source}

### Response:
"""
    return prompt


def call_hf_api(model: str, prompt: str) -> tuple[str, bool]:
    """
    Call HuggingFace Inference API.

    Args:
        model: HF model path (username/model-name)
        prompt: Formatted prompt string

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

            # Handle cold start / loading
            if response.status_code == 503:
                body = response.json()
                if "loading" in str(body).lower() or "estimated_time" in body:
                    wait = body.get("estimated_time", RETRY_WAIT)
                    time.sleep(min(wait, 30))
                    continue

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", ""), True
                elif isinstance(data, dict):
                    return data.get("generated_text", ""), True

            # Other error
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)

        except requests.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)

    return "", False


def migrate(
    ir: dict[str, Any],
    analyzer_output: dict[str, Any],
    rag_context: list[dict],
) -> dict[str, Any]:
    """
    Generate PySpark code from IR using fine-tuned LLM.

    Tries primary (fine-tuned) model first.
    Falls back to public HF model if primary fails.

    Args:
        ir: Unified IR dict
        analyzer_output: Output from Analyzer agent
        rag_context: Retrieved PySpark patterns from RAG

    Returns:
        Dict with pyspark_code, model_used, raw_response
    """
    prompt = build_prompt(ir, analyzer_output, rag_context)

    # Try primary model first
    if HF_TOKEN:
        code, success = call_hf_api(PRIMARY_MODEL, prompt)
        if success and code.strip():
            return {
                "pyspark_code": code.strip(),
                "raw_response": code,
                "model_used": PRIMARY_MODEL,
                "prompt_used": prompt,
            }

    # Fallback to public model
    if HF_TOKEN:
        code, success = call_hf_api(FALLBACK_MODEL, prompt)
        if success and code.strip():
            return {
                "pyspark_code": code.strip(),
                "raw_response": code,
                "model_used": FALLBACK_MODEL,
                "prompt_used": prompt,
            }

    # Last resort — return stub with prompt for debugging
    stub = (
        "# NOTE: LLM generation failed or HF_TOKEN not set\n"
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