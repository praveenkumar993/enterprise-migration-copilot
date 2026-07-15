"""
Migrator Agent — LLM-powered PySpark code generation.
Calls HuggingFace Space (Gradio API) with fine-tuned DeepSeek-1.3B model.
Falls back to stub if Space is unavailable.
"""

import os
import time
import json
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME", "praveends")

# HuggingFace Space Gradio API endpoint
SPACE_BASE = f"https://{HF_USERNAME}-migration-copilot-inference.hf.space"
SPACE_URL = f"{SPACE_BASE}/gradio_api/queue/join"

MAX_RETRIES = 2
RETRY_WAIT = 5


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


def call_space_api(prompt: str) -> tuple[str, bool]:
    try:
        # Gradio 6 queue API
        submit_response = requests.post(
            SPACE_URL,
            json={
                "data": [prompt],
                "fn_index": 0,
                "session_hash": "render123"
            },
            timeout=60,
        )

        if submit_response.status_code == 503:
            time.sleep(30)
            return call_space_api(prompt)

        if submit_response.status_code != 200:
            return "", False

        event_id = submit_response.json().get("event_id", "")
        if not event_id:
            return "", False

        # Poll for result
        result_response = requests.get(
            f"{SPACE_BASE}/gradio_api/queue/data?session_hash=render123",
            timeout=120,
            stream=True,
        )

        for line in result_response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    try:
                        data = json.loads(decoded[5:].strip())
                        if data.get("msg") == "process_completed":
                            output = data.get("output", {})
                            result = output.get("data", [])
                            if result:
                                return str(result[0]).strip(), True
                    except Exception:
                        continue

        return "", False

    except Exception:
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
    Generate PySpark code from IR using fine-tuned LLM via HuggingFace Space.

    Calls DeepSeek-1.3B fine-tuned model hosted on HuggingFace Spaces (ZeroGPU).
    Falls back to stub if Space is unavailable or returns empty output.

    Args:
        ir: Unified IR dict from parser
        analyzer_output: Output from Analyzer agent
        rag_context: Retrieved PySpark patterns from RAG

    Returns:
        Dict with pyspark_code, model_used, raw_response
    """
    prompt = build_prompt(ir, analyzer_output, rag_context)

    # Call HuggingFace Space
    code, success = call_space_api(prompt)

    if success and code.strip() and len(code.strip()) > 20:
        cleaned = clean_output(code)
        if len(cleaned) > 20:
            return {
                "pyspark_code": cleaned,
                "raw_response": code,
                "model_used": f"{HF_USERNAME}/migration-copilot-deepseek-coder-1-3b-instruct",
                "prompt_used": prompt,
            }

    return _stub_response(prompt, reason="Space API unavailable or returned empty output")


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