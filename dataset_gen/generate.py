"""
Bulk Dataset Generation — Ollama local generation with few-shot context.
Uses delimiter-based output format (not JSON) to avoid small-model
JSON-escaping failures with multi-line code strings.
Target: 8,000-9,000 raw pairs across all 4 source languages.
"""

import json
import random
import re
import time
import argparse
from pathlib import Path
from typing import Any

import ollama

from templates.sql_prompts import SQL_GENERATION_PROMPT, SQL_TOPICS_BY_DIFFICULTY
from templates.hiveql_prompts import HIVEQL_GENERATION_PROMPT, HIVEQL_TOPICS_BY_DIFFICULTY
from templates.plsql_prompts import PLSQL_GENERATION_PROMPT, PLSQL_TOPICS_BY_DIFFICULTY
from templates.sp_prompts import SP_GENERATION_PROMPT, SP_TOPICS_BY_DIFFICULTY


MODEL_NAME = "qwen2.5-coder:1.5b"
SEEDS_PATH = Path(__file__).parent / "seeds" / "claude_seeds.jsonl"
OUTPUT_PATH = Path(__file__).parent / "raw_pairs.jsonl"

MAX_ATTEMPTS_PER_BATCH = 3
RETRY_WAIT_SECONDS = 1

LANGUAGE_CONFIG = {
    "sql": {
        "prompt_template": SQL_GENERATION_PROMPT,
        "topics": SQL_TOPICS_BY_DIFFICULTY,
    },
    "hiveql": {
        "prompt_template": HIVEQL_GENERATION_PROMPT,
        "topics": HIVEQL_TOPICS_BY_DIFFICULTY,
    },
    "plsql": {
        "prompt_template": PLSQL_GENERATION_PROMPT,
        "topics": PLSQL_TOPICS_BY_DIFFICULTY,
    },
    "stored_procedure": {
        "prompt_template": SP_GENERATION_PROMPT,
        "topics": SP_TOPICS_BY_DIFFICULTY,
    },
}

DIFFICULTIES = ["easy", "medium", "hard", "expert"]


def load_seeds() -> dict[str, list[dict]]:
    """Load Claude seed examples grouped by source_language."""
    seeds_by_language: dict[str, list[dict]] = {
        "sql": [], "hiveql": [], "plsql": [], "stored_procedure": []
    }

    if not SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds file not found: {SEEDS_PATH}")

    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
                lang = pair.get("source_language", "")
                if lang in seeds_by_language:
                    seeds_by_language[lang].append(pair)
            except json.JSONDecodeError:
                continue

    return seeds_by_language


def format_single_example(example: dict) -> str:
    """Format ONE seed example using the same delimiter style we want back."""
    features_str = ", ".join(example.get("features", []))
    return (
        f"===FEATURES===\n{features_str}\n\n"
        f"===SOURCE_CODE===\n{example.get('source_code', '')}\n\n"
        f"===PYSPARK_CODE===\n{example.get('pyspark_code', '')}\n\n"
        f"===END==="
    )


def build_prompt(language: str, difficulty: str, seeds_by_language: dict) -> str:
    """
    Build a complete generation prompt with ONE few-shot example.
    Matches the few-shot example difficulty to the requested difficulty
    so the model sees an appropriately-scoped example to imitate.
    """
    config = LANGUAGE_CONFIG[language]
    available_seeds = seeds_by_language.get(language, [])

    # Prefer same difficulty, fall back to any available seed
    same_difficulty = [s for s in available_seeds if s.get("difficulty") == difficulty]
    pool = same_difficulty if same_difficulty else available_seeds

    if pool:
        example = random.choice(pool)
    else:
        example = {"features": [], "source_code": "", "pyspark_code": ""}

    few_shot_text = format_single_example(example)

    prompt = config["prompt_template"].format(
        difficulty=difficulty,
        few_shot_examples=few_shot_text,
    )
    return prompt

def parse_delimited_response(raw_text: str) -> dict | None:
    """
    Parse delimiter-based OR markdown-heading-based response.
    Handles both our markers (===FEATURES===) and qwen's preferred
    markdown variant (#### FEATURES:) automatically.
    """
    text = raw_text.strip()

    # Normalize qwen markdown headings to our markers
    text = re.sub(r"#{1,4}\s*FEATURES\s*:?", "===FEATURES===", text, flags=re.IGNORECASE)
    text = re.sub(r"#{1,4}\s*SOURCE\s*CODE\s*:?", "===SOURCE_CODE===", text, flags=re.IGNORECASE)
    text = re.sub(r"#{1,4}\s*PYSPARK[_\s]*CODE\s*:?", "===PYSPARK_CODE===", text, flags=re.IGNORECASE)
    text = re.sub(r"#{1,4}\s*END\s*", "===END===", text, flags=re.IGNORECASE)

    if "===SOURCE_CODE===" not in text or "===PYSPARK_CODE===" not in text:
        return None

    all_markers = ["===FEATURES===", "===SOURCE_CODE===", "===PYSPARK_CODE===", "===END==="]

    def extract_section(t, start_marker, end_markers):
        start_idx = t.find(start_marker)
        if start_idx == -1:
            return ""
        start_idx += len(start_marker)
        end_idx = len(t)
        for marker in end_markers:
            idx = t.find(marker, start_idx)
            if idx != -1:
                end_idx = min(end_idx, idx)
        return t[start_idx:end_idx].strip()

    features_raw = extract_section(text, "===FEATURES===", all_markers)
    source_code = extract_section(text, "===SOURCE_CODE===", all_markers)
    pyspark_code = extract_section(text, "===PYSPARK_CODE===", all_markers)

    # Strip markdown code fences if model added them anyway
    source_code = re.sub(r"^```\w*\n?", "", source_code)
    source_code = re.sub(r"\n?```$", "", source_code).strip()
    pyspark_code = re.sub(r"^```\w*\n?", "", pyspark_code)
    pyspark_code = re.sub(r"\n?```$", "", pyspark_code).strip()

    if len(source_code) < 10 or len(pyspark_code) < 10:
        return None

    # Handle both "- bullet item" and "item, item" feature formats
    features_raw = re.sub(r"^\s*[-*]\s*`?", "", features_raw, flags=re.MULTILINE)
    features_raw = re.sub(r"`", "", features_raw)
    features = [f.strip() for f in re.split(r"[,\n]", features_raw) if f.strip()]

    return {
        "source_code": source_code,
        "pyspark_code": pyspark_code,
        "features": features,
    }

def call_ollama(prompt: str) -> str:
    """Call Ollama once. Returns raw response text, or empty string on failure."""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.4,
                "num_predict": 1200,
                "top_p": 0.9,
            },
        )
        return response.get("response", "")
    except Exception as e:
        print(f"  Ollama call failed: {e}")
        return ""


def generate_one_pair(
    language: str,
    difficulty: str,
    seeds_by_language: dict,
    max_attempts: int = MAX_ATTEMPTS_PER_BATCH,
) -> dict | None:
    """
    Generate exactly one validated-format pair for a language/difficulty combo.
    Retries with fresh few-shot sample if parsing fails.
    """
    for attempt in range(max_attempts):
        prompt = build_prompt(language, difficulty, seeds_by_language)
        raw_response = call_ollama(prompt)

        if not raw_response:
            time.sleep(RETRY_WAIT_SECONDS)
            continue

        parsed = parse_delimited_response(raw_response)

        if parsed:
            parsed["source_language"] = language
            parsed["difficulty"] = difficulty
            parsed["generated_by"] = "ollama-bulk"
            return parsed

        if attempt < max_attempts - 1:
            time.sleep(RETRY_WAIT_SECONDS)

    return None


def run_generation(
    total_target: int,
    languages: list[str] = None,
    difficulties: list[str] = None,
) -> None:
    """Run bulk generation until total_target raw pairs are written."""
    languages = languages or list(LANGUAGE_CONFIG.keys())
    difficulties = difficulties or DIFFICULTIES

    print(f"Loading seeds from {SEEDS_PATH}...")
    seeds_by_language = load_seeds()
    for lang, seeds in seeds_by_language.items():
        print(f"  {lang}: {len(seeds)} seed examples loaded")

    combos = [(lang, diff) for lang in languages for diff in difficulties]
    random.shuffle(combos)

    total_generated = 0
    total_attempted = 0
    combo_index = 0

    print(f"\nStarting generation. Target: {total_target} pairs")
    print(f"Model: {MODEL_NAME}\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
        while total_generated < total_target:
            language, difficulty = combos[combo_index % len(combos)]
            combo_index += 1
            total_attempted += 1

            print(f"[{total_attempted}] {language}/{difficulty} "
                  f"(generated so far: {total_generated}/{total_target})...", end=" ")

            pair = generate_one_pair(language, difficulty, seeds_by_language)

            if pair:
                out_file.write(json.dumps(pair) + "\n")
                out_file.flush()
                total_generated += 1
                print("OK")
            else:
                print("FAILED (all retries exhausted)")

    yield_rate = round((total_generated / total_attempted) * 100, 1) if total_attempted else 0
    print(f"\nDone. Total pairs written: {total_generated}")
    print(f"Total attempts: {total_attempted}")
    print(f"Yield rate: {yield_rate}%")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk generate migration pairs via Ollama")
    parser.add_argument("--target", type=int, default=20, help="Total raw pairs to generate")
    parser.add_argument("--languages", nargs="+", default=None, help="Subset of languages to generate")
    parser.add_argument("--difficulties", nargs="+", default=None, help="Subset of difficulties to generate")
    args = parser.parse_args()

    run_generation(
        total_target=args.target,
        languages=args.languages,
        difficulties=args.difficulties,
    )