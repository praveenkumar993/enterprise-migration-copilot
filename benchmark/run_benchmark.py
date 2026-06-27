"""
Benchmark: Base vs Fine-tuned Models
=====================================
Evaluates 6 models (3 base + 3 fine-tuned) on 480 held-out test scripts.
Scores each output on 3 metrics:
  - syntax_valid:    ast.parse check on generated PySpark
  - has_pyspark_ops: at least 2 DataFrame operations present
  - semantic_sim:    at least 60% of source table names appear in output

Output:
  benchmark/results.json     — all 2,880 individual scores
  benchmark/summary.md       — leaderboard table + best model declaration

Usage:
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --dry-run   # 10 scripts only, fast test
"""

import json
import ast
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import requests
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveends")

TEST_SPLIT_PATH = Path("dataset_gen/test_split.jsonl")
RESULTS_PATH = Path("benchmark/results.json")
SUMMARY_PATH = Path("benchmark/summary.md")

MODELS = {
    "DeepSeek-1.3B-base": {
        "repo": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "is_finetuned": False,
        "family": "deepseek",
    },
    "DeepSeek-1.3B-finetuned": {
        "repo": f"{HF_USERNAME}/migration-copilot-deepseek-coder-1-3b-instruct",
        "is_finetuned": True,
        "family": "deepseek",
        "base_model": "deepseek-ai/deepseek-coder-1.3b-instruct",
    },
    "Qwen2.5-1.5B-base": {
        "repo": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "is_finetuned": False,
        "family": "qwen",
    },
    "Qwen2.5-1.5B-finetuned": {
        "repo": f"{HF_USERNAME}/migration-copilot-qwen2-5-coder-1-5b-instruct",
        "is_finetuned": True,
        "family": "qwen",
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    },
    "Phi-3.5-mini-base": {
        "repo": "microsoft/Phi-3.5-mini-instruct",
        "is_finetuned": False,
        "family": "phi",
    },
    "Phi-3.5-mini-finetuned": {
        "repo": f"{HF_USERNAME}/migration-copilot-phi-3-5-mini-instruct",
        "is_finetuned": True,
        "family": "phi",
        "base_model": "microsoft/Phi-3.5-mini-instruct",
    },
}

DATAFRAME_OPS = [
    r"\.select\(", r"\.filter\(", r"\.where\(", r"\.groupBy\(",
    r"\.join\(", r"\.agg\(", r"\.withColumn\(", r"\.orderBy\(",
    r"\.union\(", r"\.distinct\(", r"\.drop\(", r"\.limit\(",
    r"spark\.table\(", r"spark\.read", r"spark\.sql\(",
    r"\.show\(", r"\.count\(", r"\.collect\(", r"\.write\.",
    r"\.repartition\(", r"\.cache\(", r"DeltaTable\.",
]


def build_prompt(example: dict) -> str:
    source_lang = example.get("source_language", "sql").upper()
    difficulty = example.get("difficulty", "medium")
    source_code = example.get("source_code", "")
    return f"""### Instruction:
Convert the following {source_lang} code to PySpark.
Difficulty: {difficulty}

### Input:
{source_code}

### Response:
"""


def call_hf_inference(repo: str, prompt: str, max_retries: int = 3) -> str:
    """Call HuggingFace Inference API with retry on cold start."""
    url = f"https://api-inference.huggingface.co/models/{repo}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.1,
            "do_sample": True,
            "return_full_text": False,
        },
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code == 503:
                # Model loading — wait and retry
                wait = 15 * (attempt + 1)
                print(f"    Model loading, waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                return ""

            data = response.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            return ""

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"    API call failed: {e}")
                return ""

    return ""


def score_output(generated: str, example: dict) -> dict:
    """Score a generated PySpark output on 3 metrics."""

    # Metric 1: Syntax valid
    syntax_valid = False
    try:
        ast.parse(generated)
        syntax_valid = True
    except SyntaxError:
        pass

    # Metric 2: Has at least 2 DataFrame operations
    op_count = sum(1 for pattern in DATAFRAME_OPS if re.search(pattern, generated))
    has_pyspark_ops = op_count >= 2

    # Metric 3: Semantic similarity — table names from source appear in output
    source_code = example.get("source_code", "").lower()
    # Extract identifiers that look like table names (FROM/JOIN targets)
    table_pattern = r"(?:from|join)\s+([a-z_][a-z0-9_]*)"
    source_tables = set(re.findall(table_pattern, source_code))
    if source_tables:
        generated_lower = generated.lower()
        found = sum(1 for t in source_tables if t in generated_lower)
        semantic_sim = (found / len(source_tables)) >= 0.60
    else:
        semantic_sim = True  # no tables to check — don't penalize

    overall = syntax_valid and has_pyspark_ops and semantic_sim

    return {
        "syntax_valid": syntax_valid,
        "has_pyspark_ops": has_pyspark_ops,
        "semantic_sim": semantic_sim,
        "overall": overall,
        "op_count": op_count,
    }


def load_test_split(dry_run: bool = False) -> list[dict]:
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found: {TEST_SPLIT_PATH}\n"
            "Run: python dataset_gen/create_test_split.py"
        )
    pairs = []
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if dry_run:
        # Take 2 from each language for fast testing
        by_lang: dict[str, list] = defaultdict(list)
        for p in pairs:
            by_lang[p.get("source_language", "")].append(p)
        pairs = []
        for lang_pairs in by_lang.values():
            pairs.extend(lang_pairs[:3])
        print(f"DRY RUN: using {len(pairs)} scripts")

    return pairs


def run_benchmark(dry_run: bool = False) -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in .env")

    test_pairs = load_test_split(dry_run)
    print(f"Test scripts loaded: {len(test_pairs)}")
    print(f"Models to evaluate: {len(MODELS)}")
    print(f"Total API calls: {len(test_pairs) * len(MODELS)}")
    print()

    results = []
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for model_name, model_config in MODELS.items():
        repo = model_config["repo"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Repo: {repo}")
        print(f"{'='*60}")

        model_scores = []
        for i, example in enumerate(test_pairs):
            lang = example.get("source_language", "")
            diff = example.get("difficulty", "")

            prompt = build_prompt(example)
            generated = call_hf_inference(repo, prompt)

            scores = score_output(generated, example)
            scores.update({
                "model": model_name,
                "repo": repo,
                "is_finetuned": model_config["is_finetuned"],
                "family": model_config["family"],
                "source_language": lang,
                "difficulty": diff,
                "generated_length": len(generated),
            })
            model_scores.append(scores)
            results.append(scores)

            # Progress update every 20 scripts
            if (i + 1) % 20 == 0:
                so_far = model_scores[-20:]
                overall_rate = sum(1 for s in so_far if s["overall"]) / len(so_far)
                print(f"  [{i+1}/{len(test_pairs)}] last-20 pass rate: {overall_rate:.0%}")

            time.sleep(0.5)  # gentle rate limiting

        # Per-model summary
        passed = sum(1 for s in model_scores if s["overall"])
        print(f"\n  {model_name}: {passed}/{len(model_scores)} passed ({passed/len(model_scores):.1%})")

    # Save all results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")

    write_summary(results, test_pairs)


def write_summary(results: list[dict], test_pairs: list[dict]) -> None:
    """Write leaderboard table to benchmark/summary.md"""
    languages = ["sql", "hiveql", "plsql", "stored_procedure"]
    model_names = list(MODELS.keys())

    # Aggregate scores per model per language
    def get_rate(model: str, lang: str | None = None) -> float:
        subset = [r for r in results if r["model"] == model]
        if lang:
            subset = [r for r in subset if r["source_language"] == lang]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r["overall"]) / len(subset)

    # Find best fine-tuned model
    finetuned_models = [m for m in model_names if MODELS[m]["is_finetuned"]]
    best_model = max(finetuned_models, key=lambda m: get_rate(m))
    best_rate = get_rate(best_model)

    # Build table
    lang_headers = " | ".join(f"{l.upper():7s}" for l in languages)
    header = f"| {'Model':35s} | {lang_headers} | {'Overall':7s} |"
    separator = "|" + "-" * 37 + "|" + "---------|" * (len(languages) + 1)

    rows = [header, separator]
    for model_name in model_names:
        lang_rates = " | ".join(f"{get_rate(model_name, l):.0%}   " for l in languages)
        overall = get_rate(model_name)
        marker = " ⭐" if model_name == best_model else ""
        row = f"| {model_name + marker:35s} | {lang_rates} | {overall:.0%}     |"
        rows.append(row)

    # Improvement table
    improvement_rows = []
    for family in ["deepseek", "qwen", "phi"]:
        base = next(m for m in model_names if MODELS[m]["family"] == family and not MODELS[m]["is_finetuned"])
        ft = next(m for m in model_names if MODELS[m]["family"] == family and MODELS[m]["is_finetuned"])
        base_rate = get_rate(base)
        ft_rate = get_rate(ft)
        delta = ft_rate - base_rate
        sign = "+" if delta >= 0 else ""
        improvement_rows.append(
            f"| {family.capitalize():15s} | {base_rate:.1%}      | {ft_rate:.1%}           | {sign}{delta:.1%}       |"
        )

    summary = f"""# Benchmark Results — Enterprise Migration Copilot

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Test set: {len(test_pairs)} scripts (stratified by language and difficulty)
Metrics: syntax_valid AND has_pyspark_ops AND semantic_sim (60% table coverage)

## Leaderboard

{chr(10).join(rows)}

## Fine-tuning Improvement

| Model Family   | Base Overall | Fine-tuned Overall | Delta       |
|----------------|-------------|-------------------|-------------|
{chr(10).join(improvement_rows)}

## Best Model

**{best_model}** achieved the highest overall pass rate: **{best_rate:.1%}**

## Metric Definitions

- **syntax_valid**: Generated PySpark passes `ast.parse()` — no syntax errors
- **has_pyspark_ops**: At least 2 DataFrame operations (`.select`, `.filter`, `.groupBy`, etc.)
- **semantic_sim**: At least 60% of source table names appear in generated output
- **Overall**: All 3 metrics must pass

## Per-Language Breakdown

"""
    for lang in languages:
        summary += f"### {lang.upper()}\n\n"
        summary += f"| {'Model':35s} | Syntax | DF Ops | Semantic | Overall |\n"
        summary += f"|{'':->37s}|{'':->8s}|{'':->8s}|{'':->10s}|{'':->9s}|\n"
        for model_name in model_names:
            subset = [r for r in results if r["model"] == model_name and r["source_language"] == lang]
            if not subset:
                continue
            syn = sum(1 for r in subset if r["syntax_valid"]) / len(subset)
            ops = sum(1 for r in subset if r["has_pyspark_ops"]) / len(subset)
            sem = sum(1 for r in subset if r["semantic_sim"]) / len(subset)
            ov = sum(1 for r in subset if r["overall"]) / len(subset)
            summary += f"| {model_name:35s} | {syn:.0%}    | {ops:.0%}    | {sem:.0%}      | {ov:.0%}     |\n"
        summary += "\n"

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary written: {SUMMARY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Use 12 scripts only for fast testing")
    args = parser.parse_args()
    run_benchmark(dry_run=args.dry_run)