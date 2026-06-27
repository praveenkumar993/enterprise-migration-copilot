"""
Frontier Benchmark — Claude + Grok comparison on 50-script subset
=================================================================
For Claude: generates prompts to frontier/prompts.txt for manual evaluation
For Grok:   calls api.x.ai directly (requires XAI_TOKEN in .env)

Usage:
    python benchmark/frontier_benchmark.py --generate-prompts   # Step 1
    python benchmark/frontier_benchmark.py --score-claude       # After manual eval
    python benchmark/frontier_benchmark.py --score-grok         # Grok API
"""

import json
import re
import ast
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import os

import requests

load_dotenv()

XAI_TOKEN = os.getenv("XAI_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TEST_SPLIT_PATH = Path("dataset_gen/test_split.jsonl")
FRONTIER_DIR = Path("benchmark/frontier")
PROMPTS_PATH = FRONTIER_DIR / "prompts.txt"
CLAUDE_OUTPUTS_PATH = FRONTIER_DIR / "claude_outputs.jsonl"
GROK_OUTPUTS_PATH = FRONTIER_DIR / "grok_outputs.jsonl"
FRONTIER_SUMMARY_PATH = Path("benchmark/frontier_summary.md")

# 50-script subset: 15 SQL, 15 HiveQL, 10 PL/SQL, 10 SP
FRONTIER_COUNTS = {
    "sql": 15,
    "hiveql": 15,
    "plsql": 10,
    "stored_procedure": 10,
}

DATAFRAME_OPS = [
    r"\.select\(", r"\.filter\(", r"\.where\(", r"\.groupBy\(",
    r"\.join\(", r"\.agg\(", r"\.withColumn\(", r"\.orderBy\(",
    r"spark\.table\(", r"spark\.read", r"\.show\(", r"\.count\(",
    r"\.write\.", r"\.repartition\(", r"DeltaTable\.",
]


def load_frontier_subset() -> list[dict]:
    """Load 50-script stratified subset from test split."""
    if not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError(f"Run create_test_split.py first: {TEST_SPLIT_PATH}")

    all_pairs = []
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    from collections import defaultdict
    import random
    random.seed(42)

    by_lang: dict[str, list] = defaultdict(list)
    for p in all_pairs:
        by_lang[p.get("source_language", "")].append(p)

    subset = []
    for lang, count in FRONTIER_COUNTS.items():
        pool = by_lang.get(lang, [])
        sampled = random.sample(pool, min(count, len(pool)))
        subset.extend(sampled)

    return subset


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


def score_output(generated: str, example: dict) -> dict:
    syntax_valid = False
    try:
        ast.parse(generated)
        syntax_valid = True
    except SyntaxError:
        pass

    op_count = sum(1 for pattern in DATAFRAME_OPS if re.search(pattern, generated))
    has_pyspark_ops = op_count >= 2

    source_code = example.get("source_code", "").lower()
    table_pattern = r"(?:from|join)\s+([a-z_][a-z0-9_]*)"
    source_tables = set(re.findall(table_pattern, source_code))
    if source_tables:
        generated_lower = generated.lower()
        found = sum(1 for t in source_tables if t in generated_lower)
        semantic_sim = (found / len(source_tables)) >= 0.60
    else:
        semantic_sim = True

    return {
        "syntax_valid": syntax_valid,
        "has_pyspark_ops": has_pyspark_ops,
        "semantic_sim": semantic_sim,
        "overall": syntax_valid and has_pyspark_ops and semantic_sim,
    }


def generate_prompts() -> None:
    """Generate the 50 prompts for manual Claude evaluation."""
    subset = load_frontier_subset()
    FRONTIER_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_PATH, "w", encoding="utf-8") as f:
        for i, example in enumerate(subset, 1):
            lang = example.get("source_language", "")
            diff = example.get("difficulty", "")
            f.write(f"{'='*70}\n")
            f.write(f"PROMPT {i:02d} | language={lang} | difficulty={diff}\n")
            f.write(f"{'='*70}\n")
            f.write(build_prompt(example))
            f.write("\n\n")

    print(f"Generated {len(subset)} prompts → {PROMPTS_PATH}")
    print()
    print("Next steps for Claude evaluation:")
    print("1. Open frontier/prompts.txt")
    print("2. For each prompt, paste into claude.ai and copy the response")
    print("3. Save each response as one JSON line in frontier/claude_outputs.jsonl:")
    print('   {"prompt_id": 1, "language": "sql", "generated": "...pyspark code..."}')
    print("4. Run: python benchmark/frontier_benchmark.py --score-claude")


def score_claude() -> None:
    """Score manually collected Claude outputs."""
    if not CLAUDE_OUTPUTS_PATH.exists():
        print(f"No Claude outputs found at {CLAUDE_OUTPUTS_PATH}")
        print("Follow instructions from --generate-prompts first.")
        return

    subset = load_frontier_subset()
    outputs = []
    with open(CLAUDE_OUTPUTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    outputs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Scoring {len(outputs)} Claude outputs...")
    results = []
    for i, (example, output) in enumerate(zip(subset, outputs)):
        generated = output.get("generated", "")
        scores = score_output(generated, example)
        scores.update({
            "model": "Claude (manual)",
            "source_language": example.get("source_language", ""),
            "difficulty": example.get("difficulty", ""),
            "prompt_id": i + 1,
        })
        results.append(scores)

    _print_frontier_results("Claude (manual)", results)


def score_grok() -> None:
    """Score using Grok API directly."""
    if not XAI_TOKEN:
        print("XAI_TOKEN not set in .env — cannot call Grok API")
        return

    subset = load_frontier_subset()
    print(f"Calling Grok API for {len(subset)} scripts...")

    results = []
    outputs = []

    for i, example in enumerate(subset):
        prompt = build_prompt(example)

        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "grok-beta",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.1,
                },
                timeout=30,
            )

            if response.status_code == 200:
                generated = response.json()["choices"][0]["message"]["content"]
            else:
                print(f"  [{i+1}] Grok API error: {response.status_code}")
                generated = ""

        except Exception as e:
            print(f"  [{i+1}] Grok call failed: {e}")
            generated = ""

        scores = score_output(generated, example)
        scores.update({
            "model": "Grok",
            "source_language": example.get("source_language", ""),
            "difficulty": example.get("difficulty", ""),
            "prompt_id": i + 1,
        })
        results.append(scores)
        outputs.append({
            "prompt_id": i + 1,
            "language": example.get("source_language", ""),
            "generated": generated,
        })

        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(subset)}] done")

    FRONTIER_DIR.mkdir(parents=True, exist_ok=True)
    with open(GROK_OUTPUTS_PATH, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

    _print_frontier_results("Grok", results)


def _print_frontier_results(model_name: str, results: list[dict]) -> None:
    languages = ["sql", "hiveql", "plsql", "stored_procedure"]

    print(f"\n{'='*60}")
    print(f"FRONTIER RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"{'Language':20s} {'Syntax':8s} {'DF Ops':8s} {'Semantic':10s} {'Overall':8s}")
    print("-" * 60)

    for lang in languages:
        subset = [r for r in results if r["source_language"] == lang]
        if not subset:
            continue
        syn = sum(1 for r in subset if r["syntax_valid"]) / len(subset)
        ops = sum(1 for r in subset if r["has_pyspark_ops"]) / len(subset)
        sem = sum(1 for r in subset if r["semantic_sim"]) / len(subset)
        ov = sum(1 for r in subset if r["overall"]) / len(subset)
        print(f"{lang:20s} {syn:.0%}      {ops:.0%}      {sem:.0%}        {ov:.0%}")

    overall = sum(1 for r in results if r["overall"]) / len(results)
    print("-" * 60)
    print(f"{'OVERALL':20s} {overall:.0%}")

    # Append to frontier summary
    FRONTIER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRONTIER_SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## {model_name} (50-script frontier subset)\n\n")
        f.write(f"| Language | Syntax | DF Ops | Semantic | Overall |\n")
        f.write(f"|----------|--------|--------|----------|---------|\n")
        for lang in languages:
            subset = [r for r in results if r["source_language"] == lang]
            if not subset:
                continue
            ov = sum(1 for r in subset if r["overall"]) / len(subset)
            syn = sum(1 for r in subset if r["syntax_valid"]) / len(subset)
            ops = sum(1 for r in subset if r["has_pyspark_ops"]) / len(subset)
            sem = sum(1 for r in subset if r["semantic_sim"]) / len(subset)
            f.write(f"| {lang:20s} | {syn:.0%}    | {ops:.0%}    | {sem:.0%}      | {ov:.0%}     |\n")
        f.write(f"| **Overall**          | —      | —      | —        | **{overall:.0%}**   |\n\n")
        f.write(f"*Note: Frontier models evaluated on 50-script subset only.*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontier model benchmark")
    parser.add_argument("--generate-prompts", action="store_true")
    parser.add_argument("--score-claude", action="store_true")
    parser.add_argument("--score-grok", action="store_true")
    args = parser.parse_args()

    if args.generate_prompts:
        generate_prompts()
    elif args.score_claude:
        score_claude()
    elif args.score_grok:
        score_grok()
    else:
        print("Specify one of: --generate-prompts | --score-claude | --score-grok")