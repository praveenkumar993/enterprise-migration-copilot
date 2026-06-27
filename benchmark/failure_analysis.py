"""
Failure Analysis — Deep dive on best fine-tuned model's failures
================================================================
Reads benchmark/results.json, identifies the best fine-tuned model,
and categorizes all its failures by language and failure type.

Output: benchmark/failure_analysis.md

Usage:
    python benchmark/failure_analysis.py
"""

import json
import re
import ast
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path("benchmark/results.json")
FAILURE_ANALYSIS_PATH = Path("benchmark/failure_analysis.md")
TEST_SPLIT_PATH = Path("dataset_gen/test_split.jsonl")


def load_results() -> list[dict]:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_split() -> dict[str, dict]:
    """Load test split indexed by source_code prefix for joining."""
    pairs = {}
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pair = json.loads(line)
                    key = pair.get("source_code", "")[:80]
                    pairs[key] = pair
                except json.JSONDecodeError:
                    continue
    return pairs


def categorize_failure(result: dict) -> list[str]:
    """Identify which specific checks failed."""
    categories = []
    if not result.get("syntax_valid"):
        categories.append("syntax_error")
    if not result.get("has_pyspark_ops"):
        categories.append("missing_dataframe_ops")
    if not result.get("semantic_sim"):
        categories.append("low_semantic_alignment")
    return categories if categories else ["unknown"]


def find_best_finetuned(results: list[dict]) -> str:
    finetuned = [r for r in results if r.get("is_finetuned")]
    model_scores: dict[str, list] = defaultdict(list)
    for r in finetuned:
        model_scores[r["model"]].append(r["overall"])

    best = max(
        model_scores,
        key=lambda m: sum(model_scores[m]) / len(model_scores[m])
    )
    rate = sum(model_scores[best]) / len(model_scores[best])
    print(f"Best fine-tuned model: {best} ({rate:.1%} overall pass rate)")
    return best


def run_analysis() -> None:
    if not RESULTS_PATH.exists():
        print(f"No results found at {RESULTS_PATH}")
        print("Run benchmark/run_benchmark.py first.")
        return

    results = load_results()
    best_model = find_best_finetuned(results)

    best_results = [r for r in results if r["model"] == best_model]
    failures = [r for r in best_results if not r["overall"]]
    passes = [r for r in best_results if r["overall"]]

    languages = ["sql", "hiveql", "plsql", "stored_procedure"]

    # Per-language stats
    lang_stats: dict[str, dict] = {}
    for lang in languages:
        lang_results = [r for r in best_results if r["source_language"] == lang]
        lang_failures = [r for r in lang_results if not r["overall"]]
        lang_stats[lang] = {
            "total": len(lang_results),
            "passed": len(lang_results) - len(lang_failures),
            "failed": len(lang_failures),
            "pass_rate": (len(lang_results) - len(lang_failures)) / len(lang_results) if lang_results else 0,
            "failures": lang_failures,
        }

    # Failure categorization
    failure_categories: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for failure in failures:
        lang = failure.get("source_language", "unknown")
        for category in categorize_failure(failure):
            failure_categories[lang][category] += 1

    # Build the markdown report
    total = len(best_results)
    total_passed = len(passes)
    total_failed = len(failures)
    overall_rate = total_passed / total if total else 0

    report = f"""# Failure Analysis — {best_model}

## Overview

| Metric | Value |
|--------|-------|
| Model | {best_model} |
| Total test scripts | {total} |
| Passed | {total_passed} ({overall_rate:.1%}) |
| Failed | {total_failed} ({1-overall_rate:.1%}) |

## Per-Language Pass/Fail Rates

| Language | Total | Passed | Failed | Pass Rate |
|----------|-------|--------|--------|-----------|
"""
    for lang in languages:
        s = lang_stats[lang]
        report += f"| {lang:20s} | {s['total']:5d} | {s['passed']:6d} | {s['failed']:6d} | {s['pass_rate']:.1%}      |\n"

    report += "\n## Failure Categories by Language\n\n"
    for lang in languages:
        cats = failure_categories.get(lang, {})
        if not cats:
            report += f"### {lang.upper()} — No failures\n\n"
            continue

        report += f"### {lang.upper()}\n\n"
        report += "| Failure Type | Count | % of failures |\n"
        report += "|-------------|-------|---------------|\n"
        total_lang_failures = sum(cats.values())
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            pct = count / total_lang_failures if total_lang_failures else 0
            report += f"| {cat:30s} | {count:5d} | {pct:.0%}            |\n"
        report += "\n"

    # Main failure mode per language
    report += "## Main Failure Mode Per Language\n\n"
    for lang in languages:
        cats = failure_categories.get(lang, {})
        if cats:
            main_failure = max(cats, key=lambda k: cats[k])
            count = cats[main_failure]
            lang_total_failures = lang_stats[lang]["failed"]

            suggestions = {
                "syntax_error": "Fine-tune with stricter prompt format or add a post-processing syntax-fix step.",
                "missing_dataframe_ops": "Augment training data with more examples containing chained DataFrame ops.",
                "low_semantic_alignment": "Add table-name preservation instruction to the prompt template.",
            }
            suggestion = suggestions.get(main_failure, "Review training data quality for this language.")

            report += f"**{lang.upper()}**: `{main_failure}` — {count}/{lang_total_failures} failures ({count/lang_total_failures:.0%} of {lang} failures)\n"
            report += f"→ {suggestion}\n\n"

    report += "## Conclusions\n\n"
    report += f"The best fine-tuned model ({best_model}) achieves {overall_rate:.1%} overall pass rate "
    report += f"on {total} held-out test scripts not seen during training.\n\n"

    # Identify weakest language
    weakest_lang = min(languages, key=lambda l: lang_stats[l]["pass_rate"])
    strongest_lang = max(languages, key=lambda l: lang_stats[l]["pass_rate"])

    report += f"- **Strongest language**: {strongest_lang.upper()} ({lang_stats[strongest_lang]['pass_rate']:.1%} pass rate)\n"
    report += f"- **Weakest language**: {weakest_lang.upper()} ({lang_stats[weakest_lang]['pass_rate']:.1%} pass rate)\n"
    report += f"- Primary failure mode overall: `{max(failure_categories.get(weakest_lang, {'unknown': 1}), key=lambda k: failure_categories.get(weakest_lang, {}).get(k, 0))}`\n"
    report += "\nNext steps: targeted data augmentation for the weakest language/failure combination.\n"

    FAILURE_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURE_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nFailure analysis written: {FAILURE_ANALYSIS_PATH}")
    print(f"Overall pass rate: {overall_rate:.1%}")
    print(f"Main failure categories:")
    all_cats: dict[str, int] = defaultdict(int)
    for lang_cats in failure_categories.values():
        for cat, count in lang_cats.items():
            all_cats[cat] += count
    for cat, count in sorted(all_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    run_analysis()