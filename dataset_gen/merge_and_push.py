"""
Merge and Push — combines Claude seeds + validated Ollama pairs into
final training dataset and pushes to HuggingFace Hub.
Run after bulk generation and validation are complete.

Usage:
    python dataset_gen/merge_and_push.py --push
    python dataset_gen/merge_and_push.py --dry-run
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveends")
DATASET_REPO = f"{HF_USERNAME}/enterprise-migration-dataset"

SEEDS_PATH = Path("dataset_gen/seeds/claude_seeds.jsonl")
OLLAMA_VALID_PATH = Path("dataset_gen/valid_pairs.jsonl")
FINAL_OUTPUT_PATH = Path("dataset_gen/final_dataset.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    """Load all valid JSON lines from a JSONL file."""
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return []
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping line {i} in {path}: {e}")
    return pairs


def merge_datasets(
    seeds: list[dict],
    ollama_pairs: list[dict],
) -> list[dict]:
    """
    Merge Claude seeds and validated Ollama pairs.
    Deduplicates by source_code to avoid near-duplicates.
    """
    seen_sources = set()
    merged = []

    # Claude seeds first — highest quality, always included
    for pair in seeds:
        key = pair.get("source_code", "").strip()[:100]
        if key and key not in seen_sources:
            seen_sources.add(key)
            merged.append(pair)

    # Ollama pairs second — deduplicated against seeds and each other
    for pair in ollama_pairs:
        key = pair.get("source_code", "").strip()[:100]
        if key and key not in seen_sources:
            seen_sources.add(key)
            merged.append(pair)

    return merged


def print_stats(dataset: list[dict], label: str) -> None:
    """Print dataset statistics."""
    by_lang: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    by_source: dict[str, int] = {}

    for pair in dataset:
        lang = pair.get("source_language", "unknown")
        diff = pair.get("difficulty", "unknown")
        gen_by = pair.get("generated_by", "unknown")
        by_lang[lang] = by_lang.get(lang, 0) + 1
        by_diff[diff] = by_diff.get(diff, 0) + 1
        by_source[gen_by] = by_source.get(gen_by, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Total pairs: {len(dataset)}")
    print(f"\nBy source language:")
    for lang, count in sorted(by_lang.items()):
        print(f"  {lang:25s} {count:5d}")
    print(f"\nBy difficulty:")
    for diff in ["easy", "medium", "hard", "expert"]:
        count = by_diff.get(diff, 0)
        print(f"  {diff:25s} {count:5d}")
    print(f"\nBy generation source:")
    for gen, count in sorted(by_source.items()):
        print(f"  {gen:25s} {count:5d}")
    print(f"{'=' * 60}\n")


def push_to_huggingface(dataset_path: Path) -> bool:
    """Push the final dataset to HuggingFace Hub."""
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env — cannot push to HuggingFace")
        return False

    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=HF_TOKEN)

        # Create dataset repo if it doesn't exist
        try:
            create_repo(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                token=HF_TOKEN,
                exist_ok=True,
                private=False,
            )
            print(f"Dataset repo ready: https://huggingface.co/datasets/{DATASET_REPO}")
        except Exception as e:
            print(f"Repo creation note: {e}")

        # Upload the final dataset file
        api.upload_file(
            path_or_fileobj=str(dataset_path),
            path_in_repo="train.jsonl",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Dataset update — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )

        # Upload a README
        readme_content = f"""# Enterprise Migration Dataset

Training dataset for SQL/HiveQL/PL/SQL/Stored Procedure → PySpark migration.

## Stats
- Generated: {datetime.now().strftime('%Y-%m-%d')}
- Total pairs: see train.jsonl

## Sources
- claude-seed: hand-crafted high-quality examples (200 pairs, 99.5% validation pass rate)
- ollama-bulk: generated using qwen2.5-coder:1.5b with few-shot prompting

## Schema
Each line is a JSON object with fields:
- source_language: sql | hiveql | plsql | stored_procedure
- difficulty: easy | medium | hard | expert
- source_code: the raw SQL/HiveQL/PL/SQL/T-SQL input
- pyspark_code: the target PySpark output
- features: list of concepts demonstrated
- generated_by: claude-seed | ollama-bulk
"""
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message="Update README",
        )

        print(f"\nDataset pushed to: https://huggingface.co/datasets/{DATASET_REPO}")
        return True

    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        print(f"ERROR pushing to HuggingFace: {e}")
        return False


def main(dry_run: bool = False, seeds_only: bool = False) -> None:
    print("Loading Claude seed examples...")
    seeds = load_jsonl(SEEDS_PATH)
    print(f"  Loaded {len(seeds)} Claude seed pairs")

    if seeds_only:
        ollama_pairs = []
        print("  Skipping Ollama pairs (--seeds-only mode)")
    else:
        print(f"\nLoading validated Ollama pairs from {OLLAMA_VALID_PATH}...")
        ollama_pairs = load_jsonl(OLLAMA_VALID_PATH)
        print(f"  Loaded {len(ollama_pairs)} validated Ollama pairs")

    print("\nMerging datasets...")
    merged = merge_datasets(seeds, ollama_pairs)
    print(f"  Merged total: {len(merged)} pairs (after deduplication)")

    print_stats(merged, "FINAL DATASET STATISTICS")

    if dry_run:
        print("DRY RUN — not writing or pushing anything.")
        return

    # Write final dataset
    FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in merged:
            f.write(json.dumps(pair) + "\n")
    print(f"Final dataset written to: {FINAL_OUTPUT_PATH}")

    if not dry_run:
        print("\nPushing to HuggingFace Hub...")
        success = push_to_huggingface(FINAL_OUTPUT_PATH)
        if success:
            print("Push complete.")
        else:
            print("Push failed — check your HF_TOKEN in .env")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and push final training dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, no file writes")
    parser.add_argument("--seeds-only", action="store_true", help="Merge seeds only, ignore Ollama pairs")
    args = parser.parse_args()

    if args.dry_run:
        main(dry_run=True)
    elif args.push:
        main(dry_run=False)
    else:
        print("Specify --push to write + push, or --dry-run to preview stats only.")
        print("Example: python dataset_gen/merge_and_push.py --dry-run")