"""
Create stratified test split from validated pairs.
Saves 480 held-out examples to dataset_gen/test_split.jsonl
stratified by language and difficulty — these were NEVER seen
by any fine-tuned model during training.

Run this ONCE before the benchmark:
    python dataset_gen/create_test_split.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

VALID_PAIRS_PATH = Path("dataset_gen/valid_pairs.jsonl")
SEEDS_PATH = Path("dataset_gen/seeds/claude_seeds.jsonl")
TEST_SPLIT_PATH = Path("dataset_gen/test_split.jsonl")

# Target counts per language per difficulty (from spec)
TARGET_COUNTS = {
    "sql":              {"easy": 30, "medium": 40, "hard": 35, "expert": 15},
    "hiveql":           {"easy": 30, "medium": 40, "hard": 35, "expert": 15},
    "plsql":            {"easy": 30, "medium": 40, "hard": 35, "expert": 15},
    "stored_procedure": {"easy": 40, "medium": 40, "hard": 30, "expert": 10},
}

TOTAL_TARGET = sum(
    count
    for lang_counts in TARGET_COUNTS.values()
    for count in lang_counts.values()
)  # = 480


def load_jsonl(path: Path) -> list[dict]:
    pairs = []
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pairs


def main():
    random.seed(99)  # reproducible split

    print("Loading validated pairs...")
    ollama_pairs = load_jsonl(VALID_PAIRS_PATH)
    claude_pairs = load_jsonl(SEEDS_PATH)

    all_pairs = ollama_pairs + claude_pairs
    print(f"  Total available: {len(all_pairs)}")

    # Group by language + difficulty
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for pair in all_pairs:
        lang = pair.get("source_language", "")
        diff = pair.get("difficulty", "")
        if lang in TARGET_COUNTS and diff in TARGET_COUNTS[lang]:
            buckets[lang][diff].append(pair)

    print("\nAvailable pairs per bucket:")
    for lang in TARGET_COUNTS:
        for diff in TARGET_COUNTS[lang]:
            available = len(buckets[lang][diff])
            needed = TARGET_COUNTS[lang][diff]
            status = "✓" if available >= needed else f"⚠ only {available} available"
            print(f"  {lang:20s} {diff:8s} need={needed:3d}  have={available:4d}  {status}")

    # Sample from each bucket
    test_pairs = []
    for lang in TARGET_COUNTS:
        for diff, count in TARGET_COUNTS[lang].items():
            pool = buckets[lang][diff]
            if len(pool) >= count:
                sampled = random.sample(pool, count)
            else:
                print(f"  WARNING: {lang}/{diff} only has {len(pool)}, using all")
                sampled = pool
            test_pairs.extend(sampled)

    random.shuffle(test_pairs)

    # Write test split
    TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_SPLIT_PATH, "w", encoding="utf-8") as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nTest split written: {len(test_pairs)} pairs → {TEST_SPLIT_PATH}")

    # Stats
    by_lang: dict[str, int] = defaultdict(int)
    by_diff: dict[str, int] = defaultdict(int)
    for p in test_pairs:
        by_lang[p.get("source_language", "?")] += 1
        by_diff[p.get("difficulty", "?")] += 1

    print("\nFinal test split distribution:")
    for lang, count in sorted(by_lang.items()):
        print(f"  {lang:25s} {count}")
    print()
    for diff in ["easy", "medium", "hard", "expert"]:
        print(f"  {diff:25s} {by_diff.get(diff, 0)}")


if __name__ == "__main__":
    main()