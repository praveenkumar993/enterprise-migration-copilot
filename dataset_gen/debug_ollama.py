"""
Debug script — calls Ollama once and prints the FULL raw response
so we can see exactly what the model returned and how parsing went.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate import build_prompt, load_seeds, call_ollama, parse_delimited_response, MODEL_NAME

print(f"Model: {MODEL_NAME}")
print("Loading seeds...")
seeds_by_language = load_seeds()

print("\nBuilding prompt for sql/easy...")
prompt = build_prompt("sql", "easy", seeds_by_language)

print("\n" + "=" * 70)
print("PROMPT SENT TO OLLAMA:")
print("=" * 70)
print(prompt)

print("\n" + "=" * 70)
print("CALLING OLLAMA — this may take 30-90 seconds...")
print("=" * 70)
raw_response = call_ollama(prompt)

print("\n" + "=" * 70)
print("RAW RESPONSE FROM OLLAMA (full):")
print("=" * 70)
print(raw_response)
print("=" * 70)

print("\n" + "=" * 70)
print("PARSE RESULT:")
print("=" * 70)
parsed = parse_delimited_response(raw_response)
if parsed:
    print("SUCCESS")
    print(f"features: {parsed['features']}")
    print(f"source_code:\n{parsed['source_code']}")
    print(f"\npyspark_code:\n{parsed['pyspark_code']}")
else:
    print("FAILED — could not extract valid sections")