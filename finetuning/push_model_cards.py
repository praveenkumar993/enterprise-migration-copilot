"""
Push model cards to all 3 fine-tuned model repos on HuggingFace.
Run from repo root: python finetuning/push_model_cards.py
"""
from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "praveends")

api = HfApi(token=HF_TOKEN)

cards = [
    {
        "repo": f"{HF_USERNAME}/migration-copilot-phi-3-5-mini-instruct",
        "file": "finetuning/model_card_phi.md",
    },
    {
        "repo": f"{HF_USERNAME}/migration-copilot-qwen2-5-coder-1-5b-instruct",
        "file": "finetuning/model_card_qwen.md",
    },
    {
        "repo": f"{HF_USERNAME}/migration-copilot-deepseek-coder-1-3b-instruct",
        "file": "finetuning/model_card_deepseek.md",
    },
]

for card in cards:
    print(f"Pushing model card to {card['repo']}...")
    try:
        api.upload_file(
            path_or_fileobj=card["file"],
            path_in_repo="README.md",
            repo_id=card["repo"],
            repo_type="model",
            commit_message="Add model card with benchmark results and training details",
        )
        print(f"  ✓ Done: https://huggingface.co/{card['repo']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\nAll model cards pushed.")