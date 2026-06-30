import json

from datasets import load_dataset

# Load the HLE (Humanity's Last Exam) dataset
ds = load_dataset("skylenage-ai/HLE-Verified")

# Filter questions whose verified_Classes is "Gold subset"
gold_subset = []
for example in ds["train"]:
    if example.get("Verified_Classes") == "Gold subset":
        gold_subset.append(
            {
                "id": example.get("id"),
                "question": example.get("question"),
                "answer": example.get("answer"),
                "answer_type": example.get("answer_type"),
                "rationale": example.get("rationale"),
                "Verified_Classes": example.get("Verified_Classes"),
            }
        )

# Save to JSON
output_path = "hle_gold_subset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(gold_subset, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(gold_subset)} Gold subset questions to {output_path}")
