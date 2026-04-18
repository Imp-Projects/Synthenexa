import json
from random import sample


def clean_dataset(input_file, output_file):
    seen_prompts = set()
    cleaned_samples = []

    with open(input_file, "r") as f:
        for line in f:
            try:
                sample = json.loads(line)

                # Skip broken entries
                required_keys = [
                    "prompt",
                    "entry_point",
                    "canonical_solution",
                    "test"
                ]

                if not all(key in sample and sample[key] for key in required_keys):
                    continue

                # Remove exact duplicate prompts
                fingerprint = (
                    sample["entry_point"].strip() +
                    sample["canonical_solution"].strip()
                )

                if fingerprint not in seen_prompts:
                    seen_prompts.add(fingerprint)
                    cleaned_samples.append(sample)

            except Exception:
                continue

    # Normalize task IDs
    for idx, sample in enumerate(cleaned_samples):
        sample["task_id"] = f"HumanEval/{idx}"

    # Save cleaned dataset
    with open(output_file, "w") as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Cleaned dataset saved with {len(cleaned_samples)} samples")


if __name__ == "__main__":
    clean_dataset(
        "Dataset/generated.jsonl",
        "Dataset/clean_generated.jsonl"
    )