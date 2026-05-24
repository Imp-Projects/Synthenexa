import os
import json
import re
import concurrent.futures
import argparse
from tqdm import tqdm
from dotenv import load_dotenv


from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert in software engineering.

Generate a new HumanEval-style Python coding task.

Return output EXACTLY in this format:

[Start]
{{
    "task_id": "{task_id}",
    "prompt": "def add(a, b):\\n    pass",
    "entry_point": "add",
    "canonical_solution": "    return a + b",
    "test": "\\n\\ndef check(candidate):\\n    assert candidate(1,2)==3"
}}
[End]
"""


import random

def gen_answer(model, task_id):
    prompt = PROMPT_TEMPLATE.format(task_id=task_id)
    return call_model(model, prompt)


def extract_wrapped_content(text):
    match = re.search(r'\[Start\](.*?)\[End\]', text, re.DOTALL)
    if not match:
        return None

    content = match.group(1).strip()

    try:
        return json.loads(content)
    except Exception:
        return None


def process_task(task_num, model):
    task_id = f"HumanEval/{100 + task_num}"
    while True:
        completion = gen_answer(model, task_id)
        parsed = extract_wrapped_content(completion)
        if parsed:
            parsed["task_id"] = task_id
            return parsed
        print(f"Retrying task {task_id}...")


def main(output_path, total_tasks=10, model=None):
    samples = []
    selected_model = model or get_default_model()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_task, i, selected_model) for i in range(total_tasks)]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_tasks,
            desc="Generating dataset"
        ):
            result = future.result()
            if result:
                samples.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--total_tasks", type=int, default=10)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    out = str(resolve_pipeline_path(args.output_path))
    main(out, args.total_tasks, args.model)

def validate_sample(sample):
    required_keys = {
        "task_id",
        "prompt",
        "entry_point",
        "canonical_solution",
        "test",
    }

    if not required_keys.issubset(sample.keys()):
        return False, "Missing required keys"

    try:
        compile(sample["prompt"], "<prompt>", "exec")
        compile(sample["canonical_solution"], "<solution>", "exec")
        compile(sample["test"], "<test>", "exec")
    except Exception as e:
        return False, f"Syntax error: {e}"

    if f"def {sample['entry_point']}" not in sample["prompt"]:
        return False, "Entry point mismatch"

    test_asserts = sample["test"].count("assert")
    if test_asserts < 2:
        return False, "Weak tests"

    return True, "Valid"


def validate_dataset(input_path, output_path):
    seen_prompts = set()
    valid_samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            if sample["prompt"] in seen_prompts:
                continue

            is_valid, reason = validate_sample(sample)

            if is_valid:
                valid_samples.append(sample)
                seen_prompts.add(sample["prompt"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Validated samples: {len(valid_samples)} -> {output_path}")
