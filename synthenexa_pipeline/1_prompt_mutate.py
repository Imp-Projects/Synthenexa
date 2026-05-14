import os
import json
import re
import concurrent.futures
from tqdm import tqdm
import argparse
from evalplus.data import write_jsonl
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()

task_describe = """
You are an expert prompt engineer.
"""

information = """
Please help me improve the given prompt to get a more helpful and harmless response.
Suppose I need to generate a Python program based on natural language descriptions.
"""

format = """
Ensure the optimized prompt includes a detailed task description.
Wrap the optimized prompt in {{}}.
"""


def GEN_ANSWER(prompt, model):
    full_prompt = task_describe + information + prompt + format
    return call_model(model, full_prompt), 0


# Extract {{prompt}}
def extract_wrapped_content(text):
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"\{\{(.*?)\}\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def process_task(task_id, source_task_id, prompt, model):
    total_tokens = 0

    for attempt in range(3):  # prevent infinite loops
        completion, tokens_used = GEN_ANSWER(prompt, model)
        total_tokens += tokens_used

        # Handle None responses
        if completion is None:
            print(f"  Attempt {attempt + 1}/3 failed for task {task_id}")
            continue

        wrapped_content = extract_wrapped_content(completion)

        if wrapped_content:
            return {
                "prompt_id": task_id,
                "source_task_id": source_task_id,
                "mutated_prompt": wrapped_content
            }, total_tokens

    print(f"⚠️  Task {task_id} exhausted all attempts, skipping...")
    return None, total_tokens


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main(model, prompt_path, output_path, num_mutations=10):
    model = model or get_default_model()

    data = read_jsonl(prompt_path)
    if not data:
        raise ValueError(f"No prompts found in {prompt_path}")

    samples = []
    total_tokens = 0

    # Task-aware / prompt-group mutation:
    # build mutation jobs from multiple source tasks (not a single global prompt).
    jobs = []
    for entry in data:
        source_task_id = entry.get("task_id")
        source_prompt = entry.get("mutated_prompt", entry.get("prompt", ""))
        if not source_prompt:
            continue
        jobs.append((source_task_id, source_prompt))
        if len(jobs) >= num_mutations:
            break

    if not jobs:
        raise ValueError("No valid prompts available for mutation.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        futures = [
            executor.submit(process_task, idx, source_task_id, source_prompt, model)
            for idx, (source_task_id, source_prompt) in enumerate(jobs)
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing tasks",
        ):
            result, tokens_used = future.result()

            if result:
                samples.append(result)

            total_tokens += tokens_used

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    write_jsonl(output_path, samples)

    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_mutations", type=int, default=10)

    args = parser.parse_args()

    main(
        args.model,
        str(resolve_pipeline_path(args.prompt_path)),
        str(resolve_pipeline_path(args.output_path)),
        args.num_mutations,
    )
