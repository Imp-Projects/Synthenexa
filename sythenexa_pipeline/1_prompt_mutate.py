import os
import json
import re
import concurrent.futures
from tqdm import tqdm
import argparse
import random
from evalplus.data import write_jsonl
from dotenv import load_dotenv

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


# MOCK MODEL RESPONSE (no API call)
def GEN_ANSWER(prompt, model):
    return "{{mock prompt}}", 0


# Extract {{prompt}}
def extract_wrapped_content(text):
    match = re.search(r"\{\{(.*?)\}\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def process_task(task_id, prompt, model):
    total_tokens = 0

    for _ in range(3):  # prevent infinite loops
        completion, tokens_used = GEN_ANSWER(prompt, model)
        total_tokens += tokens_used

        wrapped_content = extract_wrapped_content(completion)

        if wrapped_content:
            return {
                "prompt_id": task_id,
                "mutated_prompt": wrapped_content
            }, total_tokens

    return None, total_tokens


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main(model, prompt_path, output_path):

    data = read_jsonl(prompt_path)
    random_entry = random.choice(data)

    # fallback if field not present
    prompt = random_entry.get("mutated_prompt", random_entry.get("prompt", ""))

    samples = []
    total_tokens = 0

    problems = {i: {"prompt": prompt} for i in range(10)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

        futures = [
            executor.submit(process_task, task_id, problem["prompt"], model)
            for task_id, problem in problems.items()
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    main(args.model, args.prompt_path, args.output_path)