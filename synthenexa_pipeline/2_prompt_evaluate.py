import json
import os
import re
import argparse
import concurrent.futures
from tqdm import tqdm
from evalplus.data import write_jsonl
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()


def GEN_SOLUTION(task_describe, prompt, model):
    return call_model(model, f"{task_describe}\n\n{prompt}"), 0


def process_task(task_id, task_describe, prompt, model):
    completion, tokens_used = GEN_SOLUTION(task_describe, prompt, model)
    return dict(task_id=task_id, completion=completion), tokens_used


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def main(model, testset_path, mutated_prompt_path, output_path):
    model = model or get_default_model()
    model_slug = model.replace(":", "_").replace("/", "_")
    tasks = list(read_jsonl(testset_path))
    os.makedirs(output_path, exist_ok=True)

    # Read mutated prompt file
    mutated_prompts = {task["prompt_id"]: task["mutated_prompt"] for task in read_jsonl(mutated_prompt_path)}

    total_tokens = 0  # Initialize total tokens counter

    for prompt_id, task_describe in mutated_prompts.items():
        samples = []

        # Use multithreading to speed up task processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    process_task,
                    task["task_id"],
                    task_describe,
                    task.get("prompt", ""),
                    model,
                )
                for task in tasks
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"Processing tasks for prompt_id {prompt_id}"):
                result, tokens_used = future.result()
                total_tokens += tokens_used  # Accumulate total tokens
                if result['completion']:
                    samples.append(result)
                else:
                    print(f"No valid completion for task_id: {result['task_id']} with prompt_id: {prompt_id}")

        output_file = os.path.join(output_path, f"train_set_{model_slug}_{prompt_id}.jsonl")
        write_jsonl(output_file, samples)

    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt evaluation using Ollama.")
    parser.add_argument('--model', type=str, default=None, help="Ollama model name from .env (e.g., 'qwen:latest').")
    parser.add_argument('--trainset_path', type=str, required=True, help="Path to the train set JSONL file.")
    parser.add_argument('--mutated_prompt_path', type=str, required=True, help="Path to the mutated prompt JSONL file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output files.")

    # Parse command-line arguments
    args = parser.parse_args()

    main(
        args.model,
        str(resolve_pipeline_path(args.trainset_path)),
        str(resolve_pipeline_path(args.mutated_prompt_path)),
        str(resolve_pipeline_path(args.output_path)),
    )