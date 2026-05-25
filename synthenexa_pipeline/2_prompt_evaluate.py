import json
import os
import re
import argparse
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()


def write_jsonl(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def GEN_SOLUTION(task_describe, prompt, model):
    return call_model(model, f"{task_describe}\n\n{prompt}"), 0


def _sanitize_completion(text):
    if not text:
        return text
    # Prefer fenced code blocks if present.
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if blocks:
        return max(blocks, key=lambda b: len(b.strip())).strip()
    return text.strip()


def process_task(task_id, task_describe, prompt, model):
    completion, tokens_used = GEN_SOLUTION(task_describe, prompt, model)
    return dict(task_id=task_id, completion=_sanitize_completion(completion)), tokens_used


def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def main(model, testset_path, mutated_prompt_path, output_path, max_workers):
    model = model or get_default_model()
    model_slug = model.replace(":", "_").replace("/", "_")
    tasks = list(read_jsonl(testset_path))
    os.makedirs(output_path, exist_ok=True)
    mutated_prompt_rows = list(read_jsonl(mutated_prompt_path))

    total_tokens = 0  # Initialize total tokens counter

    for row in mutated_prompt_rows:
        prompt_id = row["prompt_id"]
        task_describe = row["mutated_prompt"]
        samples = []

        # Full-set evaluation policy: every mutated prompt is evaluated on every task.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
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

        # Keep a stable order for easier debugging/repro comparisons.
        samples.sort(key=lambda item: item["task_id"])
        output_file = os.path.join(output_path, f"train_set_{model_slug}_{prompt_id}.jsonl")
        write_jsonl(output_file, samples)

    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt evaluation using Ollama.")
    parser.add_argument('--model', type=str, default=None, help="Ollama model name from .env (e.g., 'qwen:latest').")
    parser.add_argument('--trainset_path', type=str, required=True, help="Path to the train set JSONL file.")
    parser.add_argument('--mutated_prompt_path', type=str, required=True, help="Path to the mutated prompt JSONL file.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output files.")
    parser.add_argument('--max_workers', type=int, default=50, help="ThreadPool workers for per-prompt task evaluation.")

    # Parse command-line arguments
    args = parser.parse_args()

    main(
        args.model,
        str(resolve_pipeline_path(args.trainset_path)),
        str(resolve_pipeline_path(args.mutated_prompt_path)),
        str(resolve_pipeline_path(args.output_path)),
        args.max_workers,
    )
