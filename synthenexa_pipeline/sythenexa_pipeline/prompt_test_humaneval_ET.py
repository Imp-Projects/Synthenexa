import re
import json
import os
import argparse
from evalplus.data import write_jsonl
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path


load_dotenv()

# TODO: change the prompt
task_describe_template = """

"""


def GEN_SOLUTION(prompt, entry_point, model):
    attempts = 0
    task_describe = task_describe_template.format(entry_point=entry_point)

    while attempts < 5:
        response_text = call_model(model, f"{task_describe}\n\n{prompt}")
        match = re.search(r'```python(.*?)```', response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if entry_point in code:
                return code
        attempts += 1

    print(f"Failed to extract valid Python code after 5 attempts for prompt: {prompt}")
    return None


def process_task(task_id, prompt, entry_point, model):
    completion = GEN_SOLUTION(prompt, entry_point, model)
    if completion is not None:
        return dict(task_id=task_id, entry_point=entry_point, completion=completion)
    else:
        return dict(task_id=task_id, entry_point=entry_point, completion="")


def load_tasks_from_jsonl(file_path):
    tasks = {}
    with open(file_path, 'r') as file:
        for line in file:
            task = json.loads(line)
            tasks[task['task_id']] = task
    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HumanEval prompt test with Ollama.")
    parser.add_argument("--input_file_path", required=True)
    parser.add_argument("--output_file_path", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    problems = load_tasks_from_jsonl(str(resolve_pipeline_path(args.input_file_path)))
    model = args.model or get_default_model()
    samples = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_task, task_id, problem["prompt"], problem["entry_point"], model)
            for task_id, problem in problems.items()
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
            samples.append(future.result())

    out_path = str(resolve_pipeline_path(args.output_file_path))
    write_jsonl(out_path, samples)
    print(f"Results saved to {out_path}")