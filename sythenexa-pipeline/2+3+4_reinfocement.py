import os
import re
import json
import random
import subprocess
import concurrent.futures
from tqdm import tqdm
from evalplus.data import write_jsonl
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set API Key
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Task description and information used for generating optimized prompts
task_describe = """
You are an expert prompt engineer.
"""

information = """
Please help me improve the given prompt to get a more helpful and harmless response.
Suppose I need to generate a Python program based on natural language descriptions.
The generated Python program should be able to complete the tasks described in natural language and pass any test cases specific to those tasks.\n
"""

format = """
You may add any information you think will help improve the task's effectiveness during the prompt optimization process.
If you find certain expressions and wording in the original prompt inappropriate, you can also modify these usages.
Ensure that the optimized prompt includes a detailed task description and clear process guidance added to the original prompt.
Wrap the optimized prompt in {{}}.
"""

def GEN_SOLUTION(task_describe, prompt):
    return """
def solution(x):
    return x
""" 
# Process task
def process_task(task_id, task_describe, prompt):
    completion = GEN_SOLUTION(task_describe, prompt)
    return dict(task_id=task_id, completion=completion)

# Read JSONL file
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield json.loads(line)

# Generate solutions and save to the specified directory
def generate_solutions(test_set_path, mutated_prompt_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    mutated_prompts = {task["prompt_id"]: task["mutated_prompt"] for task in read_jsonl(mutated_prompt_path)}

    for prompt_id, task_describe in mutated_prompts.items():
        samples = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    process_task,
                    task["task_id"],
                    task_describe,
                    task["prompt"]
                )
                for task in read_jsonl(test_set_path)
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc=f"Processing tasks for prompt_id {prompt_id}"):
                result = future.result()
                if result['completion']:
                    samples.append(result)
                else:
                    print(f"No valid completion for task_id: {result['task_id']} with prompt_id: {prompt_id}")

        # Train set name
        output_file = os.path.join(output_directory, f"train_set_gpt3.5turbo_{prompt_id}.jsonl")
        write_jsonl(output_file, samples)

def evaluate_and_select_best_prompts(folder_path, problem_file_path, prompts_file_path, best_prompt_output_path):

    # Read all prompts
    with open(prompts_file_path, "r") as f:
        prompts = [json.loads(line) for line in f]

    if not prompts:
        print("No prompts found.")
        return

    # Mock: select first prompt as best
    best_prompt = prompts[0]

   
    with open(best_prompt_output_path, "w") as f:
        json.dump(best_prompt, f)
        f.write("\n")

    print("Best prompt selected (mock).")

# Generate optimized prompts
def GEN_ANSWER(prompt):
    return "{{Write a clear Python function to solve the problem described in natural language.}}"

def extract_wrapped_content(text):
    match = re.search(r'\{\{(.*?)\}\}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""

def process_optimization_task(task_id, prompt):
    while True:
        completion = GEN_ANSWER(prompt)
        wrapped_content = extract_wrapped_content(completion)
        if wrapped_content:
            return dict(prompt_id=task_id, mutated_prompt=wrapped_content)
        else:
            print(f"Task {task_id}: No wrapped content found. Retrying...")

def generate_new_prompts(existing_prompts):
    new_prompts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for task_id in range(10):
            random_prompt = random.choice(existing_prompts)
            prompt_text = random_prompt['mutated_prompt']
            formatted_prompt = f"The prompt ready to be optimized are as follows and wrapped in []:\n[{prompt_text}]\n"
            futures.append(executor.submit(process_optimization_task, task_id, formatted_prompt))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
            new_prompts.append(future.result())

    return new_prompts

def optimize_prompts(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return

    with open(input_file, 'r') as file:
        prompts = [json.loads(line) for line in file]

    new_prompts = generate_new_prompts(prompts)

    existing_ids = {prompt['prompt_id'] for prompt in prompts}
    new_id = 0
    for new_prompt in new_prompts:
        while new_id in existing_ids:
            new_id += 1
        new_prompt['prompt_id'] = new_id
        existing_ids.add(new_id)

    new_prompts = [prompt for prompt in new_prompts if prompt['prompt_id'] <= 9]

    combined_prompts = prompts + new_prompts
    with open(output_file, 'w') as out_file:
        for prompt in combined_prompts:
            json.dump(prompt, out_file)
            out_file.write('\n')

    print(f"New prompts saved to {output_file}")

# Main program entry point
if __name__ == "__main__":
    
    test_set_path = "data/train_set.jsonl"
    mutated_prompt_path = "data/mutated_prompts.jsonl"
    output_directory = "data/evaluation_output"

    generate_solutions(test_set_path, mutated_prompt_path, output_directory)

    folder_path = "data/evaluation_output"
    problem_file_path = "data/train_set.jsonl"
    prompts_file_path = "data/mutated_prompts.jsonl"
    best_prompt_output_path = "data/best_prompt.jsonl"

    evaluate_and_select_best_prompts(
        folder_path,
        problem_file_path,
        prompts_file_path,
        best_prompt_output_path
    )

    optimize_input_file = best_prompt_output_path
    optimize_output_file = "data/optimized_prompts.jsonl"

    optimize_prompts(optimize_input_file, optimize_output_file)