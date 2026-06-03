import os
import json
import re
import concurrent.futures
from tqdm import tqdm
import argparse
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()

BASE_TASK_DESCRIBE = "You are an expert prompt engineer."
BASE_INFORMATION = """
Please help me improve the given system prompt for a code generation model.
The model will generate a Python program based on natural language descriptions.
Ensure the optimized prompt includes a detailed task description and guidelines.
"""
FORMAT_INSTRUCTION = """
Wrap the optimized prompt in {{}}.
"""

def build_mutation_prompt(base_prompt, feedback_data=None):
    if feedback_data and "errors" in feedback_data and feedback_data["errors"]:
        error_context = "The previous prompt failed on the following test cases due to these errors:\n"
        for i, err in enumerate(feedback_data["errors"][:5]):
            error_context += f"{i+1}. {err}\n"
        error_context += "\nPlease rewrite the system prompt specifically to address these edge cases and errors."
        
        full_prompt = (
            BASE_TASK_DESCRIBE + "\n" +
            BASE_INFORMATION + "\n" +
            f"Here is the previous prompt:\n{base_prompt}\n\n" +
            error_context + "\n" +
            FORMAT_INSTRUCTION
        )
    else:
        full_prompt = (
            BASE_TASK_DESCRIBE + "\n" +
            BASE_INFORMATION + "\n" +
            f"Here is the base prompt to improve:\n{base_prompt}\n" +
            FORMAT_INSTRUCTION
        )
    return full_prompt

def extract_wrapped_content(text):
    if not text or not isinstance(text, str):
        return None
    match = re.search(r"\{\{(.*?)\}\}", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_task(task_id, base_prompt, feedback_data, model):
    prompt = build_mutation_prompt(base_prompt, feedback_data)
    for attempt in range(3):
        completion = call_model(model, prompt)
        if completion is None:
            continue
        wrapped_content = extract_wrapped_content(completion)
        if wrapped_content:
            return {
                "prompt_id": task_id,
                "mutated_prompt": wrapped_content
            }
    print(f"⚠️ Task {task_id} exhausted all attempts.")
    return None

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main(model, base_prompt_path, feedback_path, output_path, num_mutations=5):
    model = model or get_default_model()
    
    base_prompts = read_jsonl(base_prompt_path)
    if not base_prompts:
        raise ValueError("No base prompt found.")
    
    # Use the best prompt from the previous round if feedback exists
    feedback_data = None
    if feedback_path and os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                feedback_data = json.loads(first_line)
                if "mutated_prompt" in feedback_data:
                    base_prompt_text = feedback_data["mutated_prompt"]
                elif "prompt" in feedback_data:
                    base_prompt_text = feedback_data["prompt"]
                else:
                    base_prompt_text = base_prompts[0].get("mutated_prompt", base_prompts[0].get("prompt", ""))
            else:
                base_prompt_text = base_prompts[0].get("mutated_prompt", base_prompts[0].get("prompt", ""))
    else:
        base_prompt_text = base_prompts[0].get("mutated_prompt", base_prompts[0].get("prompt", ""))
        
    samples = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_task, idx, base_prompt_text, feedback_data, model)
            for idx in range(num_mutations)
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_mutations, desc="Mutating Prompts"):
            result = future.result()
            if result:
                samples.append(result)
                
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_jsonl(output_path, samples)
    print(f"Generated {len(samples)} mutated prompts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--base_prompt_path", required=True)
    parser.add_argument("--feedback_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_mutations", type=int, default=5)
    
    args = parser.parse_args()
    
    main(
        args.model,
        str(resolve_pipeline_path(args.base_prompt_path)),
        str(resolve_pipeline_path(args.feedback_path)) if args.feedback_path else None,
        str(resolve_pipeline_path(args.output_path)),
        args.num_mutations,
    )
