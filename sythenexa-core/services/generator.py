from models.ollama_client import call_model

def clean_code(output):
    # remove markdown ``` blocks
    if "```" in output:
        output = output.split("```")[1]
        output = output.replace("python", "")
    return output.strip()

def generate_code(problem, prompt):
    full_prompt = f"""
You are an expert Python programmer.

{prompt}

Problem:
{problem}

Return ONLY valid Python code. No explanation.
"""

    raw_output = call_model("qwen2.5:3b", full_prompt)

    return clean_code(raw_output)