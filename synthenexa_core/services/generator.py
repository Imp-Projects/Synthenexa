from synthenexa_core.models.ollama_client import call_model
import re

def clean_code(output: str) -> str:
    blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)

    if blocks:
        return max(blocks, key=lambda x: len(x.strip())).strip()

    return output.strip().replace("```", "")


def generate_code(problem: str, prompt: str) -> str:
    full_prompt = f"""
You are an expert software engineer.

Your task is to generate correct, executable code.

Instructions:
- Detect programming language automatically
- Output ONLY code
- No explanation
- No markdown
- No comments unless necessary for code execution

Problem:
{problem}

Task:
{prompt}
"""

    raw = call_model("qwen2.5:3b", full_prompt)
    return clean_code(raw)