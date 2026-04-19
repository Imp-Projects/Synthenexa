from synthenexa_core.models.ollama_client import call_qwen
import re


def clean_code(output: str) -> str:
    blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)

    if blocks:
        return max(blocks, key=lambda x: len(x.strip())).strip()

    return output.strip().replace("```", "")


def generate_code(problem: str, prompt: str) -> str:
    full_prompt = f"""
You are an expert software engineer.

You can write code in ANY programming language.

Task:
{prompt}

Problem:
{problem}

Rules:
- Detect language automatically
- Output ONLY code
- No explanation
- No markdown
"""

    raw = call_qwen(full_prompt)
    return clean_code(raw)