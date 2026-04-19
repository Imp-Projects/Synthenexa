from synthenexa_core.models.gemini_client import call_gemini

def clean_code(output: str) -> str:
    import re

    # extract all code blocks
    blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)

    if blocks:
        # prefer longest meaningful block
        return max(blocks, key=lambda x: len(x.strip())).strip()

    # fallback cleanup
    return output.strip().replace("```", "")

def generate_code(problem: str, prompt: str) -> str:
    full_prompt = f"""
You are an expert multi-language software engineer.

You write code in ANY language:
Python, Java, JavaScript, C++, Go, Rust, etc.

TASK:
{prompt}

PROBLEM:
{problem}

RULES:
- Detect language automatically from prompt/problem
- If language is mentioned, strictly follow it
- Output ONLY code
- No explanation
- No markdown
- Ensure code is runnable and complete
"""

    raw_output = call_gemini(full_prompt)
    return clean_code(raw_output)

if __name__ == "__main__":
    problem = "Write a function to check if a number is prime"
    prompt = "Generate a solution in Python"

    result = generate_code(problem, prompt)
    print("\n===== GENERATED CODE =====\n")
    print(result)