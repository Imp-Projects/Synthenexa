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

def generate_code(problem, prompt):
    full_prompt = f"""
You are an expert software engineer.

You can write code in ANY programming language including:
Python, Java, JavaScript, C++, Go, etc.

Task:
{prompt}

Problem:
{problem}

IMPORTANT RULES:
- Detect the correct programming language from the problem or prompt
- If language is explicitly mentioned, follow it
- Output ONLY code
- No explanation
- No markdown
- Code must be runnable
"""

    raw_output = call_gemini(full_prompt)
    return clean_code(raw_output)

if __name__ == "__main__":
    problem = "Write a function to check if a number is prime"
    prompt = "Generate a solution in Python"

    result = generate_code(problem, prompt)
    print("\n===== GENERATED CODE =====\n")
    print(result)