from models.ollama_client import call_model


def evaluate_code(problem, code, error):
    prompt = f"""
You are a code evaluator.

Problem:
{problem}

Generated Code:
{code}

Execution Feedback:
{error}

Explain clearly:
1. Why it failed
2. What is wrong
3. What is missing
"""

    return call_model("qwen2.5:3b", prompt)