from synthenexa_core.models.gemini_client import call_gemini

def evaluate_code(problem, code, error):
    prompt = f"""
You are a code evaluator.

Problem:
{problem}

Generated Code:
{code}

Execution Feedback:
{error}

Respond in this format:

- Failure Reason:
- Bug Explanation:
- Missing Logic:
- Suggested Fix:
"""

    return call_gemini(prompt)