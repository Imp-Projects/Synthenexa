from synthenexa_core.models.gemini_client import call_gemini

def refine_code(problem: str, code: str, feedback: str) -> str:
    prompt = f"""
You are an expert Python developer.

Problem:
{problem}

Current Code:
{code}

Feedback:
{feedback}

Fix the code so that it passes all test cases and resolves the issues.

STRICT RULES:
- Return ONLY valid Python code
- Do NOT include explanation
- Do NOT include markdown
"""

    return call_gemini(prompt)