from synthenexa_core.models.gemini_client import call_gemini

def refine_prompt(old_prompt, feedback):
    prompt = f"""
You are a prompt optimization expert.

Original Prompt:
{old_prompt}

Feedback:
{feedback}

Rewrite the prompt so that the next code generation attempt is more likely to pass all tests.

Return only the improved prompt.
"""

    return call_gemini(prompt)