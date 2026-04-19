from synthenexa_core.models.gemini_client import call_gemini

def refine_prompt(old_prompt: str, feedback: str) -> str:
    prompt = f"""
You are a senior prompt engineering expert for AI code generation systems.

Your job is to improve prompts based on structured debugging feedback.

---

ORIGINAL PROMPT:
{old_prompt}

---

FEEDBACK (JSON from evaluator):
{feedback}

---

TASK:
Rewrite the prompt so that the next generated code:
- fixes all issues mentioned in feedback
- handles missing edge cases
- avoids previous failure patterns
- improves correctness and robustness

---

STRICT RULES:
- Return ONLY the improved prompt
- Do NOT include explanations
- Do NOT include JSON
- Do NOT include markdown
- Keep it compact but powerful
- Ensure it works for ANY programming language

---

OPTIMIZATION STRATEGY:
- Fix root cause mentioned in "failure_reason"
- Address missing logic explicitly
- Strengthen constraints in prompt
- Add clarity to expected output format if needed
"""

    return call_gemini(prompt)