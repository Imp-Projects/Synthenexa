from synthenexa_core.models.gemini_client import call_gemini


def evaluate_code(problem: str, code: str, error: str) -> str:
    prompt = f"""
You are an expert code debugging and review system.

You analyze code written in ANY programming language:
Python, Java, JavaScript, C++, Go, Rust, etc.

---

TASK:
Analyze the code and execution failure.

---

PROBLEM:
{problem}

---

GENERATED CODE:
{code}

---

EXECUTION ERROR:
{error}

---

STRICT RULES:
- Return ONLY valid JSON
- Do NOT include explanations outside JSON
- Do NOT include markdown
- Be precise and structured
- Keep responses short and technical

---

OUTPUT FORMAT (JSON ONLY):

{{
  "failure_reason": "string",
  "bug_explanation": "string",
  "missing_logic": "string",
  "suggested_fix": "string"
}}
"""

    return call_gemini(prompt)