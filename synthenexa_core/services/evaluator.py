from synthenexa_core.models.gemini_client import call_gemini


def evaluate_code(problem: str, code: str, error: str) -> str:
    prompt = f"""
You are an expert code reviewer and debugging assistant.

You evaluate code written in ANY programming language:
Python, Java, JavaScript, C++, Go, Rust, etc.

---

TASK:
Analyze the generated code and execution failure.

---

PROBLEM:
{problem}

---

GENERATED CODE:
{code}

---

EXECUTION ERROR / OUTPUT:
{error}

---

STRICT INSTRUCTIONS:
- Be precise and technical
- Assume code may be in any language
- Do NOT rewrite full code unless necessary
- Focus on debugging reasoning
- Identify root cause, not symptoms

---

RESPOND ONLY IN THIS FORMAT:

Failure Reason:
<short root cause>

Bug Explanation:
<technical explanation of what exactly is wrong>

Missing Logic:
<what the code failed to handle or implement>

Suggested Fix:
<high-level fix instructions (not full rewrite unless needed)>
"""

    return call_gemini(prompt)