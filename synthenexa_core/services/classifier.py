def is_code_prompt(problem: str):
    keywords = [
        "python", "function", "code", "algorithm",
        "debug", "bug", "error", "leetcode",
        "implement", "write a program"
    ]

    text = problem.lower()

    return any(word in text for word in keywords)