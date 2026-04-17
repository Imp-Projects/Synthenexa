from services.generator import generate_code
from services.runner import run_code
from services.evaluator import evaluate_code
from services.refiner import refine_prompt
from services.classifier import is_code_prompt


def run_pipeline(problem, iterations=3):
    prompt = "Solve the user's request clearly and correctly."
    history = []

    code_mode = is_code_prompt(problem) 
    if not code_mode:
        iterations = 1

    for i in range(iterations):
        print(f"\n--- Iteration {i+1} ---")

        response = generate_code(problem, prompt)

        passed = True
        score = 1.0
        error = "No execution needed"

        # only for coding tasks
        if code_mode:
            passed, score, error = run_code(response, problem)

        history.append({
            "iteration": i + 1,
            "response": response,
            "passed": passed,
            "score": score,
            "error": error
        })

        if passed:
            return {
                "success": True,
                "final_output": response,
                "history": history,
                "mode": "code" if code_mode else "general"
            }

        feedback = evaluate_code(problem, response, error)
        prompt = refine_prompt(prompt, feedback)

    return {
        "success": False,
        "final_output": response,
        "history": history,
        "mode": "code" if code_mode else "general"
    }