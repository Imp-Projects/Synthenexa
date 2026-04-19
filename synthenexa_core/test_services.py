from synthenexa_core.services.generator import generate_code
from synthenexa_core.services.evaluator import evaluate_code
from synthenexa_core.services.refiner import refine_prompt


def run_pipeline():
    problem = "Write a function add(a, b) that returns sum."
    prompt = "Write a Python function."

    # STEP 1: Generate
    code = generate_code(problem, prompt)
    print("\nGenerated Code:\n", code)

    # STEP 2: Simulated execution error
    error = "Wrong output format or failed test cases"

    # STEP 3: Evaluate
    feedback = evaluate_code(problem, code, error)
    print("\nFeedback:\n", feedback)

    # STEP 4: Refine prompt
    new_prompt = refine_prompt(prompt, feedback)
    print("\nNew Prompt:\n", new_prompt)


if __name__ == "__main__":
    run_pipeline()