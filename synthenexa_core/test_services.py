from synthenexa_core.services.generator import generate_code
from synthenexa_core.services.evaluator import evaluate_code
from synthenexa_core.services.refiner_code import refine_code
from synthenexa_core.services.runner import run_code

def run_pipeline():
    problem = "Return the second largest number in a list"

    # STEP 1: Generate
    code = generate_code(problem, prompt="Write a simple function to solve the problem.")
    print("\nGenerated Code:\n", code)

    for i in range(3):
        print(f"\n--- ITERATION {i+1} ---")

        # STEP 2: Run actual code
        success, score, error = run_code(code, problem)

        if success:
            print("✅ Code passed!")
            break

        print("❌ Error:\n", error)

        # STEP 3: Evaluate
        feedback = evaluate_code(problem, code, error)
        print("\nFeedback:\n", feedback)

        # STEP 4: Refine CODE (not prompt)
        code = refine_code(problem, code, feedback)
        print("\nRefined Code:\n", code)


if __name__ == "__main__":
    run_pipeline()