from services.generator import generate_code
from services.evaluator import evaluate_code
from services.refiner import refine_prompt

problem = "Write a function add(a, b) that returns sum."

prompt = "Write a Python function."

# Test generator
code = generate_code(problem, prompt)
print("\nGenerated Code:\n", code)

# Fake error
error = "Wrong output"

# Test evaluator
feedback = evaluate_code(problem, code, error)
print("\nFeedback:\n", feedback)

# Test refiner
new_prompt = refine_prompt(prompt, feedback)
print("\nNew Prompt:\n", new_prompt)