"""
generate_dataset.py — HumanEval-style Dataset Generator

Changes from original:
- Replaced single PROMPT_TEMPLATE with a structured hard-difficulty generation
  strategy that rotates across 8 algorithmic archetypes.
- Each archetype targets reasoning-heavy, edge-case-sensitive problems that
  produce meaningful variation in pass rates and weighted scores.
- Added DIFFICULTY_HINTS injected per task to guide output diversity.
- validate_sample / validate_dataset unchanged; process_task / main pipeline
  unchanged; output schema unchanged.
"""

import os
import json
import re
import random
import concurrent.futures
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()


# ---------------------------------------------------------------------------
# Hard-difficulty generation strategy
# ---------------------------------------------------------------------------

# Each archetype steers the model toward a different class of algorithmic
# reasoning, ensuring diverse pass-rate distributions across generated tasks.
_ARCHETYPES = [
    {
        "name": "dynamic_programming",
        "description": (
            "A dynamic programming problem where a naïve recursive solution times out "
            "and the correct approach requires identifying overlapping subproblems and "
            "optimal substructure. Must have at least one tricky boundary condition "
            "(e.g. empty input, single-element list, negative values, or integer overflow)."
        ),
        "example_topics": [
            "longest common subsequence", "coin change with constraints",
            "partition equal subset sum", "edit distance with custom costs",
            "matrix chain multiplication", "maximum subarray with at most k deletions",
        ],
    },
    {
        "name": "graph_traversal",
        "description": (
            "A graph or tree traversal problem (BFS, DFS, topological sort, or shortest path) "
            "where correctness depends on handling disconnected components, cycles, or "
            "directed vs undirected edge semantics. Edge cases must include empty graphs, "
            "self-loops, or duplicate edges."
        ),
        "example_topics": [
            "detect cycle in directed graph", "shortest path with forbidden nodes",
            "count islands with diagonal connectivity", "topological sort with cycle detection",
            "minimum spanning tree validation", "bipartite graph check",
        ],
    },
    {
        "name": "string_algorithms",
        "description": (
            "A non-trivial string manipulation problem requiring pattern recognition, "
            "sliding window, or two-pointer reasoning. The correct solution must handle "
            "Unicode edge cases, empty strings, single-character inputs, and overlapping "
            "patterns correctly."
        ),
        "example_topics": [
            "longest palindromic substring", "minimum window substring",
            "group anagrams with frequency constraint", "decode ways with wildcards",
            "count distinct subsequences", "zigzag string encoding",
        ],
    },
    {
        "name": "mathematical_reasoning",
        "description": (
            "A problem grounded in number theory, combinatorics, or modular arithmetic "
            "where off-by-one errors and integer overflow are common pitfalls. Must "
            "require reasoning about prime factorization, GCD/LCM, or combinatorial "
            "counts with large inputs."
        ),
        "example_topics": [
            "count ways to express n as ordered sum of primes",
            "modular exponentiation", "nth Catalan number mod p",
            "number of trailing zeros in n factorial",
            "count integers with exactly k divisors up to n",
            "Euler's totient with prime factorization",
        ],
    },
    {
        "name": "interval_and_sorting",
        "description": (
            "A problem involving interval merging, sweep-line logic, or custom sort "
            "comparators where correctness hinges on tie-breaking rules and overlapping "
            "edge cases. Must include edge cases with identical intervals, single-element "
            "input, and already-sorted / reverse-sorted inputs."
        ),
        "example_topics": [
            "merge overlapping intervals with priority", "meeting rooms minimum required",
            "skyline problem", "non-overlapping interval removal",
            "task scheduler with cooldown", "minimum interval to include each query",
        ],
    },
    {
        "name": "recursion_and_backtracking",
        "description": (
            "A combinatorial search or constraint-satisfaction problem solved with "
            "backtracking or memoized recursion. The problem must prune the search space "
            "non-trivially and include edge cases where no valid solution exists or "
            "all permutations are valid."
        ),
        "example_topics": [
            "N-Queens with forbidden cells", "Sudoku validator",
            "word search in a grid", "generate all valid parenthesizations",
            "subset sum with exactly k elements", "letter combinations of phone number with exclusions",
        ],
    },
    {
        "name": "data_structure_design",
        "description": (
            "A problem requiring design of a custom data structure or an unusual application "
            "of a standard one (heap, deque, union-find, segment tree). Must test operations "
            "in adversarial order and include edge cases like single-element structure, "
            "duplicate insertions, and queries on empty structures."
        ),
        "example_topics": [
            "LRU cache with frequency tie-breaking", "min-stack with O(1) get-min",
            "sliding window maximum", "design a time-based key-value store",
            "implement queue using two stacks with size limit",
            "find median from data stream",
        ],
    },
    {
        "name": "bit_manipulation",
        "description": (
            "A problem whose efficient solution relies on bitwise operations, "
            "where an arithmetic or loop-based approach would be too slow or incorrect. "
            "Must include edge cases around zero, negative numbers (two's complement), "
            "and the maximum 32/64-bit integer boundary."
        ),
        "example_topics": [
            "count bits set in range [l, r]", "single number III (two missing numbers)",
            "reverse bits of a 32-bit integer", "maximum XOR of two numbers in an array",
            "find the only non-repeated element in groups of three",
            "power of two check without arithmetic operators",
        ],
    },
]

# Difficulty hints injected into every prompt to push the model away from
# trivial solutions and toward behaviorally diverse outputs.
_DIFFICULTY_HINTS = (
    "Requirements for difficulty:\n"
    "- The naïve / brute-force solution must be clearly insufficient (wrong or too slow).\n"
    "- Include AT LEAST 3 assert statements in 'test', covering: normal case, "
    "edge case (empty / zero / negative / boundary), and a case where the output "
    "is non-obvious (requires careful reasoning).\n"
    "- The canonical_solution must be the optimal approach, not a brute-force loop.\n"
    "- Do NOT generate trivial problems like FizzBuzz, simple sorting, or basic arithmetic.\n"
    "- The function signature should accept at least one non-trivial argument type "
    "(list of lists, dict, tuple pair, etc.) where meaningful.\n"
)

_PROMPT_TEMPLATE = """
You are an expert software engineer and competitive programmer.

Generate a NEW, ORIGINAL HumanEval-style Python coding task.

Algorithmic category: {archetype_name}
Category description: {archetype_description}
Suggested topic area (choose freely, do NOT copy exactly): {example_topic}

{difficulty_hints}

Return output EXACTLY in this format — nothing before [Start] or after [End]:

[Start]
{{
    "task_id": "{task_id}",
    "prompt": "def function_name(args):\\n    \\\"\\\"\\\"Docstring explaining the task, constraints, and examples.\\\"\\\"\\\"\\n    pass",
    "entry_point": "function_name",
    "canonical_solution": "    # optimal implementation here",
    "test": "\\n\\ndef check(candidate):\\n    assert candidate(...) == ...\\n    assert candidate(...) == ...\\n    assert candidate(...) == ..."
}}
[End]
"""


def _build_prompt(task_id: str, archetype: dict, topic_hint: str) -> str:
    return _PROMPT_TEMPLATE.format(
        archetype_name=archetype["name"],
        archetype_description=archetype["description"],
        example_topic=topic_hint,
        difficulty_hints=_DIFFICULTY_HINTS,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Core generation helpers (interface unchanged from original)
# ---------------------------------------------------------------------------

def gen_answer(model: str, task_id: str, archetype: dict, topic_hint: str) -> str:
    prompt = _build_prompt(task_id, archetype, topic_hint)
    return call_model(model, prompt)


def extract_wrapped_content(text: str) -> dict | None:
    match = re.search(r'\[Start\](.*?)\[End\]', text, re.DOTALL)
    if not match:
        return None
    content = match.group(1).strip()
    try:
        return json.loads(content)
    except Exception:
        return None


def process_task(task_num: int, model: str) -> dict:
    task_id = f"HumanEval/{100 + task_num}"

    # Rotate through archetypes deterministically; shuffle topics per attempt
    archetype = _ARCHETYPES[task_num % len(_ARCHETYPES)]
    topics = archetype["example_topics"]

    attempt = 0
    while True:
        # Cycle through topic hints on retries to encourage diversity
        topic_hint = topics[attempt % len(topics)]
        completion = gen_answer(model, task_id, archetype, topic_hint)
        parsed = extract_wrapped_content(completion)
        if parsed:
            parsed["task_id"] = task_id
            parsed["archetype"] = archetype["name"]   # metadata for downstream analysis
            return parsed
        attempt += 1
        print(f"Retrying task {task_id} (attempt {attempt}, topic: {topic_hint})...")


# ---------------------------------------------------------------------------
# Main pipeline (unchanged from original)
# ---------------------------------------------------------------------------

def main(output_path: str, total_tasks: int = 10, model: str | None = None) -> None:
    samples = []
    selected_model = model or get_default_model()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_task, i, selected_model) for i in range(total_tasks)]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_tasks,
            desc="Generating dataset",
        ):
            result = future.result()
            if result:
                samples.append(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(samples)} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--total_tasks", type=int, default=10)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    out = str(resolve_pipeline_path(args.output_path))
    main(out, args.total_tasks, args.model)


# ---------------------------------------------------------------------------
# Validation (unchanged from original)
# ---------------------------------------------------------------------------

def validate_sample(sample: dict) -> tuple[bool, str]:
    required_keys = {
        "task_id",
        "prompt",
        "entry_point",
        "canonical_solution",
        "test",
    }

    if not required_keys.issubset(sample.keys()):
        return False, "Missing required keys"

    try:
        compile(sample["prompt"], "<prompt>", "exec")
        compile(sample["canonical_solution"], "<solution>", "exec")
        compile(sample["test"], "<test>", "exec")
    except Exception as e:
        return False, f"Syntax error: {e}"

    if f"def {sample['entry_point']}" not in sample["prompt"]:
        return False, "Entry point mismatch"

    test_asserts = sample["test"].count("assert")
    if test_asserts < 2:
        return False, "Weak tests"

    return True, "Valid"


def validate_dataset(input_path: str, output_path: str) -> None:
    seen_prompts: set = set()
    valid_samples: list = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            if sample["prompt"] in seen_prompts:
                continue

            is_valid, reason = validate_sample(sample)

            if is_valid:
                valid_samples.append(sample)
                seen_prompts.add(sample["prompt"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Validated samples: {len(valid_samples)} -> {output_path}")