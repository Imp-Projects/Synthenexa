import json
import os
import re
import textwrap
import argparse
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from utils.ollama_client import call_model, get_default_model
from utils.paths import resolve_pipeline_path

load_dotenv()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Stub builder — handles tasks where prompt field is empty (e.g. MBPP)
# ---------------------------------------------------------------------------

def _build_stub(task: dict) -> str:
    """
    Return the function stub to send to the model.

    HumanEval tasks have a fully-formed stub in the 'prompt' field.
    MBPP tasks have an empty 'prompt' but a natural-language 'text' field
    and an 'entry_point'. We construct a minimal stub from those so the
    model always receives a concrete function signature to complete.
    """
    prompt = task.get("prompt", "").strip()
    if prompt:
        return prompt

    # Fallback: build a stub from entry_point + text description
    entry_point = task.get("entry_point", "solve")
    description = task.get("text", "").strip()
    stub = f'def {entry_point}(*args):\n'
    if description:
        stub += f'    """{description}"""\n'
    stub += '    pass'
    return stub


# ---------------------------------------------------------------------------
# Completion normalisation
# ---------------------------------------------------------------------------

def _extract_fenced_block(text: str) -> str:
    """
    Return the content between the first ```python fence and the closing ```.
    Falls back to the whole string if no fences are found.
    """
    match = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback: try a bare ``` fence (no language tag)
    match = re.search(r"```\s*([\s\S]*?)```", text)
    if match:
        return match.group(1)
    return text


def _reindent(code: str, spaces: int = 4) -> str:
    """
    Remove common leading whitespace with textwrap.dedent, then re-indent
    every non-empty, non-comment line by `spaces` spaces.

    A line is considered a comment only when '#' is its first non-whitespace
    character. Inline comments (code followed by #) are left untouched because
    they are not standalone comment lines.
    """
    code = textwrap.dedent(code)
    prefix = " " * spaces
    result = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            # preserve blank lines as-is (no indent added)
            result.append("")
        elif stripped.startswith("#"):
            # standalone comment — re-indent like any other line
            result.append(prefix + stripped)
        else:
            result.append(prefix + stripped)
    return "\n".join(result)


def _normalise_completion(raw: str) -> str:
    """
    Full normalisation pipeline applied to every raw model response:
      1. Extract the fenced ```python ... ``` block (fallback: whole string).
      2. textwrap.dedent to remove common leading whitespace.
      3. Re-indent every non-empty line to exactly 4 spaces per level.
      4. Strip surrounding blank lines.
    """
    block = _extract_fenced_block(raw)
    normalised = _reindent(block, spaces=4)
    # Strip leading/trailing blank lines without touching inner indentation
    lines = normalised.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Completion sanitization
# ---------------------------------------------------------------------------

def _sanitize_completion(text: str, entry_point: str | None = None) -> str:
    """
    Clean the raw model output so it contains only the function body lines
    that the HumanEval evaluator expects to be appended to the prompt stub.

    Steps:
    1. Strip model reasoning tags (<think>...</think> / </think>).
    2. Extract the longest fenced code block if present.
    3. If the model re-declared the full function signature, strip it.
    4. Ensure every line is indented by at least 4 spaces (fixes unindented output).
    5. Strip leading/trailing blank lines.
    """
    if not text:
        return text

    # Step 1: strip reasoning tags that some models leak into output
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)

    # Step 2: prefer fenced code block content
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if blocks:
        text = max(blocks, key=lambda b: len(b.strip()))

    text = text.strip()

    # Step 3: if the model repeated the function signature, drop it
    if entry_point:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
                text = "\n".join(lines[i + 1:])
                break

    # Step 4: ensure body is indented — if no line starts with whitespace,
    # the model returned flat code which causes IndentationError when appended
    # to the stub. Prefix every non-empty line with 4 spaces.
    lines = text.splitlines()
    if lines and not lines[0].startswith(" ") and not lines[0].startswith("\t"):
        lines = ["    " + line if line.strip() else line for line in lines]
        text = "\n".join(lines)

    # Remove leading and trailing blank lines, but keep indentation of content
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are an expert Python programmer. {style_instruction}

Your task is to implement the body of the Python function whose stub is provided.

Rules:
- Return ONLY the indented function body (the lines that go inside the def).
- Do NOT repeat the function signature or the docstring.
- Do NOT wrap your answer in markdown fences.
- Do NOT define any extra helper functions outside the given function.
- The implementation must be correct and handle all edge cases described in the docstring.
"""

_USER_TEMPLATE = """\
Complete the following Python function. The function you implement MUST be named `{entry_point}`.
Return only the indented body lines — do not repeat the signature.

{prompt_stub}
"""


def GEN_SOLUTION(task_describe: str, prompt_stub: str, entry_point: str, model: str) -> tuple[str, int]:
    """
    Call the model to implement *prompt_stub* under the persona/style defined
    by *task_describe* (the mutated prompt).

    The mutated prompt is embedded as a system message so it shapes *how* the
    model codes without overriding the task itself. call_model now forwards
    system= to /api/chat (Ollama) or chat completions (Groq).
    """
    system_msg = _SYSTEM_TEMPLATE.format(style_instruction=task_describe)
    user_msg = _USER_TEMPLATE.format(entry_point=entry_point, prompt_stub=prompt_stub)
    raw = call_model(model, user_msg, system=system_msg)

    # Normalise immediately after receiving the raw response so that all
    # downstream code (sanitization, evaluation) always sees clean, consistently
    # indented Python — regardless of what the model actually returned.
    normalised = _normalise_completion(raw) if raw else raw

    return normalised, 0


def process_task(
    task_id: str,
    task_describe: str,
    prompt_stub: str,
    entry_point: str,
    model: str,
) -> tuple[dict, int]:
    raw_completion, tokens_used = GEN_SOLUTION(task_describe, prompt_stub, entry_point, model)
    clean = _sanitize_completion(raw_completion, entry_point=entry_point)
    return dict(task_id=task_id, completion=clean, raw_completion=raw_completion, prompt=task_describe, prompt_length=len(task_describe)), tokens_used


# ---------------------------------------------------------------------------
# Main pipeline (interface unchanged)
# ---------------------------------------------------------------------------

def main(
    model: str | None,
    testset_path: str,
    mutated_prompt_path: str,
    output_path: str,
    max_workers: int,
) -> None:
    model = model or get_default_model()
    model_slug = model.replace(":", "_").replace("/", "_")
    tasks = list(read_jsonl(testset_path))
    os.makedirs(output_path, exist_ok=True)
    mutated_prompt_rows = list(read_jsonl(mutated_prompt_path))

    total_tokens = 0

    for row in mutated_prompt_rows:
        prompt_id = row["prompt_id"]
        task_describe = row["mutated_prompt"]
        samples = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_task,
                    task["task_id"],
                    task_describe,
                    _build_stub(task),             # handles empty MBPP prompt field
                    task.get("entry_point", ""),
                    model,
                )
                for task in tasks
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Processing tasks for prompt_id {prompt_id}",
            ):
                result, tokens_used = future.result()
                total_tokens += tokens_used
                if result["completion"]:
                    samples.append(result)
                else:
                    print(
                        f"No valid completion for task_id: {result['task_id']} "
                        f"with prompt_id: {prompt_id}"
                    )

        samples.sort(key=lambda item: item["task_id"])
        output_file = os.path.join(output_path, f"train_set_{model_slug}_{prompt_id}.jsonl")
        write_jsonl(output_file, samples)

    print(f"Total tokens used: {total_tokens}")


# ---------------------------------------------------------------------------
# CLI (unchanged)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run prompt evaluation using Ollama.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trainset_path", type=str, required=True)
    parser.add_argument("--mutated_prompt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=30)

    args = parser.parse_args()

    main(
        args.model,
        str(resolve_pipeline_path(args.trainset_path)),
        str(resolve_pipeline_path(args.mutated_prompt_path)),
        str(resolve_pipeline_path(args.output_path)),
        args.max_workers,
    )