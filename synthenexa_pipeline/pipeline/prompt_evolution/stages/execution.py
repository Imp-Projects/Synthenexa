"""
pipeline/prompt_evolution/stages/execution.py — Generate code solutions via LLM.

For each candidate prompt and each task in context.tasks:
  - Builds a structured task instruction using DatasetService.build_task_prompt()
  - Calls the LLM with the candidate prompt as the system persona
  - Stores an ExecutionResult per (prompt_id, task_id) pair
"""
from core.context import PipelineContext
from core.schema import ExecutionResult
from pipeline.prompt_evolution.stages.base import Stage
from services.dataset_service import DatasetService
from services.llm_service import LLMService


class ExecutionStage(Stage):
    """Runs code generation for every (candidate_prompt × task) pair."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def run(self, context: PipelineContext) -> PipelineContext:
        total_calls = len(context.candidate_prompts) * len(context.tasks)
        print(
            f"[ExecutionStage] Generating code for "
            f"{len(context.candidate_prompts)} prompts × {len(context.tasks)} tasks "
            f"= {total_calls} LLM calls..."
        )
        context.generated_codes = {}

        for candidate in context.candidate_prompts:
            results = []

            for task in context.tasks:
                task_id     = task.get("task_id", "unknown")
                task_prompt = DatasetService.build_task_prompt(task)

                try:
                    code_output = self.llm_service.generate(
                        prompt=task_prompt,
                        system=candidate.text,
                        temperature=0.0,   # deterministic for code generation
                    )
                except Exception as exc:
                    print(
                        f"[ExecutionStage] LLM failed for "
                        f"prompt={candidate.id} task={task_id}: {exc}"
                    )
                    code_output = ""

                # Fallback: stub function so evaluation can still run
                if not code_output.strip():
                    entry = task.get("entry_point", "solve")
                    code_output = f"def {entry}(*args, **kwargs):\n    pass  # generation failed"

                result = ExecutionResult(
                    prompt_id=candidate.id,
                    task_id=task_id,
                    output=code_output,
                )
                results.append(result)

            context.generated_codes[candidate.id] = results
            print(f"  ✓ {candidate.id}: {len(results)} code outputs generated.")

        return context
