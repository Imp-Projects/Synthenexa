"""
pipeline/prompt_evolution/stages/execution.py — Generate code solutions via LLM.

For each candidate prompt and each task in context.tasks:
  - Builds a structured task instruction using DatasetService.build_task_prompt()
  - Calls the LLM with the candidate prompt as the system persona
  - Stores an ExecutionResult per (prompt_id, task_id) pair
"""
import hashlib

from core.context import PipelineContext
from core.schema import ExecutionResult
from pipeline.prompt_evolution.stages.base import Stage
from services.dataset_service import DatasetService
from services.llm_service import LLMService
from services.llm_cache import LLMResponseCache
import time


class ExecutionStage(Stage):
    """Runs code generation for every (candidate_prompt × task) pair."""

    def __init__(self, llm_service: LLMService, cache: LLMResponseCache | None = None):
        self.llm_service = llm_service
        self.cache = cache

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
            start_time = time.perf_counter()

            for task in context.tasks:
                task_id     = task.get("task_id", "unknown")
                task_prompt = DatasetService.build_task_prompt(task)
                cache_key = self._cache_key(candidate.text, task_id, task_prompt)

                if cache_key in context.execution_cache:
                    code_output = context.execution_cache[cache_key]
                    context.call_stats["generation_cache_hits"] = context.call_stats.get("generation_cache_hits", 0) + 1
                elif self.cache is not None and (cached := self.cache.get(cache_key)) is not None:
                    code_output = cached
                    context.call_stats["generation_cache_hits"] = context.call_stats.get("generation_cache_hits", 0) + 1
                    context.call_stats["persistent_cache_hits"] = context.call_stats.get("persistent_cache_hits", 0) + 1
                else:
                    if self.cache is not None:
                        context.call_stats["persistent_cache_misses"] = context.call_stats.get("persistent_cache_misses", 0) + 1
                    try:
                        code_output = self.llm_service.generate(
                            prompt=task_prompt,
                            system=candidate.text,
                            temperature=0.0,   # deterministic for code generation
                        )
                        context.call_stats["generation_calls"] = context.call_stats.get("generation_calls", 0) + 1
                    except Exception as exc:
                        context.call_stats["generation_failures"] = context.call_stats.get("generation_failures", 0) + 1
                        print(
                            f"[ExecutionStage] LLM failed for "
                            f"prompt={candidate.id} task={task_id}: {exc}"
                        )
                        code_output = ""

                # Fallback: stub function so evaluation can still run
                if not code_output.strip():
                    entry = task.get("entry_point", "solve")
                    code_output = f"def {entry}(*args, **kwargs):\n    pass  # generation failed"

                context.execution_cache[cache_key] = code_output
                if self.cache is not None:
                    self.cache.put(cache_key, code_output)

                result = ExecutionResult(
                    prompt_id=candidate.id,
                    task_id=task_id,
                    output=code_output,
                )
                results.append(result)

            context.generated_codes[candidate.id] = results
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"  ✓ {candidate.id}: {len(results)} code outputs generated. | total={elapsed_ms:.1f}ms")

        return context

    def _cache_key(self, candidate_text: str, task_id: str, task_prompt: str) -> str:
        material = "\n".join((self.llm_service.model, candidate_text, task_id, task_prompt, "temperature=0.0"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
