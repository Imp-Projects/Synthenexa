"""
pipeline/prompt_evolution/stages/mutation.py — Generate prompt variants via LLM.

Strategy:
  - Takes context.current_prompt as the seed
  - Asks the LLM (as a structured expert prompt engineer) to produce N variations
  - Each variation is stored as a PromptCandidate via context.add_candidate()
  - Falls back to the original prompt if the LLM returns nothing useful
"""
import hashlib
import re

from core.context import PipelineContext
from core.schema import PromptCandidate
from pipeline.prompt_evolution.stages.base import Stage
from services.llm_service import LLMService
from services.llm_cache import LLMResponseCache


# System persona used for all mutation calls
_MUTATION_SYSTEM = """\
You are an elite prompt engineer specialising in code-generation instruction design.
Your job is to take an existing system prompt and produce a single improved variation.

Rules:
- Keep it concise (1–4 sentences).
- Focus on clarity, specificity, and guiding the model toward correct Python code.
- Do NOT include any explanation or commentary — output ONLY the improved prompt text.
- Do NOT wrap the output in quotes or markdown.
"""

_MUTATION_ROLES = (
    "clarity and instruction structure",
    "correctness and edge-case handling",
    "output-format reliability and constraint discipline",
    "reasoning and problem decomposition",
    "concise minimal wording without losing important constraints",
)


def normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt_text(text).encode("utf-8")).hexdigest()[:16]


class MutationStage(Stage):
    """Generates N mutated prompt variants from the current best prompt."""

    def __init__(self, llm_service: LLMService, num_mutations: int = 5, cache: LLMResponseCache | None = None):
        self.llm_service   = llm_service
        self.num_mutations = num_mutations
        self.cache = cache

    def run(self, context: PipelineContext) -> PipelineContext:
        print(f"[MutationStage] Generating {self.num_mutations} prompt variants...")

        # Seed from base_prompt on the very first generation
        if context.current_prompt is None:
            context.current_prompt = context.base_prompt

        context.candidate_prompts = []
        base_text = context.current_prompt.text
        seen_hashes: set[str] = set()
        feedback = self._build_feedback(context)

        for i in range(self.num_mutations):
            role = _MUTATION_ROLES[i % len(_MUTATION_ROLES)]
            user_msg = (
                f"Here is the current system prompt used to guide an LLM to write Python code:\n\n"
                f'"""\n{base_text}\n"""\n\n'
                f"Mutation focus: {role}.\n{feedback}"
                "\nMake a meaningful improvement related to this focus; do not merely replace synonyms. "
                "Output ONLY the new prompt text."
            )
            cache_key = LLMResponseCache.key(self.llm_service.model, base_text, role, feedback, "temperature=0.9")

            mutated_text = self.cache.get(cache_key) if self.cache is not None else None
            if mutated_text is not None:
                context.call_stats["persistent_cache_hits"] = context.call_stats.get("persistent_cache_hits", 0) + 1
            else:
                if self.cache is not None:
                    context.call_stats["persistent_cache_misses"] = context.call_stats.get("persistent_cache_misses", 0) + 1
                try:
                    context.call_stats["mutation_calls"] = context.call_stats.get("mutation_calls", 0) + 1
                    mutated_text = self.llm_service.generate(
                        prompt=user_msg,
                        system=_MUTATION_SYSTEM,
                        temperature=0.9,   # higher temperature encourages creative variation
                    )
                    if self.cache is not None:
                        self.cache.put(cache_key, mutated_text)
                except Exception as exc:
                    print(f"[MutationStage] LLM call failed for variant {i}: {exc}")
                    mutated_text = ""

            # Fallback: keep the original if mutation produced nothing
            if not mutated_text.strip():
                print(f"[MutationStage] Empty response for variant {i}, using original.")
                mutated_text = base_text

            mutated_text = mutated_text.strip()
            candidate_hash = prompt_hash(mutated_text)
            if candidate_hash in seen_hashes:
                print(f"[MutationStage] Duplicate variant {i} skipped.")
                continue
            seen_hashes.add(candidate_hash)

            candidate = PromptCandidate(
                id=f"{context.current_prompt.id}_gen{context.generation}_m{i}",
                text=mutated_text,
                origin_model=self.llm_service.model,
                generation=context.generation,
                mutation_idx=i,
                parent_id=context.current_prompt.id,
                mutation_role=role,
                prompt_hash=candidate_hash,
            )
            context.add_candidate(candidate)
            # Suppress default printing here, handled by EnsembleScoringStage later in a prettier way
            # print(f"  ✓ Variant {i}: {mutated_text[:80].strip()!r}...")

        # print(f"[MutationStage] {len(context.candidate_prompts)} candidates ready.")
        return context

    @staticmethod
    def _build_feedback(context: PipelineContext) -> str:
        if context.generation <= 1 or not context.scores:
            return ""
        parent_id = context.current_prompt.id if context.current_prompt else ""
        parent_score = context.scores.get(parent_id)
        lines = [f"Parent score: {parent_score:.4f}." if parent_score is not None else ""]
        if context.ensemble_scores and parent_id in context.ensemble_scores:
            scores = context.ensemble_scores[parent_id]
            lines.append("Parent model scores: " + ", ".join(f"{m}={s:.4f}" for m, s in sorted(scores.items())) + ".")
        result = context.eval_results.get(parent_id)
        if result and result.errors:
            lines.append(f"Parent evaluation reported {len(result.errors)} failure(s); address likely edge cases.")
        return "\n".join(line for line in lines if line)
