"""
pipeline/prompt_evolution/stages/mutation.py — Generate prompt variants via LLM.

Strategy:
  - Takes context.current_prompt as the seed
  - Asks the LLM (as a structured expert prompt engineer) to produce N variations
  - Each variation is stored as a PromptCandidate via context.add_candidate()
  - Falls back to the original prompt if the LLM returns nothing useful
"""
from core.context import PipelineContext
from core.schema import PromptCandidate
from pipeline.prompt_evolution.stages.base import Stage
from services.llm_service import LLMService


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


class MutationStage(Stage):
    """Generates N mutated prompt variants from the current best prompt."""

    def __init__(self, llm_service: LLMService, num_mutations: int = 5):
        self.llm_service   = llm_service
        self.num_mutations = num_mutations

    def run(self, context: PipelineContext) -> PipelineContext:
        print(f"[MutationStage] Generating {self.num_mutations} prompt variants...")

        # Seed from base_prompt on the very first generation
        if context.current_prompt is None:
            context.current_prompt = context.base_prompt

        context.candidate_prompts = []
        base_text = context.current_prompt.text

        for i in range(self.num_mutations):
            user_msg = (
                f"Here is the current system prompt used to guide an LLM to write Python code:\n\n"
                f'"""\n{base_text}\n"""\n\n'
                f"Produce one improved variation of this prompt. Output ONLY the new prompt text."
            )

            try:
                mutated_text = self.llm_service.generate(
                    prompt=user_msg,
                    system=_MUTATION_SYSTEM,
                    temperature=0.9,   # higher temp for creative variation
                )
            except Exception as exc:
                print(f"[MutationStage] LLM call failed for variant {i}: {exc}")
                mutated_text = ""

            # Fallback: keep the original if mutation produced nothing
            if not mutated_text.strip():
                print(f"[MutationStage] Empty response for variant {i}, using original.")
                mutated_text = base_text

            candidate = PromptCandidate(
                id=f"{context.current_prompt.id}_gen{context.generation}_m{i}",
                text=mutated_text.strip(),
                origin_model=self.llm_service.model,
                generation=context.generation,
                mutation_idx=i,
                parent_id=context.current_prompt.id,
            )
            context.add_candidate(candidate)
            # Suppress default printing here, handled by EnsembleScoringStage later in a prettier way
            # print(f"  ✓ Variant {i}: {mutated_text[:80].strip()!r}...")

        # print(f"[MutationStage] {len(context.candidate_prompts)} candidates ready.")
        return context
