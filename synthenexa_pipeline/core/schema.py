from pydantic import BaseModel, Field


class PromptCandidate(BaseModel):
    id: str
    text: str
    is_base: bool = False
    # Lineage tracking
    origin_model: str = ""
    generation: int = 0
    mutation_idx: int = 0
    parent_id: str = ""
    mutation_role: str = ""
    prompt_hash: str = ""


class EvalResult(BaseModel):
    prompt_id: str
    passed: int
    failed: int
    total: int
    evaluator_model: str = ""
    is_base: bool = False
    dataset_split: str = "all"
    task_ids: list[str] = Field(default_factory=list)
    syntax_errors: int = 0
    errors: list[str] = Field(default_factory=list)
    # Execution timing (milliseconds)
    total_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0


class ExecutionResult(BaseModel):
    prompt_id: str
    task_id: str
    output: str
