from pydantic import BaseModel

class PromptCandidate(BaseModel):
    id: str
    text: str

class EvalResult(BaseModel):
    prompt_id: str
    passed: int
    failed: int
    total: int

class ExecutionResult(BaseModel):
    prompt_id: str
    task_id: str
    output: str
