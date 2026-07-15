import pytest
import os
from synthenexa_pipeline.services.sandbox_eval_service import SandboxEvalService

def test_sandbox_eval_service():
    # Only run if E2B is configured
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        pytest.skip("E2B_API_KEY is not set")
    
    svc = SandboxEvalService(max_workers=1)
    
    tasks = [
        {
            "task_id": 1,
            "entry_point": "add",
            "test": "def check(add):\n    assert add(1, 2) == 3\n"
        },
        {
            "task_id": 2,
            "entry_point": "sub",
            "test": "def check(sub):\n    assert sub(3, 2) == 1\n"
        }
    ]
    
    generated_codes = [
        "def add(a, b): return a + b",
        "def sub(a, b): return a - b"
    ]
    
    result = svc.evaluate(generated_codes, tasks)
    
    assert result["total"] == 2
    assert result["passed"] == 2
    assert result["failed"] == 0
