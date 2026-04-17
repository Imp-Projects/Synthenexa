from fastapi import APIRouter
from services.pipeline import run_pipeline

router = APIRouter()


@router.post("/optimize")
def optimize(data: dict):
    problem = data.get("problem")

    result = run_pipeline(problem, iterations=3)

    return result