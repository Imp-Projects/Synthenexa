from fastapi import FastAPI
from pydantic import BaseModel

from synthenexa_core.services.generator import generate_code
from synthenexa_core.services.evaluator import evaluate_code
from synthenexa_core.services.refiner_promt import refine_prompt

app = FastAPI()


class CodeRequest(BaseModel):
    problem: str
    prompt: str
    error: str = ""


@app.get("/")
def home():
    return {"message": "Synthenexa ML Pipeline Running 🚀"}


@app.post("/generate")
def generate(req: CodeRequest):
    return {"generated_code": generate_code(req.problem, req.prompt)}


@app.post("/evaluate")
def evaluate(req: CodeRequest):
    return {"feedback": evaluate_code(req.problem, req.prompt, req.error)}


@app.post("/refine")
def refine(req: CodeRequest):
    return {"refined_prompt": refine_prompt(req.prompt, req.error)}


@app.post("/pipeline")
def pipeline(req: CodeRequest):
    code = generate_code(req.problem, req.prompt)
    feedback = evaluate_code(req.problem, code, req.error)
    new_prompt = refine_prompt(req.prompt, feedback)

    return {
        "generated_code": code,
        "feedback": feedback,
        "refined_prompt": new_prompt
    }