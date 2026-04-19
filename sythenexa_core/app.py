from fastapi import FastAPI
from routes.optimize import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def home():
    return {"message": "Prochemy ML Service Running 🚀"}