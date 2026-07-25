import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="Synthenexa Results Viewer")

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "synthenexa_pipeline", "Analysis", "output")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "synthenexa_pipeline", "templates")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount the output directory to serve the images statically
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Get all PNG files in the output directory, sorted by modification time (newest first)
    files = []
    if os.path.exists(OUTPUT_DIR):
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "images": files}
    )

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    print("Starting Synthenexa Results Viewer on http://localhost:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
