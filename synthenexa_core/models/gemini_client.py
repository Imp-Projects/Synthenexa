import os
from google import genai
from dotenv import load_dotenv
from pathlib import Path

# ----------------------------
# Debug info (optional but useful)
# ----------------------------
print("CWD:", os.getcwd())
print("FILE:", __file__)

# ----------------------------
# Load .env from project root
# ----------------------------
env_path = Path(__file__).resolve().parents[2] / ".env"
print("ENV PATH:", env_path)

load_dotenv(env_path, override=True)

# ----------------------------
# Get API key
# ----------------------------
api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI KEY LOADED:", bool(api_key))

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

# ----------------------------
# Initialize Gemini client
# ----------------------------
client = genai.Client(api_key=api_key)

# ----------------------------
# Function to call Gemini
# ----------------------------
def call_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ----------------------------
# TEST RUN
# ----------------------------
if __name__ == "__main__":
    
    result = call_gemini("Write a Python function to add two numbers")
    print("\n--- GEMINI RESPONSE ---\n")
    print(result)