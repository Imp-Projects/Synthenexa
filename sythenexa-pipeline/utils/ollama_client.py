import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_model(model, prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        # 🔥 DEBUG PRINT
        print("\nFULL RESPONSE:\n", data)

        return data.get("response", "")

    except Exception as e:
        return f"ERROR: {str(e)}"