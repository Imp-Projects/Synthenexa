import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def call_model(model: str, prompt: str) -> str:
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

        # Raise error if HTTP request failed
        response.raise_for_status()

        data = response.json()

        # DEBUG (optional)
        print("\nFULL RESPONSE:\n", data)

        # Ollama sometimes returns:
        # { "response": "...", ... }
        # or error format

        if "response" in data:
            return data["response"]

        # fallback safety
        if "error" in data:
            return f"OLLAMA ERROR: {data['error']}"

        return "ERROR: Unexpected response format"

    except requests.exceptions.RequestException as e:
        return f"REQUEST ERROR: {str(e)}"

    except Exception as e:
        return f"GENERAL ERROR: {str(e)}"