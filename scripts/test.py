import requests

def test_mistral():
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b",
            "prompt": "Write a Python function to add two numbers",
            "stream": False
        }
    )

    print(res.json()["response"])   
test_mistral()