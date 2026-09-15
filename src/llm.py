import requests

url = "http://localhost:11434/api/generate"


def ask_llm(prompt):
    data = {
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)

        response.raise_for_status()

        result = response.json()

        return result["response"]

    except requests.RequestException as error:
        print("Something went wrong:")
        print(error)

        return None
