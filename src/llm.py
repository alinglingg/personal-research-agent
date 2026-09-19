import requests

from errors import InvalidLLMResponseError, LLMTimeoutError, LLMUnavailableError

url = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = (5, 120)  # connect and read timeouts, in seconds


def ask_llm(prompt):
    data = {"model": "qwen3:8b", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.Timeout as error:
        raise LLMTimeoutError("Ollama timed out. Please try again.") from error
    except requests.RequestException as error:
        raise LLMUnavailableError("Ollama is unavailable. Please try again later.") from error

    try:
        result = response.json()
    except ValueError as error:
        raise InvalidLLMResponseError("Ollama returned invalid JSON.") from error
    if not isinstance(result, dict) or not isinstance(result.get("response"), str) or not result["response"].strip():
        raise InvalidLLMResponseError("Ollama returned an invalid response.")
    return result["response"]
