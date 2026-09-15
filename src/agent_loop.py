import json

from llm import ask_llm
from tools import calculate


user_request = "Explain SQLite in one sentence"

messages = []

max_steps = 5
step = 0


while step < max_steps:
    step = step + 1

    print("Step:", step)

    state_text = "\n".join(messages)

    prompt = f"""
You are an AI agent.

Your goal is to answer the user's request.

Available tool:

calculate
Arguments:
- a: number
- b: number
- operation: add, subtract, multiply, or divide

Current user request:
{user_request}

Previous observations:
{state_text}

Choose exactly one action.

If you need the calculator, return JSON like:
{{
    "action": "calculate",
    "a": 10,
    "b": 5,
    "operation": "multiply"
}}

If you already have enough information to answer the user, return JSON like:
{{
    "action": "final",
    "answer": "your final answer"
}}

Return ONLY valid JSON.
"""

    response = ask_llm(prompt)

    print("Model decision:", response)

    decision = json.loads(response)

    if decision["action"] == "calculate":
        result = calculate(
            decision["a"],
            decision["b"],
            decision["operation"]
        )

        observation = f"Calculator result: {result}"

        messages.append(observation)

        print(observation)

        continue

    if decision["action"] == "final":
        print("Final answer:")
        print(decision["answer"])

        break
