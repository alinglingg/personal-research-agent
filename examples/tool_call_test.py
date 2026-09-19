import json

from llm import ask_llm
from tools import calculate


user_request = "What is 25 multiplied by 17?"

prompt = f"""
You are an AI assistant that can choose tools.

Available tool:

calculate
Arguments:
- a: number
- b: number
- operation: add, subtract, multiply, or divide

User request:
{user_request}

Return ONLY valid JSON.

Example:
{{
    "action": "calculate",
    "a": 10,
    "b": 5,
    "operation": "multiply"
}}
"""

response = ask_llm(prompt)

print("Raw model response:")
print(response)

tool_call = json.loads(response)

print("Parsed tool call:")
print(tool_call)

print("Action:", tool_call["action"])
print("A:", tool_call["a"])
print("B:", tool_call["b"])
print("Operation:", tool_call["operation"])

if tool_call["action"] == "calculate":
    result = calculate(
        tool_call["a"],
        tool_call["b"],
        tool_call["operation"]
    )

    print("Tool result:", result)

final_prompt = f"""
The user asked:

{user_request}

You chose the calculator tool.

The calculator returned:

{result}

Answer the user's original question clearly and concisely.
"""

final_answer = ask_llm(final_prompt)

print("Final answer:")
print(final_answer)
