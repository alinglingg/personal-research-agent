from llm import ask_llm

user_request = "Explain what SQLite is."

prompt = f"""
You are an AI assistant.

You have two possible actions:

1. ANSWER
Use this when you can answer directly.

2. CALCULATE
Use this when the user asks for arithmetic.

User request:
{user_request}

Respond with only one word:
ANSWER or CALCULATE
"""

decision = ask_llm(prompt)

print("Decision:", decision)
