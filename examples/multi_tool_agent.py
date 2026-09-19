import json

from llm import ask_llm
from tools import calculate, search_notes
from memory import search_memory


state = {
    "goal": "What do I previously know about SQLite?",
    "step": 0,
    "observations": [],
    "tool_history": [],
    "final_answer": None
}

max_steps = 5


while state["step"] < max_steps:
    state["step"] = state["step"] + 1

    print("Step:", state["step"])

    observations_text = "\n".join(state["observations"])
    tool_history_text = json.dumps(state["tool_history"], indent=2)

    prompt = f"""
You are an AI agent.

Your goal:
{state["goal"]}

Available tools:

1. calculate
Use for arithmetic.

Arguments:
- a: number
- b: number
- operation: add, subtract, multiply, or divide

2. search_notes
Use when the user asks about information that may exist in their local notes.

Arguments:
- query: a short keyword or phrase to search for

3. search_memory
Use when the user asks about information that may have been saved from previous sessions.

Arguments:
- query: a short keyword or phrase to search persistent memory

Previous observations:
{observations_text}

Tool history:
{tool_history_text}

Do not repeat a successful tool call with the same arguments.

Choose exactly one action.

Calculator example:
{{
    "action": "calculate",
    "a": 10,
    "b": 5,
    "operation": "multiply"
}}

Note search example:
{{
    "action": "search_notes",
    "query": "memory"
}}

Final answer example:
{{
    "action": "final",
    "answer": "your final answer"
}}

Memory search example:
{{
    "action": "search_memory",
    "query": "SQLite"
}}


Return ONLY valid JSON.
"""

    response = ask_llm(prompt)

    decision = json.loads(response)

    print("Decision:", decision)

    if decision["action"] == "search_notes":
        results = search_notes(decision["query"])

        observation = "\n".join(results)

        state["observations"].append(observation)

        tool_record = {
            "tool": "search_notes",
            "query": decision["query"],
            "result": results
        }

        state["tool_history"].append(tool_record)

        continue

    if decision["action"] == "search_memory":
        results = search_memory(decision["query"])

        observation = "\n".join(
            [f"{row[0]}: {row[1]}" for row in results]
        )

        state["observations"].append(observation)

        tool_record = {
            "tool": "search_memory",
            "query": decision["query"],
            "result": results
        }

        state["tool_history"].append(tool_record)

        continue

    if decision["action"] == "calculate":
        result = calculate(
            decision["a"],
            decision["b"],
            decision["operation"]
        )

        observation = f"Calculator result: {result}"

        state["observations"].append(observation)

        tool_record = {
            "tool": "calculate",
            "a": decision["a"],
            "b": decision["b"],
            "operation": decision["operation"],
            "result": result
        }

        state["tool_history"].append(tool_record)

        continue

    if decision["action"] == "final":
        state["final_answer"] = decision["answer"]

        print("Final answer:")
        print(state["final_answer"])
        break
