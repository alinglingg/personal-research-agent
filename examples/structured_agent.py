import json

from llm import ask_llm
from tools import calculate


state = {
    "goal": "What is 25 multiplied by 17?",
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

Available tool:

calculate
Arguments:
- a: number
- b: number
- operation: add, subtract, multiply, or divide

Previous observations:
{observations_text}

Tool history:
{tool_history_text}

Do not repeat a tool call if the same tool with the same arguments has already succeeded.
If an existing observation already answers the user's question, return the final answer.

Choose exactly one action.

If you need the calculator, return:
{{
    "action": "calculate",
    "a": 10,
    "b": 5,
    "operation": "multiply"
}}

If you have enough information to answer, return:
{{
    "action": "final",
    "answer": "your final answer"
}}

Return ONLY valid JSON.
"""

    response = ask_llm(prompt)

    decision = json.loads(response)

    print("Decision:", decision)

    if decision["action"] == "calculate":
        already_called = False

        for record in state["tool_history"]:
            if (
                record["tool"] == "calculate"
                and record["a"] == decision["a"]
                and record["b"] == decision["b"]
                and record["operation"] == decision["operation"]
            ):
                already_called = True
                break

        if already_called:
            state["observations"].append(
                "The requested calculator call was already completed successfully."
            )
            continue

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


print("\nFinal state:")
print(json.dumps(state, indent=2))
