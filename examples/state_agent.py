import json

state = {
    "goal": "What is 25 multiplied by 17?",
    "step": 0,
    "observations": [],
    "tool_history": [],
    "final_answer": None
}

state["step"] = state["step"] + 1

state["observations"].append("Calculator result: 425")

tool_record = {
    "tool": "calculate",
    "a": 25,
    "b": 17,
    "operation": "multiply",
    "result": 425
}

state["tool_history"].append(tool_record)

print(json.dumps(state, indent=2))
