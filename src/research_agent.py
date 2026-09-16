import json

from llm import ask_llm
from tools import search_notes

state = {
    "goal": "What is the difference between agent state and agent memory?",
    "step": 0,
    "evidence": [],
    "sources": [],
    "tool_history": [],
    "final_answer": None
}

max_steps = 6

while state["step"] < max_steps:
    state["step"] = state["step"] + 1

    evidence_text = json.dumps(state["evidence"], indent=2)
    sources_text = "\n".join(state["sources"])

    prompt = f"""
You are a research agent.

Research goal:
{state["goal"]}

Available tool:

search_notes
Use this to search local research notes.

Arguments:
- query: a short keyword or phrase

Evidence gathered so far:
{evidence_text}

Sources gathered so far:
{sources_text}

Research rules:
- Gather at least 2 useful pieces of evidence before answering.
- Use search_notes when more evidence is needed.
- Do not repeat the same search unnecessarily.
- Base the final answer only on the gathered evidence.
- Mention the source filename after each supported claim, using this format: [agents.txt]

Choose one action.

Search example:
{{
    "action": "search_notes",
    "query": "agent state"
}}

Final example:
{{
    "action": "final",
    "answer": "Agent state is temporary task context [agents.txt], while memory can persist across sessions [agents.txt]."

}}

Return ONLY valid JSON.
"""

    response = ask_llm(prompt)
    decision = json.loads(response)

    print("Step:", state["step"])
    print("Decision:", decision)

    if decision["action"] == "final":
        if len(state["evidence"]) < 2:
            state["evidence"].append(
                "More evidence is required before a final answer can be given."
            )
            continue

        state["final_answer"] = decision["answer"]

        print("Final answer:")
        print(state["final_answer"])

        break

    if decision["action"] == "search_notes":
        results = search_notes(decision["query"])

        for result in results:
            source, content = result.split(": ", 1)

            evidence_record = {
                "source": source,
                "content": content
            }

            state["evidence"].append(evidence_record)

            if source not in state["sources"]:
                state["sources"].append(source)

        state["tool_history"].append({
            "tool": "search_notes",
            "query": decision["query"],
            "result": results
        })

        continue