import json

from llm import ask_llm
from tools import search_notes


def run_research_agent(goal):
    state = {
        "goal": goal,
        "step": 0,
        "evidence": [],
        "sources": [],
        "tool_history": [],
        "final_answer": None,
        "metrics": {
            "tool_calls": 0,
            "searches": 0
        }
    }

    max_steps = 6

    while state["step"] < max_steps:
        state["step"] += 1

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
- Mention the source filename after each supported claim.

Choose one action.

Search example:
{{
    "action": "search_notes",
    "query": "agent state"
}}

Final example:
{{
    "action": "final",
    "answer": "Agent state is temporary task context [agents.txt]."
}}

Return ONLY valid JSON.
"""

        response = ask_llm(prompt)
        decision = json.loads(response)

        print("Step:", state["step"])
        print("Decision:", decision)

        if decision["action"] == "search_notes":
            state["metrics"]["tool_calls"] += 1
            state["metrics"]["searches"] += 1

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

        if decision["action"] == "final":
            if len(state["evidence"]) < 2:
                continue

            state["final_answer"] = decision["answer"]

            print("\nFinal answer:")
            print(state["final_answer"])

            break

    print("\n--- Run Report ---")
    print("Goal:", state["goal"])
    print("Steps:", state["step"])
    print("Tool calls:", state["metrics"]["tool_calls"])
    print("Searches:", state["metrics"]["searches"])
    print("Evidence items:", len(state["evidence"]))
    print("Sources:", state["sources"])

    passed_evidence = len(state["evidence"]) >= 2
    passed_sources = len(state["sources"]) >= 1
    passed_final_answer = state["final_answer"] is not None
    passed_step_limit = state["step"] <= max_steps
    passed_citations = all(
    f"[{source}]" in state["final_answer"]
    for source in state["sources"]
)

    overall_pass = (
        passed_evidence
        and passed_sources
        and passed_final_answer
        and passed_step_limit
    )

    print("\n--- Evaluation ---")
    print("Enough evidence:", passed_evidence)
    print("Has sources:", passed_sources)
    print("Produced final answer:", passed_final_answer)
    print("Stayed within step limit:", passed_step_limit)
    print("Overall pass:", overall_pass)
    print("Citations present:", passed_citations)

    state["evaluation"] = {
        "passed_evidence": passed_evidence,
        "passed_sources": passed_sources,
        "passed_final_answer": passed_final_answer,
        "passed_step_limit": passed_step_limit,
        "passed_citations": passed_citations,
        "overall_pass": overall_pass
    }

    return state


if __name__ == "__main__":
    run_research_agent(
        "What is the difference between agent state and agent memory?"
    )