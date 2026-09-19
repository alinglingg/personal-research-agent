import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from llm import ask_llm
from tools import calculate, search_notes
from memory import search_memory


class AgentState(TypedDict):
    goal: str
    step: int
    evidence: list
    sources: list
    tool_history: list
    final_answer: str | None
    next_action: str | None
    tool_query: str | None
    number_a: float | None
    number_b: float | None
    operation: str | None

def make_final_answer(state: AgentState):
    evidence_text = json.dumps(state["evidence"], indent=2)

    prompt = f"""
Answer the user's goal using only the evidence below.

Goal:
{state["goal"]}

Evidence:
{evidence_text}

Rules:
- Do not use information that is not in the evidence.
- Cite the source names shown in the evidence when appropriate.
- Give a concise final answer.

Return ONLY valid JSON:

{{
    "answer": "your final answer"
}}
"""

    response = ask_llm(prompt)
    result = json.loads(response)

    return {
        "step": state["step"] + 1,
        "next_action": "final",
        "final_answer": result["answer"]
    }

def decide(state: AgentState):
    evidence_text = json.dumps(state["evidence"], indent=2)
    sources_text = "\n".join(state["sources"])
    tool_history_text = json.dumps(state["tool_history"], indent=2)

    prompt = f"""
You are a research agent.

Research goal:
{state["goal"]}

Available tools:

1. search_notes
Use when the user asks about information in their local notes.
Arguments:
- query

2. search_memory
Use when the user asks about something previously remembered, saved,
learned, or known from an earlier session.
Arguments:
- query

3. calculate
Use for arithmetic.
Arguments:
- a
- b
- operation: add, subtract, multiply, divide

Evidence gathered:
{evidence_text}

Sources gathered:
{sources_text}

Tool history:
{tool_history_text}

Rules:
- If relevant evidence already exists, prefer producing a final answer.
- Do not repeat a successful tool call with the same arguments.
- Use search_memory for previously remembered or saved information.
- Use search_notes for information in local notes.
- Use calculate for arithmetic.
- Base final answers only on gathered evidence.
- Never invent source filenames.

Choose exactly one action.

Search notes:
{{
    "action": "search_notes",
    "query": "agent memory"
}}

Search memory:
{{
    "action": "search_memory",
    "query": "SQLite"
}}

Calculate:
{{
    "action": "calculate",
    "a": 25,
    "b": 17,
    "operation": "multiply"
}}

Final:
{{
    "action": "final",
    "answer": "your evidence-based answer"
}}

Return ONLY valid JSON.
"""

    response = ask_llm(prompt)
    decision = json.loads(response)

    # -------------------------
    # SEARCH NOTES
    # -------------------------

    if decision["action"] == "search_notes":
        query = decision["query"]

        already_called = any(
            record["tool"] == "search_notes"
            and record.get("query", "").lower() == query.lower()
            for record in state["tool_history"]
        )

        if already_called:
            return make_final_answer(state)

        return {
            "step": state["step"] + 1,
            "next_action": "search_notes",
            "tool_query": query
        }

    # -------------------------
    # SEARCH MEMORY
    # -------------------------

    if decision["action"] == "search_memory":
        query = decision["query"]

        already_called = any(
            record["tool"] == "search_memory"
            and record.get("query", "").lower() == query.lower()
            for record in state["tool_history"]
        )

        if already_called:
            return make_final_answer(state)

        return {
            "step": state["step"] + 1,
            "next_action": "search_memory",
            "tool_query": query
        }

    # -------------------------
    # CALCULATE
    # -------------------------

    if decision["action"] == "calculate":
        a = decision["a"]
        b = decision["b"]
        operation = decision["operation"]

        already_called = any(
            record["tool"] == "calculate"
            and record.get("a") == a
            and record.get("b") == b
            and record.get("operation") == operation
            for record in state["tool_history"]
        )

        if already_called:
            return make_final_answer(state)

        return {
            "step": state["step"] + 1,
            "next_action": "calculate",
            "number_a": a,
            "number_b": b,
            "operation": operation
        }

    # -------------------------
    # FINAL
    # -------------------------

    if decision["action"] == "final":
        return {
            "step": state["step"] + 1,
            "next_action": "final",
            "final_answer": decision["answer"]
        }

    return {
        "step": state["step"] + 1,
        "next_action": "unknown"
    }


def search_notes_node(state: AgentState):
    results = search_notes(state["tool_query"])

    evidence = list(state["evidence"])
    sources = list(state["sources"])
    tool_history = list(state["tool_history"])

    for result in results:
        source, content = result.split(": ", 1)

        evidence.append({
            "source": source,
            "content": content
        })

        if source not in sources:
            sources.append(source)

    tool_history.append({
        "tool": "search_notes",
        "query": state["tool_query"],
        "result": results
    })

    return {
        "evidence": evidence,
        "sources": sources,
        "tool_history": tool_history,
        "next_action": None,
        "tool_query": None
    }


def search_memory_node(state: AgentState):
    results = search_memory(state["tool_query"])

    evidence = list(state["evidence"])
    sources = list(state["sources"])
    tool_history = list(state["tool_history"])

    for row in results:
        memory_id, content = row

        evidence.append({
            "source": "memory.db",
            "content": content
        })

    if results and "memory.db" not in sources:
        sources.append("memory.db")

    tool_history.append({
        "tool": "search_memory",
        "query": state["tool_query"],
        "result": results
    })

    return {
        "evidence": evidence,
        "sources": sources,
        "tool_history": tool_history,
        "next_action": None,
        "tool_query": None
    }


def calculate_node(state: AgentState):
    result = calculate(
        state["number_a"],
        state["number_b"],
        state["operation"]
    )

    evidence = list(state["evidence"])
    tool_history = list(state["tool_history"])

    evidence.append({
        "source": "calculator",
        "content": f"Calculator result: {result}"
    })

    tool_history.append({
        "tool": "calculate",
        "a": state["number_a"],
        "b": state["number_b"],
        "operation": state["operation"],
        "result": result
    })

    return {
        "evidence": evidence,
        "tool_history": tool_history,
        "next_action": None,
        "number_a": None,
        "number_b": None,
        "operation": None
    }


def route_after_decide(state: AgentState):
    if state["next_action"] == "search_notes":
        return "search_notes"

    if state["next_action"] == "search_memory":
        return "search_memory"

    if state["next_action"] == "calculate":
        return "calculate"

    if state["next_action"] == "final":
        return "end"

    return "end"


graph = StateGraph(AgentState)

graph.add_node("decide", decide)
graph.add_node("search_notes", search_notes_node)
graph.add_node("search_memory", search_memory_node)
graph.add_node("calculate", calculate_node)

graph.add_edge(START, "decide")

graph.add_conditional_edges(
    "decide",
    route_after_decide,
    {
        "search_notes": "search_notes",
        "search_memory": "search_memory",
        "calculate": "calculate",
        "end": END
    }
)

graph.add_edge("search_notes", "decide")
graph.add_edge("search_memory", "decide")
graph.add_edge("calculate", "decide")

app = graph.compile()

def run_langgraph_agent(goal):
    initial_state = {
        "goal": goal,
        "step": 0,
        "evidence": [],
        "sources": [],
        "tool_history": [],
        "final_answer": None,
        "next_action": None,
        "tool_query": None,
        "number_a": None,
        "number_b": None,
        "operation": None
    }

    return app.invoke(
        initial_state,
        config={"recursion_limit": 12}
    )

if __name__ == "__main__":
    result = run_langgraph_agent(
        "What does my notes say about agent memory?"
    )

    print("\nFinal graph state:")
    print(json.dumps(result, indent=2))