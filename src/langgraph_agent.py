import json
import math
import re
from functools import partial
from time import perf_counter
from uuid import uuid4
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langgraph.errors import GraphRecursionError

from errors import AgentError, AgentLimitError, InvalidLLMResponseError
from observability import log_event, measure_phase, active_run
from direct_answers import direct_answer, calculation_request, lookup_subject
from llm import ask_llm as generate_response
from tools import calculate, search_notes
from memory import search_memory


# Reproducible agent decisions; generic example scripts keep their defaults.
ask_llm = partial(generate_response, options={"temperature": 0})


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

def parse_response(response, *, decision=False):
    """Reject malformed JSON and tool arguments before they reach graph nodes."""
    try:
        value = json.loads(response)
    except (ValueError, TypeError) as error:
        raise InvalidLLMResponseError("Ollama returned invalid JSON.") from error
    valid = isinstance(value, dict)
    action = value.get("action") if valid and decision else "final"
    if valid and action in ("search_notes", "search_memory"):
        valid = isinstance(value.get("query"), str) and bool(value["query"].strip())
    elif valid and action == "calculate":
        try:
            valid = all(
                type(value.get(key)) in (int, float) and math.isfinite(value[key])
                for key in ("a", "b")
            ) and value.get("operation") in ("add", "subtract", "multiply", "divide")
        except OverflowError:
            valid = False
        if valid and value["operation"] == "divide" and value["b"] == 0:
            valid = False
    elif valid and action == "final":
        valid = isinstance(value.get("answer"), str) and bool(value["answer"].strip())
    else:
        valid = False
    if not valid:
        raise InvalidLLMResponseError("Ollama returned an invalid action or answer.")
    return value


def required_memory_query(state):
    """Recognize explicit recall requests, not general questions about memory."""
    if any(record["tool"] == "search_memory" for record in state["tool_history"]):
        return None
    goal = re.split(r"[?.!]", state["goal"], maxsplit=1)[0].strip()
    recall = re.search(
        r"\b(?:previously|earlier|already)\s+(?:(?:was|were|been)\s+)?"
        r"(?:save[ds]?|remember(?:ed)?|known|learn(?:ed|t))\b"
        r"|\b(?:I|we|you)\s+(?:(?:had|have)\s+)?(?:save[ds]?|remember(?:ed)?|learn(?:ed|t)|knew)\b"
        r"|\b(?:last|earlier|previous)\s+(?:session|conversation)\b"
        r"|\b(?:saved|remembered)\s+(?:memory|memories|information)\b"
        r"|\b(?:search|check|look up)\s+(?:my\s+)?(?:saved\s+)?(?:memory|memories)\b",
        goal, re.IGNORECASE,
    )
    if not recall:
        return None
    # Keep explicit search subjects short enough for the existing SQLite LIKE tool.
    subject = re.search(r"\b(?:for|about)\s+(.+)$", goal, re.IGNORECASE)
    if not subject:
        subject = re.search(r"^what\s+(.+?)\s+(?:did|had|have|was|were)\b", goal, re.IGNORECASE)
    return (subject.group(1) if subject else goal).strip(" \"'")


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

    with measure_phase("synthesis_llm"):
        response = ask_llm(prompt)
        result = parse_response(response)

    return {
        "step": state["step"] + 1,
        "next_action": "final",
        "final_answer": result["answer"]
    }

def decide(state: AgentState):
    # Finalize on the existing decision node, retaining step-count semantics and
    # the mandatory memory-search prerequisite before taking any shortcut.
    if required_memory_query(state) is None:
        answer = direct_answer(state)
        if answer is not None:
            return {"step": state["step"] + 1, "next_action": "final", "final_answer": answer}

    # Tool results appear only in evidence, not again in history and sources.
    history = [{k: v for k, v in record.items() if k != "result"}
               for record in state["tool_history"]]
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
{json.dumps(state["evidence"])}

Tool history:
{json.dumps(history)}

Rules:
- If relevant evidence already exists, prefer producing a final answer.
- Do not repeat a successful tool call with the same arguments.
- Use search_memory for previously remembered or saved information.
- Use search_notes for information in local notes.
- Always use calculate for arithmetic; never answer from mental arithmetic.
- Search queries must contain only the subject, not filenames or formatting instructions.
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

    with measure_phase("decision_llm") as measurement:
        response = ask_llm(prompt)
        decision = parse_response(response, decision=True)
        if state["tool_history"] and decision["action"] == "final":
            measurement["phase"] = "synthesis_llm"
    memory_query = required_memory_query(state)
    if memory_query is not None and decision["action"] != "search_memory":
        # Enforce the prerequisite before all actions, including duplicate-call
        # fallback finalization. The existing graph still executes the tool.
        decision = {"action": "search_memory", "query": memory_query}
    elif not state["tool_history"] and (calculation := calculation_request(state["goal"])):
        # A shortened prompt must never permit an ungrounded arithmetic answer.
        # Reuse argument validation for zero division and out-of-range operands.
        decision = parse_response(json.dumps(calculation), decision=True)
    elif not state["tool_history"]:
        for tool in ("search_memory", "search_notes"):
            subject = lookup_subject(state["goal"], tool)
            if subject:
                decision = {"action": tool, "query": subject}
                break

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
    with measure_phase("tool", tool="search_notes"):
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
    with measure_phase("tool", tool="search_memory"):
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
    with measure_phase("tool", tool="calculate"):
        result = calculate(
            state["number_a"], state["number_b"], state["operation"]
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

    run_id = str(uuid4())
    started = perf_counter()
    state = dict(initial_state)
    selected_tool = None
    status = "failed"
    error_code = None

    def emit(event, **extra):
        log_event(
            event, run_id=run_id, goal=goal, selected_tool=selected_tool,
            step_count=state["step"],
            latency_ms=round((perf_counter() - started) * 1000, 2),
            **extra,
        )

    metrics = {"decision_llm_ms": 0.0, "synthesis_llm_ms": 0.0, "tool_ms": 0.0, "llm_calls": 0}
    timing_token = active_run.set((emit, metrics))
    emit("run_started", status="running", error=None)
    try:
        for update in app.stream(initial_state, config={"recursion_limit": 12}, stream_mode="updates"):
            for node, values in update.items():
                state.update(values)
                if node == "decide":
                    action = values.get("next_action")
                    selected_tool = action if action in ("search_notes", "search_memory", "calculate") else None
                    emit("decision", status="running", error=None, action=action)
        if not state["final_answer"]:
            raise AgentError("The agent did not produce an answer.")
        status = "completed"
        return state
    except GraphRecursionError as error:
        error_code = AgentLimitError.code
        raise AgentLimitError("The agent reached its step limit. Try a narrower goal.") from error
    except AgentError as error:
        error_code = error.code
        raise
    except Exception as error:
        error_code = AgentError.code
        raise AgentError("The research run failed. Please try again.") from error
    finally:
        active_run.reset(timing_token)
        emit("run_finished", status=status, error=error_code,
             timings={key: round(value, 2) for key, value in metrics.items()})


if __name__ == "__main__":
    result = run_langgraph_agent(
        "What does my notes say about agent memory?"
    )

    print("\nFinal graph state:")
    print(json.dumps(result, indent=2))