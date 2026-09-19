from unittest.mock import Mock

import pytest

import langgraph_agent as agent


@pytest.mark.parametrize("goal,query", [
    ("Search previously saved memory for zirconium. If nothing matches, reply no.", "zirconium"),
    ("What backup color did I previously save in memory?", "backup color"),
    ("What did I previously remember about SQLite?", "SQLite"),
    ("What was previously known about SQLite?", "SQLite"),
    ("What did we learn in our earlier session about SQLite?", "SQLite"),
    ("What did I save in memory about SQLite?", "SQLite"),
    ("Search my memory for SQLite", "SQLite"),
])
def test_final_cannot_skip_memory(monkeypatch, scripted_llm, goal, query):
    search = Mock(return_value=[])
    monkeypatch.setattr(agent, "search_memory", search)
    scripted_llm({"action": "final", "answer": "Premature answer"},
                 {"action": "final", "answer": "No matching evidence found."})
    result = agent.run_langgraph_agent(goal)
    search.assert_called_once_with(query)
    assert result["tool_history"][0]["tool"] == "search_memory"
    assert result["final_answer"] == "No matching evidence found."
    assert result["step"] == 2


@pytest.mark.parametrize("goal", [
    "What do my notes say about agent memory?", "What is 25 multiplied by 17?",
    "How does SQLite save information?", "Explain memory architecture.",
])
def test_guard_does_not_capture_other_goals(goal):
    assert agent.required_memory_query({"goal": goal, "tool_history": []}) is None


def test_guard_accepts_prior_empty_attempt():
    state = {"goal": "Search previously saved memory for zirconium",
             "tool_history": [{"tool": "search_memory", "query": "zirconium", "result": []}]}
    assert agent.required_memory_query(state) is None


def test_guard_preserves_model_memory_query(monkeypatch, scripted_llm):
    search = Mock(return_value=[(1, "My backup color is amber.")])
    monkeypatch.setattr(agent, "search_memory", search)
    scripted_llm({"action": "search_memory", "query": "backup color"},
                 {"action": "final", "answer": "Your backup color is amber. (memory.db)"})
    result = agent.run_langgraph_agent("What backup color did I previously save in memory?")
    search.assert_called_once_with("backup color")
    assert result["evidence"] == [{"source": "memory.db", "content": "My backup color is amber."}]


def test_guard_precedes_unrelated_duplicate_fallback(scripted_llm):
    state = {"goal": "Search previously saved memory for zirconium", "step": 1,
             "tool_history": [{"tool": "search_notes", "query": "zirconium", "result": []}],
             "evidence": [], "sources": []}
    scripted_llm({"action": "search_notes", "query": "zirconium"})
    update = agent.decide(state)
    assert update["next_action"] == "search_memory"
    assert update["tool_query"] == "zirconium"
