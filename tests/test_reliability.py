import json
from unittest.mock import Mock

import pytest
import requests
from fastapi.testclient import TestClient

import api
import langgraph_agent as agent
import llm
import memory
from errors import AgentError, InvalidLLMResponseError, LLMTimeoutError, LLMUnavailableError
from tools import search_notes


@pytest.mark.parametrize("tool,results,source", [
    ("search_notes", ["notes.txt: SQLite is local."], "notes.txt"),
    ("search_memory", [(1, "SQLite is local.")], "memory.db"),
    ("search_notes", [], None),
    ("search_memory", [], None),
])
def test_search_routing_and_duplicates(monkeypatch, scripted_llm, tool, results, source):
    search = Mock(return_value=results)
    monkeypatch.setattr(agent, tool, search)
    scripted_llm(
        {"action": tool, "query": "SQLite"},
        {"action": tool, "query": "sqlite"},
        {"answer": "SQLite is local." if results else "No matching evidence found."},
    )
    result = agent.run_langgraph_agent("Find SQLite")
    search.assert_called_once_with("SQLite")
    assert len(result["tool_history"]) == 1
    assert result["sources"] == ([source] if source else [])
    assert len(result["evidence"]) == len(results)
    assert result["next_action"] == "final"
    assert result["step"] == 2
    assert result["final_answer"]


@pytest.mark.parametrize("operation,a,b,expected", [
    ("add", 3, 2, 5), ("subtract", 3, 2, 1),
    ("multiply", 3, 2, 6), ("divide", 3, 2, 1.5),
])
def test_calculator_duplicates(scripted_llm, operation, a, b, expected):
    call = {"action": "calculate", "a": a, "b": b, "operation": operation}
    scripted_llm(call, call, {"answer": str(expected)})
    result = agent.run_langgraph_agent("Calculate")
    assert len(result["tool_history"]) == 1
    assert result["tool_history"][0]["result"] == expected
    assert result["final_answer"] == str(expected)


@pytest.mark.parametrize("payload", [
    "not JSON", "```json\n{}\n```", "null", "[]", "{}",
    '{"action":"unknown"}', '{"action":"search_notes"}',
    '{"action":"search_memory","query":12}',
    '{"action":"search_notes","query":" "}',
    '{"action":"calculate","a":true,"b":2,"operation":"add"}',
    '{"action":"calculate","a":1,"b":0,"operation":"divide"}',
    '{"action":"calculate","a":1,"b":2,"operation":"power"}',
    '{"action":"calculate","a":NaN,"b":2,"operation":"add"}',
    '{"action":"final","answer":null}', '{"action":"final","answer":" "}',
])
def test_invalid_decisions(scripted_llm, payload):
    scripted_llm(payload)
    with pytest.raises(InvalidLLMResponseError):
        agent.run_langgraph_agent("test")


@pytest.mark.parametrize("payload", ["oops", {}, {"answer": []}])
def test_invalid_duplicate_final_response(scripted_llm, payload):
    call = {"action": "calculate", "a": 1, "b": 2, "operation": "add"}
    scripted_llm(call, call, payload)
    with pytest.raises(InvalidLLMResponseError):
        agent.run_langgraph_agent("test")


@pytest.mark.parametrize("failure,error", [
    (requests.ConnectionError("private server info"), LLMUnavailableError),
    (requests.Timeout("private server info"), LLMTimeoutError),
    (requests.HTTPError("private server info"), LLMUnavailableError),
])
def test_network_errors(monkeypatch, failure, error):
    post = Mock(side_effect=failure)
    monkeypatch.setattr(llm.requests, "post", post)
    with pytest.raises(error, match="Ollama"):
        llm.ask_llm("test")
    assert post.call_args.kwargs["timeout"] == (5, 120)


@pytest.mark.parametrize("body", [None, [], {}, {"response": None}, {"response": " "}])
def test_invalid_ollama_envelope(monkeypatch, body):
    monkeypatch.setattr(llm.requests, "post", Mock(return_value=Mock(json=Mock(return_value=body))))
    with pytest.raises(InvalidLLMResponseError):
        llm.ask_llm("test")


def test_ollama_json_and_success(monkeypatch):
    response = Mock()
    monkeypatch.setattr(llm.requests, "post", Mock(return_value=response))
    response.json.side_effect = ValueError("private body")
    with pytest.raises(InvalidLLMResponseError):
        llm.ask_llm("test")
    response.json.side_effect = None
    response.json.return_value = {"response": '{"answer":"3"}'}
    assert llm.ask_llm("test") == '{"answer":"3"}'


@pytest.mark.parametrize("failure,status,code", [
    (LLMUnavailableError("Ollama is unavailable."), 503, "llm_unavailable"),
    (LLMTimeoutError("Ollama timed out."), 504, "llm_timeout"),
    ("bad json", 502, "invalid_llm_response"),
    (RuntimeError("secret internals"), 500, "agent_failed"),
])
def test_api_clean_errors(scripted_llm, failure, status, code):
    scripted_llm(failure)
    response = TestClient(api.app).post("/research", json={"goal": "test"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "secret" not in response.text


def test_api_success(scripted_llm):
    scripted_llm({"action": "calculate", "a": 1, "b": 2, "operation": "add"},
                 {"action": "final", "answer": "3"})
    response = TestClient(api.app).post("/research", json={"goal": "1 + 2"})
    assert response.status_code == 200
    assert response.json() == {
        "goal": "1 + 2", "answer": "3", "sources": [], "steps": 2,
        "tool_history": [{"tool": "calculate", "a": 1, "b": 2, "operation": "add", "result": 3}],
    }


def test_step_limit_is_clean(monkeypatch):
    count = iter(range(20))
    monkeypatch.setattr(agent, "ask_llm", lambda prompt: json.dumps(
        {"action": "calculate", "a": next(count), "b": 1, "operation": "add"}))
    response = TestClient(api.app).post("/research", json={"goal": "never finish"})
    assert response.status_code == 504
    assert response.json()["detail"]["code"] == "agent_limit_reached"


def test_tool_failure_is_clean(monkeypatch, scripted_llm):
    scripted_llm({"action": "search_memory", "query": "test"})
    monkeypatch.setattr(agent, "search_memory", Mock(side_effect=OSError("secret path")))
    response = TestClient(api.app).post("/research", json={"goal": "test"})
    assert response.status_code == 500
    assert "secret path" not in response.text


@pytest.mark.parametrize("failure", [False, True])
def test_structured_logs(caplog, scripted_llm, failure):
    scripted_llm({"action": "calculate", "a": 1, "b": 2, "operation": "add"},
                 "bad json" if failure else {"action": "final", "answer": "3"})
    if failure:
        with pytest.raises(AgentError):
            agent.run_langgraph_agent("logging goal")
    else:
        agent.run_langgraph_agent("logging goal")
    events = [json.loads(r.message) for r in caplog.records if r.name == "personal_research_agent"]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    assert len({e["run_id"] for e in events}) == 1
    assert all(e["goal"] == "logging goal" and e["latency_ms"] >= 0 for e in events)
    assert any(e["selected_tool"] == "calculate" for e in events)
    assert events[-1]["status"] == ("failed" if failure else "completed")
    assert events[-1]["error"] == ("invalid_llm_response" if failure else None)
    assert events[-1]["step_count"] == (1 if failure else 2)


def test_search_notes_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert search_notes("anything") == []
    notes = tmp_path / "data/notes"
    notes.mkdir(parents=True)
    (notes / "sample.txt").write_text("SQLite memory\nNothing here\nMEMORY matters", encoding="utf-8")
    (notes / "ignored.md").write_text("memory")
    assert search_notes("memory") == ["sample.txt: SQLite memory", "sample.txt: MEMORY matters"]
    assert search_notes("absent") == []


def test_search_memory_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DATABASE_PATH", str(tmp_path / "memory.db"))
    memory.init_db()
    assert memory.search_memory("SQLite") == []
    memory.save_memory("SQLite stores memory")
    memory.save_memory("Different entry")
    assert [r[1] for r in memory.search_memory("sqlite")] == ["SQLite stores memory"]
    assert memory.search_memory("' OR 1=1 --") == []


def test_http_status_failure(monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("internal URL")
    monkeypatch.setattr(llm.requests, "post", Mock(return_value=response))
    with pytest.raises(LLMUnavailableError):
        llm.ask_llm("test")
    response.json.assert_not_called()


@pytest.mark.parametrize("failure,status", [(requests.ConnectionError(), 503), (requests.Timeout(), 504)])
def test_api_network_boundary(monkeypatch, failure, status):
    monkeypatch.setattr(llm.requests, "post", Mock(side_effect=failure))
    response = TestClient(api.app).post("/research", json={"goal": "test"})
    assert response.status_code == status


def test_large_operand_is_invalid(scripted_llm):
    scripted_llm({"action": "calculate", "a": 10**400, "b": 2, "operation": "multiply"})
    with pytest.raises(InvalidLLMResponseError):
        agent.run_langgraph_agent("test")
