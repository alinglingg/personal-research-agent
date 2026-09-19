import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Unit tests must not call the network")
    monkeypatch.setattr("requests.sessions.Session.request", blocked)


@pytest.fixture
def scripted_llm(monkeypatch):
    def install(*responses):
        iterator = iter(responses)
        def respond(prompt):
            response = next(iterator)
            if isinstance(response, Exception):
                raise response
            return response if isinstance(response, str) else json.dumps(response)
        monkeypatch.setattr("langgraph_agent.ask_llm", respond)
    return install
