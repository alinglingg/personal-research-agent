import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from langgraph_agent import run_langgraph_agent


def test_agent_calculator_route():
    result = run_langgraph_agent(
        "What is 25 multiplied by 17?"
    )

    assert result["final_answer"] is not None
    assert "425" in result["final_answer"]

    assert len(result["tool_history"]) > 0
    assert result["tool_history"][0]["tool"] == "calculate"

    assert result["next_action"] == "final"