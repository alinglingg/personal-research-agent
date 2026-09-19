import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tools import calculate, search_notes


def test_calculate_multiply():
    result = calculate(25, 17, "multiply")
    assert result == 425


def test_calculate_add():
    result = calculate(10, 5, "add")
    assert result == 15


def test_search_notes_memory():
    results = search_notes("memory")

    assert len(results) > 0
    assert any("memory" in result.lower() for result in results)