from copy import deepcopy

import pytest

from evaluations.run import CASES, check_result, run_case


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_fixed_evaluation(case):
    assert all(run_case(case).values())


@pytest.fixture
def golden_result():
    return {
        "tool_history": [{"tool": "calculate", "a": 25, "b": 17, "operation": "multiply", "result": 425}],
        "sources": [], "final_answer": "425", "next_action": "final", "step": 2,
        "evidence": [{"source": "calculator", "content": "Calculator result: 425"}],
    }


@pytest.mark.parametrize("mutation,check", [
    ({"final_answer": "426"}, "grounded_answer"),
    ({"final_answer": "425 according to invented.txt"}, "no_fabricated_sources"),
    ({"sources": ["invented.txt"]}, "no_fabricated_sources"),
    ({"evidence": []}, "grounded_answer"),
    ({"next_action": None}, "completion"),
    ({"final_answer": None}, "completion"),
    ({"tool_history": [{"tool": "search_notes", "query": "25"}]}, "routing"),
])
def test_evaluator_detects_regressions(golden_result, mutation, check):
    assert all(check_result(CASES[0], golden_result).values())
    golden_result.update(mutation)
    assert not check_result(CASES[0], golden_result)[check]


def test_evaluator_detects_duplicate_calls(golden_result):
    golden_result["tool_history"] *= 2
    assert not check_result(CASES[0], golden_result)["no_duplicate_calls"]


def test_evaluation_restores_cwd_and_database():
    from pathlib import Path
    import memory
    before = (Path.cwd(), memory.DATABASE_PATH)
    run_case(deepcopy(CASES[1]))
    assert (Path.cwd(), memory.DATABASE_PATH) == before


@pytest.mark.parametrize("name,answer", [
    ("memory", "My backup color is amber.(memory.db)"),
    ("memory", "Your saved backup colour is amber [memory.db]."),
    ("empty_notes", "No matching evidence found"),
    ("empty_notes", "I couldn't find any notes about zirconium."),
    ("empty_memory", "No relevant memories for zirconium were found."),
])
def test_semantic_variants_preserve_requirements(name, answer):
    case = next(c for c in CASES if c["name"] == name)
    result = run_case(case, details=True)
    state = {**result, "sources": case["sources"], "step": 2, "next_action": "final", "final_answer": answer}
    assert all(check_result(case, state).values())


@pytest.mark.parametrize("name,answer", [
    ("memory", "Your backup color is blue. (memory.db)"),
    ("memory", "Your backup color is not amber. (memory.db)"),
    ("memory", "Amber. (memory.db)"),
    ("memory", "Your backup color is amber."),
    ("memory", "Your backup color is amber. (memory.db) Your password is 1234."),
    ("memory", "Your backup color is amber or blue. (memory.db)"),
    ("empty_notes", "No matching evidence found. Zirconium is a metal."),
    ("empty_notes", "Matching evidence found."),
    ("empty_notes", "No matching evidence found about gold."),
    ("empty_memory", "No matching evidence found. You never saved anything."),
])
def test_semantic_checks_reject_unsupported_claims(name, answer):
    from evaluations.run import answer_matches
    case = next(c for c in CASES if c["name"] == name)
    assert not answer_matches(case, answer)


@pytest.mark.parametrize("mutation", [
    {"tool_history": []},
    {"tool_history": [{"tool": "search_memory", "query": "unrelated", "result": []}]},
    {"tool_history": [{"tool": "search_memory", "query": "zirconium", "result": [(1, "Found something")]}]},
])
def test_empty_answer_requires_real_relevant_empty_search(mutation):
    case = next(c for c in CASES if c["name"] == "empty_memory")
    report = run_case(case, details=True)
    state = {**report, "sources": [], "step": 2, "next_action": "final", **mutation}
    assert not check_result(case, state)["grounded_answer"]


def test_correct_answer_requires_evidence_provenance():
    case = next(c for c in CASES if c["name"] == "memory")
    report = run_case(case, details=True)
    state = {**report, "sources": case["sources"], "step": 2, "next_action": "final"}
    state["tool_history"][0]["result"] = []
    assert not check_result(case, state)["grounded_answer"]


def test_report_includes_diagnostics():
    report = run_case(CASES[0], details=True)
    assert report["selected_tools"] == ["calculate"]
    assert report["evidence"]
    assert report["final_answer"] == "425"
    assert report["passed"]
    assert report["reasons"] == ["All checks passed."]
