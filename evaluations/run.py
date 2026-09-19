"""Fixed, isolated graph evaluations: offline replay by default, --live for Ollama."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import langgraph_agent as agent
import memory

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())


def normalize(text):
    return " ".join(text.lower().split())


def answer_matches(case, answer):
    """Bounded semantic grammar for these fixtures, not a general LLM judge.

    Full matching prevents a correct keyword from hiding contradictory or extra
    claims. Citations must be present on positive retrieval answers.
    """
    text = normalize(answer)
    name = case["name"]
    if name == "calculator":
        pattern = r"(?:the (?:answer|result) is )?425[.!]?"
    elif name in ("notes", "memory"):
        source = re.escape(case["sources"][0])
        citation = rf"(?:\({source}\)|\[{source}\])"
        fact = (r"(?:the |your |my )?launch code is cedar-42" if name == "notes"
                else r"(?:my |your |the )?(?:saved |remembered )?backup colou?r is amber")
        pattern = rf"{fact}[.!]?\s*{citation}[.!]?"
    else:
        noun = r"(?:evidence|results|notes|memories|information|matches)"
        subject = r"(?: (?:about|for|on) zirconium)?"
        pattern = (
            rf"(?:no (?:matching |relevant )?{noun}{subject}(?: (?:was |were )?found)?"
            rf"|(?:i )?(?:found no|could not find any|couldn't find any) "
            rf"(?:matching |relevant )?{noun}{subject})[.!]?"
        )
    return re.fullmatch(pattern, text) is not None


def check_result(case, result):
    history = result.get("tool_history", [])
    signatures = []
    for record in history:
        arguments = {k: v for k, v in record.items() if k != "result"}
        if isinstance(arguments.get("query"), str):
            arguments["query"] = arguments["query"].lower()
        signatures.append(json.dumps(arguments, sort_keys=True))
    answer = result.get("final_answer") or ""
    cited_files = set(re.findall(r"\b[\w-]+\.(?:txt|db|md|pdf)\b", answer, re.IGNORECASE))
    source = "calculator" if case["name"] == "calculator" else next(iter(case["sources"]), None)
    expected_evidence = [{"source": source, "content": content} for content in case["evidence"]]
    # Verify provenance against actual tool history, not just answer keywords.
    retrieved = []
    for record in history:
        if record["tool"] == "search_memory":
            retrieved.extend({"source": "memory.db", "content": row[1]} for row in record.get("result", []))
        elif record["tool"] == "search_notes":
            for row in record.get("result", []):
                filename, content = row.split(": ", 1)
                retrieved.append({"source": filename, "content": content})
        elif record["tool"] == "calculate" and "result" in record:
            retrieved.append({"source": "calculator", "content": f"Calculator result: {record['result']}"})
    evidence_valid = result.get("evidence", []) == expected_evidence == retrieved
    # An empty answer must follow a real search for the requested subject.
    if not expected_evidence:
        evidence_valid = evidence_valid and any(
            r["tool"] == case["call"]["action"]
            and normalize(r.get("query", "")) == normalize(case["call"]["query"])
            and r.get("result") == [] for r in history
        )
    return {
        "routing": bool(history) and all(r["tool"] == case["call"]["action"] for r in history),
        "grounded_answer": answer_matches(case, answer) and evidence_valid,
        "no_fabricated_sources": set(result.get("sources", [])) == set(case["sources"])
            and cited_files <= set(case["sources"])
            and not re.search(r"https?://", answer, re.IGNORECASE),
        "no_duplicate_calls": len(signatures) == len(set(signatures)),
        "completion": result.get("next_action") == "final" and bool(answer.strip())
            and 0 < result.get("step", 0) <= 6,
    }


def report_result(case, result):
    checks = check_result(case, result)
    explanations = {
        "routing": "Expected the requested tool to be called, with no unrelated tools.",
        "grounded_answer": "Answer must express the expected fact or no-results statement, with matching evidence and tool results.",
        "no_fabricated_sources": "Source metadata or answer citations do not match the fixture.",
        "no_duplicate_calls": "Repeated tool call with the same arguments.",
        "completion": "Missing final answer or completion within the step budget.",
    }
    return {
        "case": case["name"], "selected_tools": [r["tool"] for r in result.get("tool_history", [])],
        "tool_history": result.get("tool_history", []), "evidence": result.get("evidence", []),
        "final_answer": result.get("final_answer"), "checks": checks,
        "passed": all(checks.values()),
        "reasons": [explanations[key] for key, passed in checks.items() if not passed] or ["All checks passed."],
    }


@contextmanager
def fixtures():
    """Exercise real SQLite and note tools without touching personal data."""
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        notes = root / "data/notes"
        notes.mkdir(parents=True)
        (notes / "project.txt").write_text("The launch code is cedar-42.\n", encoding="utf-8")
        try:
            os.chdir(root)
            with patch.object(memory, "DATABASE_PATH", str(root / "data/memory.db")):
                memory.init_db()
                memory.save_memory("My backup color is amber.")
                yield
        finally:
            os.chdir(previous_cwd)


def run_case(case, live=False, *, details=False):
    with fixtures():
        if live:
            result = agent.run_langgraph_agent(case["goal"])
        else:
            # Repeating the selected action deliberately exercises the duplicate guard.
            responses = [case["call"], case["call"], {"answer": case["answer"]}]
            with patch.object(agent, "ask_llm", side_effect=[json.dumps(r) for r in responses]):
                result = agent.run_langgraph_agent(case["goal"])
    return report_result(case, result) if details else check_result(case, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the configured local Ollama model")
    args = parser.parse_args()
    failed = False
    for case in CASES:
        try:
            report = run_case(case, live=args.live, details=True)
        except Exception as error:
            report = {"case": case["name"], "passed": False, "error": type(error).__name__}
        failed |= not report["passed"]
        print(json.dumps(report), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
