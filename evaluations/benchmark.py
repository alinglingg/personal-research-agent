"""Serial live latency benchmark using the same isolated evaluation fixtures."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluations.run import CASES, agent, fixtures, report_result


def benchmark(case):
    spans = []
    calls = 0
    original_llm = agent.ask_llm

    def timed_llm(prompt):
        nonlocal calls
        calls += 1
        started = perf_counter()
        try:
            return original_llm(prompt)
        finally:
            spans.append({"phase": "routing_llm" if calls == 1 else "followup_llm",
                          "latency_ms": (perf_counter() - started) * 1000})

    def timed_tool(name, original):
        def invoke(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                spans.append({"phase": "tool", "tool": name,
                              "latency_ms": (perf_counter() - started) * 1000})
        return invoke

    with fixtures(), ExitStack() as stack:
        stack.enter_context(patch.object(agent, "ask_llm", timed_llm))
        for name in ("calculate", "search_memory", "search_notes"):
            stack.enter_context(patch.object(agent, name, timed_tool(name, getattr(agent, name))))
        started = perf_counter()
        try:
            result = agent.run_langgraph_agent(case["goal"])
            report = report_result(case, result)
        except Exception as error:
            report = {"case": case["name"], "passed": False, "error": type(error).__name__}
        elapsed = (perf_counter() - started) * 1000
    return {**report, "total_ms": elapsed, "llm_calls": calls, "spans": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    failed = False
    for repeat in range(args.repeats):
        for case in CASES[:3]:
            report = {"repeat": repeat + 1, **benchmark(case)}
            failed |= not report["passed"]
            print(json.dumps(report), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
