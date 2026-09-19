# Personal Research Agent

A local-first research agent using Python 3.12, LangGraph, Ollama (Qwen3 8B), SQLite, and FastAPI. It searches local text notes, retrieves saved memories, and performs arithmetic. Evidence, sources, and tool history remain in graph state; repeated calls with the same arguments are prevented.

## Architecture

```text
Client → FastAPI → LangGraph decision node
                       ├── search_notes ──┐
                       ├── search_memory ┤ → decision node → final answer
                       └── calculate ────┘
```

The existing tools, SQLite storage, graph nodes, and successful API response shape are preserved. A deterministic decision guard recognizes explicit recall requests (previously saved/remembered information or earlier sessions) and requires a memory search before any other action can finalize the run. It preserves a model-selected memory query, or extracts a search subject from the request when overriding another action. A completed empty search satisfies the prerequisite; general questions about memory architecture do not trigger it. There is no frontend, embedding service, or web search.

## Setup and run

Run commands from the repository root so relative data paths resolve correctly:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3:8b
# Start Ollama separately with `ollama serve` if it is not already running.
mkdir -p data/notes
PYTHONPATH=src python -c 'from memory import init_db; init_db()'
uvicorn api:app --app-dir src
```

Place `.txt` notes in `data/notes/`. Memories are stored in `data/memory.db`.

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"goal":"What is 25 multiplied by 17?"}'
```

Successful responses contain `goal`, `answer`, `sources`, `steps`, and `tool_history`.

## Testing

```bash
pytest -v
```

Tests run offline: model responses are scripted and unexpected network requests fail. The original calculator test retains its assertions with a deterministic model fixture. Search tests use temporary notes and SQLite databases; existing tests remain in place.

Coverage includes note and memory routing, empty results, all calculator operations, duplicate calls, invalid JSON and argument shapes, invalid Ollama response envelopes, connection/HTTP failures, timeouts, graph limits, API errors, and structured logs. `httpx` supports FastAPI's test client.

## Evaluations

```bash
python evaluations/run.py          # offline graph and evaluator regression checks
python evaluations/run.py --live   # evaluate the configured Ollama model
```

Five fixed prompts in `evaluations/cases.json` exercise calculator, notes, memory, and empty searches. Both modes use real tools against temporary synthetic notes and a temporary SQLite database, leaving personal data untouched. Run the suite as a standalone process because fixture setup temporarily changes its working directory.

Each case checks routing, an evidence-grounded answer, source names, duplicate calls, and completion. Results are JSON lines on stdout and include selected tools, tool history, evidence, the final answer, and pass/fail reasons; any failed check or run returns a nonzero exit status. Offline mode deliberately repeats a scripted action to test the duplicate guard. It validates graph behavior and evaluator checks, **not model routing quality**. Live mode uses Ollama to test actual model decisions and answers.

Answer checks use a bounded, case-specific semantic grammar: positive answers must state the expected fact and cite the fixture source; empty answers must state that no matches were found. Spacing, punctuation, and supported paraphrases are accepted. Full-answer matching rejects contradictions and added claims. Evidence must match both the fixture and actual tool results; empty cases require a search for the requested subject with an empty result. Source checks compare returned source sets and reject unexpected filename citations and URLs. These small fixtures are regression checks, not a general proof that arbitrary answers are factual. Tests also inject wrong answers, invented sources, wrong routing, and duplicate calls to verify the evaluator detects failures.

## Observability

The `personal_research_agent` Python logger emits one JSON object per event to stderr:

- `run_started`: a run has begun.
- `decision`: the model selected a tool or final answer.
- `run_finished`: completion or failure, including a stable error code.

All events include `run_id`, `goal`, `selected_tool`, `step_count`, `latency_ms`, `status`, and `error`. Decision events also include `action`. Step count is the number of completed decision nodes, matching the API's existing `steps` semantics; latency is elapsed time since the run started. A failure while deciding keeps the last completed count. `selected_tool` is null at startup and for a final decision.

Example completion event:

```json
{"event":"run_finished","run_id":"example-id","goal":"1 + 2","selected_tool":null,"step_count":2,"latency_ms":153.4,"status":"completed","error":null}
```

Logs include the user goal, which may be sensitive; restrict access and retention accordingly. Full model prompts, responses, evidence, and exception messages are not logged. Applications can configure the named logger to integrate with their own handlers.

## Error handling

Ollama requests use a 5-second connection timeout and a 120-second read timeout. These bound network waits, not the total run duration. There are no automatic retries. The graph retains its recursion limit of 12 graph steps.

Invalid JSON, missing fields, unsupported actions/operations, non-finite calculator operands, and division by zero are rejected before tool execution. Errors use this response shape:

```json
{"detail":{"code":"llm_unavailable","message":"Ollama is unavailable. Please try again later."}}
```

| HTTP status | Code | Meaning |
| --- | --- | --- |
| 503 | `llm_unavailable` | Ollama connection or HTTP failure |
| 504 | `llm_timeout` | Ollama request timed out |
| 502 | `invalid_llm_response` | Invalid model JSON, response envelope, action, or answer |
| 504 | `agent_limit_reached` | Graph step limit reached |
| 500 | `agent_failed` | Unexpected agent or tool failure |

Failure responses omit internal exception details. FastAPI continues to return its normal 422 response for invalid request bodies.

## CI

`.github/workflows/tests.yml` runs on every push and pull request. It installs requirements on Python 3.12, runs `pytest -v`, and runs the offline evaluation suite. CI requires no Ollama server or model downloads. Live model evaluations are an explicit local check.

## Project structure

```text
src/             # API, graph, model client, tools, memory, errors, logging
tests/          # offline pytest suite
evaluations/    # fixed cases and offline/live runner
data/notes/     # local text notes
.github/workflows/tests.yml
```
