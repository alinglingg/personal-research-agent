# Personal Research Agent

A local-first AI research agent that can search local notes, retrieve saved memories, perform calculations, and answer using evidence from its tools.

Built with Python, LangGraph, Ollama (Qwen3 8B), SQLite, and FastAPI, the project demonstrates tool routing, persistent memory, grounded answers, deterministic guardrails, observability, automated evaluation, latency optimization, and a lightweight web interface.

## What it does

The agent decides which tool to use based on the user's request:

- **Notes Search** — retrieves information from local text files
- **Memory Search** — retrieves information stored in SQLite across sessions
- **Calculator** — performs deterministic arithmetic
- **Grounded Answering** — answers using retrieved tool evidence instead of relying only on the language model

Example questions:

- “What do my notes say about agent memory?”
- “What do I previously know about SQLite?”
- “What is 25 multiplied by 17?”

The project is designed as a small but extensible foundation for internal knowledge assistants, project research tools, and business systems that need to retrieve information before answering.

---

## Demo

Start the API:

```bash
uvicorn api:app --app-dir src --reload
```

Then open the web interface:

```text
http://127.0.0.1:8000/app
```

The browser interface shows:

- final answer
- tool used
- source
- step count
- expandable technical execution details

You can also use the FastAPI Swagger interface:

```text
http://127.0.0.1:8000/docs
```

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"goal":"What is 25 multiplied by 17?"}'
```

Successful responses contain:

- `goal`
- `answer`
- `sources`
- `steps`
- `tool_history`

---

## Architecture

```text
Client / Web UI
      ↓
    FastAPI
      ↓
LangGraph decision node
      ├── search_notes ─────┐
      ├── search_memory ────┤
      └── calculate ────────┘
                ↓
      Evidence + Tool History
                ↓
         Final Answer
```

The graph keeps evidence, sources, tool history, and execution state across each run.

A deterministic routing guard recognizes explicit recall requests involving previously saved, remembered, or earlier-session information and requires a memory search before the run can finalize.

Repeated tool calls with the same arguments are prevented.

A lightweight frontend is included for interactive demos. The current version does not include embeddings or web search.

---

## Tech Stack

- Python 3.12
- LangGraph
- Ollama
- Qwen3 8B
- FastAPI
- SQLite
- Pytest
- GitHub Actions
- HTML / CSS / JavaScript frontend

---

## Validation

Current project validation:

- **137 pytest tests passing**
- **5/5 offline evaluations passing**
- **5/5 live Ollama evaluations passing**
- **9/9 latency benchmark checks passing**
- duplicate tool calls prevented
- fabricated sources rejected
- deterministic routing guardrails validated

One upstream Starlette / AnyIO deprecation warning remains in the test environment. It originates from a dependency rather than application code.

---

## Setup and Run

Run commands from the repository root so relative paths resolve correctly.

### Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install the Ollama model

```bash
ollama pull qwen3:8b
```

Start Ollama separately if it is not already running:

```bash
ollama serve
```

### Initialize local storage

```bash
mkdir -p data/notes
PYTHONPATH=src python -c 'from memory import init_db; init_db()'
```

Place `.txt` note files inside:

```text
data/notes/
```

Persistent memories are stored locally in:

```text
data/memory.db
```

### Start the API

```bash
uvicorn api:app --app-dir src --reload
```

Open the web demo:

```text
http://127.0.0.1:8000/app
```

---

## Testing

Run the full test suite:

```bash
pytest -v
```

Tests run offline by default.

Model responses are scripted during automated tests, and unexpected network access is rejected.

Search and memory tests use temporary files and temporary SQLite databases so personal local data is not modified.

Test coverage includes:

- note routing
- memory routing
- empty search results
- calculator operations
- duplicate tool-call prevention
- deterministic recall routing
- malformed model JSON
- invalid action schemas
- invalid Ollama response envelopes
- connection failures
- HTTP failures
- request timeouts
- graph recursion limits
- API error responses
- structured logging
- direct-answer latency paths

---

## Evaluations

Run offline evaluations:

```bash
python evaluations/run.py
```

Run live evaluations against the configured Ollama model:

```bash
python evaluations/run.py --live
```

The evaluation suite includes five fixed cases covering:

- calculator routing
- notes search
- memory retrieval
- empty notes search
- empty memory search

Each case checks:

- correct routing
- grounded answers
- expected sources
- evidence consistency
- duplicate tool calls
- completion behavior
- fabricated source rejection

Offline evaluations use scripted model behavior and temporary fixtures.

Live mode tests the actual Ollama model's routing and answer behavior.

The evaluator accepts supported paraphrases while rejecting contradictions, unsupported claims, wrong routing, duplicate calls, and fabricated sources.

These evaluations are regression checks for known cases and are not intended as proof that every arbitrary model response is factual.

---

## Observability

The `personal_research_agent` logger emits structured JSON events to stderr.

Events include:

- `run_started`
- `decision`
- `phase_finished`
- `run_finished`
- `request_finished`

Tracked information includes:

- `run_id`
- goal
- selected tool
- step count
- latency
- completion status
- error code
- decision LLM time
- synthesis LLM time
- tool execution time
- number of LLM calls

Example completion event:

```json
{
  "event": "run_finished",
  "run_id": "example-id",
  "goal": "1 + 2",
  "selected_tool": null,
  "step_count": 2,
  "latency_ms": 153.4,
  "status": "completed",
  "error": null
}
```

Full model prompts, model responses, evidence contents, and internal exception messages are not written to logs.

The user goal is logged and may contain sensitive information, so production deployments should apply appropriate retention and access controls.

---

## Error Handling

Ollama requests use:

- 5-second connection timeout
- 120-second read timeout

There are currently no automatic retries.

The graph uses a recursion limit of 12 steps.

The application validates and rejects:

- invalid model JSON
- missing action fields
- unsupported actions
- invalid calculator operations
- non-finite operands
- division by zero
- malformed Ollama responses

API errors use a structured response format:

```json
{
  "detail": {
    "code": "llm_unavailable",
    "message": "Ollama is unavailable. Please try again later."
  }
}
```

| HTTP Status | Code | Meaning |
| --- | --- | --- |
| 503 | `llm_unavailable` | Ollama connection or HTTP failure |
| 504 | `llm_timeout` | Ollama request timed out |
| 502 | `invalid_llm_response` | Invalid model JSON, action, or response |
| 504 | `agent_limit_reached` | Graph step limit reached |
| 500 | `agent_failed` | Unexpected agent or tool failure |

FastAPI continues to use its normal `422` response for invalid request bodies.

---

## Latency Optimizations

The project includes optimized execution paths that reduce unnecessary model calls while preserving the LangGraph architecture.

After a tool runs, the system can finalize without another LLM call when the result is deterministic and sufficiently clear.

### Calculator shortcut

For a single explicit arithmetic request:

```text
What is 25 multiplied by 17?
```

the agent still routes through the calculator tool, but once the tool returns `425`, another synthesis call is not required.

### Single-fact lookup shortcut

For a clear notes or memory lookup that returns one short declarative fact, the system can return that grounded fact directly with its source.

Requests that are ambiguous, contain multiple results, require explanation, or involve empty searches continue through the normal model synthesis path.

The mandatory recall-routing guard runs before these shortcuts.

API response fields remain unchanged.

---

## Benchmarking

Run the latency benchmark against the live Ollama model:

```bash
python evaluations/benchmark.py --repeats 3
```

The benchmark covers:

- calculator
- notes lookup
- memory lookup

Reports include:

- answer checks
- evidence checks
- model call count
- routing latency
- synthesis latency
- tool time
- total agent latency

Current benchmark validation:

- **9/9 checks passing**

Observed optimizations reduced latency substantially on common request types, with the largest improvements coming from eliminating unnecessary second model calls.

Because Ollama runs locally, latency varies based on machine load and generation speed.

---

## CI

GitHub Actions runs on every push and pull request.

The workflow:

- uses Python 3.12
- installs project requirements
- runs `pytest -v`
- runs the offline evaluation suite

CI does not require Ollama or model downloads.

Live model evaluations remain an explicit local validation step.

Workflow file:

```text
.github/workflows/tests.yml
```

---

## Project Structure

```text
personal-research-agent/
├── src/
│   ├── api.py
│   ├── langgraph_agent.py
│   ├── llm.py
│   ├── memory.py
│   ├── tools.py
│   ├── direct_answers.py
│   └── ...
├── frontend/
│   └── index.html
├── tests/
│   └── ...
├── evaluations/
│   ├── run.py
│   ├── benchmark.py
│   └── cases.json
├── data/
│   └── notes/
├── examples/
│   └── ...
├── .github/
│   └── workflows/
│       └── tests.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## What I Learned

This project was built as a hands-on introduction to agentic AI architecture.

It demonstrates:

- LLM tool selection
- LangGraph orchestration
- temporary agent state
- persistent SQLite memory
- evidence-grounded responses
- deterministic routing guardrails
- duplicate-call prevention
- API design
- structured error handling
- observability
- automated testing
- offline and live evaluation
- latency profiling and optimization
- lightweight frontend integration

The project evolved from a simple local LLM script into a tested, observable, API-accessible agent system.

---

## Business Use Cases

The same architecture can be adapted into an internal business assistant.

Possible use cases include:

- searching company SOPs
- retrieving project documentation
- looking up client notes
- answering internal policy questions
- retrieving previously recorded decisions
- performing pricing or operational calculations
- supporting customer service teams
- creating an internal knowledge assistant

Instead of answering only from the language model's built-in knowledge, the agent retrieves relevant information from connected tools before producing its response.

---

## Future Improvements

Possible next steps include:

- semantic search with embeddings
- web research tools
- conversation and session memory
- human-in-the-loop approvals
- additional business data connectors
- Docker deployment
- richer frontend UX
- authentication and multi-user sessions
- production database support
- distributed tracing
- hosted deployment