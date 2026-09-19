# Personal Research Agent

A local-first AI research agent built with Python, Ollama, LangGraph, SQLite, and FastAPI.

## Features

- Local LLM inference with Ollama
- LangGraph-based agent workflow
- Multiple tools:
  - local note search
  - persistent memory search
  - calculator
- SQLite-backed persistent memory
- Tool history and evidence tracking
- Duplicate tool-call guardrails
- FastAPI endpoint
- Automated tests with pytest

## Architecture

Client
↓
FastAPI
↓
LangGraph Agent
↓
Decision Node
├── search_notes
├── search_memory
└── calculate
↓
Evidence / Tool History
↓
Final Answer

## Tech Stack

- Python 3.12
- Ollama
- Qwen3 8B
- LangGraph
- FastAPI
- SQLite
- Pytest

## Project Structure

```text
personal-research-agent/
├── src/
│   ├── api.py
│   ├── langgraph_agent.py
│   ├── llm.py
│   ├── memory.py
│   └── tools.py
├── data/
│   └── notes/
├── tests/
├── examples/
├── requirements.txt
├── pytest.ini
└── README.md