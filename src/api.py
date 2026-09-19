from fastapi import FastAPI
from pydantic import BaseModel

from langgraph_agent import run_langgraph_agent


app = FastAPI(
    title="Personal Research Agent API",
    description="API for a local LangGraph-powered research agent.",
    version="1.0.0"
)


class ResearchRequest(BaseModel):
    goal: str


@app.get("/")
def home():
    return {
        "message": "Personal Research Agent API is running."
    }


@app.post("/research")
def research(request: ResearchRequest):
    result = run_langgraph_agent(request.goal)

    return {
        "goal": result["goal"],
        "answer": result["final_answer"],
        "sources": result["sources"],
        "steps": result["step"],
        "tool_history": result["tool_history"]
    }