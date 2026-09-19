from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langgraph_agent import run_langgraph_agent
from errors import AgentError


app = FastAPI(
    title="Personal Research Agent API",
    description="API for a local LangGraph-powered research agent.",
    version="1.0.0"
)


@app.exception_handler(AgentError)
def agent_error_handler(request: Request, error: AgentError):
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": {"code": error.code, "message": str(error)}},
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
