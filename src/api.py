from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from langgraph_agent import run_langgraph_agent
from errors import AgentError
from observability import log_event


app = FastAPI(
    title="Personal Research Agent API",
    description="API for a local LangGraph-powered research agent.",
    version="1.0.0"
)


@app.exception_handler(AgentError)
def agent_error_handler(request: Request, error: AgentError):
    return JSONResponse(status_code=error.status_code,
                        content={"detail": {"code": error.code, "message": str(error)}})


@app.middleware("http")
async def request_latency(request: Request, call_next):
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        log_event("request_finished", method=request.method, path=request.url.path,
                  status_code=status_code, latency_ms=round((perf_counter() - started) * 1000, 2))


class ResearchRequest(BaseModel):
    goal: str


@app.get("/")
def home():
    return {
        "message": "Personal Research Agent API is running."
    }


@app.get("/app")
def frontend():
    return FileResponse("frontend/index.html")


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
