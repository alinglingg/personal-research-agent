from fastapi import FastAPI
from pydantic import BaseModel

from research_agent import run_research_agent


app = FastAPI()


class ResearchRequest(BaseModel):
    goal: str


@app.get("/")
def home():
    return {
        "message": "Personal Research Agent API is running."
    }


@app.post("/research")
def research(request: ResearchRequest):
    result = run_research_agent(request.goal)

    return {
        "goal": result["goal"],
        "answer": result["final_answer"],
        "sources": result["sources"],
        "steps": result["step"],
        "evaluation": result["evaluation"]
    }