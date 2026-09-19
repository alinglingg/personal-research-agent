"""One JSON object per event; no full prompts, evidence or model responses are logged."""
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

logger = logging.getLogger("personal_research_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event, **fields):
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))


# Context-local state keeps concurrent requests' measurements separate, including
# LangGraph's context-propagating worker threads.
active_run = ContextVar("agent_timing_run", default=None)


@contextmanager
def measure_phase(phase, *, tool=None):
    started = perf_counter()
    error = None
    measurement = {"phase": phase}
    try:
        yield measurement
    except Exception as failure:
        error = getattr(failure, "code", "agent_failed")
        raise
    finally:
        phase = measurement["phase"]
        elapsed = (perf_counter() - started) * 1000
        current = active_run.get()
        if current is not None:
            emit, metrics = current
            metrics[f"{phase}_ms"] += elapsed
            if phase.endswith("llm"):
                metrics["llm_calls"] += 1
            emit("phase_finished", phase=phase, tool=tool,
                 duration_ms=round(elapsed, 2),
                 status="failed" if error else "completed", error=error)
