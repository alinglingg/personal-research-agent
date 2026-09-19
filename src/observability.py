"""One JSON object per event; no full prompts, evidence or model responses are logged."""
import json
import logging

logger = logging.getLogger("personal_research_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event, **fields):
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False))
