"""Public, safe errors shared by the model, agent and API boundaries."""


class AgentError(Exception):
    code = "agent_failed"
    status_code = 500


class LLMUnavailableError(AgentError):
    code = "llm_unavailable"
    status_code = 503


class LLMTimeoutError(AgentError):
    code = "llm_timeout"
    status_code = 504


class InvalidLLMResponseError(AgentError):
    code = "invalid_llm_response"
    status_code = 502


class AgentLimitError(AgentError):
    code = "agent_limit_reached"
    status_code = 504
