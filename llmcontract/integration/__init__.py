from llmcontract.integration.client import MonitoredClient
from llmcontract.integration.middleware import ToolMiddleware, ToolResult
from llmcontract.integration.types import LLMResponse, ToolCall
from llmcontract.integration.exceptions import ProtocolViolationError

__all__ = [
    "MonitoredClient",
    "ToolMiddleware",
    "ToolResult",
    "LLMResponse",
    "ToolCall",
    "ProtocolViolationError",
]

# Lazy import for optional langfuse dependency
def __getattr__(name: str):
    if name == "LangfuseMonitor":
        from llmcontract.integration.langfuse import LangfuseMonitor
        return LangfuseMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
