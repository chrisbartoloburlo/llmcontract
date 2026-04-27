from llmcontract.monitor.monitor import Monitor, MonitorResult, Ok, Violation, Blocked
from llmcontract.integration import (
    MonitoredClient, ToolMiddleware, ToolResult,
    LLMResponse, ToolCall, ProtocolViolationError,
)

__all__ = [
    "Monitor", "MonitorResult", "Ok", "Violation", "Blocked",
    "MonitoredClient", "ToolMiddleware", "ToolResult",
    "LLMResponse", "ToolCall", "ProtocolViolationError",
]

# Lazy import for optional langfuse dependency
def __getattr__(name: str):
    if name == "LangfuseMonitor":
        from llmcontract.integration.langfuse import LangfuseMonitor
        return LangfuseMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
