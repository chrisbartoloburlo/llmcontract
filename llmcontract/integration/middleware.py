"""Tool middleware that monitors tool execution against a protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llmcontract.monitor.monitor import Monitor, Ok
from llmcontract.integration.types import ToolCall, LLMResponse
from llmcontract.integration.exceptions import ProtocolViolationError

ToolFunction = Callable[..., Any]


@dataclass
class ToolResult:
    """Result of a monitored tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any


class ToolMiddleware:
    """Intercepts LLM tool calls and checks them against the protocol.

    Parameters
    ----------
    monitor:
        The same monitor instance used by MonitoredClient.
    tools:
        Registry mapping tool name → implementation function.
    receive_label:
        Label for ?Receive when the LLM requests a tool call.
        If None, uses the tool name itself.
        If a string, uses that string for all tools.
        If callable, receives the ToolCall and returns a label.
    send_label:
        Label for !Send when returning tool results to the LLM.
        If None, uses the tool name itself.
        If a string, uses that string for all tools.
        If callable, receives (tool_name, tool_result) and returns a label.
    """

    def __init__(
        self,
        monitor: Monitor,
        tools: dict[str, ToolFunction] | None = None,
        receive_label: str | Callable[[ToolCall], str] | None = None,
        send_label: str | Callable[[str, Any], str] | None = None,
    ) -> None:
        self._monitor = monitor
        self._tools: dict[str, ToolFunction] = dict(tools) if tools else {}
        self._receive_label = receive_label
        self._send_label = send_label

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def register(self, name: str, fn: ToolFunction) -> None:
        """Register a tool function."""
        self._tools[name] = fn

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call with protocol checks.

        1. monitor.receive(label) — the LLM chose this tool
        2. Execute the tool function
        3. monitor.send(label) — sending the result back
        """
        if tool_call.name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_call.name!r}")

        # Receive check — LLM requested this tool
        recv_label = self._resolve_receive_label(tool_call)
        result = self._monitor.receive(recv_label)
        if not isinstance(result, Ok):
            raise ProtocolViolationError(result, "receive")

        # Execute
        output = self._tools[tool_call.name](**tool_call.arguments)

        # Send check — returning result to LLM
        send_label = self._resolve_send_label(tool_call.name, output)
        result = self._monitor.send(send_label)
        if not isinstance(result, Ok):
            raise ProtocolViolationError(result, "send")

        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=output,
        )

    def process(self, response: LLMResponse) -> list[ToolResult]:
        """Process all tool calls in an LLM response."""
        return [self.execute(tc) for tc in response.tool_calls]

    def _resolve_receive_label(self, tool_call: ToolCall) -> str:
        if self._receive_label is None:
            return tool_call.name
        if callable(self._receive_label):
            return self._receive_label(tool_call)
        return self._receive_label

    def _resolve_send_label(self, tool_name: str, tool_result: Any) -> str:
        if self._send_label is None:
            return tool_name
        if callable(self._send_label):
            return self._send_label(tool_name, tool_result)
        return self._send_label
