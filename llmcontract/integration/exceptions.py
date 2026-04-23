"""Exceptions raised by the integration layer."""

from __future__ import annotations

from llmcontract.monitor.monitor import MonitorResult, Violation, Blocked


class ProtocolViolationError(Exception):
    """Raised when an LLM interaction violates the session protocol."""

    def __init__(self, result: MonitorResult, phase: str) -> None:
        self.result = result
        self.phase = phase
        if isinstance(result, Violation):
            expected = ", ".join(result.expected)
            super().__init__(
                f"Protocol violation during {phase}: "
                f"got {result.got}, expected one of [{expected}]"
            )
        elif isinstance(result, Blocked):
            super().__init__(
                f"Protocol blocked during {phase}: {result.reason}"
            )
        else:
            super().__init__(f"Protocol error during {phase}")
