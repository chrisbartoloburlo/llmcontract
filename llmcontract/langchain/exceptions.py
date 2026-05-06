"""Exception type for protocol violations.

The library never raises this itself — it is provided as a convenience for
``on_violation`` callbacks that want to halt execution by raising. The
runtime decision of *what* to do on a violation belongs to the user, not
to the library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmcontract.langchain.fsm import ViolationEvent


class ProtocolViolationError(RuntimeError):
    """Convenience exception for use in ``on_violation`` callbacks.

    The library never raises this itself. Construct and raise it inside
    your handler if you want a violation to abort the agent invocation.
    """

    def __init__(self, message: str, violation: "ViolationEvent") -> None:
        super().__init__(message)
        self.violation = violation
