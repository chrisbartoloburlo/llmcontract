"""Langfuse integration for llmcontract protocol monitoring.

Records each protocol event (send/receive) as a guardrail observation
in Langfuse, with boolean scores for pass/fail compliance tracking.

Requires: pip install llmsessioncontract[langfuse]
"""

from __future__ import annotations

from typing import Any

from llmcontract.monitor.monitor import Monitor, MonitorResult, Ok, Violation, Blocked


class LangfuseMonitor:
    """A Monitor wrapper that records protocol events to Langfuse.

    Each send()/receive() call:
    - Delegates to the underlying Monitor
    - Creates a guardrail observation in Langfuse with the action and result
    - Attaches a boolean score (protocol_compliance) — 1 for Ok, 0 for Violation/Blocked

    Usage:
        from langfuse import get_client
        from llmcontract.integration.langfuse import LangfuseMonitor

        langfuse = get_client()
        monitor = LangfuseMonitor(
            protocol="!Request.?Response.end",
            langfuse=langfuse,
        )

        monitor.send("Request")    # Ok — recorded as passing guardrail
        monitor.receive("Response") # Ok — recorded as passing guardrail
        assert monitor.is_terminal

    With an existing trace:
        with langfuse.start_as_current_observation(name="agent-run") as trace:
            monitor = LangfuseMonitor(protocol="...", langfuse=langfuse)
            monitor.send("Request")  # guardrail nested under trace
    """

    def __init__(
        self,
        protocol: str,
        langfuse: Any,
        *,
        monitor: Monitor | None = None,
    ) -> None:
        self._monitor = monitor or Monitor(protocol)
        self._langfuse = langfuse
        self._protocol = protocol
        self._step_count = 0

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    @property
    def current_state(self) -> int:
        return self._monitor.current_state

    @property
    def is_terminal(self) -> bool:
        return self._monitor.is_terminal

    @property
    def is_halted(self) -> bool:
        return self._monitor.is_halted

    def send(self, label: str) -> MonitorResult:
        """Record a send event and log to Langfuse."""
        result = self._monitor.send(label)
        self._record(f"!{label}", "send", label, result)
        return result

    def receive(self, label: str) -> MonitorResult:
        """Record a receive event and log to Langfuse."""
        result = self._monitor.receive(label)
        self._record(f"?{label}", "receive", label, result)
        return result

    def _record(
        self,
        action: str,
        direction: str,
        label: str,
        result: MonitorResult,
    ) -> None:
        self._step_count += 1
        passed = isinstance(result, Ok)

        if isinstance(result, Violation):
            output = {
                "passed": False,
                "result": "violation",
                "expected": result.expected,
                "got": result.got,
            }
        elif isinstance(result, Blocked):
            output = {
                "passed": False,
                "result": "blocked",
                "reason": result.reason,
            }
        else:
            output = {
                "passed": True,
                "result": "ok",
            }

        name = f"protocol-step-{self._step_count}"

        with self._langfuse.start_as_current_observation(
            as_type="guardrail",
            name=name,
            input={
                "action": action,
                "direction": direction,
                "label": label,
                "protocol": self._protocol,
                "state_before": self._monitor.current_state
                    if not passed else self._monitor.current_state,
            },
            metadata={
                "protocol": self._protocol,
                "step": self._step_count,
                "monitor_halted": self._monitor.is_halted,
                "monitor_terminal": self._monitor.is_terminal,
            },
        ) as guardrail:
            guardrail.update(output=output)
            guardrail.score(
                name="protocol_compliance",
                value=1 if passed else 0,
                data_type="boolean",
                comment=f"{action} — {'ok' if passed else output['result']}",
            )
