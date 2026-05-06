"""Stateful runner that drives a ``ProtocolFSM`` through one chain
execution.

One ``ProtocolMonitor`` instance per agent invocation. Holds the current
state and the trace of every event that has fired so far. Calls the
user-supplied ``on_violation`` handler whenever the FSM rejects a
transition.

This module does not import LangChain.
"""

from __future__ import annotations

from typing import Any

from llmcontract.langchain.fsm import (
    MonitorContext,
    ProtocolFSM,
    ViolationEvent,
    ViolationHandler,
)
from llmcontract.langchain.tool_ref import ToolRef


class ProtocolMonitor:
    """Owns the mutable state for one chain execution.

    Construct once per ``agent.invoke`` (or call ``reset()`` between
    invocations). The middleware calls ``transition()`` on every tool
    call — once with ``phase="send"`` before the tool runs, once with
    ``phase="recv"`` after it returns successfully.
    """

    def __init__(
        self,
        fsm: ProtocolFSM,
        on_violation: ViolationHandler,
        initial_state: str | None = None,
    ) -> None:
        self._fsm = fsm
        self._on_violation = on_violation
        self._initial = initial_state if initial_state is not None else fsm.initial
        self._state: str = self._initial
        self._trace: list[str] = []

    # ── Read-only views ─────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def trace(self) -> list[str]:
        # Snapshot copy — callers must not mutate the monitor's history.
        return list(self._trace)

    # ── Driving the FSM ─────────────────────────────────────

    def transition(
        self,
        tool_ref: ToolRef,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Attempt one FSM step for a tool-backed event.

        Returns ``True`` on success (state advanced), ``False`` on
        violation (state unchanged, ``on_violation`` invoked).
        """
        return self._step(
            label=tool_ref.label,
            phase=phase,
            tool_ref=tool_ref,
            event_label=None,
            metadata=metadata,
        )

    def transition_event(
        self,
        event_label: str,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Attempt one FSM step for a free-form (non-tool) event.

        Use this for transitions whose ``Transition`` was declared with
        ``event_label=...`` instead of a ``ToolRef`` — agent text
        replies, projected user replies, timeouts, or any other signal
        the orchestrator chooses to feed into the protocol.

        Returns ``True`` on success (state advanced), ``False`` on
        violation (state unchanged, ``on_violation`` invoked).
        """
        return self._step(
            label=event_label,
            phase=phase,
            tool_ref=None,
            event_label=event_label,
            metadata=metadata,
        )

    def _step(
        self,
        *,
        label: str,
        phase: str,
        tool_ref: ToolRef | None,
        event_label: str | None,
        metadata: dict[str, Any] | None,
    ) -> bool:
        event = f"{phase}:{label}"
        # The trace records the *attempted* event regardless of outcome,
        # so violation handlers see the full history including the
        # violating step. ``ViolationEvent.trace`` and
        # ``MonitorContext.trace`` are both snapshot copies, never
        # references to this list.
        self._trace.append(event)

        ctx = MonitorContext(
            current_state=self._state,
            event=event,
            tool_ref=tool_ref,
            phase=phase,
            trace=list(self._trace),
            event_label=event_label,
            metadata=dict(metadata) if metadata else {},
        )

        next_state, ok = self._fsm.step(self._state, event, ctx)
        if ok:
            self._state = next_state
            return True

        self._on_violation(
            ViolationEvent(
                current_state=self._state,
                event=event,
                expected=self._fsm.valid_events(self._state),
                trace=list(self._trace),
                tool_ref=tool_ref,
                phase=phase,
                event_label=event_label,
            )
        )
        return False

    def reset(self) -> None:
        """Restore the monitor to its initial state and clear the trace.

        Use this if you want to reuse one ``ProtocolMonitor`` instance
        across multiple ``agent.invoke`` calls. The cleaner pattern is
        to construct a fresh monitor per invocation, but reset is
        provided for callers who pool resources.
        """
        self._state = self._initial
        self._trace.clear()

    def is_complete(self) -> bool:
        """Whether the monitor's current state is terminal."""
        return self._fsm.is_terminal(self._state)
