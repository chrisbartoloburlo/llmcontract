"""Stateful runner that drives a ``ProtocolFSM`` through one chain
execution.

One ``ProtocolMonitor`` instance per agent invocation. Holds the current
state and the trace of every event that has fired so far. Calls the
user-supplied ``on_violation`` handler whenever the FSM rejects a
transition.

The module also exposes a pure ``fire_step`` helper that
``CheckpointedProtocolMiddleware`` uses to drive the FSM through
LangGraph-checkpointed state instead of through an in-memory monitor.

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


def fire_step(
    *,
    fsm: ProtocolFSM,
    state: str,
    trace: list[str],
    label: str,
    phase: str,
    tool_ref: ToolRef | None,
    event_label: str | None,
    metadata: dict[str, Any] | None,
    on_violation: ViolationHandler,
) -> tuple[str, list[str], bool]:
    """Pure transition firing — no instance state held anywhere.

    Builds the event string from ``phase``/``label``, appends it to a
    fresh copy of ``trace``, runs the FSM step, and on violation calls
    ``on_violation`` with a populated ``ViolationEvent``.

    Returns ``(new_state, new_trace, ok)``. ``new_trace`` always
    includes the attempted event, even on violation, so the trace
    history matches what ``ProtocolMonitor`` would record.

    Used by both ``ProtocolMonitor`` (for in-process state) and the
    ``CheckpointedProtocolMiddleware`` (for state held in
    LangGraph's ``AgentState`` checkpoint).
    """
    event = f"{phase}:{label}"
    new_trace = [*trace, event]
    ctx = MonitorContext(
        current_state=state,
        event=event,
        tool_ref=tool_ref,
        phase=phase,
        trace=list(new_trace),
        event_label=event_label,
        metadata=dict(metadata) if metadata else {},
    )
    next_state, ok = fsm.step(state, event, ctx)
    if ok:
        return next_state, new_trace, True
    on_violation(
        ViolationEvent(
            current_state=state,
            event=event,
            expected=fsm.valid_events(state),
            trace=list(new_trace),
            tool_ref=tool_ref,
            phase=phase,
            event_label=event_label,
        )
    )
    return state, new_trace, False


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
        new_state, new_trace, ok = fire_step(
            fsm=self._fsm,
            state=self._state,
            trace=self._trace,
            label=label,
            phase=phase,
            tool_ref=tool_ref,
            event_label=event_label,
            metadata=metadata,
            on_violation=self._on_violation,
        )
        # On violation, fire_step returns the unchanged state but still
        # includes the attempted event in the trace — match that here.
        self._state = new_state
        self._trace[:] = new_trace
        return ok

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
