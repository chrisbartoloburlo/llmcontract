"""Pure finite-state-machine protocol definition.

This module has zero LangChain imports. ``ProtocolFSM`` is an explicit
transition table you build via ``add_transition`` calls; it has no notion
of recursion, choice, or any other DSL primitive — those compose by
hand from individual ``Transition`` edges.

State is held entirely in ``ProtocolMonitor`` (a sibling module). The FSM
itself is immutable after the developer finishes adding transitions and
marking terminal states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llmcontract.langchain.tool_ref import ToolRef


# ── Per-step contextual data ────────────────────────────────


@dataclass
class MonitorContext:
    """Transient data passed to ``guard``/``action`` callables and embedded
    in ``ViolationEvent``. Built fresh by the monitor for each transition
    attempt; do not retain references."""

    current_state: str
    """FSM state *before* the transition attempt."""

    event: str
    """Full event string, e.g. ``"send:search"``."""

    tool_ref: ToolRef
    """The ``ToolRef`` whose call triggered this event."""

    phase: str
    """Either ``"send"`` (tool call about to run) or ``"recv"`` (result returned)."""

    trace: list[str]
    """Snapshot copy of all events fired so far. Mutating this list does
    not affect the monitor's internal trace."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Per-phase context — tool ``args`` on ``send`` events, tool
    ``result`` on ``recv`` events. The library does not interpret it;
    it's threaded through to user-supplied guards and actions."""


@dataclass
class ViolationEvent:
    """Argument passed to the user's ``on_violation`` callback when a
    transition cannot fire (no rule matches, or a guard returned
    ``False``)."""

    current_state: str
    event: str
    expected: list[str]
    """Event strings that *would* have been valid from ``current_state``.
    May be empty if no transitions are defined from this state."""
    trace: list[str]
    """All events fired so far, *including* the violating one."""
    tool_ref: ToolRef
    phase: str


ViolationHandler = Callable[[ViolationEvent], None]


# ── Transitions ─────────────────────────────────────────────


_VALID_PHASES = ("send", "recv")


@dataclass
class Transition:
    """One edge in the FSM graph.

    ``event`` is computed from ``phase`` and ``tool.label``; it must not
    be assigned by the developer. ``guard`` (if set) decides whether the
    edge fires; ``action`` (if set) runs as a side effect when the edge
    commits.
    """

    source: str
    tool: ToolRef
    phase: str
    target: str
    guard: Callable[[MonitorContext], bool] | None = None
    action: Callable[[MonitorContext], None] | None = None

    def __post_init__(self) -> None:
        if self.phase not in _VALID_PHASES:
            raise ValueError(
                f"Transition.phase must be one of {_VALID_PHASES!r}; "
                f"got {self.phase!r}"
            )

    @property
    def event(self) -> str:
        """The lookup key for this transition: ``"<phase>:<tool.label>"``.
        Always recomputed; never stored — keeps correctness if internals
        of ``ToolRef`` ever shift.
        """
        return f"{self.phase}:{self.tool.label}"


# ── The FSM ─────────────────────────────────────────────────


class ProtocolFSM:
    """Pure FSM definition — initial state, transition table, terminal
    set. No reference to LangChain or to monitor state.

    ``step()`` is the workhorse: given a current state and an incoming
    event, it consults the table, runs any guard, fires any action, and
    returns ``(next_state, ok)``. Failures (no rule, or guard rejected)
    return ``(state, False)`` — the monitor decides what to do with that.
    """

    def __init__(self, initial: str) -> None:
        self.initial: str = initial
        # Indexed by (source_state, event_string) for O(1) lookup.
        self._transitions: dict[tuple[str, str], Transition] = {}
        self._terminal: set[str] = set()

    # ── Building the FSM (fluent) ────────────────────────────

    def add_transition(self, t: Transition) -> "ProtocolFSM":
        """Register a transition. Returns ``self`` so calls chain.

        Raises ``ValueError`` if a transition with the same
        ``(source, event)`` already exists — duplicates would make the
        FSM non-deterministic, which we forbid by construction.
        """
        key = (t.source, t.event)
        if key in self._transitions:
            raise ValueError(
                f"duplicate transition for state={t.source!r}, "
                f"event={t.event!r}"
            )
        self._transitions[key] = t
        return self

    def mark_terminal(self, *states: str) -> "ProtocolFSM":
        """Flag one or more states as protocol-complete. Returns ``self``."""
        self._terminal.update(states)
        return self

    # ── Querying the FSM ─────────────────────────────────────

    def valid_events(self, state: str) -> list[str]:
        """All event strings with a registered transition out of ``state``.
        Returns ``[]`` when ``state`` is unknown or has no outgoing edges.
        """
        return [event for (src, event) in self._transitions if src == state]

    def step(
        self,
        state: str,
        event: str,
        ctx: MonitorContext,
    ) -> tuple[str, bool]:
        """Try to fire transition ``(state, event)``.

        On success: run the action (if any), return ``(target, True)``.
        On failure: return ``(state, False)`` *without* mutating anything.
        Calling ``on_violation`` is the monitor's responsibility, not the
        FSM's — keeping the FSM pure makes it trivially unit-testable.
        """
        transition = self._transitions.get((state, event))
        if transition is None:
            return state, False
        if transition.guard is not None and not transition.guard(ctx):
            return state, False
        if transition.action is not None:
            transition.action(ctx)
        return transition.target, True

    def is_terminal(self, state: str) -> bool:
        return state in self._terminal
