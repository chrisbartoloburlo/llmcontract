"""Runtime monitor for session type protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from llmcontract.dsl.parser import parse
from llmcontract.monitor.automaton import Automaton, compile_ast


# ── Result types ─────────────────────────────────────────────

@dataclass(frozen=True)
class Ok:
    """The event was accepted."""
    pass


@dataclass(frozen=True)
class Violation:
    """Protocol violation: expected one thing, got another."""
    expected: list[str]
    got: str


@dataclass(frozen=True)
class Blocked:
    """The monitor is halted and no further events are accepted."""
    reason: str


@dataclass(frozen=True)
class Unrecognized:
    """The projection couldn't classify the event into a known label.

    Distinct from `Violation`: a violation means the agent did the wrong
    thing, an `Unrecognized` means the projection layer (typically over
    natural language) couldn't decide which label to emit. Outer-loop code
    is expected to react by asking the underlying agent to clarify with the
    user — not by halting as if the protocol had been broken.

    The monitor's state is NOT advanced when it returns `Unrecognized` and
    the monitor is NOT halted, so a follow-up event after clarification
    can be fed normally.

    A protocol can opt out of this behavior by including a literal
    `Unrecognized` transition at any state — in that case the monitor
    follows the transition and returns `Ok`, treating clarification as a
    first-class branch of the protocol.
    """
    expected: list[str]
    direction: str


MonitorResult = Union[Ok, Violation, Blocked, Unrecognized]


# Sentinel label that triggers Unrecognized handling. Use this constant
# rather than a bare string so callers don't typo their way around the
# special case.
UNRECOGNIZED = "Unrecognized"


# ── Monitor ──────────────────────────────────────────────────

class Monitor:
    """Runtime monitor that checks a stream of send/receive events against a protocol."""

    def __init__(self, protocol: str) -> None:
        ast = parse(protocol)
        self._automaton: Automaton = compile_ast(ast)
        self._current_state: int = self._automaton.initial_state
        self._halted: bool = False
        # The trace records the *attempted* event for every send/receive
        # call, including Unrecognized and violations and Blocked
        # attempts. This makes the trace a faithful audit log:
        # serializing it is sufficient to reconstruct the monitor's
        # state by replay (see ``to_dict`` / ``from_dict``).
        self._trace: list[str] = []

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        return self._automaton.is_terminal(self._current_state)

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def trace(self) -> list[str]:
        """Snapshot copy of every event recorded since construction or
        last ``reset()``. Format mirrors the DSL: ``"!Label"`` for send,
        ``"?Label"`` for receive, ``"?Unrecognized"`` for the sentinel.
        Mutating the returned list does not affect the monitor.
        """
        return list(self._trace)

    def reset(self) -> None:
        """Restore initial state and clear trace. Useful when reusing
        one ``Monitor`` instance across multiple protocol sessions —
        the cleaner pattern is one fresh ``Monitor`` per session, but
        ``reset`` is provided for callers who pool resources.
        """
        self._current_state = self._automaton.initial_state
        self._halted = False
        self._trace.clear()

    # ── Persistence ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the monitor's runtime state to a JSON-friendly
        dict. Pair with ``from_dict(protocol, state)`` to persist state
        across process restarts.

        The protocol string is *not* included — callers pass the same
        string to ``from_dict``. ``from_dict`` rebuilds state by
        replaying the trace, which makes the round-trip robust against
        internal automaton-numbering changes between releases: if the
        protocol's semantics still accept the trace, restoration
        succeeds; if they don't, restoration produces a halted monitor
        whose trace records the exact event that no longer fits.
        """
        return {"trace": list(self._trace)}

    @classmethod
    def from_dict(cls, protocol: str, state: dict) -> "Monitor":
        """Restore a monitor from a snapshot produced by ``to_dict``.

        Replays the saved trace through a fresh monitor; the replay
        advances ``current_state`` and sets ``is_halted`` exactly as
        the original ran. Side effects of the original session — model
        calls, tool calls, anything outside the monitor — are *not*
        re-executed; ``from_dict`` only restores the monitor's own
        bookkeeping.
        """
        m = cls(protocol)
        for event in state.get("trace", []):
            direction, label = _parse_event(event)
            m._step(direction, label)
        return m

    # ── Stepping ───────────────────────────────────────────

    def send(self, label: str) -> MonitorResult:
        """Record a send event."""
        return self._step("send", label)

    def receive(self, label: str) -> MonitorResult:
        """Record a receive event."""
        return self._step("receive", label)

    def _step(self, direction: str, label: str) -> MonitorResult:
        event = f"{'!' if direction == 'send' else '?'}{label}"
        # Record before any branching: trace captures every attempt
        # regardless of outcome.
        self._trace.append(event)

        if self._halted:
            return Blocked("monitor halted after a previous violation")

        transitions = self._automaton.transitions.get(self._current_state, {})
        key = (direction, label)

        if key in transitions:
            self._current_state = transitions[key]
            return Ok()

        expected = [f"{'!' if d == 'send' else '?'}{l}" for d, l in transitions]

        # Soft fail-open path for projection-induced uncertainty: if the
        # event's label is the UNRECOGNIZED sentinel and the protocol does
        # not declare a transition for it at this state, return Unrecognized
        # without halting and without advancing state — the outer loop is
        # expected to drive a clarification turn and re-feed the result.
        if label == UNRECOGNIZED:
            return Unrecognized(expected=expected, direction=direction)

        self._halted = True
        return Violation(expected=expected, got=event)


def _parse_event(event: str) -> tuple[str, str]:
    """Reverse of ``f"{'!' if direction == 'send' else '?'}{label}"``.
    Used by ``Monitor.from_dict`` to replay a serialized trace.
    """
    if not event or event[0] not in ("!", "?"):
        raise ValueError(
            f"trace entry {event!r} is not in the expected '!Label' / "
            f"'?Label' format produced by Monitor.to_dict()"
        )
    direction = "send" if event[0] == "!" else "receive"
    return direction, event[1:]
