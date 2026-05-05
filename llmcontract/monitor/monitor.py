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

    @property
    def current_state(self) -> int:
        return self._current_state

    @property
    def is_terminal(self) -> bool:
        return self._automaton.is_terminal(self._current_state)

    @property
    def is_halted(self) -> bool:
        return self._halted

    def send(self, label: str) -> MonitorResult:
        """Record a send event."""
        return self._step("send", label)

    def receive(self, label: str) -> MonitorResult:
        """Record a receive event."""
        return self._step("receive", label)

    def _step(self, direction: str, label: str) -> MonitorResult:
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

        got = f"{'!' if direction == 'send' else '?'}{label}"
        self._halted = True
        return Violation(expected=expected, got=got)
