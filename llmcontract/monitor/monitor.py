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


MonitorResult = Union[Ok, Violation, Blocked]


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

        # Build a useful violation message
        expected = [f"{'!' if d == 'send' else '?'}{l}" for d, l in transitions]
        got = f"{'!' if direction == 'send' else '?'}{label}"
        self._halted = True
        return Violation(expected=expected, got=got)
