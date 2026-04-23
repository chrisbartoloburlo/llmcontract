"""Compiles a session type AST into a finite state automaton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from llmcontract.dsl.ast import (
    End, ExternalChoice, InternalChoice, ProtocolNode,
    Receive, Recursion, RecVar, Send, Sequence,
)

Direction = Literal["send", "receive"]
TransitionKey = tuple[Direction, str]


@dataclass
class Automaton:
    """Finite state automaton compiled from a session type AST."""

    transitions: dict[int, dict[TransitionKey, int]] = field(default_factory=dict)
    terminal_states: set[int] = field(default_factory=set)
    initial_state: int = 0
    _next_id: int = field(default=0, repr=False)

    def _new_state(self) -> int:
        sid = self._next_id
        self._next_id += 1
        if sid not in self.transitions:
            self.transitions[sid] = {}
        return sid

    def is_terminal(self, state: int) -> bool:
        return state in self.terminal_states


def compile_ast(node: ProtocolNode) -> Automaton:
    """Compile an AST into a finite state automaton."""
    aut = Automaton()
    start = aut._new_state()
    aut.initial_state = start
    rec_env: dict[str, int] = {}
    _compile(node, start, aut, rec_env)
    return aut


def _compile(
    node: ProtocolNode,
    current: int,
    aut: Automaton,
    rec_env: dict[str, int],
) -> None:
    """Recursively compile *node* starting from *current* state."""

    if isinstance(node, End):
        aut.terminal_states.add(current)

    elif isinstance(node, Send):
        nxt = aut._new_state()
        aut.transitions[current][("send", node.label)] = nxt
        # The next state is terminal by default; a Sequence wrapper overrides this.
        aut.terminal_states.add(nxt)

    elif isinstance(node, Receive):
        nxt = aut._new_state()
        aut.transitions[current][("receive", node.label)] = nxt
        aut.terminal_states.add(nxt)

    elif isinstance(node, InternalChoice):
        for label, branch in node.branches.items():
            nxt = aut._new_state()
            aut.transitions[current][("send", label)] = nxt
            _compile(branch, nxt, aut, rec_env)

    elif isinstance(node, ExternalChoice):
        for label, branch in node.branches.items():
            nxt = aut._new_state()
            aut.transitions[current][("receive", label)] = nxt
            _compile(branch, nxt, aut, rec_env)

    elif isinstance(node, Sequence):
        # Compile left, find its "leaf" states (non-choice terminal sinks),
        # then wire those into right.
        _compile(node.left, current, aut, rec_env)
        # The leaf states produced by left are terminal states reachable from current
        # that were just added. We need to find them and make them non-terminal,
        # then compile right from each.
        leaf_states = _collect_leaf_states(node.left, current, aut)
        for s in leaf_states:
            aut.terminal_states.discard(s)
            _compile(node.right, s, aut, rec_env)

    elif isinstance(node, Recursion):
        rec_env_copy = dict(rec_env)
        rec_env_copy[node.var] = current
        _compile(node.body, current, aut, rec_env_copy)

    elif isinstance(node, RecVar):
        # Back-edge: wire current state to the recursion point.
        # We mark current as an epsilon-transition target by copying transitions.
        target = rec_env[node.var]
        # Copy all transitions from the target to current state
        for key, dest in aut.transitions.get(target, {}).items():
            aut.transitions[current][key] = dest

    else:
        raise TypeError(f"Unknown AST node: {type(node)}")


def _collect_leaf_states(
    node: ProtocolNode,
    current: int,
    aut: Automaton,
) -> list[int]:
    """Return the states that a compiled node ends in (its continuation points)."""

    if isinstance(node, End):
        return [current]

    elif isinstance(node, (Send, Receive)):
        # The single transition target
        direction = "send" if isinstance(node, Send) else "receive"
        label = node.label
        nxt = aut.transitions[current].get((direction, label))
        if nxt is not None:
            return [nxt]
        return []

    elif isinstance(node, (InternalChoice, ExternalChoice)):
        direction = "send" if isinstance(node, InternalChoice) else "receive"
        leaves: list[int] = []
        for label, branch in node.branches.items():
            nxt = aut.transitions[current].get((direction, label))
            if nxt is not None:
                leaves.extend(_collect_leaf_states(branch, nxt, aut))
        return leaves

    elif isinstance(node, Sequence):
        left_leaves = _collect_leaf_states(node.left, current, aut)
        all_leaves: list[int] = []
        for s in left_leaves:
            all_leaves.extend(_collect_leaf_states(node.right, s, aut))
        return all_leaves

    elif isinstance(node, Recursion):
        return _collect_leaf_states(node.body, current, aut)

    elif isinstance(node, RecVar):
        return [current]

    return [current]
