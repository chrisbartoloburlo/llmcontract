from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Send:
    """!label — send action."""
    label: str


@dataclass(frozen=True)
class Receive:
    """?label — receive action."""
    label: str


@dataclass(frozen=True)
class InternalChoice:
    """!{a, b, ...} — sender chooses among branches."""
    branches: dict[str, ProtocolNode]


@dataclass(frozen=True)
class ExternalChoice:
    """?{a, b, ...} — receiver chooses among branches."""
    branches: dict[str, ProtocolNode]


@dataclass(frozen=True)
class Sequence:
    """left.right — sequential composition."""
    left: ProtocolNode
    right: ProtocolNode


@dataclass(frozen=True)
class Recursion:
    """rec X. body — recursive protocol."""
    var: str
    body: ProtocolNode


@dataclass(frozen=True)
class RecVar:
    """X — recursion variable reference."""
    var: str


@dataclass(frozen=True)
class End:
    """end — terminal state."""
    pass


ProtocolNode = Union[
    Send, Receive, InternalChoice, ExternalChoice,
    Sequence, Recursion, RecVar, End,
]
