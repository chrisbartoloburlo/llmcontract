"""Stable, hashable references to LangChain tools.

A ``ToolRef`` wraps a ``BaseTool`` instance, a ``@tool``-decorated callable,
or any plain callable, and exposes a single read-only string label derived
once at construction. Two ``ToolRef`` objects are equal (and share a hash)
iff their labels match — which lets developers refer to the same tool from
multiple FSM transitions without juggling identity.

Crucially, this module does **not** import LangChain. The label resolution
walks duck-typed attributes (``.name`` first, ``.__name__`` second) so the
core FSM/monitor stays importable in environments without LangChain
installed. The actual ``BaseTool`` import only happens inside the
middleware module.
"""

from __future__ import annotations

from typing import Any, Callable


class ToolRef:
    """Stable label-bearing reference to a LangChain tool or callable.

    Label resolution order, applied once at construction:

    1. If ``tool`` has a ``.name`` attribute that is a non-empty string,
       use it. (Covers ``BaseTool`` instances and ``@tool``-decorated
       callables, which expose ``.name`` on the resulting
       ``StructuredTool``.)
    2. Otherwise, if ``tool`` is callable and has ``__name__``, use that.
    3. Otherwise, raise ``TypeError``.

    The label is read-only. Comparing or hashing two ``ToolRef`` objects
    uses the label only — wrapping different callables that happen to
    share a name yields equal refs.
    """

    __slots__ = ("_label", "_tool")

    def __init__(self, tool: Any) -> None:
        # `.name` first — covers BaseTool subclasses and @tool wrappers
        # (StructuredTool exposes .name) without importing langchain.
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            label = name
        elif callable(tool) and getattr(tool, "__name__", None):
            label = tool.__name__
        else:
            raise TypeError(
                f"ToolRef expects a BaseTool, @tool callable, or named "
                f"callable; got {type(tool).__name__}"
            )
        # __slots__ disables __dict__; assign through object.__setattr__
        # so future attempts to overwrite (label/tool are read-only) fail.
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_tool", tool)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"ToolRef is immutable; cannot set {name!r}")

    @property
    def label(self) -> str:
        return self._label

    @property
    def tool(self) -> Any:
        return self._tool

    def __repr__(self) -> str:
        return f"ToolRef({self._label!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolRef):
            return NotImplemented
        return self._label == other._label

    def __hash__(self) -> int:
        return hash(("ToolRef", self._label))


def ref(tool: Any) -> ToolRef:
    """Convenience shorthand for ``ToolRef(tool)``.

    This is the primary API developers reach for — they always pass the
    tool function, never a name string.
    """
    return ToolRef(tool)
