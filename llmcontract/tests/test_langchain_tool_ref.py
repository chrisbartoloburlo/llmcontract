"""Tests for ``llmcontract.langchain.tool_ref``.

Cases enumerated by the spec in section 9. We deliberately don't import
LangChain — duck-typed fakes stand in for ``BaseTool`` and ``@tool``
results, exercising the same resolution paths the real types take.
"""

from __future__ import annotations

import pytest

from llmcontract.langchain import ToolRef, ref


# ── Fakes ───────────────────────────────────────────────────


class _FakeBaseTool:
    """Stand-in for langchain_core.tools.BaseTool. Has a `.name`."""

    def __init__(self, name: str) -> None:
        self.name = name


def _fake_decorated(name: str):
    """Stand-in for `@tool`-decorated callable: a function with `.name`."""

    def fn(*args, **kwargs):
        return f"called {name}"

    fn.name = name  # type: ignore[attr-defined]
    return fn


def plain_function(x: int) -> int:
    """A plain callable with no `.name` attribute."""
    return x + 1


# ── Tests ───────────────────────────────────────────────────


class TestLabelResolution:
    def test_basetool_uses_name(self) -> None:
        t = _FakeBaseTool("search")
        assert ToolRef(t).label == "search"

    def test_decorated_callable_uses_name(self) -> None:
        fn = _fake_decorated("book")
        assert ToolRef(fn).label == "book"

    def test_plain_function_uses_dunder_name(self) -> None:
        assert ToolRef(plain_function).label == "plain_function"

    def test_non_callable_without_name_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            ToolRef(42)

    def test_empty_name_falls_back_to_dunder(self) -> None:
        # If a tool's `.name` is an empty string, we fall through to
        # the callable check rather than producing an empty label.
        fn = _fake_decorated("")
        assert ToolRef(fn).label == "fn"


class TestEqualityAndHashing:
    def test_equal_by_label_only(self) -> None:
        a = ToolRef(_FakeBaseTool("search"))
        b = ToolRef(_fake_decorated("search"))
        assert a == b

    def test_different_labels_not_equal(self) -> None:
        assert ToolRef(_FakeBaseTool("search")) != ToolRef(_FakeBaseTool("book"))

    def test_hash_matches_equality(self) -> None:
        a = ToolRef(_FakeBaseTool("search"))
        b = ToolRef(_fake_decorated("search"))
        assert hash(a) == hash(b)

    def test_usable_as_dict_key(self) -> None:
        a = ToolRef(_FakeBaseTool("search"))
        b = ToolRef(_fake_decorated("search"))
        d = {a: "first"}
        d[b] = "second"  # same label → overwrites
        assert d == {a: "second"}

    def test_eq_returns_notimplemented_for_non_toolref(self) -> None:
        # __eq__ should hand back NotImplemented so Python tries the
        # other operand; comparing to an unrelated object yields False.
        assert (ToolRef(_FakeBaseTool("search")) == "search") is False


class TestImmutability:
    def test_label_is_read_only(self) -> None:
        t = ToolRef(_FakeBaseTool("search"))
        with pytest.raises(AttributeError):
            t.label = "other"  # type: ignore[misc]

    def test_tool_is_read_only(self) -> None:
        t = ToolRef(_FakeBaseTool("search"))
        with pytest.raises(AttributeError):
            t._tool = None  # type: ignore[attr-defined]


class TestRefShorthand:
    def test_ref_equivalent_to_constructor(self) -> None:
        fn = _fake_decorated("search")
        assert ref(fn) == ToolRef(fn)


class TestRepr:
    def test_repr_format(self) -> None:
        t = ToolRef(_FakeBaseTool("search"))
        assert repr(t) == "ToolRef('search')"
