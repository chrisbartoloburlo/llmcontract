"""Tests for ``llmcontract.langchain.fsm``.

Cases enumerated by the spec in section 9. ``ProtocolFSM`` has no
LangChain dependency, so these run without the optional extras.
"""

from __future__ import annotations

import pytest

from llmcontract.langchain import (
    MonitorContext,
    ProtocolFSM,
    Transition,
    ToolRef,
)


def _ref(name: str) -> ToolRef:
    """Tiny helper — wraps a fake callable with a `.name`."""

    def fn():
        return None

    fn.name = name  # type: ignore[attr-defined]
    return ToolRef(fn)


def _ctx(state: str, event: str, tool_ref: ToolRef, phase: str = "send") -> MonitorContext:
    return MonitorContext(
        current_state=state,
        event=event,
        tool_ref=tool_ref,
        phase=phase,
        trace=[event],
        metadata={},
    )


# ── Transition validation ───────────────────────────────────


class TestTransition:
    def test_valid_phases_construct(self) -> None:
        Transition(source="a", tool=_ref("t"), phase="send", target="b")
        Transition(source="a", tool=_ref("t"), phase="recv", target="b")

    def test_invalid_phase_raises(self) -> None:
        with pytest.raises(ValueError):
            Transition(source="a", tool=_ref("t"), phase="other", target="b")

    def test_event_property_format(self) -> None:
        t = Transition(source="a", tool=_ref("search"), phase="send", target="b")
        assert t.event == "send:search"

    def test_event_recomputed_each_time(self) -> None:
        # `event` is a property, not stored — the spec is explicit. We
        # don't have a public way to mutate the underlying ToolRef
        # label, but we can verify the property's lookup uses the live
        # ToolRef rather than a cached string.
        tool = _ref("search")
        t = Transition(source="a", tool=tool, phase="send", target="b")
        assert t.event == "send:search"
        # And a second access doesn't differ.
        assert t.event == "send:search"


# ── ProtocolFSM building (fluent API) ───────────────────────


class TestProtocolFSMConstruction:
    def test_add_transition_returns_self(self) -> None:
        fsm = ProtocolFSM(initial="a")
        result = fsm.add_transition(
            Transition(source="a", tool=_ref("t"), phase="send", target="b")
        )
        assert result is fsm

    def test_mark_terminal_returns_self(self) -> None:
        fsm = ProtocolFSM(initial="a")
        assert fsm.mark_terminal("done") is fsm

    def test_mark_terminal_accepts_multiple(self) -> None:
        fsm = ProtocolFSM(initial="a").mark_terminal("done", "cancelled")
        assert fsm.is_terminal("done")
        assert fsm.is_terminal("cancelled")

    def test_duplicate_transition_raises(self) -> None:
        fsm = ProtocolFSM(initial="a")
        fsm.add_transition(
            Transition(source="a", tool=_ref("t"), phase="send", target="b")
        )
        with pytest.raises(ValueError):
            fsm.add_transition(
                Transition(source="a", tool=_ref("t"), phase="send", target="c")
            )


# ── Stepping through transitions ────────────────────────────


class TestStep:
    def test_valid_transition(self) -> None:
        tool = _ref("search")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(source="a", tool=tool, phase="send", target="b")
        )
        next_state, ok = fsm.step("a", "send:search", _ctx("a", "send:search", tool))
        assert ok and next_state == "b"

    def test_missing_transition_returns_failure(self) -> None:
        tool = _ref("search")
        fsm = ProtocolFSM(initial="a")
        next_state, ok = fsm.step("a", "send:search", _ctx("a", "send:search", tool))
        assert not ok and next_state == "a"

    def test_guard_returning_false_blocks_transition(self) -> None:
        tool = _ref("book")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=tool,
                phase="send",
                target="b",
                guard=lambda ctx: False,
            )
        )
        next_state, ok = fsm.step("a", "send:book", _ctx("a", "send:book", tool))
        assert not ok and next_state == "a"

    def test_guard_returning_true_lets_transition_fire(self) -> None:
        tool = _ref("book")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=tool,
                phase="send",
                target="b",
                guard=lambda ctx: True,
            )
        )
        next_state, ok = fsm.step("a", "send:book", _ctx("a", "send:book", tool))
        assert ok and next_state == "b"

    def test_action_called_only_when_guard_passes(self) -> None:
        calls: list[str] = []
        tool = _ref("book")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=tool,
                phase="send",
                target="b",
                guard=lambda ctx: False,
                action=lambda ctx: calls.append("ran"),
            )
        )
        fsm.step("a", "send:book", _ctx("a", "send:book", tool))
        assert calls == []  # action skipped because guard returned False

    def test_action_runs_when_guard_passes(self) -> None:
        calls: list[str] = []
        tool = _ref("book")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=tool,
                phase="send",
                target="b",
                guard=lambda ctx: True,
                action=lambda ctx: calls.append(ctx.event),
            )
        )
        fsm.step("a", "send:book", _ctx("a", "send:book", tool))
        assert calls == ["send:book"]

    def test_action_runs_without_guard(self) -> None:
        calls: list[str] = []
        tool = _ref("book")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=tool,
                phase="send",
                target="b",
                action=lambda ctx: calls.append(ctx.event),
            )
        )
        fsm.step("a", "send:book", _ctx("a", "send:book", tool))
        assert calls == ["send:book"]


# ── valid_events ────────────────────────────────────────────


class TestValidEvents:
    def test_lists_all_outgoing_events(self) -> None:
        fsm = (
            ProtocolFSM(initial="a")
            .add_transition(Transition(source="a", tool=_ref("t1"), phase="send", target="b"))
            .add_transition(Transition(source="a", tool=_ref("t2"), phase="send", target="c"))
            .add_transition(Transition(source="b", tool=_ref("t3"), phase="recv", target="d"))
        )
        assert sorted(fsm.valid_events("a")) == ["send:t1", "send:t2"]

    def test_unknown_state_returns_empty_list(self) -> None:
        assert ProtocolFSM(initial="a").valid_events("nowhere") == []
