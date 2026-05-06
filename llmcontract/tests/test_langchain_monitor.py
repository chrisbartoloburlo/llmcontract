"""Tests for ``llmcontract.langchain.monitor``."""

from __future__ import annotations

from llmcontract.langchain import (
    ProtocolFSM,
    ProtocolMonitor,
    ToolRef,
    Transition,
    ViolationEvent,
)


def _ref(name: str) -> ToolRef:
    def fn():
        return None

    fn.name = name  # type: ignore[attr-defined]
    return ToolRef(fn)


def _build_search_book_fsm() -> tuple[ProtocolFSM, ToolRef, ToolRef]:
    """Returns (fsm, search_ref, book_ref) wired as a four-edge FSM."""
    search = _ref("search")
    book = _ref("book")
    fsm = (
        ProtocolFSM(initial="idle")
        .add_transition(Transition(source="idle", tool=search, phase="send", target="searching"))
        .add_transition(Transition(source="searching", tool=search, phase="recv", target="results"))
        .add_transition(Transition(source="results", tool=book, phase="send", target="booking"))
        .add_transition(Transition(source="booking", tool=book, phase="recv", target="done"))
        .mark_terminal("done")
    )
    return fsm, search, book


# ── State updates ───────────────────────────────────────────


class TestStateUpdates:
    def test_valid_transition_advances_state(self) -> None:
        fsm, search, _ = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        assert m.transition(search, "send") is True
        assert m.state == "searching"

    def test_violation_does_not_advance_state(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        # First event should be send:search; book first is a violation.
        assert m.transition(book, "send") is False
        assert m.state == "idle"

    def test_full_happy_path_terminates(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        for tool, phase in [(search, "send"), (search, "recv"), (book, "send"), (book, "recv")]:
            assert m.transition(tool, phase) is True
        assert m.state == "done"
        assert m.is_complete()


# ── Violation handler is called correctly ──────────────────


class TestViolationHandler:
    def test_handler_receives_violation_event(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        captured: list[ViolationEvent] = []
        m = ProtocolMonitor(fsm=fsm, on_violation=captured.append)

        m.transition(book, "send")  # illegal first move

        assert len(captured) == 1
        v = captured[0]
        assert v.event == "send:book"
        assert v.current_state == "idle"
        assert v.expected == ["send:search"]
        assert v.tool_ref == book
        assert v.phase == "send"

    def test_violating_event_appears_in_trace(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        captured: list[ViolationEvent] = []
        m = ProtocolMonitor(fsm=fsm, on_violation=captured.append)

        m.transition(search, "send")
        m.transition(search, "recv")
        m.transition(book, "recv")  # wrong phase first

        # The violating event is in the trace alongside the prior valid ones.
        assert captured[-1].trace == ["send:search", "recv:search", "recv:book"]

    def test_returns_true_on_success_false_on_failure(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        assert m.transition(search, "send") is True
        assert m.transition(book, "send") is False  # not allowed from "searching"


# ── Trace snapshot semantics ────────────────────────────────


class TestTraceSnapshots:
    def test_monitor_trace_property_is_a_copy(self) -> None:
        fsm, search, _ = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        m.transition(search, "send")
        snapshot = m.trace
        snapshot.append("garbage")
        assert m.trace == ["send:search"]  # internal trace untouched

    def test_context_trace_is_a_copy(self) -> None:
        # The MonitorContext passed to the guard receives a snapshot —
        # mutating it must not corrupt the monitor's running trace.
        fsm, search, _ = _build_search_book_fsm()
        traces: list[list[str]] = []

        def grabbing_guard(ctx) -> bool:
            ctx.trace.append("garbage")
            traces.append(list(ctx.trace))
            return True

        # Replace the search/send transition with a guarded version.
        fsm._transitions[("idle", "send:search")] = Transition(
            source="idle", tool=search, phase="send", target="searching",
            guard=grabbing_guard,
        )

        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        m.transition(search, "send")

        assert traces == [["send:search", "garbage"]]
        assert m.trace == ["send:search"]


# ── Reset and is_complete ──────────────────────────────────


class TestResetAndCompletion:
    def test_reset_restores_initial_state(self) -> None:
        fsm, search, _ = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        m.transition(search, "send")
        m.reset()
        assert m.state == "idle"
        assert m.trace == []

    def test_reset_honours_explicit_initial_state(self) -> None:
        fsm, search, _ = _build_search_book_fsm()
        m = ProtocolMonitor(
            fsm=fsm, on_violation=lambda v: None, initial_state="results"
        )
        m.transition(search, "send")  # not legal from "results" — violation
        m.reset()
        assert m.state == "results"

    def test_is_complete_only_when_in_terminal_state(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        assert not m.is_complete()
        for tool, phase in [(search, "send"), (search, "recv"), (book, "send"), (book, "recv")]:
            m.transition(tool, phase)
        assert m.is_complete()


# ── Metadata is threaded through ───────────────────────────


class TestMetadata:
    def test_send_metadata_reaches_guard(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        seen_args: list[dict] = []

        # Guard the book/send transition: only allow if args are non-empty.
        fsm._transitions[("results", "send:book")] = Transition(
            source="results", tool=book, phase="send", target="booking",
            guard=lambda ctx: (seen_args.append(ctx.metadata.get("args", {})) or True),
        )

        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        for tool, phase, meta in [
            (search, "send", None),
            (search, "recv", None),
            (book, "send", {"args": {"id": "F1"}}),
        ]:
            m.transition(tool, phase, metadata=meta)

        assert seen_args == [{"id": "F1"}]

    def test_recv_metadata_reaches_action(self) -> None:
        fsm, search, _ = _build_search_book_fsm()
        results: list[str] = []

        fsm._transitions[("searching", "recv:search")] = Transition(
            source="searching", tool=search, phase="recv", target="results",
            action=lambda ctx: results.append(ctx.metadata.get("result", "")),
        )

        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        m.transition(search, "send")
        m.transition(search, "recv", metadata={"result": "AA101, UA205"})

        assert results == ["AA101, UA205"]


# ── Action fires only on success ───────────────────────────


class TestActionOnlyOnSuccess:
    def test_action_skipped_on_violation(self) -> None:
        # Pre-register a guarded transition that always fails; the
        # action must not run.
        called: list[str] = []
        search = _ref("search")
        fsm = ProtocolFSM(initial="a").add_transition(
            Transition(
                source="a",
                tool=search,
                phase="send",
                target="b",
                guard=lambda ctx: False,
                action=lambda ctx: called.append("ran"),
            )
        )
        m = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        m.transition(search, "send")
        assert called == []
