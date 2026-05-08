"""Tests for ``llmcontract.langchain.middleware`` (0.4.0 architecture).

The middleware now stores FSM state in LangGraph's checkpointed
``AgentState`` and exposes ``before_agent`` / ``before_model`` /
``after_model`` / ``wrap_tool_call`` hooks. Tests drive each hook
directly with synthetic ``state`` dicts and stand-in ``request``
objects — same approach as 0.3.x tests, just against the new surface.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

# The middleware imports LangChain at construction time; skip if absent.
pytest.importorskip("langchain.agents.middleware")
pytest.importorskip("langgraph.types")

from langchain_core.messages import ToolMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from llmcontract.langchain import (  # noqa: E402
    CheckpointedProtocolMiddleware,
    ProtocolFSM,
    ToolRef,
    Transition,
    ViolationEvent,
)


# ── Fake helpers ────────────────────────────────────────────


def _ref(name: str) -> ToolRef:
    def fn():
        return None

    fn.name = name  # type: ignore[attr-defined]
    return ToolRef(fn)


def _request(
    name: str,
    args: dict | None = None,
    tc_id: str = "tc-1",
    state: dict | None = None,
) -> Any:
    """Mimic LangGraph's ``ToolCallRequest`` shape — only ``.tool_call``
    and ``.state`` are read by the middleware."""
    return SimpleNamespace(
        tool_call={"name": name, "args": args or {}, "id": tc_id},
        state=state if state is not None else {},
    )


def _tool_message(name: str = "search", content: str = "results") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="tc-1", name=name)


def _build_search_book_fsm() -> tuple[ProtocolFSM, ToolRef, ToolRef]:
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


def _build_mw(
    fsm: ProtocolFSM,
    refs: list[ToolRef],
    on_violation=lambda v: None,
):
    return CheckpointedProtocolMiddleware(
        fsm=fsm, on_violation=on_violation, tool_refs=refs
    ).middleware


# ── before_agent initializes state ──────────────────────────


class TestBeforeAgent:
    def test_initializes_fsm_state_and_trace_when_absent(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        update = mw.before_agent({}, runtime=None)
        assert update == {"fsm_state": "idle", "fsm_trace": []}

    def test_skips_initialization_when_already_present(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        update = mw.before_agent(
            {"fsm_state": "results", "fsm_trace": ["send:search", "recv:search"]},
            runtime=None,
        )
        assert update is None


# ── wrap_tool_call drives FSM through state ─────────────────


class TestWrapToolCall:
    def test_registered_tool_returns_command_with_state_updates(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        def handler(req: Any) -> ToolMessage:
            return _tool_message()

        result = mw.wrap_tool_call(
            _request("search", state={"fsm_state": "idle", "fsm_trace": []}),
            handler,
        )

        assert isinstance(result, Command)
        assert result.update["fsm_state"] == "results"
        assert result.update["fsm_trace"] == ["send:search", "recv:search"]
        assert len(result.update["messages"]) == 1

    def test_unregistered_tool_passes_through(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        called = {"n": 0}

        def handler(req: Any) -> ToolMessage:
            called["n"] += 1
            return _tool_message(name="weather", content="sunny")

        result = mw.wrap_tool_call(
            _request("weather", state={"fsm_state": "idle", "fsm_trace": []}),
            handler,
        )

        assert called["n"] == 1
        # Untouched return — no Command wrapping when tool isn't registered.
        assert isinstance(result, ToolMessage)
        assert result.content == "sunny"

    def test_handler_exception_skips_recv_and_propagates(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        def handler(req: Any) -> Any:
            raise RuntimeError("tool failure")

        with pytest.raises(RuntimeError, match="tool failure"):
            mw.wrap_tool_call(
                _request("search", state={"fsm_state": "idle", "fsm_trace": []}),
                handler,
            )

    def test_violation_invokes_callback(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        captured: list[ViolationEvent] = []
        mw = _build_mw(fsm, [search, book], on_violation=captured.append)

        def handler(req: Any) -> ToolMessage:
            return _tool_message(name="book", content="x")

        # Booking from idle is illegal.
        mw.wrap_tool_call(
            _request("book", state={"fsm_state": "idle", "fsm_trace": []}),
            handler,
        )

        # Same as 0.3.x: when on_violation returns normally, both
        # send and recv fire (and both violate from idle).
        assert len(captured) == 2
        assert captured[0].event == "send:book"
        assert captured[1].event == "recv:book"

    def test_send_metadata_carries_args(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        seen: dict = {}
        # Override the search/send transition to capture metadata.
        fsm._transitions[("idle", "send:search")] = Transition(
            source="idle", tool=search, phase="send", target="searching",
            action=lambda ctx: seen.update(ctx.metadata),
        )
        mw = _build_mw(fsm, [search, book])

        def handler(req: Any) -> ToolMessage:
            return _tool_message()

        mw.wrap_tool_call(
            _request("search", args={"q": "Rome"},
                     state={"fsm_state": "idle", "fsm_trace": []}),
            handler,
        )
        assert seen == {"args": {"q": "Rome"}}


# ── awrap_tool_call mirrors sync ───────────────────────────


class TestAwrapToolCall:
    def test_async_full_cycle(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        async def handler(req: Any) -> ToolMessage:
            return _tool_message()

        result = asyncio.run(
            mw.awrap_tool_call(
                _request("search", state={"fsm_state": "idle", "fsm_trace": []}),
                handler,
            )
        )

        assert isinstance(result, Command)
        assert result.update["fsm_state"] == "results"
        assert result.update["fsm_trace"] == ["send:search", "recv:search"]


# ── before_model: interrupt-gated transitions ───────────────


class TestBeforeModelInterrupt:
    def test_no_interrupt_transition_returns_none(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])

        # idle has only send:search outgoing; no interrupts.
        update = mw.before_model({"fsm_state": "idle", "fsm_trace": []}, runtime=None)
        assert update is None

    def test_single_interrupt_transition_fires_on_resume(self) -> None:
        # before_model uses langgraph.types.interrupt() internally; we
        # drive the inner _maybe_interrupt with a fake interrupt fn.
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="presented")
            .add_transition(Transition(
                source="presented", phase="recv", target="approved",
                event_label="UserApproval", interrupt=True,
            ))
        )
        cpm = CheckpointedProtocolMiddleware(
            fsm=fsm, on_violation=lambda v: None, tool_refs=[search]
        )
        captured_payload: dict = {}

        def fake_interrupt(payload):
            captured_payload.update(payload)
            return {"approval": "DL317"}

        result = cpm._maybe_interrupt(
            {"fsm_state": "presented", "fsm_trace": []},
            fake_interrupt,
        )
        assert result == {
            "fsm_state": "approved",
            "fsm_trace": ["recv:UserApproval"],
        }
        assert captured_payload["current_state"] == "presented"
        assert captured_payload["expected"] == ["recv:UserApproval"]

    def test_explicit_event_label_in_resume_disambiguates(self) -> None:
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="s")
            .add_transition(Transition(
                source="s", phase="recv", target="a",
                event_label="A", interrupt=True,
            ))
            .add_transition(Transition(
                source="s", phase="recv", target="b",
                event_label="B", interrupt=True,
            ))
        )
        cpm = CheckpointedProtocolMiddleware(
            fsm=fsm, on_violation=lambda v: None, tool_refs=[search]
        )

        def fake_interrupt(payload):
            return {"event_label": "B", "metadata": {"x": 1}}

        result = cpm._maybe_interrupt(
            {"fsm_state": "s", "fsm_trace": []}, fake_interrupt
        )
        assert result == {"fsm_state": "b", "fsm_trace": ["recv:B"]}

    def test_ambiguous_resume_without_label_raises(self) -> None:
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="s")
            .add_transition(Transition(
                source="s", phase="recv", target="a",
                event_label="A", interrupt=True,
            ))
            .add_transition(Transition(
                source="s", phase="recv", target="b",
                event_label="B", interrupt=True,
            ))
        )
        cpm = CheckpointedProtocolMiddleware(
            fsm=fsm, on_violation=lambda v: None, tool_refs=[search]
        )

        def fake_interrupt(payload):
            return "unstructured value"

        with pytest.raises(ValueError, match="ambiguous"):
            cpm._maybe_interrupt(
                {"fsm_state": "s", "fsm_trace": []}, fake_interrupt
            )


# ── after_model: structured response matching ───────────────


class _PresentOptionsResp:
    def __init__(self, options: list[str]) -> None:
        self.options = options


class _OtherResp:
    pass


class TestAfterModelStructuredResponse:
    def test_no_structured_response_returns_none(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        mw = _build_mw(fsm, [search, book])
        update = mw.after_model({"fsm_state": "idle", "fsm_trace": []}, runtime=None)
        assert update is None

    def test_matching_response_fires_transition(self) -> None:
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="results")
            .add_transition(Transition(
                source="results", phase="send", target="presented",
                event_label="PresentOptions",
                match_structured_response=_PresentOptionsResp,
            ))
        )
        mw = _build_mw(fsm, [search])
        update = mw.after_model(
            {
                "fsm_state": "results",
                "fsm_trace": [],
                "structured_response": _PresentOptionsResp(["AA101", "DL317"]),
            },
            runtime=None,
        )
        assert update == {
            "fsm_state": "presented",
            "fsm_trace": ["send:PresentOptions"],
        }

    def test_non_matching_response_returns_none(self) -> None:
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="results")
            .add_transition(Transition(
                source="results", phase="send", target="presented",
                event_label="PresentOptions",
                match_structured_response=_PresentOptionsResp,
            ))
        )
        mw = _build_mw(fsm, [search])
        update = mw.after_model(
            {
                "fsm_state": "results",
                "fsm_trace": [],
                "structured_response": _OtherResp(),
            },
            runtime=None,
        )
        assert update is None

    def test_ambiguous_match_raises(self) -> None:
        search = _ref("search")
        fsm = (
            ProtocolFSM(initial="s")
            .add_transition(Transition(
                source="s", phase="send", target="a",
                event_label="A",
                match_structured_response=_PresentOptionsResp,
            ))
            .add_transition(Transition(
                source="s", phase="send", target="b",
                event_label="B",
                match_structured_response=_PresentOptionsResp,
            ))
        )
        mw = _build_mw(fsm, [search])
        with pytest.raises(ValueError, match="ambiguous"):
            mw.after_model(
                {
                    "fsm_state": "s",
                    "fsm_trace": [],
                    "structured_response": _PresentOptionsResp([]),
                },
                runtime=None,
            )
