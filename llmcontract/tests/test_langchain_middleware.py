"""Tests for ``llmcontract.langchain.middleware``.

The middleware imports from LangChain at construction time. Tests are
skipped automatically if LangChain isn't available — the rest of the
submodule (FSM, monitor, ToolRef) is fully testable without it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

# Skip the whole module cleanly if LangChain isn't present.
pytest.importorskip("langchain.agents.middleware")

from llmcontract.langchain import (  # noqa: E402
    ProtocolEnforcerMiddleware,
    ProtocolFSM,
    ProtocolMonitor,
    ToolRef,
    Transition,
    ViolationEvent,
)


# ── Fakes that mimic LangChain's ToolCallRequest / handler shapes ─


def _ref(name: str) -> ToolRef:
    def fn():
        return None

    fn.name = name  # type: ignore[attr-defined]
    return ToolRef(fn)


def _request(name: str, args: dict | None = None, tc_id: str = "tc-1") -> Any:
    """Mimic the ``ToolCallRequest`` shape — only ``.tool_call`` is read
    by the middleware."""
    return SimpleNamespace(tool_call={"name": name, "args": args or {}, "id": tc_id})


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


# ── Synchronous wrap_tool_call ─────────────────────────────


class TestSyncMiddleware:
    def test_registered_tool_fires_send_then_recv(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        def handler(req: Any) -> Any:
            return SimpleNamespace(content="results", tool_call_id=req.tool_call["id"])

        result = mw.wrap_tool_call(_request("search"), handler)

        assert monitor.state == "results"
        assert monitor.trace == ["send:search", "recv:search"]
        assert result.content == "results"

    def test_unregistered_tool_passes_through_unmonitored(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        called = {"n": 0}

        def handler(req: Any) -> Any:
            called["n"] += 1
            return SimpleNamespace(content="passthrough")

        # `weather` isn't in tool_refs.
        result = mw.wrap_tool_call(_request("weather"), handler)

        assert called["n"] == 1
        assert result.content == "passthrough"
        assert monitor.state == "idle"
        assert monitor.trace == []  # nothing recorded

    def test_handler_exception_skips_recv_and_propagates(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        def handler(req: Any) -> Any:
            raise RuntimeError("tool failure")

        with pytest.raises(RuntimeError, match="tool failure"):
            mw.wrap_tool_call(_request("search"), handler)

        # `send` fired before the handler; `recv` did not — the
        # protocol stays in the post-send state. The monitor is *not*
        # in violation; this is a tool exception, not a sequencing bug.
        assert monitor.state == "searching"
        assert monitor.trace == ["send:search"]

    def test_violation_invokes_on_violation_callback(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        captured: list[ViolationEvent] = []
        monitor = ProtocolMonitor(fsm=fsm, on_violation=captured.append)
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        def handler(req: Any) -> Any:
            return SimpleNamespace(content="x")

        # Booking before searching — !book is illegal from idle.
        mw.wrap_tool_call(_request("book"), handler)

        # Per spec section 7: when on_violation returns normally,
        # execution continues — handler still runs, recv still fires.
        # `recv:book` is also illegal from idle, so the handler is
        # called twice (once per phase). A handler that raises would
        # short-circuit at the send.
        assert len(captured) == 2
        assert captured[0].event == "send:book"
        assert captured[0].current_state == "idle"
        assert captured[1].event == "recv:book"
        assert captured[1].current_state == "idle"

    def test_send_metadata_carries_args(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        seen: dict = {}
        # Override the search/send transition to capture metadata.
        fsm._transitions[("idle", "send:search")] = Transition(
            source="idle", tool=search, phase="send", target="searching",
            action=lambda ctx: seen.update(ctx.metadata),
        )
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: None)
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        def handler(req: Any) -> Any:
            return SimpleNamespace(content="results")

        mw.wrap_tool_call(_request("search", args={"q": "Rome"}), handler)
        assert seen == {"args": {"q": "Rome"}}


# ── Async awrap_tool_call mirrors sync ─────────────────────


class TestAsyncMiddleware:
    def test_async_full_cycle(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        async def handler(req: Any) -> Any:
            return SimpleNamespace(content="async result")

        result = asyncio.run(mw.awrap_tool_call(_request("search"), handler))

        assert monitor.state == "results"
        assert monitor.trace == ["send:search", "recv:search"]
        assert result.content == "async result"

    def test_async_handler_exception_skips_recv(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        async def handler(req: Any) -> Any:
            raise RuntimeError("async tool failure")

        with pytest.raises(RuntimeError, match="async tool failure"):
            asyncio.run(mw.awrap_tool_call(_request("search"), handler))

        assert monitor.state == "searching"
        assert monitor.trace == ["send:search"]

    def test_async_unregistered_passes_through(self) -> None:
        fsm, search, book = _build_search_book_fsm()
        monitor = ProtocolMonitor(fsm=fsm, on_violation=lambda v: pytest.fail(str(v)))
        mw = ProtocolEnforcerMiddleware(monitor=monitor, tool_refs=[search, book]).middleware

        async def handler(req: Any) -> Any:
            return SimpleNamespace(content="passthrough")

        result = asyncio.run(mw.awrap_tool_call(_request("weather"), handler))
        assert result.content == "passthrough"
        assert monitor.trace == []
