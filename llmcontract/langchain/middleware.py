"""LangChain integration: a thin ``AgentMiddleware`` that drives a
``ProtocolMonitor`` from ``wrap_tool_call`` / ``awrap_tool_call``.

This is the only module in the submodule that imports LangChain. The
FSM, the monitor, and ``ToolRef`` are all framework-agnostic; they can
be unit-tested in isolation without LangChain installed.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from llmcontract.langchain.monitor import ProtocolMonitor
from llmcontract.langchain.tool_ref import ToolRef


class ProtocolEnforcerMiddleware:
    """LangChain ``AgentMiddleware`` that fires ``send`` and ``recv``
    transitions on the wrapped monitor for every registered tool call.

    Construction does the import-and-subclass dance against LangChain's
    ``AgentMiddleware`` lazily so the rest of the package stays
    importable without LangChain. The actual middleware object exposed
    via ``.middleware`` is a real ``AgentMiddleware`` subclass instance
    that you pass to ``create_agent(middleware=[...])``.
    """

    def __init__(
        self,
        monitor: ProtocolMonitor,
        tool_refs: list[ToolRef],
    ) -> None:
        self._monitor = monitor
        # The only place tool *name strings* surface inside the library —
        # we look up by the name LangChain hands us in the request, so
        # the developer never has to write or see a string.
        self._ref_by_label: dict[str, ToolRef] = {t.label: t for t in tool_refs}
        self._impl = self._build_impl()

    @property
    def middleware(self) -> Any:
        """The ``AgentMiddleware`` subclass instance to pass to
        ``create_agent(middleware=[...])``."""
        return self._impl

    # ── Build the real AgentMiddleware subclass instance ────

    def _build_impl(self) -> Any:
        # Imports happen here, not at module load — keeps the rest of
        # the langchain submodule usable in environments without
        # langchain installed.
        from langchain.agents.middleware import AgentMiddleware

        outer = self  # captured by closure into the methods below

        class _Impl(AgentMiddleware):
            def wrap_tool_call(self, request, handler):  # type: ignore[override]
                return outer._dispatch_sync(request, handler)

            async def awrap_tool_call(self, request, handler):  # type: ignore[override]
                return await outer._dispatch_async(request, handler)

        return _Impl()

    # ── Sync and async dispatch share one logical body ──────

    def _dispatch_sync(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        name = request.tool_call["name"]
        tool_ref = self._ref_by_label.get(name)
        if tool_ref is None:
            # Tool isn't registered in this protocol — pass through
            # unmonitored. Partial protocol coverage is a valid use case
            # (e.g., monitoring only the booking subset of a larger tool
            # surface).
            return handler(request)

        args = request.tool_call.get("args", {}) or {}
        self._monitor.transition(tool_ref, phase="send", metadata={"args": args})

        # Tool exception path: let it propagate. We deliberately do NOT
        # fire the recv transition — the protocol stays in the
        # post-send state, mirroring reality (the tool didn't actually
        # produce a result). A tool exception is *not* a protocol
        # violation; it's an orthogonal failure mode and the user's
        # outer error handling owns it.
        result = handler(request)

        self._monitor.transition(tool_ref, phase="recv", metadata={"result": result})
        return result

    async def _dispatch_async(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        name = request.tool_call["name"]
        tool_ref = self._ref_by_label.get(name)
        if tool_ref is None:
            return await handler(request)

        args = request.tool_call.get("args", {}) or {}
        self._monitor.transition(tool_ref, phase="send", metadata={"args": args})

        result = await handler(request)

        self._monitor.transition(tool_ref, phase="recv", metadata={"result": result})
        return result
