"""LangChain integration: ``AgentMiddleware`` that drives a
``ProtocolFSM`` through LangGraph-checkpointed state.

The previous (0.3.x) ``ProtocolEnforcerMiddleware`` held FSM state on
a ``ProtocolMonitor`` instance attribute. That works in single-process
happy paths but breaks the moment LangGraph rehydrates the graph from
a checkpoint — after a ``HumanInTheLoopMiddleware`` interrupt, after a
worker restart, or in any multi-pod deployment — because the
middleware is reconstructed but instance attributes are not part of
the checkpoint. It also raced when one ``ProtocolEnforcerMiddleware``
was reused across concurrent agent invocations.

``CheckpointedProtocolMiddleware`` (0.4.0) fixes both by storing FSM
state in an ``AgentState`` subclass — LangGraph checkpoints it
automatically, keys it by ``thread_id``, and resumes correctly. It
also adds two design improvements:

* ``Transition.interrupt=True`` — the middleware suspends the agent
  via ``langgraph.types.interrupt(...)`` when the FSM enters such a
  transition's source state, eliminating the "orchestrator forgot to
  fire the gate" failure class for human-approval steps.
* ``Transition.match_structured_response=<type>`` — the middleware
  fires the transition deterministically when the model emits a typed
  ``structured_response`` of that type, replacing brittle text
  pattern detection.

This is the only module in the submodule that imports LangChain (or
LangGraph). The FSM, the ``fire_step`` helper, and ``ToolRef`` stay
framework-agnostic and unit-testable in isolation.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from llmcontract.langchain.fsm import (
    ProtocolFSM,
    Transition,
    ViolationHandler,
)
from llmcontract.langchain.monitor import fire_step
from llmcontract.langchain.tool_ref import ToolRef


# ── State schema ────────────────────────────────────────────


def _make_state_schema() -> Any:
    """Build the ``ProtocolState`` TypedDict against LangChain's
    ``AgentState``. Done lazily so the module is importable without
    LangChain installed (the FSM/monitor pieces stay usable on their
    own).
    """
    from langchain.agents.middleware import AgentState

    class ProtocolState(AgentState):
        fsm_state: str
        fsm_trace: list[str]

    return ProtocolState


# ── The middleware ──────────────────────────────────────────


class CheckpointedProtocolMiddleware:
    """Build an ``AgentMiddleware`` that enforces a ``ProtocolFSM``
    against an agent run, with FSM state held in LangGraph's
    checkpointed ``AgentState``.

    Pass the resulting ``.middleware`` into
    ``create_agent(middleware=[...])``. The middleware:

    * Initializes ``state["fsm_state"]`` from ``fsm.initial`` and
      ``state["fsm_trace"]`` from ``[]`` in ``before_agent``.
    * In ``wrap_tool_call``, fires the ``send`` and ``recv``
      transitions for every registered tool, threading state through
      the returned ``Command``.
    * In ``before_model``, suspends the agent via
      ``langgraph.types.interrupt(...)`` when the current FSM state
      has any outgoing ``Transition(interrupt=True)``. The resume
      value is treated as the transition's metadata.
    * In ``after_model``, fires any ``Transition`` that declares
      ``match_structured_response=<type>`` whose type matches
      ``state["structured_response"]``.

    After the agent returns, read ``result["fsm_state"]`` and
    ``result["fsm_trace"]`` for the final FSM state and full event
    trace. ``fsm.is_terminal(result["fsm_state"])`` tells you whether
    the protocol completed.
    """

    def __init__(
        self,
        *,
        fsm: ProtocolFSM,
        on_violation: ViolationHandler,
        tool_refs: list[ToolRef],
    ) -> None:
        self._fsm = fsm
        self._on_violation = on_violation
        # Tool-name lookup is the single place the library reads a
        # string from LangChain's request — the developer never writes
        # or sees a tool name string in the public API.
        self._ref_by_label: dict[str, ToolRef] = {
            t.label: t for t in tool_refs
        }
        self._impl = self._build_impl()

    @property
    def middleware(self) -> Any:
        """The ``AgentMiddleware`` instance to pass to
        ``create_agent(middleware=[...])``."""
        return self._impl

    # ── Build the AgentMiddleware subclass ────────────────────

    def _build_impl(self) -> Any:
        # All LangChain/LangGraph imports happen here, not at module
        # load — keeps the rest of the submodule importable in
        # environments without those packages installed.
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command, interrupt

        outer = self
        ProtocolState = _make_state_schema()

        class _Impl(AgentMiddleware):
            state_schema = ProtocolState  # type: ignore[assignment]

            def before_agent(self, state, runtime):  # type: ignore[override]
                # Initialize FSM state on first entry. ``state.get`` is
                # safe — TypedDicts are dicts at runtime.
                updates: dict[str, Any] = {}
                if "fsm_state" not in state:
                    updates["fsm_state"] = outer._fsm.initial
                if "fsm_trace" not in state:
                    updates["fsm_trace"] = []
                return updates or None

            def before_model(self, state, runtime):  # type: ignore[override]
                return outer._maybe_interrupt(state, interrupt)

            def after_model(self, state, runtime):  # type: ignore[override]
                return outer._maybe_match_structured_response(state)

            def wrap_tool_call(self, request, handler):  # type: ignore[override]
                return outer._dispatch_sync(
                    request, handler, ToolMessage, Command
                )

            async def awrap_tool_call(self, request, handler):  # type: ignore[override]
                return await outer._dispatch_async(
                    request, handler, ToolMessage, Command
                )

        return _Impl()

    # ── Hook bodies (sync; pulled out so they're testable) ───

    def _maybe_interrupt(
        self,
        state: Any,
        interrupt_fn: Callable[[Any], Any],
    ) -> dict[str, Any] | None:
        fsm_state = state.get("fsm_state", self._fsm.initial)
        candidates = self._fsm.interrupt_transitions_from(fsm_state)
        if not candidates:
            return None

        payload = {
            "current_state": fsm_state,
            "expected": [t.event for t in candidates],
            "trace": list(state.get("fsm_trace", [])),
        }
        resume_value = interrupt_fn(payload)

        # Resolve which transition the resume value targets. Two
        # supported shapes:
        #   1. dict with explicit "event_label" — disambiguates when
        #      multiple interrupt transitions share a source state
        #   2. anything else — only legal when there's exactly one
        #      candidate; the value becomes the metadata
        if isinstance(resume_value, dict) and "event_label" in resume_value:
            label = resume_value["event_label"]
            metadata = resume_value.get("metadata", {})
            t = _find_transition(candidates, label)
        elif len(candidates) == 1:
            t = candidates[0]
            metadata = (
                resume_value
                if isinstance(resume_value, dict)
                else {"resume": resume_value}
            )
        else:
            raise ValueError(
                f"interrupt resume value is ambiguous: {len(candidates)} "
                f"interrupt-gated transitions from state {fsm_state!r}; "
                f"resume payload must include 'event_label' to "
                f"disambiguate"
            )

        new_state, new_trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=list(state.get("fsm_trace", [])),
            label=t.event_label,  # type: ignore[arg-type]
            phase=t.phase,
            tool_ref=None,
            event_label=t.event_label,
            metadata=metadata,
            on_violation=self._on_violation,
        )
        return {"fsm_state": new_state, "fsm_trace": new_trace}

    def _maybe_match_structured_response(
        self,
        state: Any,
    ) -> dict[str, Any] | None:
        structured = state.get("structured_response")
        if structured is None:
            return None

        fsm_state = state.get("fsm_state", self._fsm.initial)
        candidates = self._fsm.structured_response_transitions_from(fsm_state)
        matched = [
            t for t in candidates
            if isinstance(structured, t.match_structured_response)  # type: ignore[arg-type]
        ]
        if not matched:
            return None
        if len(matched) > 1:
            raise ValueError(
                f"ambiguous structured_response match: "
                f"{len(matched)} transitions from state {fsm_state!r} "
                f"match the same response type"
            )
        t = matched[0]
        new_state, new_trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=list(state.get("fsm_trace", [])),
            label=t.event_label,  # type: ignore[arg-type]
            phase=t.phase,
            tool_ref=None,
            event_label=t.event_label,
            metadata={"structured_response": structured},
            on_violation=self._on_violation,
        )
        return {"fsm_state": new_state, "fsm_trace": new_trace}

    # ── Tool-call dispatch (sync + async share one body) ─────

    def _dispatch_sync(
        self,
        request: Any,
        handler: Callable[[Any], Any],
        ToolMessage: type,
        Command: type,
    ) -> Any:
        name = request.tool_call["name"]
        tool_ref = self._ref_by_label.get(name)
        if tool_ref is None:
            # Unregistered tool — pass through unmonitored. Partial
            # protocol coverage is a valid use case (e.g., monitoring
            # only the booking subset of a larger tool surface).
            return handler(request)

        fsm_state = request.state.get("fsm_state", self._fsm.initial)
        trace = list(request.state.get("fsm_trace", []))
        args = request.tool_call.get("args", {}) or {}

        fsm_state, trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=trace,
            label=tool_ref.label,
            phase="send",
            tool_ref=tool_ref,
            event_label=None,
            metadata={"args": args},
            on_violation=self._on_violation,
        )

        # Tool exceptions propagate; recv does not fire. The protocol
        # stays in the post-send state, which mirrors reality (the
        # tool didn't produce a result). A tool exception is *not* a
        # protocol violation; it's an orthogonal failure mode and the
        # caller's outer error handling owns it.
        result = handler(request)

        fsm_state, trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=trace,
            label=tool_ref.label,
            phase="recv",
            tool_ref=tool_ref,
            event_label=None,
            metadata={"result": result},
            on_violation=self._on_violation,
        )

        return _wrap_with_state_update(result, fsm_state, trace, ToolMessage, Command)

    async def _dispatch_async(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
        ToolMessage: type,
        Command: type,
    ) -> Any:
        name = request.tool_call["name"]
        tool_ref = self._ref_by_label.get(name)
        if tool_ref is None:
            return await handler(request)

        fsm_state = request.state.get("fsm_state", self._fsm.initial)
        trace = list(request.state.get("fsm_trace", []))
        args = request.tool_call.get("args", {}) or {}

        fsm_state, trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=trace,
            label=tool_ref.label,
            phase="send",
            tool_ref=tool_ref,
            event_label=None,
            metadata={"args": args},
            on_violation=self._on_violation,
        )

        result = await handler(request)

        fsm_state, trace, _ = fire_step(
            fsm=self._fsm,
            state=fsm_state,
            trace=trace,
            label=tool_ref.label,
            phase="recv",
            tool_ref=tool_ref,
            event_label=None,
            metadata={"result": result},
            on_violation=self._on_violation,
        )

        return _wrap_with_state_update(result, fsm_state, trace, ToolMessage, Command)


# ── Helpers ─────────────────────────────────────────────────


def _find_transition(
    candidates: list[Transition], label: str
) -> Transition:
    for t in candidates:
        if t.event_label == label:
            return t
    raise ValueError(
        f"interrupt resume specified event_label={label!r}, but no "
        f"interrupt-gated transition with that label is available; "
        f"valid options: {[t.event_label for t in candidates]}"
    )


def _wrap_with_state_update(
    result: Any,
    fsm_state: str,
    trace: list[str],
    ToolMessage: type,
    Command: type,
) -> Any:
    """Bundle the handler's return with FSM state updates so LangGraph
    persists ``fsm_state`` / ``fsm_trace`` through its checkpoint.

    ``handler`` may return a ``ToolMessage`` (the common case) or a
    ``Command`` (when other middleware is also injecting state). Both
    paths fold the FSM state updates in alongside.
    """
    state_update = {"fsm_state": fsm_state, "fsm_trace": trace}

    if isinstance(result, Command):
        existing = result.update if isinstance(result.update, dict) else {}
        merged = {**existing, **state_update}
        # Preserve other Command fields if present (goto, graph, etc.)
        kwargs: dict[str, Any] = {"update": merged}
        for attr in ("goto", "graph", "resume"):
            value = getattr(result, attr, None)
            if value is not None:
                kwargs[attr] = value
        return Command(**kwargs)

    if isinstance(result, ToolMessage):
        return Command(update={"messages": [result], **state_update})

    # Anything else — most likely an AIMessage from a custom handler.
    # Wrap it the same way the framework would; a list-of-messages
    # update is the standard contract.
    return Command(update={"messages": [result], **state_update})
