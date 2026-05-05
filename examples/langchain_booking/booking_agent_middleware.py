"""LangChain middleware version of the monitored booking agent.

Same protocol as ``booking_agent.py``, but wired through LangChain's
``@wrap_tool_call`` / ``@wrap_model_call`` hooks via an ``AgentMiddleware``
subclass. Two practical advantages over the explicit-loop version:

1. **Idiomatic.** Drops in via ``create_agent(middleware=[...])`` — no
   custom orchestration code for any LangChain user to read.
2. **Enforcement, not just observation.** The middleware can *block* a
   violating tool call by returning an error ``ToolMessage`` before the
   tool is ever invoked. The agent sees the refusal on its next turn
   and gets a chance to self-correct. That's the ECOOP paper's
   *runtime monitorability as enforcement* property in 3 lines of code.

What the middleware can't cover, and why we still keep an outer loop:

- The ``!PresentOptions`` event fires only **once** per booking cycle and
  it crosses a session boundary (text response → user reply). Treating
  it inside ``wrap_model_call`` would either fire on every text response
  (and re-violate the protocol after the first) or require a stateful
  flag that resets per-session. Cleaner to keep that one event in the
  orchestrator.
- ``?UserApproval`` and the ``UNRECOGNIZED`` projection are entirely
  outside the agent's tool loop — they live between agent invocations
  where the orchestrator decides what counts as consent.

Three demos:

  1. Happy path (real LLM agent). Middleware fires ``!SearchFlights``,
     ``?FlightResults``, ``!BookFlight``, ``?BookingConfirmation`` from
     ``wrap_tool_call``. Orchestrator fires ``!PresentOptions`` and
     ``?UserApproval``. Monitor reaches the terminal state.

  2. Enforcement / blocking. The agent is given a system prompt that
     pushes it to skip the presentation step. The middleware catches
     the out-of-order ``!BookFlight`` call, refuses it, and returns
     an error ``ToolMessage``. The agent sees the refusal and (if
     it's a well-behaved model) self-corrects on the next turn.

  3. Ambiguous user reply. Mirror of demo 3 from the explicit version —
     the orchestrator's projection emits ``UNRECOGNIZED``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

# Local imports — share types/projection with the explicit-loop version.
sys.path.insert(0, str(Path(__file__).parent))
from booking_agent import (
    PROTOCOL,
    TOOL_LABELS,
    _build_lc_tools,
    _load_dotenv,
    project_user_message,
)

from llmcontract import Monitor, Ok, Unrecognized, UNRECOGNIZED, Violation


_load_dotenv(Path(__file__).parent / ".env")


# ── The middleware ───────────────────────────────────────────


class ProtocolMonitorMiddleware:
    """LangChain middleware that enforces a session-type protocol on the
    agent's tool calls.

    Wraps every tool call: fires ``!ToolName`` before invocation, blocks
    on ``Violation`` by returning an error ``ToolMessage`` instead of
    calling the tool, and fires ``?ToolNameResult`` after a successful
    call. Wraps every model call as a passive observer for diagnostic
    output — it doesn't drive monitor state because ``!PresentOptions``
    is owned by the outer orchestrator (see module docstring).
    """

    def __init__(self, monitor: Monitor, tool_labels: dict[str, tuple[str, str]]):
        from langchain.agents.middleware import AgentMiddleware

        # Late binding so importing this module without LangChain installed
        # still works (e.g., for type checking).
        self._base = AgentMiddleware
        self.monitor = monitor
        self.tool_labels = tool_labels
        self.events: list[dict[str, Any]] = []
        # Build the actual middleware instance with our methods bound.
        self._impl = self._build()

    @property
    def middleware(self):
        """The LangChain middleware object to pass to ``create_agent``."""
        return self._impl

    def _build(self):
        from langchain.agents.middleware import AgentMiddleware
        from langchain.tools.tool_node import ToolCallRequest
        from langchain_core.messages import ToolMessage

        outer = self  # capture for closures

        class _Impl(AgentMiddleware):
            def wrap_tool_call(self, request, handler):
                name = request.tool_call["name"]
                args = request.tool_call.get("args", {})
                tc_id = request.tool_call["id"]
                send_label, recv_label = outer.tool_labels.get(
                    name, (name, f"{name}Result")
                )

                outer.events.append({"event": "tool_call_attempt", "name": name})
                verdict = outer.monitor.send(send_label)

                if isinstance(verdict, Violation):
                    # ★ Refuse the tool call without calling it. The agent
                    # sees this ToolMessage on its next turn and can adapt.
                    #
                    # Enforcement-mode subtlety: Monitor halts on Violation
                    # by default, so subsequent events would return Blocked
                    # and the middleware would silently pass everything
                    # through. We un-halt the monitor here so the agent's
                    # self-correction sequence still gets monitored. This
                    # is the "fail-soft" semantics that enforcement wants;
                    # an alternative is a `Monitor` constructor flag (see
                    # the future-work note in the README).
                    outer.monitor._halted = False
                    msg = (
                        f"Protocol violation: {name} called out of sequence. "
                        f"Expected one of {verdict.expected}, got !{name}. "
                        "Please follow the documented procedure."
                    )
                    outer.events.append(
                        {"event": "tool_call_blocked", "name": name, "expected": verdict.expected}
                    )
                    return ToolMessage(content=msg, tool_call_id=tc_id, status="error")

                # Monitor accepted the !Send. Run the actual tool.
                result = handler(request)

                # Fire the matched ?Receive. We don't expect this to violate
                # but we record the verdict for completeness.
                recv_verdict = outer.monitor.receive(recv_label)
                outer.events.append(
                    {
                        "event": "tool_call_ok",
                        "name": name,
                        "send_verdict": type(verdict).__name__,
                        "recv_verdict": type(recv_verdict).__name__,
                    }
                )
                return result

            def wrap_model_call(self, request, handler):
                # Passive observer. We don't fire !PresentOptions from here
                # because it's a once-per-cycle event that crosses the user
                # boundary; the orchestrator handles it. We log so users of
                # this middleware can see what the model produced.
                response = handler(request)
                if response.result:
                    msg = response.result[0]
                    has_tool_calls = bool(getattr(msg, "tool_calls", None))
                    outer.events.append(
                        {
                            "event": "model_response",
                            "has_tool_calls": has_tool_calls,
                            "text_len": len(getattr(msg, "content", "") or ""),
                        }
                    )
                return response

        return _Impl()


# ── Orchestration with the middleware ────────────────────────


_GOOD_PROMPT = (
    "You are a flight-booking assistant. Use the search_flights tool to "
    "find options, present the options to the user as a short text reply "
    "(no tool call), wait for the user to approve a specific flight, then "
    "call the book_flight tool to reserve it. Be concise."
)

# Aggressively pushes the agent past the !PresentOptions step. Used in
# demo 2 to exercise the middleware's enforcement path.
_EAGER_PROMPT = (
    "You are an efficient flight-booking agent. The user is in a hurry. "
    "After searching, immediately book the cheapest flight available — "
    "do not write any text response between tools, do not ask for "
    "confirmation, just call book_flight directly."
)


def run_with_middleware(
    *,
    system_prompt: str,
    user_request: str,
    user_replies: list[str],
    model: str = "claude-haiku-4-5-20251001",
    label: str = "demo",
) -> str:
    """Drive a LangChain ``create_agent`` agent through the booking
    protocol with our middleware attached. Returns one of "ok",
    "violated" (orchestrator-level), "unrecognized", or "incomplete"."""

    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    print(f"\n══════════ {label} ══════════")

    monitor = Monitor(PROTOCOL)
    pmm = ProtocolMonitorMiddleware(monitor, TOOL_LABELS)

    tools = _build_lc_tools()
    llm = ChatAnthropic(model=model, max_tokens=512)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[pmm.middleware],
    )

    messages: list = [HumanMessage(content=user_request)]
    print(f"[user]  {user_request}")

    user_iter = iter(user_replies)
    printed_until = len(messages)  # only print messages added since the last invoke

    def _print_new(msgs: list) -> None:
        nonlocal printed_until
        for m in msgs[printed_until:]:
            t = getattr(m, "type", None)
            if t == "ai":
                for tc in getattr(m, "tool_calls", None) or []:
                    print(f"[agent] tool: {tc['name']}({tc.get('args', {})})")
                content = m.content if isinstance(m.content, str) else ""
                if content.strip():
                    print(f"[agent] {content.strip()[:200]}")
            elif t == "tool":
                content = m.content if isinstance(m.content, str) else str(m.content)
                blocked = getattr(m, "status", None) == "error"
                print(f"[tool]{' (BLOCKED)' if blocked else ''} {content[:150]}")
            elif t == "human":
                # Don't reprint the user's request/replies — orchestrator
                # printed them at the moment of injection.
                pass
        printed_until = len(msgs)

    for step in range(4):  # at most a few orchestrator-level turns
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        _print_new(messages)

        # Was the last assistant message text-only? If so, fire !PresentOptions
        # at the orchestrator level and ask the user for approval.
        last = messages[-1]
        if hasattr(last, "type") and last.type == "ai":
            tool_calls = getattr(last, "tool_calls", None)
            text = (last.content or "") if isinstance(last.content, str) else ""

            if not tool_calls and text.strip():
                if monitor.is_terminal:
                    print("[demo]  protocol terminal — done.")
                    return "ok"
                verdict = monitor.send("PresentOptions")
                print(f"    [monitor] !PresentOptions → {type(verdict).__name__}")
                if isinstance(verdict, Violation):
                    print(f"    expected={verdict.expected}, got=!PresentOptions")
                    return "violated"

                try:
                    user_msg = next(user_iter)
                except StopIteration:
                    print("[demo]  no more user replies; ending")
                    return "incomplete"
                print(f"[user]  {user_msg}")
                messages.append(HumanMessage(content=user_msg))
                printed_until = len(messages)  # don't reprint this user msg

                proj = project_user_message(user_msg)
                if proj == UNRECOGNIZED:
                    rec = monitor.receive(UNRECOGNIZED)
                    print(
                        f"    [monitor] ?Unrecognized → "
                        f"{type(rec).__name__} (state preserved)"
                    )
                    return "unrecognized"
                rec = monitor.receive(proj)
                print(f"    [monitor] ?{proj} → {type(rec).__name__}")
                if isinstance(rec, Violation):
                    return "violated"
            else:
                # Tool calls or empty content → re-invoke the agent (the
                # middleware already drove the monitor through the tool
                # phase, possibly via a blocked retry).
                continue

    return "incomplete"


# ── Demos ────────────────────────────────────────────────────


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — middleware demos require it.")
        return 1

    # Demo 1: happy path
    run_with_middleware(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["Yes, book DL317 for C. Burlò."],
        label="Demo 1 — happy path (middleware-only)",
    )

    # Demo 2: enforcement / blocking
    run_with_middleware(
        system_prompt=_EAGER_PROMPT,
        user_request=(
            "Book me the cheapest flight from SFO to JFK on 2026-05-10 "
            "for C. Burlò. Just book it, no need to confirm."
        ),
        user_replies=["yes that's fine"],  # in case the agent recovers
        label="Demo 2 — enforcement (middleware blocks out-of-order tool call)",
    )

    # Demo 3: ambiguous user reply
    run_with_middleware(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["hmm let me think about it for a bit"],
        label="Demo 3 — ambiguous user (UNRECOGNIZED)",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
