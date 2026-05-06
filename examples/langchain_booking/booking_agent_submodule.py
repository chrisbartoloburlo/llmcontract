"""Booking agent wired up via the canonical ``llmcontract.langchain``
submodule (shipped in 0.3.0).

This is the production-shape integration. Compare with:

  * ``booking_agent.py`` — explicit ``monitor.send/receive`` loop, no
    middleware. Pedagogical baseline.
  * ``booking_agent_middleware.py`` — hand-rolled ``AgentMiddleware``
    subclass against the DSL-based ``llmcontract.Monitor``. Useful for
    understanding what the submodule abstracts away.
  * **this file** — drop-in ``ProtocolEnforcerMiddleware`` from
    ``llmcontract.langchain``. ~40 lines of FSM definition and the rest
    is standard ``create_agent`` boilerplate.

The FSM here is the tool-call subset of the canonical booking protocol —
``!PresentOptions`` and ``?UserApproval`` aren't tool-backed events, so
they're handled at the orchestrator boundary between agent invocations.
This split (middleware for tool events, orchestrator for user events)
is the right factoring for any agent monitored against a session-type
contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from booking_agent import (  # noqa: E402
    _load_dotenv,
    book_flight,
    project_user_message,
    search_flights,
)

from llmcontract import UNRECOGNIZED  # noqa: E402
from llmcontract.langchain import (  # noqa: E402
    ProtocolEnforcerMiddleware,
    ProtocolFSM,
    ProtocolMonitor,
    ProtocolViolationError,
    Transition,
    ViolationEvent,
    ref,
)


_load_dotenv(Path(__file__).parent / ".env")


# ── Tool refs and FSM ────────────────────────────────────────

# Tool refs derive their stable label from the @tool callables — never
# from a string. Adding a new tool means adding one decorator and one
# `ref(fn)` line; no string lookups anywhere in the user-facing API.
search_ref = ref(search_flights)
book_ref = ref(book_flight)


def _build_fsm() -> ProtocolFSM:
    """The canonical booking protocol's tool-call subset, as an explicit
    FSM table. The non-tool events ``!PresentOptions`` and
    ``?UserApproval`` are handled by the orchestrator, not the FSM.

    The FSM is rebuilt per session because ``ProtocolFSM`` itself is
    immutable but ``ProtocolMonitor`` (which holds runtime state)
    expects to be constructed fresh per agent invocation. We could
    share one FSM across monitors — it's stateless after construction —
    but the per-session pattern keeps the wiring uniform with the rest
    of the example.
    """
    return (
        ProtocolFSM(initial="idle")
        .add_transition(Transition(source="idle", tool=search_ref, phase="send", target="searching"))
        .add_transition(Transition(source="searching", tool=search_ref, phase="recv", target="search_done"))
        .add_transition(Transition(source="search_done", tool=book_ref, phase="send", target="booking"))
        .add_transition(Transition(source="booking", tool=book_ref, phase="recv", target="done"))
        .mark_terminal("done")
    )


def _raise_on_violation(v: ViolationEvent) -> None:
    """Strict enforcement — surface as an exception. The library never
    raises itself; we choose to here because the booking flow has no
    way to recover from an out-of-order tool call mid-session.

    A self-correcting alternative is to return normally from this
    handler and have ``wrap_tool_call`` surface a ``ToolMessage`` with
    ``status="error"`` so the agent sees the refusal on its next turn —
    that's the pattern the hand-rolled ``booking_agent_middleware.py``
    demonstrates. Pick whichever fits your operational stance.
    """
    raise ProtocolViolationError(
        f"Illegal {v.phase}:{v.tool_ref.label} from state {v.current_state!r}; "
        f"expected one of {v.expected}; trace={v.trace}",
        violation=v,
    )


# ── Orchestrator that wraps create_agent + the canonical middleware ──


_GOOD_PROMPT = (
    "You are a flight-booking assistant. Use the search_flights tool to "
    "find options, present the options to the user as a short text reply "
    "(no tool call), wait for the user to approve a specific flight, then "
    "call the book_flight tool to reserve it. Be concise."
)

# Pushes the agent to call ``book_flight`` *first*, before ``search``.
# Matches what an FSM keyed on tool-call ordering can actually catch:
# the FSM has no notion of "agent text responses", so we can't enforce
# ``!PresentOptions`` from this side. The DSL-based examples in this
# folder catch text-step violations because the DSL models them; the
# submodule's FSM is strictly about which tool fires when.
_BOOK_FIRST_PROMPT = (
    "You are a quick booking agent. If the user names a flight (e.g. "
    "DL317), call book_flight directly with that flight_id and the "
    "passenger name. Do not search first — the user has already chosen."
)


def run_with_submodule(
    *,
    system_prompt: str,
    user_request: str,
    user_replies: list[str],
    label: str,
    model: str = "claude-haiku-4-5-20251001",
) -> str:
    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    print(f"\n══════════ {label} ══════════")
    print(f"[user]  {user_request}")

    monitor = ProtocolMonitor(fsm=_build_fsm(), on_violation=_raise_on_violation)
    middleware = ProtocolEnforcerMiddleware(
        monitor=monitor, tool_refs=[search_ref, book_ref]
    ).middleware

    agent = create_agent(
        model=ChatAnthropic(model=model, max_tokens=512),
        tools=[search_flights, book_flight],
        system_prompt=system_prompt,
        middleware=[middleware],
    )

    messages: list = [HumanMessage(content=user_request)]
    user_iter = iter(user_replies)
    printed_until = len(messages)

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
                print(f"[tool]  {content[:150]}")
        printed_until = len(msgs)

    for _ in range(4):
        try:
            result = agent.invoke({"messages": messages})
        except ProtocolViolationError as exc:
            print(f"[demo]  PROTOCOL ABORT: {exc}")
            return "violated"
        messages = result["messages"]
        _print_new(messages)

        last = messages[-1]
        if getattr(last, "type", None) != "ai":
            break
        tool_calls = getattr(last, "tool_calls", None)
        text = (last.content or "") if isinstance(last.content, str) else ""

        if not tool_calls and text.strip():
            if monitor.is_complete():
                print(f"[demo]  protocol terminal — state={monitor.state}, "
                      f"trace={monitor.trace}")
                return "ok"
            try:
                user_msg = next(user_iter)
            except StopIteration:
                print("[demo]  no more user replies; ending")
                return "incomplete"
            print(f"[user]  {user_msg}")
            messages.append(HumanMessage(content=user_msg))
            printed_until = len(messages)

            if project_user_message(user_msg) == UNRECOGNIZED:
                print(
                    "[demo]  user reply UNRECOGNIZED — outer loop would "
                    "ask user to clarify. Ending."
                )
                return "unrecognized"
        else:
            continue

    return "incomplete"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    run_with_submodule(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["Yes, book DL317 for C. Burlò."],
        label="Demo 1 — happy path (llmcontract.langchain submodule)",
    )

    run_with_submodule(
        system_prompt=_BOOK_FIRST_PROMPT,
        user_request=(
            "Book DL317 for C. Burlò. SFO to JFK on 2026-05-10."
        ),
        user_replies=[],
        label="Demo 2 — enforcement (book_flight before search_flights)",
    )

    run_with_submodule(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["hmm let me think about it for a bit"],
        label="Demo 3 — ambiguous user reply",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
