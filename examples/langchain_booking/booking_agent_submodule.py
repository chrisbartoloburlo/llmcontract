"""Booking agent wired up via the canonical ``llmcontract.langchain``
submodule.

This is the production-shape integration as of 0.4.0. Compare with:

  * ``booking_agent.py`` — explicit ``monitor.send/receive`` loop, no
    middleware. Pedagogical baseline.
  * ``booking_agent_middleware.py`` — hand-rolled ``AgentMiddleware``
    subclass against the DSL-based ``llmcontract.Monitor``. Useful for
    understanding what the submodule abstracts away.
  * **this file** — drop-in ``CheckpointedProtocolMiddleware`` from
    ``llmcontract.langchain``. FSM state lives in LangGraph's
    checkpointed ``AgentState``, the user-approval gate is enforced
    via ``langgraph.types.interrupt``, and the rest is standard
    ``create_agent`` boilerplate.

What changed in 0.4.0
─────────────────────

* **State persistence** — FSM state is held in ``AgentState``
  (``fsm_state``, ``fsm_trace``), so it survives ``interrupt()``
  resumes, worker restarts, and multi-pod deployments. The 0.3.x
  middleware stored state as an instance attribute, which broke the
  moment LangGraph rehydrated from a checkpoint.

* **Approval-gate enforcement** — the ``?UserApproval`` transition is
  declared with ``interrupt=True``. The middleware suspends the agent
  via ``langgraph.types.interrupt`` automatically when the FSM enters
  ``presented``; the orchestrator *cannot* forget to gate. Resuming
  with ``Command(resume=...)`` carries the approval back.

* **Tool-call coverage** — ``!SearchFlights``/``?FlightResults`` and
  ``!BookFlight``/``?BookingConfirmation`` still flow through
  ``wrap_tool_call``, exactly as in 0.3.x.

The ``!PresentOptions`` event remains an orchestrator-fired
``transition_event`` for now — switching it to
``match_structured_response`` would require constraining the agent's
text reply via ``response_format``, which is a separate change in
agent setup. The 0.4.0 middleware supports both modes; this example
shows the simpler one.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from booking_agent import (  # noqa: E402
    _load_dotenv,
    book_flight,
    project_user_message,
    search_flights,
)

from llmcontract import UNRECOGNIZED  # noqa: E402
from llmcontract.langchain import (  # noqa: E402
    CheckpointedProtocolMiddleware,
    ProtocolFSM,
    ProtocolViolationError,
    Transition,
    ViolationEvent,
    ref,
)


_load_dotenv(Path(__file__).parent / ".env")


# ── Tool refs and FSM ────────────────────────────────────────

search_ref = ref(search_flights)
book_ref = ref(book_flight)


def _build_fsm() -> ProtocolFSM:
    """Booking protocol as an explicit FSM table.

    The 0.4.0 demo deliberately compresses ``!PresentOptions`` into the
    ``?UserApproval`` interrupt: the agent's text reply listing the
    options is delivered as part of the interrupt payload, and the
    interrupt itself *is* the approval gate. This makes the protocol
    fully framework-enforced — no event in the FSM is fired from the
    orchestrator's side.

    ``UserApproval`` carries ``interrupt=True``: when the FSM enters
    the ``search_done`` state, the middleware pauses the agent on
    ``langgraph.types.interrupt`` before the next model call, and the
    caller drives the approval via ``Command(resume=...)``. The
    orchestrator *cannot* skip this gate — that was the leak in
    0.3.x's "orchestrator must remember to call ``transition_event``"
    pattern.

    For protocols that genuinely need a separate ``!PresentOptions``
    event, the cleanest path is structured output: declare
    ``response_format=PresentOptionsResponse`` on the agent and
    ``match_structured_response=PresentOptionsResponse`` on the
    transition. The middleware then fires it deterministically from
    ``after_model``. See ``test_langchain_middleware.py`` for the
    pattern; not shown in this demo to keep agent setup minimal.
    """
    return (
        ProtocolFSM(initial="idle")
        .add_transition(Transition(source="idle", tool=search_ref, phase="send", target="searching"))
        .add_transition(Transition(source="searching", tool=search_ref, phase="recv", target="search_done"))
        .add_transition(Transition(
            source="search_done", phase="recv", target="approved",
            event_label="UserApproval", interrupt=True,
        ))
        .add_transition(Transition(source="approved", tool=book_ref, phase="send", target="booking"))
        .add_transition(Transition(source="booking", tool=book_ref, phase="recv", target="done"))
        .mark_terminal("done")
    )


def _raise_on_violation(v: ViolationEvent) -> None:
    """Strict enforcement — surface as an exception."""
    label = v.tool_ref.label if v.tool_ref is not None else v.event_label
    raise ProtocolViolationError(
        f"Illegal {v.phase}:{label} from state {v.current_state!r}; "
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

_BOOK_FIRST_PROMPT = (
    "You are a quick booking agent. If the user names a flight (e.g. "
    "DL317), call book_flight directly with that flight_id and the "
    "passenger name. Do not search first — the user has already chosen."
)


def _print_messages(messages: list, printed_until: int) -> int:
    for m in messages[printed_until:]:
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
    return len(messages)


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
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    print(f"\n══════════ {label} ══════════")
    print(f"[user]  {user_request}")

    fsm = _build_fsm()
    middleware = CheckpointedProtocolMiddleware(
        fsm=fsm, on_violation=_raise_on_violation,
        tool_refs=[search_ref, book_ref],
    ).middleware

    # interrupt() requires a checkpointer. InMemorySaver is fine for the
    # demo; production would use SqliteSaver / PostgresSaver / etc.
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=ChatAnthropic(model=model, max_tokens=512),
        tools=[search_flights, book_flight],
        system_prompt=system_prompt,
        middleware=[middleware],
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": label}}
    user_iter = iter(user_replies)
    printed_until = 0
    inputs: Any = {"messages": [HumanMessage(content=user_request)]}

    for _ in range(4):
        try:
            result = agent.invoke(inputs, config=config)
        except ProtocolViolationError as exc:
            print(f"[demo]  PROTOCOL ABORT: {exc}")
            return "violated"

        printed_until = _print_messages(result.get("messages", []), printed_until)

        # interrupt() fired — drain the payload, get user input,
        # resume with the projected event_label.
        if result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            print(f"[demo]  interrupt fired — current_state="
                  f"{payload['current_state']!r}, expected={payload['expected']}")

            try:
                user_msg = next(user_iter)
            except StopIteration:
                print("[demo]  no more user replies; ending")
                return "incomplete"
            print(f"[user]  {user_msg}")

            projection = project_user_message(user_msg)
            if projection == UNRECOGNIZED:
                print("[demo]  user reply UNRECOGNIZED — outer loop "
                      "would clarify. Ending.")
                return "unrecognized"

            # Two channels at once: ``resume`` carries the protocol-
            # level event_label that drives the FSM transition;
            # ``update`` injects the user's natural-language reply
            # into ``messages`` so the model sees it on its next turn.
            # Without the message, the model wouldn't know which
            # flight to book — the protocol-level fact "approved" is
            # not the same as the natural-language fact "DL317".
            inputs = Command(
                update={"messages": [HumanMessage(content=user_msg)]},
                resume={
                    "event_label": projection,
                    "metadata": {"text": user_msg},
                },
            )
            continue

        # No interrupt — agent terminated. Either the protocol
        # reached its terminal state (success) or the agent stopped
        # short of it (incomplete).
        fsm_state = result.get("fsm_state", "?")
        if fsm.is_terminal(fsm_state):
            print(f"[demo]  protocol terminal — fsm_state={fsm_state!r}, "
                  f"trace={result.get('fsm_trace', [])}")
            return "ok"
        break

    print(f"[demo]  loop exhausted; fsm_state={result.get('fsm_state')!r}, "
          f"trace={result.get('fsm_trace')}")
    return "incomplete"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    run_with_submodule(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["Yes, book DL317 for C. Burlò."],
        label="demo-1-happy-path",
    )

    run_with_submodule(
        system_prompt=_BOOK_FIRST_PROMPT,
        user_request="Book DL317 for C. Burlò. SFO to JFK on 2026-05-10.",
        user_replies=[],
        label="demo-2-enforcement",
    )

    run_with_submodule(
        system_prompt=_GOOD_PROMPT,
        user_request="Book me a flight from SFO to JFK on 2026-05-10.",
        user_replies=["hmm let me think about it for a bit"],
        label="demo-3-ambiguous-user",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
