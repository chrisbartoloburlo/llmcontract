"""LangChain + llmcontract: monitor a flight-booking agent at runtime.

Encodes the canonical booking protocol from the README

    !SearchFlights.?FlightResults.!PresentOptions.?UserApproval
    .!BookFlight.?BookingConfirmation.end

and drives it three ways:

  1. Happy path — a real Anthropic LLM agent (LangChain ChatAnthropic +
     LangChain tools) follows the protocol cleanly. Monitor stays Ok.

  2. Skip-presentation violation — a deterministic scripted agent that
     calls book_flight directly after search_flights, omitting the
     ?UserApproval step. Monitor returns Violation; outer loop halts.

  3. Ambiguous user — real LLM agent again, but the user replies with an
     ambiguous "uhh maybe". The user-message projection emits UNRECOGNIZED;
     monitor preserves state and signals the outer loop to clarify rather
     than treating the trajectory as failed.

Run with `ANTHROPIC_API_KEY` exported. Demo 2 doesn't need the API key.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

from llmcontract import (
    Monitor,
    Ok,
    Unrecognized,
    UNRECOGNIZED,
    Violation,
)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Lightweight `.env` loader — no python-dotenv dep. Reads `KEY=value`
    pairs and pushes them into ``os.environ`` only if not already set, so an
    explicit shell ``export`` still wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent / ".env")


# ── Protocol ─────────────────────────────────────────────────

PROTOCOL = (
    "!SearchFlights.?FlightResults.!PresentOptions.?UserApproval"
    ".!BookFlight.?BookingConfirmation.end"
)

# Tool name → (send-label fired when the agent calls it,
#              receive-label fired when the result comes back).
TOOL_LABELS = {
    "search_flights": ("SearchFlights", "FlightResults"),
    "book_flight": ("BookFlight", "BookingConfirmation"),
}


# ── User-side projection ─────────────────────────────────────

# The protocol's `?UserApproval` is satisfied by an explicit affirmative.
# Everything else (empty turn, "I'm not sure", "let me think") is
# UNRECOGNIZED — the projection cannot resolve consent and asks the
# outer loop to drive a clarification turn.
#
# Lessons from the tau2 case study: "sure" and "please" as standalone
# tokens produce false positives ("not sure", "please don't"); we drop
# them here in favour of higher-precision patterns. Better still would
# be an LLM-as-judge for ambiguous cases — see the case-study repos.
_APPROVE = re.compile(
    r"\b(yes|yeah|yep|okay|book\s+it|confirm|go\s+ahead|sounds\s+good)\b",
    re.IGNORECASE,
)


def project_user_message(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return UNRECOGNIZED
    return "UserApproval" if _APPROVE.search(text) else UNRECOGNIZED


# ── Mock booking system (the "tool" backend) ─────────────────

FLIGHTS = {
    ("SFO", "JFK", "2026-05-10"): [
        {"id": "AA101", "depart": "08:00", "arrive": "16:30", "price": 380},
        {"id": "UA205", "depart": "11:00", "arrive": "19:50", "price": 420},
        {"id": "DL317", "depart": "21:00", "arrive": "05:30+1", "price": 295},
    ],
}


def _search(origin: str, destination: str, date: str) -> str:
    key = (origin.upper(), destination.upper(), date)
    flights = FLIGHTS.get(key, [])
    if not flights:
        return f"No flights found for {origin}→{destination} on {date}."
    return "\n".join(
        f"{f['id']}: depart {f['depart']}, arrive {f['arrive']}, ${f['price']}"
        for f in flights
    )


def _book(flight_id: str, passenger: str) -> str:
    return f"Reserved {flight_id} for {passenger}. Confirmation: BK-{flight_id}-7421"


# ── Monitor wiring helpers ───────────────────────────────────


def _step(monitor: Monitor, kind: str, label: str, *, fail_on_violation: bool = True) -> bool:
    """Fire a monitor event, print the result, return True if the trajectory
    should continue. ``kind`` is "send" or "receive"."""
    if kind == "send":
        result = monitor.send(label)
        sym = "!"
    else:
        result = monitor.receive(label)
        sym = "?"
    if isinstance(result, Ok):
        print(f"    [monitor] {sym}{label} → Ok")
        return True
    if isinstance(result, Unrecognized):
        print(
            f"    [monitor] {sym}{label} → Unrecognized "
            f"(state preserved, outer loop should clarify)"
        )
        return False
    if isinstance(result, Violation):
        print(
            f"    [monitor] {sym}{label} → Violation "
            f"(expected={result.expected}, got={result.got})"
        )
        return not fail_on_violation
    print(f"    [monitor] {sym}{label} → {type(result).__name__}: {result}")
    return False


# ── LangChain-based real agent ───────────────────────────────


def _build_lc_tools():
    """Define the LangChain tools that the LLM will call.

    Imported lazily so demo 2 (which doesn't need the API or LangChain)
    can run without those dependencies installed.
    """
    from langchain_core.tools import tool

    @tool
    def search_flights(origin: str, destination: str, date: str) -> str:
        """Search available flights matching the origin, destination, and date.

        Args:
            origin: 3-letter airport code, e.g. "SFO"
            destination: 3-letter airport code, e.g. "JFK"
            date: ISO date, e.g. "2026-05-10"
        """
        return _search(origin, destination, date)

    @tool
    def book_flight(flight_id: str, passenger: str) -> str:
        """Reserve a seat on the chosen flight."""
        return _book(flight_id, passenger)

    return [search_flights, book_flight]


def run_real_agent(
    system_prompt: str,
    user_request: str,
    user_replies: Iterable[str],
    *,
    max_turns: int = 8,
) -> str:
    """Drive a LangChain ChatAnthropic agent through the booking protocol.

    Returns one of: "ok" (terminal state reached), "violated", "unrecognized",
    "incomplete" (ran out of turns).
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    tools = _build_lc_tools()
    tools_by_name = {t.name: t for t in tools}
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512).bind_tools(
        tools
    )

    monitor = Monitor(PROTOCOL)
    messages: list = [SystemMessage(content=system_prompt), HumanMessage(content=user_request)]
    print(f"[user]  {user_request}")
    user_iter: Iterator[str] = iter(user_replies)

    for turn in range(max_turns):
        response: AIMessage = llm.invoke(messages)
        messages.append(response)

        # Tool calls fire `!ToolName.?ToolNameResult` pairs.
        if response.tool_calls:
            for tc in response.tool_calls:
                send_label, recv_label = TOOL_LABELS.get(
                    tc["name"], (tc["name"], f"{tc['name']}Result")
                )
                print(f"[agent] tool: {tc['name']}({tc['args']})")
                if not _step(monitor, "send", send_label):
                    return "violated"
                output = tools_by_name[tc["name"]].invoke(tc["args"])
                print(f"[tool]  {output}")
                if not _step(monitor, "receive", recv_label):
                    return "violated"
                messages.append(
                    ToolMessage(content=str(output), tool_call_id=tc["id"])
                )
            continue  # keep going until we get a text response

        # Text-only response → the agent presented options to the user.
        text = (response.content or "").strip() if isinstance(response.content, str) else ""
        if not text:
            # Some models emit empty text + no tool calls. Treat as end.
            break
        print(f"[agent] {text}")

        if monitor.is_terminal:
            # Final wrap-up message after BookingConfirmation.
            return "ok"

        if not _step(monitor, "send", "PresentOptions"):
            return "violated"

        try:
            user_msg = next(user_iter)
        except StopIteration:
            print("[demo]  no more user replies; ending")
            break
        print(f"[user]  {user_msg}")
        messages.append(HumanMessage(content=user_msg))

        label = project_user_message(user_msg)
        # `?UserApproval` advances state; UNRECOGNIZED preserves it and
        # signals "ask for clarification".
        if label == UNRECOGNIZED:
            _step(monitor, "receive", UNRECOGNIZED)
            return "unrecognized"
        if not _step(monitor, "receive", label):
            return "violated"

    return "incomplete"


# ── Synthetic agent for the deterministic skip demo ──────────


def run_skipping_agent() -> str:
    """A scripted agent that calls book_flight directly after search_flights,
    deliberately skipping the !PresentOptions and ?UserApproval steps so the
    monitor's catch is visible without depending on LLM behaviour."""

    monitor = Monitor(PROTOCOL)

    # The user makes a request — but we deliberately skip the conversation
    # part and go straight from search → book.
    print("[user]  Book me the cheapest flight from SFO to JFK on 2026-05-10. Just book it.")
    print("[demo]  (synthetic agent: skipping !PresentOptions/?UserApproval)")

    print("[agent] tool: search_flights({'origin': 'SFO', 'destination': 'JFK', 'date': '2026-05-10'})")
    if not _step(monitor, "send", "SearchFlights"):
        return "violated"
    print(f"[tool]  {_search('SFO', 'JFK', '2026-05-10')}")
    if not _step(monitor, "receive", "FlightResults"):
        return "violated"

    # Skip ahead to the booking call without presenting options or getting
    # approval — the monitor should reject this.
    print("[agent] tool: book_flight({'flight_id': 'DL317', 'passenger': 'C. Burlò'})")
    if not _step(monitor, "send", "BookFlight"):
        return "violated"

    return "ok"


# ── Demos ────────────────────────────────────────────────────


_GOOD_PROMPT = (
    "You are a flight-booking assistant. Use search_flights to find options, "
    "present the options to the user as a short text reply (no tool call), "
    "wait for the user to approve a specific flight, then call book_flight "
    "to reserve it. Be concise."
)


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Demo 2 (synthetic violation) will run; "
            "demos 1 and 3 require an Anthropic API key.\n",
            file=sys.stderr,
        )

    print("\n══════════ Demo 1 — happy path (real LangChain agent) ══════════")
    if os.environ.get("ANTHROPIC_API_KEY"):
        outcome = run_real_agent(
            system_prompt=_GOOD_PROMPT,
            user_request="Book me a flight from SFO to JFK on 2026-05-10.",
            user_replies=["Yes, book DL317 for me — passenger C. Burlò."],
        )
        print(f"[demo]  outcome: {outcome}")
    else:
        print("(skipped — no ANTHROPIC_API_KEY)")

    print("\n══════════ Demo 2 — skip violation (synthetic) ══════════")
    outcome = run_skipping_agent()
    print(f"[demo]  outcome: {outcome}")

    print("\n══════════ Demo 3 — ambiguous user reply ══════════")
    if os.environ.get("ANTHROPIC_API_KEY"):
        outcome = run_real_agent(
            system_prompt=_GOOD_PROMPT,
            user_request="Book me a flight from SFO to JFK on 2026-05-10.",
            user_replies=["hmm let me think about it for a bit"],
        )
        print(f"[demo]  outcome: {outcome}")
    else:
        print("(skipped — no ANTHROPIC_API_KEY)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
