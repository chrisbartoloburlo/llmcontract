# LangChain + llmcontract: monitored flight booking

A self-contained example wiring an llmcontract `Monitor` into a LangChain
agent that drives the canonical booking protocol from the project README:

```
!SearchFlights.?FlightResults.!PresentOptions.?UserApproval
.!BookFlight.?BookingConfirmation.end
```

Three demos in one script (`booking_agent.py`):

| # | What it shows | Outcome |
|---|---|---|
| **1. Happy path** | Real `ChatAnthropic` agent + LangChain tools. Agent searches, presents options, waits for approval, books. | Monitor stays `Ok`; protocol reaches the terminal state. |
| **2. Skip violation** | Synthetic scripted agent that calls `book_flight` directly after `search_flights`, omitting the `!PresentOptions`/`?UserApproval` steps. Deterministic — runs without an API key. | Monitor returns `Violation` at `!BookFlight`; outer loop halts. |
| **3. Ambiguous user** | Real LLM agent again, but the user replies *"uhhh maybe? not sure yet"*. The user-message projection emits `UNRECOGNIZED`. | Monitor returns `Unrecognized` (state preserved, *not* a violation). The outer loop is expected to drive a clarification turn. |

## Run

```bash
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-…' > .env  # demos 1 and 3 only
python3 booking_agent.py
```

Demo 2 runs without the API key, so you can see the violation-catching
behaviour even before signing up for Anthropic.

## What the integration looks like

The script projects three event sources into the protocol's label alphabet:

| Event source | Projected label |
|---|---|
| Agent calls `search_flights` tool | `!SearchFlights` |
| Tool returns search results | `?FlightResults` |
| Agent emits a text reply (no tool call) | `!PresentOptions` |
| User reply matches the `_APPROVE` regex | `?UserApproval` |
| User reply is empty or doesn't match | `UNRECOGNIZED` |
| Agent calls `book_flight` tool | `!BookFlight` |
| Tool returns booking confirmation | `?BookingConfirmation` |

The `_step(monitor, "send"|"receive", label)` helper fires the event,
prints the result, and tells the caller whether to continue. That's the
entire integration — no callbacks, no decorators, no framework magic.
The pattern transports cleanly to any other agent loop (LangGraph,
AutoGen, CrewAI, plain Anthropic SDK, custom).

## Why this is interesting

Most LLM agent frameworks let you specify *what tools* the agent has,
but not *what order it must call them in* or *what other events must
happen between calls*. The protocol-and-monitor pattern fills that gap:
the contract lives outside the agent's prompt, gets enforced at runtime,
and produces auditable evidence (every `!`/`?` event in order) of
whether the agent followed it.

The `UNRECOGNIZED` path is the part that wouldn't be obvious from the
protocol DSL alone — it lets the projection layer say *"I can't
classify this user reply"* without the monitor either accepting (which
would be a soundness violation) or rejecting (which would be a false
positive). That distinction is what the [`Unrecognized`
result](https://github.com/chrisbartoloburlo/llmcontract/blob/main/llmcontract/monitor/monitor.py)
adds in `llmsessioncontract>=0.2.2`.

## Next: turn this into an empirical case study

The same harness — same tools, same protocol, same projection — scales
into a measurement study by varying tasks, models, and trial counts and
counting violations. See the
[`llmcontract-playwright-mcp`](https://github.com/chrisbartoloburlo/llmcontract-playwright-mcp)
case study for the pattern. A booking-domain version of that would
encode multiple booking protocols (modify, cancel, refund, multi-leg)
and measure how often LangChain agents respect each one.

## License

MIT, mirroring the parent project.
