# LangChain + llmcontract: monitored flight booking

Three self-contained scripts wiring an llmcontract monitor into a
LangChain agent that drives the canonical booking protocol from the
project README:

```
!SearchFlights.?FlightResults.!PresentOptions.?UserApproval
.!BookFlight.?BookingConfirmation.end
```

| File | What it shows | When to read |
|---|---|---|
| **`booking_agent.py`** | Explicit orchestration loop with `monitor.send()` / `monitor.receive()` called inline against the DSL-based `Monitor`. Pedagogical — every event fires from a visible line of code. | First, to understand what monitoring an agent looks like end-to-end. |
| **`booking_agent_middleware.py`** | Same protocol, hand-rolled `AgentMiddleware` subclass against the DSL-based `Monitor`. Useful for understanding what the canonical submodule abstracts away, including the `monitor._halted = False` workaround needed before 0.3.0. | Second, to see how an integration looks when written from scratch. |
| **`booking_agent_submodule.py`** | Production shape — uses `llmcontract.langchain.ProtocolEnforcerMiddleware` (shipped in 0.3.0). Tool refs via `ref(fn)`, FSM-as-data with guards/actions per transition, user-controlled `on_violation`. About 40 lines of FSM definition; the rest is `create_agent` boilerplate. | Last. This is what real adopters should copy. |

## `booking_agent.py` — three demos with explicit orchestration

| # | What it shows | Outcome |
|---|---|---|
| **1. Happy path** | Real `ChatAnthropic` agent + LangChain tools. Agent searches, presents options, waits for approval, books. | Monitor stays `Ok`; protocol reaches the terminal state. |
| **2. Skip violation** | Synthetic scripted agent that calls `book_flight` directly after `search_flights`, omitting the `!PresentOptions`/`?UserApproval` steps. Deterministic — runs without an API key. | Monitor returns `Violation` at `!BookFlight`; outer loop halts. |
| **3. Ambiguous user** | Real LLM agent again, but the user replies *"uhhh maybe? not sure yet"*. The user-message projection emits `UNRECOGNIZED`. | Monitor returns `Unrecognized` (state preserved, *not* a violation). The outer loop is expected to drive a clarification turn. |

## `booking_agent_middleware.py` — three demos with LangChain middleware

Same protocol, wired through `AgentMiddleware.wrap_tool_call` instead of an
explicit loop. Three demos:

| # | What it shows | Outcome |
|---|---|---|
| **1. Happy path** | `create_agent` + `ProtocolMonitorMiddleware`. Agent searches, presents, books. | Monitor reaches terminal state. Same result as `booking_agent.py` Demo 1 — different integration point. |
| **2. Enforcement / blocking** | An "eager" system prompt pushes the agent to skip `!PresentOptions` and call `book_flight` directly. The middleware refuses the call by returning an error `ToolMessage`. | Agent sees the refusal and **self-corrects** — emits a presentation, gets the user's approval, then books cleanly. The protocol forced the agent back on track. |
| **3. Ambiguous user** | Same orchestration as the explicit version. | `Unrecognized` from the user-side projection. |

The enforcement demo is the new capability the middleware integration
unlocks: `monitor.send(label)` returning `Violation` becomes a refusal of
the actual action — not just a logged error. Real production agents that
fail-open into Stripe / GitHub / production databases benefit from this
in a way logging alone doesn't deliver.

## Run

```bash
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-…' > .env

# Explicit-loop version (Demo 2 runs without an API key)
python3 booking_agent.py

# Middleware version (all demos require the API key)
python3 booking_agent_middleware.py
```

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

In the **middleware** version (`booking_agent_middleware.py`), the
agent-side events are handled by an `AgentMiddleware` subclass instead:

```python
class ProtocolMonitorMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        send_label, recv_label = TOOL_LABELS[request.tool_call["name"]]
        verdict = monitor.send(send_label)
        if isinstance(verdict, Violation):
            # Refuse the tool call — agent sees the error on its next turn
            return ToolMessage(content="Protocol violation: ...",
                               tool_call_id=request.tool_call["id"],
                               status="error")
        result = handler(request)
        monitor.receive(recv_label)
        return result
```

The user-side events (`?UserApproval`, `UNRECOGNIZED`) stay in the outer
orchestrator because they cross the agent's invocation boundary. That
split — middleware for tool/model events, orchestrator for user events —
is the right factoring for any LangChain agent monitored against a
session-type contract.

### One sharp edge: enforcement-mode monitors

The default `Monitor` halts on `Violation` — subsequent events return
`Blocked`. For a *logging* monitor that's correct. For an *enforcement*
monitor that refuses violating tool calls and lets the agent self-correct,
we need the monitor to stay live. The middleware example does this with
`monitor._halted = False` after a refusal — touching a private flag.

A clean fix lives in `llmcontract` itself: a `Monitor(protocol,
halt_on_violation=False)` constructor flag. Until that lands, the
private-flag dance is the workaround. See the docstring on
`ProtocolMonitorMiddleware.wrap_tool_call` for the rationale.

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

## Empirical study (already in this directory)

The same harness ran across **10 tasks × 3 models × 2 trials = 60
trajectories** (`run_all.py` + `sweep.py`). Headline:

| Model | violation rate |
|---|---:|
| Claude Haiku 4.5 | **15%** |
| Claude Sonnet 4.6 | **10%** |
| Claude Opus 4.7 | **0%** |

The gradient runs *opposite* to the
[Playwright case study](https://github.com/chrisbartoloburlo/llmcontract-playwright-mcp)
— there, larger models took shortcuts; here, smaller models do.
Different protocols, different "skip" semantics, same monitor surfacing
both. Full breakdown with per-task analysis and a sample violating
trajectory in [`reports/findings.md`](reports/findings.md).

## Deriving the protocol from the @tool decorations

The tools, the `TOOL_LABELS` dict, and the `PROTOCOL` DSL string are
*not* three things you maintain in parallel. They're derived from a
single source — each LangChain `@tool` includes a `Protocol response:
<Label>` annotation in its docstring:

```python
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search available flights matching the origin, destination, and date.

    Protocol response: FlightResults

    Args:
        ...
    """
    return _search(origin, destination, date)


@tool
def book_flight(flight_id: str, passenger: str) -> str:
    """Reserve a seat on the chosen flight.

    Protocol response: BookingConfirmation
    """
    return _book(flight_id, passenger)
```

A small `linear_protocol(*events)` helper then stitches the tools and
non-tool events into the DSL string:

```python
PROTOCOL = linear_protocol(
    search_flights,    # !SearchFlights.?FlightResults
    "PresentOptions",  # !PresentOptions  (agent text)
    "?UserApproval",   # ?UserApproval    (user reply, projected)
    book_flight,       # !BookFlight.?BookingConfirmation
)
# → "!SearchFlights.?FlightResults.!PresentOptions.?UserApproval.!BookFlight.?BookingConfirmation.end"
```

Adding a tool means adding one decorator with one new docstring line;
the labels and protocol fall out automatically.

## License

MIT, mirroring the parent project.
