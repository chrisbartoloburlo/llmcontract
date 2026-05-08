# llmcontract

A runtime monitor for LLM agent interaction protocols based on session type theory.

`llmcontract` lets you define communication protocols using a concise DSL inspired by session types, then monitor agent interactions at runtime to catch protocol violations the moment they happen.

## Installation

```bash
pip install -e .
```

## Protocol DSL

Protocols are written as strings using this syntax:

| Syntax | Meaning |
|--------|---------|
| `!label` | Send action |
| `?label` | Receive action |
| `!{a, b}` | Internal choice (sender chooses) |
| `?{a, b}` | External choice (receiver chooses) |
| `.` | Sequence |
| `rec X. ...X...` | Recursion |
| `end` | Terminal state |

### Examples

A flight booking protocol — a strict linear sequence:

```
!SearchFlights.?FlightResults.!PresentOptions.?UserApproval.!BookFlight.?BookingConfirmation.end
```

A card payment protocol — with branching and recursion:

```
!CreateCard.?{CardCreated.rec X.!Transaction.?{TransactionOK.X, SessionEnd}, CardError}.end
```

## Usage

```python
from llmcontract import Monitor, Ok, Violation, Blocked

protocol = "!SearchFlights.?FlightResults.!BookFlight.?BookingConfirmation.end"
m = Monitor(protocol)

m.send("SearchFlights")    # Ok()
m.receive("FlightResults") # Ok()
m.send("BookFlight")       # Ok()
m.receive("BookingConfirmation") # Ok()
assert m.is_terminal
```

### Catching violations

```python
m = Monitor("!Ping.?Pong.end")
m.send("Ping")       # Ok()
m.send("Pong")       # Violation(expected=['?Pong'], got='!Pong')
m.send("Anything")   # Blocked('monitor halted after a previous violation')
```

### Working with choices

```python
protocol = "!CreateCard.?{CardCreated.!Done.end, CardError.end}"
m = Monitor(protocol)
m.send("CreateCard")       # Ok()
m.receive("CardError")     # Ok() — the receiver chose this branch
assert m.is_terminal
```

### Recursion

```python
protocol = "rec X.!Ping.?Pong.X"
m = Monitor(protocol)
for _ in range(100):
    m.send("Ping")     # Ok()
    m.receive("Pong")  # Ok()
```

### Handling natural-language input: `Unrecognized`

When the projection layer (typically over user chat) can't classify an event into a known label, it can emit the sentinel `UNRECOGNIZED` instead. The monitor treats this as a soft signal — distinct from `Violation` — without halting or advancing state, so the outer loop can drive a clarification turn:

```python
from llmcontract import Monitor, Ok, Unrecognized, UNRECOGNIZED

m = Monitor("?{Yes.end, No.end}")
result = m.receive(UNRECOGNIZED)         # projection couldn't decide
assert isinstance(result, Unrecognized)  # not a Violation
# state preserved; ask the agent to ask the user to clarify, then:
m.receive("Yes")                         # Ok()
```

A protocol can also handle `Unrecognized` *explicitly* as a first-class branch — useful for "ask again" loops:

```python
protocol = "rec Loop.!Ask.?{Yes.end, No.end, Unrecognized.Loop}"
m = Monitor(protocol)
m.send("Ask")
m.receive(UNRECOGNIZED)  # Ok — protocol routes back to Loop
m.send("Ask")
m.receive("Yes")         # Ok — terminal
```

The distinction matters at the system boundary: `Violation` means the agent broke the rules; `Unrecognized` means we don't have enough information to decide yet. Different responses (halt vs. clarify) come naturally from the typed result.

## Integration Layer

For real agent loops, `llmcontract` provides a client wrapper and tool middleware that share a single monitor — so the full interaction is tracked automatically.

### Client Wrapper

Wraps any LLM client call. Checks `!Send` before calling the LLM and `?Receive` after getting the response. SDK-agnostic — you provide a small adapter function.

```python
from llmcontract import Monitor, MonitoredClient, LLMResponse, ToolCall

monitor = Monitor(
    "rec Loop.!Request.?{ToolCall.!ToolResult.Loop, FinalAnswer.end}"
)

# Adapt your SDK's response to LLMResponse
def adapt(raw):
    if raw.tool_calls:
        return LLMResponse(tool_calls=[
            ToolCall(name=tc.function.name, arguments=tc.arguments, id=tc.id)
            for tc in raw.tool_calls
        ])
    return LLMResponse(content=raw.content)

client = MonitoredClient(
    llm_call=openai.chat.completions.create,
    response_adapter=adapt,
    monitor=monitor,
    send_label="Request",
    receive_label=lambda r: "ToolCall" if r.has_tool_calls else "FinalAnswer",
)

response = client.call(model="gpt-4", messages=[...])
# Automatically fires !Request then ?ToolCall or ?FinalAnswer
```

### Tool Middleware

Wraps tool execution. When the LLM requests a tool, the middleware checks `?Receive` (tool requested) and `!Send` (result returned) against the protocol.

```python
from llmcontract import ToolMiddleware

middleware = ToolMiddleware(
    monitor=monitor,  # same monitor as the client
    tools={
        "search": search_fn,
        "book": book_fn,
    },
)

# Process all tool calls from a response
results = middleware.process(response)
# Each tool call checks ?receive and !send against the protocol
```

### Combined Agent Loop

```python
from llmcontract import (
    Monitor, MonitoredClient, ToolMiddleware,
    LLMResponse, ToolCall, ProtocolViolationError,
)

protocol = "rec Loop.!Request.?{ToolCall.!ToolResult.Loop, FinalAnswer.end}"
monitor = Monitor(protocol)

client = MonitoredClient(
    llm_call=my_llm_fn,
    response_adapter=my_adapter,
    monitor=monitor,
    send_label="Request",
    receive_label=lambda r: "ToolCall" if r.has_tool_calls else "FinalAnswer",
)

while True:
    try:
        response = client.call(messages=messages)
    except ProtocolViolationError as e:
        print(f"Protocol violated: {e}")
        break

    if not response.has_tool_calls:
        break  # FinalAnswer — protocol complete

    # Execute tools, send results back
    for tc in response.tool_calls:
        result = tools[tc.name](**tc.arguments)
        monitor.send("ToolResult")  # record the send
        messages.append(tool_result_msg(tc.id, result))
```

## LangChain Integration (`llmcontract.langchain`, 0.3.0+)

A focused FSM-as-data API for users who want to wire protocol monitoring
into LangChain agents without touching the DSL parser. Tool references
are real Python callables, transitions are explicit objects with
optional guards, actions, and approval-gate interrupts, and violation
handling is fully user-controlled.

```bash
pip install llmsessioncontract[langchain]
```

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from llmcontract.langchain import (
    ProtocolFSM, Transition,
    CheckpointedProtocolMiddleware, ViolationEvent,
    ProtocolViolationError, ref,
)

@tool
def search(query: str) -> str:
    """Search for available flights."""
    return f"Results for: {query}"

@tool
def book(result: str) -> str:
    """Book a selected flight."""
    return f"Booked: {result}"

search_ref = ref(search)
book_ref = ref(book)

fsm = (
    ProtocolFSM(initial="idle")
    .add_transition(Transition(source="idle", tool=search_ref, phase="send", target="searching"))
    .add_transition(Transition(source="searching", tool=search_ref, phase="recv", target="results"))
    .add_transition(Transition(
        source="results", phase="recv", target="approved",
        event_label="UserApproval", interrupt=True,   # framework-enforced gate
    ))
    .add_transition(Transition(source="approved", tool=book_ref, phase="send", target="booking"))
    .add_transition(Transition(source="booking", tool=book_ref, phase="recv", target="done"))
    .mark_terminal("done")
)

def on_violation(v: ViolationEvent) -> None:
    label = v.tool_ref.label if v.tool_ref else v.event_label
    raise ProtocolViolationError(f"Illegal {v.phase}:{label} from {v.current_state!r}", violation=v)

middleware = CheckpointedProtocolMiddleware(
    fsm=fsm, on_violation=on_violation, tool_refs=[search_ref, book_ref],
).middleware

# interrupt-gated transitions require a checkpointer.
agent = create_agent(
    model=..., tools=[search, book], middleware=[middleware],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "demo-1"}}
result = agent.invoke({"messages": [("user", "Book me a flight to Rome.")]}, config=config)

# When the FSM enters `results`, the middleware suspends on `interrupt()`.
# The result carries an `__interrupt__` payload with the protocol context.
if result.get("__interrupt__"):
    result = agent.invoke(
        Command(resume={"event_label": "UserApproval"}),
        config=config,
    )

print(result["fsm_state"])     # → "done"
print(result["fsm_trace"])     # → ["send:search", "recv:search",
                               #    "recv:UserApproval",
                               #    "send:book", "recv:book"]
```

### Non-tool events: three firing surfaces (0.4.0+)

`Transition` accepts either a `tool: ToolRef` (fired automatically by
the middleware on tool calls) or an `event_label: str` (for free-form
events — agent text replies, user replies, timeouts, etc.).
Event-label transitions support three firing modes:

1. **Interrupt-gated** (`interrupt=True`) — the middleware suspends
   the agent on `langgraph.types.interrupt(...)` from the source
   state and fires the transition on `Command(resume=...)`. The
   gate is framework-enforced; the orchestrator *cannot* skip it.
   Best for human-approval steps on irreversible actions.

2. **Structured-response matched** (`match_structured_response=Type`) —
   the middleware fires the transition deterministically in
   `after_model` when `state["structured_response"]` is an instance
   of the given type. Use with `response_format=Type` on the agent.
   Best for non-tool agent events that need to be reliably detected.

3. **Orchestrator-fired** — the orchestrator (or any non-LangChain
   loop) calls `monitor.transition_event(label, phase, metadata)` on
   a `ProtocolMonitor`. Best for events the framework can't see —
   e.g., natural-language user replies in a custom orchestration loop.

The library never interprets natural language; the orchestrator owns
the projection from raw input to protocol-level event labels.

When to pick this over the DSL `Monitor`:

- You're already in a LangChain stack and want a drop-in `AgentMiddleware`
- You need per-transition guards and actions (e.g., audit logs, business rules)
- You want enforcement (block tool calls), not just observation
- You don't need recursion / choice / `Unrecognized` from the DSL

When to stick with the DSL `Monitor`:

- You want to write protocols as concise strings (`!Search.?Result.end`)
- You need recursion or compositional choice
- You're outside LangChain (Anthropic SDK, OpenAI SDK, custom loop)
- You want first-class natural-language ambiguity via `Unrecognized`

Worked example: [`examples/langchain_booking/booking_agent_submodule.py`](examples/langchain_booking/booking_agent_submodule.py).

## Langfuse Integration

Track protocol compliance in [Langfuse](https://langfuse.com) — every send/receive is recorded as a guardrail observation with a pass/fail score.

```bash
pip install llmsessioncontract[langfuse]
```

```python
from langfuse import get_client
from llmcontract.integration.langfuse import LangfuseMonitor

langfuse = get_client()

with langfuse.start_as_current_observation(name="agent-run") as trace:
    monitor = LangfuseMonitor(
        protocol="!Request.?{ToolCall.!ToolResult.end, FinalAnswer.end}",
        langfuse=langfuse,
    )

    monitor.send("Request")       # guardrail: ok ✓
    monitor.receive("ToolCall")   # guardrail: ok ✓
    monitor.send("ToolResult")    # guardrail: ok ✓
    monitor.send("ExtraCall")     # guardrail: VIOLATION ✗

langfuse.flush()
```

Each step appears as a guardrail observation in your Langfuse trace with:
- **Input**: the action attempted, direction, label, protocol
- **Output**: `passed: true/false`, violation details if applicable
- **Score**: `protocol_compliance` (boolean) for filtering and analytics

## Claude Code Plugin

A Claude Code plugin ships with this repo: **protocol-builder** walks you through designing a session-type protocol conversationally, validates it as you go, and emits a ready-to-paste Python integration snippet.

```bash
# Install in Claude Code
/plugin marketplace add chrisbartoloburlo/llmcontract
/plugin install protocol-builder@llmcontract

# Then in any conversation
/protocol-builder
```

The skill validates each draft DSL against `llmcontract`'s parser, so anything it produces is guaranteed to load with `Monitor(...)`. Source lives under `skills/protocol-builder/`.

## Case Studies

- **[`llmcontract-tau2`](https://github.com/chrisbartoloburlo/llmcontract-tau2)** — *user ↔ agent layer.* Standalone replay of [tau2-bench](https://github.com/sierra-research/tau2-bench)'s shipped trajectories through `Monitor`. Headline: 11/1755 (0.6%) of trajectories that tau2 scored as passing violate the documented "obtain user confirmation before mutating the database" policy. Discussion upstream: [tau2-bench#298](https://github.com/sierra-research/tau2-bench/issues/298).
- **[`llmcontract-playwright-mcp`](https://github.com/chrisbartoloburlo/llmcontract-playwright-mcp)** — *agent ↔ tool layer.* 90-trajectory sweep across Claude Haiku 4.5 / Sonnet 4.6 / Opus 4.7 driving [`@playwright/mcp`](https://github.com/microsoft/playwright-mcp), checked against two invariants from the server's documented usage. Headline: 9% violate `snap-before-interact`, 29% violate `stay-on-snapshot-refs` — and the failure modes scale opposite directions with model capability (Haiku snapshots religiously but ignores the snapshot 57% of the time; Opus skips the snapshot 13% of the time but follows through cleanly when it commits).

## Research

This work is based on the theory developed in:

> Christian Bartolo Burlò, Adrian Francalanza, Alceste Scalas. **"On the Monitorability of Session Types, in Theory and in Practice"**. *35th European Conference on Object-Oriented Programming (ECOOP 2021)*, pp. 20:1–20:30, Schloss Dagstuhl, 2021.
> [[PDF]](https://orbit.dtu.dk/files/261948257/LIPIcs_ECOOP_2021_20.pdf) [[Google Scholar]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=oxv-4o8AAAAJ&citation_for_view=oxv-4o8AAAAJ:9yKSN-GCB0IC)

## Architecture

```
DSL string ──▶ Parser ──▶ AST ──▶ FSM Compiler ──▶ Automaton ──▶ Monitor
```

- **Parser** (`llmcontract.dsl.parser`) — hand-written recursive descent parser that produces an AST
- **AST** (`llmcontract.dsl.ast`) — frozen dataclasses: `Send`, `Receive`, `InternalChoice`, `ExternalChoice`, `Sequence`, `Recursion`, `RecVar`, `End`
- **FSM Compiler** (`llmcontract.monitor.automaton`) — compiles the AST into a finite state automaton with transitions keyed by `(direction, label)`
- **Monitor** (`llmcontract.monitor.monitor`) — steps through the automaton on each `send`/`receive` call, returning `Ok`, `Violation`, or `Blocked`
- **MonitoredClient** (`llmcontract.integration.client`) — wraps any LLM client call with automatic protocol checks
- **ToolMiddleware** (`llmcontract.integration.middleware`) — intercepts tool execution with protocol checks
- **LangfuseMonitor** (`llmcontract.integration.langfuse`) — records protocol events as Langfuse guardrail observations

## Tests

```bash
pip install -e ".[dev]"
pytest llmcontract/tests/ -v
```

## License

MIT
