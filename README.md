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
