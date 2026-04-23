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

## Architecture

```
DSL string ──▶ Parser ──▶ AST ──▶ FSM Compiler ──▶ Automaton ──▶ Monitor
```

- **Parser** (`llmcontract.dsl.parser`) — hand-written recursive descent parser that produces an AST
- **AST** (`llmcontract.dsl.ast`) — frozen dataclasses: `Send`, `Receive`, `InternalChoice`, `ExternalChoice`, `Sequence`, `Recursion`, `RecVar`, `End`
- **FSM Compiler** (`llmcontract.monitor.automaton`) — compiles the AST into a finite state automaton with transitions keyed by `(direction, label)`
- **Monitor** (`llmcontract.monitor.monitor`) — steps through the automaton on each `send`/`receive` call, returning `Ok`, `Violation`, or `Blocked`

## Tests

```bash
pip install -e ".[dev]"
pytest llmcontract/tests/ -v
```

## License

MIT
