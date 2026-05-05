---
name: protocol-builder
description: Design a session-type protocol for an LLM agent's interaction with its MCP tools, and emit a runtime-checkable DSL string plus a Python integration snippet for the llmsessioncontract monitor. Use when the user asks to design, write, sketch, or generate an interaction protocol / contract / session type for an agent or MCP server, or mentions llmcontract / llmsessioncontract.
---

# Protocol Builder

Walk the user through designing a **session-type protocol** for their LLM agent and emit the protocol DSL string plus a Python integration snippet for the [llmsessioncontract](https://pypi.org/project/llmsessioncontract/) runtime monitor.

A session type is a precise description of the back-and-forth between two parties — exactly the shape of an LLM ↔ MCP-server conversation. Once written down, a runtime monitor can check at every step that the agent is following it.

## DSL syntax (cheat sheet)

| Token          | Meaning                          |
| -------------- | -------------------------------- |
| `!Label`       | Send (agent → world)             |
| `?Label`       | Receive (world → agent)          |
| `!{a, b}`      | Internal choice (sender chooses) |
| `?{a, b}`      | External choice (world chooses)  |
| `.`            | Sequence                         |
| `rec X. ... X` | Recursion (loop back to X)       |
| `end`          | Terminal state                   |

Two examples to anchor on:

- Strict linear: `!SearchFlights.?FlightResults.!BookFlight.?BookingConfirmation.end`
- With branching + recursion: `!CreateCard.?{CardCreated.rec X.!Transaction.?{TransactionOK.X, SessionEnd}, CardError}.end`

## How to run this skill

Be efficient. Don't ask 20 questions — ask broad ones, propose a draft, and iterate.

### 1. Discover the agent shape

Open with: "What does this agent do, and what tools does it call?" Let the user describe it in their own words.

If the user has an MCP config nearby (e.g. `.mcp.json`, `claude_desktop_config.json`, a tool list in code), offer to read it so you can use real tool names rather than placeholders. **Do not** read files unless they invite you to — just ask.

If they're already inside a project that uses the [Inspector Protocol Builder](https://github.com/modelcontextprotocol/inspector/pull/1281), point that out as the visual alternative and continue here only if they want the conversational flow.

### 2. Propose a draft

From their description, draft the simplest plausible protocol — usually a flat `!Tool1.?Tool1Result. ... .end`. Render it inside a fenced block so it's easy to copy. Then ask **one** question at a time:

- "Does the agent ever loop (e.g. retry, multi-turn)?" → wrap a section in `rec X. ... .X`
- "Does the response branch into success / error / multiple shapes?" → use `?{Ok.<rest>, Err.end}`
- "Does the agent itself decide between alternatives?" → `!{Confirm.<rest>, Cancel.end}`

Update the draft after each answer. Show the diff inline so the user sees the protocol grow.

### 3. Validate as you go

Whenever the protocol changes, validate it before showing it:

```bash
python3 "${CLAUDE_PLUGIN_ROOT:-$HOME/.claude/skills/protocol-builder}/scripts/validate.py" '<DSL string>'
```

If `llmcontract` isn't installed, the script will tell you to run `pip install llmsessioncontract`. Pass that on.

If validation fails, show the error verbatim and walk the user through fixing it (usually a missing `end`, an unbalanced `{...}`, or a stray `.`).

### 4. Emit the final artifact

When the user is happy, output:

1. The final DSL string in a code block.
2. A ready-to-paste Python snippet wired up with `Monitor` + `MonitoredClient` + `ToolMiddleware`. Use this template, substituting the protocol:

```python
from llmsessioncontract import Monitor, MonitoredClient, ToolMiddleware, LLMResponse

protocol = "<DSL>"
monitor = Monitor(protocol)

client = MonitoredClient(
    llm_call=your_llm_fn,
    response_adapter=your_adapter,
    monitor=monitor,
    send_label="Request",
    receive_label=lambda r: "ToolCall" if r.has_tool_calls else "FinalAnswer",
)

tools = ToolMiddleware(
    monitor=monitor,
    tools={
        # "tool_name": tool_fn,
    },
)
```

Offer to write it to a file — `protocol.py` in the cwd by default — and ask before doing so. If the user has Langfuse, also mention `LangfuseMonitor` as a drop-in replacement that records each step as a guardrail observation.

## Things to avoid

- Don't invent tool names. If the user hasn't given you specific names, use placeholders (`!ActionA`, `?ActionAResult`) and say so explicitly.
- Don't over-design. A flat send/receive sequence is often the right answer; reach for `rec` and `?{}` only when the user describes loops or branches.
- Don't let the protocol grow unbounded. If it's getting past ~10 steps, ask whether some part should be factored into a recursive scope.
- Don't claim the protocol works at runtime until you've actually run `validate.py` against the latest version.
