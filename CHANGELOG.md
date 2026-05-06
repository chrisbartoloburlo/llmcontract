# Changelog

## 0.3.1 — 2026-05-04

### Added

- **Mixed transitions in `llmcontract.langchain`** — `Transition` now
  accepts either a `tool: ToolRef` (for tool-backed events, fired
  through the middleware) **or** an `event_label: str` (for free-form
  events, fired explicitly by the orchestrator via the new
  `ProtocolMonitor.transition_event(event_label, phase, metadata)`
  method). The two are mutually exclusive — a `Transition` constructor
  raises `ValueError` if both or neither are supplied.

  This closes the expressivity gap between the FSM-as-data API and the
  DSL-based `Monitor`: protocols mixing tool calls with non-tool events
  (agent text replies, projected user replies, timeouts, system
  signals) can now be expressed entirely as a single FSM. `MonitorContext`
  and `ViolationEvent` gain an `event_label: str | None` field so guards,
  actions, and violation handlers can branch on which firing surface
  triggered them.

  Backwards compatible — the existing `tool=...` keyword and
  `monitor.transition(...)` API are unchanged. Existing FSM definitions
  keep working without modification.

  The `examples/langchain_booking/booking_agent_submodule.py` example
  is updated to model the full canonical booking protocol end-to-end:
  `!SearchFlights`/`?FlightResults` and `!BookFlight`/`?BookingConfirmation`
  via the middleware, `!PresentOptions` and `?UserApproval` via
  `transition_event` from the orchestrator.

  Adds 8 tests covering construction validation, the new
  `transition_event` method, and an end-to-end mixed protocol. Total
  test count: 120.

## 0.3.0 — 2026-05-06

### Added

- **`llmcontract.langchain` submodule** — a focused, FSM-as-data API for
  users who want to wire protocol monitoring into LangChain agents
  without writing or parsing a DSL string. Tool references are real
  Python callables (`ref(search)`, never `"search"`); transitions are
  explicit `Transition` objects with optional `guard` and `action`
  callbacks; violation handling is fully user-controlled via an
  `on_violation: Callable[[ViolationEvent], None]` hook.

  Public API:

  - `ToolRef` / `ref()` — stable, hashable references derived from
    `tool.name` (or `__name__` for plain callables), no magic strings
  - `ProtocolFSM` — pure FSM definition built fluently via
    `add_transition(...)` / `mark_terminal(...)`, no LangChain imports
  - `Transition`, `MonitorContext`, `ViolationEvent` — dataclasses for
    the FSM data model
  - `ProtocolMonitor` — stateful runner that wraps an FSM, calls the
    user's `on_violation` handler when transitions are rejected
  - `ProtocolEnforcerMiddleware` — `AgentMiddleware` subclass exposing
    `wrap_tool_call` / `awrap_tool_call`, drops in via
    `create_agent(middleware=[...])`
  - `ProtocolViolationError` — convenience exception for handlers that
    want to raise

  Optional dependency: install with `pip install
  llmsessioncontract[langchain]`. The FSM and monitor modules
  themselves have zero LangChain imports, so they're testable in
  isolation.

  Validated against the canonical booking example from the technical
  spec — six-state FSM with guard + action, real `ChatAnthropic` agent
  via `create_agent`. Reaches the `done` terminal state with a clean
  four-event trace (`send:search`, `recv:search`, `send:book`,
  `recv:book`).

  Adds 79 tests across `test_langchain_tool_ref.py`,
  `test_langchain_fsm.py`, `test_langchain_monitor.py`, and
  `test_langchain_middleware.py`. Middleware tests skip cleanly if
  LangChain isn't installed.

  This is a **second API alongside the existing DSL-based `Monitor`**,
  not a replacement — the two coexist and target different audiences.
  See the README for guidance on which to pick.

## 0.2.2 — 2026-05-05

### Added

- **`Unrecognized` MonitorResult** for projection-induced uncertainty.
  When the projection layer (typically over natural-language user input)
  can't decide which label to emit, it can pass the `UNRECOGNIZED`
  sentinel to `Monitor.send` / `Monitor.receive`. The monitor returns an
  `Unrecognized(expected, direction)` result *without* halting and
  *without* advancing state — distinct from `Violation` (agent broke the
  rules) so outer-loop code can react by driving a clarification turn
  rather than treating the trajectory as failed.

  Protocols may also include `Unrecognized` as an explicit choice branch
  (`?{Yes.end, No.end, Unrecognized.Loop}`); in that case the monitor
  follows the transition normally and returns `Ok`. Both modes — soft
  fall-through and explicit branch — are useful in different settings.

  Motivation: any projection from natural language to a finite alphabet
  is necessarily lossy. Conflating "the projection couldn't classify"
  with "the agent violated the protocol" loses important information at
  the system boundary; this change makes the distinction first-class.

  New exports: `Unrecognized`, `UNRECOGNIZED` from `llmcontract`.

## 0.2.1 — 2026-05-05

### Fixed

- **FSM compiler bug in recursion + multi-branch choice.** Protocols of the
  shape `rec X.!{a.X, b.X, c.X}` (or the external-choice equivalent
  `rec X.?{a.X, b.X, c.X}`) used to silently lock the loop to whichever
  branch was taken first — the second-and-later branches became
  unreachable after the first iteration. The cause was that the back-edge
  was compiled by snapshot-copying the target state's transitions before
  later choice branches were compiled. Fixed in
  [`b887e7e`](https://github.com/chrisbartoloburlo/llmcontract/commit/b887e7e)
  by replacing the snapshot with a state alias that resolves after compile
  finishes.

  Affects every protocol with a multi-branch choice that loops directly
  back to a recursion variable — common in agent loops where the agent
  picks one of several actions on every turn (e.g. `rec Loop.!{Search.Loop,
  Book.Loop, Cancel.Loop}`). Protocols where the choice is preceded by a
  non-choice step (e.g. `rec Loop.!Request.?{ToolCall.!Result.Loop,
  Final.end}`) were unaffected and the README example continued to work.

  Three regression tests added to `test_monitor.py`.

## 0.2.0 — 2026-05-04

- Add Langfuse integration: `LangfuseMonitor` records each send/receive as
  a guardrail observation with a pass/fail score.
- New optional dependency: `pip install llmsessioncontract[langfuse]`.

## 0.1.x — earlier

- Initial parser, FSM compiler, monitor.
- `MonitoredClient` and `ToolMiddleware` integration helpers.
