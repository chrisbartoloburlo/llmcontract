# Changelog

## 0.4.1 — 2026-05-08

### Added

- **`Monitor.trace`** — every send/receive call now records the
  attempted event into a per-monitor trace list (format mirrors the
  DSL: `"!Label"` / `"?Label"`). Captures violations, `Blocked`
  attempts, and `Unrecognized` attempts alongside accepted events,
  so the trace is a faithful audit log of the session.

- **`Monitor.to_dict()` / `Monitor.from_dict(protocol, state)`** —
  serializable runtime-state snapshot for persistence across process
  restarts. `from_dict` rebuilds state by replaying the saved trace
  through a fresh monitor, which makes the round-trip robust against
  internal automaton-numbering changes between releases.

  Pair with any persistence layer (Redis, postgres, S3, files) — the
  library itself stays storage-agnostic. The protocol string is *not*
  embedded in the snapshot; callers pass it to `from_dict`.

- **`Monitor.reset()`** — restore initial state and clear the trace.
  Symmetric with the langchain submodule's `ProtocolMonitor.reset()`.

These bring the DSL-based core monitor up to the same operational
shape as the 0.4.0 LangChain submodule: serializable state, audit
trail, restartable. No breaking changes — every addition is purely
additive on top of the existing 0.4.0 API.

## 0.4.0 — 2026-05-08

### Changed (breaking)

- **`ProtocolEnforcerMiddleware` is replaced by
  `CheckpointedProtocolMiddleware`.** The 0.3.x middleware stored FSM
  state on a `ProtocolMonitor` instance attribute, which was lost
  whenever LangGraph rehydrated the graph from a checkpoint — after a
  `HumanInTheLoopMiddleware` interrupt, after a worker restart, or in
  any multi-pod deployment. It also raced when one middleware instance
  was reused across concurrent agent invocations. The new middleware
  stores FSM state in a `ProtocolState` `AgentState` subclass
  (`fsm_state: str`, `fsm_trace: list[str]`); LangGraph checkpoints
  it automatically, keys it by `thread_id`, and resumes correctly.

  Read the final FSM state from `result["fsm_state"]` and the full
  trace from `result["fsm_trace"]` after `agent.invoke(...)`.

  No deprecation period — this is a known production correctness fix
  and pre-1.0, so the old class is removed rather than aliased.

### Added

- **`Transition.interrupt: bool = False`** — when `True`, the
  middleware suspends the agent via `langgraph.types.interrupt(...)`
  before any model call from the transition's source state, and
  fires the transition with the resume value as metadata. Eliminates
  the "orchestrator forgot to fire the gate" failure class — the gate
  becomes framework-enforced. The resume value either supplies an
  explicit `event_label` (for protocols where multiple interrupt
  transitions share a source state) or, when there's only one
  candidate, becomes the transition metadata directly.

  Only valid on `event_label`-based transitions; tool-backed
  transitions are inherently driven by `wrap_tool_call` rather than
  by interrupt resumption.

- **`Transition.match_structured_response: type | None = None`** —
  when set, the middleware fires the transition deterministically in
  `after_model` whenever `state["structured_response"]` is an instance
  of that type. Use with `response_format=...` on the agent for
  reliable detection of non-tool agent events without text-pattern
  parsing.

- **`fire_step` helper** in `llmcontract.langchain.monitor` — pure
  transition-firing function shared by `ProtocolMonitor` (in-process
  state) and `CheckpointedProtocolMiddleware` (state in the
  LangGraph checkpoint). Exposed publicly for callers who want to
  drive the FSM from custom contexts.

### Updated

- The `examples/langchain_booking/booking_agent_submodule.py` example
  now uses `CheckpointedProtocolMiddleware`, an `interrupt=True`
  approval gate, and the `Command(resume=...)` pattern. The protocol
  was simplified to compress `!PresentOptions` into the
  `?UserApproval` interrupt — the agent's text reply is delivered
  with the interrupt payload as user-facing context, and the
  interrupt itself is the gate. Demonstrates a fully
  framework-enforced approval flow with no orchestrator-side `fire_step`.

- 16 tests (`test_langchain_middleware.py`) rewritten for the new
  architecture. Total test count: 128.

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
