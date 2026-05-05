# Changelog

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
