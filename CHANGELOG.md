# Changelog

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
