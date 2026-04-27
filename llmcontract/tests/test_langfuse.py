"""Tests for Langfuse integration using a mock Langfuse client."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from llmcontract.monitor.monitor import Ok, Violation, Blocked
from llmcontract.integration.langfuse import LangfuseMonitor


# ── Mock Langfuse ──────────────────────────────────────────────


@dataclass
class MockScore:
    name: str
    value: Any
    data_type: str = ""
    comment: str = ""


@dataclass
class MockObservation:
    as_type: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    output: dict | None = None
    metadata: dict = field(default_factory=dict)
    scores: list[MockScore] = field(default_factory=list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def score(self, **kwargs):
        self.scores.append(MockScore(**kwargs))


class MockLangfuse:
    def __init__(self):
        self.observations: list[MockObservation] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        obs = MockObservation(**{k: v for k, v in kwargs.items() if k in MockObservation.__dataclass_fields__})
        self.observations.append(obs)
        yield obs


# ── Tests ──────────────────────────────────────────────────────


def test_ok_events_recorded():
    lf = MockLangfuse()
    m = LangfuseMonitor("!Ping.?Pong.end", langfuse=lf)

    r1 = m.send("Ping")
    r2 = m.receive("Pong")

    assert isinstance(r1, Ok)
    assert isinstance(r2, Ok)
    assert m.is_terminal
    assert len(lf.observations) == 2

    # First observation: !Ping
    obs1 = lf.observations[0]
    assert obs1.as_type == "guardrail"
    assert obs1.input["action"] == "!Ping"
    assert obs1.output["passed"] is True
    assert obs1.scores[0].value == 1

    # Second observation: ?Pong
    obs2 = lf.observations[1]
    assert obs2.input["action"] == "?Pong"
    assert obs2.output["passed"] is True
    assert obs2.scores[0].name == "protocol_compliance"


def test_violation_recorded():
    lf = MockLangfuse()
    m = LangfuseMonitor("!Ping.?Pong.end", langfuse=lf)

    m.send("Ping")
    result = m.send("Pong")  # wrong — should be receive

    assert isinstance(result, Violation)
    assert len(lf.observations) == 2

    obs = lf.observations[1]
    assert obs.output["passed"] is False
    assert obs.output["result"] == "violation"
    assert "expected" in obs.output
    assert obs.output["got"] == "!Pong"
    assert obs.scores[0].value == 0


def test_blocked_recorded():
    lf = MockLangfuse()
    m = LangfuseMonitor("!Ping.?Pong.end", langfuse=lf)

    m.send("Ping")
    m.send("Pong")  # violation
    result = m.send("Anything")  # blocked

    assert isinstance(result, Blocked)
    assert len(lf.observations) == 3

    obs = lf.observations[2]
    assert obs.output["passed"] is False
    assert obs.output["result"] == "blocked"
    assert obs.scores[0].value == 0


def test_step_counter():
    lf = MockLangfuse()
    m = LangfuseMonitor("!A.?B.!C.?D.end", langfuse=lf)

    m.send("A")
    m.receive("B")
    m.send("C")
    m.receive("D")

    assert len(lf.observations) == 4
    assert lf.observations[0].name == "protocol-step-1"
    assert lf.observations[3].name == "protocol-step-4"


def test_metadata_includes_protocol():
    lf = MockLangfuse()
    protocol = "!Ping.?Pong.end"
    m = LangfuseMonitor(protocol, langfuse=lf)

    m.send("Ping")

    obs = lf.observations[0]
    assert obs.metadata["protocol"] == protocol
    assert obs.metadata["step"] == 1


def test_existing_monitor():
    """Can wrap an existing Monitor instance."""
    from llmcontract.monitor.monitor import Monitor

    lf = MockLangfuse()
    base = Monitor("!Ping.?Pong.end")
    m = LangfuseMonitor("!Ping.?Pong.end", langfuse=lf, monitor=base)

    m.send("Ping")
    assert m.monitor is base
    assert base.current_state == m.current_state


def test_choice_protocol():
    lf = MockLangfuse()
    m = LangfuseMonitor("!Req.?{Ok.end, Err.end}", langfuse=lf)

    m.send("Req")
    m.receive("Err")

    assert m.is_terminal
    assert len(lf.observations) == 2
    assert lf.observations[1].output["passed"] is True


def test_recursion_protocol():
    lf = MockLangfuse()
    m = LangfuseMonitor("rec X.!Ping.?Pong.X", langfuse=lf)

    for i in range(3):
        m.send("Ping")
        m.receive("Pong")

    assert len(lf.observations) == 6
    # All should pass
    for obs in lf.observations:
        assert obs.scores[0].value == 1
