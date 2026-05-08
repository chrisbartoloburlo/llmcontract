"""Tests for the runtime monitor."""

import pytest

from llmcontract.monitor.monitor import (
    Blocked, Monitor, Ok, UNRECOGNIZED, Unrecognized, Violation,
)


FLIGHT_PROTOCOL = (
    "!SearchFlights.?FlightResults.!PresentOptions"
    ".?UserApproval.!BookFlight.?BookingConfirmation.end"
)

CARD_PROTOCOL = (
    "!CreateCard.?{CardCreated.rec X.!Transaction"
    ".?{TransactionOK.X, SessionEnd}, CardError}.end"
)


class TestValidSequence:
    """Test that a valid sequence of events passes."""

    def test_flight_happy_path(self):
        m = Monitor(FLIGHT_PROTOCOL)
        assert isinstance(m.send("SearchFlights"), Ok)
        assert isinstance(m.receive("FlightResults"), Ok)
        assert isinstance(m.send("PresentOptions"), Ok)
        assert isinstance(m.receive("UserApproval"), Ok)
        assert isinstance(m.send("BookFlight"), Ok)
        assert isinstance(m.receive("BookingConfirmation"), Ok)
        assert m.is_terminal


class TestViolation:
    """Test that protocol violations are caught."""

    def test_book_without_approval(self):
        """!BookFlight without preceding ?UserApproval should violate."""
        m = Monitor(FLIGHT_PROTOCOL)
        assert isinstance(m.send("SearchFlights"), Ok)
        assert isinstance(m.receive("FlightResults"), Ok)
        assert isinstance(m.send("PresentOptions"), Ok)
        # Skip UserApproval, go straight to BookFlight
        result = m.send("BookFlight")
        assert isinstance(result, Violation)
        assert "?UserApproval" in result.expected
        assert result.got == "!BookFlight"

    def test_wrong_direction(self):
        """Sending when we should receive is a violation."""
        m = Monitor(FLIGHT_PROTOCOL)
        assert isinstance(m.send("SearchFlights"), Ok)
        # Should receive FlightResults, but we try to send
        result = m.send("FlightResults")
        assert isinstance(result, Violation)

    def test_halted_after_violation(self):
        """After a violation, all further events return Blocked."""
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        m.send("Wrong")  # violation
        assert m.is_halted
        result = m.send("SearchFlights")
        assert isinstance(result, Blocked)


class TestChoice:
    """Test that choices resolve correctly."""

    def test_external_choice_card_created(self):
        m = Monitor(CARD_PROTOCOL)
        assert isinstance(m.send("CreateCard"), Ok)
        assert isinstance(m.receive("CardCreated"), Ok)

    def test_external_choice_card_error(self):
        m = Monitor(CARD_PROTOCOL)
        assert isinstance(m.send("CreateCard"), Ok)
        assert isinstance(m.receive("CardError"), Ok)
        assert m.is_terminal

    def test_invalid_choice(self):
        m = Monitor(CARD_PROTOCOL)
        assert isinstance(m.send("CreateCard"), Ok)
        result = m.receive("UnknownBranch")
        assert isinstance(result, Violation)


class TestRecursion:
    """Test that recursion cycles correctly."""

    def test_single_transaction_then_end(self):
        m = Monitor(CARD_PROTOCOL)
        assert isinstance(m.send("CreateCard"), Ok)
        assert isinstance(m.receive("CardCreated"), Ok)
        assert isinstance(m.send("Transaction"), Ok)
        assert isinstance(m.receive("SessionEnd"), Ok)
        assert m.is_terminal

    def test_multiple_transactions(self):
        """Recursion should allow repeating the transaction loop."""
        m = Monitor(CARD_PROTOCOL)
        assert isinstance(m.send("CreateCard"), Ok)
        assert isinstance(m.receive("CardCreated"), Ok)

        # First transaction
        assert isinstance(m.send("Transaction"), Ok)
        assert isinstance(m.receive("TransactionOK"), Ok)

        # Second transaction (recursion)
        assert isinstance(m.send("Transaction"), Ok)
        assert isinstance(m.receive("TransactionOK"), Ok)

        # Third transaction then end
        assert isinstance(m.send("Transaction"), Ok)
        assert isinstance(m.receive("SessionEnd"), Ok)
        assert m.is_terminal

    def test_recursive_choice_preserves_all_branches_after_loop(self):
        """Regression: `rec X.!{a.X, b.X, c.X}` previously locked the loop to
        whichever branch was taken first, because RecVar copied a snapshot of
        the target's transitions before later choice branches were compiled.
        """
        m = Monitor("rec X.!{A.X, B.X, C.X}")
        for label in ["A", "B", "C", "A", "B", "C"]:
            assert isinstance(m.send(label), Ok), f"{label!r} rejected after loop"

    def test_recursive_choice_violation_still_fires(self):
        m = Monitor("rec X.!{A.X, B.X}")
        assert isinstance(m.send("A"), Ok)
        result = m.send("C")
        assert not isinstance(result, Ok)
        assert hasattr(result, "expected")
        assert set(result.expected) == {"!A", "!B"}

    def test_nested_recursive_choices(self):
        """Two nested `rec` scopes, each with multi-branch choices that loop."""
        protocol = (
            "rec Outer.!{"
            "Enter.rec Inner.!{Step.Inner, Done.Outer}, "
            "Skip.Outer"
            "}"
        )
        m = Monitor(protocol)
        for label in ["Skip", "Enter", "Step", "Step", "Done", "Skip", "Enter"]:
            assert isinstance(m.send(label), Ok), f"{label!r} rejected"


class TestSimpleProtocol:
    """Test with minimal protocols."""

    def test_end_only(self):
        m = Monitor("end")
        assert m.is_terminal

    def test_single_send(self):
        m = Monitor("!Ping.end")
        assert not m.is_terminal
        assert isinstance(m.send("Ping"), Ok)
        assert m.is_terminal


class TestUnrecognized:
    """Soft fail-open path for projection layers that can't classify input.

    Distinct from `Violation`: signals that the projection (typically over
    natural language) couldn't decide which label to emit, so the outer loop
    should drive a clarification turn rather than treat the agent as having
    broken the protocol.
    """

    def test_unrecognized_does_not_advance_state_or_halt(self):
        m = Monitor("?{Yes.end, No.end}")
        result = m.receive(UNRECOGNIZED)
        assert isinstance(result, Unrecognized)
        assert result.direction == "receive"
        assert set(result.expected) == {"?Yes", "?No"}
        # State preserved → a real classification can still fire
        assert isinstance(m.receive("Yes"), Ok)
        assert m.is_terminal

    def test_repeated_unrecognized_keeps_monitor_live(self):
        m = Monitor("?{Yes.end, No.end}")
        for _ in range(5):
            assert isinstance(m.receive(UNRECOGNIZED), Unrecognized)
        # After many unrecognized events the monitor is still live
        assert not m.is_halted
        assert isinstance(m.receive("No"), Ok)

    def test_protocol_can_handle_unrecognized_explicitly(self):
        """If the protocol declares an `Unrecognized` branch, the monitor
        treats it as a regular transition and returns Ok."""
        m = Monitor(
            "rec Loop."
            "!Ask.?{Yes.end, No.end, Unrecognized.Loop}"
        )
        assert isinstance(m.send("Ask"), Ok)
        # First time the user is unclear → protocol routes us back to Loop
        assert isinstance(m.receive(UNRECOGNIZED), Ok)
        assert isinstance(m.send("Ask"), Ok)
        assert isinstance(m.receive("Yes"), Ok)
        assert m.is_terminal

    def test_unrecognized_works_for_send_direction_too(self):
        m = Monitor("!{A.end, B.end}")
        result = m.send(UNRECOGNIZED)
        assert isinstance(result, Unrecognized)
        assert result.direction == "send"
        assert set(result.expected) == {"!A", "!B"}

    def test_unrecognized_after_a_real_violation_returns_blocked(self):
        m = Monitor("?Yes.end")
        assert isinstance(m.receive("No"), Violation)
        # Once halted, even Unrecognized is rejected as Blocked
        assert isinstance(m.receive(UNRECOGNIZED), Blocked)


class TestTrace:
    """The trace records every send/receive attempt, regardless of outcome."""

    def test_trace_starts_empty(self):
        m = Monitor(FLIGHT_PROTOCOL)
        assert m.trace == []

    def test_trace_records_each_event(self):
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        m.receive("FlightResults")
        assert m.trace == ["!SearchFlights", "?FlightResults"]

    def test_trace_records_violations_and_subsequent_blocked(self):
        m = Monitor("!A.?B.end")
        m.send("A")
        m.send("Wrong")  # Violation — halts
        m.send("Whatever")  # Blocked
        assert m.trace == ["!A", "!Wrong", "!Whatever"]
        assert m.is_halted

    def test_trace_records_unrecognized_attempts(self):
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        m.receive("FlightResults")
        m.send("PresentOptions")
        m.receive(UNRECOGNIZED)  # doesn't advance, but recorded
        assert m.trace == [
            "!SearchFlights", "?FlightResults",
            "!PresentOptions", "?Unrecognized",
        ]

    def test_trace_property_is_a_copy(self):
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        snapshot = m.trace
        snapshot.append("garbage")
        assert m.trace == ["!SearchFlights"]  # internal trace untouched


class TestReset:
    def test_reset_restores_initial_state_and_clears_trace(self):
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        m.receive("FlightResults")
        m.reset()
        assert m.current_state == m._automaton.initial_state
        assert m.trace == []
        assert not m.is_halted

    def test_reset_clears_halted_flag(self):
        m = Monitor("?Yes.end")
        m.receive("No")  # violation
        assert m.is_halted
        m.reset()
        assert not m.is_halted
        # Fresh monitor accepts the right event again
        assert isinstance(m.receive("Yes"), Ok)


class TestSerialization:
    """to_dict / from_dict round-trip preserves runtime state by replay."""

    def test_to_dict_returns_trace(self):
        m = Monitor(FLIGHT_PROTOCOL)
        m.send("SearchFlights")
        m.receive("FlightResults")
        assert m.to_dict() == {"trace": ["!SearchFlights", "?FlightResults"]}

    def test_from_dict_restores_current_state(self):
        original = Monitor(FLIGHT_PROTOCOL)
        original.send("SearchFlights")
        original.receive("FlightResults")
        original.send("PresentOptions")

        restored = Monitor.from_dict(FLIGHT_PROTOCOL, original.to_dict())
        assert restored.current_state == original.current_state
        assert restored.trace == original.trace
        assert not restored.is_halted

    def test_from_dict_continues_normally(self):
        original = Monitor(FLIGHT_PROTOCOL)
        original.send("SearchFlights")
        original.receive("FlightResults")
        snapshot = original.to_dict()

        restored = Monitor.from_dict(FLIGHT_PROTOCOL, snapshot)
        # The next legal event should work just like on `original`.
        assert isinstance(restored.send("PresentOptions"), Ok)
        assert restored.trace == [
            "!SearchFlights", "?FlightResults", "!PresentOptions",
        ]

    def test_from_dict_replays_violation_and_blocked(self):
        original = Monitor("!A.?B.end")
        original.send("A")
        original.send("Wrong")  # halts
        original.send("Whatever")  # blocked

        restored = Monitor.from_dict("!A.?B.end", original.to_dict())
        assert restored.is_halted
        assert restored.trace == original.trace

    def test_from_dict_with_empty_state_is_fresh_monitor(self):
        m = Monitor.from_dict(FLIGHT_PROTOCOL, {})
        assert m.trace == []
        assert m.current_state == m._automaton.initial_state

    def test_from_dict_rejects_malformed_event(self):
        with pytest.raises(ValueError, match="not in the expected"):
            Monitor.from_dict(FLIGHT_PROTOCOL, {"trace": ["malformed"]})
