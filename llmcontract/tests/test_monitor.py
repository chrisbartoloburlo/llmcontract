"""Tests for the runtime monitor."""

import pytest

from llmcontract.monitor.monitor import Blocked, Monitor, Ok, Violation


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
