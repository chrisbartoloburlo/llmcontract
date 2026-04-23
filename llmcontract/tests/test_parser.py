"""Tests for the session type DSL parser."""

import pytest

from llmcontract.dsl.ast import (
    End, ExternalChoice, InternalChoice,
    Receive, Recursion, RecVar, Send, Sequence,
)
from llmcontract.dsl.parser import ParseError, parse


class TestBasicParsing:
    def test_send(self):
        assert parse("!Ping") == Send("Ping")

    def test_receive(self):
        assert parse("?Pong") == Receive("Pong")

    def test_end(self):
        assert parse("end") == End()

    def test_sequence(self):
        result = parse("!Ping.?Pong.end")
        expected = Sequence(Send("Ping"), Sequence(Receive("Pong"), End()))
        assert result == expected


class TestChoices:
    def test_internal_choice(self):
        result = parse("!{a, b}")
        assert isinstance(result, InternalChoice)
        assert set(result.branches.keys()) == {"a", "b"}

    def test_external_choice(self):
        result = parse("?{a, b}")
        assert isinstance(result, ExternalChoice)
        assert set(result.branches.keys()) == {"a", "b"}

    def test_choice_with_continuation(self):
        result = parse("?{ok.!Done.end, err.!Retry.end}")
        assert isinstance(result, ExternalChoice)
        assert "ok" in result.branches
        assert "err" in result.branches


class TestRecursion:
    def test_simple_recursion(self):
        result = parse("rec X.!Ping.?Pong.X")
        assert isinstance(result, Recursion)
        assert result.var == "X"

    def test_recursion_variable(self):
        result = parse("rec X.!Ping.X")
        assert isinstance(result, Recursion)
        body = result.body
        assert isinstance(body, Sequence)
        assert body.right == RecVar("X")


class TestPaymentProtocol:
    """Test the flight booking protocol from the spec."""

    PROTOCOL = (
        "!SearchFlights.?FlightResults.!PresentOptions"
        ".?UserApproval.!BookFlight.?BookingConfirmation.end"
    )

    def test_parses_without_error(self):
        result = parse(self.PROTOCOL)
        assert result is not None

    def test_structure(self):
        result = parse(self.PROTOCOL)
        # Should be a nested Sequence starting with Send("SearchFlights")
        assert isinstance(result, Sequence)
        assert result.left == Send("SearchFlights")


class TestCardProtocol:
    """Test the card protocol with recursion and external choice."""

    PROTOCOL = (
        "!CreateCard.?{CardCreated.rec X.!Transaction"
        ".?{TransactionOK.X, SessionEnd}, CardError}.end"
    )

    def test_parses_without_error(self):
        result = parse(self.PROTOCOL)
        assert result is not None

    def test_has_external_choice(self):
        result = parse(self.PROTOCOL)
        # Top level: Sequence(Send("CreateCard"), Sequence(ExternalChoice(...), End()))
        assert isinstance(result, Sequence)
        assert result.left == Send("CreateCard")
        rest = result.right
        assert isinstance(rest, Sequence)
        assert isinstance(rest.left, ExternalChoice)
        assert set(rest.left.branches.keys()) == {"CardCreated", "CardError"}


class TestParseErrors:
    def test_empty_input(self):
        with pytest.raises(ParseError):
            parse("")

    def test_invalid_character(self):
        with pytest.raises(ParseError):
            parse("@foo")

    def test_unclosed_brace(self):
        with pytest.raises(ParseError):
            parse("!{a, b")

    def test_trailing_dot(self):
        with pytest.raises(ParseError):
            parse("!Ping.")
