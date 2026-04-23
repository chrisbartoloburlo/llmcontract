"""Tests for the integration layer (client wrapper + tool middleware)."""

import pytest

from llmcontract.monitor.monitor import Monitor
from llmcontract.integration import (
    MonitoredClient,
    ToolMiddleware,
    LLMResponse,
    ToolCall,
    ProtocolViolationError,
)


# ── Helpers ──────────────────────────────────────────────────

def fake_llm(*args, **kwargs):
    """Fake LLM that returns a raw dict."""
    return kwargs.get("_response", {"content": "ok"})


def identity_adapter(raw):
    """Adapter that passes through an LLMResponse or builds one from a dict."""
    if isinstance(raw, LLMResponse):
        return raw
    return LLMResponse(content=raw.get("content"))


def tool_response_adapter(raw):
    """Adapter that returns the raw value directly (already an LLMResponse)."""
    return raw


# ── MonitoredClient tests ────────────────────────────────────

class TestMonitoredClient:
    def test_happy_path(self):
        monitor = Monitor("!Request.?Response.end")
        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="Request",
            receive_label="Response",
        )
        resp = client.call()
        assert resp.content == "ok"
        assert monitor.is_terminal

    def test_send_violation(self):
        """Sending the wrong label raises ProtocolViolationError."""
        monitor = Monitor("!Request.?Response.end")
        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="WrongLabel",
            receive_label="Response",
        )
        with pytest.raises(ProtocolViolationError, match="send"):
            client.call()

    def test_receive_violation(self):
        """Receiving the wrong label raises ProtocolViolationError."""
        monitor = Monitor("!Request.?Response.end")
        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="Request",
            receive_label="WrongLabel",
        )
        with pytest.raises(ProtocolViolationError, match="receive"):
            client.call()

    def test_dynamic_receive_label(self):
        """receive_label can be a callable that inspects the response."""
        monitor = Monitor("!Request.?{ToolCall.end, Answer.end}")

        def dynamic_receive(resp: LLMResponse) -> str:
            return "ToolCall" if resp.has_tool_calls else "Answer"

        # Response without tool calls → Answer
        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="Request",
            receive_label=dynamic_receive,
        )
        resp = client.call()
        assert monitor.is_terminal

    def test_dynamic_send_label(self):
        """send_label can be a callable that inspects the request args."""
        monitor = Monitor("!Search.?Results.end")

        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label=lambda **kw: kw.get("action", "Search"),
            receive_label="Results",
        )
        resp = client.call(action="Search")
        assert monitor.is_terminal

    def test_blocked_after_violation(self):
        """After a violation, further calls raise with 'blocked'."""
        monitor = Monitor("!A.?B.end")
        client = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="Wrong",
            receive_label="B",
        )
        with pytest.raises(ProtocolViolationError):
            client.call()

        # Now monitor is halted
        client2 = MonitoredClient(
            llm_call=fake_llm,
            response_adapter=identity_adapter,
            monitor=monitor,
            send_label="A",
            receive_label="B",
        )
        with pytest.raises(ProtocolViolationError, match="blocked"):
            client2.call()


# ── ToolMiddleware tests ─────────────────────────────────────

class TestToolMiddleware:
    def test_single_tool_call(self):
        monitor = Monitor("?search.!search.end")
        middleware = ToolMiddleware(
            monitor=monitor,
            tools={"search": lambda query: f"results for {query}"},
        )
        tc = ToolCall(name="search", arguments={"query": "flights"}, id="1")
        result = middleware.execute(tc)
        assert result.result == "results for flights"
        assert result.tool_call_id == "1"
        assert monitor.is_terminal

    def test_unknown_tool(self):
        monitor = Monitor("?search.!search.end")
        middleware = ToolMiddleware(monitor=monitor, tools={})
        tc = ToolCall(name="search", arguments={}, id="1")
        with pytest.raises(ValueError, match="Unknown tool"):
            middleware.execute(tc)

    def test_tool_violation(self):
        """Calling the wrong tool raises ProtocolViolationError."""
        monitor = Monitor("?search.!search.end")
        middleware = ToolMiddleware(
            monitor=monitor,
            tools={"book": lambda: "booked"},
        )
        tc = ToolCall(name="book", arguments={}, id="1")
        with pytest.raises(ProtocolViolationError):
            middleware.execute(tc)

    def test_process_multiple_tool_calls(self):
        """process() handles all tool calls in a response."""
        monitor = Monitor("?a.!a.?b.!b.end")
        middleware = ToolMiddleware(
            monitor=monitor,
            tools={
                "a": lambda: "result_a",
                "b": lambda: "result_b",
            },
        )
        response = LLMResponse(
            tool_calls=[
                ToolCall(name="a", id="1"),
                ToolCall(name="b", id="2"),
            ]
        )
        results = middleware.process(response)
        assert len(results) == 2
        assert results[0].result == "result_a"
        assert results[1].result == "result_b"
        assert monitor.is_terminal

    def test_custom_labels(self):
        """Custom receive/send labels override tool name defaults."""
        monitor = Monitor("?ToolCall.!ToolResult.end")
        middleware = ToolMiddleware(
            monitor=monitor,
            tools={"search": lambda: "found"},
            receive_label="ToolCall",
            send_label="ToolResult",
        )
        tc = ToolCall(name="search", arguments={}, id="1")
        result = middleware.execute(tc)
        assert result.result == "found"
        assert monitor.is_terminal

    def test_register_tool(self):
        monitor = Monitor("?calc.!calc.end")
        middleware = ToolMiddleware(monitor=monitor)
        middleware.register("calc", lambda x: x * 2)
        tc = ToolCall(name="calc", arguments={"x": 5}, id="1")
        result = middleware.execute(tc)
        assert result.result == 10


# ── Combined client + middleware tests ───────────────────────

class TestCombinedFlow:
    """Client and middleware share a single monitor for full agent loop tracking."""

    PROTOCOL = (
        "rec Loop."
        "!Request."
        "?{ToolCall.!ToolResult.Loop, FinalAnswer.end}"
    )

    def _make_llm(self, responses):
        """Create a fake LLM that returns canned responses in order."""
        it = iter(responses)

        def llm_call(**kwargs):
            return next(it)

        return llm_call

    def _make_adapter(self):
        def adapter(raw):
            if "tool_calls" in raw:
                tool_calls = [
                    ToolCall(name=tc["name"], arguments=tc.get("args", {}), id=tc.get("id", ""))
                    for tc in raw["tool_calls"]
                ]
                return LLMResponse(tool_calls=tool_calls)
            return LLMResponse(content=raw.get("content", ""))

        return adapter

    def test_tool_then_answer(self):
        """One tool call round, then final answer."""
        monitor = Monitor(self.PROTOCOL)
        adapter = self._make_adapter()

        responses = [
            {"tool_calls": [{"name": "search", "args": {"q": "test"}, "id": "1"}]},
            {"content": "Here are the results"},
        ]
        llm = self._make_llm(responses)

        def dynamic_receive(resp: LLMResponse) -> str:
            return "ToolCall" if resp.has_tool_calls else "FinalAnswer"

        client = MonitoredClient(
            llm_call=llm,
            response_adapter=adapter,
            monitor=monitor,
            send_label="Request",
            receive_label=dynamic_receive,
        )
        middleware = ToolMiddleware(
            monitor=monitor,
            tools={"search": lambda q: f"found {q}"},
            receive_label=lambda tc: "ToolCall",
            send_label=lambda name, result: "ToolResult",
        )

        # Step 1: send request, LLM responds with tool call
        resp1 = client.call()
        assert resp1.has_tool_calls

        # Step 2: execute tool via middleware
        # Note: ToolCall receive was already consumed by client's dynamic label,
        # so middleware should NOT re-check receive. We handle this by having
        # middleware use a passthrough receive label.
        # Actually, let's restructure: the client handles !Request and ?ToolCall,
        # the middleware handles !ToolResult only.
        # Let me fix this — the middleware should only do the send side here.

        # Actually the protocol after client.call() is at the state after ?ToolCall.
        # So middleware needs to do !ToolResult. Let's use the middleware with
        # only send_label, and manually execute the tool.
        tool_call = resp1.tool_calls[0]
        tool_output = middleware._tools[tool_call.name](**tool_call.arguments)

        send_result = monitor.send("ToolResult")
        from llmcontract.monitor.monitor import Ok
        assert isinstance(send_result, Ok)

        # Step 3: send request again, LLM responds with final answer
        resp2 = client.call()
        assert resp2.content == "Here are the results"
        assert monitor.is_terminal

    def test_multiple_tool_rounds(self):
        """Multiple tool call rounds before final answer."""
        monitor = Monitor(self.PROTOCOL)
        adapter = self._make_adapter()

        responses = [
            {"tool_calls": [{"name": "search", "id": "1"}]},
            {"tool_calls": [{"name": "refine", "id": "2"}]},
            {"content": "Done"},
        ]
        llm = self._make_llm(responses)

        client = MonitoredClient(
            llm_call=llm,
            response_adapter=adapter,
            monitor=monitor,
            send_label="Request",
            receive_label=lambda r: "ToolCall" if r.has_tool_calls else "FinalAnswer",
        )

        # Round 1
        resp = client.call()
        assert resp.has_tool_calls
        monitor.send("ToolResult")

        # Round 2
        resp = client.call()
        assert resp.has_tool_calls
        monitor.send("ToolResult")

        # Final
        resp = client.call()
        assert resp.content == "Done"
        assert monitor.is_terminal

    def test_violation_on_wrong_sequence(self):
        """Skipping a required protocol step raises."""
        monitor = Monitor(self.PROTOCOL)
        adapter = self._make_adapter()

        responses = [{"content": "Done"}]
        llm = self._make_llm(responses)

        client = MonitoredClient(
            llm_call=llm,
            response_adapter=adapter,
            monitor=monitor,
            send_label="Request",
            receive_label=lambda r: "ToolCall" if r.has_tool_calls else "FinalAnswer",
        )

        # First call is fine — !Request.?FinalAnswer
        resp = client.call()
        assert monitor.is_terminal

        # Second call should fail — protocol is done
        responses2 = [{"content": "Extra"}]
        client2 = MonitoredClient(
            llm_call=self._make_llm(responses2),
            response_adapter=adapter,
            monitor=monitor,
            send_label="Request",
            receive_label=lambda r: "FinalAnswer",
        )
        with pytest.raises(ProtocolViolationError):
            client2.call()
