"""Client wrapper that enforces protocol compliance on LLM calls."""

from __future__ import annotations

from typing import Any, Callable

from llmcontract.monitor.monitor import Monitor, Ok
from llmcontract.integration.types import LLMResponse
from llmcontract.integration.exceptions import ProtocolViolationError


class MonitoredClient:
    """Wraps an LLM call function and checks protocol events automatically.

    Parameters
    ----------
    llm_call:
        A function that sends a request to the LLM and returns a raw response.
        Signature is flexible — MonitoredClient passes through *args/**kwargs.
    response_adapter:
        Converts the vendor-specific response object into an LLMResponse.
    monitor:
        Shared monitor instance that tracks protocol state.
    send_label:
        Label (or label-producing function) for the !Send event.
        If callable, receives the same (*args, **kwargs) as llm_call.
    receive_label:
        Label (or label-producing function) for the ?Receive event.
        If callable, receives the LLMResponse.
    """

    def __init__(
        self,
        llm_call: Callable[..., Any],
        response_adapter: Callable[[Any], LLMResponse],
        monitor: Monitor,
        send_label: str | Callable[..., str] = "Request",
        receive_label: str | Callable[[LLMResponse], str] = "Response",
    ) -> None:
        self._llm_call = llm_call
        self._response_adapter = response_adapter
        self._monitor = monitor
        self._send_label = send_label
        self._receive_label = receive_label

    @property
    def monitor(self) -> Monitor:
        return self._monitor

    def call(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """Send a request to the LLM, checking protocol on both sides.

        1. Resolve send_label → monitor.send(label) → raise on violation
        2. Call llm_call(*args, **kwargs)
        3. Adapt response via response_adapter
        4. Resolve receive_label → monitor.receive(label) → raise on violation
        5. Return LLMResponse
        """
        # Send check
        label = self._send_label(*args, **kwargs) if callable(self._send_label) else self._send_label
        result = self._monitor.send(label)
        if not isinstance(result, Ok):
            raise ProtocolViolationError(result, "send")

        # LLM call
        raw = self._llm_call(*args, **kwargs)
        response = self._response_adapter(raw)

        # Receive check
        label = self._receive_label(response) if callable(self._receive_label) else self._receive_label
        result = self._monitor.receive(label)
        if not isinstance(result, Ok):
            raise ProtocolViolationError(result, "receive")

        return response
