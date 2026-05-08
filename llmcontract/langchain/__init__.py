"""LangChain-native protocol enforcement for ``llmcontract``.

A focused, FSM-as-data API for users who want to wire protocol monitoring
into LangChain agents with no DSL parsing and no magic strings. Tool
references are real Python callables; transitions are explicit objects
with optional guards and actions; violation handling is fully
user-controlled.

The full design and rationale live at
https://llmcontract.dev/findings/ and in the project README.

Submodules:
    tool_ref     ToolRef, ref()
    fsm          ProtocolFSM, Transition, MonitorContext, ViolationEvent
    monitor      ProtocolMonitor, fire_step
    middleware   CheckpointedProtocolMiddleware
    exceptions   ProtocolViolationError
"""

from llmcontract.langchain.exceptions import ProtocolViolationError
from llmcontract.langchain.fsm import (
    MonitorContext,
    ProtocolFSM,
    Transition,
    ViolationEvent,
    ViolationHandler,
)
from llmcontract.langchain.middleware import CheckpointedProtocolMiddleware
from llmcontract.langchain.monitor import ProtocolMonitor, fire_step
from llmcontract.langchain.tool_ref import ToolRef, ref

__all__ = [
    "ToolRef",
    "ref",
    "ProtocolFSM",
    "Transition",
    "MonitorContext",
    "ViolationEvent",
    "ViolationHandler",
    "ProtocolMonitor",
    "fire_step",
    "CheckpointedProtocolMiddleware",
    "ProtocolViolationError",
]
