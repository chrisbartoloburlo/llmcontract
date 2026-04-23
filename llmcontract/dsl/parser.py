"""Hand-written recursive descent parser for the session type DSL."""

from __future__ import annotations

from llmcontract.dsl.ast import (
    End, ExternalChoice, InternalChoice, ProtocolNode,
    Receive, Recursion, RecVar, Send, Sequence,
)


class ParseError(Exception):
    """Raised on invalid input with position information."""

    def __init__(self, message: str, pos: int) -> None:
        self.pos = pos
        super().__init__(f"Parse error at position {pos}: {message}")


class _Parser:
    def __init__(self, src: str) -> None:
        self.src = src
        self.pos = 0

    # ── helpers ──────────────────────────────────────────────

    def _skip_ws(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] in " \t\n\r":
            self.pos += 1

    def _peek(self) -> str | None:
        self._skip_ws()
        if self.pos >= len(self.src):
            return None
        return self.src[self.pos]

    def _expect(self, ch: str) -> None:
        self._skip_ws()
        if self.pos >= len(self.src) or self.src[self.pos] != ch:
            found = self.src[self.pos] if self.pos < len(self.src) else "EOF"
            raise ParseError(f"expected '{ch}', got '{found}'", self.pos)
        self.pos += 1

    def _read_ident(self) -> str:
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == '_'):
            self.pos += 1
        if self.pos == start:
            found = self.src[self.pos] if self.pos < len(self.src) else "EOF"
            raise ParseError(f"expected identifier, got '{found}'", self.pos)
        return self.src[start:self.pos]

    def _at_keyword(self, kw: str) -> bool:
        self._skip_ws()
        end = self.pos + len(kw)
        if self.src[self.pos:end] == kw:
            # Make sure it's not a prefix of a longer identifier
            if end >= len(self.src) or not (self.src[end].isalnum() or self.src[end] == '_'):
                return True
        return False

    # ── grammar ──────────────────────────────────────────────
    #
    # protocol  ::= atom ('.' protocol)?
    # atom      ::= '!' choice_or_label
    #             | '?' choice_or_label
    #             | 'rec' IDENT '.' protocol
    #             | 'end'
    #             | IDENT              (recursion variable)
    #
    # choice_or_label ::= '{' branch (',' branch)* '}'
    #                   | IDENT
    #
    # branch ::= IDENT ('.' protocol)?

    def parse(self) -> ProtocolNode:
        node = self._parse_protocol()
        self._skip_ws()
        if self.pos < len(self.src):
            raise ParseError(f"unexpected character '{self.src[self.pos]}'", self.pos)
        return node

    def _parse_protocol(self) -> ProtocolNode:
        left = self._parse_atom()
        self._skip_ws()
        if self._peek() == '.':
            # Could be sequence or end-of-input
            # We need to check that what follows '.' is another atom, not EOF
            save = self.pos
            self.pos += 1  # consume '.'
            self._skip_ws()
            if self.pos >= len(self.src):
                raise ParseError("unexpected EOF after '.'", self.pos)
            right = self._parse_protocol()
            left = Sequence(left, right)
        return left

    def _parse_atom(self) -> ProtocolNode:
        ch = self._peek()
        if ch is None:
            raise ParseError("unexpected EOF", self.pos)

        if ch == '!':
            self.pos += 1
            return self._parse_send()
        if ch == '?':
            self.pos += 1
            return self._parse_receive()
        if self._at_keyword('rec'):
            return self._parse_rec()
        if self._at_keyword('end'):
            self.pos += 3
            return End()

        # Must be a recursion variable
        ident = self._read_ident()
        return RecVar(ident)

    def _parse_send(self) -> ProtocolNode:
        if self._peek() == '{':
            return self._parse_internal_choice()
        label = self._read_ident()
        return Send(label)

    def _parse_receive(self) -> ProtocolNode:
        if self._peek() == '{':
            return self._parse_external_choice()
        label = self._read_ident()
        return Receive(label)

    def _parse_internal_choice(self) -> InternalChoice:
        self._expect('{')
        branches = self._parse_branches()
        self._expect('}')
        return InternalChoice(branches)

    def _parse_external_choice(self) -> ExternalChoice:
        self._expect('{')
        branches = self._parse_branches()
        self._expect('}')
        return ExternalChoice(branches)

    def _parse_branches(self) -> dict[str, ProtocolNode]:
        branches: dict[str, ProtocolNode] = {}
        label, body = self._parse_branch()
        branches[label] = body
        while self._peek() == ',':
            self.pos += 1  # consume ','
            label, body = self._parse_branch()
            if label in branches:
                raise ParseError(f"duplicate branch label '{label}'", self.pos)
            branches[label] = body
        return branches

    def _parse_branch(self) -> tuple[str, ProtocolNode]:
        label = self._read_ident()
        self._skip_ws()
        if self._peek() == '.':
            self.pos += 1  # consume '.'
            body = self._parse_protocol()
        else:
            body = End()
        return label, body

    def _parse_rec(self) -> Recursion:
        self.pos += 3  # consume 'rec'
        var = self._read_ident()
        self._expect('.')
        body = self._parse_protocol()
        return Recursion(var, body)


def parse(src: str) -> ProtocolNode:
    """Parse a session type DSL string into an AST."""
    return _Parser(src).parse()
