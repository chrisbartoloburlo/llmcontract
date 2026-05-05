#!/usr/bin/env python3
"""Validate a session-type DSL string by parsing it with llmcontract.

Usage:
    python3 validate.py '<DSL string>'

Exits 0 if the protocol parses cleanly, 1 otherwise. On parse error,
prints the exception message to stderr.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: validate.py '<DSL string>'", file=sys.stderr)
        return 2

    dsl = argv[1]

    try:
        from llmcontract import Monitor
    except ImportError:
        print(
            "llmcontract is not installed. Install it with:\n"
            "    pip install llmsessioncontract",
            file=sys.stderr,
        )
        return 3

    try:
        Monitor(dsl)
    except Exception as e:
        print(f"INVALID: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
