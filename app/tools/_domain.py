"""Current ticket domain for tools (set by investigator before tool calls)."""

from __future__ import annotations

import contextvars

_current_domain: contextvars.ContextVar[str] = contextvars.ContextVar("domain", default="support")


def get_domain() -> str:
    return _current_domain.get()


def set_domain(domain: str) -> None:
    _current_domain.set(domain)
