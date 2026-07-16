"""Parse relative time windows like 1h, 30m, 2d."""

from __future__ import annotations

import re
from datetime import timedelta

_SINCE_RE = re.compile(r"^(\d+)(s|m|h|d)$", re.IGNORECASE)

_UNITS = {
    "s": lambda n: timedelta(seconds=n),
    "m": lambda n: timedelta(minutes=n),
    "h": lambda n: timedelta(hours=n),
    "d": lambda n: timedelta(days=n),
}


def parse_since(since: str) -> timedelta:
    """Parse a duration string such as ``1h``, ``30m``, or ``2d``."""
    text = since.strip().lower()
    match = _SINCE_RE.match(text)
    if not match:
        raise ValueError(f"invalid since value: {since!r} (expected e.g. 1h, 30m, 2d)")
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return _UNITS[unit](amount)
