"""Fetch prior tickets for a user."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from app.tools._domain import get_domain

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@lru_cache(maxsize=8)
def _load_history(domain: str) -> list[dict]:
    path = DATA_DIR / domain / "historical_tickets.jsonl"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@tool
def get_ticket_history(user_id: str, k: int = 5) -> list[dict]:
    """Return the last k historical tickets for a user (matched by sender email)."""
    domain = get_domain()
    matches = [
        row
        for row in _load_history(domain)
        if row.get("sender") == user_id or row.get("user_id") == user_id
    ]
    return matches[-k:]
