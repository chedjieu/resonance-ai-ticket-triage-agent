"""Query mock service logs for investigation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from app.tools._domain import get_domain
from app.tools._time import parse_since

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@lru_cache(maxsize=8)
def _load_logs(domain: str) -> dict[str, list[dict]]:
    path = DATA_DIR / domain / "mock_logs.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@tool
def query_logs(service: str, since: str = "1h") -> list[dict]:
    """Fetch recent log lines for a service. Each entry has timestamp, level, and message."""
    domain = get_domain()
    all_logs = _load_logs(domain).get(service, [])
    if not all_logs:
        return []

    window = parse_since(since)
    now = datetime.now(UTC)
    cutoff = now - window
    filtered = [entry for entry in all_logs if _parse_ts(entry["timestamp"]) >= cutoff]

    # Mock data uses fixed dates; return all service logs if the window is empty.
    return filtered or list(all_logs)
