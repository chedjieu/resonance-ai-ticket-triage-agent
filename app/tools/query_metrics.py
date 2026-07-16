"""Query mock service metrics for investigation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

from app.tools._domain import get_domain

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


@lru_cache(maxsize=8)
def _load_metrics(domain: str) -> dict[str, dict[str, dict]]:
    path = DATA_DIR / domain / "mock_metrics.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@tool
def query_metrics(service: str, metric: str, since: str = "1h") -> dict:
    """Fetch a metric snapshot for a service (current, avg, p95, trend)."""
    _ = since  # window is illustrative for mock snapshots
    domain = get_domain()
    service_metrics = _load_metrics(domain).get(service, {})
    snapshot = service_metrics.get(metric)
    if snapshot is None:
        return {
            "service": service,
            "metric": metric,
            "current": None,
            "avg": None,
            "p95": None,
            "trend": "unknown",
        }
    return {
        "service": service,
        "metric": metric,
        "current": snapshot.get("current"),
        "avg": snapshot.get("avg"),
        "p95": snapshot.get("p95"),
        "trend": snapshot.get("trend", "flat"),
    }
