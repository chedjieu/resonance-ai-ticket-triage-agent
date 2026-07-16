"""Append-only JSONL log of HITL approve/edit/reject outcomes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "data" / "hitl_outcomes.jsonl"


def append_hitl_outcome(record: dict[str, Any], path: Path | None = None) -> None:
    """Append one HITL outcome as a JSON line."""
    log_path = path or DEFAULT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **record,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_hitl_outcomes(path: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Return the last `limit` HITL outcome records (oldest → newest within the window)."""
    log_path = path or DEFAULT_LOG
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if limit <= 0:
        return rows
    return rows[-limit:]


__all__ = ["DEFAULT_LOG", "append_hitl_outcome", "load_hitl_outcomes"]
