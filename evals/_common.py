"""Shared helpers for Project 2 evals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langsmith import Client

EVALS_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ensure_langsmith_dataset(name: str, rows: list[dict], description: str) -> str:
    """Create LangSmith dataset if missing; return dataset name."""
    client = Client()
    try:
        client.read_dataset(dataset_name=name)
    except Exception:
        client.create_dataset(dataset_name=name, description=description)
        client.create_examples(
            dataset_name=name,
            examples=[{"inputs": row, "outputs": {}} for row in rows],
        )
    return name


def eval_data(rows: list[dict], dataset_name: str, description: str) -> Any:
    """LangSmith dataset when API key present; otherwise local examples."""
    if os.getenv("LANGSMITH_API_KEY"):
        return ensure_langsmith_dataset(dataset_name, rows, description)
    return [{"inputs": row} for row in rows]


def should_upload() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY"))


def empty_ticket_state(ticket: dict, domain: str, **overrides: Any) -> dict:
    state = {
        "ticket_id": str(ticket.get("id") or "eval"),
        "raw": ticket,
        "domain": domain,
        "classification": None,
        "severity": None,
        "findings": [],
        "draft": None,
        "approval": "pending",
        "sent": False,
        "step_log": [],
        "next": "END",
    }
    state.update(overrides)
    return state
