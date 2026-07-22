"""Resonance Technologies - Ticket Triage Agent (Project 2).

Loads `.env` from this project root, or falls back to `../RAIRA-AI-Research-Assistant/.env`
so both projects can share one config file during the lab.
"""
from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent.parent
    _candidates = [
        _project_root / ".env",
        _project_root.parent / "RAIRA-AI-Research-Assistant" / ".env",
    ]
    for _env in _candidates:
        if _env.exists():
            # .env is the source of truth — ignore stale RTTA_MODEL=fake from the shell.
            load_dotenv(_env, override=True)
            break
except Exception:
    pass
