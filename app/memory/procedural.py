"""Procedural memory — versioned responder style prompts on disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "prompts"

DEFAULT_PROMPT = (
    "You are a support responder for Resonance Technologies. Write a clear, empathetic "
    "reply to the customer. Cite findings when relevant. Never invent refunds, credits, "
    "SLAs, or timelines. Prefer concrete next steps. Escalate when unsure or when the "
    "issue involves payment, legal risk, or PII handling."
)


def _path(domain: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in domain)
    return PROMPTS_DIR / f"responder_{safe}.json"


def _default_doc(domain: str) -> dict:
    return {
        "domain": domain,
        "latest": "v1",
        "versions": {
            "v1": {
                "prompt": DEFAULT_PROMPT,
                "created_at": "2026-01-01T00:00:00Z",
            }
        },
    }


def _load(domain: str) -> dict:
    path = _path(domain)
    if not path.exists():
        return _default_doc(domain)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(domain: str, doc: dict) -> None:
    path = _path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def get_responder_prompt(domain: str, version: str = "latest") -> str:
    """Return the responder style prompt for a domain (default: latest)."""
    doc = _load(domain)
    versions = doc.get("versions") or {}
    key = doc.get("latest", "v1") if version == "latest" else version
    entry = versions.get(key) or versions.get(doc.get("latest", "v1"))
    if not entry:
        return DEFAULT_PROMPT
    return str(entry.get("prompt") or DEFAULT_PROMPT)


def set_responder_prompt(domain: str, prompt: str) -> str:
    """Append a new prompt version and mark it latest. Returns the version id."""
    doc = _load(domain)
    versions = dict(doc.get("versions") or {})
    nums = []
    for key in versions:
        if key.startswith("v") and key[1:].isdigit():
            nums.append(int(key[1:]))
    next_n = (max(nums) if nums else 0) + 1
    version = f"v{next_n}"
    versions[version] = {
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    doc["versions"] = versions
    doc["latest"] = version
    doc["domain"] = domain
    _save(domain, doc)
    return version


__all__ = ["get_responder_prompt", "set_responder_prompt"]
