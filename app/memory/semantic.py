"""Semantic memory — per-user facts via LangGraph Store."""

from __future__ import annotations

import os
from uuid import uuid4

from langgraph.store.base import BaseStore

_store: BaseStore | None = None


def get_store() -> BaseStore:
    """Return the shared store (InMemoryStore or PostgresStore)."""
    global _store
    if _store is not None:
        return _store

    backend = os.getenv("RTTA_MEMORY", "memory").strip().lower()
    if backend == "postgres":
        from langgraph.store.postgres import PostgresStore

        dsn = os.getenv(
            "POSTGRES_DSN",
            "postgresql://postgres:postgres@localhost:5433/resonance",
        )
        store = PostgresStore.from_conn_string(dsn)
        store.setup()
        _store = store
        return _store

    from langgraph.store.memory import InMemoryStore

    _store = InMemoryStore()
    return _store


def _namespace(user_id: str) -> tuple[str, ...]:
    # LangGraph namespaces cannot contain '.' — sanitize email-like ids.
    label = (user_id or "anonymous").strip().replace(".", "_").replace("@", "_at_")
    return ("users", label)


def recall_user(user_id: str, k: int = 3) -> list[dict]:
    """Return up to k stored facts/preferences for this user."""
    store = get_store()
    ns = _namespace(user_id)
    try:
        items = store.search(ns, query=user_id or "preferences", limit=k)
    except Exception:
        items = []

    if not items:
        # Fall back to listing keys when the store has no search index.
        try:
            items = list(store.search(ns, limit=k))
        except Exception:
            return []

    out: list[dict] = []
    for item in items[:k]:
        value = item.value if isinstance(item.value, dict) else {"content": str(item.value)}
        out.append(
            {
                "key": item.key,
                "content": str(value.get("content", "")),
                "kind": str(value.get("kind", "fact")),
                "score": getattr(item, "score", None),
            }
        )
    return out


def remember_user(user_id: str, content: str, kind: str = "fact") -> str:
    """Persist a per-user fact. Returns the store key."""
    store = get_store()
    key = str(uuid4())
    store.put(
        _namespace(user_id),
        key,
        {"content": content, "kind": kind, "user_id": user_id},
    )
    return key


__all__ = ["get_store", "recall_user", "remember_user"]
