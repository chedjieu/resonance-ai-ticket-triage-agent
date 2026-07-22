"""Episodic memory — similar past resolutions via pgvector (read-only at runtime)."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import psycopg

from app.llm import get_embeddings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/resonance"
TABLE = "past_resolutions"


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _postgres_available(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _from_postgres(ticket_text: str, domain: str, k: int) -> list[dict] | None:
    dsn = os.getenv("POSTGRES_DSN", DEFAULT_DSN)
    if not _postgres_available(dsn):
        return None

    try:
        embedder = get_embeddings()
        vec = _vec_literal(embedder.embed_query(ticket_text))
        sql = f"""
            SELECT ticket_text, resolution_text,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE}
            WHERE domain = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(sql, (vec, domain, vec, k))
            rows = cur.fetchall()
        return [
            {
                "ticket_text": str(ticket_text_),
                "resolution_text": str(resolution_text),
                "score": float(score),
                "source": "past_resolutions",
            }
            for ticket_text_, resolution_text, score in rows
        ]
    except Exception as exc:
        logger.warning("Episodic pgvector search failed (%s) — using file fallback", exc)
        return None


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in {"the", "and", "for", "with"}}


def _from_file(ticket_text: str, domain: str, k: int) -> list[dict]:
    path = DATA_DIR / domain / "historical_tickets.jsonl"
    if not path.exists():
        return []

    query_tokens = _tokenize(ticket_text)
    scored: list[tuple[float, dict]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            body = str(row.get("body") or row.get("ticket_text") or "")
            subject = str(row.get("subject") or "")
            ticket = f"{subject}\n{body}".strip()
            resolution = str(row.get("resolution") or row.get("resolution_text") or "")
            overlap = len(query_tokens & _tokenize(f"{ticket} {resolution}"))
            score = overlap / max(len(query_tokens), 1)
            scored.append(
                (
                    score,
                    {
                        "ticket_text": ticket,
                        "resolution_text": resolution,
                        "score": float(score),
                        "source": f"file:{path.name}",
                    },
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]


def similar_past_cases(ticket_text: str, domain: str, k: int = 3) -> list[dict]:
    """Return up to k similar past ticket → resolution pairs for few-shot prompting."""
    hits = _from_postgres(ticket_text, domain, k)
    if hits is not None:
        return hits
    return _from_file(ticket_text, domain, k)


__all__ = ["similar_past_cases"]
