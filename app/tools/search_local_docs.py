"""Vector search over ingested local documentation."""

from __future__ import annotations

import os
import re

import psycopg
from langchain_core.tools import tool

from app.llm import get_embeddings

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/resonance"


def _sanitize_table(table: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError(f"invalid table name: {table!r}")
    return table


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def search_local_docs(query: str, k: int = 5, table: str = "docs") -> list[dict]:
    """Search an ingested document corpus. Returns citations with source_url and text."""
    table = _sanitize_table(table)
    embedder = get_embeddings()
    vec = _vec_literal(embedder.embed_query(query))
    dsn = os.getenv("POSTGRES_DSN", DEFAULT_DSN)

    sql = f"""
        SELECT chunk_id, source_url, 1 - (embedding <=> %s::vector) AS score, text
        FROM {table}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with psycopg.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, vec, k))
        rows = cur.fetchall()

    return [
        {
            "chunk_id": str(chunk_id),
            "source_url": str(source_url),
            "score": float(score),
            "text": str(text),
        }
        for chunk_id, source_url, score, text in rows
    ]


@tool
def search_local_docs_tool(query: str, k: int = 5, table: str = "docs") -> list[dict]:
    """Search the ingested document corpus for content relevant to a query."""
    return search_local_docs(query=query, k=k, table=table)
