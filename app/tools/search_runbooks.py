"""Search runbooks for the current ticket domain."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import psycopg
from langchain_core.tools import tool

from app.tools._domain import get_domain
from app.tools.search_local_docs import DEFAULT_DSN, search_local_docs

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_SOURCE_RE = re.compile(r"<!--\s*source:\s*(.+?)\s*-->")


def _keyword_score(query: str, text: str) -> float:
    q_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query) if len(w) > 2}
    if not q_words:
        return 0.0
    body = text.lower()
    hits = sum(1 for w in q_words if w in body)
    return hits / len(q_words)


def _postgres_available(dsn: str | None = None) -> bool:
    dsn = dsn or os.getenv("POSTGRES_DSN", DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _search_runbooks_from_files(domain: str, query: str, k: int) -> list[dict]:
    """Offline fallback: keyword search over data/{domain}/runbooks/*.md."""
    runbook_dir = DATA_DIR / domain / "runbooks"
    if not runbook_dir.is_dir():
        return []

    scored: list[tuple[float, dict]] = []
    for path in sorted(runbook_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        source = str(path)
        match = _SOURCE_RE.search(text)
        if match:
            source = match.group(1).strip()
        score = _keyword_score(query, text)
        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "chunk_id": path.stem,
                    "source_url": source,
                    "score": score,
                    "text": text[:2000],
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        for path in sorted(runbook_dir.glob("*.md"))[:k]:
            text = path.read_text(encoding="utf-8")
            source = str(path)
            match = _SOURCE_RE.search(text)
            if match:
                source = match.group(1).strip()
            scored.append(
                (
                    0.1,
                    {
                        "chunk_id": path.stem,
                        "source_url": source,
                        "score": 0.1,
                        "text": text[:2000],
                    },
                )
            )

    return [item for _, item in scored[:k]]


@tool
def search_runbooks(query: str, k: int = 3) -> list[dict]:
    """Search domain runbooks for remediation guidance relevant to the query."""
    domain = get_domain()
    mode = os.getenv("MONK_RUNBOOKS", "auto").strip().lower()

    if mode == "file" or (mode == "auto" and not _postgres_available()):
        if mode == "auto":
            logger.info("Postgres unavailable — searching runbooks from data/%s/runbooks/", domain)
        return _search_runbooks_from_files(domain, query, k)

    table = "runbooks_" + domain.replace("-", "_")
    try:
        return search_local_docs(query=query, k=k, table=table)
    except Exception as exc:
        logger.warning("Runbook pgvector search failed (%s) — using file fallback", exc)
        return _search_runbooks_from_files(domain, query, k)
