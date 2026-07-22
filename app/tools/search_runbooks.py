"""Search runbooks via AWS Bedrock Knowledge Bases (with offline file fallback)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.tools._domain import get_domain

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


def _kb_id_for_domain(domain: str) -> str:
    """Resolve Knowledge Base id: per-domain env, then shared fallbacks."""
    safe = domain.upper().replace("-", "_")
    for key in (
        f"BEDROCK_KB_ID_{safe}",
        f"BEDROCK_KNOWLEDGE_BASE_ID_{safe}",
        "BEDROCK_KB_ID",
        "BEDROCK_KNOWLEDGE_BASE_ID",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _source_url_from_location(location: dict[str, Any] | None) -> str:
    if not location:
        return ""
    loc_type = location.get("type")
    if loc_type == "S3":
        s3 = location.get("s3Location") or {}
        return str(s3.get("uri") or "")
    if loc_type == "WEB":
        web = location.get("webLocation") or {}
        return str(web.get("url") or "")
    if loc_type == "CONFLUENCE":
        conf = location.get("confluenceLocation") or {}
        return str(conf.get("url") or "")
    if loc_type == "SALESFORCE":
        sf = location.get("salesforceLocation") or {}
        return str(sf.get("url") or "")
    if loc_type == "SHAREPOINT":
        sp = location.get("sharePointLocation") or {}
        return str(sp.get("url") or "")
    if loc_type == "CUSTOM":
        custom = location.get("customDocumentLocation") or {}
        return str(custom.get("id") or "")
    return ""


def _search_runbooks_bedrock_kb(domain: str, query: str, k: int) -> list[dict]:
    """Retrieve runbook chunks from an Amazon Bedrock Knowledge Base."""
    import boto3

    kb_id = _kb_id_for_domain(domain)
    if not kb_id:
        raise RuntimeError(
            "BEDROCK_KB_ID (or BEDROCK_KNOWLEDGE_BASE_ID) is not set — "
            "cannot query Bedrock Knowledge Bases"
        )

    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("BEDROCK_REGION")
        or "us-east-1"
    )
    # Scope the query to the active ticket domain when one shared KB holds all runbooks.
    retrieval_text = f"[{domain} runbooks] {query}"

    client = boto3.client("bedrock-agent-runtime", region_name=region)
    response = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": retrieval_text},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": max(1, int(k)),
            }
        },
    )

    results: list[dict] = []
    for i, item in enumerate(response.get("retrievalResults") or []):
        content = item.get("content") or {}
        text = str(content.get("text") or "")
        location = item.get("location") or {}
        source_url = _source_url_from_location(location)
        metadata = item.get("metadata") or {}
        chunk_id = str(
            metadata.get("x-amz-bedrock-kb-chunk-id")
            or metadata.get("chunk_id")
            or f"kb-{i}"
        )
        score = float(item.get("score") or 0.0)
        results.append(
            {
                "chunk_id": chunk_id,
                "source_url": source_url or f"bedrock-kb://{kb_id}/{chunk_id}",
                "score": score,
                "text": text[:2000],
            }
        )
    return results


@tool
def search_runbooks(query: str, k: int = 3) -> list[dict]:
    """Search domain runbooks for remediation guidance relevant to the query."""
    domain = get_domain()
    mode = os.getenv("RTTA_RUNBOOKS", "auto").strip().lower()

    # Explicit offline mode, or auto with no KB configured.
    if mode == "file" or (mode == "auto" and not _kb_id_for_domain(domain)):
        if mode == "auto":
            logger.info(
                "Bedrock KB id unset — searching runbooks from data/%s/runbooks/",
                domain,
            )
        return _search_runbooks_from_files(domain, query, k)

    if mode == "bedrock" or mode == "auto":
        try:
            return _search_runbooks_bedrock_kb(domain, query, k)
        except Exception as exc:
            logger.warning("Bedrock Knowledge Base retrieve failed (%s) — using file fallback", exc)
            return _search_runbooks_from_files(domain, query, k)

    # Unknown RTTA_RUNBOOKS value — keep previous file behaviour.
    logger.warning("Unknown RTTA_RUNBOOKS=%r — using file fallback", mode)
    return _search_runbooks_from_files(domain, query, k)
