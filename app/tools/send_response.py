"""Mock outbound ticket response sender."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import tool

SENT_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "sent_responses.log"


@tool
def send_response(ticket_id: str, subject: str, body: str, recipient: str) -> dict:
    """Send the approved response to the customer (mock — appends to a log file)."""
    out_id = f"OUT-{uuid4().hex[:8].upper()}"
    record = {
        "out_id": out_id,
        "ticket_id": ticket_id,
        "subject": subject,
        "body": body,
        "recipient": recipient,
        "sent_at": datetime.now(UTC).isoformat(),
    }
    SENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"out_id": out_id, "ticket_id": ticket_id, "status": "sent"}
