"""Slack notifier — post Block Kit messages to a channel."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

SLACK_LOG = Path(__file__).resolve().parent.parent.parent / "data" / "slack_notifications.log"
DEFAULT_API = "https://slack.com/api/chat.postMessage"


def notify_slack(channel: str, blocks: list[dict]) -> dict[str, Any]:
    """Post Block Kit `blocks` to a Slack `channel` (e.g. ``#incidents``).

    Uses ``SLACK_BOT_TOKEN`` + chat.postMessage when set; otherwise appends a
    mock record to ``data/slack_notifications.log`` so local demos stay offline.
    """
    record = {
        "notification_id": f"SLK-{uuid4().hex[:8].upper()}",
        "channel": channel,
        "blocks": blocks,
        "sent_at": datetime.now(UTC).isoformat(),
    }

    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if token:
        payload = {
            "channel": channel.lstrip("#") if channel.startswith("#") else channel,
            "blocks": blocks,
            "text": _fallback_text(blocks),
        }
        # Allow posting with a leading # via name; Slack API prefers channel id,
        # but bot tokens often accept #channel names when the bot is a member.
        if channel.startswith("#"):
            payload["channel"] = channel
        req = urllib.request.Request(
            os.getenv("SLACK_API_URL", DEFAULT_API).strip() or DEFAULT_API,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            record["status"] = "sent" if body.get("ok") else "error"
            record["slack_response"] = body
            if not body.get("ok"):
                logger.warning("Slack API error: %s", body.get("error"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            logger.warning("Slack notify failed: %s", exc)
    else:
        record["status"] = "mocked"
        logger.info("SLACK_BOT_TOKEN unset — mocking Slack post to %s", channel)

    SLACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SLACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "notification_id": record["notification_id"],
        "channel": channel,
        "status": record["status"],
    }


def _fallback_text(blocks: list[dict]) -> str:
    """Plain-text fallback required by Slack when blocks are present."""
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "section":
            text = block.get("text") or {}
            if isinstance(text, dict) and text.get("text"):
                parts.append(str(text["text"]))
        elif block.get("type") == "header":
            text = block.get("text") or {}
            if isinstance(text, dict) and text.get("text"):
                parts.append(str(text["text"]))
    return "\n".join(parts) or "RTTA ticket triage notification"


def p1_incident_blocks(state: dict) -> list[dict]:
    """Build Slack blocks for an approved P1 send."""
    raw = state.get("raw") or {}
    draft = state.get("draft") or {}
    classification = state.get("classification") or {}
    ticket_id = state.get("ticket_id", "unknown")
    subject = raw.get("subject") or draft.get("subject") or "(no subject)"
    category = classification.get("category", "unknown")
    sender = raw.get("sender", "unknown")
    preview = str(draft.get("body") or "")[:280]

    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"P1 incident approved: {ticket_id}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Subject:*\n{subject}"},
                {"type": "mrkdwn", "text": f"*Category:*\n{category}"},
                {"type": "mrkdwn", "text": f"*Sender:*\n{sender}"},
                {"type": "mrkdwn", "text": f"*Severity:*\n{state.get('severity')}"},
            ],
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Reply preview:*\n```{preview}```"},
        },
    ]


__all__ = ["notify_slack", "p1_incident_blocks"]
