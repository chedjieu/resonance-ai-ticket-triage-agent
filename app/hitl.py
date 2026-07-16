"""Human-in-the-loop approval node."""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import interrupt

from app.hitl_log import append_hitl_outcome
from app.state import Approval, TicketState


def hitl_node(state: TicketState) -> dict:
    """Pause for human approval; resume with action approve/edit/reject."""
    queued_at = datetime.now(timezone.utc)
    payload = interrupt(
        {
            "draft": state["draft"],
            "classification": state["classification"],
            "severity": state["severity"],
            "findings": state["findings"],
            "raw": state["raw"],
            "queued_at": queued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    decided_at = datetime.now(timezone.utc)
    latency_s = max(0.0, (decided_at - queued_at).total_seconds())

    action = str(payload.get("action", "approve")).lower()
    edited_body = payload.get("edited_body")
    draft_before = dict(state["draft"]) if state["draft"] else {}
    draft = dict(draft_before)

    approval: Approval
    if action == "approve":
        approval = "approved"
        log = "HITL: approved"
    elif action == "edit":
        approval = "edited"
        if edited_body:
            draft["body"] = str(edited_body)
        log = "HITL: edited and approved"
    else:
        approval = "rejected"
        log = "HITL: rejected"

    try:
        append_hitl_outcome(
            {
                "ticket_id": state.get("ticket_id"),
                "domain": state.get("domain"),
                "action": action,
                "approval": approval,
                "severity": state.get("severity"),
                "classification": state.get("classification"),
                "draft_before": draft_before.get("body"),
                "draft_after": draft.get("body"),
                "recommended_action": draft_before.get("recommended_action"),
                "queued_at": queued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hitl_latency_seconds": round(latency_s, 3),
            }
        )
    except Exception:
        # Logging must never block the approval path.
        pass

    return {
        "approval": approval,
        "draft": draft,
        "step_log": state["step_log"] + [log],
    }
