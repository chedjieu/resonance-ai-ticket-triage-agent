"""Human-in-the-loop approval node."""

from __future__ import annotations

from langgraph.types import interrupt

from app.state import Approval, TicketState


def hitl_node(state: TicketState) -> dict:
    """Pause for human approval; resume with action approve/edit/reject."""
    payload = interrupt(
        {
            "draft": state["draft"],
            "classification": state["classification"],
            "severity": state["severity"],
            "findings": state["findings"],
            "raw": state["raw"],
        }
    )

    action = str(payload.get("action", "approve")).lower()
    edited_body = payload.get("edited_body")
    draft = dict(state["draft"]) if state["draft"] else {}

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

    return {
        "approval": approval,
        "draft": draft,
        "step_log": state["step_log"] + [log],
    }
