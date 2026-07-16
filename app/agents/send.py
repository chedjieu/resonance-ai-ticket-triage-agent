"""Send agent — dispatches approved responses."""

from __future__ import annotations

from app.state import TicketState
from app.tools.notify_slack import notify_slack, p1_incident_blocks
from app.tools.send_response import send_response


def send_node(state: TicketState) -> dict:
    if state["approval"] not in ("approved", "edited"):
        return {"step_log": state["step_log"] + ["Send: skipped (not approved)"]}

    draft = state["draft"] or {}
    raw = state["raw"]
    result = send_response.invoke(
        {
            "ticket_id": state["ticket_id"],
            "subject": draft.get("subject", "Re: your ticket"),
            "body": draft.get("body", ""),
            "recipient": raw.get("sender", "unknown@example.com"),
        }
    )
    out_id = result.get("out_id", "unknown")
    logs = [f"Send: response dispatched ({out_id})"]

    # After a P1 is approved and sent, also notify #incidents.
    if state.get("severity") == "P1":
        try:
            slack = notify_slack("#incidents", p1_incident_blocks(state))
            logs.append(
                f"Send: Slack #incidents notified "
                f"({slack.get('notification_id')}, status={slack.get('status')})"
            )
        except Exception as exc:
            logs.append(f"Send: Slack #incidents notify failed ({exc})")

    return {
        "sent": True,
        "step_log": state["step_log"] + logs,
    }
