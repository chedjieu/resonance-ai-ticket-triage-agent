"""Supervisor — routes tickets to the next worker."""

from __future__ import annotations

from app.state import Route, TicketState


def supervisor_node(state: TicketState) -> dict:
    """Decide the next node based on current ticket progress."""
    nxt: Route = "END"

    # Guardrail / operator rejection ends the run (no further workers).
    if state["approval"] == "rejected":
        nxt = "END"
    elif state["classification"] is None:
        nxt = "triager"
    elif state["findings"] == []:
        nxt = "investigator"
    elif state["draft"] is None:
        nxt = "responder"
    elif state["approval"] == "pending":
        nxt = "hitl"
    elif state["approval"] in ("approved", "edited") and not state["sent"]:
        nxt = "send"
    else:
        nxt = "END"

    return {
        "next": nxt,
        "step_log": state["step_log"] + [f"Supervisor: route -> {nxt}"],
    }
