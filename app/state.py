"""Ticket triage agent state."""

from __future__ import annotations

from typing import Literal, TypedDict

Domain = Literal["support", "it-helpdesk", "oncall"]
Severity = Literal["P1", "P2", "P3", "P4"]
Approval = Literal["pending", "approved", "edited", "rejected"]
Route = Literal["triager", "investigator", "responder", "hitl", "send", "END"]


class TicketState(TypedDict):
    ticket_id: str
    raw: dict
    domain: Domain
    classification: dict | None
    severity: Severity | None
    findings: list[dict]
    draft: dict | None
    approval: Approval
    sent: bool
    step_log: list[str]
    next: Route
