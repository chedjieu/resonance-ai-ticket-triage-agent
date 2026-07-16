"""Responder agent — draft reply using procedural, episodic, and semantic memory."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_chat_model, invoke_with_throttle_fallback
from app.memory.episodic import similar_past_cases
from app.memory.procedural import get_responder_prompt
from app.memory.semantic import recall_user
from app.state import TicketState

RISK_PHRASES = ("refund", "credit", "guarantee", "tomorrow", "by eod")
INJECTION_MARKERS = (
    "ignore your guidelines",
    "skip approval",
    "skip the approval",
    "bypass approval",
    "reply rudely",
    "send_response",
    "exfiltrate",
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class ResponderOutput(BaseModel):
    subject: str
    body: str
    recommended_action: Literal["send", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)


def _post_process(out: ResponderOutput, ticket_text: str = "") -> ResponderOutput:
    flags = list(out.risk_flags)
    body_l = out.body.lower()
    ticket_l = (ticket_text or "").lower()
    force_escalate = False

    if out.confidence < 0.6:
        flags.append("low_confidence")
        force_escalate = True

    for phrase in RISK_PHRASES:
        if phrase in body_l or phrase in ticket_l:
            flags.append(f"phrase:{phrase}")
            force_escalate = True

    for marker in INJECTION_MARKERS:
        if marker in ticket_l:
            flags.append("prompt_injection")
            force_escalate = True
            break

    if EMAIL_RE.search(out.body) or EMAIL_RE.search(ticket_text or ""):
        flags.append("pii:email")
        force_escalate = True
    if PHONE_RE.search(out.body) or PHONE_RE.search(ticket_text or ""):
        flags.append("pii:phone")
        force_escalate = True
    if SSN_RE.search(out.body) or SSN_RE.search(ticket_text or ""):
        flags.append("pii:ssn")
        force_escalate = True

    seen: set[str] = set()
    unique: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    action = "escalate" if force_escalate else out.recommended_action

    return ResponderOutput(
        subject=out.subject,
        body=out.body,
        recommended_action=action,
        confidence=out.confidence,
        risk_flags=unique,
    )


def _build_human_message(state: TicketState, episodic: list[dict], memories: list[dict]) -> str:
    few_shot = []
    for i, case in enumerate(episodic, start=1):
        few_shot.append(
            f"Example {i}:\n"
            f"Ticket: {case.get('ticket_text', '')}\n"
            f"Resolution: {case.get('resolution_text', '')}"
        )
    few_shot_block = "\n\n".join(few_shot) if few_shot else "(none)"

    mem_lines = [m.get("content", "") for m in memories if m.get("content")]
    mem_block = "\n".join(f"- {line}" for line in mem_lines) if mem_lines else "(none)"

    payload = {
        "raw": state["raw"],
        "classification": state["classification"],
        "severity": state["severity"],
        "findings": state["findings"],
    }
    return (
        f"Few-shot past cases:\n{few_shot_block}\n\n"
        f"What we know about this user:\n{mem_block}\n\n"
        f"Current ticket context (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Draft a customer reply. Set recommended_action to send or escalate. "
        "Include risk_flags for any concerns."
    )


def responder_node(state: TicketState) -> dict:
    domain = state["domain"]
    raw = state["raw"] or {}
    ticket_body = str(raw.get("body") or raw.get("subject") or "")
    sender = str(raw.get("sender") or "anonymous")

    style = get_responder_prompt(domain, version="latest")
    episodic = similar_past_cases(ticket_body, domain, k=3)
    memories = recall_user(sender, k=3)

    system = SystemMessage(content=style)
    human = HumanMessage(content=_build_human_message(state, episodic, memories))

    def run_draft() -> ResponderOutput:
        llm = get_chat_model().with_structured_output(ResponderOutput)
        return llm.invoke([system, human])

    drafted = invoke_with_throttle_fallback(run_draft)
    out = _post_process(drafted, ticket_text=ticket_body)

    log = (
        f"Responder: action={out.recommended_action} "
        f"confidence={out.confidence:.2f} "
        f"episodic={len(episodic)} semantic={len(memories)}"
    )
    if out.risk_flags:
        log += f" flags={out.risk_flags}"

    return {
        "draft": out.model_dump(),
        "approval": "pending",
        "step_log": state["step_log"] + [log],
    }
