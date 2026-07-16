"""Triager agent — classifies tickets by category and severity."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.guardrails import check_ticket_guardrail, is_guardrail_exception
from app.llm import get_chat_model, invoke_with_throttle_fallback
from app.state import Severity, TicketState

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class TriageOutput(BaseModel):
    category: str
    severity: Literal["P1", "P2", "P3", "P4"]
    confidence: float
    rationale: str


@lru_cache(maxsize=8)
def _load_taxonomy(domain: str) -> dict:
    path = DATA_DIR / domain / "taxonomy.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_taxonomy(taxonomy: dict) -> str:
    parts = [f"{c['name']}: {c['description']}" for c in taxonomy.get("categories", [])]
    severities = taxonomy.get("severities", ["P1", "P2", "P3", "P4"])
    return f"categories=[{'; '.join(parts)}]; severities={severities}"


def _valid_categories(taxonomy: dict) -> set[str]:
    return {c["name"] for c in taxonomy.get("categories", [])}


def triager_node(state: TicketState) -> dict:
    domain = state["domain"]
    taxonomy = _load_taxonomy(domain)
    valid = _valid_categories(taxonomy)
    raw = state["raw"] or {}
    ticket_text = f"{raw.get('subject', '')}\n{raw.get('body', '')}"
    refusal = check_ticket_guardrail(ticket_text)
    if refusal:
        return {
            "classification": None,
            "severity": None,
            "approval": "rejected",
            "step_log": state["step_log"] + [f"GUARDRAIL_REFUSAL: {refusal}"],
        }

    ticket = json.dumps(raw, ensure_ascii=False)

    # TODO: episodic memory examples
    system_prompt = (
        f"You are a {domain} triage analyst. "
        f"Available categories: {_format_taxonomy(taxonomy)}. "
        f"Given this ticket: {ticket}, choose the best category and severity. "
        "Provide a brief rationale. Be conservative on severity."
    )

    def run_triage() -> TriageOutput:
        llm = get_chat_model().with_structured_output(TriageOutput)
        return llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=ticket),
            ]
        )

    try:
        out = invoke_with_throttle_fallback(run_triage)
    except Exception as exc:
        if is_guardrail_exception(exc):
            return {
                "classification": None,
                "severity": None,
                "approval": "rejected",
                "step_log": state["step_log"]
                + [f"GUARDRAIL_REFUSAL: Bedrock/Vertex guardrail intervened ({exc})"],
            }
        raise

    category = out.category if out.category in valid else "unknown"
    severity: Severity = out.severity if category != "unknown" else "P3"

    log = f"Triager: {category} / {severity} (confidence={out.confidence:.2f})"
    if category == "unknown":
        log += " [category not in taxonomy — defaulted to unknown/P3]"

    return {
        "classification": {
            "category": category,
            "confidence": out.confidence,
            "rationale": out.rationale,
        },
        "severity": severity,
        "step_log": state["step_log"] + [log],
    }
