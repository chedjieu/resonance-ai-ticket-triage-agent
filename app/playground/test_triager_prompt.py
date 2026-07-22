#!/usr/bin/env python
"""Print the triage system prompt and run structured output against the sample ticket.

Run:
    cd RTTA-AI-Multi-Agent-Ticket-Triage
    uv run python -m app.playground.test_triager_prompt

Uses RTTA_MODEL from RAIRA-AI-Research-Assistant/.env (Bedrock by default).
"""

from __future__ import annotations

import json
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.triager import (
    TriageOutput,
    _format_taxonomy,
    _load_taxonomy,
    _valid_categories,
)
from app.graph import SAMPLE_TICKET, make_initial_state
from app.llm import get_chat_model, invoke_with_throttle_fallback


def main() -> int:
    domain = sys.argv[1] if len(sys.argv) > 1 else "support"
    state = make_initial_state("TKT-1001", SAMPLE_TICKET, domain)
    taxonomy = _load_taxonomy(domain)
    ticket = json.dumps(state["raw"], ensure_ascii=False)

    system_prompt = (
        f"You are a {domain} triage analyst. "
        f"Available categories: {_format_taxonomy(taxonomy)}. "
        f"Given this ticket: {ticket}, choose the best category and severity. "
        "Provide a brief rationale. Be conservative on severity."
    )

    print("=== SYSTEM PROMPT ===")
    print(system_prompt)
    print()
    print("=== HUMAN (ticket JSON) ===")
    print(ticket)
    print()

    def run() -> TriageOutput:
        llm = get_chat_model().with_structured_output(TriageOutput)
        return llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=ticket),
            ]
        )

    out = invoke_with_throttle_fallback(run)
    valid = _valid_categories(taxonomy)
    category = out.category if out.category in valid else "unknown"
    severity = out.severity if category != "unknown" else "P3"

    print("=== TRIAGE OUTPUT ===")
    print(f"category:   {category}")
    print(f"severity:   {severity}")
    print(f"confidence: {out.confidence:.2f}")
    print(f"rationale:  {out.rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
