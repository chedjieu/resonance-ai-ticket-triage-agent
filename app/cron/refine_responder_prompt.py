"""Auto-prompt-tuning cron — propose a v+1 responder procedural prompt from HITL edits.

Pulls the last N HITL outcomes, asks an LLM to summarise common human edits, and
prints a proposed next prompt version to stdout. Does NOT write to disk; the
instructor reviews and applies manually via procedural memory if desired.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import os

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PRE_IMPORT_MODEL = os.environ.get("RTTA_MODEL", "").strip()
importlib.import_module("app")

from langchain_core.messages import HumanMessage, SystemMessage

from app.hitl_log import DEFAULT_LOG, load_hitl_outcomes
from app.llm import get_chat_model
from app.memory.procedural import get_responder_prompt
from app._fake_llm import is_fake_chat_model

if _PRE_IMPORT_MODEL.lower() in ("fake", "stub", "offline"):
    os.environ["RTTA_MODEL"] = _PRE_IMPORT_MODEL
    get_chat_model.cache_clear()

REFINE_SYSTEM = (
    "You help improve a support-agent system prompt using human-in-the-loop edits. "
    "Be concrete and conservative. Do not invent new product policies. "
    "Preserve safety rules about refunds, credits, SLAs, timelines, and PII."
)

REFINE_HUMAN = """Domain: {domain}
Current procedural prompt (version tip → latest):
---
{current_prompt}
---

HITL outcomes (last {n}, JSON). Focus on rows where action is "edit" or "reject",
and on differences between draft_before and draft_after:
---
{outcomes_json}
---

Tasks:
1. Summarise the common patterns in human edits/rejections (bullet list).
2. Propose an improved system prompt that would reduce those edits next time.
3. Keep the prompt as a single plain-text instruction block (no markdown fences).

Return JSON only:
{{
  "summary": ["...", "..."],
  "proposed_prompt": "...",
  "rationale": "one short paragraph"
}}
"""


def _parse_proposal(text: str) -> dict:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    blob = fence.group(1).strip() if fence else text.strip()
    try:
        data = json.loads(blob)
        if isinstance(data, dict) and data.get("proposed_prompt"):
            return data
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return {
        "summary": ["(could not parse structured LLM reply — raw text follows)"],
        "proposed_prompt": text.strip(),
        "rationale": "fallback: used raw model output as proposed_prompt",
    }


def _heuristic_proposal(current: str, outcomes: list[dict]) -> dict:
    """Offline fallback when the model is fake or returns unusable JSON."""
    edits = [o for o in outcomes if o.get("action") == "edit"]
    rejects = [o for o in outcomes if o.get("action") == "reject"]
    summary = [
        f"{len(edits)} edits and {len(rejects)} rejects in the last {len(outcomes)} HITL outcomes.",
    ]
    themes: list[str] = []
    for o in edits:
        before = str(o.get("draft_before") or "")
        after = str(o.get("draft_after") or "")
        if not after or after == before:
            continue
        if "sorry" in after.lower() and "sorry" not in before.lower():
            themes.append("humans often add a clearer apology / empathy opener")
        if len(after) < len(before) * 0.85:
            themes.append("humans shorten drafts — prefer concise replies")
        if "escalate" in after.lower() or "specialist" in after.lower():
            themes.append("humans add escalation / specialist handoff language")
        if "eod" in before.lower() or "tomorrow" in before.lower():
            if "eod" not in after.lower() and "tomorrow" not in after.lower():
                themes.append("humans remove invented timelines (EOD / tomorrow)")
    # Deduplicate while keeping order
    seen: set[str] = set()
    for t in themes:
        if t not in seen:
            seen.add(t)
            summary.append(t)

    addendum_bits = []
    blob = " ".join(summary).lower()
    if "concise" in blob or "shorten" in blob:
        addendum_bits.append("Keep replies under ~120 words unless steps require more.")
    if "apology" in blob or "empathy" in blob:
        addendum_bits.append("Open with one short empathy sentence before steps.")
    if "timeline" in blob or "eod" in blob:
        addendum_bits.append("Never promise same-day / EOD / tomorrow timelines.")
    if "escalat" in blob:
        addendum_bits.append("When unsure, say a specialist will follow up rather than guessing.")
    if not addendum_bits:
        addendum_bits.append(
            "Mirror HITL edits: be concrete, avoid over-promising, and escalate billing/PII."
        )

    proposed = current.rstrip() + "\n\nAdditional guidance from recent HITL review:\n- "
    proposed += "\n- ".join(addendum_bits)
    return {
        "summary": summary,
        "proposed_prompt": proposed,
        "rationale": "Heuristic proposal from edit/reject patterns (fake/offline path).",
    }


def propose_prompt(domain: str, outcomes: list[dict], current: str) -> dict:
    if not outcomes:
        return {
            "summary": ["No HITL outcomes found — nothing to refine."],
            "proposed_prompt": current,
            "rationale": "Empty log; returning the current prompt unchanged.",
        }

    # Offline fake model has no useful refine behaviour — use heuristics.
    if is_fake_chat_model(os.getenv("RTTA_MODEL", "")):
        return _heuristic_proposal(current, outcomes)

    try:
        llm = get_chat_model()
        reply = llm.invoke(
            [
                SystemMessage(content=REFINE_SYSTEM),
                HumanMessage(
                    content=REFINE_HUMAN.format(
                        domain=domain,
                        current_prompt=current,
                        n=len(outcomes),
                        outcomes_json=json.dumps(outcomes, ensure_ascii=False, indent=2)[:12000],
                    )
                ),
            ]
        )
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        parsed = _parse_proposal(content)
        proposed = str(parsed.get("proposed_prompt") or "")
        # Reject clearly-wrong stubs (e.g. unrelated refusal messages).
        bad = (
            len(proposed) < 80
            or "cooking" in proposed.lower()
            or "research assistant" in proposed.lower()
        )
        if bad:
            return _heuristic_proposal(current, outcomes)
        return parsed
    except Exception:
        return _heuristic_proposal(current, outcomes)


def next_version_label(domain: str) -> str:
    """Best-effort v+1 label from the on-disk prompt file (display only)."""
    path = _ROOT / "data" / "prompts" / f"responder_{domain}.json"
    if not path.exists():
        return "v2"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        latest = str(doc.get("latest") or "v1")
        if latest.startswith("v") and latest[1:].isdigit():
            return f"v{int(latest[1:]) + 1}"
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return "v+1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Propose a v+1 responder procedural prompt from recent HITL edits."
    )
    parser.add_argument("--domain", default="support", help="Responder domain (default: support)")
    parser.add_argument("--limit", type=int, default=50, help="Max HITL outcomes to read (default: 50)")
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help=f"HITL JSONL path (default: {DEFAULT_LOG})",
    )
    args = parser.parse_args(argv)

    outcomes = load_hitl_outcomes(path=args.log, limit=args.limit)
    # Prefer same-domain rows when available; otherwise use the full window.
    domain_rows = [o for o in outcomes if o.get("domain") in (None, args.domain)]
    rows = domain_rows if domain_rows else outcomes
    current = get_responder_prompt(args.domain, version="latest")
    proposal = propose_prompt(args.domain, rows, current)
    version = next_version_label(args.domain)

    print("=" * 72)
    print(f"Responder prompt refine proposal - domain={args.domain} -> {version}")
    print(f"HITL log: {args.log}  (using {len(rows)} of last {args.limit})")
    print("=" * 72)
    print("\n## Common HITL edit / reject patterns\n")
    for item in proposal.get("summary") or []:
        print(f"- {item}")
    print("\n## Rationale\n")
    print(proposal.get("rationale") or "(none)")
    print(f"\n## Proposed {version} prompt\n")
    print(proposal.get("proposed_prompt") or current)
    print("\n" + "=" * 72)
    print(
        "NOT APPLIED. Instructor: review above, then manually update "
        f"data/prompts/responder_{args.domain}.json (or call set_responder_prompt) if approved."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
