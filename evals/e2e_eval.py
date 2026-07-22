"""End-to-end eval — full pipeline vs golden category/response/escalation."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langsmith.evaluation import evaluate

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
importlib.import_module("app")

from app._fake_llm import is_fake_chat_model
from app.graph import build_graph, make_initial_state
from app.llm import get_chat_model
from evals._common import EVALS_DIR, eval_data, load_jsonl, should_upload

GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
DATASET_NAME = "rtta-ticket-e2e-golden"
EXPERIMENT = "e2e-eval"
JUDGE_PROMPT = (
    "On a scale of 1-5, is this support reply reasonable for the ticket?\n"
    "Ticket: {ticket}\nDraft: {draft}\n\n"
    'Return JSON: {{"score": <1-5>, "feedback": "<short>"}}'
)
RESPONSE_PASS = 3.0


def _has_pending_interrupt(graph, config: dict) -> bool:
    snap = graph.get_state(config)
    for task in snap.tasks or ():
        if getattr(task, "interrupts", None):
            return True
    return False


def run_full_pipeline(inputs: dict) -> dict:
    domain = inputs.get("domain", "support")
    ticket = inputs["ticket"]
    ticket_id = str(inputs.get("id") or ticket.get("id") or "eval")
    graph = build_graph()
    config = {"configurable": {"thread_id": f"e2e-{ticket_id}-{uuid.uuid4().hex[:8]}"}}
    state = make_initial_state(ticket_id, ticket, domain)  # type: ignore[arg-type]

    for _ in graph.stream(state, config, stream_mode="updates"):
        pass

    # Auto-approve HITL for offline eval (escalation still reflected in draft action).
    while _has_pending_interrupt(graph, config):
        graph.invoke(Command(resume={"action": "approve", "edited_body": None}), config)

    final = graph.get_state(config).values
    classification = final.get("classification") or {}
    draft = final.get("draft") or {}
    return {
        "category": classification.get("category"),
        "severity": final.get("severity"),
        "recommended_action": draft.get("recommended_action"),
        "subject": draft.get("subject"),
        "body": draft.get("body"),
        "approval": final.get("approval"),
        "sent": final.get("sent"),
        "findings": final.get("findings") or [],
    }


def category_evaluator(run, example) -> dict:
    pred = run.outputs.get("category")
    expected = example.inputs.get("expected_category")
    return {"key": "category_exact", "score": 1.0 if pred == expected else 0.0}


def escalation_evaluator(run, example) -> dict:
    pred = run.outputs.get("recommended_action")
    expected = example.inputs.get("expected_action")
    return {"key": "escalation_exact", "score": 1.0 if pred == expected else 0.0}


def _parse_quality(text: str) -> tuple[float, str]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    blob = fence.group(1).strip() if fence else text.strip()
    try:
        data = json.loads(blob)
        return max(1.0, min(5.0, float(data.get("score", 0)))), str(data.get("feedback", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = re.search(r"[1-5](?:\.\d+)?", text)
        return (float(match.group()) if match else 2.0), text[:200]


def _fake_response_score(ticket: dict, body: str) -> tuple[float, str]:
    if not (body or "").strip():
        return 1.0, "empty response"
    blob = " ".join(
        [
            str(ticket.get("subject", "")),
            str(ticket.get("body", "")),
        ]
    ).lower()
    words = {w for w in re.findall(r"[a-z]{4,}", blob)}
    body_l = body.lower()
    hits = sum(1 for w in words if w in body_l)
    ratio = hits / max(len(words), 1)
    score = max(1.0, min(5.0, 2.5 + 2.5 * ratio))
    return score, f"fake judge coverage={ratio:.2f}"


def response_quality(ticket: dict, draft_body: str) -> tuple[float, str]:
    if is_fake_chat_model(os.getenv("RTTA_MODEL", "")):
        return _fake_response_score(ticket, draft_body)
    prompt = JUDGE_PROMPT.format(
        ticket=json.dumps(ticket, ensure_ascii=False),
        draft=(draft_body or "")[:3000],
    )
    try:
        reply = get_chat_model().invoke([HumanMessage(content=prompt)])
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return _parse_quality(content)
    except Exception as exc:
        score, _ = _fake_response_score(ticket, draft_body)
        return score, f"judge fallback: {exc}"


def response_evaluator(run, example) -> dict:
    score, feedback = response_quality(
        example.inputs.get("ticket") or {},
        run.outputs.get("body") or "",
    )
    return {"key": "response_quality", "score": score, "comment": feedback}


def print_summary(results) -> float:
    passed = 0
    total = 0
    for row in results:
        total += 1
        cat_ok = esc_ok = False
        quality = 0.0
        feedback = ""
        for result in row["evaluation_results"]["results"]:
            if result.key == "category_exact" and result.score is not None:
                cat_ok = float(result.score) >= 1.0
            if result.key == "escalation_exact" and result.score is not None:
                esc_ok = float(result.score) >= 1.0
            if result.key == "response_quality" and result.score is not None:
                quality = float(result.score)
                feedback = result.comment or ""

        ok = cat_ok and esc_ok and quality >= RESPONSE_PASS
        if ok:
            passed += 1
        tid = row["example"].inputs.get("id", "?")
        print(
            f"{'PASS' if ok else 'FAIL'}  {tid}  "
            f"cat={'Y' if cat_ok else 'N'} esc={'Y' if esc_ok else 'N'} "
            f"response={quality:.1f}"
        )
        if feedback and not ok:
            print(f"       {feedback[:120]}")

    rate = passed / total if total else 0.0
    print(f"\nPass-rate: {passed}/{total} ({100 * rate:.0f}%)")
    if getattr(results, "url", None):
        print(f"LangSmith: {results.url}")
    return rate


def main() -> None:
    rows = load_jsonl(GOLDEN_PATH)
    data = eval_data(rows, DATASET_NAME, "Ticket triage end-to-end golden")
    results = evaluate(
        run_full_pipeline,
        data=data,
        evaluators=[category_evaluator, escalation_evaluator, response_evaluator],
        experiment_prefix=EXPERIMENT,
        description="E2E category + response quality + escalation",
        upload_results=should_upload(),
    )
    rate = print_summary(results)
    if rate < 1.0 and os.getenv("EVAL_STRICT", "").lower() in ("1", "true", "yes"):
        sys.exit(1)


if __name__ == "__main__":
    main()
