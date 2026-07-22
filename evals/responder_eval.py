"""Responder eval — escalation precision/recall + LLM quality 1-5."""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langsmith.evaluation import evaluate

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
importlib.import_module("app")

from app._fake_llm import is_fake_chat_model
from app.agents.responder import responder_node
from app.llm import get_chat_model
from evals._common import EVALS_DIR, empty_ticket_state, eval_data, load_jsonl, should_upload

GOLDEN_PATH = EVALS_DIR / "responder_golden.jsonl"
DATASET_NAME = "rtta-ticket-responder-golden"
EXPERIMENT = "responder-eval"
QUALITY_PASS = 3.0
JUDGE_PROMPT = (
    "On a scale of 1-5, rate this support draft reply.\n"
    "Ticket: {ticket}\nFindings: {findings}\nDraft: {draft}\n"
    "Action: {action}\n\n"
    'Return JSON: {{"score": <1-5>, "feedback": "<short>"}}'
)


def run_responder(inputs: dict) -> dict:
    state = empty_ticket_state(
        inputs["ticket"],
        inputs.get("domain", "support"),
        ticket_id=str(inputs.get("id") or inputs["ticket"].get("id", "eval")),
        classification=inputs.get("classification"),
        severity=inputs.get("severity"),
        findings=inputs.get("findings") or [],
    )
    out = responder_node(state)
    draft = out.get("draft") or {}
    return {
        "recommended_action": draft.get("recommended_action"),
        "subject": draft.get("subject"),
        "body": draft.get("body"),
        "confidence": draft.get("confidence"),
        "risk_flags": draft.get("risk_flags") or [],
    }


def action_match(run, example) -> dict:
    pred = run.outputs.get("recommended_action")
    expected = example.inputs.get("expected_action")
    return {"key": "action_exact", "score": 1.0 if pred == expected else 0.0}


def _parse_quality(text: str) -> tuple[float, str]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    blob = fence.group(1).strip() if fence else text.strip()
    try:
        data = json.loads(blob)
        return max(1.0, min(5.0, float(data.get("score", 0)))), str(data.get("feedback", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = re.search(r"[1-5](?:\.\d+)?", text)
        return (float(match.group()) if match else 2.0), text[:200]


def _fake_quality(body: str, action: str, expected: str) -> tuple[float, str]:
    if not (body or "").strip():
        return 1.0, "empty body"
    score = 3.5
    if action == expected:
        score += 0.5
    if len(body) > 40:
        score += 0.5
    return max(1.0, min(5.0, score)), "fake judge"


def quality_score(inputs: dict, outputs: dict) -> tuple[float, str]:
    if is_fake_chat_model(os.getenv("RTTA_MODEL", "")):
        return _fake_quality(
            outputs.get("body") or "",
            outputs.get("recommended_action") or "",
            inputs.get("expected_action") or "",
        )
    prompt = JUDGE_PROMPT.format(
        ticket=json.dumps(inputs.get("ticket") or {}, ensure_ascii=False),
        findings=json.dumps(inputs.get("findings") or [], ensure_ascii=False)[:2000],
        draft=json.dumps(
            {"subject": outputs.get("subject"), "body": outputs.get("body")},
            ensure_ascii=False,
        )[:3000],
        action=outputs.get("recommended_action"),
    )
    try:
        reply = get_chat_model().invoke([HumanMessage(content=prompt)])
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return _parse_quality(content)
    except Exception as exc:
        score, _ = _fake_quality(
            outputs.get("body") or "",
            outputs.get("recommended_action") or "",
            inputs.get("expected_action") or "",
        )
        return score, f"judge fallback: {exc}"


def quality_evaluator(run, example) -> dict:
    score, feedback = quality_score(example.inputs, run.outputs or {})
    return {"key": "quality", "score": score, "comment": feedback}


def _precision_recall(pairs: list[tuple[str, str]]) -> tuple[float, float, float]:
    """Binary escalate vs not; escalate is the positive class."""
    tp = fp = fn = tn = 0
    for expected, predicted in pairs:
        exp_pos = expected == "escalate"
        pred_pos = predicted == "escalate"
        if exp_pos and pred_pos:
            tp += 1
        elif not exp_pos and pred_pos:
            fp += 1
        elif exp_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def print_summary(results) -> float:
    passed = 0
    total = 0
    pairs: list[tuple[str, str]] = []

    for row in results:
        total += 1
        example = row["example"].inputs
        run = row.get("run")
        outputs = getattr(run, "outputs", None) if run is not None else {}
        if not isinstance(outputs, dict):
            outputs = {}

        action_ok = False
        quality = 0.0
        feedback = ""
        for result in row["evaluation_results"]["results"]:
            if result.key == "action_exact" and result.score is not None:
                action_ok = float(result.score) >= 1.0
            if result.key == "quality" and result.score is not None:
                quality = float(result.score)
                feedback = result.comment or ""

        expected = str(example.get("expected_action", "?"))
        predicted = str(outputs.get("recommended_action") or "?")
        pairs.append((expected, predicted))

        ok = action_ok and quality >= QUALITY_PASS
        if ok:
            passed += 1
        tid = example.get("id", "?")
        print(
            f"{'PASS' if ok else 'FAIL'}  {tid}  "
            f"action {predicted} vs {expected}  quality={quality:.1f}"
        )
        if feedback and not ok:
            print(f"       {feedback[:120]}")

    precision, recall, f1 = _precision_recall(pairs)
    print(f"\nEscalation precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")

    rate = passed / total if total else 0.0
    print(f"Pass-rate: {passed}/{total} ({100 * rate:.0f}%)")
    if getattr(results, "url", None):
        print(f"LangSmith: {results.url}")
    return rate


def main() -> None:
    rows = load_jsonl(GOLDEN_PATH)
    data = eval_data(rows, DATASET_NAME, "Responder golden with expected escalation")
    results = evaluate(
        run_responder,
        data=data,
        evaluators=[action_match, quality_evaluator],
        experiment_prefix=EXPERIMENT,
        description="Responder escalation + draft quality",
        upload_results=should_upload(),
    )
    rate = print_summary(results)
    if rate < 1.0 and os.getenv("EVAL_STRICT", "").lower() in ("1", "true", "yes"):
        sys.exit(1)


if __name__ == "__main__":
    main()
