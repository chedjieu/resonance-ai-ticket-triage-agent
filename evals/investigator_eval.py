"""Investigator eval — keyword grounding + LLM-as-judge sufficiency."""

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
from app.agents.investigator import investigator_node
from app.llm import get_chat_model
from evals._common import EVALS_DIR, empty_ticket_state, eval_data, load_jsonl, should_upload

GOLDEN_PATH = EVALS_DIR / "investigator_golden.jsonl"
DATASET_NAME = "rtta-ticket-investigator-golden"
EXPERIMENT = "investigator-eval"
PASS_THRESHOLD = 0.5
JUDGE_PROMPT = (
    "You are judging ticket investigation findings.\n"
    "Ticket: {ticket}\nClassification: {classification}\nFindings: {findings}\n\n"
    "Are these findings sufficient and grounded in plausible evidence for drafting a reply?\n"
    'Return JSON: {{"score": <0.0-1.0>, "feedback": "<short>"}}'
)


def run_investigator(inputs: dict) -> dict:
    state = empty_ticket_state(
        inputs["ticket"],
        inputs.get("domain", "support"),
        ticket_id=str(inputs.get("id") or inputs["ticket"].get("id", "eval")),
        classification=inputs.get("classification"),
        severity=inputs.get("severity"),
    )
    out = investigator_node(state)
    findings = out.get("findings") or []
    return {"findings": findings}


def _keyword_score(findings: list[dict], keywords: list[str]) -> float:
    """Each finding claim must contain at least one expected keyword."""
    if not findings:
        return 0.0
    if not keywords:
        return 1.0
    kws = [k.lower() for k in keywords]
    hits = 0
    for finding in findings:
        claim = str(finding.get("claim", "")).lower()
        if any(k in claim for k in kws):
            hits += 1
    return hits / len(findings)


def keyword_evaluator(run, example) -> dict:
    findings = run.outputs.get("findings") or []
    keywords = example.inputs.get("expected_finding_keywords") or []
    return {"key": "keyword_grounding", "score": _keyword_score(findings, keywords)}


def _parse_judge(text: str) -> tuple[float, str]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    blob = fence.group(1).strip() if fence else text.strip()
    try:
        data = json.loads(blob)
        return max(0.0, min(1.0, float(data.get("score", 0)))), str(data.get("feedback", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = re.search(r"0?\.\d+|1\.0|1|0", text)
        return (float(match.group()) if match else 0.0), text[:200]


def _fake_judge(findings: list[dict], keywords: list[str]) -> tuple[float, str]:
    score = _keyword_score(findings, keywords)
    return score, f"fake judge: keyword grounding {score:.2f}"


def llm_judge_score(ticket: dict, classification: dict, findings: list[dict], keywords: list[str]) -> tuple[float, str]:
    if is_fake_chat_model(os.getenv("RTTA_MODEL", "")):
        return _fake_judge(findings, keywords)
    prompt = JUDGE_PROMPT.format(
        ticket=json.dumps(ticket, ensure_ascii=False),
        classification=json.dumps(classification or {}, ensure_ascii=False),
        findings=json.dumps(findings, ensure_ascii=False)[:4000],
    )
    try:
        reply = get_chat_model().invoke([HumanMessage(content=prompt)])
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return _parse_judge(content)
    except Exception as exc:
        return _fake_judge(findings, keywords)[0], f"judge fallback: {exc}"


def judge_evaluator(run, example) -> dict:
    findings = run.outputs.get("findings") or []
    score, feedback = llm_judge_score(
        example.inputs.get("ticket") or {},
        example.inputs.get("classification") or {},
        findings,
        example.inputs.get("expected_finding_keywords") or [],
    )
    return {"key": "llm_judge", "score": score, "comment": feedback}


def print_summary(results) -> float:
    passed = 0
    total = 0
    for row in results:
        total += 1
        kw = judge = 0.0
        feedback = ""
        for result in row["evaluation_results"]["results"]:
            if result.key == "keyword_grounding" and result.score is not None:
                kw = float(result.score)
            if result.key == "llm_judge" and result.score is not None:
                judge = float(result.score)
                feedback = result.comment or ""
        combined = 0.5 * kw + 0.5 * judge
        ok = combined >= PASS_THRESHOLD
        if ok:
            passed += 1
        tid = row["example"].inputs.get("id", "?")
        print(
            f"{'PASS' if ok else 'FAIL'}  {tid}  "
            f"keyword={kw:.2f} judge={judge:.2f} combined={combined:.2f}"
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
    data = eval_data(rows, DATASET_NAME, "Investigator golden with expected finding keywords")
    results = evaluate(
        run_investigator,
        data=data,
        evaluators=[keyword_evaluator, judge_evaluator],
        experiment_prefix=EXPERIMENT,
        description="Investigator keyword grounding + LLM judge",
        upload_results=should_upload(),
    )
    rate = print_summary(results)
    if rate < 1.0 and os.getenv("EVAL_STRICT", "").lower() in ("1", "true", "yes"):
        sys.exit(1)


if __name__ == "__main__":
    main()
