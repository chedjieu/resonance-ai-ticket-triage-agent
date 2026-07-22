"""Triager eval — category/severity exact-match + confusion matrices."""

from __future__ import annotations

import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

from langsmith.evaluation import evaluate

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
importlib.import_module("app")  # load .env via app/__init__.py

from app.agents.triager import triager_node
from evals._common import EVALS_DIR, empty_ticket_state, eval_data, load_jsonl, should_upload

GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
DATASET_NAME = "rtta-ticket-triager-golden"
EXPERIMENT = "triager-eval"


def run_triager(inputs: dict) -> dict:
    state = empty_ticket_state(
        inputs["ticket"],
        inputs.get("domain", "support"),
        ticket_id=str(inputs.get("id") or inputs["ticket"].get("id", "eval")),
    )
    out = triager_node(state)
    classification = out.get("classification") or {}
    return {
        "category": classification.get("category"),
        "severity": out.get("severity"),
        "confidence": classification.get("confidence"),
        "rationale": classification.get("rationale"),
    }


def category_match(run, example) -> dict:
    pred = run.outputs.get("category")
    expected = example.inputs.get("expected_category")
    return {"key": "category_exact", "score": 1.0 if pred == expected else 0.0}


def severity_match(run, example) -> dict:
    pred = run.outputs.get("severity")
    expected = example.inputs.get("expected_severity")
    return {"key": "severity_exact", "score": 1.0 if pred == expected else 0.0}


def _print_confusion(title: str, pairs: list[tuple[str, str]]) -> None:
    labels = sorted({p for pair in pairs for p in pair})
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1

    print(f"\n{title} confusion matrix (rows=expected, cols=predicted)")
    header = "exp\\pred".ljust(18) + "".join(lab.ljust(16) for lab in labels)
    print(header)
    for exp in labels:
        row = exp.ljust(18)
        for pred in labels:
            row += str(matrix[exp][pred]).ljust(16)
        print(row)


def print_summary(results) -> float:
    passed = 0
    total = 0
    cat_pairs: list[tuple[str, str]] = []
    sev_pairs: list[tuple[str, str]] = []

    for row in results:
        total += 1
        example = row["example"].inputs
        run = row.get("run")
        outputs = getattr(run, "outputs", None) if run is not None else {}
        if not isinstance(outputs, dict):
            outputs = {}

        cat_ok = sev_ok = False
        for result in row["evaluation_results"]["results"]:
            if result.key == "category_exact" and result.score is not None:
                cat_ok = float(result.score) >= 1.0
            if result.key == "severity_exact" and result.score is not None:
                sev_ok = float(result.score) >= 1.0

        expected_c = str(example.get("expected_category", "?"))
        expected_s = str(example.get("expected_severity", "?"))
        pred_c = str(outputs.get("category") or "?")
        pred_s = str(outputs.get("severity") or "?")
        cat_pairs.append((expected_c, pred_c))
        sev_pairs.append((expected_s, pred_s))

        ok = cat_ok and sev_ok
        if ok:
            passed += 1
        tid = example.get("id") or example.get("ticket", {}).get("id", "?")
        print(
            f"{'PASS' if ok else 'FAIL'}  {tid}  "
            f"cat {pred_c} vs {expected_c}  sev {pred_s} vs {expected_s}"
        )

    _print_confusion("Category", cat_pairs)
    _print_confusion("Severity", sev_pairs)

    rate = passed / total if total else 0.0
    print(f"\nPass-rate: {passed}/{total} ({100 * rate:.0f}%)")
    if getattr(results, "url", None):
        print(f"LangSmith: {results.url}")
    return rate


def main() -> None:
    rows = load_jsonl(GOLDEN_PATH)
    data = eval_data(rows, DATASET_NAME, "Ticket triager golden labels")
    results = evaluate(
        run_triager,
        data=data,
        evaluators=[category_match, severity_match],
        experiment_prefix=EXPERIMENT,
        description="Triager category + severity exact-match",
        upload_results=should_upload(),
    )
    rate = print_summary(results)
    if rate < 1.0 and os.getenv("EVAL_STRICT", "").lower() in ("1", "true", "yes"):
        sys.exit(1)


if __name__ == "__main__":
    main()
