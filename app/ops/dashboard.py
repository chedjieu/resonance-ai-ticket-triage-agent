"""Ops dashboard — read-only Streamlit view of ticket triage metrics.

Run:
    uv run streamlit run app/ops/dashboard.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
HITL_LOG = DATA_DIR / "hitl_outcomes.jsonl"
SENT_LOG = DATA_DIR / "sent_responses.log"

STATUS_COLORS = {
    "resolved": "#2e7d32",
    "escalated": "#ef6c00",
    "rejected": "#c62828",
    "approved": "#2e7d32",
    "edited": "#1565c0",
    "sent": "#2e7d32",
    "unknown": "#616161",
}


def _parse_ts(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _derive_status(row: dict) -> str:
    if row.get("status"):
        return str(row["status"])
    if row.get("recommended_action") == "escalate":
        return "escalated"
    approval = str(row.get("approval") or row.get("action") or "").lower()
    if approval == "rejected":
        return "rejected"
    if approval in ("approved", "edited", "approve", "edit"):
        return "resolved"
    return "unknown"


def _normalize_hitl(rows: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows:
        ts = _parse_ts(row.get("timestamp"))
        classification = row.get("classification") or {}
        category = classification.get("category") if isinstance(classification, dict) else None
        latency = row.get("hitl_latency_seconds")
        if latency is None:
            queued = _parse_ts(row.get("queued_at"))
            if queued and ts:
                latency = max(0.0, (ts - queued).total_seconds())
        records.append(
            {
                "ticket_id": row.get("ticket_id") or "unknown",
                "timestamp": ts,
                "day": ts.date() if ts else None,
                "category": category or "unknown",
                "severity": row.get("severity") or "unknown",
                "approval": row.get("approval") or row.get("action") or "unknown",
                "recommended_action": row.get("recommended_action") or "send",
                "hitl_latency_seconds": float(latency) if latency is not None else None,
                "status": _derive_status(row),
                "domain": row.get("domain") or "support",
            }
        )
    return pd.DataFrame.from_records(records)


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'background:{color};color:white;font-size:0.85rem;font-weight:600;">'
        f"{status}</span>"
    )


def main() -> None:
    st.set_page_config(page_title="RTTA Ticket Ops", layout="wide")
    st.title("Resonance Ticket Triage — Ops Dashboard")
    st.caption("Read-only metrics from HITL outcomes (and sent log when present).")

    df = _normalize_hitl(_load_jsonl(HITL_LOG))
    if df.empty:
        st.warning(f"No HITL outcomes found at `{HITL_LOG}`. Approve some tickets first.")
        return

    today = date.today()
    available_days = sorted({d for d in df["day"].dropna().tolist()})
    default_day = today if today in available_days else (available_days[-1] if available_days else today)

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        selected_day = st.date_input("Day", value=default_day)
    with col_f2:
        if selected_day != today:
            st.info(f"Showing {selected_day} (today is {today}).")

    day_df = df[df["day"] == selected_day].copy()
    if day_df.empty:
        st.warning(f"No tickets for {selected_day}. Pick another day or run the approval UI.")
        day_df = df.copy()
        st.caption("Falling back to all logged outcomes for charts below.")

    # --- KPI row ---
    escalations = int((day_df["recommended_action"] == "escalate").sum())
    total = len(day_df)
    escalation_rate = (escalations / total) if total else 0.0
    latency_series = day_df["hitl_latency_seconds"].dropna()
    avg_latency = float(latency_series.mean()) if not latency_series.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tickets (selected day)", total)
    k2.metric("Escalation rate", f"{100 * escalation_rate:.0f}%")
    k3.metric("Avg HITL latency", f"{avg_latency:.0f}s")
    k4.metric("P1 count", int((day_df["severity"] == "P1").sum()))

    # --- Charts ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classifications by category")
        cat_counts = (
            day_df["category"].value_counts().rename_axis("category").reset_index(name="count")
        )
        st.bar_chart(cat_counts.set_index("category"))

    with c2:
        st.subheader("Severity distribution")
        sev_order = ["P1", "P2", "P3", "P4", "unknown"]
        sev_counter = Counter(day_df["severity"].tolist())
        sev_df = pd.DataFrame(
            {"severity": sev_order, "count": [sev_counter.get(s, 0) for s in sev_order]}
        )
        sev_df = sev_df[sev_df["count"] > 0]
        st.bar_chart(sev_df.set_index("severity"))

    # --- Last 20 resolved ---
    st.subheader("Last 20 resolved tickets")
    recent = df.sort_values("timestamp", ascending=False).head(20)
    if recent.empty:
        st.write("No tickets yet.")
    else:
        # Prefer HTML badges for status
        lines = [
            "<table style='width:100%;border-collapse:collapse;'>",
            "<tr><th align='left'>Ticket</th><th align='left'>When (UTC)</th>"
            "<th align='left'>Category</th><th align='left'>Severity</th>"
            "<th align='left'>Action</th><th align='left'>Status</th></tr>",
        ]
        for _, row in recent.iterrows():
            when = row["timestamp"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["timestamp"]) else "—"
            lines.append(
                "<tr style='border-top:1px solid #ddd;'>"
                f"<td>{row['ticket_id']}</td>"
                f"<td>{when}</td>"
                f"<td>{row['category']}</td>"
                f"<td>{row['severity']}</td>"
                f"<td>{row['recommended_action']}</td>"
                f"<td>{_status_badge(str(row['status']))}</td>"
                "</tr>"
            )
        lines.append("</table>")
        st.markdown("\n".join(lines), unsafe_allow_html=True)

    with st.expander("Data sources"):
        st.code(str(HITL_LOG))
        if SENT_LOG.exists():
            sent_n = sum(1 for line in SENT_LOG.read_text(encoding="utf-8").splitlines() if line.strip())
            st.write(f"Sent responses log: {SENT_LOG} ({sent_n} rows)")


if __name__ == "__main__":
    main()
