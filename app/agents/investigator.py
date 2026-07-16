"""Investigator agent — gathers context via tools before responding."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.llm import get_chat_model, invoke_with_throttle_fallback
from app.state import TicketState
from app.tools._domain import set_domain
from app.tools.get_ticket_history import get_ticket_history
from app.tools.query_logs import query_logs
from app.tools.query_metrics import query_metrics
from app.tools.search_runbooks import search_runbooks

INVESTIGATOR_SYSTEM = (
    "You are an investigator. Given a classified ticket, gather enough context to write "
    "an informed response. Use tools to fetch logs, metrics, runbooks, and the user's "
    "ticket history. Stop calling tools when you can clearly explain what happened and "
    "what should be done. Budget: 8 tool calls max."
)

SUMMARIZE_SYSTEM = (
    "Summarise the investigation as a JSON list of objects with keys "
    '"claim", "source", and "tool". Each claim should cite which tool/source supported it. '
    "Return only the JSON list."
)

TOOLS = [query_logs, query_metrics, search_runbooks, get_ticket_history]
TOOL_BY_NAME = {t.name: t for t in TOOLS}
MAX_TOOL_CALLS = 8
ARGS_PREVIEW_LEN = 80


def _tool_result_str(result: object) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _truncate_args(args: dict) -> str:
    text = json.dumps(args, ensure_ascii=False)
    if len(text) <= ARGS_PREVIEW_LEN:
        return text
    return text[: ARGS_PREVIEW_LEN - 3] + "..."


def _format_tool_log(name: str, args: dict) -> str:
    return f"Investigator: {name}({_truncate_args(args)})"


def _parse_findings_json(content: str) -> list[dict]:
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if isinstance(data, dict) and "findings" in data:
        data = data["findings"]
    if not isinstance(data, list):
        return []
    findings: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        findings.append(
            {
                "claim": claim,
                "source": str(item.get("source", "")),
                "tool": str(item.get("tool", "")),
            }
        )
    return findings


def _ticket_payload(state: TicketState) -> str:
    return json.dumps(
        {
            "ticket_id": state["ticket_id"],
            "domain": state["domain"],
            "raw": state["raw"],
            "classification": state["classification"],
            "severity": state["severity"],
        },
        ensure_ascii=False,
    )


def _findings_from_tool_messages(messages: list) -> list[dict]:
    """Build investigator findings from tool results when LLM JSON parse fails."""
    findings: list[dict] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = msg.name or "unknown"
        content = _tool_result_str(msg.content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = content

        if name == "query_logs" and isinstance(data, list):
            for entry in data[:3]:
                if isinstance(entry, dict) and entry.get("message"):
                    findings.append(
                        {
                            "claim": str(entry["message"]),
                            "source": str(entry.get("timestamp", "logs")),
                            "tool": "query_logs",
                        }
                    )
        elif name == "query_metrics" and isinstance(data, dict):
            metric = data.get("metric", "metric")
            current = data.get("current")
            if current is not None:
                findings.append(
                    {
                        "claim": f"{metric} current={current}, trend={data.get('trend', 'unknown')}",
                        "source": str(data.get("service", "metrics")),
                        "tool": "query_metrics",
                    }
                )
        elif name == "search_runbooks" and isinstance(data, list):
            for hit in data[:2]:
                if isinstance(hit, dict):
                    text = str(hit.get("text", ""))[:200]
                    findings.append(
                        {
                            "claim": text or "Runbook guidance found",
                            "source": str(hit.get("source_url", "runbook")),
                            "tool": "search_runbooks",
                        }
                    )
        elif name == "get_ticket_history" and isinstance(data, list):
            for ticket in data[:2]:
                if isinstance(ticket, dict):
                    resolution = ticket.get("resolution") or ticket.get("subject", "")
                    findings.append(
                        {
                            "claim": str(resolution),
                            "source": str(ticket.get("id", "history")),
                            "tool": "get_ticket_history",
                        }
                    )
    return findings


def investigator_node(state: TicketState) -> dict:
    set_domain(state["domain"])
    step_log = list(state["step_log"])
    messages: list = [
        SystemMessage(content=INVESTIGATOR_SYSTEM),
        HumanMessage(content=_ticket_payload(state)),
    ]
    tool_calls_used = 0

    while tool_calls_used < MAX_TOOL_CALLS:
        def invoke_with_tools():
            llm = get_chat_model().bind_tools(TOOLS)
            return llm.invoke(messages)

        ai_msg = invoke_with_throttle_fallback(invoke_with_tools)
        if not ai_msg.tool_calls:
            break

        messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            if tool_calls_used >= MAX_TOOL_CALLS:
                break
            name = tc["name"]
            args = tc["args"]
            result = TOOL_BY_NAME[name].invoke(args)
            messages.append(
                ToolMessage(
                    content=_tool_result_str(result),
                    tool_call_id=tc["id"],
                    name=name,
                )
            )
            tool_calls_used += 1
            step_log.append(_format_tool_log(name, args))

    if tool_calls_used >= MAX_TOOL_CALLS:
        step_log.append(f"Investigator: max tool calls ({MAX_TOOL_CALLS}) reached")

    def summarize():
        llm = get_chat_model()
        return llm.invoke(
            [
                SystemMessage(content=SUMMARIZE_SYSTEM),
                *messages,
            ]
        )

    summary_msg = invoke_with_throttle_fallback(summarize)
    content = summary_msg.content if isinstance(summary_msg.content, str) else str(summary_msg.content)

    findings: list[dict] = []
    try:
        findings = _parse_findings_json(content)
    except (json.JSONDecodeError, TypeError):
        findings = _findings_from_tool_messages(messages)
        if findings:
            step_log.append("Investigator: synthesized findings from tool results")
        else:
            step_log.append("Investigator: failed to parse findings JSON")

    step_log.append(f"Investigator: {len(findings)} findings gathered")
    return {"findings": findings, "step_log": step_log}
