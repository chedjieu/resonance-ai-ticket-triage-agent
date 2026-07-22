"""Deterministic offline stand-in for any real chat model + embeddings.

When `RTTA_MODEL=fake` (and/or `RTTA_EMBEDDINGS=fake`), the project routes
through this module instead of Bedrock/Vertex. The point is to let the entire
graph - planner / researcher / writer, plus tools, guardrails, evals and the
HTMX UI - run end-to-end with zero cloud credentials, so we can dry-run the
lab offline. Real cloud models go through the normal `init_chat_model`
path in `app/llm.py`.

The fake model recognises a small number of "shapes" from the prompts in this
repo and emits matching output:

- structured planner output -> `PlannerOutput` with 3-7 tagged sub-questions
- structured triager output  -> `TriageOutput`        (Project 2)
- structured responder output -> `ResponderOutput`    (Project 2)
- a tool-using loop with web/local/fetch/summarize tools -> alternating tool
  calls then a JSON list of findings
- a hello-agent style tool loop with weather/news tools -> one call each then
  a one-line summary
- a writer-style call (no tools) carrying findings JSON -> a markdown report
  with `[n]` inline citations and a numbered Sources section

Everything is keyed off message content so two runs on the same input produce
the same output, which is exactly what the evals need.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from typing import Any, ClassVar
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

# ---------- helpers ------------------------------------------------------------


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "is", "are",
    "what", "which", "how", "why", "do", "does", "did", "can", "should", "with",
    "this", "that", "these", "those", "i", "you", "we", "they", "it", "by", "as",
    "be", "been", "from", "at", "into", "about", "between", "your", "our",
}


def _keywords(text: str, k: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", (text or "").lower())
    out: list[str] = []
    for w in words:
        if w in _STOPWORDS or len(w) <= 2:
            continue
        if w in out:
            continue
        out.append(w)
        if len(out) >= k:
            break
    return out


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _system_text(messages: list[BaseMessage]) -> str:
    for m in messages:
        if isinstance(m, SystemMessage):
            c = m.content
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _all_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _urls_in_tool_messages(messages: list[BaseMessage]) -> list[str]:
    urls: list[str] = []
    for m in _all_tool_messages(messages):
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        for u in re.findall(r"https?://[^\s\"'\)\]]+", content):
            if u not in urls:
                urls.append(u)
    return urls


def _input_to_messages(value: Any) -> list[BaseMessage]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [HumanMessage(content=value)]
    if hasattr(value, "to_messages"):
        return value.to_messages()
    return [HumanMessage(content=str(value))]


def _seed_text(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage | HumanMessage):
            parts.append(m.content if isinstance(m.content, str) else json.dumps(m.content))
    return "\n".join(parts)


# ---------- structured-output synthesiser -------------------------------------


_PLANNER_TOPIC_EXPANSIONS: list[tuple[tuple[str, ...], list[str]]] = [
    (("kubernetes", "1.30"), ["dynamic resource allocation", "apparmor", "kubernetes"]),
    (("sox",), ["internal control", "audit", "cloud", "sox"]),
    (("rotate", "iam"), ["rotate", "deactivate", "iam"]),
    (("zero-trust",), ["zero trust", "identity", "microsegmentation"]),
    (("zero trust",), ["zero trust", "identity", "microsegmentation"]),
    (("federal reserve",), ["rate", "technology", "market"]),
    (("bedrock", "vertex"), ["bedrock", "vertex", "rag"]),
    (("gdpr", "ccpa"), ["gdpr", "ccpa", "transfer"]),
    (("openai", "anthropic"), ["openai", "anthropic", "google"]),
    (("eu ai act",), ["eu ai act", "deployer", "obligation"]),
    (("ransomware",), ["ransomware", "attack", "response"]),
    (("climate", "wheat"), ["climate", "wheat", "price"]),
    (("who",), ["who", "screen time", "children"]),
    (("screen time",), ["who", "screen time", "children"]),
    (("sec", "climate"), ["sec", "climate", "disclosure"]),
    (("agpl",), ["agpl", "license", "saas"]),
    (("apple",), ["quarter", "revenue", "apple"]),
    (("data retention",), ["openai", "anthropic", "google"]),
]


def _planner_coverage_topics(question: str) -> list[str]:
    """Build topic labels so each sub-question names a distinct coverage area."""
    q_lower = (question or "").lower()
    topics: list[str] = []

    def add(topic: str) -> None:
        t = topic.strip()
        if t and t.lower() not in {x.lower() for x in topics}:
            topics.append(t)

    for needles, extras in _PLANNER_TOPIC_EXPANSIONS:
        if all(n in q_lower for n in needles):
            for extra in extras:
                add(extra)

    for token in re.findall(r"\b[A-Z]{2,}\b", question or ""):
        add(token.lower())

    for token in re.findall(r"\b\d+\.\d+\b", question or ""):
        add(token)

    for match in re.finditer(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+)?", question or ""
    ):
        add(match.group(0))

    for kw in _keywords(question, k=12):
        if len(topics) >= 7:
            break
        add(kw)

    if not topics:
        topics = ["topic"]
    return topics[:7]


def _make_sub_questions(question: str) -> list[dict[str, str]]:
    topics = _planner_coverage_topics(question)
    sources = ["web", "local", "both"]
    n = max(3, min(7, len(topics)))
    out: list[dict[str, str]] = []
    for i in range(n):
        topic = topics[i]
        out.append(
            {
                "text": f"What are the key facts about {topic} relevant to: {question}?",
                "source": sources[i % len(sources)],
            }
        )
    return out


def _structured_planner_output(schema: type, question: str) -> Any:
    sqs = _make_sub_questions(question)
    return schema(sub_questions=sqs)


def _classify_ticket(body: str, subject: str, categories: list[str]) -> tuple[str, str, float, str]:
    """Return (category, severity, confidence, rationale) heuristically.

    Used by the Project 2 Triager when run in fake mode.
    """
    text = f"{subject}\n{body}".lower()
    rules: list[tuple[str, list[str], str]] = [
        ("account_security", ["suspicious", "unauthorized", "compromised", "another country", "stolen"], "P1"),
        ("login_issue", ["mfa", "login", "log in", "password", "authenticator", "2fa", "locked"], "P2"),
        ("billing", ["refund", "invoice", "charge", "billed", "subscription", "plan", "credit"], "P2"),
        ("bug_report", ["bug", "broken", "crash", "error", "hangs", "exception", "stack trace"], "P3"),
        ("integration_help", ["integration", "webhook", "slack", "stripe", "api key", "callback"], "P3"),
        ("feature_request", ["feature", "wish", "would be nice", "please add", "suggest"], "P4"),
        ("data_export", ["export", "delete my account", "gdpr", "download my data"], "P3"),
    ]
    for cat, kws, sev in rules:
        if cat in categories and any(k in text for k in kws):
            rationale = f"Matched keyword(s) for category {cat}."
            return cat, sev, 0.82, rationale
    if "other" in categories:
        return "other", "P3", 0.55, "No high-confidence keyword match; defaulting to 'other'."
    return categories[0], "P3", 0.5, "Defaulting to first known category."


def _structured_triage_output(schema: type, messages: list[BaseMessage]) -> Any:
    text = _seed_text(messages)
    # Pull category list out of the system prompt if it lists "Available categories: [...]".
    m = re.search(r"Available categories:\s*([^\n]+)", text)
    cats = []
    if m:
        cats = [c.strip().strip("'\"") for c in re.findall(r"[A-Za-z_]+", m.group(1))]
    if not cats:
        cats = ["login_issue", "billing", "bug_report", "integration_help",
                "feature_request", "data_export", "account_security", "other"]
    # Try to find ticket subject/body in the human message; otherwise use all text.
    body = _last_human_text(messages)
    category, severity, conf, rationale = _classify_ticket(body, "", cats)
    fields = schema.model_fields if hasattr(schema, "model_fields") else {}
    payload: dict[str, Any] = {}
    if "category" in fields:
        payload["category"] = category
    if "severity" in fields:
        payload["severity"] = severity
    if "confidence" in fields:
        payload["confidence"] = conf
    if "rationale" in fields:
        payload["rationale"] = rationale
    return schema(**payload)


_PII_PATTERNS: tuple[tuple[str, str], ...] = (
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("phone", r"(?:(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"),
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
)

_ESCALATION_KEYWORDS = ("refund", "credit", "guarantee", "tomorrow", "by eod",
                       "by end of day", "compensation", "rebate")

_INJECTION_PATTERNS = (
    "ignore your guidelines", "ignore previous instructions",
    "ignore all instructions", "you are now", "disregard the above",
    "your new instructions are", "system override", "developer mode",
    "jailbreak", "reply rudely", "skip approval", "skip the approval",
    "bypass approval", "exfiltrate", "send me the api key",
    "tell me the ssn", "send the social security",
)


def _detect_risk_flags(body: str) -> list[str]:
    flags: list[str] = []
    low = body.lower()
    for kw in _ESCALATION_KEYWORDS:
        if kw in low:
            flags.append(f"keyword:{kw}")
            break
    for name, pat in _PII_PATTERNS:
        if re.search(pat, body):
            flags.append(f"pii:{name}")
    for pat in _INJECTION_PATTERNS:
        if pat in low:
            flags.append("prompt_injection")
            break
    if "send_response" in low and ("skip" in low or "without" in low or "bypass" in low):
        flags.append("tool_abuse_attempt")
    # Deduplicate but keep order.
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _structured_responder_output(schema: type, messages: list[BaseMessage]) -> Any:
    human = _last_human_text(messages)
    # The Responder is fed a JSON blob of {raw, classification, findings, ...}. Extract body if present.
    body_text = human
    sender = ""
    try:
        if human.strip().startswith("{"):
            data = json.loads(human)
            raw = data.get("raw") or data.get("ticket") or {}
            body_text = raw.get("body", "") or human
            sender = raw.get("sender", "")
    except Exception:
        pass

    flags = _detect_risk_flags(body_text)
    escalate = bool(flags)
    confidence = 0.5 if escalate else 0.85
    action = "escalate" if (escalate or confidence < 0.6) else "send"

    subject = "Re: your request"
    body = (
        "Hello,\n\n"
        "Thanks for reaching out. I have reviewed the details you sent and the supporting "
        "information collected by our team. Based on what I can see, here is what I recommend "
        "next.\n\n"
        "If this is time-sensitive please reply to this thread and our on-call engineer will "
        "follow up.\n\nBest regards,\nRTTA Support"
    )
    if "mfa" in body_text.lower() or "login" in body_text.lower():
        body = (
            "Hello,\n\nWe have looked at the authentication logs for your account and can see "
            "the MFA loop you reported. Please close and reopen your authenticator app to "
            "resync the time, then try logging in again from a fresh browser window. If the "
            "issue persists, we can reset your MFA device.\n\nThanks,\nRTTA Support"
        )
    if sender:
        body = body.replace("Hello,", f"Hello {sender.split('@')[0]},", 1)

    fields = schema.model_fields if hasattr(schema, "model_fields") else {}
    payload: dict[str, Any] = {}
    if "subject" in fields:
        payload["subject"] = subject
    if "body" in fields:
        payload["body"] = body
    if "recommended_action" in fields:
        payload["recommended_action"] = action
    if "confidence" in fields:
        payload["confidence"] = confidence
    if "risk_flags" in fields:
        payload["risk_flags"] = flags
    return schema(**payload)


def _structured_generic(schema: type, messages: list[BaseMessage]) -> Any:
    """Last-resort fallback: fill required fields with placeholders."""
    if not hasattr(schema, "model_fields"):
        return schema()
    payload: dict[str, Any] = {}
    for name, info in schema.model_fields.items():
        ann = info.annotation
        if ann is str:
            payload[name] = "n/a"
        elif ann is int:
            payload[name] = 0
        elif ann is float:
            payload[name] = 0.0
        elif ann is bool:
            payload[name] = False
        elif getattr(ann, "__origin__", None) is list:
            payload[name] = []
        else:
            payload[name] = None
    return schema(**payload)


def _structured_for(schema: type, raw_input: Any) -> Any:
    messages = _input_to_messages(raw_input)
    name = getattr(schema, "__name__", "")
    if name == "PlannerOutput":
        return _structured_planner_output(schema, _last_human_text(messages))
    if name == "TriageOutput":
        return _structured_triage_output(schema, messages)
    if name == "ResponderOutput":
        return _structured_responder_output(schema, messages)
    if name == "Findings" or name.endswith("FindingsList") or name == "FindingList":
        return _structured_generic(schema, messages)
    return _structured_generic(schema, messages)


# ---------- tool-call routing -------------------------------------------------


def _pick_researcher_query(messages: list[BaseMessage]) -> tuple[str, str]:
    """Decide which tool to call next for the P1 Researcher.

    Returns (tool_name, query). Looks at the human message for a 'web|local|both'
    tag and at the system message for cues.
    """
    human = _last_human_text(messages)
    system = _system_text(messages).lower()
    low = human.lower()
    if "search_local_docs" in system and ("local" in low or "in our internal docs" in system):
        return "search_local_docs", human[:200]
    if "search_local" in [t for t in []]:
        pass
    if " local " in f" {low} " or "documentation" in low or "internal" in low:
        return "search_local_docs", human[:200]
    return "web_search", human[:200]


def _findings_from_tool_history(messages: list[BaseMessage], sub_q_text: str) -> str:
    """Build the JSON the researcher node parses out as findings."""
    findings: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for m in _all_tool_messages(messages):
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        # Try to parse JSON tool output (web_search / search_local_docs return list[dict]).
        results: list[dict[str, Any]] = []
        try:
            data = json.loads(content)
            if isinstance(data, list):
                results = [r for r in data if isinstance(r, dict)]
        except Exception:
            for u in re.findall(r"https?://[^\s\"'\)\]]+", content):
                results.append({"url": u, "content": content[:200]})
        for r in results:
            url = r.get("source_url") or r.get("url")
            if not url or url in seen_urls:
                continue
            text = (r.get("content") or r.get("text") or r.get("snippet") or "")[:240]
            findings.append({
                "claim": f"Per the source, {text or 'relevant context for ' + sub_q_text[:80]}",
                "evidence_url": url,
                "evidence_text": text,
            })
            seen_urls.add(url)
            if len(findings) >= 2:
                break
        if len(findings) >= 2:
            break
    if not findings:
        findings = [{
            "claim": f"Background information addressing: {sub_q_text[:120]}",
            "evidence_url": "https://example.com/fake-mode-note",
            "evidence_text": "fake-mode fallback finding (no tool URL surfaced)",
        }]
    return json.dumps(findings, ensure_ascii=False)


def _pick_investigator_tool(messages: list[BaseMessage], used: list[str]) -> tuple[str, dict[str, Any]] | None:
    """Decide the next investigator tool call (P2). None means 'stop'."""
    system = _system_text(messages).lower()
    human = _last_human_text(messages)
    low = (human + "\n" + system).lower()
    # Try, in order: query_logs, search_runbooks, query_metrics, get_ticket_history.
    candidates: list[tuple[str, dict[str, Any]]] = []
    service = "auth-service" if "login" in low or "mfa" in low else (
        "billing-service" if "billing" in low or "refund" in low else (
            "reports-service" if "report" in low or "export" in low else "app"))
    candidates.append(("query_logs", {"service": service, "since": "1h"}))
    candidates.append(("search_runbooks", {"query": " ".join(_keywords(human, k=4)) or "general"}))
    candidates.append(("query_metrics", {"service": service, "metric": "errors_per_min", "since": "1h"}))
    user_email = ""
    em = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", human)
    if em:
        user_email = em.group(0)
    candidates.append(("get_ticket_history", {"user_id": user_email or "unknown@example.com", "k": 3}))
    for name, args in candidates:
        if name not in used:
            return name, args
    return None


def _build_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid4().hex[:12]}"}


def _route_with_tools(messages: list[BaseMessage], bound_tools: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Decide tool_calls or final text for a tool-bound chat invocation."""
    tool_names = {getattr(t, "name", None) or getattr(t, "__name__", "") for t in bound_tools}
    used = [m.name for m in _all_tool_messages(messages) if m.name]
    system = _system_text(messages).lower()
    human = _last_human_text(messages)

    # 1) Researcher pattern (P1): four-tool family bound, asks for findings.
    p1_tools = {"web_search", "fetch_url", "search_local_docs", "summarize"}
    if p1_tools.issubset(tool_names) or ("focused researcher" in system) or ("supporting facts" in system):
        if len(used) >= 1:
            return _findings_from_tool_history(messages, human), []
        name, query = _pick_researcher_query(messages)
        if name not in tool_names:
            name = next(iter(tool_names & p1_tools)) if (tool_names & p1_tools) else "web_search"
        return "", [_build_tool_call(name, {"query": query})]

    # 2) Investigator pattern (P2).
    p2_tools = {"query_logs", "query_metrics", "search_runbooks", "get_ticket_history"}
    if p2_tools & tool_names:
        if len(used) >= min(3, len(p2_tools & tool_names)):
            summary = (
                "Based on the logs, runbooks, and ticket history, the most likely root cause is "
                "identified and remediation steps are documented in the runbook. Recommending the "
                "standard remediation playbook."
            )
            return summary, []
        nxt = _pick_investigator_tool(messages, used)
        if nxt is None:
            return "All available signals examined; ready to summarise findings.", []
        name, args = nxt
        if name not in tool_names:
            remaining = [n for n in tool_names if n not in used]
            if remaining:
                name = remaining[0]
        return "", [_build_tool_call(name, args)]

    # 3) Hello-agent style: weather + news + maybe time.
    if {"get_weather", "search_news"} & tool_names:
        if "get_weather" in tool_names and "get_weather" not in used:
            city = "Bangalore"
            m = re.search(r"in ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)", human)
            if m:
                city = m.group(1)
            return "", [_build_tool_call("get_weather", {"city": city})]
        if "search_news" in tool_names and "search_news" not in used:
            topic_kws = _keywords(human, k=2) or ["technology"]
            return "", [_build_tool_call("search_news", {"topic": " ".join(topic_kws)})]
        if "get_time" in tool_names and "get_time" not in used and "time" in human.lower():
            return "", [_build_tool_call("get_time", {"timezone": "Asia/Kolkata"})]
        # Wrap up.
        weather = ""
        news = ""
        for m in _all_tool_messages(messages):
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            if m.name == "get_weather":
                weather = content
            elif m.name == "search_news":
                news = content
        return f"Here is what I found:\n- Weather: {weather}\n- News: {news}", []

    # 4) Generic tool-bound: just answer plainly.
    return _generic_text(messages), []


# ---------- text generators ---------------------------------------------------


def _writer_markdown(human: str) -> str:
    """The writer node sends a JSON blob; produce a markdown report from it."""
    try:
        data = json.loads(human)
    except Exception:
        return f"# Report\n\n{human[:400]}\n"
    question = data.get("question", "Research question")
    sub_qs = data.get("sub_questions") or []
    findings = data.get("findings") or []

    urls: list[str] = []
    for f in findings:
        u = f.get("evidence_url")
        if u and u not in urls:
            urls.append(u)
    url_to_n = {u: i + 1 for i, u in enumerate(urls)}

    lines: list[str] = []
    lines.append(f"# {question}")
    lines.append("")
    lines.append("## Executive summary")
    if findings:
        lines.append(
            f"This report compiles {len(findings)} findings across {len(sub_qs) or 1} sub-questions "
            f"to answer: {question}. Each claim is grounded in a cited source."
        )
    else:
        lines.append(
            f"This report covers {question}. No external findings were available; the analysis "
            "below relies on general background and should be treated as preliminary."
        )
    lines.append("")
    for idx, sq in enumerate(sub_qs or [{"text": question}]):
        title = sq.get("text", f"Sub-question {idx + 1}") if isinstance(sq, dict) else str(sq)
        lines.append(f"## {title}")
        local_findings = [f for f in findings if f.get("sub_question_index", idx) == idx]
        if not local_findings:
            local_findings = findings[:1]
        for f in local_findings:
            claim = f.get("claim", "")
            url = f.get("evidence_url")
            n = url_to_n.get(url, len(url_to_n) + 1) if url else None
            cite = f" [{n}]" if n else ""
            lines.append(f"- {claim}{cite}")
        lines.append("")
    lines.append("## Sources")
    for u, n in url_to_n.items():
        lines.append(f"{n}. {u}")
    lines.append("")
    return "\n".join(lines)


def _investigator_findings_json(messages: list[BaseMessage]) -> list[dict]:
    """Synthesize P2 investigator findings from tool message history."""
    findings: list[dict] = []
    for msg in _all_tool_messages(messages):
        name = msg.name or "unknown"
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
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
        elif name == "query_metrics" and isinstance(data, dict) and data.get("current") is not None:
            findings.append(
                {
                    "claim": (
                        f"{data.get('metric', 'metric')} current={data.get('current')}, "
                        f"trend={data.get('trend', 'unknown')}"
                    ),
                    "source": str(data.get("service", "metrics")),
                    "tool": "query_metrics",
                }
            )
        elif name == "search_runbooks" and isinstance(data, list):
            for hit in data[:2]:
                if isinstance(hit, dict):
                    findings.append(
                        {
                            "claim": str(hit.get("text", ""))[:200] or "Runbook guidance found",
                            "source": str(hit.get("source_url", "runbook")),
                            "tool": "search_runbooks",
                        }
                    )
        elif name == "get_ticket_history" and isinstance(data, list):
            for ticket in data[:2]:
                if isinstance(ticket, dict):
                    findings.append(
                        {
                            "claim": str(ticket.get("resolution") or ticket.get("subject", "")),
                            "source": str(ticket.get("id", "history")),
                            "tool": "get_ticket_history",
                        }
                    )
    return findings


def _generic_text(messages: list[BaseMessage]) -> str:
    system = _system_text(messages).lower()
    human = _last_human_text(messages)

    # writer node: no tools, asked to produce a markdown report.
    if "research report" in system or "markdown report" in system or "executive summary" in system:
        return _writer_markdown(human)

    # P2 investigator summarise step: JSON list of {claim, source, tool}.
    if '"claim"' in system and '"tool"' in system:
        return json.dumps(_investigator_findings_json(messages), ensure_ascii=False)

    # summarize tool's LLM call: produce a 3-sentence summary.
    if "summary" in system or "summarise" in system or "summarize" in system:
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", human)
        first = " ".join(words[:25]) or "The input was empty"
        return (
            f"{first}. The key takeaways are pragmatic and align with standard best practice. "
            "Apply the recommended steps and re-evaluate."
        )

    # P1 memory extract_node.
    if "worth_remembering" in system or "preference or stable fact" in system:
        return json.dumps({"worth_remembering": False, "content": ""})

    # LangSmith-as-judge for evals returning a number.
    if "return only a number" in system or "return a number 0.0-1.0" in system:
        return "0.85"

    # Default plausible reply.
    if human:
        return f"Acknowledged: {human[:200]}"
    return "OK."


# ---------- the model + the embeddings -----------------------------------------


class FakeRTTAChatModel(BaseChatModel):
    """A deterministic chat model that pretends to be a real LLM.

    Stays inside the langchain `BaseChatModel` contract so `init_chat_model`
    callers can swap it in without code changes elsewhere.
    """

    model_name: str = "fake"
    bound_tools: ClassVar[list[Any] | None] = None  # placeholder; actual lives on instance via attr
    structured_schema: type | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_bound_tools", [])

    @property
    def _llm_type(self) -> str:
        return "fake-rtta"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeRTTAChatModel:
        new = FakeRTTAChatModel(model_name=self.model_name)
        object.__setattr__(new, "_bound_tools", list(tools))
        return new

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
        method: str | None = None,
        **kwargs: Any,
    ) -> Any:
        def _run(value: Any) -> Any:
            obj = _structured_for(schema, value)
            if include_raw:
                ai = AIMessage(content=obj.model_dump_json() if isinstance(obj, BaseModel) else json.dumps(obj))
                return {"raw": ai, "parsed": obj, "parsing_error": None}
            return obj
        return RunnableLambda(_run).with_config({"run_name": f"fake_structured_{getattr(schema, '__name__', 'output')}"})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        bound = getattr(self, "_bound_tools", []) or []
        if bound:
            text, tool_calls = _route_with_tools(messages, bound)
        else:
            text, tool_calls = _generic_text(messages), []
        ai = AIMessage(content=text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


class FakeRTTAEmbeddings(Embeddings):
    """Deterministic hashed embeddings.

    Bedrock Titan v2 is 1024-dim; we match that so the existing pgvector schema
    (`vector(1024)`) works without a migration.
    """

    DIM: ClassVar[int] = 1024

    def _embed_one(self, text: str) -> list[float]:
        text = text or ""
        n_floats_per_block = 16
        n_blocks = math.ceil(self.DIM / n_floats_per_block)
        out: list[float] = []
        for i in range(n_blocks):
            digest = hashlib.sha256(f"{i}|{text}".encode()).digest()[:64]
            for j in range(n_floats_per_block):
                start = (j * 4) % (len(digest) - 4)
                (val,) = struct.unpack(">i", digest[start:start + 4])
                out.append(val / 2_147_483_647.0)
        out = out[: self.DIM]
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


# ---------- factory funcs used by app/llm.py ----------------------------------


def is_fake_chat_model(name: str | None) -> bool:
    if not name:
        return True
    return name.strip().lower() in {"fake", "fake-rtta", "stub", "rtta:fake"}


def is_fake_embeddings(name: str | None) -> bool:
    if not name:
        return False
    return name.strip().lower() in {"fake", "fake-rtta", "stub", "rtta:fake"}


def fake_chat_model(**_kwargs: Any) -> FakeRTTAChatModel:
    return FakeRTTAChatModel()


def fake_embeddings(**_kwargs: Any) -> FakeRTTAEmbeddings:
    return FakeRTTAEmbeddings()


def _force_fake_via_env() -> bool:
    return os.getenv("RTTA_FORCE_FAKE", "").strip().lower() in {"1", "true", "yes"}
