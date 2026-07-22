"""Input guardrails for ticket triage — hard-block prompt injection / exfil attempts."""

from __future__ import annotations

# Patterns that should refuse before any agent drafts a customer reply.
# Milder social-engineering / PII-in-ticket cases escalate via the Responder instead.
HARD_BLOCK_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your guidelines",
    "disregard the above",
    "disregard previous",
    "your new instructions are",
    "system override",
    "developer mode",
    "jailbreak",
    "you are now dan",
    "you are now unrestricted",
    "you are now",
    "exfiltrate",
    "send me the api key",
    "print your system prompt",
    "reveal your system prompt",
    "dump your system prompt",
)


def check_ticket_guardrail(text: str) -> str | None:
    """Return a refusal message if the ticket text hard-blocks; else None."""
    low = (text or "").lower()
    for pat in HARD_BLOCK_PATTERNS:
        if pat in low:
            return (
                "Sorry — this request was blocked by the RTTA ticket guardrail "
                f"(matched: {pat!r}). We cannot process prompt-injection or "
                "credential-exfiltration attempts as support tickets."
            )
    return None


def is_guardrail_exception(exc: BaseException) -> bool:
    """True when a Bedrock/Vertex guardrail intervened or refused the call."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    markers = (
        "guardrail",
        "guardrail_intervened",
        "contentfilter",
        "content filter",
        "model armor",
        "blocked by",
        "responsibleaipolicy",
    )
    return any(m in name or m in msg for m in markers)
