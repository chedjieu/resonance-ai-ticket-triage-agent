"""Provider-agnostic chat-model and embeddings factory.

Every Resonance Technologies project imports `get_chat_model()` and `get_embeddings()`
from here. Swap Bedrock <-> Vertex by changing `RTTA_MODEL` / `RTTA_EMBEDDINGS`
in `.env`. Set either to `fake` to use the offline `FakeRTTAChatModel` /
`FakeRTTAEmbeddings` for local dry-runs without cloud credentials.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import HumanMessage

from app._fake_llm import (
    FakeRTTAChatModel,
    FakeRTTAEmbeddings,
    fake_chat_model,
    fake_embeddings,
    is_fake_chat_model,
    is_fake_embeddings,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"
DEFAULT_EMBEDDINGS = "bedrock:amazon.titan-embed-text-v2:0"

T = TypeVar("T")
_fallback_logged = False


def is_throttling_error(exc: BaseException) -> bool:
    """True when Bedrock/Vertex rejected the call due to rate or quota limits."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "throttl" in name or "throttl" in msg or "too many tokens" in msg


def _fake_fallback_enabled() -> bool:
    """Auto-fallback to fake is opt-in only (RTTA_ALLOW_FAKE_FALLBACK=1)."""
    return os.getenv("RTTA_ALLOW_FAKE_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def activate_fake_fallback() -> None:
    """Switch this process to offline fake models and clear cached cloud clients."""
    global _fallback_logged
    os.environ["RTTA_MODEL"] = "fake"
    os.environ["RTTA_EMBEDDINGS"] = "fake"
    get_chat_model.cache_clear()
    get_embeddings.cache_clear()
    if not _fallback_logged:
        logger.warning(
            "Cloud model throttled (daily quota) — switched to RTTA_MODEL=fake for this process. "
            "Wait for Bedrock quota reset, or use Vertex (gcloud auth application-default login)."
        )
        print(
            "\n!! Bedrock daily quota hit — continuing with RTTA_MODEL=fake for this run.\n"
            "   Fix later: wait for quota reset, or set RTTA_MODEL=google_vertexai:gemini-2.5-pro\n"
            "   after: gcloud auth application-default login\n"
        )
        _fallback_logged = True


def invoke_with_throttle_fallback(fn: Callable[[], T]) -> T:
    """Run fn(); on Bedrock/Vertex throttle, fall back to fake and retry once.

    Always falls back on daily-quota throttling so labs stay runnable.
    Set RTTA_ALLOW_FAKE_FALLBACK=0 to raise instead.
    """
    try:
        return fn()
    except Exception as exc:
        deny = os.getenv("RTTA_ALLOW_FAKE_FALLBACK", "1").strip().lower() in (
            "0",
            "false",
            "no",
        )
        if is_throttling_error(exc) and not deny:
            activate_fake_fallback()
            return fn()
        raise


def ensure_chat_model_available() -> bool:
    """Probe the configured chat model. Returns True only when RTTA_MODEL=fake."""
    if is_fake_chat_model(_resolved_chat_name(None)):
        return True
    get_chat_model().invoke([HumanMessage(content="Reply with exactly: ok")])
    return False


def _resolved_chat_name(name: str | None) -> str:
    return (name or os.getenv("RTTA_MODEL") or DEFAULT_MODEL).strip()


def _resolved_embedding_name(name: str | None) -> str:
    return (name or os.getenv("RTTA_EMBEDDINGS") or DEFAULT_EMBEDDINGS).strip()


@lru_cache(maxsize=4)
def get_chat_model(name: str | None = None, **kwargs: Any):
    """Return a LangChain chat model. Reads `RTTA_MODEL` when name is None.

    If `RTTA_MODEL=fake` (the offline-demo default) we return a deterministic
    `FakeRTTAChatModel`; otherwise we hand off to `init_chat_model`.
    """
    resolved = _resolved_chat_name(name)
    if is_fake_chat_model(resolved):
        return fake_chat_model(**kwargs)

    # Vertex Model Armor (VERTEX_MODEL_ARMOR_POLICY) is configured on the GCP side — out of scope here.
    model_kwargs = dict(kwargs)
    guardrail_id = os.getenv("BEDROCK_GUARDRAIL_ID", "").strip()
    if guardrail_id and resolved.startswith("bedrock"):
        model_kwargs["guardrails"] = {
            "guardrailIdentifier": guardrail_id,
            "guardrailVersion": os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT",
            "trace": "enabled",
        }

    if "google_vertexai" in resolved or resolved.startswith("vertexai"):
        project = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GCP_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
        if project and "project" not in model_kwargs:
            model_kwargs["project"] = project
        if location and "location" not in model_kwargs:
            model_kwargs["location"] = location

    return init_chat_model(resolved, **model_kwargs)


@lru_cache(maxsize=4)
def get_embeddings(name: str | None = None, **kwargs: Any):
    """Return a LangChain embeddings model. Reads `RTTA_EMBEDDINGS` when name is None.

    If `RTTA_EMBEDDINGS=fake` we return deterministic hashed 1024-dim vectors so
    pgvector ingest + search work without a real embeddings API. Otherwise we
    defer to `init_embeddings`.
    """
    resolved = _resolved_embedding_name(name)
    if is_fake_embeddings(resolved):
        return fake_embeddings(**kwargs)
    return init_embeddings(resolved, **kwargs)


__all__ = [
    "DEFAULT_EMBEDDINGS",
    "DEFAULT_MODEL",
    "FakeRTTAChatModel",
    "FakeRTTAEmbeddings",
    "activate_fake_fallback",
    "ensure_chat_model_available",
    "get_chat_model",
    "get_embeddings",
    "invoke_with_throttle_fallback",
    "is_throttling_error",
]
