"""FastAPI entrypoint for the ticket triage approval UI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from app.graph import SAMPLE_TICKET, build_graph, make_initial_state

UI_DIR = Path(__file__).resolve().parent / "ui"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_threads: dict[str, dict[str, str | None]] = {}
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm up the graph on startup so the first /pending poll is fast."""
    _get_graph()
    yield


class IngestRequest(BaseModel):
    ticket: dict
    domain: str


class ApproveRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    edited_body: str | None = None


def _has_pending_interrupt(graph, config: dict) -> bool:
    snap = graph.get_state(config)
    return any(intr for task in snap.tasks for intr in task.interrupts)


def _run_until_pause(thread_id: str, ticket: dict, domain: str) -> None:
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    ticket_id = str(ticket.get("id") or thread_id)
    state = make_initial_state(ticket_id, ticket, domain)  # type: ignore[arg-type]
    try:
        for _ in graph.stream(state, config, stream_mode="updates"):
            pass
        if _has_pending_interrupt(graph, config):
            _threads[thread_id]["status"] = "pending_hitl"
        else:
            _threads[thread_id]["status"] = "completed"
    except Exception as exc:
        _threads[thread_id]["status"] = "failed"
        _threads[thread_id]["error"] = str(exc)
        logger.exception("Ticket processing failed for %s", thread_id)


def _resume(thread_id: str, payload: dict) -> None:
    graph = _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph.invoke(Command(resume=payload), config)
        _threads[thread_id]["status"] = "completed"
        _threads[thread_id]["error"] = None
    except Exception as exc:
        _threads[thread_id]["status"] = "failed"
        _threads[thread_id]["error"] = str(exc)
        logger.exception("Resume failed for %s", thread_id)


app = FastAPI(title="Resonance Ticket Triage", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "approval.html")


@app.post("/ingest")
async def ingest(body: IngestRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    thread_id = str(uuid4())
    _threads[thread_id] = {"domain": body.domain, "status": "running", "error": None}
    background_tasks.add_task(_run_until_pause, thread_id, body.ticket, body.domain)
    return {"thread_id": thread_id}


@app.post("/ingest/demo")
async def ingest_demo(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Ingest the built-in MFA-loop sample ticket (support domain)."""
    thread_id = str(uuid4())
    _threads[thread_id] = {"domain": "support", "status": "running", "error": None}
    background_tasks.add_task(_run_until_pause, thread_id, SAMPLE_TICKET, "support")
    return {"thread_id": thread_id}


@app.get("/threads")
async def list_threads() -> list[dict]:
    return [
        {"thread_id": thread_id, **meta}
        for thread_id, meta in _threads.items()
    ]


@app.get("/pending")
async def pending() -> list[dict]:
    graph = _get_graph()
    pending_items: list[dict] = []
    for thread_id in list(_threads):
        config = {"configurable": {"thread_id": thread_id}}
        snap = graph.get_state(config)
        for task in snap.tasks:
            for intr in task.interrupts:
                pending_items.append(
                    {
                        "thread_id": thread_id,
                        "payload": intr.value,
                    }
                )
    return pending_items


@app.post("/approve/{thread_id}")
async def approve(thread_id: str, body: ApproveRequest, background_tasks: BackgroundTasks) -> dict:
    if thread_id not in _threads:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    payload = {"action": body.action, "edited_body": body.edited_body}
    _threads[thread_id]["status"] = "resuming"
    background_tasks.add_task(_resume, thread_id, payload)
    return {"thread_id": thread_id, "action": body.action, "status": "resumed"}


def main() -> None:
    """Run with: uv run python -m app.main"""
    import os

    import uvicorn

    from app.llm import DEFAULT_MODEL, is_fake_chat_model

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8002"))
    model = os.getenv("RTTA_MODEL", DEFAULT_MODEL)
    mode = "offline fake" if is_fake_chat_model(model) else "real cloud"
    print(f"\n  Resonance Ticket Triage — Approval UI")
    print(f"  Model: {model} ({mode})")
    print(f"  Open: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
