"""Ticket triage LangGraph — supervisor routes workers in a loop."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from app.agents.hitl import hitl_node
from app.agents.investigator import investigator_node
from app.agents.responder import responder_node
from app.agents.send import send_node
from app.agents.supervisor import supervisor_node
from app.agents.triager import triager_node
from app.state import TicketState

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints.sqlite"


def build_graph_with_backends(saver: Any, store: BaseStore | None = None):
    """Compile the ticket triage graph with injected checkpoint saver and store."""
    builder = StateGraph(TicketState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triager", triager_node)
    builder.add_node("investigator", investigator_node)
    builder.add_node("responder", responder_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("send", send_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "triager": "triager",
            "investigator": "investigator",
            "responder": "responder",
            "hitl": "hitl",
            "send": "send",
            "END": END,
        },
    )
    for worker in ("triager", "investigator", "responder", "hitl", "send"):
        builder.add_edge(worker, "supervisor")

    return builder.compile(checkpointer=saver, store=store)


def build_graph():
    """Compile the ticket triage graph with a SqliteSaver checkpointer (local dev)."""
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    return build_graph_with_backends(SqliteSaver(conn))


def make_initial_state(ticket_id: str, raw: dict, domain: str) -> TicketState:
    return {
        "ticket_id": ticket_id,
        "raw": raw,
        "domain": domain,  # type: ignore[typeddict-item]
        "classification": None,
        "severity": None,
        "findings": [],
        "draft": None,
        "approval": "pending",
        "sent": False,
        "step_log": [],
        "next": "END",
    }


SAMPLE_TICKET = {
    "id": "TKT-1001",
    "subject": "Cannot log in - MFA loop",
    "body": (
        "Hi, every time I enter my code from the authenticator app it sends me "
        "back to the login page. I tried 3 different codes. Chrome on macOS."
    ),
    "sender": "priya.s@example.com",
    "attachments": [],
}


if __name__ == "__main__":
    graph = build_graph()
    state = make_initial_state("TKT-1001", SAMPLE_TICKET, "support")
    config = {"configurable": {"thread_id": "demo-ticket-1001"}}

    print(f"Processing ticket {state['ticket_id']} ({state['domain']})\n")
    for update in graph.stream(state, config, stream_mode="updates"):
        for node_name, node_update in update.items():
            if node_update.get("step_log"):
                print(f"  [{node_name}] {node_update['step_log'][-1]}")

    final = graph.get_state(config).values
    print(f"\nDone — sent={final['sent']}, severity={final['severity']}")
    print(f"Total steps: {len(final['step_log'])}")
