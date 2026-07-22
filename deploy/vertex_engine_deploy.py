"""Deploy RTTA-AI-Multi-Agent-Ticket-Triage to Vertex AI Agent Engine."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is importable when run as `python deploy/vertex_engine_deploy.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    try:
        from dotenv import load_dotenv

        for _env in (_ROOT / ".env", _ROOT.parent / "RAIRA-AI-Research-Assistant" / ".env"):
            if _env.exists():
                load_dotenv(_env, override=False)
                break
    except Exception:
        pass

    import vertexai
    from vertexai import agent_engines

    from app.graph import build_graph

    vertexai.init(
        project=os.environ["GCP_PROJECT"],
        location=os.environ.get("GCP_LOCATION", "us-central1"),
        staging_bucket=f"gs://{os.environ['GCP_BUCKET']}",
    )

    langgraph_agent = agent_engines.LanggraphAgent(
        model="gemini-2.5-pro",
        runnable=build_graph(),
        enable_tracing=True,
    )

    deployed = agent_engines.create(
        langgraph_agent,
        requirements=[
            "langgraph>=1.2,<2",
            "langchain-google-vertexai",
            "psycopg[binary]",
            "pydantic>=2",
            "pyyaml",
        ],
        display_name="RTTA-AI-Multi-Agent-Ticket-Triage",
    )
    print("Deployed:", deployed.resource_name)


if __name__ == "__main__":
    main()
