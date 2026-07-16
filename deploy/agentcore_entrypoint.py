"""Re-export root entrypoint (kept for Day 7 prompt path deploy/agentcore_entrypoint.py)."""

from agentcore_entrypoint import app, handler

__all__ = ["app", "handler"]
