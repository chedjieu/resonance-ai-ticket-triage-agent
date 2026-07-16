"""Run all Project 2 eval scripts sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    "triager_eval.py",
    "investigator_eval.py",
    "responder_eval.py",
    "e2e_eval.py",
)


def main() -> None:
    root = Path(__file__).resolve().parent
    failed = 0
    for name in SCRIPTS:
        print(f"\n{'=' * 60}\nRunning {name}\n{'=' * 60}")
        proc = subprocess.run([sys.executable, str(root / name)], cwd=str(root.parent))
        if proc.returncode != 0:
            failed += 1
            print(f"{name} exited {proc.returncode}")
    if failed:
        sys.exit(1)
    print("\nAll evals finished.")


if __name__ == "__main__":
    main()
