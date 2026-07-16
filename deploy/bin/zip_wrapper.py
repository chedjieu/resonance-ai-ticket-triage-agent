"""Zip shim for Windows — AgentCore direct_code_deploy requires `zip` on PATH."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "-r":
        print("zip shim only supports: zip -r ARCHIVE PATH ...", file=sys.stderr)
        return 1

    archive = Path(argv[1])
    paths = [Path(p) for p in argv[2:]]
    if not paths:
        print("zip -r requires at least one path", file=sys.stderr)
        return 1

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.as_posix())
            elif path.is_file():
                zf.write(path, path.as_posix())
            else:
                print(f"zip shim: path not found: {path}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
