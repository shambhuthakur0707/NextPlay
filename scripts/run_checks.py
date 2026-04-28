#!/usr/bin/env python3
"""Run lightweight repository quality checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> int:
    print(f"\n[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    checks = [
        [sys.executable, "-m", "unittest", "tests/test_metadata_sync.py"],
        [sys.executable, "-m", "unittest", "tests/test_model_switching.py"],
    ]

    for cmd in checks:
        code = run(cmd, cwd=repo_root)
        if code != 0:
            print("\n[fail] one or more checks failed.")
            return code

    print("\n[ok] all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
