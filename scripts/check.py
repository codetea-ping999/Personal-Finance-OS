#!/usr/bin/env python3
"""Run the repository's portable, deterministic verification harness.

Keep this script dependency-light so local development and CI execute exactly
the same gates. It assumes the project's ``.[dev]`` dependencies are installed.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    """Run one gate from the repository root and fail fast on an error."""
    command = [sys.executable, *args]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def missing_development_dependencies() -> list[str]:
    """Return the optional test tools that the full gate needs."""
    required = {"pytest": "pytest", "pytest_cov": "pytest-cov", "coverage": "coverage"}
    return [package for module, package in required.items() if find_spec(module) is None]


def main() -> int:
    missing = missing_development_dependencies()
    if missing:
        joined = ", ".join(missing)
        print(
            f"Missing development dependency/dependencies: {joined}. "
            'Install them with: python -m pip install -e ".[dev]"',
            file=sys.stderr,
        )
        return 2
    run("-m", "compileall", "-q", "src", "tests")
    run(
        "-m",
        "pytest",
        "--cov=personal_finance_os",
        "--cov-report=term-missing",
        "--cov-report=xml",
    )
    run("-m", "coverage", "report", "--fail-under=80")
    run(
        "-m",
        "coverage",
        "report",
        "--include=src/personal_finance_os/ai_cfo.py,src/personal_finance_os/ai_intents.py",
        "--fail-under=90",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
