"""Run ruff check --fix, ruff format, and mypy on src/."""
import subprocess
import sys

SRC = "src"


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


failures: list[str] = []

if run(["ruff", "check", "--fix", SRC]) != 0:
    failures.append("ruff check")

if run(["ruff", "format", SRC]) != 0:
    failures.append("ruff format")

if run(["mypy", SRC]) != 0:
    failures.append("mypy")

if failures:
    print(f"\nFailed: {', '.join(failures)}", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll checks passed.")
