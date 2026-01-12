"""Repo-root wrapper for INCEPTION smoke harness.

Run:
  python smoke_inception.py

This keeps the entry point stable even if package paths evolve.
"""

from inception.smoke_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
