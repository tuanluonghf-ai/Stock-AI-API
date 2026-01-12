from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

def test_golden_regression_compare():
    repo = Path(__file__).resolve().parents[2]
    golden = repo / "golden"
    manifest = golden / "_manifest.json"
    if not manifest.exists():
        # Baselines not created yet; do not fail.
        return
    data = json.loads(manifest.read_text(encoding="utf-8"))
    tickers = data.get("tickers") or []
    if not tickers:
        return
    cmd = [sys.executable, str(repo / "regression_golden.py"), "--tickers", *tickers]
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
