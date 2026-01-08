"""Validation helpers (non-raising by default).

These checks are intended for diagnostics and early detection of pack drift.
They should not crash the app in production.

Step 9 adds a small schema/required-path registry and an issue collector so
Executive Snapshot can surface warnings/errors without breaking rendering.

Design rules:
- Streamlit-free.
- Non-raising by default.
- Defensive about non-dict inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import as_dict


def _get_path(root: Any, path: Sequence[str]) -> Tuple[bool, Any]:
    cur = root
    for p in path:
        if not isinstance(cur, dict):
            return False, None
        if p not in cur:
            return False, None
        cur = cur.get(p)
    return True, cur


def validate_required_paths(pack: Any, paths: List[Sequence[str]]) -> List[str]:
    """Return list of missing path strings. Non-raising."""
    p = as_dict(pack)
    missing: List[str] = []
    for path in paths:
        ok, _ = _get_path(p, path)
        if not ok:
            missing.append(".".join(path))
    return missing


def validate_analysis_pack(ap: Any) -> Dict[str, Any]:
    """Validate a normalized analysis pack. Returns diagnostics."""
    apd = as_dict(ap)
    required = [
        ["Ticker"],
        ["Last", "Close"],
        ["PrimarySetup"],
        ["MasterScore", "Total"],
    ]
    missing = validate_required_paths(apd, required)
    return {"ok": len(missing) == 0, "missing": missing}


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, tuple, dict, set)):
        return len(v) == 0
    return False


def _add_issue(
    issues: List[Dict[str, Any]],
    severity: str,
    path: str,
    message: str,
    issue_type: str,
    value: Any = None,
) -> None:
    it = {
        "severity": severity,
        "path": path,
        "message": message,
        "type": issue_type,
    }
    if value is not None:
        it["value_type"] = type(value).__name__
    issues.append(it)


def collect_data_quality_pack(
    analysis_pack: Any,
    character_pack: Any,
    extra: Optional[Dict[str, Any]] = None,
    max_issues: int = 10,
) -> Dict[str, Any]:
    """Return a JSON-safe DataQualityPack.v1.

    The goal is high-signal early alerts, not exhaustive validation.
    """
    ap = as_dict(analysis_pack)
    cp = as_dict(character_pack)

    issues: List[Dict[str, Any]] = []

    # --- ERROR: core runtime / dashboard must-have ---
    core_checks = [
        ("AnalysisPack.Ticker", ["Ticker"], str, "Ticker missing/empty"),
        ("AnalysisPack.Last.Close", ["Last", "Close"], (int, float), "Last.Close missing"),
        ("AnalysisPack.MasterScore.Total", ["MasterScore", "Total"], (int, float), "MasterScore.Total missing"),
        ("AnalysisPack.Conviction", ["Conviction"], (int, float), "Conviction missing"),
        ("AnalysisPack.PrimarySetup.Name", ["PrimarySetup", "Name"], str, "PrimarySetup.Name missing"),
    ]

    for full_path, path, exp_t, msg in core_checks:
        ok, v = _get_path(ap, path)
        if not ok:
            _add_issue(issues, "ERROR", full_path, msg, "MISSING")
            continue
        if exp_t is not None and v is not None and not isinstance(v, exp_t):
            _add_issue(issues, "ERROR", full_path, msg, "TYPE", value=v)
            continue
        if isinstance(v, str) and _is_empty(v):
            _add_issue(issues, "ERROR", full_path, msg, "EMPTY")

    # --- WARN: completeness (safe to run but should be visible) ---
    warn_checks = [
        ("AnalysisPack.Scenario12.Name", ["Scenario12", "Name"], str, "Scenario12.Name missing"),
        ("AnalysisPack.ProTech.MA.Regime", ["ProTech", "MA", "Regime"], str, "ProTech.MA.Regime missing"),
        ("AnalysisPack.ProTech.Volume.Regime", ["ProTech", "Volume", "Regime"], str, "ProTech.Volume.Regime missing"),
        ("AnalysisPack.Fibonacci", ["Fibonacci"], dict, "Fibonacci pack missing"),
        ("AnalysisPack.Fibonacci.ShortWindow.Band", ["Fibonacci", "ShortWindow", "Band"], str, "Fib short band missing"),
        ("AnalysisPack.Fibonacci.LongWindow.Band", ["Fibonacci", "LongWindow", "Band"], str, "Fib long band missing"),
        ("CharacterPack.ClassName", ["ClassName"], str, "ClassName missing"),
        ("CharacterPack.Conviction", ["Conviction"], dict, "Conviction pack missing"),
    ]

    for full_path, path, exp_t, msg in warn_checks:
        root = ap if full_path.startswith("AnalysisPack") else cp
        ok, v = _get_path(root, path)
        if not ok:
            _add_issue(issues, "WARN", full_path, msg, "MISSING")
            continue
        if exp_t is not None and v is not None and not isinstance(v, exp_t):
            _add_issue(issues, "WARN", full_path, msg, "TYPE", value=v)
            continue
        if (isinstance(v, (dict, list)) and _is_empty(v)) or (isinstance(v, str) and _is_empty(v)):
            _add_issue(issues, "WARN", full_path, msg, "EMPTY")

    # Extra packs: only check schema key existence (keep minimal)
    if isinstance(extra, dict):
        for name, pack in extra.items():
            pdict = as_dict(pack)
            if not isinstance(pdict.get("schema"), str) or _is_empty(pdict.get("schema")):
                _add_issue(issues, "WARN", f"{name}.schema", "schema missing", "MISSING")

    # deterministic ordering: ERROR first then WARN, then by path
    def _k(i: Dict[str, Any]) -> Tuple[int, str]:
        sev = i.get("severity") or "WARN"
        rank = 0 if sev == "ERROR" else 1
        return rank, str(i.get("path") or "")

    issues_sorted = sorted(issues, key=_k)

    err = sum(1 for x in issues_sorted if x.get("severity") == "ERROR")
    warn = sum(1 for x in issues_sorted if x.get("severity") == "WARN")

    return {
        "schema": "DataQualityPack.v1",
        "error_count": int(err),
        "warn_count": int(warn),
        "issues": issues_sorted[: max(0, int(max_issues))],
    }
