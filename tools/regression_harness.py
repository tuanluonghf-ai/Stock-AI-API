"""Regression harness for deterministic ResultPack snapshots.

Usage:
  python tools/regression_harness.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from inception.core.pipeline import build_result

TICKERS = ["VNM", "CII", "MSN"]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _normalize_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _normalize_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_normalize_json(v) for v in x]
    if isinstance(x, tuple):
        return [_normalize_json(v) for v in x]
    if isinstance(x, set):
        return sorted([_normalize_json(v) for v in x], key=lambda v: str(v))
    if np is not None:
        try:
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
        except Exception:
            pass
    return x


def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and x != x
    except Exception:
        return False


def _dump_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_whitelisted_path(path: str) -> bool:
    parts = [p.lower() for p in path.replace("[", ".").replace("]", "").split(".") if p]
    dynamic = {"generated_at", "runtime_ms", "timestamp", "seed", "cache", "cached", "cache_hit", "cache_key"}
    for p in parts:
        if p in dynamic:
            return True
        if p.endswith("_time") or p.endswith("_ts") or p.endswith("_date"):
            return True
        if p in {"time", "time_ms", "time_s"}:
            return True
    return False


def _diff(a: Any, b: Any, path: str = "") -> List[Tuple[str, Any, Any]]:
    diffs: List[Tuple[str, Any, Any]] = []
    if _is_nan(a) and _is_nan(b):
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a.keys()) | set(b.keys()), key=lambda x: str(x))
        for k in keys:
            kp = f"{path}.{k}" if path else str(k)
            if k not in a:
                diffs.append((kp, None, b.get(k)))
                continue
            if k not in b:
                diffs.append((kp, a.get(k), None))
                continue
            diffs.extend(_diff(a.get(k), b.get(k), kp))
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((f"{path}[len]", len(a), len(b)))
        for i in range(min(len(a), len(b))):
            diffs.extend(_diff(a[i], b[i], f"{path}[{i}]"))
        return diffs
    if a != b:
        diffs.append((path, a, b))
    return diffs


def _collect_strings(x: Any, path: str = "") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(x, dict):
        for k, v in x.items():
            kp = f"{path}.{k}" if path else str(k)
            out.extend(_collect_strings(v, kp))
        return out
    if isinstance(x, list):
        for i, v in enumerate(x):
            out.extend(_collect_strings(v, f"{path}[{i}]"))
        return out
    if isinstance(x, str):
        out.append((path, x))
    return out


def _collect_warnings(flat: Dict[str, Any]) -> List[str]:
    warns: List[str] = []
    ap = flat.get("AnalysisPack") if isinstance(flat.get("AnalysisPack"), dict) else {}
    for src in (flat, ap):
        w = src.get("_Warnings") if isinstance(src, dict) else None
        if isinstance(w, list):
            warns.extend([str(x) for x in w if str(x).strip()])
    return warns


def _collect_data_quality_issues(x: Any) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    if isinstance(x, dict):
        if "issues" in x and isinstance(x.get("issues"), list):
            for it in x.get("issues"):
                if isinstance(it, dict):
                    issues.append(it)
        for v in x.values():
            issues.extend(_collect_data_quality_issues(v))
    elif isinstance(x, list):
        for v in x:
            issues.extend(_collect_data_quality_issues(v))
    return issues


def _has_narrative_guardrail_warnings(flat: Dict[str, Any]) -> List[str]:
    hits: List[str] = []
    tokens = ["NARRATIVE", "GUARDRAIL", "SAFETY_LINE", "TOO_MANY_NUMBERS", "KEY_ORDER", "BANNED_WORD", "FORBID_TOKEN", "DNA"]
    warns = _collect_warnings(flat)
    for w in warns:
        wl = w.upper()
        if any(t in wl for t in tokens):
            hits.append(w)
    dq_issues = _collect_data_quality_issues(flat)
    for it in dq_issues:
        msg = str(it.get("message") or "")
        path = str(it.get("path") or "")
        s = f"{msg} {path}".upper()
        if "NARRATIVE" in s or "DNA" in s:
            hits.append(f"DATA_QUALITY:{msg}:{path}")
    return hits


def _check_css_header_size(repo_root: str) -> Tuple[bool, str]:
    styles_path = os.path.join(repo_root, "inception", "ui", "styles.py")
    try:
        with open(styles_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if "60px" in content:
            return True, ""
        return False, "styles.py missing 60px"
    except Exception as exc:
        return False, f"styles.py read failed: {exc.__class__.__name__}"


def _build_flat(result: Dict[str, Any]) -> Dict[str, Any]:
    ap = result.get("AnalysisPack") if isinstance(result.get("AnalysisPack"), dict) else {}
    mods = result.get("Modules") if isinstance(result.get("Modules"), dict) else {}
    flat: Dict[str, Any] = {
        "AnalysisPack": ap,
        "DashboardSummaryPack": result.get("DashboardSummaryPack"),
        "Modules": {"character": mods.get("character")},
        "NarrativeDraftPack": result.get("NarrativeDraftPack") or ap.get("NarrativeDraftPack"),
        "TradePlanPack": ap.get("TradePlanPack"),
        "DecisionPack": ap.get("DecisionPack"),
        "_Warnings": result.get("_Warnings") or ap.get("_Warnings"),
        "_ModuleErrors": result.get("_ModuleErrors"),
    }
    return _normalize_json(flat)


def run() -> int:
    os.environ.setdefault("INCEPTION_DISABLE_STABILITY_STATE", "1")
    os.environ.setdefault("INCEPTION_PERSIST_STABILITY", "0")

    repo_root = _repo_root()
    golden_dir = os.path.join(repo_root, "golden")
    os.makedirs(golden_dir, exist_ok=True)

    css_ok, css_msg = _check_css_header_size(repo_root)
    if not css_ok:
        print(f"FAIL CSS: {css_msg}")

    overall_ok = css_ok

    for ticker in TICKERS:
        ok = True
        reasons: List[str] = []

        result = build_result(ticker=ticker)
        if not isinstance(result, dict) or result.get("Error"):
            ok = False
            reasons.append(f"build_result error: {result.get('Error') if isinstance(result, dict) else 'invalid result'}")
            print(f"FAIL {ticker}: {', '.join(reasons)}")
            overall_ok = False
            continue

        flat = _build_flat(result)
        golden_path = os.path.join(golden_dir, f"{ticker}.flat.json")

        # N/A check (UI-facing text)
        na_hits = [(p, v) for p, v in _collect_strings(flat) if "N/A" in v]
        if na_hits:
            ok = False
            reasons.append(f"N/A strings: {len(na_hits)}")

        # Narrative guardrail warnings check
        guard_hits = _has_narrative_guardrail_warnings(flat)
        if guard_hits:
            ok = False
            reasons.append(f"narrative guardrails: {len(guard_hits)}")

        if not os.path.exists(golden_path):
            _dump_json(golden_path, flat)
        else:
            old = _load_json(golden_path)
            diffs = _diff(old, flat)
            diffs = [d for d in diffs if not _is_whitelisted_path(d[0])]
            if diffs:
                ok = False
                reasons.append(f"diffs: {len(diffs)}")
                for path, a, b in diffs[:10]:
                    print(f"DIFF {ticker}: {path} -> {a} | {b}")

        if ok:
            print(f"PASS {ticker}")
        else:
            print(f"FAIL {ticker}: {', '.join(reasons)}")
            overall_ok = False

    print("PASS overall" if overall_ok else "FAIL overall")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(run())
