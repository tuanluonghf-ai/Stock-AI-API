"""INCEPTION Smoke Test & Contract Validation Harness.

Design goals
------------
- Fast, deterministic, and safe-by-default.
- Works even when local data files are missing.
- Never requires OpenAI / network.
- Surfaces schema drift and narrative guardrail violations early.

How to run
----------
From repo root (same folder as app.py):
  python smoke_inception.py

Or:
  python -m inception.smoke_runner

Optional integration mode (requires local data):
- Place Price_Vol.xlsx (and optionally Tickers Target Price.xlsx) under repo root,
  OR set INCEPTION_DATA_DIR to the folder containing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import sys
import time
import traceback
import re


# ------------------------------
# Internal helpers (no core deps)
# ------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<!\d)[\.!?]+(?!\d)")  # do not split 2.5
_NUM_RE = re.compile(r"\d+(?:[\.,]\d+)?")


def _as_text(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _split_sentences_v2(text: str) -> List[str]:
    t = _as_text(text).strip()
    if not t:
        return []
    # Normalize newlines to sentence boundaries
    t = re.sub(r"[\r\n]+", ". ", t)
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(t) if p and p.strip()]
    return parts or [t]


def _count_numbers(s: str) -> int:
    return len(_NUM_RE.findall(s or ""))


def validate_short_text_v2(
    text: Any,
    *,
    max_sentences: int = 2,
    max_numbers_per_sentence: int = 2,
    forbid_bullets: bool = True,
) -> Dict[str, Any]:
    """Local validator to avoid false sentence splits (e.g., decimals 2.5)."""
    t = _as_text(text).strip()
    issues: List[str] = []
    if not t:
        return {"ok": False, "issues": ["EMPTY"]}

    if "\n" in t or "\r" in t:
        issues.append("NEWLINE")

    if forbid_bullets:
        if re.search(r"(^|\s)([-*•]|\d+\))\s+", t):
            issues.append("BULLET_STYLE")

    sents = _split_sentences_v2(t)
    if len(sents) > int(max_sentences):
        issues.append(f"TOO_MANY_SENTENCES:{len(sents)}")

    for i, s in enumerate(sents, start=1):
        n = _count_numbers(s)
        if n > int(max_numbers_per_sentence):
            issues.append(f"TOO_MANY_NUMBERS:S{i}:{n}")

    return {"ok": len(issues) == 0, "issues": issues}


@dataclass
class SmokeResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    meta: Dict[str, Any]


def _require_dict(root: Dict[str, Any], key: str, errors: List[str]) -> Dict[str, Any]:
    v = root.get(key)
    if not isinstance(v, dict):
        errors.append(f"MISSING_OR_INVALID:{key} (expected dict)")
        return {}
    return v


def validate_result_contract(result: Any) -> SmokeResult:
    """Validate minimal Engine→UI contract.

    This is intentionally minimal and non-breaking: it catches the most common
    schema drifts that break UI rendering.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(result, dict):
        return SmokeResult(False, ["RESULT_NOT_DICT"], [], {"type": type(result).__name__})

    # Core packs
    ap = _require_dict(result, "AnalysisPack", errors)
    _require_dict(result, "DashboardSummaryPack", errors)

    # Modules is expected in most runs, but allow missing in contract-only mode
    if "Modules" in result and not isinstance(result.get("Modules"), dict):
        warnings.append("INVALID:Modules (expected dict)")

    # AnalysisPack sanity
    if isinstance(ap, dict):
        if not _as_text(ap.get("Ticker")).strip():
            errors.append("MISSING:AnalysisPack.Ticker")
        last = ap.get("Last") if isinstance(ap.get("Last"), dict) else {}
        if last is None or not isinstance(last, dict) or last.get("Close") is None:
            warnings.append("MISSING:AnalysisPack.Last.Close")

    # Narrative guardrails (local, robust)
    ndp = result.get("NarrativeDraftPack")
    if isinstance(ndp, dict):
        plan = ndp.get("plan") if isinstance(ndp.get("plan"), dict) else {}
        plan_text = (plan or {}).get("line_final") or (plan or {}).get("line_draft")
        v = validate_short_text_v2(plan_text, max_sentences=2, max_numbers_per_sentence=2)
        if not v.get("ok"):
            warnings.append("NarrativeDraftPack.plan.validator:" + ",".join(v.get("issues") or []))

    # If ContractPack exists, surface it (do not fail hard unless it says ERROR)
    cp = result.get("ContractPack")
    if isinstance(cp, dict):
        ok = cp.get("ok")
        if ok is False:
            warnings.append("ContractPack.ok=False")
        issues = cp.get("issues")
        if isinstance(issues, list) and issues:
            warnings.append(f"ContractPack.issues:{len(issues)}")

    return SmokeResult(len(errors) == 0, errors, warnings, {"keys": sorted(list(result.keys()))})


def _try_import_pipeline() -> Tuple[Optional[Any], Optional[str]]:
    try:
        from inception.core.pipeline import build_result  # type: ignore

        return build_result, None
    except Exception as e:
        return None, f"IMPORT_PIPELINE_FAIL: {e}"


def _has_local_data() -> bool:
    # If DataHub can resolve it under cwd, existence is enough.
    # We also respect INCEPTION_DATA_DIR.
    base = os.environ.get("INCEPTION_DATA_DIR")
    cand_dirs = [base] if base else []
    cand_dirs.append(os.getcwd())

    for d in cand_dirs:
        if not d:
            continue
        pv = os.path.join(d, "Price_Vol.xlsx")
        if os.path.exists(pv):
            return True
    return False


def run_smoke() -> SmokeResult:
    t0 = time.time()
    errors: List[str] = []
    warnings: List[str] = []
    meta: Dict[str, Any] = {}

    # 1) Contract-only mode: validate a synthetic minimal ResultPack
    synthetic = {
        "AnalysisPack": {"Ticker": "_SMOKE_", "Last": {"Close": 1.0}},
        "DashboardSummaryPack": {"schema": "DashboardSummaryPack.v1"},
        "NarrativeDraftPack": {
            "schema": "NarrativeDraftPack.v1",
            "plan": {"line_draft": "Kế hoạch theo kỷ luật; nếu sai kịch bản thì hạ rủi ro."},
        },
    }
    r1 = validate_result_contract(synthetic)
    if not r1.ok:
        errors.extend(["SYNTHETIC:" + x for x in r1.errors])
    warnings.extend(["SYNTHETIC:" + x for x in r1.warnings])

    # 2) Integration mode: run pipeline if local data exists
    build_result, imp_err = _try_import_pipeline()
    if imp_err:
        warnings.append(imp_err)
    elif build_result is not None:
        if _has_local_data():
            ticker = os.environ.get("INCEPTION_SMOKE_TICKER", "VCB").strip().upper() or "VCB"
            try:
                out = build_result(ticker=ticker)
                r2 = validate_result_contract(out)
                if not r2.ok:
                    errors.extend(["PIPELINE:" + x for x in r2.errors])
                warnings.extend(["PIPELINE:" + x for x in r2.warnings])
                meta["pipeline_keys"] = r2.meta.get("keys")
            except Exception as e:
                errors.append(f"PIPELINE_EXCEPTION:{e}")
                meta["trace"] = traceback.format_exc(limit=10)
        else:
            warnings.append("PIPELINE_SKIPPED: Price_Vol.xlsx not found (set INCEPTION_DATA_DIR or place file in repo root)")

    meta["elapsed_ms"] = int((time.time() - t0) * 1000)
    ok = len(errors) == 0
    return SmokeResult(ok, errors, warnings, meta)


def _print_report(r: SmokeResult) -> None:
    status = "PASSED" if r.ok else "FAILED"
    print(f"[SMOKE] {status} | errors={len(r.errors)} warnings={len(r.warnings)} | {r.meta.get('elapsed_ms','?')}ms")
    if r.errors:
        print("\n[ERRORS]")
        for e in r.errors:
            print(" -", e)
    if r.warnings:
        print("\n[WARNINGS]")
        for w in r.warnings:
            print(" -", w)


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv or sys.argv[1:]
    r = run_smoke()
    _print_report(r)
    return 0 if r.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
