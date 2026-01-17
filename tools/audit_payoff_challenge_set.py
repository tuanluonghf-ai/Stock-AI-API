from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "inception" / "tests" / "data" / "frozen"
OUT_DIR = ROOT / "audits"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from inception.core.payoff_tier_pack import LookbackCfg, compute_payoff_tier_v1  # noqa: E402
from inception.core.pipeline import build_result  # noqa: E402
from inception.infra.datahub import DataHub  # noqa: E402

TARGETS_PER_TIER = {"LOW": 10, "MEDIUM": 10, "HIGH": 10}
AUDIT_KEYWORDS = [
    "biên",
    "biên độ",
    "payoff",
    "opportunity cost",
    "đáng để",
    "không đủ biên",
    "biên hạn chế",
]


@dataclass
class Candidate:
    ticker: str
    payoff_tier: str
    payoff_span_norm: float
    shape: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_text(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _extract_shape(result: Dict[str, Any]) -> Optional[str]:
    mods = result.get("Modules") if isinstance(result.get("Modules"), dict) else {}
    cp = mods.get("character") if isinstance(mods.get("character"), dict) else {}
    for key in ("CharacterClass", "Class", "ClassName"):
        v = cp.get(key) if isinstance(cp, dict) else None
        if v:
            return _safe_text(v).strip()
    ap = result.get("AnalysisPack") if isinstance(result.get("AnalysisPack"), dict) else {}
    v = ap.get("CharacterClass")
    return _safe_text(v).strip() if v else None


def _persona_from_pack(result: Dict[str, Any]) -> str:
    ap = result.get("AnalysisPack") if isinstance(result.get("AnalysisPack"), dict) else {}
    pack = ap.get("InvestorMappingPack") if isinstance(ap.get("InvestorMappingPack"), dict) else None
    if not isinstance(pack, dict):
        pack = result.get("InvestorMappingPack") if isinstance(result.get("InvestorMappingPack"), dict) else {}
    personas = None
    for key in ("Personas", "PersonaMatch", "Compatibility", "personas", "persona_match", "compatibility"):
        if key in pack:
            personas = pack.get(key)
            break
    items: List[Dict[str, Any]] = []
    if isinstance(personas, list):
        for it in personas:
            if isinstance(it, dict):
                items.append(it)
    elif isinstance(personas, dict):
        for k, it in personas.items():
            if isinstance(it, dict):
                it = dict(it)
                it.setdefault("name", k)
                items.append(it)

    def _score(item: Dict[str, Any]) -> float:
        for k in ("score_10", "Score10", "score"):
            if k in item:
                try:
                    return float(item.get(k))
                except Exception:
                    return float("nan")
        return float("nan")

    def _name(item: Dict[str, Any]) -> str:
        return _safe_text(item.get("name") or item.get("Name") or item.get("Persona") or "").strip()

    def _label(item: Dict[str, Any]) -> str:
        return _safe_text(item.get("label") or item.get("Label") or "").strip()

    items.sort(key=lambda it: (-( _score(it) if _score(it) == _score(it) else -1e9), _name(it)))
    if items:
        top = items[0]
        return _name(top) or _label(top) or "-"
    return "-"


def _narrative_text(result: Dict[str, Any]) -> str:
    ap = result.get("AnalysisPack") if isinstance(result.get("AnalysisPack"), dict) else {}
    ndp = result.get("NarrativeDraftPack") if isinstance(result.get("NarrativeDraftPack"), dict) else None
    if not isinstance(ndp, dict):
        ndp = ap.get("NarrativeDraftPack") if isinstance(ap.get("NarrativeDraftPack"), dict) else {}
    texts: List[str] = []
    for section in ("status", "plan", "dna", "summary", "market"):
        sec = ndp.get(section) if isinstance(ndp, dict) else None
        if isinstance(sec, dict):
            for k in ("text", "line", "line_final", "line_draft", "content"):
                if isinstance(sec.get(k), str):
                    texts.append(sec.get(k))
    if isinstance(ndp, dict):
        for v in ndp.values():
            if isinstance(v, str):
                texts.append(v)
    return " ".join(texts).lower()


def _persona_payoff_flags(payoff_tier: str, persona: str) -> Tuple[bool, str]:
    persona_norm = _safe_text(persona).strip().lower()
    if payoff_tier == "LOW" and persona_norm in {"speculator", "compounder"}:
        return True, "LOW_payoff_high_persona"
    if payoff_tier == "MEDIUM" and persona_norm == "compounder":
        return True, "MEDIUM_payoff_overreach"
    return False, "CONSISTENT"


def _narrative_tradeoff_missing(payoff_tier: str, persona: str, text: str) -> bool:
    persona_norm = _safe_text(persona).strip().lower()
    if payoff_tier in {"LOW", "MEDIUM"} and persona_norm in {"speculator", "compounder", "alphahunter"}:
        return not any(k in text for k in AUDIT_KEYWORDS)
    return False


def _select_bucket(items: List[Candidate], target: int) -> List[Candidate]:
    if not items:
        return []
    items = sorted(items, key=lambda x: x.payoff_span_norm)
    if len(items) <= target:
        return items
    low = items[:3]
    high = items[-3:]
    remaining = [x for x in items if x not in low and x not in high]
    if len(remaining) >= 4:
        mid_start = max(0, (len(remaining) // 2) - 2)
        mid = remaining[mid_start: mid_start + 4]
    else:
        mid = remaining
    selected = low + mid + high
    if len(selected) > target:
        selected = selected[:target]
    return selected


def _payoff_from_df(df: pd.DataFrame, lookback_cfg: LookbackCfg) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None, "Empty OHLC"
    if "Close" not in df.columns or "High" not in df.columns or "Low" not in df.columns:
        return None, None, "Missing OHLC columns"
    try:
        df = df.sort_values("Date") if "Date" in df.columns else df
    except Exception:
        pass
    try:
        ref = float(df["Close"].iloc[-1])
    except Exception:
        return None, None, "Invalid reference price"
    if not np.isfinite(ref) or ref == 0:
        return None, None, "Invalid reference price"
    payoff = compute_payoff_tier_v1(df=df, reference_price=ref, lookback_cfg=lookback_cfg)
    tier = _safe_text(payoff.get("payoff_tier")).strip().upper()
    span = payoff.get("payoff_span_norm")
    try:
        span_f = float(span)
    except Exception:
        return None, None, "Invalid payoff_span_norm"
    if tier not in {"LOW", "MEDIUM", "HIGH"}:
        return None, None, f"Unknown tier: {tier}"
    return tier, span_f, None


def _percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}
    out = {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }
    return out


def _select_stratified(items: List[Candidate], n_low: int, n_mid: int, n_high: int) -> List[Candidate]:
    if not items:
        return []
    ordered = sorted(items, key=lambda x: x.payoff_span_norm)
    total = len(ordered)
    n_low = max(0, min(n_low, total))
    n_high = max(0, min(n_high, total - n_low))
    remaining = total - n_low - n_high
    n_mid = max(0, min(n_mid, remaining))

    low_part = ordered[:n_low]
    high_part = ordered[-n_high:] if n_high else []

    mid_part: List[Candidate] = []
    if n_mid:
        mid_start = n_low
        mid_end = total - n_high
        mid_pool = ordered[mid_start:mid_end]
        if len(mid_pool) <= n_mid:
            mid_part = mid_pool
        else:
            center = len(mid_pool) // 2
            half = n_mid // 2
            start = max(0, center - half)
            end = start + n_mid
            if end > len(mid_pool):
                end = len(mid_pool)
                start = max(0, end - n_mid)
            mid_part = mid_pool[start:end]

    selected = low_part + mid_part + high_part
    if len(selected) > (n_low + n_mid + n_high):
        selected = selected[: (n_low + n_mid + n_high)]
    return selected


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FROZEN_DIR.exists():
        print(f"ERROR: frozen data dir missing: {FROZEN_DIR}")
        return 1

    os.environ["INCEPTION_DATA_DIR"] = str(FROZEN_DIR)

    hub = DataHub.from_env(default_dir=str(FROZEN_DIR))
    df_all = hub.load_price_vol("Price_Vol.xlsx")
    if df_all.empty or "Ticker" not in df_all.columns:
        print("ERROR: Price_Vol.xlsx missing or invalid")
        return 1

    universe = sorted({str(t).strip().upper() for t in df_all["Ticker"].dropna().unique().tolist()})
    lookback_cfg = LookbackCfg(id="PAYOFF_V1_250D", window=250)
    candidates: Dict[str, List[Candidate]] = {"LOW": [], "MEDIUM": [], "HIGH": []}
    payoff_failures: List[Dict[str, Any]] = []
    universe_rows: List[Dict[str, Any]] = []

    for ticker in universe:
        df_t = df_all[df_all["Ticker"].astype(str).str.upper() == ticker].copy()
        tier, span_f, err = _payoff_from_df(df_t, lookback_cfg)
        if err:
            payoff_failures.append({"ticker": ticker, "error": err})
            continue
        universe_rows.append({"ticker": ticker, "payoff_span_norm": span_f, "payoff_tier": tier})
        candidates[tier].append(Candidate(ticker=ticker, payoff_tier=tier, payoff_span_norm=span_f))

    universe_csv = OUT_DIR / "payoff_universe_tier_distribution.csv"
    with universe_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "payoff_span_norm", "payoff_tier"])
        writer.writeheader()
        for row in sorted(universe_rows, key=lambda r: r["ticker"]):
            writer.writerow(row)

    spans = [float(r["payoff_span_norm"]) for r in universe_rows if r.get("payoff_span_norm") is not None]
    pct = _percentiles(spans)
    summary = {
        "universe_total": len(universe_rows),
        "counts": {k: len(v) for k, v in candidates.items()},
        "min": pct["min"],
        "p10": pct["p10"],
        "p25": pct["p25"],
        "p50": pct["p50"],
        "p75": pct["p75"],
        "p90": pct["p90"],
        "max": pct["max"],
        "min_span_norm": pct["min"],
        "max_span_norm": pct["max"],
    }
    summary_path = OUT_DIR / "payoff_universe_tier_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selected: List[Candidate] = []
    counts: Dict[str, int] = {}
    available: Dict[str, int] = {k: len(v) for k, v in candidates.items()}

    low_bucket = sorted(candidates.get("LOW", []), key=lambda x: x.payoff_span_norm)
    if low_bucket:
        selected.append(low_bucket[0])
    counts["LOW"] = 1 if low_bucket else 0

    medium_bucket = _select_stratified(candidates.get("MEDIUM", []), n_low=3, n_mid=4, n_high=3)
    selected.extend(medium_bucket)
    counts["MEDIUM"] = len(medium_bucket)

    high_bucket = _select_stratified(candidates.get("HIGH", []), n_low=6, n_mid=7, n_high=6)
    selected.extend(high_bucket)
    counts["HIGH"] = len(high_bucket)

    selection_debug = {
        "generated_at": _utc_now(),
        "targets_per_tier": TARGETS_PER_TIER,
        "available_per_tier": available,
        "selected": [
            {"ticker": c.ticker, "payoff_tier": c.payoff_tier, "payoff_span_norm": c.payoff_span_norm}
            for c in selected
        ],
        "selection_method": {
            "LOW": "single_lowest_span",
            "MEDIUM": "3_low + 4_mid + 3_high by payoff_span_norm",
            "HIGH": "6_low + 7_mid + 6_high by payoff_span_norm",
        },
    }
    selection_debug_path = OUT_DIR / "payoff_challenge_set_selection_debug.json"
    selection_debug_path.write_text(
        json.dumps(selection_debug, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    shortfalls = {}
    desired = {"LOW": 1, "MEDIUM": 10, "HIGH": 19}
    for tier, target in desired.items():
        if available.get(tier, 0) < target:
            shortfalls[tier] = {"target": target, "available": available.get(tier, 0), "selected": counts.get(tier, 0)}

    manifest = {
        "generated_at": _utc_now(),
        "source": "pinned_dataset",
        "targets_per_tier": {"LOW": 1, "MEDIUM": 10, "HIGH": 19},
        "available_per_tier": available,
        "selected": [
            {
                "ticker": c.ticker,
                "payoff_tier": c.payoff_tier,
                "payoff_span_norm": c.payoff_span_norm,
                "shape": c.shape,
            }
            for c in selected
        ],
        "counts": counts,
        "payoff_failures": payoff_failures,
        "shortfalls": shortfalls,
        "artifacts": {
            "universe_distribution_csv": str(universe_csv),
            "universe_distribution_summary": str(summary_path),
            "selection_debug": str(selection_debug_path),
        },
        "notes": "selection is deterministic; tier buckets from OHLC-only PayoffTier v1",
    }

    manifest_path = OUT_DIR / "payoff_challenge_set_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    reports: List[Dict[str, Any]] = []
    agg = {
        "total": 0,
        "persona_mismatch": 0,
        "narrative_missing": 0,
        "pipeline_failed": 0,
        "evaluated_total": 0,
        "by_tier": {},
    }

    for c in selected:
        result = build_result(ticker=c.ticker, data_dir=str(FROZEN_DIR))
        if not isinstance(result, dict) or result.get("Error"):
            err = result.get("Error") if isinstance(result, dict) else "Invalid result"
            reports.append(
                {
                    "ticker": c.ticker,
                    "payoff_tier": c.payoff_tier,
                    "payoff_span_norm": c.payoff_span_norm,
                    "persona": "-",
                    "persona_payoff_mismatch": False,
                    "persona_payoff_reason": "PIPELINE_FAILED",
                    "narrative_tradeoff_missing": False,
                    "shape": None,
                    "pipeline_failed": True,
                    "pipeline_error": err,
                }
            )
            agg["total"] += 1
            agg["pipeline_failed"] += 1
            agg["by_tier"].setdefault(
                c.payoff_tier,
                {"total": 0, "persona_mismatch": 0, "narrative_missing": 0, "pipeline_failed": 0, "evaluated_total": 0},
            )
            agg["by_tier"][c.payoff_tier]["total"] += 1
            agg["by_tier"][c.payoff_tier]["pipeline_failed"] += 1
            continue
        payoff_tier = c.payoff_tier
        payoff_span_norm = c.payoff_span_norm
        persona = _persona_from_pack(result)
        text = _narrative_text(result)
        mismatch, reason = _persona_payoff_flags(payoff_tier, persona)
        narr_missing = _narrative_tradeoff_missing(payoff_tier, persona, text)
        shape = _extract_shape(result)

        reports.append(
            {
                "ticker": c.ticker,
                "payoff_tier": payoff_tier,
                "payoff_span_norm": payoff_span_norm,
                "persona": persona,
                "persona_payoff_mismatch": mismatch,
                "persona_payoff_reason": reason,
                "narrative_tradeoff_missing": narr_missing,
                "shape": shape,
                "pipeline_failed": False,
                "pipeline_error": "",
            }
        )

        agg["total"] += 1
        agg["evaluated_total"] += 1
        if mismatch:
            agg["persona_mismatch"] += 1
        if narr_missing:
            agg["narrative_missing"] += 1
        agg["by_tier"].setdefault(
            payoff_tier,
            {"total": 0, "persona_mismatch": 0, "narrative_missing": 0, "pipeline_failed": 0, "evaluated_total": 0},
        )
        agg["by_tier"][payoff_tier]["total"] += 1
        agg["by_tier"][payoff_tier]["evaluated_total"] += 1
        if mismatch:
            agg["by_tier"][payoff_tier]["persona_mismatch"] += 1
        if narr_missing:
            agg["by_tier"][payoff_tier]["narrative_missing"] += 1

    if agg["total"]:
        agg["persona_mismatch_pct"] = round((agg["persona_mismatch"] / agg["total"]) * 100, 2)
        agg["narrative_missing_pct"] = round((agg["narrative_missing"] / agg["total"]) * 100, 2)

    report = {"per_ticker": reports, "aggregate": agg}

    json_path = OUT_DIR / "payoff_persona_narrative_audit_challenge.json"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / "payoff_persona_narrative_audit_challenge.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "payoff_tier",
                "payoff_span_norm",
                "persona",
                "persona_payoff_mismatch",
                "persona_payoff_reason",
                "narrative_tradeoff_missing",
                "shape",
                "pipeline_failed",
                "pipeline_error",
            ],
        )
        writer.writeheader()
        for row in reports:
            writer.writerow(row)

    print(f"WROTE {manifest_path}")
    print(f"WROTE {json_path}")
    print(f"WROTE {csv_path}")
    if payoff_failures:
        print(f"WARN payoff failures: {len(payoff_failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
