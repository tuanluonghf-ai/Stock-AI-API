from __future__ import annotations

"""ZonePack builder (UI-side only).

Purpose:
- Normalize "zones" (support/resistance/positive/neutral/risk/reclaim) from existing
  Current Status / TradePlan facts.
- Provide a single source of truth for chart shading and narrative phrasing.
This module MUST NOT compute new signals; it only packages existing levels/zones.
"""

from typing import Any, Dict, List, Optional, Tuple
import math

def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if not s:
            return default
        return float(s)
    except Exception:
        return default

def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def _zone_from(d: Any) -> Tuple[float, float]:
    if not isinstance(d, dict):
        return (float("nan"), float("nan"))
    lo = _sf(d.get("Low"))
    hi = _sf(d.get("High"))
    if _is_nan(lo) or _is_nan(hi):
        return (float("nan"), float("nan"))
    return (float(min(lo, hi)), float(max(lo, hi)))

def compute_zone_pack(analysis_pack: Dict[str, Any], df_price: Any = None) -> Dict[str, Any]:
    ap = analysis_pack or {}
    if not isinstance(ap, dict):
        ap = {}

    # Current price (for classification)
    last_close = float("nan")
    try:
        if df_price is not None and hasattr(df_price, "__len__") and len(df_price) > 0:
            # prefer Close column
            if hasattr(df_price, "iloc") and hasattr(df_price, "columns") and "Close" in list(df_price.columns):
                last_close = _sf(df_price["Close"].iloc[-1])
    except Exception:
        pass

    tpp = ap.get("TradePlanPack") or {}
    if not isinstance(tpp, dict):
        tpp = {}

    plan_primary = tpp.get("plan_primary") or {}
    plan_alt = tpp.get("plan_alt") or {}
    plans_all = tpp.get("plans_all") or []
    if not isinstance(plan_primary, dict):
        plan_primary = {}
    if not isinstance(plan_alt, dict):
        plan_alt = {}
    if not isinstance(plans_all, list):
        plans_all = []

    entry_lo, entry_hi = _zone_from(plan_primary.get("entry_zone"))
    reclaim_lo, reclaim_hi = _zone_from(plan_alt.get("defensive_reclaim_zone"))
    hard_stop = _sf(plan_alt.get("defensive_hard_stop"))

    # fallback from first DEFENSIVE plan (same logic you already had in chart)
    if (_is_nan(reclaim_lo) or _is_nan(reclaim_hi) or _is_nan(hard_stop)) and plans_all:
        for c in plans_all:
            if not isinstance(c, dict):
                continue
            if str(c.get("type") or "").strip().upper() != "DEFENSIVE":
                continue
            if _is_nan(hard_stop):
                hard_stop = _sf(c.get("defensive_hard_stop"))
            if _is_nan(reclaim_lo) or _is_nan(reclaim_hi):
                zl, zh = _zone_from(c.get("defensive_reclaim_zone"))
                if not _is_nan(zl) and not _is_nan(zh):
                    reclaim_lo, reclaim_hi = zl, zh
            break

    zones: List[Dict[str, Any]] = []
    if not _is_nan(entry_lo) and not _is_nan(entry_hi) and entry_lo != entry_hi:
        zones.append({"name": "POSITIVE", "low": entry_lo, "high": entry_hi, "source": "TradePlan.entry_zone"})
    if not _is_nan(reclaim_lo) and not _is_nan(reclaim_hi) and reclaim_lo != reclaim_hi:
        zones.append({"name": "RECLAIM", "low": reclaim_lo, "high": reclaim_hi, "source": "TradePlan.defensive_reclaim_zone"})
    if not _is_nan(hard_stop):
        zones.append({"name": "RISK", "low": float("-inf"), "high": float(hard_stop), "source": "TradePlan.defensive_hard_stop"})

    # Classify current location (simple, deterministic)
    zone_now = "NEUTRAL"
    if not _is_nan(last_close):
        for z in zones:
            lo = z["low"]
            hi = z["high"]
            if lo == float("-inf"):
                if last_close <= hi:
                    zone_now = z["name"]
                    break
            else:
                if last_close >= lo and last_close <= hi:
                    zone_now = z["name"]
                    break

    return {
        "version": "1.0",
        "last_close": None if _is_nan(last_close) else float(last_close),
        "zone_now": zone_now,
        "zones": zones,
    }
