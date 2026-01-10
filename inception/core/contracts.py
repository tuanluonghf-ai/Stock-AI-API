"""Contracts & normalization utilities for INCEPTION packs.

This module standardizes pack shapes (types) across the pipeline.

Why this exists
- UI/report code often uses dict access patterns (`x.get(...)`).
- During refactors, upstream producers might return `int/float/str` where a `dict`
  is expected (e.g., a window length stored at a key that used to hold a dict).
- Normalizing packs at the source prevents runtime crashes and reduces the amount
  of defensive code sprinkled across UI/renderers.

Design
- Non-raising: returns safe defaults.
- Streamlit-free.
- Minimal opinion: we preserve existing values when possible.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


# ============================================================
# Primitive coercions
# ============================================================

def as_dict(x: Any) -> Dict[str, Any]:
    """Return x if it's a dict, else an empty dict."""
    return x if isinstance(x, dict) else {}


def as_list(x: Any) -> List[Any]:
    """Return x if it's a list/tuple, else empty list."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return []


def as_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


def as_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


# ============================================================
# Pack normalizers
# ============================================================

def normalize_fibonacci_pack(fib_in: Any) -> Dict[str, Any]:
    """Normalize AnalysisPack['Fibonacci'].

    Important compatibility shim:
    - Some report/UI code expects `Fibonacci.ShortWindow` and `Fibonacci.LongWindow`
      to be dict-like packs (e.g., `.get('Band')`).
    - The pipeline may store these as raw ints (window lengths).

    We normalize them to dicts:
      ShortWindow = {'Value': <int>, 'Band': <ctx.ShortBand or 'N/A'>}
      LongWindow  = {'Value': <int>, 'Band': <ctx.LongBand  or 'N/A'>}

    If ShortWindow/LongWindow are already dicts, we keep them.
    """
    fib = as_dict(fib_in)

    ctx = as_dict(fib.get("Context"))
    fib["Context"] = ctx

    # Convert window scalars to dict packs for compatibility.
    sw = fib.get("ShortWindow")
    if not isinstance(sw, dict):
        band = ctx.get("ShortBand") or ctx.get("ShortWindowBand") or "N/A"
        val = sw
        try:
            # keep int-like values neat
            if isinstance(sw, float) and (np.isnan(sw) or np.isinf(sw)):
                val = sw
            elif isinstance(sw, (int, float)):
                val = int(sw)
        except Exception:
            pass
        fib["ShortWindow"] = {"Value": val, "Band": band}
    else:
        swd = as_dict(sw)
        if "Value" not in swd and "Window" in swd:
            swd["Value"] = swd.get("Window")
        if "Band" not in swd:
            swd["Band"] = ctx.get("ShortBand") or "N/A"
        fib["ShortWindow"] = swd

    lw = fib.get("LongWindow")
    if not isinstance(lw, dict):
        band = ctx.get("LongBand") or ctx.get("LongWindowBand") or "N/A"
        val = lw
        try:
            if isinstance(lw, float) and (np.isnan(lw) or np.isinf(lw)):
                val = lw
            elif isinstance(lw, (int, float)):
                val = int(lw)
        except Exception:
            pass
        fib["LongWindow"] = {"Value": val, "Band": band}
    else:
        lwd = as_dict(lw)
        if "Value" not in lwd and "Window" in lwd:
            lwd["Value"] = lwd.get("Window")
        if "Band" not in lwd:
            lwd["Band"] = ctx.get("LongBand") or "N/A"
        fib["LongWindow"] = lwd

    # Ensure nested structures exist as dicts where used.
    fib["Short"] = as_dict(fib.get("Short"))
    fib["Long"] = as_dict(fib.get("Long"))
    fib["AltShort"] = as_dict(fib.get("AltShort"))

    return fib


def normalize_protech_pack(pro_in: Any) -> Dict[str, Any]:
    pro = as_dict(pro_in)
    # Expected sub-packs used widely in UI/reporting
    for k in ("MA", "RSI", "MACD", "Volume", "Bias", "PriceAction", "RSIContext", "VolumeContext", "LevelContext"):
        pro[k] = as_dict(pro.get(k))
    return pro


def normalize_rrsim_pack(rrsim_in: Any) -> Dict[str, Any]:
    r = as_dict(rrsim_in)
    r["Setups"] = as_list(r.get("Setups"))
    return r


def normalize_analysis_pack(ap_in: Any) -> Dict[str, Any]:
    """Normalize the main AnalysisPack to a stable dict contract."""
    ap = as_dict(ap_in)

    ap["Last"] = as_dict(ap.get("Last"))
    ap["Market"] = as_dict(ap.get("Market"))

    m = ap["Market"]
    m["VNINDEX"] = as_dict(m.get("VNINDEX"))
    m["VN30"] = as_dict(m.get("VN30"))
    ap["Market"] = m

    ap["Fibonacci"] = normalize_fibonacci_pack(ap.get("Fibonacci"))
    ap["ProTech"] = normalize_protech_pack(ap.get("ProTech"))

    ap["Scenario12"] = as_dict(ap.get("Scenario12"))
    ap["MasterScore"] = as_dict(ap.get("MasterScore"))
    ap["Fundamental"] = as_dict(ap.get("Fundamental"))
    ap["StructureQuality"] = as_dict(ap.get("StructureQuality"))

    ap["RRSim"] = normalize_rrsim_pack(ap.get("RRSim"))

    # TradePlans: keep as list-of-dicts
    plans = ap.get("TradePlans")
    if isinstance(plans, dict):
        plans = list(plans.values())
    plans_list = as_list(plans)
    ap["TradePlans"] = [as_dict(p) for p in plans_list]

    ap["PrimarySetup"] = as_dict(ap.get("PrimarySetup"))

    # PositionStatePack: normalize common key drift (TitleCase -> snake_case)
    ap["PositionStatePack"] = normalize_position_state_pack(ap.get("PositionStatePack"))

    return ap


def normalize_position_state_pack(pack_in: Any) -> Dict[str, Any]:
    """Normalize PositionStatePack to a stable snake_case contract.

    Historical drift notes
    - Older pipeline injectors used TitleCase keys (e.g., "Mode", "IsHolding").
    - Core modules and dashboard expect snake_case (e.g., "mode", "is_holding").

    We keep both key styles for backward compatibility, but always populate snake_case.
    """
    p = as_dict(pack_in)

    # Map TitleCase -> snake_case when needed
    mapping = {
        "Mode": "mode",
        "IsHolding": "is_holding",
        "Timeframe": "timeframe",
        "HoldingHorizon": "holding_horizon",
        "AvgCost": "avg_cost",
        "CurrentPrice": "current_price",
        "PnlPct": "pnl_pct",
        "InProfit": "in_profit",
        "PositionSizePctNAV": "position_size_pct_nav",
        "RiskBudgetPctNAV": "risk_budget_pct_nav",
    }
    for src, dst in mapping.items():
        if dst not in p and src in p:
            p[dst] = p.get(src)

    # Canonicalize a few fields
    mode = as_str(p.get("mode") or "FLAT", default="FLAT").strip().upper()
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    p["mode"] = mode
    p["is_holding"] = bool(p.get("is_holding")) if mode == "HOLDING" else False
    p["timeframe"] = as_str(p.get("timeframe") or "D", default="D").strip().upper()
    p["holding_horizon"] = as_str(p.get("holding_horizon") or "SWING", default="SWING").strip().upper()

    # Numeric fields
    for k in ("avg_cost", "current_price", "pnl_pct", "position_size_pct_nav", "risk_budget_pct_nav"):
        if k in p:
            p[k] = as_float(p.get(k), default=np.nan)

    # Boolean convenience
    if "in_profit" in p:
        try:
            p["in_profit"] = bool(p.get("in_profit"))
        except Exception:
            p["in_profit"] = False



    # Distress overlay (HOLDING risk sanity)
    # This does not change trading logic; it only provides stable labels for downstream gating.
    pnl = as_float(p.get("pnl_pct"), default=np.nan)
    distress_level = "-"
    if np.isfinite(pnl):
        if pnl <= -25.0:
            distress_level = "SEVERE"
        elif pnl <= -15.0:
            distress_level = "MEDIUM"
        elif pnl <= -7.0:
            distress_level = "MILD"
        else:
            distress_level = "OK"
    p["distress_level"] = distress_level
    p["underwater_pct"] = pnl if np.isfinite(pnl) else np.nan
    p["avg_cost_gap_pct"] = abs(pnl) if np.isfinite(pnl) else np.nan

    # Legacy aliases
    if "DistressLevel" not in p:
        p["DistressLevel"] = p.get("distress_level")
    if "UnderwaterPct" not in p:
        p["UnderwaterPct"] = p.get("underwater_pct")
    if "AvgCostGapPct" not in p:
        p["AvgCostGapPct"] = p.get("avg_cost_gap_pct")
    # Re-populate legacy TitleCase to avoid breaking any downstream renderers.
    for src, dst in mapping.items():
        if src not in p and dst in p:
            p[src] = p.get(dst)

    return p


def normalize_character_pack(pack_in: Any) -> Dict[str, Any]:
    p = as_dict(pack_in)
    for k in ("CoreStats", "CombatStats", "Traits", "Risks", "Notes", "Meta"):
        p[k] = as_dict(p.get(k))

    # Some older renderers treat Conviction as dict; keep scalar too.
    p["Conviction"] = p.get("Conviction")

    # Ensure ClassName exists (for DataQuality/Dashboard)
    if not p.get("ClassName"):
        class_name = p.get("CharacterClass")

        c = p.get("Class")
        if not class_name and isinstance(c, dict):
            class_name = c.get("Name") or c.get("ClassName")

        dna = p.get("DNA")
        if not class_name and isinstance(dna, dict):
            class_name = dna.get("ClassName") or dna.get("Name")

        sdna = p.get("StockDNA")
        if not class_name and isinstance(sdna, dict):
            class_name = sdna.get("ClassName") or sdna.get("Name")

        p["ClassName"] = class_name or "UNKNOWN"

    # Ensure a stable nested Class object for downstream consumers
    if not isinstance(p.get("Class"), dict):
        p["Class"] = {"Name": p.get("ClassName") or "UNKNOWN"}
    else:
        p["Class"].setdefault("Name", p.get("ClassName") or "UNKNOWN")

    return p


def normalize_tradeplan_pack(pack_in: Any) -> Dict[str, Any]:
    p = as_dict(pack_in)
    p["Plans"] = as_list(p.get("Plans"))
    return p


def normalize_decision_pack(pack_in: Any) -> Dict[str, Any]:
    p = as_dict(pack_in)
    p["Flags"] = as_dict(p.get("Flags"))
    p["Weakness"] = as_list(p.get("Weakness"))
    return p


def normalize_position_manager_pack(pack_in: Any) -> Dict[str, Any]:
    p = as_dict(pack_in)
    p["Levels"] = as_dict(p.get("Levels"))
    p["Notes"] = as_list(p.get("Notes"))
    return p
