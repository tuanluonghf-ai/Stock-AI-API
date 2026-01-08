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

    return ap


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
