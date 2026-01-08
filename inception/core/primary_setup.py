"""Primary setup selection (v3).

Purpose
-------
Pick a single "PrimarySetup" from RRSim.Setups with a stable contract.

Why a dedicated module
----------------------
- Keeps selection logic out of app.py to reduce churn.
- TradePlanPack depends on Entry/Stop/TP fields; earlier versions only exposed Risk/Reward.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .helpers import _safe_float


def pick_primary_setup_v3(rrsim: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe PrimarySetup dict.

    Ranking policy (stable + simple):
    1) Prefer Status=Active over Watch over others.
    2) Within same status, prefer higher RR.

    Output contract includes fields used by TradePlanPack:
    Name, Status, Entry, Stop, TP1, TP2, RiskPct, RewardPct, RR, Confidence (Tech), ReasonTags.
    """

    r = rrsim or {}
    setups = r.get("Setups", []) or []
    if not isinstance(setups, list):
        setups = []

    if not setups:
        return {
            "Name": "N/A",
            "Status": "N/A",
            "Entry": np.nan,
            "Stop": np.nan,
            "TP1": np.nan,
            "TP2": np.nan,
            "RiskPct": np.nan,
            "RewardPct": np.nan,
            "RR": np.nan,
            "Confidence (Tech)": "N/A",
            "ReasonTags": [],
        }

    def status_rank(s: Dict[str, Any]) -> int:
        stt = str((s.get("Status") or "Watch")).strip().lower()
        if stt == "active":
            return 0
        if stt == "watch":
            return 1
        return 2

    best = None
    for s in setups:
        if not isinstance(s, dict):
            continue
        rr = _safe_float(s.get("RR"))
        if pd.isna(rr):
            continue
        if best is None:
            best = s
            continue
        if status_rank(s) < status_rank(best):
            best = s
            continue
        if status_rank(s) > status_rank(best):
            continue
        if rr > _safe_float(best.get("RR")):
            best = s

    best = best or (setups[0] if isinstance(setups[0], dict) else {})

    name = best.get("Setup") or best.get("Name") or "N/A"
    status = best.get("Status") or "Watch"

    entry = _safe_float(best.get("Entry"))
    stop = _safe_float(best.get("Stop"))
    tp1 = _safe_float(best.get("TP"))
    tp2 = np.nan

    # Best-effort TP2: extension 1.6x from TP1
    try:
        if pd.notna(tp1) and pd.notna(entry):
            tp2 = float(entry) + 1.6 * (float(tp1) - float(entry))
    except Exception:
        tp2 = np.nan

    tags = best.get("ReasonTags") or []
    if not isinstance(tags, list):
        tags = []

    return {
        "Name": name,
        "Status": status,
        "Entry": entry,
        "Stop": stop,
        "TP1": tp1,
        "TP2": tp2,
        "RiskPct": _safe_float(best.get("RiskPct")),
        "RewardPct": _safe_float(best.get("RewardPct")),
        "RR": _safe_float(best.get("RR")),
        "Confidence (Tech)": best.get("Confidence (Tech)", best.get("Probability", "N/A")),
        "ReasonTags": tags,
    }
