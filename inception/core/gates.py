"""Core gating utilities.

Centralizes gate logic used by both core pack builders and UI renderers.

Trade plan gate status:
- ACTIVE: high conviction
- WATCH: medium conviction
- LOCK: low conviction

The intent is to prevent anchoring into a detailed trade plan when conviction is
not high enough.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .helpers import _safe_float


def trade_plan_gate(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Anti-anchoring gate for Trade Plan.

    Returns:
        (status, meta) where status ∈ {"ACTIVE","WATCH","LOCK"}

    Uses the same thresholds as the UI had historically (Appendix E):
    - ACTIVE if Conviction >= 6.5 OR character conviction tier >= 4
    - WATCH  if 5.5 <= Conviction < 6.5 OR character conviction tier == 3
    - else LOCK
    """
    ap = analysis_pack or {}
    cp = character_pack or {}

    score = _safe_float(ap.get("Conviction"), default=np.nan)
    tier = None
    try:
        tier = (cp.get("Conviction") or {}).get("Tier", None)
    except Exception:
        tier = None

    active = (pd.notna(score) and score >= 6.5) or (isinstance(tier, (int, float)) and float(tier) >= 4)
    watch = (pd.notna(score) and 5.5 <= score < 6.5) or (isinstance(tier, (int, float)) and int(tier) == 3)

    if active:
        status = "ACTIVE"
    elif watch:
        status = "WATCH"
    else:
        status = "LOCK"

    return status, {"ConvictionScore": score, "Tier": tier}
