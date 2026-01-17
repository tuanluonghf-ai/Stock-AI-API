from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LookbackCfg:
    id: str
    window: int


def _cfg_id(cfg: Any) -> str:
    if isinstance(cfg, dict):
        return str(cfg.get("id") or "N/A")
    return str(getattr(cfg, "id", "N/A"))


def _cfg_window(cfg: Any, default: int) -> int:
    if isinstance(cfg, dict):
        w = cfg.get("window")
    else:
        w = getattr(cfg, "window", None)
    try:
        w_int = int(w)
        return w_int if w_int > 0 else default
    except Exception:
        return default


def compute_payoff_tier_v1(
    df: pd.DataFrame,
    reference_price: float,
    lookback_cfg: Any,
) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        payoff_span_norm = 0.0
    else:
        window = _cfg_window(lookback_cfg, default=len(df))
        win = df.tail(min(window, len(df)))
        hi = pd.to_numeric(win.get("High"), errors="coerce").max()
        lo = pd.to_numeric(win.get("Low"), errors="coerce").min()
        try:
            ref = float(reference_price)
        except Exception:
            ref = np.nan
        if not np.isfinite(hi) or not np.isfinite(lo) or not np.isfinite(ref) or ref == 0:
            payoff_span_norm = 0.0
        else:
            payoff_span_norm = (float(hi) - float(lo)) / ref

    payoff_span_norm = float(round(payoff_span_norm, 6))
    if payoff_span_norm < 0.15:
        tier = "LOW"
    elif payoff_span_norm < 0.40:
        tier = "MEDIUM"
    else:
        tier = "HIGH"

    return {
        "payoff_span_norm": payoff_span_norm,
        "payoff_tier": tier,
        "method": "swing_range_norm_v1",
        "lookback": _cfg_id(lookback_cfg),
        "notes": "economic viability only; no probability",
    }
