"""Character / Stock DNA pack builder (core).

This module centralizes CharacterPack generation so app_INCEPTION stays thin.
No Streamlit dependency.

v1 contract:
- compute_character_pack_v1(df, analysis_pack) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from inception.core.helpers import (
    _as_scalar,
    _coalesce,
    _safe_bool,
    _safe_float,
    _safe_text,
)

from inception.core.tradeplan_pack import compute_trade_plan_pack_v1
from inception.core.decision_pack import compute_decision_pack_v1
from inception.core.position_manager_pack import compute_position_manager_pack_v1
from inception.core.status_pack import compute_status_pack_v1


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR (EWMA alpha=1/period)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)
    high = pd.to_numeric(df["High"], errors="coerce").astype(float)
    low = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def _dynamic_vol_proxy(df: pd.DataFrame, lookback: int = 20) -> float:
    """Fallback volatility proxy when ATR is unavailable (data-driven, no fixed %)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df.columns:
        return np.nan
    d = df.tail(max(int(lookback), 10)).copy()
    if d.empty:
        return np.nan
    cd = pd.to_numeric(d["Close"], errors="coerce").astype(float).diff().abs().dropna()
    rng = (
        (pd.to_numeric(d["High"], errors="coerce") - pd.to_numeric(d["Low"], errors="coerce")).abs().dropna()
        if ("High" in d.columns and "Low" in d.columns)
        else pd.Series(dtype=float)
    )
    m1 = _safe_float(cd.median()) if not cd.empty else np.nan
    m2 = _safe_float(rng.median()) if not rng.empty else np.nan
    if pd.notna(m1) and pd.notna(m2):
        return float(m1 + 0.25 * m2)
    if pd.notna(m1):
        return float(m1)
    if pd.notna(m2):
        return float(0.25 * m2)
    return np.nan


def _clip(x: float, lo: float, hi: float) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return np.nan


def _bucket_score(x: float, bins: list, scores: list) -> float:
    if pd.isna(x):
        return np.nan
    for i, b in enumerate(bins):
        if x < b:
            return float(scores[i])
    return float(scores[-1])


def _avg(*xs: float) -> float:
    vals = [x for x in xs if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan


def _atr_last(df: pd.DataFrame, period: int = 14) -> float:
    a = atr_wilder(df, period)
    if a is None or getattr(a, "empty", True):
        return np.nan
    s = a.dropna()
    return _safe_float(s.iloc[-1]) if not s.empty else np.nan


def _candle_strength_from_class(candle_class: str) -> float:
    c = (candle_class or "").strip().lower()
    if c in ("strong_bull", "bull_engulf", "bull_engulfing", "marubozu_bull"):
        return 1.0
    if c in ("bull", "hammer", "bullish_hammer"):
        return 0.7
    if c in ("doji_high_vol", "shooting_star", "gravestone", "spinning_top"):
        return 0.35
    if c in ("strong_bear", "bear_engulf", "bear_engulfing", "marubozu_bear"):
        return 0.0
    if c in ("bear",):
        return 0.2
    return 0.5


def _derive_liquidity_score(vol_ratio: float, liquidity_base: float) -> float:
    if pd.notna(liquidity_base):
        return _clip(liquidity_base, 0, 10)
    vr = _safe_float(vol_ratio)
    if pd.isna(vr):
        return 5.0
    return _bucket_score(vr, bins=[0.6, 0.9, 1.2, 1.6, 2.2], scores=[3, 4.5, 6, 7.5, 8.5, 9.5])


def _pretty_level_label(level_type: str, side: str = "overhead") -> str:
    """Human-friendly label for structure-quality level components."""
    t = (level_type or "").strip().upper()
    if not t:
        return ""
    mapping = {
        "MA200": "MA200",
        "MA50": "MA50",
        "MA20": "MA20",
        "FIB": "Fibonacci",
        "FIBO": "Fibonacci",
        "SR": "S/R",
        "S/R": "S/R",
        "CHANNEL": "Channel",
        "TRENDLINE": "Trendline",
    }
    return mapping.get(t, t)


# ============================================================
# Style Hybrid Scoring (Tier-1 StyleAxis) — v15.1
# - Hard Filters (ILLIQ/GAP) keep NO tilt/runner-up for Style.
# - For "clean" stocks, compute Trend/Momentum/Range scores and expose tilt when truly near-boundary.
# ============================================================

_STYLE_HYBRID_CFG_V1: Dict[str, float] = {
    "PRIMARY_MIN_SCORE": 6.0,     # below this => StyleAxis = "Hybrid"
    "RUNNERUP_MIN_SCORE": 6.0,    # below this => no tilt
    "NEAR_ABS_MAX": 0.80,         # absolute score gap max
    "NEAR_REL_PCT": 0.10,         # relative gap max (10% of score1)
}


def _style_tilt_strength(gap: float, near_thr: float) -> str:
    try:
        g = float(gap)
        if not np.isfinite(g):
            return "-"
        if g <= 0.35:
            return "Strong"
        if g <= 0.65:
            return "Medium"
        if g <= float(near_thr):
            return "Weak"
        return "-"
    except Exception:
        return "-"


def _compute_style_hybrid_v1(
    *,
    trend_integrity: float,
    breakout_quality: float,
    momentum_adj: float,
    meanrev_prop: float,
    whipsaw: bool,
    autocorr1: Optional[float],
    enabled: bool = True,
    disabled_reason: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Return (StyleAxis, StyleHybridPack).

    StyleAxis is one of {"Trend","Momentum","Range","Hybrid"}.

    Notes:
    - Deterministic; long-run oriented.
    - Tilt is only exposed when near-boundary AND runner-up is strong enough.
    """
    ti = float(trend_integrity) if np.isfinite(trend_integrity) else 5.0
    bq = float(breakout_quality) if np.isfinite(breakout_quality) else 5.0
    ma = float(momentum_adj) if np.isfinite(momentum_adj) else 5.0
    mr = float(meanrev_prop) if np.isfinite(meanrev_prop) else 5.0

    ac1 = float(autocorr1) if (autocorr1 is not None and np.isfinite(autocorr1)) else np.nan

    # Base scores (0–10)
    s_trend = _clip(ti, 0, 10)
    s_momo = _clip(0.70 * bq + 0.30 * ma, 0, 10)
    s_range = _clip(mr, 0, 10)

    # Gentle nudges (keep simple, preserve known structure signals)
    if np.isfinite(ac1) and ac1 >= -0.02:
        s_trend = _clip(s_trend + 0.35, 0, 10)
    if mr <= 5.7:
        s_trend = _clip(s_trend + 0.25, 0, 10)

    if mr <= 6.5:
        s_momo = _clip(s_momo + 0.20, 0, 10)

    if bool(whipsaw) or (np.isfinite(ac1) and ac1 <= -0.05):
        s_range = _clip(s_range + 0.35, 0, 10)

    scores = {"Trend": float(s_trend), "Momentum": float(s_momo), "Range": float(s_range)}
    ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)

    (k1, v1) = ranked[0]
    (k2, v2) = ranked[1]
    (k3, v3) = ranked[2]

    primary_min = float(_STYLE_HYBRID_CFG_V1["PRIMARY_MIN_SCORE"])
    runner_min = float(_STYLE_HYBRID_CFG_V1["RUNNERUP_MIN_SCORE"])
    near_thr = float(max(_STYLE_HYBRID_CFG_V1["NEAR_ABS_MAX"], _STYLE_HYBRID_CFG_V1["NEAR_REL_PCT"] * v1))
    gap12 = float(v1 - v2)

    # Primary style
    if v1 < primary_min:
        style_axis = "Hybrid"
    else:
        style_axis = str(k1)

    # Tilt only when enabled + primary is a concrete style
    near = bool((enabled is True) and (style_axis in ("Trend", "Momentum", "Range")) and (v2 >= runner_min) and (gap12 <= near_thr))
    tilt_to = str(k2) if near else "-"
    tilt_strength = _style_tilt_strength(gap12, near_thr) if near else "-"

    style_pack: Dict[str, Any] = {
        "Version": "StyleHybrid_v1.0",
        "Enabled": bool(enabled),
        "DisabledReason": str(disabled_reason or "") if not enabled else "",
        "Scores": scores,
        "Primary": str(style_axis),
        "TiltTo": tilt_to,
        "TiltStrength": tilt_strength,
        "NearBoundary": bool(near),
        # Diagnostics (keep; UI can ignore)
        "ScoreGap12": float(gap12),
        "NearThreshold": float(near_thr),
        "RunnerUp2": {"Style": str(k2), "Score": float(v2)},
        "RunnerUp3": {"Style": str(k3), "Score": float(v3)},
    }
    return str(style_axis), style_pack


def compute_character_pack_v1(df: pd.DataFrame, analysis_pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces CharacterPack without breaking any existing keys.
    Uses ONLY already-computed features inside AnalysisPack + df series.
    """
    # Defensive normalization: avoid truthiness checks on pandas objects (Series/DataFrame)
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    # Sanitize AnalysisPack values to avoid pandas objects leaking into boolean/math operations
    def _sanitize_obj(obj: Any):
        if isinstance(obj, (pd.Series, pd.Index, pd.DataFrame)):
            return _as_scalar(obj)
        if isinstance(obj, dict):
            return {str(k): _sanitize_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_obj(v) for v in obj]
        return obj

    ap = _sanitize_obj(ap)


    last = ap.get("Last", {})
    last = last if isinstance(last, dict) else {}

    protech = ap.get("ProTech", {})
    protech = protech if isinstance(protech, dict) else {}

    ma = protech.get("MA", {})
    ma = ma if isinstance(ma, dict) else {}
    rsi = protech.get("RSI", {})
    rsi = rsi if isinstance(rsi, dict) else {}
    macd = protech.get("MACD", {})
    macd = macd if isinstance(macd, dict) else {}
    vol = protech.get("Volume", {})
    vol = vol if isinstance(vol, dict) else {}
    pa = protech.get("PriceAction", {})
    pa = pa if isinstance(pa, dict) else {}
    # Pre-read close/MA levels from Last pack (needed for LevelContext fallback inference)
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    # ---- Level Context (Support/Resistance distances) ----
    # Depending on pipeline version, LevelContext may live in different locations.
    # We MUST NOT accept an "empty" dict (or a dict with only null values), otherwise UpsidePower becomes N/A.
    def _lvl_num(v: Any) -> float:
        if v is None:
            return np.nan
        if isinstance(v, dict):
            v = v.get("Value")
        return _safe_float(v)

    def _lvl_strength(d: Any) -> int:
        if not isinstance(d, dict) or len(d) == 0:
            return 0
        # count how many expected fields have finite numeric content
        keys = [
            "NearestResistance", "NearestSupport",
            "UpsideToResistance", "DownsideToSupport",
            # common alternates
            "Upside", "Downside",
            "Resistance", "Support",
        ]
        score = 0
        for k in keys:
            if k in d:
                x = _lvl_num(d.get(k))
                if pd.notna(x):
                    score += 1
        return score

    lvl_source = "None"
    lvl: Dict[str, Any] = {}

    _candidates = [
        ("ProTech.LevelContext", protech.get("LevelContext")),
        ("Top.LevelContext", ap.get("LevelContext")),
        ("ProTech.Levels", protech.get("Levels")),
        ("Top.Levels", ap.get("Levels")),
    ]

    best_score = 0
    best_name = "None"
    best_cand = None
    for _name, _cand in _candidates:
        s = _lvl_strength(_cand)
        if s > best_score:
            best_score, best_name, best_cand = s, _name, _cand

    if best_score > 0 and isinstance(best_cand, dict):
        lvl = best_cand
        lvl_source = best_name

    # Final safety net: infer nearest S/R locally (MA levels + recent swing high/low)
    # This guarantees Upside/Downside when upstream packs are absent.
    if not isinstance(lvl, dict) or len(lvl) == 0:
        def _infer_nearest_sr(_df: pd.DataFrame, _close: float,
                              _ma20: float, _ma50: float, _ma200: float,
                              lookback: int = 60) -> Tuple[float, float]:
            if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty or pd.isna(_close):
                return (np.nan, np.nan)

            # Candidates from MA levels (if available)
            ma_vals = [x for x in [_ma20, _ma50, _ma200] if pd.notna(x)]
            res_cands = [x for x in ma_vals if x > _close]
            sup_cands = [x for x in ma_vals if x < _close]

            # Candidates from recent swing extremes (robust fallback)
            try:
                if "High" in _df.columns:
                    hh = float(pd.to_numeric(_df["High"], errors="coerce").tail(lookback).max())
                    # If price is at/near a recent high, treat it as the nearest "resistance" (upside room ~ 0).
                    # This avoids UpsidePower=N/A when the stock is making new highs within lookback.
                    if pd.notna(hh) and hh >= _close:
                        res_cands.append(hh)
                if "Low" in _df.columns:
                    ll = float(pd.to_numeric(_df["Low"], errors="coerce").tail(lookback).min())
                    if pd.notna(ll) and ll < _close:
                        sup_cands.append(ll)
            except Exception:
                pass

            nearest_res = min(res_cands) if res_cands else np.nan
            nearest_sup = max(sup_cands) if sup_cands else np.nan
            return (nearest_res, nearest_sup)

        _nr, _ns = _infer_nearest_sr(df, close, ma20, ma50, ma200, lookback=60)
        if pd.notna(_nr) or pd.notna(_ns):
            lvl = {
                "NearestResistance": {"Value": _nr} if pd.notna(_nr) else None,
                "NearestSupport": {"Value": _ns} if pd.notna(_ns) else None,
                "UpsideToResistance": (max(0.0, _nr - close) if (pd.notna(_nr) and pd.notna(close)) else np.nan),
                "DownsideToSupport": (max(0.0, close - _ns) if (pd.notna(_ns) and pd.notna(close)) else np.nan),
            }
            lvl_source = "Local.Infer"
    fib_ctx = ap.get("FibonacciContext", {})
    fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}
    # prefer nested AnalysisPack["Fibonacci"]["Context"] if available
    if not fib_ctx:
        fib = ap.get("Fibonacci", {})
        if isinstance(fib, dict):
            ctx = fib.get("Context", {})
            if isinstance(ctx, dict):
                fib_ctx = ctx

    # --------------------------
    # STRUCTURE QUALITY
    # Prefer AnalysisPack['StructureQuality'] as single source of truth.
    # If missing, leave empty (no cross-module fallback from app).
    # --------------------------
    struct_q = ap.get("StructureQuality", {})
    struct_q = struct_q if isinstance(struct_q, dict) else {}


    primary = ap.get("PrimarySetup", {})
    primary = primary if isinstance(primary, dict) else {}
    rrsim = ap.get("RRSim", {})
    rrsim = rrsim if isinstance(rrsim, dict) else {}

    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    def _safe_label(obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            v = obj.get("Label")
            return str(v) if v is not None else None
        # allow plain strings like "Positive"/"Negative"/"Flat"
        if isinstance(obj, (str, int, float)):
            s = str(obj).strip()
            return s if s else None
        return None
    s20 = _coalesce(_safe_label(ma.get("SlopeMA20")), ma.get("SlopeMA20Label"))
    s50 = _coalesce(_safe_label(ma.get("SlopeMA50")), ma.get("SlopeMA50Label"))
    s200 = _coalesce(_safe_label(ma.get("SlopeMA200")), ma.get("SlopeMA200Label"))
    structure = _coalesce(ma.get("Structure"), ma.get("StructureSnapshot"), ma.get("StructureSnapshotV2"))

    rsi14 = _safe_float(rsi.get("RSI"))
    rsi_state = _safe_text(_coalesce(rsi.get("State"), rsi.get("Zone"))).strip()
    rsi_div = _safe_text(rsi.get("Divergence")).strip().lower()

    macd_state = _safe_text(macd.get("State")).strip().lower()
    macd_zero = _safe_text(macd.get("ZeroLine")).strip().lower()
    hist_state = _safe_text(macd.get("HistState")).strip().lower()
    macd_div = _safe_text(macd.get("Divergence")).strip().lower()

    vol_ratio = _safe_float(vol.get("Ratio"))
    vol_regime = _safe_text(vol.get("Regime")).strip().lower()

    candle_class = _safe_text(_coalesce(pa.get("CandleClass"), pa.get("Candle"), "")).strip()
    climax = _safe_bool(_coalesce(pa.get("ClimaxFlag"), pa.get("ClimacticFlag")))
    gap = _safe_bool(_coalesce(pa.get("GapFlag"), pa.get("Gap")))
    atr = _atr_last(df, 14)
    atr = _as_scalar(atr)
    vol_proxy = _safe_float(ap.get("VolProxy")) if pd.notna(_safe_float(ap.get("VolProxy"))) else _dynamic_vol_proxy(df, 20)

    # Levels / distances (prefer LevelContext / FibonacciContext packs)
    ns = lvl.get("NearestSupport"); nearest_sup = (ns.get("Value") if isinstance(ns, dict) else _as_scalar(ns))
    nr = lvl.get("NearestResistance"); nearest_res = (nr.get("Value") if isinstance(nr, dict) else _as_scalar(nr))
    nearest_sup = _safe_float(nearest_sup)
    nearest_res = _safe_float(nearest_res)

    upside = _safe_float(lvl.get("UpsideToResistance"))
    downside = _safe_float(lvl.get("DownsideToSupport"))
    if pd.isna(upside) and pd.notna(nearest_res) and pd.notna(close):
        upside = max(0.0, nearest_res - close)
    if pd.isna(downside) and pd.notna(nearest_sup) and pd.notna(close):
        downside = max(0.0, close - nearest_sup)

    denom = atr if pd.notna(atr) and atr > 0 else (vol_proxy if pd.notna(vol_proxy) and vol_proxy > 0 else np.nan)
    upside_n = upside / denom if pd.notna(denom) and denom > 0 else np.nan
    downside_n = downside / denom if pd.notna(denom) and denom > 0 else np.nan
    rr = (upside / downside) if (pd.notna(upside) and pd.notna(downside) and downside > 0) else _safe_float(primary.get("RR"))

    # If we have a valid downside but no resistance/upside, infer an upside target from RR.
    # This prevents UpsidePower=N/A in "open sky" cases (new highs / no nearby resistance).
    # Conservative approach: only infer when RR is explicitly available and downside>0.
    if pd.isna(upside) and pd.notna(rr) and pd.notna(downside) and downside > 0 and pd.notna(close):
        try:
            upside = float(rr) * float(downside)
            if pd.isna(nearest_res):
                nearest_res = float(close) + float(upside)
            if isinstance(lvl, dict):
                lvl["NearestResistance"] = {"Value": float(nearest_res)}
                lvl["UpsideToResistance"] = float(upside)
                # keep existing support if present
            # recompute normalized upside
            if pd.notna(denom) and denom > 0:
                upside_n = upside / denom
        except Exception:
            pass
    fib_conflict = False
    try:
        fc = fib_ctx if isinstance(fib_ctx, dict) else {}
        fib_conflict = _safe_bool(fc.get("FiboConflictFlag"))
    except Exception:
        fib_conflict = False

    confluence_count = np.nan
    try:
        fc = fib_ctx if isinstance(fib_ctx, dict) else {}
        confluence_count = _safe_float(fc.get("ConfluenceCount"))
    except Exception:
        confluence_count = np.nan

    if pd.isna(confluence_count):
        # robust fallback: infer from any iterable hits inside Confluence*WithMA
        try:
            fc = fib_ctx if isinstance(fib_ctx, dict) else {}
            conf_short = fc.get("ConfluenceShortWithMA")
            conf_long = fc.get("ConfluenceLongWithMA")

            def _count_hits(obj):
                # obj can be dict/list/str/number
                if obj is None:
                    return 0
                if isinstance(obj, dict):
                    return sum(_count_hits(v) for v in obj.values())
                if isinstance(obj, (list, tuple, set)):
                    return len(obj)
                # if it's a scalar/str -> treat as 1 hit only if non-empty string
                if isinstance(obj, str):
                    return 1 if obj.strip() else 0
                return 1

            confluence_count = float(_count_hits(conf_short) + _count_hits(conf_long))
            confluence_count = min(confluence_count, 5.0) if pd.notna(confluence_count) else np.nan
        except Exception:
            confluence_count = np.nan

    # --------------------------
    # CORE STATS (0–10)
    # --------------------------
    # Trend: 4 components
    t1 = 2.5 if (pd.notna(close) and pd.notna(ma200) and close >= ma200) else 0.5
    t2 = 2.5 if (pd.notna(ma20) and pd.notna(ma50) and ma20 >= ma50) else 0.5
    t3 = 2.5 if (str(s50).lower() == "positive" and str(s200).lower() != "negative") else (1.25 if str(s50).lower() == "positive" else 0.5)
    cross_obj = ma.get("Cross", {})
    cross_event = ""
    if isinstance(cross_obj, dict):
        cross_ma = cross_obj.get("MA50VsMA200")
        if isinstance(cross_ma, dict):
            cross_event = str(_as_scalar(cross_ma.get("Event")) if cross_ma.get("Event") is not None else "")
        else:
            cross_event = str(_as_scalar(cross_ma) if cross_ma is not None else "")
        if not cross_event:
            cross_event = str(_coalesce(cross_obj.get("Event"), cross_obj.get("Name"), ""))
    else:
        cross_event = str(_as_scalar(cross_obj) if cross_obj is not None else "")
    cross_l = cross_event.strip().lower()
    # CrossUp = golden cross, CrossDown = death cross
    t4 = 2.5 if ("crossup" in cross_l or "golden" in cross_l) else (0.5 if ("crossdown" in cross_l or "death" in cross_l) else 1.25)
    trend = _clip(_avg(t1, t2, t3, t4) * 4.0, 0, 10)  # NOTE: sum of 4 sub-scores (max 10)

    # Momentum: RSI + MACD + Hist + candle
    # RSI best zone: ~60–70 (bullish but not overheated)
    if pd.isna(rsi14):
        m1 = 1.25
    else:
        if 60 <= rsi14 <= 70: m1 = 2.5
        elif 55 <= rsi14 < 60: m1 = 2.0
        elif 70 < rsi14 <= 78: m1 = 1.8
        elif 45 <= rsi14 < 55: m1 = 1.25
        else: m1 = 0.8
    m2 = 2.5 if ("bull" in macd_state and "above" in macd_zero) else (1.8 if "bull" in macd_state else 0.8)
    m3 = 2.5 if "expanding" in hist_state else (1.6 if "flat" in hist_state or "neutral" in hist_state else 1.0)
    m4 = 2.5 * _candle_strength_from_class(candle_class)
    momentum = _clip(_avg(m1, m2, m3, m4) * 4.0, 0, 10)  # NOTE: sum of 4 sub-scores (max 10)

    # Stability: inverse volatility + whipsaw penalty + shock penalty
    # Use denom as proxy for ATR; higher denom relative to price => more volatile.
    if pd.notna(close) and close > 0 and pd.notna(denom):
        vol_pct = denom / close * 100
        s1 = _bucket_score(vol_pct, bins=[1.2, 2.0, 3.0, 4.5], scores=[9.0, 8.0, 6.5, 5.0, 3.5])
    else:
        s1 = 5.5
    structure_l = _safe_text(structure).strip().lower()
    whipsaw = ("mixed" in structure_l) or ("side" in structure_l)
    s2 = 7.5 if not whipsaw else 4.5
    s3 = 7.0 if (str(s50).lower() != "flat") else 5.0
    s4_pen = 1.5 if (_safe_bool(climax) or _safe_bool(gap)) else 0.0
    stability = _clip(_avg(s1, s2, s3) - s4_pen, 0, 10)

    # Reliability: alignment + volume confirm - divergence - whipsaw
    align = 1.0 if (pd.notna(close) and pd.notna(ma50) and pd.notna(ma200) and ((close >= ma50 >= ma200) or (close < ma50 < ma200))) else 0.5
    r1 = 2.5 if align >= 1.0 else 1.25
    r2 = _bucket_score(vol_ratio, bins=[0.8, 1.0, 1.3, 1.8], scores=[0.8, 1.4, 2.0, 2.3, 2.5])
    div_pen = 2.0 if ("bear" in rsi_div or "bear" in macd_div) else 0.0
    r4_pen = 1.5 if whipsaw else 0.0
    reliability = _clip(_avg(r1, r2, 2.0) * 3.0 - div_pen - r4_pen, 0, 10)  # NOTE: sum of 3 sub-scores

    # Liquidity: base if exists, else from vol_ratio
    liquidity_base = _safe_float(ap.get("LiquidityScoreBase"))
    liquidity = _derive_liquidity_score(vol_ratio, liquidity_base)

    core_stats = {
        "Trend": float(trend),
        "Momentum": float(momentum),
        "Stability": float(stability),
        "Reliability": float(reliability),
        "Liquidity": float(liquidity)
    }

    # --------------------------
    # COMBAT STATS (0–10)
    # --------------------------
    upside_power = _bucket_score(upside_n, bins=[0.8, 1.5, 2.5], scores=[2.5, 5.0, 7.5, 9.5])
    downside_risk = (10.0 - _bucket_score(downside_n, bins=[0.8, 1.5, 2.5], scores=[2.5, 5.0, 7.5, 9.5])) if pd.notna(downside_n) else 5.5
    downside_risk = _clip(downside_risk, 0, 10)

    rr_eff = _bucket_score(rr, bins=[1.2, 1.8, 2.5], scores=[2.5, 5.0, 7.5, 9.5]) if pd.notna(rr) else 5.0

    # Breakout Force: close above key + vol confirm + candle - divergence/overheat
    above_ma200 = (pd.notna(close) and pd.notna(ma200) and close >= ma200)
    b1 = 3.5 if above_ma200 else 1.5
    b2 = _bucket_score(vol_ratio, bins=[0.9, 1.2, 1.6, 2.2], scores=[0.8, 1.6, 2.4, 3.0, 3.5])
    b3 = 3.0 * _candle_strength_from_class(candle_class)
    overheat_pen = 1.5 if (pd.notna(rsi14) and rsi14 >= 75 and "contract" in hist_state) else 0.0
    div_pen2 = 1.5 if ("bear" in rsi_div or "bear" in macd_div) else 0.0
    breakout_force = _clip(b1 + b2 + b3 - overheat_pen - div_pen2, 0, 10)

    # Support Resilience: confluence + absorption + RSI integrity
    conf = 3.5 if (pd.notna(confluence_count) and confluence_count >= 3) else (2.0 if pd.notna(confluence_count) and confluence_count >= 2 else 1.0)
    absorption = 2.5 if (_safe_bool(pa.get("NoSupplyFlag")) or "hammer" in _safe_text(candle_class).lower()) else 1.2
    rsi_ok = 2.5 if (pd.notna(rsi14) and rsi14 >= 50) else 1.2
    support_resilience = _clip(conf + absorption + rsi_ok, 0, 10)

    # Upside quality (0–10): keep UpsidePower as raw "room", then score a quality-adjusted variant.
    # - Confidence (Tech) acts as a stabilizer: High > Medium > Low.
    # - BreakoutForce / VolumeRatio / RR reinforce quality (not room).
    ps = ap.get("PrimarySetup") or {}
    conf_label = _safe_text(ps.get("Confidence (Tech)", ps.get("Probability", ""))).strip().lower()

    conf_mult = 1.0
    if "high" in conf_label:
        conf_mult = 1.15
    elif "med" in conf_label:
        conf_mult = 1.00
    elif "low" in conf_label:
        conf_mult = 0.80

    m_breakout = 0.85 + 0.30 * (float(breakout_force) / 10.0)
    m_vol = 1.0
    if pd.notna(vol_ratio):
        m_vol = 0.90 + 0.20 * _clip((float(vol_ratio) - 0.80) / 1.80, 0, 1)
    m_rr = 1.0
    if pd.notna(rr):
        m_rr = 0.90 + 0.20 * _clip((float(rr) - 1.20) / 3.00, 0, 1)

    total_mult = _clip(conf_mult * m_breakout * m_vol * m_rr, 0.65, 1.35)
    upside_quality = _clip(float(_clip(upside_power, 0, 10)) * float(total_mult), 0, 10)

    combat_stats = {
        # Naming: UpsidePower is kept for backward compatibility; UI should call it "Upside Room".
        "UpsideRoom": float(_clip(upside_power, 0, 10)),
        "UpsideQuality": float(upside_quality),
        "UpsidePower": float(_clip(upside_power, 0, 10)),
        "DownsideRisk": float(downside_risk),
        "RREfficiency": float(_clip(rr_eff, 0, 10)),
        "BreakoutForce": float(breakout_force),
        "SupportResilience": float(support_resilience)
    }

    # --------------------------
    # WEAKNESS FLAGS (severity 1–3)
    # --------------------------
    flags = []
    def add_flag(code: str, severity: int, note: str, meta: Optional[Dict[str, Any]] = None):
        rec = {"code": code, "severity": int(severity), "note": note}
        if isinstance(meta, dict) and meta:
            rec["meta"] = meta
        flags.append(rec)

    # NearMajorResistance: quality-aware (structural vs tactical; tier-aware).
    # Prevents false positives where a breakout clears a weak tactical level but hits a structural ceiling (e.g., MA200).
    try:
        ov = ((struct_q or {}).get("OverheadResistance", {}) or {}).get("Nearest", {}) or {}
        meta_p = (struct_q or {}).get("Meta", {}) if isinstance((struct_q or {}).get("Meta", {}), dict) else {}
        near_th = _safe_float(meta_p.get("NearThresholdPct"), default=np.nan)
        dist = _safe_float(ov.get("DistancePct"), default=np.nan)
        hz = _safe_text(ov.get("Horizon") or "N/A").upper()
        tier = _safe_text(ov.get("Tier") or "N/A").upper()

        comps = ov.get("ComponentsTop") if isinstance(ov.get("ComponentsTop"), list) else []
        type_top = _safe_text((comps[0] or {}).get("Type")) if (len(comps) > 0 and isinstance(comps[0], dict)) else ""
        type_top = type_top.strip()
        type_pretty = _pretty_level_label(type_top, side="overhead") if type_top else ""

        is_near = (pd.notna(dist) and pd.notna(near_th) and dist <= near_th)
        if is_near and hz in ("STRUCTURAL", "BOTH") and tier in ("HEAVY", "CONFLUENCE"):
            sev = 3 if tier == "CONFLUENCE" else 2
            note = "Trần cấu trúc gần – upside ngắn bị nén"
            if type_pretty:
                note = f"{note} ({type_pretty})"
            add_flag("NearMajorResistance", sev, note, meta={"Horizon": hz, "Tier": tier, "TypeTop": type_top, "TypeLabel": type_pretty, "DistancePct": dist, "NearThPct": near_th})
        elif is_near and hz == "TACTICAL" and tier in ("LIGHT", "MED"):
            # Minor reminder only (does not block; reduces over-warning on weak tactical levels)
            add_flag("NearMinorResistance", 1, "Cản ngắn hạn gần (tactical)")
    except Exception:
        # Legacy fallback
        if pd.notna(upside_n) and upside_n < 1.0:
            add_flag("NearMajorResistance", 2, "Upside ngắn trước kháng cự gần")
    if pd.notna(vol_ratio) and vol_ratio < 0.9:
        add_flag("NoVolumeConfirm", 2, "Thiếu xác nhận dòng tiền")
    if "bear" in rsi_div:
        add_flag("RSI_BearDiv", 3, "RSI phân kỳ giảm")
    if "bear" in macd_div:
        add_flag("MACD_BearDiv", 3, "MACD phân kỳ giảm")
    if fib_conflict:
        add_flag("TrendConflict", 2, "Xung đột Fib short vs long (ưu tiên luật cấu trúc)")
    if whipsaw:
        add_flag("WhipsawZone", 2, "Vùng nhiễu quanh MA/structure pha trộn")
    if pd.notna(rsi14) and rsi14 >= 75 and "contract" in hist_state:
        add_flag("Overheated", 2, "Đà nóng nhưng histogram co lại")
    if liquidity <= 4.5:
        add_flag("LiquidityLow", 2, "Thanh khoản thấp, dễ trượt giá")
    if (_safe_bool(climax) or _safe_bool(gap)):
        add_flag("VolShockRisk", 2, "Có dấu hiệu shock/gap")

    # --------------------------
    # CONVICTION TIER (0–7)
    # --------------------------
    points = 0.0
    points += 1.0 if trend >= 7 else (0.5 if trend >= 5 else 0.0)
    points += 1.0 if momentum >= 7 else (0.5 if momentum >= 5 else 0.0)
    # Location: confluence or breakout strength
    points += 1.0 if (pd.notna(confluence_count) and confluence_count >= 3) else (0.5 if breakout_force >= 6.5 else 0.0)
    points += 1.0 if (pd.notna(vol_ratio) and vol_ratio >= 1.2) else (0.5 if pd.notna(vol_ratio) and vol_ratio >= 1.0 else 0.0)
    points += 1.0 if (pd.notna(rr) and rr >= 1.8) else (0.5 if pd.notna(rr) and rr >= 1.4 else 0.0)
    points += 1.0 if reliability >= 7 else (0.5 if reliability >= 5 else 0.0)

    # Bonus: killer zone (confluence>=4 + strong candle + vol confirm)
    if (pd.notna(confluence_count) and confluence_count >= 4 and _candle_strength_from_class(candle_class) >= 0.7 and pd.notna(vol_ratio) and vol_ratio >= 1.2):
        points += 1.0

    # Penalties
    for f in flags:
        if f["severity"] == 2: points -= 0.5
        if f["severity"] == 3: points -= 1.0
        if f["code"] == "TrendConflict" and f["severity"] >= 2: points -= 0.5

    points = float(max(0.0, min(7.0, points)))

    # Map points to tier (same thresholds as spec)
    if points <= 1.5: tier = 1
    elif points <= 2.5: tier = 2
    elif points <= 3.5: tier = 3
    elif points <= 4.5: tier = 4
    elif points <= 5.5: tier = 5
    elif points <= 6.5: tier = 6
    else: tier = 7

    # Size guidance
    size_map = {
        1: "No edge — đứng ngoài",
        2: "Edge yếu — quan sát / size nhỏ",
        3: "Trade được — 30–50% size",
        4: "Edge tốt — full size",
        5: "Edge mạnh — full size + có thể add",
        6: "Hiếm — có thể overweight có kiểm soát",
        7: "God-tier — ưu tiên cao nhất, quản trị rủi ro chặt"
    }
    size_guidance = size_map.get(tier, "N/A")
    # --------------------------
    # STOCK TRAITS (5Y OHLCV) — Composite Scores (0–10)
    # Goal: improve class assignment without touching Report A–D logic.
    # --------------------------
    # Notes:
    # - Scores are designed to be stable over time and reflect "stock character".
    # - If OHLCV is missing/insufficient, traits fall back to neutral (5.0).
    def _get_series(col_names):
        # Case-insensitive lookup
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype=float)
        cols = {str(c).strip().lower(): c for c in df.columns}
        for n in col_names:
            key = str(n).strip().lower()
            if key in cols:
                try:
                    return pd.to_numeric(df[cols[key]], errors="coerce")
                except Exception:
                    return pd.Series(dtype=float)
        return pd.Series(dtype=float)

    o = _get_series(["Open", "O"])
    h = _get_series(["High", "H"])
    l = _get_series(["Low", "L"])
    c = _get_series(["Close", "C", "Adj Close", "AdjClose"])
    v = _get_series(["Volume", "Vol", "V"])

    # Ensure consistent length and recent window (up to ~5y daily)
    _n = int(min(len(c), 1260)) if hasattr(c, "__len__") else 0
    o = o.tail(_n) if _n else o
    h = h.tail(_n) if _n else h
    l = l.tail(_n) if _n else l
    c = c.tail(_n) if _n else c
    v = v.tail(_n) if _n else v

    def _roll_median(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).median()

    def _roll_mean(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).mean()

    def _roll_std(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).std(ddof=0)

    def _atr14(_h, _l, _c):
        prev_c = _c.shift(1)
        tr = pd.concat([(_h - _l).abs(), (_h - prev_c).abs(), (_l - prev_c).abs()], axis=1).max(axis=1)
        return _roll_mean(tr, 14), tr

    def _adx14(_h, _l, _c):
        # Wilder's ADX14 (robust, simplified smoothing)
        up = _h.diff()
        dn = -_l.diff()
        dm_p = up.where((up > dn) & (up > 0), 0.0)
        dm_m = dn.where((dn > up) & (dn > 0), 0.0)
        atr, tr = _atr14(_h, _l, _c)
        # Wilder smoothing (EMA with alpha=1/14)
        alpha = 1.0 / 14.0
        tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        dm_p_s = dm_p.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        dm_m_s = dm_m.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        di_p = 100.0 * (dm_p_s / tr_s).replace([np.inf, -np.inf], np.nan)
        di_m = 100.0 * (dm_m_s / tr_s).replace([np.inf, -np.inf], np.nan)
        dx = (100.0 * (di_p - di_m).abs() / (di_p + di_m)).replace([np.inf, -np.inf], np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        return adx

    def _percentile_rank(hist: pd.Series, x: float) -> float:
        # Robust percentile rank of x within hist (0..1)
        try:
            s = pd.to_numeric(hist, errors="coerce").dropna()
            if len(s) < 30 or (x is None) or (not np.isfinite(float(x))):
                return np.nan
            x = float(x)
            less = float((s < x).sum())
            eq = float((s == x).sum())
            return (less + 0.5 * eq) / float(len(s))
        except Exception:
            return np.nan

    def _score_from_bins(x, bins, scores, default=5.0):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        try:
            return float(_bucket_score(float(x), bins=bins, scores=scores))
        except Exception:
            return float(default)

    # ===== Trend Integrity (TI) =====
    if _n >= 260 and pd.notna(c.iloc[-1]):
        ma20_s = c.rolling(20, min_periods=20).mean()
        ma50_s = c.rolling(50, min_periods=50).mean()
        ma200_s = c.rolling(200, min_periods=200).mean()

        # PctAboveMA200 over 1y and 2y (if available)
        w1 = int(min(252, _n))
        w2 = int(min(504, _n))
        pct_above_1y = float((c.tail(w1) > ma200_s.tail(w1)).mean()) if w1 >= 50 else np.nan
        pct_above_2y = float((c.tail(w2) > ma200_s.tail(w2)).mean()) if w2 >= 100 else np.nan
        pct_above = np.nanmean([pct_above_1y, pct_above_2y])

        # MA stack consistency (bull or bear)
        stack_bull = (ma20_s > ma50_s) & (ma50_s > ma200_s)
        stack_bear = (ma20_s < ma50_s) & (ma50_s < ma200_s)
        stack_cons = float((stack_bull.tail(w1) | stack_bear.tail(w1)).mean()) if w1 >= 50 else np.nan

        # Flip rate (churn) on MA200 side + MA20/MA50 cross
        side200 = np.sign((c - ma200_s).dropna())
        flips200 = float((side200.tail(w1).diff().fillna(0) != 0).sum()) if len(side200.tail(w1)) > 5 else np.nan
        spread2050 = (ma20_s - ma50_s).dropna()
        flips2050 = float((np.sign(spread2050.tail(w1)).diff().fillna(0) != 0).sum()) if len(spread2050.tail(w1)) > 5 else np.nan
        flip_rate = np.nanmean([flips200, flips2050]) * (252.0 / max(1.0, float(w1)))

        # ADX strength (median last ~6 months)
        adx = _adx14(h, l, c)
        adx_med = float(adx.dropna().tail(126).median()) if len(adx.dropna()) >= 30 else np.nan
    else:
        pct_above = stack_cons = flip_rate = adx_med = np.nan

    ti_pct_score = _score_from_bins(pct_above, bins=[0.35, 0.50, 0.65, 0.80], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    ti_stack_score = _score_from_bins(stack_cons, bins=[0.15, 0.30, 0.45, 0.60], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    ti_flip_score = _score_from_bins(flip_rate, bins=[3, 7, 12, 18], scores=[10.0, 8.0, 6.0, 4.0, 2.0])  # lower flip is better
    ti_adx_score = _score_from_bins(adx_med, bins=[15, 20, 25, 35], scores=[3.0, 5.0, 7.0, 9.0, 10.0])
    trend_integrity = _clip(0.30*ti_pct_score + 0.20*ti_stack_score + 0.30*ti_flip_score + 0.20*ti_adx_score, 0, 10)

    # ===== Volatility Structure (VS) — we use VolRisk (0–10) =====
    if _n >= 60 and pd.notna(c.iloc[-1]):
        r = np.log(c).diff()
        rv20 = float(_roll_std(r, 20).iloc[-1] * np.sqrt(252.0) * 100.0) if len(r.dropna()) >= 25 else np.nan
        rv60 = float(_roll_std(r, 60).iloc[-1] * np.sqrt(252.0) * 100.0) if len(r.dropna()) >= 65 else np.nan
        rv = np.nanmean([rv20, rv60])

        atr, tr = _atr14(h, l, c)
        atr_pct = float((atr.iloc[-1] / c.iloc[-1]) * 100.0) if pd.notna(atr.iloc[-1]) and c.iloc[-1] != 0 else np.nan

        rv20_series = _roll_std(r, 20) * np.sqrt(252.0) * 100.0
        vol_of_vol = float(_roll_std(rv20_series, 252).iloc[-1]) if len(rv20_series.dropna()) >= 300 else np.nan

        tr_med20 = _roll_median(tr, 20)
        exp_rate = float((tr.tail(60) > tr_med20.tail(60)).mean()) if len(tr.dropna()) >= 80 else np.nan
    else:
        rv = atr_pct = vol_of_vol = exp_rate = np.nan

    # Convert to "risk score" where higher = more volatile / harder to control
    vs_atr_risk = _score_from_bins(atr_pct, bins=[1.2, 2.0, 3.0, 4.5], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_rv_risk  = _score_from_bins(rv,      bins=[18, 25, 35, 50],    scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_vov_risk = _score_from_bins(vol_of_vol, bins=[4, 7, 12, 18],   scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_exp_risk = _score_from_bins(exp_rate, bins=[0.35, 0.45, 0.55, 0.65], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vol_risk = _clip(0.30*vs_atr_risk + 0.25*vs_rv_risk + 0.25*vs_vov_risk + 0.20*vs_exp_risk, 0, 10)

    # ===== Tail & Gap Risk (TGR) — higher = worse =====
    if _n >= 260 and pd.notna(c.iloc[-1]):
        ret = c.pct_change()
        ret_1y = ret.tail(252)
        mu = float(ret_1y.mean()) if len(ret_1y.dropna()) >= 50 else np.nan
        sd = float(ret_1y.std(ddof=0)) if len(ret_1y.dropna()) >= 50 else np.nan
        thr = (mu - 2.0*sd) if (np.isfinite(mu) and np.isfinite(sd)) else np.nan
        left_tail_freq = float((ret_1y < thr).mean()) if np.isfinite(thr) else np.nan

        # ES 5% (absolute magnitude)
        q05 = float(ret_1y.quantile(0.05)) if len(ret_1y.dropna()) >= 50 else np.nan
        es5 = float(ret_1y[ret_1y <= q05].mean()) if np.isfinite(q05) and (ret_1y <= q05).sum() >= 5 else np.nan
        es5_abs = abs(es5) * 100.0 if np.isfinite(es5) else np.nan

        prev_c = c.shift(1)
        gap_pct = (o - prev_c).abs() / prev_c
        gap_freq = float((gap_pct.tail(252) > 0.015).mean()) if len(gap_pct.dropna()) >= 80 else np.nan

        # Crash clusters: count sequences of >=2 consecutive days with ret <= -2%
        crash = (ret <= -0.02).astype(int)
        crash_1y = crash.tail(252).fillna(0).to_numpy()
        clusters = 0
        run = 0
        for x in crash_1y:
            if x == 1:
                run += 1
            else:
                if run >= 2:
                    clusters += 1
                run = 0
        if run >= 2:
            clusters += 1
        crash_clusters = float(clusters) * (252.0 / 252.0)
    else:
        left_tail_freq = es5_abs = gap_freq = crash_clusters = np.nan

    tgr_tail = _score_from_bins(left_tail_freq, bins=[0.01, 0.025, 0.05, 0.08], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_es   = _score_from_bins(es5_abs,       bins=[2.0, 3.5, 5.0, 7.0],     scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_gap  = _score_from_bins(gap_freq,      bins=[0.01, 0.03, 0.06, 0.10],  scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_clu  = _score_from_bins(crash_clusters,bins=[1, 2, 4, 6],             scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tail_risk = _clip(0.35*tgr_tail + 0.30*tgr_es + 0.20*tgr_gap + 0.15*tgr_clu, 0, 10)

    # ===== Mean-Reversion / Whipsaw Propensity (MRW) — higher = more range/whipsaw =====
    def _variance_ratio(_c, k=5, w=252):
        _c = pd.to_numeric(_c, errors="coerce")
        r1 = _c.pct_change()
        rk = _c.pct_change(k)
        r1w = r1.tail(w).dropna()
        rkw = rk.tail(w).dropna()
        if len(r1w) < max(30, k*6) or len(rkw) < max(30, k*6):
            return np.nan
        v1 = float(r1w.var(ddof=0))
        vk = float(rkw.var(ddof=0))
        if v1 <= 0:
            return np.nan
        return vk / (k * v1)

    def _half_life_spread(spread, w=252):
        s = pd.to_numeric(spread, errors="coerce").dropna().tail(w)
        if len(s) < 60:
            return np.nan
        x = s.shift(1).dropna()
        y = s.loc[x.index]
        if len(x) < 50:
            return np.nan
        # OLS slope b
        b = np.polyfit(x.values, y.values, 1)[0]
        if b <= 0 or b >= 0.999:
            return np.nan
        return float(-np.log(2.0) / np.log(b))

    if _n >= 300 and pd.notna(c.iloc[-1]):
        vr5 = _variance_ratio(c, k=5, w=504)
        vr10 = _variance_ratio(c, k=10, w=504)
        vr = np.nanmean([vr5, vr10])

        r = c.pct_change()
        ac_list = []
        for lag in [1,2,3,4,5]:
            ac = float(r.tail(504).autocorr(lag=lag)) if len(r.dropna()) >= 200 else np.nan
            if np.isfinite(ac):
                ac_list.append(ac)
        ac_mean = float(np.nanmean(ac_list)) if len(ac_list) else np.nan

        atr, tr = _atr14(h, l, c)
        spread = (c - c.rolling(20, min_periods=20).mean()) / (atr.replace(0, np.nan))
        hl = _half_life_spread(spread, w=504)

        ma20_s = c.rolling(20, min_periods=20).mean()
        ma50_s = c.rolling(50, min_periods=50).mean()
        cross = ((ma20_s - ma50_s).apply(np.sign)).diff().fillna(0)
        cross_churn = float((cross.tail(252) != 0).sum())
    else:
        vr = ac_mean = hl = cross_churn = np.nan

    mr_vr = _score_from_bins(vr, bins=[0.85, 0.95, 1.05, 1.15], scores=[10.0, 8.0, 5.0, 3.0, 1.0])  # lower VR => more mean-reversion
    mr_ac = _score_from_bins(ac_mean, bins=[-0.08, -0.03, 0.03, 0.08], scores=[10.0, 8.0, 5.0, 3.0, 1.0])  # negative autocorr => mean-reversion
    mr_hl = _score_from_bins(hl, bins=[5, 10, 20, 40], scores=[10.0, 8.0, 6.0, 4.0, 2.0])  # smaller half-life => faster reversion
    mr_ch = _score_from_bins(cross_churn, bins=[2, 5, 10, 18], scores=[2.0, 4.0, 6.0, 8.0, 10.0])  # more churn => more whipsaw
    meanrev_prop = _clip(0.30*mr_vr + 0.20*mr_ac + 0.20*mr_hl + 0.30*mr_ch, 0, 10)

    # ===== Breakout Quality (BQ) — higher = better follow-through =====
    breakout_quality = 5.0
    bq_conf = 0.0
    ft_rate = fb_rate = retest_rate = np.nan
    if _n >= 300 and pd.notna(c.iloc[-1]) and len(v.dropna()) >= 80:
        atr, tr = _atr14(h, l, c)
        hh55 = h.rolling(55, min_periods=55).max().shift(1)
        level = hh55
        # Breakout day: close > HH55 and prior close <= HH55
        bday = (c > hh55) & (c.shift(1) <= hh55)
        vol_med20 = v.rolling(20, min_periods=20).median().shift(1)
        vconf = v > (vol_med20 * 1.2)
        events = (bday & vconf).tail(252)
        idxs = list(np.where(events.fillna(False).to_numpy())[0])
        # Map idxs to absolute indices in tail window
        base = int(len(c.tail(252)) - len(events))  # usually 0
        # We'll iterate using positions within the tail(252) window for simplicity
        c252 = c.tail(252).reset_index(drop=True)
        h252 = h.tail(252).reset_index(drop=True)
        l252 = l.tail(252).reset_index(drop=True)
        lvl252 = level.tail(252).reset_index(drop=True)
        atr252 = atr.tail(252).reset_index(drop=True)
        evpos = np.where((bday & vconf).tail(252).fillna(False).to_numpy())[0].tolist()

        if len(evpos) >= 3:
            ft = fb = rt = 0
            for p in evpos:
                lvl = float(lvl252.iloc[p]) if pd.notna(lvl252.iloc[p]) else np.nan
                if not np.isfinite(lvl):
                    continue
                # False break: within next 5 days, close falls back below level
                fb_win = c252.iloc[p+1:min(p+6, len(c252))]
                if len(fb_win) > 0 and (fb_win < lvl).any():
                    fb += 1
                # Follow-through: within next 10 days, (a) stays above level for most days OR (b) reaches +1 ATR from breakout close
                ft_win = c252.iloc[p+1:min(p+11, len(c252))]
                h_win = h252.iloc[p+1:min(p+11, len(h252))]
                atr_p = float(atr252.iloc[p]) if pd.notna(atr252.iloc[p]) else np.nan
                c_p = float(c252.iloc[p]) if pd.notna(c252.iloc[p]) else np.nan
                cond_a = (len(ft_win) >= 5 and (ft_win > lvl).mean() >= 0.70)
                cond_b = (np.isfinite(atr_p) and np.isfinite(c_p) and len(h_win) > 0 and (h_win.max() >= c_p + atr_p))
                if cond_a or cond_b:
                    ft += 1
                # Retest success: touches near level then closes meaningfully above within a few days
                lo_win = l252.iloc[p+1:min(p+11, len(l252))]
                if len(lo_win) > 0 and (lo_win <= lvl * 1.005).any():
                    # after first touch, require a close > lvl*1.01 within next 5 days
                    touch_idx = int(np.where((lo_win <= lvl * 1.005).to_numpy())[0][0]) + (p+1)
                    rec_win = c252.iloc[touch_idx:min(touch_idx+6, len(c252))]
                    if len(rec_win) > 0 and (rec_win >= lvl * 1.01).any():
                        rt += 1
            n_ev = max(1, len(evpos))
            ft_rate = ft / n_ev
            fb_rate = fb / n_ev
            retest_rate = rt / n_ev
            breakout_quality = _clip((0.50*ft_rate + 0.30*retest_rate + 0.20*(1.0 - fb_rate)) * 10.0, 0, 10)
            bq_conf = 1.0
        else:
            # Not enough events — neutral score, low confidence
            breakout_quality = 5.0
            bq_conf = 0.3

    # ===== Liquidity & Tradability (LT) — higher = more tradable =====
    liq_tradability = 5.0
    dv20 = amihud20 = vol_cv20 = np.nan
    dv_score = ami_score = cv_score = 5.0
    if _n >= 120 and pd.notna(c.iloc[-1]) and len(v.dropna()) >= 60:
        dollar_vol = (c * v).replace([np.inf, -np.inf], np.nan)
        dv20_s = dollar_vol.rolling(20, min_periods=20).median()
        dv20 = float(dv20_s.iloc[-1]) if pd.notna(dv20_s.iloc[-1]) else np.nan
        dv_pct = _percentile_rank(dv20_s.dropna(), dv20)
        dv_score = _clip((dv_pct * 10.0) if np.isfinite(dv_pct) else 5.0, 0, 10)

        ret = c.pct_change()
        amihud = (ret.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
        amihud_s = amihud.rolling(20, min_periods=20).mean()
        amihud20 = float(amihud_s.iloc[-1]) if pd.notna(amihud_s.iloc[-1]) else np.nan
        ami_pct = _percentile_rank(amihud_s.dropna(), amihud20)
        ami_score = _clip(((1.0 - ami_pct) * 10.0) if np.isfinite(ami_pct) else 5.0, 0, 10)

        vol_cv = (v.rolling(20, min_periods=20).std(ddof=0) / v.rolling(20, min_periods=20).mean()).replace([np.inf, -np.inf], np.nan)
        vol_cv20 = float(vol_cv.iloc[-1]) if pd.notna(vol_cv.iloc[-1]) else np.nan
        cv_pct = _percentile_rank(vol_cv.dropna(), vol_cv20)
        cv_score = _clip(((1.0 - cv_pct) * 10.0) if np.isfinite(cv_pct) else 5.0, 0, 10)

        liq_tradability = _clip(0.50*dv_score + 0.30*ami_score + 0.20*cv_score, 0, 10)

    # ===== Drawdown & Recovery (DR) — higher = worse (riskier) =====
    mdd_abs = dd_freq = rec_days = np.nan
    mdd_risk = dd_freq_risk = rec_risk = 5.0
    if _n >= 260 and len(c.dropna()) >= 260:
        c2 = c.dropna()
        roll_max = c2.cummax()
        dd = (c2 / roll_max) - 1.0
        mdd = float(dd.min()) if len(dd) else np.nan
        mdd_abs = abs(mdd) * 100.0 if np.isfinite(mdd) else np.nan
    
        # Count drawdown "episodes" deeper than -10%
        thresh = -0.10
        below = (dd <= thresh).astype(int)
        dd_events = int(((below.diff() == 1).sum())) if len(below) > 1 else 0
        years = max(1.0, float(len(dd)) / 252.0)
        dd_freq = float(dd_events) / years
    
        # Recovery speed: median days from DD episode start back to prior peak
        rec_days_list: List[int] = []
        in_dd = False
        start_i = None
        peak_val = None
        vals = c2.values
        peaks = roll_max.values
        for i in range(len(vals)):
            ddv = float(dd.iloc[i])
            if (not in_dd) and (ddv <= thresh):
                in_dd = True
                start_i = i
                peak_val = float(peaks[i])
            if in_dd and peak_val and (vals[i] >= peak_val * 0.999):
                rec_days_list.append(int(i - (start_i or 0)))
                in_dd = False
                start_i = None
                peak_val = None
        if rec_days_list:
            rec_days = float(np.median(rec_days_list))
    
        mdd_risk = _score_from_bins(mdd_abs, bins=[12, 20, 30, 45], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
        dd_freq_risk = _score_from_bins(dd_freq, bins=[0.5, 1.2, 2.5, 4.0], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
        rec_risk = _score_from_bins(rec_days, bins=[20, 45, 90, 150], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    
    # Autocorrelation (lag-1) — momentum vs mean-reversion tendency (higher = more momentum)
    autocorr1 = np.nan
    autocorr_score = 5.0
    try:
        _ret = c.pct_change().dropna()
        if len(_ret) >= 260:
            autocorr1 = float(_ret.tail(756).autocorr(lag=1))
            autocorr_score = _score_from_bins(autocorr1, bins=[-0.15, -0.05, 0.05, 0.15],
                                              scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    except Exception:
        pass


    stock_traits = {
        "TrendIntegrity": float(trend_integrity),
        "VolRisk": float(vol_risk),
        "TailGapRisk": float(tail_risk),
        "MeanReversionWhipsaw": float(meanrev_prop),
        "BreakoutQuality": float(breakout_quality),
        "LiquidityTradability": float(liq_tradability),
        "Confidence": {
            "BreakoutQuality": float(bq_conf)
        },
        "Raw": {
            "PctAboveMA200": float(pct_above) if np.isfinite(pct_above) else np.nan,
            "MAStackConsistency": float(stack_cons) if np.isfinite(stack_cons) else np.nan,
            "TrendFlipRatePerYear": float(flip_rate) if np.isfinite(flip_rate) else np.nan,
            "ADXMedian": float(adx_med) if np.isfinite(adx_med) else np.nan,
            "ATRpct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
            "RealizedVol": float(rv) if np.isfinite(rv) else np.nan,
            "VolOfVol": float(vol_of_vol) if np.isfinite(vol_of_vol) else np.nan,
            "RangeExpansionRate": float(exp_rate) if np.isfinite(exp_rate) else np.nan,
            "LeftTailFreq": float(left_tail_freq) if np.isfinite(left_tail_freq) else np.nan,
            "ES5AbsPct": float(es5_abs) if np.isfinite(es5_abs) else np.nan,
            "GapFreq": float(gap_freq) if np.isfinite(gap_freq) else np.nan,
            "CrashClusters": float(crash_clusters) if np.isfinite(crash_clusters) else np.nan,
            "VarianceRatio": float(vr) if np.isfinite(vr) else np.nan,
            "AutocorrMean": float(ac_mean) if np.isfinite(ac_mean) else np.nan,
            "HalfLife": float(hl) if np.isfinite(hl) else np.nan,
            "CrossChurn": float(cross_churn) if np.isfinite(cross_churn) else np.nan,
            "FollowThroughRate": float(ft_rate) if np.isfinite(ft_rate) else np.nan,
            "FalseBreakRate": float(fb_rate) if np.isfinite(fb_rate) else np.nan,
            "RetestSuccessRate": float(retest_rate) if np.isfinite(retest_rate) else np.nan,
            "DollarVolMedian20": float(dv20) if np.isfinite(dv20) else np.nan,
            "Amihud20": float(amihud20) if np.isfinite(amihud20) else np.nan,
            "VolumeCV20": float(vol_cv20) if np.isfinite(vol_cv20) else np.nan
        }
    }

    # Adjust existing stats using traits (bounded, non-invasive)
    trend_adj = _clip(trend + ((trend_integrity - 5.0) / 5.0) * 1.2, 0, 10)
    momentum_adj = _clip(momentum + ((breakout_quality - 5.0) / 5.0) * 0.8 - ((meanrev_prop - 5.0) / 5.0) * 0.5, 0, 10)
    stability_adj = _clip(stability - ((vol_risk - 5.0) / 5.0) * 1.2 - ((tail_risk - 5.0) / 5.0) * 1.0 - ((meanrev_prop - 5.0) / 5.0) * 0.6, 0, 10)
    reliability_adj = _clip(reliability + ((trend_integrity - 5.0) / 5.0) * 1.0 + ((breakout_quality - 5.0) / 5.0) * 0.8 + ((liq_tradability - 5.0) / 5.0) * 0.6
                           - ((tail_risk - 5.0) / 5.0) * 1.0 - ((meanrev_prop - 5.0) / 5.0) * 0.7, 0, 10)

    adjusted_stats = {
        "TrendAdj": float(trend_adj),
        "MomentumAdj": float(momentum_adj),
        "StabilityAdj": float(stability_adj),
        "ReliabilityAdj": float(reliability_adj)
    }
    # Character class (2-tier DNA taxonomy; stable, long-run oriented)
    # Tier-1: StyleAxis ∈ {"Trend","Momentum","Range","Hybrid"}; RiskRegime ∈ {"Low","Mid","High"}
    # Tier-2: 6–8 classes mapped via thresholds (designed to be stable, not "current snapshot").
    # Notes:
    # - Risk metrics (VolRisk/TailGapRisk/Drawdown*) are "higher = worse".
    # - Style metrics (TrendIntegrity/BreakoutQuality/MeanReversion*) define how the stock typically trades.
    def _tier1_style() -> str:
        # Momentum attempt dominates only when breakout quality + momentum are persistent.
        if (breakout_quality >= 6.8 and momentum_adj >= 6.5 and meanrev_prop <= 6.5):
            return "Momentum"
        if (meanrev_prop >= 6.8 or whipsaw or (autocorr1 is not None and np.isfinite(autocorr1) and autocorr1 <= -0.05)):
            return "Range"
        if (trend_integrity >= 6.7 and meanrev_prop <= 5.7 and (autocorr1 is None or (not np.isfinite(autocorr1)) or autocorr1 >= -0.02)):
            return "Trend"
        return "Hybrid"

    def _tier1_risk() -> Tuple[str, float]:
        risks = []
        for x in [vol_risk, tail_risk, mdd_risk, rec_risk]:
            try:
                xv = float(x)
                if np.isfinite(xv):
                    risks.append(xv)
            except Exception:
                pass
        rscore = float(np.mean(risks)) if risks else 5.0
        if rscore <= 4.5:
            return "Low", rscore
        if rscore >= 6.5:
            return "High", rscore
        return "Mid", rscore

    style_axis = _tier1_style()
    risk_regime, risk_score = _tier1_risk()

    # Tier-2 mapping (8 Classes) — STRICTLY long-run (3–5y) only
    # Modifiers (derived from long-run risks / tradability)
    liq_cons = float(liq_tradability) if np.isfinite(liq_tradability) else 5.0
    liq_level = float(dv_score) if np.isfinite(dv_score) else 5.0
    tail_r = float(tail_risk) if np.isfinite(tail_risk) else 5.0
    vol_r = float(vol_risk) if np.isfinite(vol_risk) else 5.0
    dd_r = float(mdd_risk) if np.isfinite(mdd_risk) else 5.0
    vov_r = float(vs_vov_risk) if np.isfinite(vs_vov_risk) else 5.0

    # Optional raw gap frequency (if available)
    gap_f = stock_traits.get("Raw", {}).get("GapFreq", np.nan)
    try:
        gap_f = float(gap_f)
    except Exception:
        gap_f = np.nan

    modifiers: List[str] = []
    if liq_cons <= 3.0 or liq_level <= 3.0:
        modifiers.append("ILLIQ")
    if (tail_r >= 7.5) or (np.isfinite(gap_f) and gap_f >= 0.08):
        modifiers.append("GAP")
    if (vol_r >= 7.2) or (dd_r >= 7.2):
        modifiers.append("HIVOL")
    if vov_r >= 7.0:
        modifiers.append("CHOPVOL")
    if (vol_r <= 3.8 and dd_r <= 5.2 and tail_r <= 6.0 and liq_tradability >= 6.0):
        modifiers.append("DEF")

    # --- v15.1: StyleHybrid v1 (soft scoring + near-boundary tilt; style-level only) ---
    style_hybrid_pack: Dict[str, Any] = {"Version": "StyleHybrid_v1.0", "Enabled": False, "Scores": {}, "Primary": str(style_axis), "TiltTo": "-", "TiltStrength": "-", "NearBoundary": False}
    hard_style_disabled = bool(("ILLIQ" in modifiers) or ("GAP" in modifiers))
    if not hard_style_disabled:
        try:
            style_axis, style_hybrid_pack = _compute_style_hybrid_v1(
                trend_integrity=float(trend_integrity),
                breakout_quality=float(breakout_quality),
                momentum_adj=float(momentum_adj),
                meanrev_prop=float(meanrev_prop),
                whipsaw=bool(whipsaw),
                autocorr1=float(autocorr1) if (autocorr1 is not None and np.isfinite(autocorr1)) else None,
                enabled=True,
            )
        except Exception:
            # keep legacy style_axis; keep pack minimal
            style_hybrid_pack = {"Version": "StyleHybrid_v1.0", "Enabled": False, "Scores": {}, "Primary": str(style_axis), "TiltTo": "-", "TiltStrength": "-", "NearBoundary": False}
    else:
        # Hard Filters: keep NO tilt/runner-up for style (still compute scores for diagnostics)
        try:
            _sa, style_hybrid_pack = _compute_style_hybrid_v1(
                trend_integrity=float(trend_integrity),
                breakout_quality=float(breakout_quality),
                momentum_adj=float(momentum_adj),
                meanrev_prop=float(meanrev_prop),
                whipsaw=bool(whipsaw),
                autocorr1=float(autocorr1) if (autocorr1 is not None and np.isfinite(autocorr1)) else None,
                enabled=False,
                disabled_reason="HardFilter",
            )
            style_hybrid_pack["Primary"] = str(style_axis)  # preserve legacy label
            style_hybrid_pack["TiltTo"] = "-"
            style_hybrid_pack["TiltStrength"] = "-"
            style_hybrid_pack["NearBoundary"] = False
        except Exception:
            style_hybrid_pack = {"Version": "StyleHybrid_v1.0", "Enabled": False, "Scores": {}, "Primary": str(style_axis), "TiltTo": "-", "TiltStrength": "-", "NearBoundary": False}


    # Priority: execution risk first
    if "ILLIQ" in modifiers:
        cclass = "Illiquid / Noisy"
    elif "GAP" in modifiers:
        cclass = "Event / Gap-Prone"
    elif style_axis == "Trend":
        cclass = "Aggressive Trend" if ("HIVOL" in modifiers or risk_regime == "High") else "Smooth Trend"
    elif style_axis == "Momentum":
        cclass = "Momentum Trend"
    elif style_axis == "Range":
        cclass = "Volatile Range" if ("HIVOL" in modifiers or "CHOPVOL" in modifiers or risk_regime == "High") else "Range / Mean-Reversion (Stable)"
    else:
        cclass = "Mixed / Choppy Trader"


# Enrich StockTraits with stable 2-tier DNA taxonomy + 15-parameter pack (for Python tagging & UI)
    try:
        stock_traits.setdefault("DNA", {})
        # DNAConfidence / ClassLockFlag — stability proxy for the long-run DNA label (0–100)
        try:
            n_bars = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        except Exception:
            n_bars = 0

        vol_stability = 10.0 - float(vs_vov_risk) if np.isfinite(vs_vov_risk) else 5.0  # higher = more stable
        liq_cons_score = float(cv_score) if np.isfinite(cv_score) else 5.0
        liq_level_score = float(dv_score) if np.isfinite(dv_score) else 5.0

        dna_confidence = 60.0
        dna_confidence += 10.0 if n_bars >= 900 else (5.0 if n_bars >= 600 else 0.0)
        dna_confidence += 10.0 if vol_stability >= 6.0 else 0.0
        dna_confidence -= 15.0 if (np.isfinite(tail_risk) and float(tail_risk) >= 8.0) else 0.0
        dna_confidence -= 10.0 if (np.isfinite(mdd_risk) and float(mdd_risk) >= 8.0) else 0.0
        dna_confidence -= 15.0 if (liq_cons_score <= 3.0 or liq_level_score <= 3.0) else 0.0
        dna_confidence = float(max(0.0, min(100.0, dna_confidence)))
        class_lock = bool(dna_confidence < 50.0)


        def _pick_primary_modifier(mods: List[str]) -> str:
            for mm in ["ILLIQ","GAP","HIVOL","CHOPVOL","HBETA","DEF"]:
                if mm in mods:
                    return mm
            return ""

        primary_mod = _pick_primary_modifier(modifiers if isinstance(modifiers, list) else [])
        stock_traits["DNA"]["Tier1"] = {
            "StyleAxis": str(style_axis),
            "StyleHybrid": style_hybrid_pack,
            "RiskRegime": str(risk_regime),
            "RiskScore": float(risk_score) if np.isfinite(risk_score) else np.nan,
            # DNA confidence is a stability/coverage proxy (0–100) — strictly long-run inputs only
            "DNAConfidence": float(dna_confidence),
            "ClassLockFlag": bool(class_lock),
            "Modifiers": list(modifiers) if isinstance(modifiers, list) else [],
            "PrimaryModifier": str(primary_mod),
        }
        stock_traits["DNA"]["StyleHybrid"] = style_hybrid_pack
        stock_traits["DNA"]["Params"] = {
            # Group 1: Trend Structure (higher = better)
            "TrendIntegrity": float(trend_integrity),
            "TrendPersistence": float(ti_pct_score),
            "TrendChurnControl": float(ti_flip_score),
    
            # Group 2: Volatility & Tail (higher = worse)
            "VolRisk": float(vol_risk),
            "TailGapRisk": float(tail_risk),
            "VolOfVolRisk": float(vs_vov_risk),
    
            # Group 3: Drawdown & Recovery (higher = worse)
            "MaxDrawdownRisk": float(mdd_risk),
            "RecoverySlownessRisk": float(rec_risk),
            "DrawdownFrequencyRisk": float(dd_freq_risk),
    
            # Group 4: Liquidity & Tradability (higher = better)
            "LiquidityTradability": float(liq_tradability),
            "LiquidityLevel": float(dv_score),
            "LiquidityConsistency": float(cv_score),
    
            # Group 5: Behavior / Setup Bias
            "BreakoutQuality": float(breakout_quality),
            "MeanReversionWhipsaw": float(meanrev_prop),
            "AutoCorrMomentum": float(autocorr_score),
        }
        stock_traits["DNA"]["Groups"] = {
            "TrendStructure": ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"],
            "VolatilityTail": ["VolRisk", "TailGapRisk", "VolOfVolRisk"],
            "DrawdownRecovery": ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"],
            "LiquidityTradability": ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"],
            "BehaviorSetup": ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"],
        }
        # Store extra raw metrics (optional diagnostics)
        stock_traits.setdefault("Raw", {})
        stock_traits["Raw"].update({
            "RealizedVolPct": float(rv) if np.isfinite(rv) else np.nan,
            "ATRpct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
            "VolOfVol": float(vol_of_vol) if np.isfinite(vol_of_vol) else np.nan,
            "TRExpansionRate": float(exp_rate) if np.isfinite(exp_rate) else np.nan,
            "LeftTailFreq": float(left_tail_freq) if np.isfinite(left_tail_freq) else np.nan,
            "ES5AbsPct": float(es5_abs) if np.isfinite(es5_abs) else np.nan,
            "GapFreq": float(gap_freq) if np.isfinite(gap_freq) else np.nan,
            "MaxDrawdownAbsPct": float(mdd_abs) if np.isfinite(mdd_abs) else np.nan,
            "DrawdownEpisodesPerYear": float(dd_freq) if np.isfinite(dd_freq) else np.nan,
            "RecoveryDaysMedian": float(rec_days) if np.isfinite(rec_days) else np.nan,
            "AutoCorr1": float(autocorr1) if np.isfinite(autocorr1) else np.nan,
        })
    except Exception:
        # Hard-fail is not allowed; keep backward compatibility.
        pass

    # Action tags (lightweight, for GPT/UI)
    tags = []
    if tier >= 4 and (pd.notna(confluence_count) and confluence_count >= 3):
        tags.append("Pullback-buy zone (confluence)")
    if breakout_force >= 7:
        tags.append("Breakout attempt (needs follow-through)")
    if any(f["code"] == "NoVolumeConfirm" for f in flags):
        tags.append("Wait for volume confirmation")
    if any(f["code"] in ("NearMajorResistance", "Overheated") for f in flags):
        tags.append("Tight risk control near resistance")
    if fib_conflict:
        tags.append("Use LongStructure_ShortTactical rule")


    # --- v5.9.9: ensure ATR is scalar-safe (avoid pandas truthiness) ---
    atr_scalar = _as_scalar(atr)
    atr_f = None
    try:
        if atr_scalar is not None and atr_scalar == atr_scalar:
            atr_f = float(atr_scalar)
    except Exception:
        atr_f = None
    atr_pos = (atr_f is not None and atr_f > 0)

    # ------------------------------------------------------------
    # v15.2: Linear Pipeline Packs (DNA -> Status -> TradePlan -> Decision)
    # - DNA pack defines Rules of Engagement (plan priors) but does NOT emit actions.
    # - Status pack contextualizes technicals through DNA lens, but does NOT emit actions.
    # ------------------------------------------------------------
    def _plan_prior_map_v1(_class: str, _style: str, _mods: List[str]) -> Dict[str, Any]:
        """Return Strategy_Probability_Map / PlanPriorMap (Python-only, deterministic).

        PlanTypes: ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"].
        Output fields are intentionally lightweight so TradePlanBuilder can score.
        """
        # Base tiers -> numeric scores (0-10)
        tier_score = {
            "HIGH": 8.5,
            "MED": 6.8,
            "LOW": 4.5,
            "VLOW": 3.3,
        }

        hard_ill = "ILLIQ" in (_mods or [])
        hard_gap = "GAP" in (_mods or [])

        cls = (_class or "").lower()
        sty = (_style or "").strip().title()

        # Default: balanced
        pri = {
            "PULLBACK": "MED",
            "BREAKOUT": "MED",
            "MEAN_REV": "MED",
            "RECLAIM": "MED",
            "DEFENSIVE": "MED",
        }

        if "illiquid" in cls or hard_ill:
            pri.update({"DEFENSIVE": "HIGH", "BREAKOUT": "VLOW", "PULLBACK": "LOW", "MEAN_REV": "LOW", "RECLAIM": "LOW"})
        elif "gap" in cls or hard_gap:
            pri.update({"DEFENSIVE": "HIGH", "BREAKOUT": "LOW", "PULLBACK": "LOW", "MEAN_REV": "LOW", "RECLAIM": "MED"})
        elif "momentum" in cls or sty == "Momentum":
            pri.update({"BREAKOUT": "HIGH", "PULLBACK": "MED", "RECLAIM": "MED", "MEAN_REV": "LOW", "DEFENSIVE": "LOW"})
        elif "range" in cls or sty == "Range":
            pri.update({"MEAN_REV": "HIGH", "PULLBACK": "MED", "RECLAIM": "MED", "BREAKOUT": "LOW", "DEFENSIVE": "MED"})
        elif "trend" in cls or sty == "Trend":
            pri.update({"PULLBACK": "HIGH", "BREAKOUT": "MED", "RECLAIM": "MED", "MEAN_REV": "LOW", "DEFENSIVE": "LOW"})
        else:
            # Hybrid / mixed
            pri.update({"PULLBACK": "MED", "BREAKOUT": "MED", "MEAN_REV": "MED", "RECLAIM": "MED", "DEFENSIVE": "MED"})

        # Risk modifiers: be more defensive under high volatility or chop
        if "HIVOL" in (_mods or []) or "CHOPVOL" in (_mods or []):
            # downgrade breakout and upgrade defensive slightly
            if pri.get("BREAKOUT") == "HIGH":
                pri["BREAKOUT"] = "MED"
            if pri.get("DEFENSIVE") in ("LOW", "MED"):
                pri["DEFENSIVE"] = "MED"

        out: Dict[str, Any] = {}
        for k in ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"]:
            t = pri.get(k, "MED")
            score = float(tier_score.get(t, 6.8))
            out[k] = {
                "tier": t,
                "fit_score": score,
            }
        return {
            "schema": "PlanPriorMap.v1",
            "plan_types": ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"],
            "priors": out,
            "notes": "Priors are class/style-based; hard gates (ILLIQ/GAP) override to DEFENSIVE bias.",
        }

    # Build DNAPack (Rules of Engagement)
    dna_pack: Dict[str, Any] = {
        "schema": "DNAPack.v1",
        "class_primary": str(cclass),
        "style_primary": str(style_hybrid_pack.get("Primary") or style_axis),
        "style_tilt_to": str(style_hybrid_pack.get("TiltTo") or "-"),
        "style_tilt_strength": str(style_hybrid_pack.get("TiltStrength") or "-"),
        "near_boundary": bool(style_hybrid_pack.get("NearBoundary")) if isinstance(style_hybrid_pack, dict) else False,
        "risk_regime": str(risk_regime),
        "hard_gates": {
            "illiquid": bool("ILLIQ" in (modifiers or [])),
            "gap_prone": bool("GAP" in (modifiers or [])),
        },
        "modifiers": list(modifiers) if isinstance(modifiers, list) else [],
    }
    try:
        dna_pack["plan_prior_map"] = _plan_prior_map_v1(dna_pack.get("class_primary"), dna_pack.get("style_primary"), dna_pack.get("modifiers"))
    except Exception:
        dna_pack["plan_prior_map"] = {"schema": "PlanPriorMap.v1", "plan_types": ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"], "priors": {}, "notes": "-"}

    # Compute StatusPack (contextualized technicals) — NO actions
    status_pack: Dict[str, Any] = {}
    try:
        status_pack = compute_status_pack_v1(analysis_pack, dna_pack) or {}
    except Exception:
        status_pack = {"schema": "StatusPack.v1", "class_context": {}, "technicals": {}}

    # Attach to AnalysisPack for downstream consumption (safe mutation)
    if isinstance(analysis_pack, dict):
        analysis_pack["DNAPack"] = dna_pack
        analysis_pack["StatusPack"] = status_pack

    
    # --- TradePlanPack v1 (scenario-driven + gate-driven blueprint) ---
    trade_plan_pack: Dict[str, Any] = {}
    try:
        trade_plan_pack = compute_trade_plan_pack_v1(analysis_pack, {
            "CharacterClass": cclass,
            "CombatStats": combat_stats,
            "Flags": flags,
            "StructureQuality": struct_q,
            "DNAPack": dna_pack,
            "StatusPack": status_pack,
        }) or {}
        # Attach to AnalysisPack for single-source-of-truth rendering (safe mutation)
        if isinstance(analysis_pack, dict):
            analysis_pack["TradePlanPack"] = trade_plan_pack
            # DecisionPack (portfolio-ready; long-only)
            try:
                decision_pack = compute_decision_pack_v1(analysis_pack, trade_plan_pack) or {}
                if isinstance(analysis_pack, dict):
                    analysis_pack["DecisionPack"] = decision_pack
                    # PositionManagerPack (portfolio-ready; long-only; sizing-lite)
                    try:
                        position_manager_pack = compute_position_manager_pack_v1(analysis_pack, trade_plan_pack, decision_pack) or {}
                        if isinstance(analysis_pack, dict):
                            analysis_pack["PositionManagerPack"] = position_manager_pack
                    except Exception:
                        position_manager_pack = {}
            except Exception:
                decision_pack = {}

    except Exception:
        trade_plan_pack = {}

    pack = {
        "CharacterClass": cclass,
        "StyleHybrid": stock_traits.get("DNA", {}).get("StyleHybrid", {}),
        "CoreStats": core_stats,
        "AdjustedStats": adjusted_stats,
        "StockTraits": stock_traits,
        "CombatStats": combat_stats,
        "StructureQuality": struct_q,
        "TradePlanPack": trade_plan_pack,
        "Flags": flags,
        "Conviction": {"Points": points, "Tier": tier, "SizeGuidance": size_guidance},
        "ActionTags": tags,
        "Meta": {
            "DenomUsed": "ATR14" if atr_pos else "VolProxy",
            "ConfidenceTech": ps.get("Confidence (Tech)", ps.get("Probability", "N/A")),
            "UpsideQualityMult": float(total_mult) if pd.notna(total_mult) else np.nan,
            "LevelCtxSource": lvl_source,
            "LevelCtxKeys": ",".join(list(lvl.keys())[:8]) if isinstance(lvl, dict) else "",
            "Close": float(close) if pd.notna(close) else np.nan,
            "NearestRes": float(nearest_res) if pd.notna(nearest_res) else np.nan,
            "NearestSup": float(nearest_sup) if pd.notna(nearest_sup) else np.nan,
            "UpsideRaw": float(upside) if pd.notna(upside) else np.nan,
            "DownsideRaw": float(downside) if pd.notna(downside) else np.nan,
            "ATR14": atr_f if (atr_f is not None) else np.nan,
            "VolProxy": float(vol_proxy) if pd.notna(vol_proxy) else np.nan,
            "UpsideNorm": float(upside_n) if pd.notna(upside_n) else np.nan,
            "DownsideNorm": float(downside_n) if pd.notna(downside_n) else np.nan,
            "RR": float(rr) if pd.notna(rr) else np.nan
        }
    }

    # Step 8: normalize pack contract (fail-safe)
    try:
        from inception.core.contracts import normalize_character_pack
        pack = normalize_character_pack(pack)
    except Exception:
        pass


    # Ensure CharacterPack.ClassName exists (for DataQuality/Dashboard)
    try:
        if isinstance(pack, dict):
            class_name = pack.get("ClassName") or pack.get("CharacterClass")
            # common alternates
            if not class_name:
                c = pack.get("Class")
                if isinstance(c, dict):
                    class_name = c.get("Name") or c.get("ClassName")
            if not class_name:
                dna = pack.get("DNA")
                if isinstance(dna, dict):
                    class_name = dna.get("ClassName") or dna.get("Name")
            if not class_name:
                sdna = pack.get("StockDNA")
                if isinstance(sdna, dict):
                    class_name = sdna.get("ClassName") or sdna.get("Name")
            pack["ClassName"] = class_name or "UNKNOWN"
            # provide a normalized Class dict for downstream renders
            if not isinstance(pack.get("Class"), dict):
                pack["Class"] = {"Name": pack["ClassName"]}
            else:
                pack["Class"].setdefault("Name", pack["ClassName"])
    except Exception:
        pass

    return pack


