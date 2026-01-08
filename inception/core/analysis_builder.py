"""inception.core.analysis_builder

Step 12: centralize analysis building into core (no Streamlit/UI).

This module constructs the base AnalysisPack used by Character / Report modules.

Notes:
- Data loading is handled by inception.infra.datahub.DataHub.
- TradePlanPack / DecisionPack are NOT attached here. They are produced by the
  Character module (compute_character_pack_v1) after PositionStatePack injection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd

from inception.infra.datahub import DataHub, DataError

from inception.core.helpers import _as_scalar, _clip, _safe_float
from inception.core.indicators import sma, rsi_wilder, macd
from inception.core.scoring import (
    build_rr_sim,
    classify_scenario,
    classify_scenario12,
    compute_conviction,
    compute_conviction_pack,
    compute_master_score,
)
from inception.core.tradeplan import build_trade_plan
from inception.core.primary_setup import pick_primary_setup_v3
from inception.core.contracts import normalize_analysis_pack


def sanitize_pack(obj: Any) -> Any:
    """Recursively convert obj to JSON-safe / render-safe python primitives.
    - pandas Series/Index -> scalar (latest)
    - pandas DataFrame -> list[dict] (records) (only if needed)
    - numpy scalar -> python scalar
    - NaN/Inf -> None
    """
    try:
        if obj is pd.NaT:
            return None
    except Exception:
        pass

    # pandas containers
    if isinstance(obj, pd.Series):
        return sanitize_pack(_as_scalar(obj))
    if isinstance(obj, pd.Index):
        return [sanitize_pack(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        # Avoid embedding big dataframes by default; keep minimal representation.
        try:
            return [sanitize_pack(r) for r in obj.to_dict(orient="records")]
        except Exception:
            return None

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): sanitize_pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_pack(v) for v in obj]

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        fx = float(obj)
        return fx if math.isfinite(fx) else None
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_pack(v) for v in obj.tolist()]

    # python scalars
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj



def _isnan(x) -> bool:
    try: return x is None or (isinstance(x, float) and np.isnan(x))
    except: return True

def _sgn(x: float) -> int:
    if pd.isna(x): return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0: return np.nan
    return (a - b) / b * 100

def _trend_label_from_slope(slope: float, eps: float = 1e-9) -> str:
    if pd.isna(slope): return "N/A"
    if slope > eps: return "Up"
    if slope < -eps: return "Down"
    return "Flat"

def _find_last_cross(series_a: pd.Series, series_b: pd.Series, lookback: int = 20) -> Dict[str, Any]:
    a = series_a.dropna()
    b = series_b.dropna()
    if a.empty or b.empty:
        return {"Event": "None", "BarsAgo": None}
    
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return {"Event": "None", "BarsAgo": None}
    
    df = df.tail(lookback + 2)
    diff = df["a"] - df["b"]
    sign = diff.apply(_sgn)
    
    last_event = None
    last_bars_ago = None
    
    s = sign.values
    for i in range(len(s) - 1, 0, -1):
        if s[i] == 0 or s[i-1] == 0:
            continue
        if s[i] != s[i-1]:
            if s[i-1] < s[i]:
                last_event = "CrossUp"
            else:
                last_event = "CrossDown"
            last_bars_ago = (len(s) - 1) - i
            break
            
    return {"Event": last_event or "None", "BarsAgo": int(last_bars_ago) if last_bars_ago is not None else None}

def _detect_divergence_simple(close: pd.Series, osc: pd.Series, lookback: int = 60) -> Dict[str, Any]:
    c = close.dropna().tail(lookback).reset_index(drop=True)
    o = osc.dropna().tail(lookback).reset_index(drop=True)
    n = min(len(c), len(o))
    if n < 10:
        return {"Type": "None", "Detail": "N/A"}
    
    c = c.tail(n).reset_index(drop=True)
    o = o.tail(n).reset_index(drop=True)
    
    lows = []
    highs = []
    
    for i in range(2, n-2):
        if c[i] < c[i-1] and c[i] < c[i+1]:
            lows.append(i)
        if c[i] > c[i-1] and c[i] > c[i+1]:
            highs.append(i)
            
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        price_ll = c[i2] < c[i1]
        osc_hl = o[i2] > o[i1]
        if price_ll and osc_hl:
            return {"Type": "Bullish", "Detail": f"Price LL vs Osc HL (swings {i1}->{i2})"}
            
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        price_hh = c[i2] > c[i1]
        osc_lh = o[i2] < o[i1]
        if price_hh and osc_lh:
            return {"Type": "Bearish", "Detail": f"Price HH vs Osc LH (swings {i1}->{i2})"}
            
    return {"Type": "None", "Detail": "N/A"}

# ============================================================
# 4. LOADERS
# ============================================================

def _file_mtime(path: str) -> int:
    """Best-effort file mtime helper.

    This is intentionally defensive and returns 0 on any failure.
    """
    try:
        import os
        return int(os.path.getmtime(path))
    except Exception:
        return 0


def _dynamic_vol_proxy(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Fallback volatility proxy when ATR is unavailable.
    Data-driven (no fixed %):
    - median(|Close.diff|) + 0.25 * median(High-Low)
    """
    if df.empty or "Close" not in df.columns:
        return np.nan
    d = df.tail(max(lookback, 10)).copy()
    if d.empty:
        return np.nan
    cd = d["Close"].astype(float).diff().abs().dropna()
    rng = (d["High"].astype(float) - d["Low"].astype(float)).abs().dropna() if ("High" in d.columns and "Low" in d.columns) else pd.Series(dtype=float)
    m1 = _safe_float(cd.median()) if not cd.empty else np.nan
    m2 = _safe_float(rng.median()) if not rng.empty else np.nan
    if pd.notna(m1) and pd.notna(m2):
        return float(m1 + 0.25 * m2)
    if pd.notna(m1):
        return float(m1)
    if pd.notna(m2):
        return float(0.25 * m2)
    return np.nan



def _fib_levels(low, high):
    rng = high - low
    if rng <= 0: return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng,
        "127.2": high + 0.272 * rng,
        "161.8": high + 0.618 * rng
    }

def _compute_fib_window(df: pd.DataFrame, w: int) -> Dict[str, Any]:
    L = w if len(df) >= w else len(df)
    win = df.tail(L)
    hi, lo = win["High"].max(), win["Low"].min()
    return {"window": L, "swing_high": hi, "swing_low": lo, "levels": _fib_levels(lo, hi)}

def _score_fib_relevance(close: float, fib: Dict[str, Any]) -> float:
    lv = fib.get("levels", {})
    hi = _safe_float(fib.get("swing_high"))
    lo = _safe_float(fib.get("swing_low"))
    if pd.isna(close) or pd.isna(hi) or pd.isna(lo) or hi <= lo:
        return -1e9
    rng = hi - lo
    range_pct = rng / close if close != 0 else np.nan
    s382 = _safe_float(lv.get("38.2"))
    s618 = _safe_float(lv.get("61.8"))
    score = 0.0
    if pd.notna(range_pct):
        score += max(0.0, 1.2 - abs(range_pct - 0.25) * 3.0)
    if range_pct < 0.08:
        score -= 0.8
    if pd.notna(s382) and pd.notna(s618):
        loz = min(s382, s618)
        hiz = max(s382, s618)
        if loz <= close <= hiz:
            score += 2.0
        elif close > hiz:
            score += 1.0
        else:
            score += 0.5
    return score

def compute_dual_fibonacci_auto(df: pd.DataFrame, long_window: int = 250) -> Dict[str, Any]:
    if df.empty:
        return {"short_window": None, "long_window": None, "auto_short": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "fixed_long": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "alt_short": {"window": None, "swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "selection_reason": "N/A"}
    last_close = _safe_float(df.iloc[-1].get("Close"))
    fib60 = _compute_fib_window(df, 60)
    fib90 = _compute_fib_window(df, 90)
    s60 = _score_fib_relevance(last_close, fib60)
    s90 = _score_fib_relevance(last_close, fib90)
    if s90 > s60:
        chosen = fib90
        alt = fib60
        reason = "AutoSelect=90 (higher relevance score)"
    else:
        chosen = fib60
        alt = fib90
        reason = "AutoSelect=60 (higher relevance score)"
    L_long = long_window if len(df) >= long_window else len(df)
    win_long = df.tail(L_long)
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()
    return {
        "short_window": chosen.get("window"),
        "long_window": L_long,
        "auto_short": {"swing_high": chosen.get("swing_high"), "swing_low": chosen.get("swing_low"), "levels": chosen.get("levels", {})},
        "fixed_long": {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)},
        "alt_short": alt,
        "selection_reason": reason
    }

def _sorted_levels(levels: Dict[str, Any]) -> List[Tuple[str, float]]:
    out = []
    for k, v in (levels or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            out.append((str(k), fv))
    return sorted(out, key=lambda x: x[1])

def _nearest_levels_around_price(levels: Dict[str, Any], close: float, n_each: int = 2) -> Dict[str, Any]:
    if pd.isna(close):
        return {"Supports": [], "Resistances": []}
    lv = _sorted_levels(levels)
    supports = [(k, v) for (k, v) in lv if v <= close]
    resistances = [(k, v) for (k, v) in lv if v >= close]
    supports = sorted(supports, key=lambda kv: abs(close - kv[1]))[:n_each]
    resistances = sorted(resistances, key=lambda kv: abs(close - kv[1]))[:n_each]
    def pack(items):
        arr = []
        for k, v in items:
            dist_pct = ((v - close) / close * 100) if close != 0 else np.nan
            arr.append({"Level": k, "Value": v, "DistPct": dist_pct})
        return arr
    return {"Supports": pack(supports), "Resistances": pack(resistances)}

def _position_band(close: float, levels: Dict[str, Any]) -> str:
    if pd.isna(close): return "N/A"
    l382 = _safe_float((levels or {}).get("38.2"))
    l618 = _safe_float((levels or {}).get("61.8"))
    if pd.isna(l382) or pd.isna(l618): return "N/A"
    lo = min(l382, l618)
    hi = max(l382, l618)
    if close >= hi: return "Above61.8"
    if close <= lo: return "Below38.2"
    return "Between38.2_61.8"

def _confluence_with_ma(levels: Dict[str, Any], ma_values: Dict[str, float], tol_pct: float = 0.6) -> Dict[str, Any]:
    res = {}
    lv = _sorted_levels(levels)
    for ma_key, ma_val in (ma_values or {}).items():
        if pd.isna(ma_val) or ma_val == 0:
            res[ma_key] = []
            continue
        hits = []
        for k, v in lv:
            dist_pct = abs(v - ma_val) / ma_val * 100
            if dist_pct <= tol_pct:
                hits.append({"FibLevel": k, "FibValue": v, "DistPct": dist_pct})
        res[ma_key] = hits
    return res

def compute_fibonacci_context_pack(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    close = _safe_float(last.get("Close"))
    ma_vals = {
        "MA20": _safe_float(last.get("MA20")),
        "MA50": _safe_float(last.get("MA50")),
        "MA200": _safe_float(last.get("MA200")),
    }
    short_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
    long_lv  = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
    short_near = _nearest_levels_around_price(short_lv, close, n_each=2)
    long_near  = _nearest_levels_around_price(long_lv, close, n_each=2)
    short_band = _position_band(close, short_lv)
    long_band  = _position_band(close, long_lv)
    conf_short = _confluence_with_ma(short_lv, ma_vals, tol_pct=0.6)
    conf_long  = _confluence_with_ma(long_lv, ma_vals, tol_pct=0.6)
    fib_conflict = (short_band != "N/A" and long_band != "N/A" and short_band != long_band)
    priority_rule = "LongStructure_ShortTactical" if fib_conflict else "None"
    return {
        "Close": close,
        "ShortBand": short_band,
        "LongBand": long_band,
        "NearestShort": short_near,
        "NearestLong": long_near,
        "ConfluenceShortWithMA": conf_short,
        "ConfluenceLongWithMA": conf_long,
        "FiboConflictFlag": bool(fib_conflict),
        "FiboPriorityRuleApplied": priority_rule
    }

# ============================================================
# 6B. PRO TECH FEATURES (PYTHON-ONLY)
# ============================================================
def compute_ma_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    # --------------------------
    # Helper coercions (avoid Series truth-value ambiguity)
    # --------------------------
    def _as_scalar(x: Any) -> Any:
        try:
            if isinstance(x, pd.Series):
                s = x.dropna()
                if not s.empty:
                    return s.iloc[-1]
                return x.iloc[-1] if len(x) else None
            if isinstance(x, (list, tuple, np.ndarray)):
                return x[-1] if len(x) else None
        except Exception:
            return None
        return x

    def _coalesce(*vals: Any) -> Any:
        for v in vals:
            if v is None:
                continue
            v2 = _as_scalar(v)
            if v2 is None:
                continue
            if isinstance(v2, float) and pd.isna(v2):
                continue
            if isinstance(v2, str) and not v2.strip():
                continue
            return v2
        return None

    def _safe_bool(x: Any) -> bool:
        v = _as_scalar(x)
        if v is None:
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return bool(int(v))
        if isinstance(v, (float, np.floating)):
            return bool(v) if pd.notna(v) else False
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "t", "yes", "y", "1", "ok"):
                return True
            if s in ("false", "f", "no", "n", "0", ""):
                return False
        return False
    
    def slope_value(series: pd.Series, n: int = 10) -> float:
        s = series.dropna()
        if len(s) < n + 1: return np.nan
        return _safe_float(s.iloc[-1] - s.iloc[-(n+1)])
    
    def slope_label(val: float, eps: float = 1e-9) -> str:
        if pd.isna(val): return "N/A"
        if val > eps: return "Positive"
        if val < -eps: return "Negative"
        return "Flat"
        
    s20_v = slope_value(df["MA20"], 10)
    s50_v = slope_value(df["MA50"], 10)
    s200_v = slope_value(df["MA200"], 10)
    
    structure_snapshot = "N/A"
    if pd.notna(close) and pd.notna(ma50) and pd.notna(ma200):
        if close >= ma50 and ma50 >= ma200:
            structure_snapshot = "Close>=MA50>=MA200"
        elif close < ma50 and ma50 < ma200:
            structure_snapshot = "Close<MA50<MA200"
        else:
            structure_snapshot = "MixedStructure"
            
    dist50 = ((close - ma50) / ma50 * 100) if (pd.notna(close) and pd.notna(ma50) and ma50 != 0) else np.nan
    dist200 = ((close - ma200) / ma200 * 100) if (pd.notna(close) and pd.notna(ma200) and ma200 != 0) else np.nan
    cross_price_ma50 = _find_last_cross(df["Close"], df["MA50"], lookback=20)
    cross_price_ma200 = _find_last_cross(df["Close"], df["MA200"], lookback=60)
    cross_ma20_ma50 = _find_last_cross(df["MA20"], df["MA50"], lookback=60)
    cross_ma50_ma200 = _find_last_cross(df["MA50"], df["MA200"], lookback=120)
    
    return {
        "Regime": structure_snapshot,
        "SlopeMA20": slope_label(s20_v),
        "SlopeMA50": slope_label(s50_v),
        "SlopeMA200": slope_label(s200_v),
        "SlopeMA20Value": s20_v,
        "SlopeMA50Value": s50_v,
        "SlopeMA200Value": s200_v,
        "DistToMA50Pct": dist50,
        "DistToMA200Pct": dist200,
        "Cross": {
            "PriceVsMA50": cross_price_ma50,
            "PriceVsMA200": cross_price_ma200,
            "MA20VsMA50": cross_ma20_ma50,
            "MA50VsMA200": cross_ma50_ma200
        },
        "Structure": {
            "PriceAboveMA50": bool(pd.notna(close) and pd.notna(ma50) and close >= ma50),
            "PriceAboveMA200": bool(pd.notna(close) and pd.notna(ma200) and close >= ma200),
            "MA20AboveMA50": bool(pd.notna(ma20) and pd.notna(ma50) and ma20 >= ma50),
            "MA50AboveMA200": bool(pd.notna(ma50) and pd.notna(ma200) and ma50 >= ma200),
        }
    }

def compute_rsi_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    rsi = df["RSI"].dropna()
    if rsi.empty: return {}
    last_rsi = _safe_float(rsi.iloc[-1])
    prev5 = _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else (_safe_float(rsi.iloc[0]) if len(rsi) >= 1 else np.nan)
    direction = "N/A"
    if pd.notna(last_rsi) and pd.notna(prev5):
        delta5 = last_rsi - prev5
        if delta5 > 1.0: direction = "Rising"
        elif delta5 < -1.0: direction = "Falling"
        else: direction = "Flat"
    else:
        delta5 = np.nan
    zone = "N/A"
    if pd.notna(last_rsi):
        if last_rsi >= 70: zone = "Zone70Plus"
        elif last_rsi >= 60: zone = "Zone60_70"
        elif last_rsi >= 50: zone = "Zone50_60"
        elif last_rsi >= 40: zone = "Zone40_50"
        elif last_rsi >= 30: zone = "Zone30_40"
        else: zone = "ZoneBelow30"
    tail6 = rsi.tail(6).tolist()
    tail6 = [(_safe_float(x) if pd.notna(x) else np.nan) for x in tail6]
    tail20 = rsi.tail(20)
    rsi_max20 = _safe_float(tail20.max()) if not tail20.empty else np.nan
    rsi_min20 = _safe_float(tail20.min()) if not tail20.empty else np.nan
    def _streak(cond_series: pd.Series) -> int:
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    streak_above70 = _streak((rsi >= 70).tail(60))
    streak_below30 = _streak((rsi <= 30).tail(60))
    div = _detect_divergence_simple(df["Close"], df["RSI"], lookback=60)
    return {
        "Value": last_rsi,
        "State": zone,
        "Direction": direction,
        "Divergence": div,
        "Delta5": delta5,
        "RSI_Series_6": tail6,
        "Max20": rsi_max20,
        "Min20": rsi_min20,
        "StreakAbove70": streak_above70,
        "StreakBelow30": streak_below30
    }

def compute_macd_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    macd_v = df["MACD"].dropna()
    sig = df["MACDSignal"].dropna()
    hist = df["MACDHist"].dropna()
    if macd_v.empty or sig.empty: return {}
    last_m = _safe_float(macd_v.iloc[-1])
    last_s = _safe_float(sig.iloc[-1])
    last_h = _safe_float(hist.iloc[-1]) if not hist.empty else np.nan
    state = "N/A"
    diff_ms = np.nan
    if pd.notna(last_m) and pd.notna(last_s):
        diff_ms = last_m - last_s
        if diff_ms > 0: state = "MACD_Above_Signal"
        elif diff_ms < 0: state = "MACD_Below_Signal"
        else: state = "MACD_Near_Signal"
    cross = _find_last_cross(df["MACD"], df["MACDSignal"], lookback=30)
    zero = "N/A"
    if pd.notna(last_m):
        if last_m > 0.0: zero = "Above"
        elif last_m < 0.0: zero = "Below"
        else: zero = "Near"
    h_tail6 = []
    if not hist.empty:
        h_tail6 = hist.tail(6).tolist()
        h_tail6 = [(_safe_float(x) if pd.notna(x) else np.nan) for x in h_tail6]
    hist_state = "N/A"
    if len(hist) >= 4:
        h0 = _safe_float(hist.iloc[-1])
        h1 = _safe_float(hist.iloc[-2])
        h2 = _safe_float(hist.iloc[-3])
        if pd.notna(h0) and pd.notna(h1) and pd.notna(h2):
            if h0 >= 0 and h1 >= 0:
                if (h0 > h1 > h2): hist_state = "Expanding_Positive"
                elif (h0 < h1 < h2): hist_state = "Contracting_Positive"
                else: hist_state = "Mixed_Positive"
            elif h0 < 0 and h1 < 0:
                if (h0 < h1 < h2): hist_state = "Expanding_Negative"
                elif (h0 > h1 > h2): hist_state = "Contracting_Negative"
                else: hist_state = "Mixed_Negative"
            else:
                hist_state = "Sign_Flip"
    def _streak(cond_series: pd.Series) -> int:
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    streak_pos = _streak((hist >= 0).tail(60)) if not hist.empty else 0
    streak_neg = _streak((hist < 0).tail(60)) if not hist.empty else 0
    delta5 = np.nan
    if len(macd_v) >= 6:
        delta5 = _safe_float(macd_v.iloc[-1] - macd_v.iloc[-6])
    div = _detect_divergence_simple(df["Close"], df["MACD"], lookback=60)
    return {
        "Value": last_m,
        "Signal": last_s,
        "Hist": last_h,
        "State": state,
        "Cross": cross,
        "ZeroLine": zero,
        "HistState": hist_state,
        "Divergence": div,
        "Diff_MACD_Signal": diff_ms,
        "MACD_Delta5": delta5,
        "Hist_Series_6": h_tail6,
        "Hist_Streak_Pos": streak_pos,
        "Hist_Streak_Neg": streak_neg
    }

def compute_volume_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    ratio = (vol / avg) if (pd.notna(vol) and pd.notna(avg) and avg != 0) else np.nan
    regime = "N/A"
    if pd.notna(ratio):
        if ratio >= 1.8: regime = "Spike"
        elif ratio >= 1.2: regime = "High"
        elif ratio >= 0.8: regime = "Normal"
        else: regime = "Low"
    return {"Vol": vol, "Avg20Vol": avg, "Ratio": ratio, "Regime": regime}

# --- STEP 6B: RSI+MACD BIAS (UPDATED) ---
def compute_rsi_macd_bias_features(rsi_feat: Dict[str, Any], macd_feat: Dict[str, Any]) -> Dict[str, Any]:
    if not rsi_feat or not macd_feat:
        return {"BiasCode": "N/A", "Alignment": "N/A", "Tags": ["MissingData"], "Facts": {}, "Notes": []}
    
    rsi_v = _safe_float(rsi_feat.get("Value"))
    rsi_zone = rsi_feat.get("State", "N/A")
    rsi_dir = rsi_feat.get("Direction", "N/A")
    rsi_div = rsi_feat.get("Divergence", {})
    rsi_div_type = rsi_div.get("Type", "None") if rsi_div else "None"
    
    macd_rel = macd_feat.get("State", "N/A")
    zero = macd_feat.get("ZeroLine", "N/A")
    hist_state = macd_feat.get("HistState", "N/A")
    cross = macd_feat.get("Cross", {})
    cross_event = cross.get("Event", "None") if cross else "None"
    cross_bars = cross.get("BarsAgo") if cross else None
    
    # 6B: Neutralized Bias Logic (Condition-based)
    alignment = "Mixed"
    rsi_pos = (rsi_zone in ["Zone60_70", "Zone70Plus"]) or (rsi_dir == "Rising" and rsi_v > 50)
    rsi_neg = (rsi_zone in ["Zone30_40", "ZoneBelow30"]) or (rsi_dir == "Falling" and rsi_v < 50)
    
    if rsi_zone == "Zone70Plus": alignment = "RSI_70Plus"
    elif rsi_zone == "ZoneBelow30": alignment = "RSI_Below30"
    else:
        # Check alignment in middle zones
        if rsi_pos and macd_rel == "MACD_Above_Signal":
            alignment = "Aligned_Positive"
        elif rsi_neg and macd_rel == "MACD_Below_Signal":
            alignment = "Aligned_Negative"
        else:
            alignment = "Mixed"
            
    tags: List[str] = [
        rsi_zone,
        macd_rel,
        f"MACD_ZeroLine={zero}",
        f"MACD_HistState={hist_state}",
        f"MACD_Cross={cross_event}",
        f"RSI_Direction={rsi_dir}",
        f"RSI_Divergence={rsi_div_type}",
        f"Alignment={alignment}",
    ]
    
    bias_code = "__".join([
        rsi_zone,
        macd_rel,
        f"Zero={zero}",
        f"Hist={hist_state}",
        f"Cross={cross_event}",
        f"Align={alignment}",
    ])
    
    notes: List[str] = []
    notes.append("Bias mô tả bằng điều kiện (facts), không kết luận tốt/xấu.")
    if cross_event != "None" and cross_bars is not None:
        notes.append(f"MACD_CrossEvent={cross_event}; BarsAgo={cross_bars}")
        
    return {
        "BiasCode": bias_code,
        "Alignment": alignment,
        "Tags": tags,
        "Facts": {
            "RSIZone": rsi_zone,
            "RSIValue": rsi_v if pd.notna(rsi_v) else np.nan,
            "RSIDirection": rsi_dir,
            "RSIDivergenceType": rsi_div_type,
            "MACDRelation": macd_rel,
            "MACDZeroLine": zero,
            "MACDHistState": hist_state,
            "MACDCrossEvent": cross_event,
            "MACDCrossBarsAgo": cross_bars
        },
        "Notes": notes
    }

# --- STEP 6: PRICE ACTION & PATTERNS ---
def _pct_dist(a: float, b: float, base: float) -> float:
    if pd.isna(a) or pd.isna(b) or pd.isna(base) or base == 0:
        return np.nan
    return abs(a - b) / abs(base) * 100

def _range_percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty or pd.isna(value): return np.nan
    return float((s <= value).mean() * 100)

def compute_price_action_features(df: pd.DataFrame, fib_ctx: Optional[Dict[str, Any]] = None, vol_feat: Optional[Dict[str, Any]] = None, tol_pct: float = 0.8) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    o = _safe_float(last.get("Open"), _safe_float(last.get("Close")))
    h = _safe_float(last.get("High"))
    l = _safe_float(last.get("Low"))
    c = _safe_float(last.get("Close"))
    if pd.isna(h) or pd.isna(l) or pd.isna(c): return {"Error": "Missing OHLC"}
    rng = h - l
    body = abs(c - o) if (pd.notna(o) and pd.notna(c)) else np.nan
    upper = (h - max(o, c)) if (pd.notna(o) and pd.notna(c)) else np.nan
    lower = (min(o, c) - l) if (pd.notna(o) and pd.notna(c)) else np.nan
    body_pct = (body / rng * 100) if (pd.notna(body) and rng > 0) else np.nan
    upper_pct = (upper / rng * 100) if (pd.notna(upper) and rng > 0) else np.nan
    lower_pct = (lower / rng * 100) if (pd.notna(lower) and rng > 0) else np.nan
    range_pct_close = (rng / c * 100) if (pd.notna(c) and c != 0 and rng >= 0) else np.nan
    direction = "N/A"
    if pd.notna(o) and pd.notna(c):
        direction = "Bull" if c > o else ("Bear" if c < o else "Flat")
    gap_pct = np.nan
    if prev is not None:
        pc = _safe_float(prev.get("Close"))
        if pd.notna(o) and pd.notna(pc) and pc != 0:
            gap_pct = (o - pc) / pc * 100
    ranges = (df["High"] - df["Low"]).tail(60)
    range_pctl_60 = _range_percentile(ranges, rng)
    doji = bool(pd.notna(body_pct) and body_pct <= 10)
    close_from_high_pct = ((h - c) / rng * 100) if (rng > 0 and pd.notna(c)) else np.nan
    close_from_low_pct = ((c - l) / rng * 100) if (rng > 0 and pd.notna(c)) else np.nan
    hammer = False
    shooting_star = False
    if (rng > 0) and pd.notna(body) and pd.notna(upper) and pd.notna(lower):
        hammer = (lower >= 2.0 * body) and (upper <= 0.6 * body) and (pd.notna(close_from_high_pct) and close_from_high_pct <= 30)
        shooting_star = (upper >= 2.0 * body) and (lower <= 0.6 * body) and (pd.notna(close_from_low_pct) and close_from_low_pct <= 30)
    bullish_engulf = False
    bearish_engulf = False
    inside_bar = False
    outside_bar = False
    if prev is not None:
        po = _safe_float(prev.get("Open"), _safe_float(prev.get("Close")))
        ph = _safe_float(prev.get("High"))
        pl = _safe_float(prev.get("Low"))
        pc = _safe_float(prev.get("Close"))
        if pd.notna(ph) and pd.notna(pl):
            inside_bar = bool(h < ph and l > pl)
            outside_bar = bool(h > ph and l < pl)
        if pd.notna(po) and pd.notna(pc) and pd.notna(o) and pd.notna(c):
            prev_bear = pc < po
            prev_bull = pc > po
            last_bull = c > o
            last_bear = c < o
            bullish_engulf = bool(last_bull and prev_bear and (c >= po) and (o <= pc))
            bearish_engulf = bool(last_bear and prev_bull and (o >= pc) and (c <= po))
    patterns = {
        "Doji": doji, "Hammer": hammer, "ShootingStar": shooting_star,
        "BullishEngulfing": bullish_engulf, "BearishEngulfing": bearish_engulf,
        "InsideBar": inside_bar, "OutsideBar": outside_bar,
    }
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    near_ma = {
        "MA20": {"Value": ma20, "DistPct": _pct_dist(c, ma20, c), "Near": bool(pd.notna(ma20) and _pct_dist(c, ma20, c) <= tol_pct)},
        "MA50": {"Value": ma50, "DistPct": _pct_dist(c, ma50, c), "Near": bool(pd.notna(ma50) and _pct_dist(c, ma50, c) <= tol_pct)},
        "MA200": {"Value": ma200, "DistPct": _pct_dist(c, ma200, c), "Near": bool(pd.notna(ma200) and _pct_dist(c, ma200, c) <= tol_pct)},
    }
    near_fib_short = []
    near_fib_long = []
    if fib_ctx:
        for side in ["Supports", "Resistances"]:
            for it in (fib_ctx.get("NearestShort", {}) or {}).get(side, []) or []:
                d = _safe_float(it.get("DistPct"))
                if pd.notna(d) and abs(d) <= tol_pct:
                    near_fib_short.append({"Side": side, **it})
            for it in (fib_ctx.get("NearestLong", {}) or {}).get(side, []) or []:
                d = _safe_float(it.get("DistPct"))
                if pd.notna(d) and abs(d) <= tol_pct:
                    near_fib_long.append({"Side": side, **it})
    vol_regime = (vol_feat.get("Regime") if vol_feat else "N/A")
    notes = []
    notes.append(f"Dir={direction}, BodyPct={body_pct}, WickU={upper_pct}, WickL={lower_pct}, RangePct={range_pct_close}")
    if pd.notna(range_pctl_60): notes.append(f"RangePercentile60={range_pctl_60}")
    if pd.notna(gap_pct): notes.append(f"GapPct={gap_pct}")
    notes.append(f"VolRegime={vol_regime}")
    return {
        "Candle": {
            "Open": o, "High": h, "Low": l, "Close": c, "Direction": direction,
            "BodyPct": body_pct, "UpperWickPct": upper_pct, "LowerWickPct": lower_pct,
            "RangePctOfClose": range_pct_close, "GapPct": gap_pct,
            "RangePercentile60": range_pctl_60, "CloseFromHighPct": close_from_high_pct, "CloseFromLowPct": close_from_low_pct,
        },
        "Patterns": patterns,
        "Context": {
            "NearMA": near_ma, "NearFibShort": near_fib_short, "NearFibLong": near_fib_long, "VolumeRegime": vol_regime,
            "FiboConflictFlag": bool(fib_ctx.get("FiboConflictFlag")) if fib_ctx else False,
            "FiboPriorityRuleApplied": (fib_ctx.get("FiboPriorityRuleApplied") if fib_ctx else "None"),
        },
        "Notes": notes
    }

# --- STEP 7B: NEW CONTEXT FEATURES ---
def compute_rsi_context_features(df: pd.DataFrame, rsi_col: str = "RSI") -> Dict[str, Any]:
    if df.empty or rsi_col not in df.columns:
        return {"Streak70": 0, "Cross70BarsAgo": None, "Delta3": np.nan, "Delta5": np.nan}
    
    rsi = df[rsi_col].dropna()
    if rsi.empty: return {}
    
    last_r = _safe_float(rsi.iloc[-1])
    
    def _streak(cond_series):
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    
    streak70 = _streak((rsi >= 70).tail(60))
    
    # Cross 70 up check
    # We look back to find when it crossed 70 from below
    cross_70_idx = None
    vals = rsi.values
    for i in range(len(vals)-2, -1, -1):
        if vals[i] < 70 and vals[i+1] >= 70:
            cross_70_idx = len(vals) - 1 - (i+1)
            break
            
    delta3 = last_r - _safe_float(rsi.iloc[-4]) if len(rsi) >= 4 else np.nan
    delta5 = last_r - _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else np.nan
    
    return {
        "Streak70": streak70,
        "Cross70BarsAgo": int(cross_70_idx) if cross_70_idx is not None else None,
        "Delta3": delta3,
        "Delta5": delta5,
        "Turning": "Falling" if pd.notna(delta3) and delta3 < -2 else ("Rising" if pd.notna(delta3) and delta3 > 2 else "Flat")
    }

def compute_volume_context_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "Volume" not in df.columns:
        return {"VolStreakUp": 0, "VolTrend": "N/A"}
        
    vol = df["Volume"].dropna()
    if len(vol) < 5: return {"VolStreakUp": 0, "VolTrend": "N/A"}
    
    # Streak of rising volume
    cnt = 0
    vals = vol.values
    for i in range(len(vals)-1, 0, -1):
        if vals[i] > vals[i-1]: cnt += 1
        else: break
        
    # Simple slope of Vol MA
    avg20 = df["Avg20Vol"] if "Avg20Vol" in df.columns else sma(vol, 20)
    slope = "Flat"
    if len(avg20) >= 5:
        a = avg20.dropna()
        if len(a) >= 5:
            delta = a.iloc[-1] - a.iloc[-5]
            if delta > 0: slope = "Rising"
            elif delta < 0: slope = "Falling"
            
    return {"VolStreakUp": int(cnt), "VolTrend": slope}

def compute_level_context_features(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    if pd.isna(c): return {}
    
    # Merge levels
    levels = {}
    levels.update(dual_fib.get("auto_short", {}).get("levels", {}))
    levels.update(dual_fib.get("fixed_long", {}).get("levels", {}))
    
    sorted_lv = sorted([(k, v) for k,v in levels.items() if pd.notna(v)], key=lambda x: x[1])
    
    sup = [(k,v, (v-c)/c*100) for k,v in sorted_lv if v <= c]
    res = [(k,v, (v-c)/c*100) for k,v in sorted_lv if v >= c]
    
    sup = sorted(sup, key=lambda x: abs(x[2]))[:1] # Nearest
    res = sorted(res, key=lambda x: abs(x[2]))[:1] # Nearest
    
    sup_pack = {"Label": sup[0][0], "Level": sup[0][1], "DistPct": sup[0][2]} if sup else {"Label": "N/A", "Level": np.nan, "DistPct": np.nan}
    res_pack = {"Label": res[0][0], "Level": res[0][1], "DistPct": res[0][2]} if res else {"Label": "N/A", "Level": np.nan, "DistPct": np.nan}
    
    return {
        "NearestSupport": sup_pack,
        "NearestResistance": res_pack
    }

# ============================================================
# 7C. STRUCTURE QUALITY PACK (Support/Resistance Quality-Aware)
# ============================================================
def compute_structure_quality_pack(
    df: pd.DataFrame,
    last: pd.Series,
    dual_fib: Optional[Dict[str, Any]] = None,
    fib_ctx: Optional[Dict[str, Any]] = None,
    *,
    daily_lookback: int = 60,
    weekly_lookback_weeks: int = 78
) -> Dict[str, Any]:
    """
    Computes a quality-aware Support/Resistance pack to prevent structural blind spots:
    - Distinguishes Tactical (60–90D/daily) vs Structural (~250D/weekly) levels
    - Distinguishes strength tiers (LIGHT/MED/HEAVY/CONFLUENCE)
    - Provides a CeilingGate for bull-trap control (breakout over weak tactical vs structural ceiling)

    Output is facts-only; GPT/renderer should only interpret.
    """

    dual_fib = dual_fib if isinstance(dual_fib, dict) else {}
    fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}

    c = _safe_float(last.get("Close"))
    if pd.isna(c) or c == 0:
        return {
            "Meta": {"Price": np.nan, "VolPct_ATRProxy": np.nan, "NearThresholdPct": np.nan, "ZoneWidthPct": np.nan},
            "OverheadResistance": {"Nearest": {}},
            "UnderlyingSupport": {"Nearest": {}},
            "Gates": {"CeilingGate": {"Status": "N/A", "Reason": "Missing price", "Horizon": "N/A", "Tier": "N/A",
                                      "NearestLevelPrice": None, "DistancePct": None}}
        }

    denom = _dynamic_vol_proxy(df, 20)
    vol_pct = (denom / c * 100.0) if (pd.notna(denom) and pd.notna(c) and c != 0) else np.nan

    near_th = float(_clip(max(1.0, 0.8 * vol_pct) if pd.notna(vol_pct) else 1.5, 1.0, 2.5))
    zone_w = float(_clip(max(0.8, 0.6 * vol_pct) if pd.notna(vol_pct) else 1.0, 0.8, 2.0))

    def _cand(type_: str, horizon: str, price: float, weight: float) -> Dict[str, Any]:
        dist_pct = ((price - c) / c * 100.0) if (pd.notna(price) and pd.notna(c) and c != 0) else np.nan
        return {"Type": type_, "Horizon": horizon, "Price": float(price), "Weight": float(weight), "DistPct": float(dist_pct)}

    W = {
        "MA200": 4.0,
        "WEEKLY_SWING": 4.0,
        "FIB_LONG_61.8": 3.5,
        "FIB_LONG_50.0": 2.5,
        "FIB_LONG_38.2": 2.0,
        "MA50": 3.0,
        "DAILY_SWING": 2.5,
        "FIB_SHORT_61.8": 2.5,
        "FIB_SHORT_50.0": 2.0,
        "FIB_SHORT_38.2": 1.5,
        "MA20": 1.5,
        "FIB_EXT": 1.0,
    }

    candidates: List[Dict[str, Any]] = []

    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    if pd.notna(ma20) and ma20 > 0:
        candidates.append(_cand("MA20", "TACTICAL", ma20, W["MA20"]))
    if pd.notna(ma50) and ma50 > 0:
        candidates.append(_cand("MA50", "STRUCTURAL", ma50, W["MA50"]))
    if pd.notna(ma200) and ma200 > 0:
        candidates.append(_cand("MA200", "STRUCTURAL", ma200, W["MA200"]))

    short_lv = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    long_lv = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}

    def _add_fib(levels: Dict[str, Any], prefix: str, horizon: str):
        for k, v in (levels or {}).items():
            lv = _safe_float(v)
            if pd.isna(lv) or lv <= 0:
                continue
            kk = str(k).strip()
            if kk in ("61.8", "61.80"):
                typ = f"{prefix}_61.8"; w = W.get(f"FIB_{prefix}_61.8", 2.0)
            elif kk in ("50.0", "50", "50.00"):
                typ = f"{prefix}_50.0"; w = W.get(f"FIB_{prefix}_50.0", 1.5)
            elif kk in ("38.2", "38.20"):
                typ = f"{prefix}_38.2"; w = W.get(f"FIB_{prefix}_38.2", 1.2)
            else:
                typ = f"{prefix}_EXT"; w = W.get("FIB_EXT", 1.0)
            # Type example: FIB_SHORT_61.8 / FIB_LONG_61.8
            candidates.append(_cand(f"FIB_{typ}", horizon, lv, float(w)))

    _add_fib(short_lv, "SHORT", "TACTICAL")
    _add_fib(long_lv, "LONG", "STRUCTURAL")

    def _extract_daily_pivots(_df: pd.DataFrame, lb: int = 60) -> Tuple[List[float], List[float]]:
        if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty:
            return ([], [])
        if "High" not in _df.columns or "Low" not in _df.columns:
            return ([], [])
        d = _df.tail(int(lb)).copy()
        hi = d["High"]
        lo = d["Low"]
        piv_hi, piv_lo = [], []
        for i in range(1, len(d) - 1):
            h0, h1, h2 = _safe_float(hi.iloc[i-1]), _safe_float(hi.iloc[i]), _safe_float(hi.iloc[i+1])
            l0, l1, l2 = _safe_float(lo.iloc[i-1]), _safe_float(lo.iloc[i]), _safe_float(lo.iloc[i+1])
            if pd.notna(h1) and pd.notna(h0) and pd.notna(h2) and h1 > h0 and h1 > h2:
                piv_hi.append(float(h1))
            if pd.notna(l1) and pd.notna(l0) and pd.notna(l2) and l1 < l0 and l1 < l2:
                piv_lo.append(float(l1))
        return (piv_hi, piv_lo)

    dh, dl = _extract_daily_pivots(df, daily_lookback)
    for p in dh:
        candidates.append(_cand("DAILY_SWING_HIGH", "TACTICAL", p, W["DAILY_SWING"]))
    for p in dl:
        candidates.append(_cand("DAILY_SWING_LOW", "TACTICAL", p, W["DAILY_SWING"]))

    def _extract_weekly_pivots(_df: pd.DataFrame, lb_weeks: int = 78) -> Tuple[List[float], List[float]]:
        if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty:
            return ([], [])
        if "High" not in _df.columns or "Low" not in _df.columns:
            return ([], [])
        d = _df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            for col in ("Date", "Datetime", "date", "datetime"):
                if col in d.columns:
                    try:
                        d[col] = pd.to_datetime(d[col], errors="coerce")
                        d = d.set_index(col)
                        break
                    except Exception:
                        pass
        if not isinstance(d.index, pd.DatetimeIndex):
            return ([], [])
        d = d.sort_index()
        w = d[["High", "Low"]].resample("W").agg({"High": "max", "Low": "min"}).dropna()
        if w.empty:
            return ([], [])
        w = w.tail(int(lb_weeks)).copy()
        wh = w["High"]
        wl = w["Low"]
        piv_hi, piv_lo = [], []
        for i in range(1, len(w) - 1):
            h0, h1, h2 = _safe_float(wh.iloc[i-1]), _safe_float(wh.iloc[i]), _safe_float(wh.iloc[i+1])
            l0, l1, l2 = _safe_float(wl.iloc[i-1]), _safe_float(wl.iloc[i]), _safe_float(wl.iloc[i+1])
            if pd.notna(h1) and pd.notna(h0) and pd.notna(h2) and h1 > h0 and h1 > h2:
                piv_hi.append(float(h1))
            if pd.notna(l1) and pd.notna(l0) and pd.notna(l2) and l1 < l0 and l1 < l2:
                piv_lo.append(float(l1))
        return (piv_hi, piv_lo)

    wh, wl = _extract_weekly_pivots(df, weekly_lookback_weeks)
    for p in wh:
        candidates.append(_cand("WEEKLY_SWING_HIGH", "STRUCTURAL", p, W["WEEKLY_SWING"]))
    for p in wl:
        candidates.append(_cand("WEEKLY_SWING_LOW", "STRUCTURAL", p, W["WEEKLY_SWING"]))

    def _cluster_nearest(side: str) -> Dict[str, Any]:
        if side == "overhead":
            pool = [x for x in candidates if pd.notna(x.get("Price")) and x["Price"] >= c]
            pool = sorted(pool, key=lambda x: x["Price"])
        else:
            pool = [x for x in candidates if pd.notna(x.get("Price")) and x["Price"] <= c]
            pool = sorted(pool, key=lambda x: x["Price"], reverse=True)

        if not pool:
            return {}

        clusters: List[List[Dict[str, Any]]] = []
        for it in pool:
            placed = False
            for cl in clusters:
                center = float(np.mean([z["Price"] for z in cl]))
                if abs(it["Price"] - center) / c * 100.0 <= zone_w:
                    cl.append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([it])

        def _center_dist(cl: List[Dict[str, Any]]) -> float:
            center = float(np.mean([z["Price"] for z in cl]))
            return abs((center - c) / c * 100.0)

        clusters = sorted(clusters, key=_center_dist)
        chosen = clusters[0]
        center = float(np.mean([z["Price"] for z in chosen]))
        dist_pct = abs((center - c) / c * 100.0)

        horizons = {str(z.get("Horizon", "N/A")).upper() for z in chosen}
        has_struct = "STRUCTURAL" in horizons
        has_tact = "TACTICAL" in horizons
        horizon = "BOTH" if (has_struct and has_tact) else ("STRUCTURAL" if has_struct else "TACTICAL")

        confluence_count = int(len(chosen))
        confluence_mult = float(_clip(1.0 + 0.25 * max(0, confluence_count - 1), 1.0, 1.75))

        if dist_pct <= near_th:
            near_factor = 1.0
        else:
            near_factor = max(0.6, near_th / dist_pct) if dist_pct > 0 else 1.0

        raw = sum(float(z.get("Weight", 0.0)) for z in chosen) * confluence_mult * float(near_factor)
        quality = float(_clip(raw * 1.2, 0.0, 10.0))

        if quality >= 8.5 or (confluence_count >= 3 and has_struct):
            tier = "CONFLUENCE"
        elif quality >= 6.5:
            tier = "HEAVY"
        elif quality >= 4.0:
            tier = "MED"
        else:
            tier = "LIGHT"

        low = float(min(z["Price"] for z in chosen))
        high = float(max(z["Price"] for z in chosen))
        comps = sorted(chosen, key=lambda x: (-float(x.get("Weight", 0.0)), abs(float(x.get("DistPct", 0.0)))))[:3]

        return {
            "Zone": {"Low": low, "High": high},
            "Center": center,
            "DistancePct": float(dist_pct),
            "Horizon": horizon,
            "Tier": tier,
            "QualityScore": quality,
            "ComponentsTop": comps,
            "ConfluenceCount": confluence_count
        }

    overhead = _cluster_nearest("overhead")
    support = _cluster_nearest("support")

    gate = {"Status": "N/A", "Reason": "N/A", "Horizon": "N/A", "Tier": "N/A", "NearestLevelPrice": None, "DistancePct": None}
    if overhead:
        dist = _safe_float(overhead.get("DistancePct"), default=np.nan)
        tier = str(overhead.get("Tier", "N/A")).upper()
        hz = str(overhead.get("Horizon", "N/A")).upper()
        gate.update({"Horizon": hz, "Tier": tier, "NearestLevelPrice": float(overhead.get("Center", np.nan)), "DistancePct": float(dist)})

        if pd.isna(dist) or dist > near_th:
            gate.update({"Status": "PASS", "Reason": "No near ceiling"})
        else:
            if hz in ("STRUCTURAL", "BOTH") and tier in ("HEAVY", "CONFLUENCE"):
                status = "WAIT"
                reason = "Structural ceiling near"
                if hz == "BOTH" and tier == "CONFLUENCE" and dist <= 0.5 * near_th:
                    status = "FAIL"
                    reason = "Confluence ceiling too close"
                gate.update({"Status": status, "Reason": reason})
            else:
                gate.update({"Status": "PASS", "Reason": "Ceiling manageable"})

    return {
        "Meta": {
            "Price": float(c),
            "VolPct_ATRProxy": float(vol_pct) if pd.notna(vol_pct) else np.nan,
            "NearThresholdPct": float(near_th),
            "ZoneWidthPct": float(zone_w),
        },
        "OverheadResistance": {"Nearest": overhead or {}},
        "UnderlyingSupport": {"Nearest": support or {}},
        "Gates": {"CeilingGate": gate}
    }

def compute_market_context(df_all: pd.DataFrame) -> Dict[str, Any]:
    def pack(tick: str) -> Dict[str, Any]:
        d = df_all[df_all["Ticker"].astype(str).str.upper() == tick].copy()
        if d.empty or len(d) < 2: return {"Ticker": tick, "Close": np.nan, "ChangePct": np.nan, "Regime": "N/A"}
        d = d.sort_values("Date")
        c = _safe_float(d.iloc[-1].get("Close"))
        p = _safe_float(d.iloc[-2].get("Close"))
        chg = _pct_change(c, p)
        regime = "N/A"
        try:
            d["MA50"] = sma(d["Close"], 50)
            d["MA200"] = sma(d["Close"], 200)
            ma50 = _safe_float(d.iloc[-1].get("MA50"))
            ma200 = _safe_float(d.iloc[-1].get("MA200"))
            if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
                if c >= ma50 and ma50 >= ma200: regime = "Up"
                elif c < ma50 and ma50 < ma200: regime = "Down"
                else: regime = "Neutral"
        except: regime = "N/A"
        return {"Ticker": tick, "Close": c, "ChangePct": chg, "Regime": regime}
    vnindex = pack("VNINDEX")
    vn30 = pack("VN30")
    return {"VNINDEX": vnindex, "VN30": vn30}

# ============================================================


def build_base_result_v1(
    ticker: str,
    df_all: pd.DataFrame,
    *,
    hsc_targets: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build the base (pre-modules) analysis result.

    Design goals:
    - Deterministic + JSON-safe packs.
    - Mirrors the stable v14.6 contracts so downstream UI/renderers do not drift.
    - Does NOT attach TradePlanPack/DecisionPack; those are produced by the Character module
      after PositionState injection (Step 11/12 refactor).
    """

    tick = str(ticker or "").strip().upper()
    if not tick:
        return {"Error": "Missing ticker"}

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        return {"Error": "Empty price/volume dataset"}

    df = df_all[df_all["Ticker"].astype(str).str.upper() == tick].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {tick}"}

    # Ensure chronological order
    if "Date" in df.columns:
        try:
            df = df.sort_values("Date")
        except Exception:
            pass

    # --- Core indicators ---
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h

    dual_fib = compute_dual_fibonacci_auto(df, long_window=250)
    last = df.iloc[-1]

    # --- Context packs ---
    fib_ctx = compute_fibonacci_context_pack(last, dual_fib)
    conviction = compute_conviction(last)
    conviction_pack = compute_conviction_pack(last)
    scenario = classify_scenario(last)
    scenario12 = classify_scenario12(last)

    # --- Trade plans & RR ---
    trade_plans = build_trade_plan(df, dual_fib)
    rrsim = build_rr_sim(trade_plans)
    primary = pick_primary_setup_v3(rrsim)

    # --- Master score ---
    master = compute_master_score(last, dual_fib, trade_plans)

    # --- Pro-tech features ---
    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    bias_feat = compute_rsi_macd_bias_features(rsi_feat, macd_feat)
    pa_feat = compute_price_action_features(df, fib_ctx=fib_ctx, vol_feat=vol_feat, tol_pct=0.8)
    rsi_ctx = compute_rsi_context_features(df)
    vol_ctx = compute_volume_context_features(df)
    lvl_ctx = compute_level_context_features(last, dual_fib)

    # --- Structure quality ---
    try:
        struct_q = compute_structure_quality_pack(df, last, dual_fib=dual_fib, fib_ctx=fib_ctx)
    except Exception:
        struct_q = {}

    # --- Market context & relative strength ---
    market_ctx = compute_market_context(df_all)
    stock_chg = np.nan
    try:
        if len(df) >= 2:
            stock_chg = _pct_change(_safe_float(df.iloc[-1].get("Close")), _safe_float(df.iloc[-2].get("Close")))
    except Exception:
        stock_chg = np.nan

    mkt_chg = _safe_float((market_ctx.get("VNINDEX") or {}).get("ChangePct"))
    rel = "N/A"
    try:
        if pd.notna(stock_chg) and pd.notna(mkt_chg):
            if stock_chg > mkt_chg + 0.3:
                rel = "Stronger"
            elif stock_chg < mkt_chg - 0.3:
                rel = "Weaker"
            else:
                rel = "InLine"
    except Exception:
        rel = "N/A"

    # --- Fundamental (pass-through) ---
    hsc = hsc_targets if isinstance(hsc_targets, pd.DataFrame) else pd.DataFrame()
    fund_row: Dict[str, Any] = {}
    try:
        if not hsc.empty and "Ticker" in hsc.columns:
            fund = hsc[hsc["Ticker"].astype(str).str.upper() == tick]
            fund_row = fund.iloc[0].to_dict() if not fund.empty else {}
    except Exception:
        fund_row = {}

    close = _safe_float(last.get("Close"))
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)

    # Handle mixed units (VND vs thousand VND) like v14.6
    close_for_calc = close
    target_for_calc = target_vnd
    if pd.notna(close) and pd.notna(target_vnd):
        if (close < 500) and (target_vnd > 1000):
            target_for_calc = target_vnd / 1000.0
        elif (close > 1000) and (target_vnd < 500):
            target_for_calc = target_vnd * 1000.0

    upside_pct = (
        (target_for_calc - close_for_calc) / close_for_calc * 100
        if (pd.notna(target_for_calc) and pd.notna(close_for_calc) and close_for_calc != 0)
        else np.nan
    )

    # Keep the raw row but add convenience fields (mirrors v14.6)
    if pd.notna(target_vnd):
        fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct
    fund_row["TargetK"] = (target_vnd / 1000.0) if pd.notna(target_vnd) else np.nan

    analysis_pack: Dict[str, Any] = {
        "_schema_version": "1.0",
        "Ticker": tick,
        "Last": {
            "Close": _safe_float(last.get("Close")),
            "MA20": _safe_float(last.get("MA20")),
            "MA50": _safe_float(last.get("MA50")),
            "MA200": _safe_float(last.get("MA200")),
            "RSI": _safe_float(last.get("RSI")),
            "MACD": _safe_float(last.get("MACD")),
            "MACDSignal": _safe_float(last.get("MACDSignal")),
            "MACDHist": _safe_float(last.get("MACDHist")),
            "Volume": _safe_float(last.get("Volume")),
            "Avg20Vol": _safe_float(last.get("Avg20Vol")),
        },
        "ScenarioBase": scenario,
        "Scenario12": scenario12,
        "Conviction": conviction,
        "ConvictionPack": conviction_pack,
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {}),
            "AltShort": dual_fib.get("alt_short", {}),
            "SelectionReason": dual_fib.get("selection_reason", "N/A"),
            "Context": fib_ctx,
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": fund_row.get("TargetK", np.nan),
            "UpsidePct": upside_pct,
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(getattr(v, "entry", np.nan)),
                "Stop": _safe_float(getattr(v, "stop", np.nan)),
                "TP": _safe_float(getattr(v, "tp", np.nan)),
                "RR": _safe_float(getattr(v, "rr", np.nan)),
                "Confidence (Tech)": getattr(v, "probability", "N/A"),
                "Status": getattr(v, "status", "Watch"),
                "ReasonTags": list(getattr(v, "reason_tags", []) or []),
            }
            for k, v in (trade_plans or {}).items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "ProTech": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat,
            "Bias": bias_feat,
            "PriceAction": pa_feat,
            "RSIContext": rsi_ctx,
            "VolumeContext": vol_ctx,
            "LevelContext": lvl_ctx,
        },
        "Market": {
            "VNINDEX": market_ctx.get("VNINDEX", {}),
            "VN30": market_ctx.get("VN30", {}),
            "StockChangePct": stock_chg,
            "RelativeStrengthVsVNINDEX": rel,
        },
        "StructureQuality": sanitize_pack(struct_q) if isinstance(struct_q, dict) else {},
        "PrimarySetup": sanitize_pack(primary) if isinstance(primary, dict) else {},
    }

    # Contracts: normalize to prevent type drift
    analysis_pack = normalize_analysis_pack(analysis_pack)

    last_dict = {
        str(k): _as_scalar(v)
        for k, v in (
            (last.to_dict() if hasattr(last, "to_dict") else dict(last))
        ).items()
    }

    return {
        "Ticker": tick,
        "Last": last_dict,
        "Scenario": scenario,
        "Conviction": conviction,
        "DualFibo": dual_fib,
        "Scenario12": scenario12,
        "MasterScore": master,
        "RRSim": rrsim,
        "AnalysisPack": analysis_pack,
        "_DF": df,
        "Fundamental": fund_row,
    }

