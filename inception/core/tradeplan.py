"""Trade plan builder (pure-ish domain logic).

Depends on pandas/numpy + core helpers. No Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers import _safe_float

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str
    status: str = "Watch"
    reason_tags: List[str] = field(default_factory=list)

def _compute_rr(entry: float, stop: float, tp: float) -> float:
    if any(pd.isna([entry, stop, tp])) or entry <= stop:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return (reward / risk) if risk > 0 else np.nan

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

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

def _buffer_price_dynamic(df: pd.DataFrame, entry: float) -> float:
    """
    Buffer = max(0.5*ATR14, vol_proxy)
    - No fixed percent
    - Entirely derived from price/volatility series
    """
    if pd.isna(entry) or entry == 0 or df.empty:
        return np.nan

    a = atr_wilder(df, 14)
    atr_last = _safe_float(a.dropna().iloc[-1]) if not a.dropna().empty else np.nan
    b_atr = (0.5 * atr_last) if pd.notna(atr_last) else np.nan

    b_proxy = _dynamic_vol_proxy(df, lookback=20)

    cands = [x for x in [b_atr, b_proxy] if pd.notna(x) and x > 0]
    if not cands:
        return np.nan
    return float(max(cands))

def _collect_levels(levels: Dict[str, Any]) -> List[Tuple[str, float]]:
    out = []
    for k, v in (levels or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            out.append((str(k), float(fv)))
    out.sort(key=lambda x: x[1])
    return out

def _nearest_above(levels: Dict[str, Any], x: float) -> Tuple[Optional[str], float]:
    best_k, best_v = None, np.nan
    for k, v in _collect_levels(levels):
        if pd.isna(x): 
            continue
        if v > x:
            if best_k is None or v < best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _nearest_below(levels: Dict[str, Any], x: float) -> Tuple[Optional[str], float]:
    best_k, best_v = None, np.nan
    for k, v in _collect_levels(levels):
        if pd.isna(x):
            continue
        if v < x:
            if best_k is None or v > best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _vol_ratio(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns:
        return np.nan
    vol = _safe_float(df.iloc[-1].get("Volume"))
    avg = _safe_float(df.iloc[-1].get("Avg20Vol")) if "Avg20Vol" in df.columns else np.nan
    if pd.notna(vol) and pd.notna(avg) and avg != 0:
        return float(vol / avg)
    return np.nan

def _probability_label_from_facts(df: pd.DataFrame, rr: float, status: str, vr: float) -> str:
    """
    Neutral probability tag derived only from technical facts already computed in df:
    - Trend (Close/MA50/MA200)
    - Momentum (RSI + MACD vs Signal)
    - Volume ratio
    - RR quality
    """
    if df.empty or pd.isna(rr) or status == "Invalid":
        return "N/A"

    last = df.iloc[-1]
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))

    pts = 0.0

    # Status weight
    st = (status or "Watch").strip().lower()
    if st == "active":
        pts += 0.8
    elif st == "watch":
        pts += 0.3

    # RR weight
    if rr >= 4.0:
        pts += 1.4
    elif rr >= 3.0:
        pts += 1.0
    elif rr >= 1.8:
        pts += 0.5
    else:
        pts += 0.1

    # Trend (structure)
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            pts += 0.9
        elif (c < ma50) and (ma50 < ma200):
            pts -= 0.7
        else:
            pts += 0.2

    # Momentum (RSI + MACD relation)
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            pts += 0.7
        elif (rsi <= 45) and (macd_v < sig):
            pts -= 0.5
        else:
            pts += 0.15

        # RSI>=70 is treated as neutral regime (no extra bullish pts)
        if rsi >= 70:
            pts += 0.0

    # Volume confirmation
    if pd.notna(vr):
        if vr >= 1.1:
            pts += 0.35
        elif vr < 0.8:
            pts -= 0.25

    if pts >= 2.4:
        return "High"
    if pts >= 1.3:
        return "Medium"
    return "Low"

def _build_anchor_level_map(df: pd.DataFrame, fib_short: Dict[str, Any], fib_long: Dict[str, Any]) -> Dict[str, float]:
    """
    Merge MA + fib(short+long) into a single anchor map.
    Keys are tags; values are prices.
    """
    last = df.iloc[-1] if not df.empty else pd.Series(dtype=object)
    anchors: Dict[str, float] = {}

    for ma_key in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(ma_key))
        if pd.notna(v):
            anchors[ma_key] = float(v)

    for k, v in (fib_short or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            anchors[f"FibS_{k}"] = float(fv)

    for k, v in (fib_long or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            anchors[f"FibL_{k}"] = float(fv)

    return anchors

def _nearest_support_below(anchors: Dict[str, float], x: float, exclude_vals: Optional[List[float]] = None) -> Tuple[str, float]:
    """
    Choose the nearest support strictly below x (i.e., the HIGHEST level < x).
    """
    exclude_vals = exclude_vals or []
    best_k, best_v = "N/A", np.nan
    for k, v in (anchors or {}).items():
        if pd.isna(x) or pd.isna(v):
            continue
        if any(pd.notna(ev) and abs(v - ev) <= 1e-9 for ev in exclude_vals):
            continue
        if v < x:
            if pd.isna(best_v) or v > best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _nearest_resistance_above(anchors: Dict[str, float], x: float, exclude_vals: Optional[List[float]] = None) -> Tuple[str, float]:
    """
    Choose the nearest resistance strictly above x (i.e., the LOWEST level > x).
    """
    exclude_vals = exclude_vals or []
    best_k, best_v = "N/A", np.nan
    for k, v in (anchors or {}).items():
        if pd.isna(x) or pd.isna(v):
            continue
        if any(pd.notna(ev) and abs(v - ev) <= 1e-9 for ev in exclude_vals):
            continue
        if v > x:
            if pd.isna(best_v) or v < best_v:
                best_k, best_v = k, v
    return best_k, best_v

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty:
        return {}

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))

    fib_short = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
    fib_long  = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}

    # unified TP pool (fib only)
    levels_tp = {}
    levels_tp.update(fib_short)
    levels_tp.update(fib_long)

    anchors = _build_anchor_level_map(df, fib_short, fib_long)
    vr = _vol_ratio(df)

    plans: Dict[str, TradeSetup] = {}

    # ----------------------------
    # 1) BREAKOUT PLAN
    # Entry anchor: nearest resistance above close (prefer fib short 61.8 if available)
    # Stop anchor: nearest support below entry (MA/Fib short/long) - buffer
    # TP: nearest fib above entry (else 3R fallback)
    # ----------------------------
    # Choose a "base resistance" for breakout trigger:
    # - Prefer FibS_61.8 if exists and >= close (acts like resistance), else nearest anchor above close.
    base_res = np.nan
    if pd.notna(close):
        s618 = _safe_float(fib_short.get("61.8"))
        if pd.notna(s618) and s618 >= close:
            base_res = s618
            base_res_tag = "Anchor=FibS_61.8"
        else:
            k_res, v_res = _nearest_resistance_above(anchors, close)
            base_res = v_res
            base_res_tag = f"Anchor={k_res}" if k_res != "N/A" else "Anchor=Fallback_Close"
            if pd.isna(base_res):
                base_res = close

    entry_b = _round_price(base_res * 1.01) if pd.notna(base_res) else np.nan
    buf_b = _buffer_price_dynamic(df, entry_b) if pd.notna(entry_b) else np.nan

    stop_ref_tag_b, stop_ref_val_b = _nearest_support_below(anchors, entry_b)
    stop_b = _round_price(stop_ref_val_b - buf_b) if (pd.notna(stop_ref_val_b) and pd.notna(buf_b)) else np.nan

    tp_label_b, tp_val_b = _nearest_above(levels_tp, entry_b) if pd.notna(entry_b) else (None, np.nan)
    if pd.notna(tp_val_b):
        tp_b = _round_price(tp_val_b)
    else:
        # fallback 3R
        if pd.notna(entry_b) and pd.notna(stop_b) and entry_b > stop_b:
            tp_b = _round_price(entry_b + 3.0 * (entry_b - stop_b))
        else:
            tp_b = np.nan

    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    tags_b: List[str] = []
    if base_res_tag: tags_b.append(base_res_tag)
    if stop_ref_tag_b != "N/A": tags_b.append(f"StopRef={stop_ref_tag_b}")
    if pd.notna(vr): tags_b.append(f"VolRatio={round(vr,2)}")
    if pd.notna(buf_b): tags_b.append("Buffer=Dynamic(ATR/Proxy)")
    if tp_label_b: tags_b.append(f"TP=Fib{tp_label_b}")

    status_b = "Watch"
    if any(pd.isna([entry_b, stop_b, tp_b, rr_b])) or (entry_b <= stop_b) or (rr_b < 1.2):
        status_b = "Invalid"
        tags_b.append("Invalid=GeometryOrRR")
    else:
        near_entry = (abs(close - entry_b) / close * 100) <= 1.2 if (pd.notna(close) and close != 0) else False
        vol_ok = (vr >= 1.1) if pd.notna(vr) else True
        if near_entry and vol_ok:
            status_b = "Active"
            tags_b.append("Trigger=NearEntry")
            if pd.notna(vr) and vr >= 1.1:
                tags_b.append("Trigger=VolumeSupport")

    prob_b = _probability_label_from_facts(df, rr_b, status_b, vr)
    breakout = TradeSetup(
        name="Breakout",
        entry=entry_b, stop=stop_b, tp=tp_b, rr=rr_b,
        probability=prob_b,
        status=status_b,
        reason_tags=tags_b
    )
    plans["Breakout"] = breakout

    # ----------------------------
    # 2) PULLBACK PLAN
    # Entry anchor: nearest support below close (prefer FibS_50 / FibS_38.2 / MA50 if available)
    # Stop anchor: next support below entry (exclude entry anchor) - buffer
    # TP: nearest fib above entry (else 2.6R fallback)
    # ----------------------------
    entry_anchor_tag = "EntryAnchor=Fallback_Close"
    entry_anchor_val = close

    # Preferred candidates if below close
    candidates: List[Tuple[str, float]] = []
    if pd.notna(close):
        for lab, val in [
            ("FibS_50.0", _safe_float(fib_short.get("50.0"))),
            ("FibS_38.2", _safe_float(fib_short.get("38.2"))),
            ("MA50", _safe_float(last.get("MA50"))),
            ("MA20", _safe_float(last.get("MA20"))),
        ]:
            if pd.notna(val) and val < close:
                candidates.append((lab, float(val)))

    if candidates:
        candidates.sort(key=lambda kv: abs(close - kv[1]))
        entry_anchor_tag, entry_anchor_val = candidates[0]
        entry_anchor_tag = f"EntryAnchor={entry_anchor_tag}"
    else:
        # fallback: nearest support below close from merged anchors
        k_sup, v_sup = _nearest_support_below(anchors, close)
        if pd.notna(v_sup):
            entry_anchor_tag = f"EntryAnchor={k_sup}"
            entry_anchor_val = v_sup

    entry_p = _round_price(entry_anchor_val) if pd.notna(entry_anchor_val) else np.nan
    buf_p = _buffer_price_dynamic(df, entry_p) if pd.notna(entry_p) else np.nan

    # stop = nearest support below entry, excluding entry anchor value (so "next level down")
    stop_ref_tag_p, stop_ref_val_p = _nearest_support_below(anchors, entry_p, exclude_vals=[entry_anchor_val])
    if pd.isna(stop_ref_val_p):
        # if no lower support, allow MA200 if below entry, else mark invalid by geometry later
        ma200 = _safe_float(last.get("MA200"))
        if pd.notna(ma200) and pd.notna(entry_p) and ma200 < entry_p and abs(ma200 - entry_anchor_val) > 1e-9:
            stop_ref_tag_p, stop_ref_val_p = "MA200", float(ma200)

    stop_p = _round_price(stop_ref_val_p - buf_p) if (pd.notna(stop_ref_val_p) and pd.notna(buf_p)) else np.nan

    tp_label_p, tp_val_p = _nearest_above(levels_tp, entry_p) if pd.notna(entry_p) else (None, np.nan)
    if pd.notna(tp_val_p):
        tp_p = _round_price(tp_val_p)
    else:
        # fallback 2.6R
        if pd.notna(entry_p) and pd.notna(stop_p) and entry_p > stop_p:
            tp_p = _round_price(entry_p + 2.6 * (entry_p - stop_p))
        else:
            tp_p = np.nan

    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    tags_p: List[str] = [entry_anchor_tag]
    if stop_ref_tag_p != "N/A": tags_p.append(f"StopRef={stop_ref_tag_p}")
    if pd.notna(vr): tags_p.append(f"VolRatio={round(vr,2)}")
    if pd.notna(buf_p): tags_p.append("Buffer=Dynamic(ATR/Proxy)")
    if tp_label_p: tags_p.append(f"TP=Fib{tp_label_p}")

    status_p = "Watch"
    if any(pd.isna([entry_p, stop_p, tp_p, rr_p])) or (entry_p <= stop_p) or (rr_p < 1.2):
        status_p = "Invalid"
        tags_p.append("Invalid=GeometryOrRR")
    else:
        near_entry = (abs(close - entry_p) / close * 100) <= 1.2 if (pd.notna(close) and close != 0) else False
        if near_entry:
            status_p = "Active"
            tags_p.append("Trigger=NearEntry")

    prob_p = _probability_label_from_facts(df, rr_p, status_p, vr)
    pullback = TradeSetup(
        name="Pullback",
        entry=entry_p, stop=stop_p, tp=tp_p, rr=rr_p,
        probability=prob_p,
        status=status_p,
        reason_tags=tags_p
    )
    plans["Pullback"] = pullback

    return plans
