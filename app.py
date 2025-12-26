# ============================================================
# INCEPTION v4.6
# app.py — Streamlit + GPT-4 Turbo
# Author: INCEPTION AI Research Framework
# Purpose: Technical–Fundamental Integrated Research Assistant
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any, List

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="INCEPTION v4.6",
    layout="wide",
    page_icon="🟣"
)

st.markdown("""
<style>
body {
    background-color: #0B0E11;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
strong { color: #E5E7EB; font-weight: 700; }
h1, h2, h3 { color: #E5E7EB; }

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextInput textarea {
    border-radius: 10px;
}

/* Make the action button full width and glossy black */
section[data-testid="stSidebar"] button[kind="primary"]{
    width: 100% !important;
    border-radius: 10px !important;
    background: linear-gradient(180deg, #111827 0%, #000000 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.40) !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover{
    filter: brightness(1.08);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. PATHS & CONSTANTS
# ============================================================

PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
TICKER_NAME_PATH = "Ticker name.xlsx"

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
    "KH03": {"name": "Khách mời 03", "quota": 5},
    "KH04": {"name": "Khách mời 04", "quota": 5},
    "KH05": {"name": "Khách mời 05", "quota": 5},
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def _fmt_price(x, ndigits=2):
    if pd.isna(x): return ""
    return f"{float(x):.{ndigits}f}"

def _fmt_int(x):
    if pd.isna(x): return ""
    return f"{int(round(float(x))):,}"

def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"{float(x):.1f}%"

def _fmt_thousand(x, ndigits=1):
    if pd.isna(x): return ""
    return f"{float(x):.{ndigits}f}"

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except:
        return default

def _round_price(x: float, ndigits: int = 2) -> float:
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

def _pct_dist(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return abs((a - b) / b) * 100

def _scenario_vi(s: str) -> str:
    mp = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng – Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng – Nhịp điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm – Yếu",
        "Neutral / Sideways": "Đi ngang / Trung tính",
    }
    return mp.get(s, s)

# ============================================================
# 4. LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"Lỗi khi đọc file {path}: {e}")
        return pd.DataFrame()
    df.columns = [c.strip().title() for c in df.columns]
    rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
    df.rename(columns=rename, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
    return df

@st.cache_data
def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name"])
    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Name"])
    name_col = "Stock Name" if "Stock Name" in df.columns else "Name"
    df = df.rename(columns={name_col: "Name"})
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    return df[["Ticker", "Name"]].drop_duplicates()

@st.cache_data
def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    rename_map = {}
    for c in df.columns:
        c0 = c.strip()
        c1 = c0.lower()

        if c1 in ["ticker", "ma", "symbol", "code"]:
            rename_map[c] = "Ticker"
        if c0 in ["TP (VND)", "Target", "Target Price", "TargetPrice", "TP"]:
            rename_map[c] = "Target"
        if c1 in ["recommendation", "khuyennghi", "khuyến nghị"]:
            rename_map[c] = "Recommendation"

    df.rename(columns=rename_map, inplace=True)

    if "Ticker" not in df.columns:
        for c in df.columns:
            if "ticker" in c.lower() or c.strip().lower() == "ma":
                df.rename(columns={c: "Ticker"}, inplace=True)
                break

    if "Target" not in df.columns:
        for c in df.columns:
            c1 = c.lower()
            if ("tp" in c1) or ("target" in c1) or ("muc tieu" in c1) or ("mục tiêu" in c1):
                df.rename(columns={c: "Target"}, inplace=True)
                break

    if "Ticker" not in df.columns or "Target" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    if "Recommendation" not in df.columns:
        df["Recommendation"] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")

    return df[["Ticker", "Target", "Recommendation"]].drop_duplicates(subset=["Ticker"], keep="last")

# ============================================================
# 5. INDICATORS
# ============================================================

def sma(series, window): return series.rolling(window=window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ============================================================
# 6. FIBONACCI DUAL-FRAME (SHORT AUTO 60/90 + LONG 250)
# ============================================================

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

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(np.nan, index=df.index)
    tr = _true_range(df["High"], df["Low"], df["Close"])
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def choose_short_fib_window(df: pd.DataFrame, candidates=(60, 90)) -> int:
    # deterministic, python-only: choose the window with clearer swing (range vs ATR)
    if df is None or df.empty:
        return candidates[0]
    best_w = candidates[0]
    best_score = -np.inf

    atr14 = atr_wilder(df, 14)
    last_atr = atr14.iloc[-1] if len(atr14) else np.nan

    for w in candidates:
        L = w if len(df) >= w else len(df)
        win = df.tail(L)
        hi = win["High"].max() if "High" in win.columns else np.nan
        lo = win["Low"].min() if "Low" in win.columns else np.nan
        rng = (hi - lo) if (pd.notna(hi) and pd.notna(lo)) else np.nan
        # if ATR missing, fallback to close std
        if pd.notna(last_atr) and last_atr != 0 and pd.notna(rng):
            score = float(rng / last_atr)
        else:
            sd = float(win["Close"].std()) if "Close" in win.columns else np.nan
            score = float(rng / sd) if (pd.notna(rng) and pd.notna(sd) and sd != 0) else -np.inf

        # slight preference to longer window if scores tie
        score = score + (0.0001 * w)
        if score > best_score:
            best_score = score
            best_w = w
    return int(best_w)

def compute_dual_fibonacci(df: pd.DataFrame, long_window: int = 250) -> Dict[str, Any]:
    short_window = choose_short_fib_window(df, candidates=(60, 90))

    L_short = short_window if len(df) >= short_window else len(df)
    L_long = long_window if len(df) >= long_window else len(df)

    win_short = df.tail(L_short)
    win_long = df.tail(L_long)

    s_hi, s_lo = win_short["High"].max(), win_short["Low"].min()
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()

    return {
        "short_window": L_short,
        "long_window": L_long,
        "auto_short": {"swing_high": s_hi, "swing_low": s_lo, "levels": _fib_levels(s_lo, s_hi)},
        "fixed_long": {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)}
    }

# ============================================================
# 6B. PRICE ACTION PACK (PYTHON-ONLY) + SIMPLE DIVERGENCE
# ============================================================

def _find_swings(series: pd.Series, mode: str = "low", lookback: int = 50) -> List[int]:
    # mode: "low" or "high"
    if series is None or series.empty:
        return []
    s = series.tail(lookback)
    idxs = list(s.index)
    vals = s.values
    out = []
    # 3-bar swing
    for i in range(1, len(vals) - 1):
        if mode == "low":
            if vals[i] <= vals[i-1] and vals[i] < vals[i+1]:
                out.append(idxs[i])
        else:
            if vals[i] >= vals[i-1] and vals[i] > vals[i+1]:
                out.append(idxs[i])
    return out[-5:]  # keep last few

def compute_divergence_pack(df: pd.DataFrame) -> Dict[str, Any]:
    # uses Close + RSI only, python-only
    if df is None or df.empty or "Close" not in df.columns or "RSI" not in df.columns:
        return {"Status": "N/A"}

    close = df["Close"]
    rsi = df["RSI"]

    lows = _find_swings(close, "low", lookback=60)
    highs = _find_swings(close, "high", lookback=60)

    bull = "N/A"
    bear = "N/A"

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        p1, p2 = _safe_float(close.loc[i1]), _safe_float(close.loc[i2])
        r1, r2 = _safe_float(rsi.loc[i1]), _safe_float(rsi.loc[i2])
        if pd.notna(p1) and pd.notna(p2) and pd.notna(r1) and pd.notna(r2):
            if (p2 < p1) and (r2 > r1):
                bull = "Bullish divergence"
            else:
                bull = "No clear bullish divergence"

    if len(highs) >= 2:
        j1, j2 = highs[-2], highs[-1]
        p1, p2 = _safe_float(close.loc[j1]), _safe_float(close.loc[j2])
        r1, r2 = _safe_float(rsi.loc[j1]), _safe_float(rsi.loc[j2])
        if pd.notna(p1) and pd.notna(p2) and pd.notna(r1) and pd.notna(r2):
            if (p2 > p1) and (r2 < r1):
                bear = "Bearish divergence"
            else:
                bear = "No clear bearish divergence"

    return {"Status": "OK", "Bullish": bull, "Bearish": bear}

def compute_price_action_pack(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty or len(df) < 2:
        return {"Status": "N/A", "Reason": "Not enough bars"}
    cols_ok = all(c in df.columns for c in ["High", "Low", "Close"])
    if not cols_ok:
        return {"Status": "N/A", "Reason": "Missing High/Low/Close"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    has_open = ("Open" in df.columns) and pd.notna(last.get("Open")) and pd.notna(prev.get("Open"))

    c = _safe_float(last.get("Close"))
    h = _safe_float(last.get("High"))
    l = _safe_float(last.get("Low"))
    o = _safe_float(last.get("Open")) if has_open else np.nan

    pc = _safe_float(prev.get("Close"))
    ph = _safe_float(prev.get("High"))
    pl = _safe_float(prev.get("Low"))
    po = _safe_float(prev.get("Open")) if has_open else np.nan

    atr14 = atr_wilder(df, 14).iloc[-1] if len(df) >= 14 else np.nan

    rng = (h - l) if (pd.notna(h) and pd.notna(l)) else np.nan
    body = abs(c - o) if (has_open and pd.notna(c) and pd.notna(o)) else np.nan
    upper_wick = (h - max(o, c)) if (has_open and pd.notna(h) and pd.notna(o) and pd.notna(c)) else np.nan
    lower_wick = (min(o, c) - l) if (has_open and pd.notna(l) and pd.notna(o) and pd.notna(c)) else np.nan

    body_pct = (body / rng * 100) if (pd.notna(body) and pd.notna(rng) and rng != 0) else np.nan
    upper_wick_pct = (upper_wick / rng * 100) if (pd.notna(upper_wick) and pd.notna(rng) and rng != 0) else np.nan
    lower_wick_pct = (lower_wick / rng * 100) if (pd.notna(lower_wick) and pd.notna(rng) and rng != 0) else np.nan
    range_to_atr = (rng / atr14) if (pd.notna(rng) and pd.notna(atr14) and atr14 != 0) else np.nan

    patterns = []

    # Inside / Outside (doesn't need Open)
    if pd.notna(h) and pd.notna(l) and pd.notna(ph) and pd.notna(pl):
        if (h < ph) and (l > pl):
            patterns.append({"Name": "Inside Bar", "Bias": "Neutral", "Strength": "Info"})
        if (h > ph) and (l < pl):
            patterns.append({"Name": "Outside Bar", "Bias": "Volatile", "Strength": "Info"})

    # Open-dependent patterns
    if has_open and pd.notna(o) and pd.notna(po) and pd.notna(c) and pd.notna(pc) and pd.notna(rng) and rng != 0:
        if pd.notna(body_pct) and body_pct <= 10:
            patterns.append({"Name": "Doji-like", "Bias": "Indecision", "Strength": "Info"})

        if pd.notna(body) and pd.notna(upper_wick) and pd.notna(lower_wick):
            if (lower_wick >= 2.0 * max(body, 1e-9)) and (upper_wick_pct <= 30 if pd.notna(upper_wick_pct) else False):
                patterns.append({"Name": "Hammer-like (Pinbar)", "Bias": "Bullish (potential)", "Strength": "Medium"})
            if (upper_wick >= 2.0 * max(body, 1e-9)) and (lower_wick_pct <= 30 if pd.notna(lower_wick_pct) else False):
                patterns.append({"Name": "Shooting Star-like (Pinbar)", "Bias": "Bearish (potential)", "Strength": "Medium"})

        bull_prev = pc < po
        bear_prev = pc > po
        bull_now = c > o
        bear_now = c < o

        if bull_prev and bull_now and (c >= po) and (o <= pc):
            patterns.append({"Name": "Bullish Engulfing", "Bias": "Bullish", "Strength": "High"})
        if bear_prev and bear_now and (c <= po) and (o >= pc):
            patterns.append({"Name": "Bearish Engulfing", "Bias": "Bearish", "Strength": "High"})

    # Context proximity to MA/Fibo
    near = {}
    for k in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(k))
        near[k] = {"Value": v, "DistPct": _pct_dist(c, v)}

    try:
        s_lv = dual_fib.get("auto_short", {}).get("levels", {})
        l_lv = dual_fib.get("fixed_long", {}).get("levels", {})
        for tag, lv in [("Short", s_lv), ("Long", l_lv)]:
            for key in ["38.2", "50.0", "61.8"]:
                v = _safe_float(lv.get(key))
                near[f"Fibo{tag}_{key}"] = {"Value": v, "DistPct": _pct_dist(c, v)}
    except:
        pass

    near_hits = []
    for k, obj in near.items():
        d = _safe_float(obj.get("DistPct"))
        if pd.notna(d) and d <= 1.0:
            near_hits.append(k)

    return {
        "Status": "OK",
        "HasOpen": bool(has_open),
        "LastCandle": {
            "Range": rng,
            "Body": body,
            "UpperWick": upper_wick,
            "LowerWick": lower_wick,
            "BodyPct": body_pct,
            "UpperWickPct": upper_wick_pct,
            "LowerWickPct": lower_wick_pct,
            "ATR14": atr14,
            "RangeToATR": range_to_atr
        },
        "Patterns": patterns,
        "NearHits": near_hits
    }

# ============================================================
# 7. CONVICTION SCORE
# ============================================================

def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)

# ============================================================
# 8. TRADE PLAN LOGIC
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str

def _compute_rr(entry: float, stop: float, tp: float) -> float:
    if any(pd.isna([entry, stop, tp])) or entry <= stop:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty: return {}

    last = df.iloc[-1]
    close = last["Close"]
    ma20 = last["MA20"]

    fib_short = dual_fib["auto_short"]["levels"]

    res_zone = fib_short.get("61.8", close * 1.05)
    sup_zone = fib_short.get("38.2", close * 0.95)

    entry_b = _round_price(res_zone * 1.01)
    stop_b = _round_price(max(ma20 * 0.985, sup_zone * 0.99))
    tp_b = _round_price(entry_b * 1.25)
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    entry_p = _round_price(sup_zone)
    stop_p = _round_price(entry_p * 0.94)
    tp_p = _round_price(entry_p * 1.20)
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    setups = {}
    if rr_b >= 2.5:
        setups["Breakout"] = TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao")
    if rr_p >= 2.5:
        setups["Pullback"] = TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, "TB")

    return setups

# ============================================================
# 9. SCENARIO CLASSIFICATION
# ============================================================

def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20:
            return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200:
            return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200:
            return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# ============================================================
# 9B. 12-SCENARIO (PYTHON-ONLY)
# ============================================================

def classify_scenario12(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    # Trend regime (3)
    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"
        elif c < ma50 and ma50 < ma200:
            trend = "Down"
        else:
            trend = "Neutral"

    # Momentum regime (4)
    mom = "Neutral"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = "Bull"
        elif (rsi <= 45) and (macd_v < sig):
            mom = "Bear"
        elif (rsi >= 70):
            mom = "Exhaust"
        else:
            mom = "Neutral"

    # Volume regime
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"

    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Bull": 0, "Neutral": 1, "Bear": 2, "Exhaust": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1

    name_map = {
        ("Up","Bull"): "S1 – Uptrend + Bullish Momentum",
        ("Up","Neutral"): "S2 – Uptrend + Neutral Momentum",
        ("Up","Bear"): "S3 – Uptrend + Bearish Pullback",
        ("Up","Exhaust"): "S4 – Uptrend + Overbought/Exhaust",
        ("Neutral","Bull"): "S5 – Range + Bullish Attempt",
        ("Neutral","Neutral"): "S6 – Range + Balanced",
        ("Neutral","Bear"): "S7 – Range + Bearish Pressure",
        ("Neutral","Exhaust"): "S8 – Range + Overbought Risk",
        ("Down","Bull"): "S9 – Downtrend + Short-covering Bounce",
        ("Down","Neutral"): "S10 – Downtrend + Weak Stabilization",
        ("Down","Bear"): "S11 – Downtrend + Bearish Momentum",
        ("Down","Exhaust"): "S12 – Downtrend + Overbought Rebound Risk",
    }

    return {
        "Code": int(code),
        "TrendRegime": trend,
        "MomentumRegime": mom,
        "VolumeRegime": vol_reg,
        "Name": name_map.get((trend, mom), "Scenario – N/A"),
    }

# ============================================================
# 9C. MASTER SCORE (PYTHON-ONLY)
# ============================================================

def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup], fund_row: Dict[str, Any]) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))
    upside_pct = _safe_float(fund_row.get("UpsidePct"))

    comps = {}

    trend = 0.0
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            trend = 2.0
        elif (c >= ma200):
            trend = 1.2
        else:
            trend = 0.4
    comps["Trend"] = trend

    mom = 0.0
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = 2.0
        elif (rsi <= 45) and (macd_v < sig):
            mom = 0.4
        else:
            mom = 1.1
    comps["Momentum"] = mom

    vcomp = 0.0
    if pd.notna(vol) and pd.notna(avg_vol):
        vcomp = 1.6 if vol > avg_vol else 0.9
    comps["Volume"] = vcomp

    fibc = 0.0
    try:
        s_lv = dual_fib.get("auto_short", {}).get("levels", {})
        l_lv = dual_fib.get("fixed_long", {}).get("levels", {})
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))

        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            fibc += 1.2 if c >= s_618 else (0.8 if c >= s_382 else 0.4)
        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            fibc += 0.8 if c >= l_618 else (0.5 if c >= l_382 else 0.2)
    except:
        fibc = 0.0
    comps["Fibonacci"] = fibc

    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(s.rr)]
        best_rr = max(rrs) if rrs else np.nan
    rrcomp = 0.0
    if pd.notna(best_rr):
        rrcomp = 2.0 if best_rr >= 4.0 else (1.5 if best_rr >= 3.0 else 1.0)
    comps["RRQuality"] = rrcomp

    fcomp = 0.0
    if pd.notna(upside_pct):
        fcomp = 2.0 if upside_pct >= 25 else (1.5 if upside_pct >= 15 else (1.0 if upside_pct >= 5 else 0.5))
    comps["FundamentalUpside"] = fcomp

    total = float(sum(comps.values()))
    if total >= 9.0:
        tier, sizing = "A+", "Aggressive (2.0x) if risk control ok"
    elif total >= 7.5:
        tier, sizing = "A", "Full size (1.0x) + consider pyramiding"
    elif total >= 6.0:
        tier, sizing = "B", "Medium size (0.6–0.8x)"
    elif total >= 4.5:
        tier, sizing = "C", "Small / tactical (0.3–0.5x)"
    else:
        tier, sizing = "D", "No edge / avoid or hedge"

    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": tier,
        "PositionSizing": sizing,
        "BestRR": best_rr if pd.notna(best_rr) else np.nan
    }

# ============================================================
# 9D. RR SIM (PYTHON-ONLY)
# ============================================================

def build_rr_sim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    rows = []
    best_rr = np.nan
    for k, s in trade_plans.items():
        entry = _safe_float(s.entry)
        stop = _safe_float(s.stop)
        tp = _safe_float(s.tp)
        rr = _safe_float(s.rr)
        risk_pct = ((entry - stop) / entry * 100) if (pd.notna(entry) and pd.notna(stop) and entry != 0) else np.nan
        reward_pct = ((tp - entry) / entry * 100) if (pd.notna(tp) and pd.notna(entry) and entry != 0) else np.nan
        rows.append({
            "Setup": k,
            "Entry": entry,
            "Stop": stop,
            "TP": tp,
            "RR": rr,
            "RiskPct": risk_pct,
            "RewardPct": reward_pct,
            "Probability": s.probability
        })
        if pd.notna(rr):
            best_rr = rr if pd.isna(best_rr) else max(best_rr, rr)
    return {"Setups": rows, "BestRR": best_rr if pd.notna(best_rr) else np.nan}

# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================

def _normalize_target_to_close_unit(target_raw: float, close: float) -> Dict[str, Any]:
    # Goal: compute target in the SAME unit as Close.
    # Typical case: Close is in "ngàn" (e.g., 30.5) but target file may be in VND (e.g., 42500).
    t = _safe_float(target_raw, np.nan)
    c = _safe_float(close, np.nan)
    if pd.isna(t) or pd.isna(c) or c == 0:
        return {"TargetNorm": np.nan, "TargetDisplayK": np.nan, "Rule": "N/A"}

    # Heuristic:
    # - If close < 1000 and target > 1000 => target likely VND => divide by 1000
    # - Else keep as-is
    rule = "Keep"
    t_norm = t
    if (c < 1000) and (t > 1000):
        t_norm = t / 1000.0
        rule = "Target/1000"
    return {"TargetNorm": t_norm, "TargetDisplayK": t_norm, "Rule": rule}

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Error": "Không đọc được dữ liệu Price_Vol.xlsx"}

    df = df_all[df_all["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {ticker}"}

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h

    dual_fib = compute_dual_fibonacci(df, long_window=250)
    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].astype(str).str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    close = _safe_float(last.get("Close"), np.nan)
    target_raw = _safe_float(fund_row.get("Target"), np.nan)
    norm = _normalize_target_to_close_unit(target_raw, close)

    target_norm = _safe_float(norm.get("TargetNorm"), np.nan)
    upside_pct = ((target_norm - close) / close * 100) if (pd.notna(target_norm) and pd.notna(close) and close != 0) else np.nan

    fund_row["Target"] = target_raw
    fund_row["TargetNorm"] = target_norm
    fund_row["UpsidePct"] = upside_pct
    fund_row["TargetNormRule"] = norm.get("Rule")

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

    price_action = compute_price_action_pack(df, dual_fib)
    divergence = compute_divergence_pack(df)

    # MA/MACD cross quick flags (python-only)
    ma_cross = {
        "MA20xMA50": "N/A",
        "MA50xMA200": "N/A"
    }
    if len(df) >= 2:
        p = df.iloc[-2]
        if pd.notna(p.get("MA20")) and pd.notna(p.get("MA50")) and pd.notna(last.get("MA20")) and pd.notna(last.get("MA50")):
            prev_rel = float(p["MA20"] - p["MA50"])
            now_rel = float(last["MA20"] - last["MA50"])
            if prev_rel <= 0 and now_rel > 0:
                ma_cross["MA20xMA50"] = "Golden cross (MA20 lên MA50)"
            elif prev_rel >= 0 and now_rel < 0:
                ma_cross["MA20xMA50"] = "Death cross (MA20 xuống MA50)"
            else:
                ma_cross["MA20xMA50"] = "No cross"

        if pd.notna(p.get("MA50")) and pd.notna(p.get("MA200")) and pd.notna(last.get("MA50")) and pd.notna(last.get("MA200")):
            prev_rel = float(p["MA50"] - p["MA200"])
            now_rel = float(last["MA50"] - last["MA200"])
            if prev_rel <= 0 and now_rel > 0:
                ma_cross["MA50xMA200"] = "Golden cross (MA50 lên MA200)"
            elif prev_rel >= 0 and now_rel < 0:
                ma_cross["MA50xMA200"] = "Death cross (MA50 xuống MA200)"
            else:
                ma_cross["MA50xMA200"] = "No cross"

    macd_cross = "N/A"
    if len(df) >= 2 and pd.notna(df.iloc[-2].get("MACD")) and pd.notna(df.iloc[-2].get("MACDSignal")) and pd.notna(last.get("MACD")) and pd.notna(last.get("MACDSignal")):
        prev_rel = float(df.iloc[-2]["MACD"] - df.iloc[-2]["MACDSignal"])
        now_rel = float(last["MACD"] - last["MACDSignal"])
        if prev_rel <= 0 and now_rel > 0:
            macd_cross = "MACD cắt lên Signal"
        elif prev_rel >= 0 and now_rel < 0:
            macd_cross = "MACD cắt xuống Signal"
        else:
            macd_cross = "No cross"

    analysis_pack = {
        "Ticker": ticker.upper(),
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
        "MA_Cross": ma_cross,
        "MACD_Cross": macd_cross,
        "Divergence": divergence,
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {})
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetRaw": target_raw,
            "TargetNorm": target_norm,
            "TargetK": target_norm,
            "UpsidePct": upside_pct,
            "TargetNormRule": fund_row.get("TargetNormRule")
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(v.entry),
                "Stop": _safe_float(v.stop),
                "TP": _safe_float(v.tp),
                "RR": _safe_float(v.rr),
                "Probability": v.probability
            } for k, v in trade_plans.items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "PriceAction": price_action
    }

    return {
        "Ticker": ticker.upper(),
        "Last": last.to_dict(),
        "Scenario": scenario,
        "Conviction": conviction,
        "DualFibo": dual_fib,
        "TradePlans": trade_plans,
        "Fundamental": fund_row,
        "Scenario12": scenario12,
        "MasterScore": master,
        "RRSim": rrsim,
        "AnalysisPack": analysis_pack
    }

# ============================================================
# 11. GPT REPORT
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick = data["Ticker"]
    last = data["Last"]
    conviction = data["Conviction"]
    scenario = _scenario_vi(data["Scenario"])
    analysis_pack = data.get("AnalysisPack", {})

    close = _fmt_price(last.get("Close"))
    header_html = f"""
    <div style="font-size:22px; font-weight:800; color:#E5E7EB; line-height:1.3;">
      {tick} — {close} | Điểm Conviction: {conviction:.1f}/10 | {scenario}
    </div>
    """

    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    prompt = f"""
Bạn là chuyên gia phân tích chứng khoán cao cấp. Viết báo cáo tiếng Việt, rõ ràng, mạch lạc, thân thiện (xưng hô với "bạn").

QUY TẮC BẮT BUỘC:
- Không bịa số.
- Không tự tính toán bất kỳ con số nào.
- Chỉ được dùng đúng dữ liệu trong JSON "AnalysisPack".
- Tuyệt đối không dùng emoji/bullets kiểu 1️⃣ 2️⃣ hoặc ký hiệu trang trí.
- Không in/nhắc "Fibo Conflict" (nếu có dữ liệu liên quan thì chỉ dùng để suy luận nội bộ).

FORMAT OUTPUT BẮT BUỘC (giữ đúng cấu trúc):
(1) Một đoạn TÓM TẮT 3–5 câu (không gắn tiêu đề).

A. Kỹ thuật
1. MA
2. RSI
3. MACD
4. RSI + MACD Bias
5. Fibonacci (2 khung: ShortWindow và LongWindow)
6. Volume & Price Action (bao gồm nến/mẫu hình nếu có)
7. 12-Scenario
8. MasterScore (tier + sizing)

B. Cơ bản
- Nêu khuyến nghị, giá mục tiêu (TargetK), upside.

C. Trade plan
- Tóm tắt các setup trong TradePlans (nếu không có thì ghi rõ).

D. Rủi ro vs lợi nhuận
- Diễn giải dựa trên RRSim (RiskPct/RewardPct/RR/Probability).

Lưu ý văn phong:
- Mỗi câu chỉ nên dùng 1–2 con số (tối đa).
- Tránh kiểu câu khô cứng liệt kê: "MA20 là..., MA50 là...".
- Nếu thiếu dữ liệu thì ghi "N/A" đúng chỗ.

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1600
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return header_html + "\n\n" + content

# ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.6</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập")
    user_key = st.text_input("Mã VIP", type="password")
    ticker_input = st.text_input("Mã cổ phiếu", value="VCB").upper()
    run_btn = st.button("Phân tích", type="primary")

col_main, = st.columns([1])

# ============================================================
# 13. MAIN EXECUTION
# ============================================================

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng. Vui lòng nhập lại.")
    else:
        # Keep missing-file warnings only (no "ready" / no guides)
        missing_files = [f for f in [PRICE_VOL_PATH, HSC_TARGET_PATH, TICKER_NAME_PATH] if not os.path.exists(f)]
        if missing_files:
            st.warning(f"⚠️ Thiếu file dữ liệu: {', '.join(missing_files)}.")
        with st.spinner(f"Đang phân tích {ticker_input}..."):
            try:
                result = analyze_ticker(ticker_input)
                report = generate_insight_report(result)
                st.markdown(report, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Lỗi xử lý: {e}")

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <p style='text-align:center; color:#6B7280; font-size:13px;'>
    © 2025 INCEPTION Research Framework
    </p>
    """,
    unsafe_allow_html=True
)
