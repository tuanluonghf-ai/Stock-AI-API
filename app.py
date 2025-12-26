# ============================================================
# INCEPTION v4.8
# app.py — Streamlit + GPT-4 Turbo
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(page_title="INCEPTION v4.8", layout="wide", page_icon="🟣")

st.markdown(
    """
<style>
body {
    background-color: #0B0E11;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
strong { color: #E5E7EB; font-weight: 700; }
h1, h2, h3 { color: #E5E7EB; }

/* Full-width glossy black button */
.stButton>button {
    width: 100%;
    background: linear-gradient(180deg, #111827 0%, #000000 100%);
    color: #FFFFFF !important;
    font-weight: 700;
    border-radius: 10px;
    height: 44px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 6px 14px rgba(0,0,0,0.45);
}
.stButton>button:hover {
    background: linear-gradient(180deg, #0B1220 0%, #000000 100%);
    border: 1px solid rgba(255,255,255,0.18);
}
</style>
""",
    unsafe_allow_html=True,
)

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
# 3. HELPERS
# ============================================================

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except:
        return default

def _fmt_price(x, ndigits=2):
    if pd.isna(x):
        return ""
    return f"{float(x):.{ndigits}f}"

def _fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{float(x):.2f}%"

def _fmt_thousand(x, ndigits=1):
    if pd.isna(x):
        return ""
    return f"{float(x)/1000:.{ndigits}f}"

def _round_price(x: float, ndigits: int = 2) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x), ndigits)

def _sgn(x: float) -> int:
    if pd.isna(x):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a - b) / b * 100

def _trend_label_from_slope(slope: float, eps: float = 1e-9) -> str:
    if pd.isna(slope):
        return "N/A"
    if slope > eps:
        return "Up"
    if slope < -eps:
        return "Down"
    return "Flat"

def _scenario_vi(x: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng — Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng — Pha điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm — Yếu",
        "Neutral / Sideways": "Đi ngang / Trung tính",
    }
    return m.get(x, x)

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
    sign = diff.apply(_sgn).values

    last_event = None
    last_bars_ago = None
    for i in range(len(sign) - 1, 0, -1):
        if sign[i] == 0 or sign[i - 1] == 0:
            continue
        if sign[i] != sign[i - 1]:
            last_event = "CrossUp" if sign[i - 1] < sign[i] else "CrossDown"
            last_bars_ago = (len(sign) - 1) - i
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

    lows, highs = [], []
    for i in range(2, n - 2):
        if c[i] < c[i - 1] and c[i] < c[i + 1]:
            lows.append(i)
        if c[i] > c[i - 1] and c[i] > c[i + 1]:
            highs.append(i)

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if (c[i2] < c[i1]) and (o[i2] > o[i1]):
            return {"Type": "Bullish", "Detail": "Price LL vs Osc HL"}
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if (c[i2] > c[i1]) and (o[i2] < o[i1]):
            return {"Type": "Bearish", "Detail": "Price HH vs Osc LH"}
    return {"Type": "None", "Detail": "N/A"}

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

    df.columns = [str(c).strip().title() for c in df.columns]
    rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
    df.rename(columns=rename, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values(["Ticker", "Date"])
    else:
        df = df.sort_values(["Ticker"]).copy()

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    return df

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

    if "Ticker" not in df.columns or "Target" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    if "Recommendation" not in df.columns:
        df["Recommendation"] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    # Normalize Target:
    # - parse numeric even if string like "42,500"
    # - if numeric < 500 => assume "thousand" style, convert to VND by *1000
    tgt = pd.to_numeric(df["Target"], errors="coerce")
    df["Target"] = tgt
    m = df["Target"].notna() & (df["Target"] < 500)
    df.loc[m, "Target"] = df.loc[m, "Target"] * 1000.0

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

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Requires High/Low/Close; fallback to abs(close diff) proxy
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = df["Close"].diff().abs()
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ============================================================
# 6. FIBONACCI DUAL-FRAME (AUTO SELECT 60 OR 90 + LONG 250)
# ============================================================

def _fib_levels(low, high):
    rng = high - low
    if rng <= 0:
        return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng,
        "127.2": high + 0.272 * rng,
        "161.8": high + 0.618 * rng,
    }

def _compute_fib_window(df: pd.DataFrame, w: int) -> Dict[str, Any]:
    L = w if len(df) >= w else len(df)
    win = df.tail(L)
    hi, lo = win["High"].max(), win["Low"].min() if ("High" in win.columns and "Low" in win.columns) else (win["Close"].max(), win["Close"].min())
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
        return {
            "short_window": None,
            "long_window": None,
            "auto_short": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "fixed_long": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "alt_short": {"window": None, "swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "selection_reason": "N/A",
        }

    last_close = _safe_float(df.iloc[-1].get("Close"))
    fib60 = _compute_fib_window(df, 60)
    fib90 = _compute_fib_window(df, 90)

    s60 = _score_fib_relevance(last_close, fib60)
    s90 = _score_fib_relevance(last_close, fib90)

    if s90 > s60:
        chosen, alt = fib90, fib60
        reason = "AutoSelect=90 (higher relevance score)"
    else:
        chosen, alt = fib60, fib90
        reason = "AutoSelect=60 (higher relevance score)"

    L_long = long_window if len(df) >= long_window else len(df)
    win_long = df.tail(L_long)
    l_hi = win_long["High"].max() if "High" in win_long.columns else win_long["Close"].max()
    l_lo = win_long["Low"].min() if "Low" in win_long.columns else win_long["Close"].min()

    return {
        "short_window": chosen.get("window"),
        "long_window": L_long,
        "auto_short": {"swing_high": chosen.get("swing_high"), "swing_low": chosen.get("swing_low"), "levels": chosen.get("levels", {})},
        "fixed_long": {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)},
        "alt_short": alt,
        "selection_reason": reason,
    }

# ============================================================
# 6B. PRO TECH FEATURES (PYTHON-ONLY): MA/RSI/MACD/VOL
# ============================================================

def compute_ma_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    def slope(series: pd.Series, n: int = 10) -> float:
        s = series.dropna()
        if len(s) < n + 1:
            return np.nan
        return _safe_float(s.iloc[-1] - s.iloc[-(n + 1)])

    s20 = slope(df["MA20"], 10)
    s50 = slope(df["MA50"], 10)
    s200 = slope(df["MA200"], 10)

    regime = "Neutral"
    if pd.notna(close) and pd.notna(ma50) and pd.notna(ma200):
        if close >= ma50 and ma50 >= ma200:
            regime = "Up"
        elif close < ma50 and ma50 < ma200:
            regime = "Down"

    dist50 = ((close - ma50) / ma50 * 100) if (pd.notna(close) and pd.notna(ma50) and ma50 != 0) else np.nan
    dist200 = ((close - ma200) / ma200 * 100) if (pd.notna(close) and pd.notna(ma200) and ma200 != 0) else np.nan

    cross_price_ma50 = _find_last_cross(df["Close"], df["MA50"], lookback=20)
    cross_price_ma200 = _find_last_cross(df["Close"], df["MA200"], lookback=60)
    cross_ma20_ma50 = _find_last_cross(df["MA20"], df["MA50"], lookback=60)
    cross_ma50_ma200 = _find_last_cross(df["MA50"], df["MA200"], lookback=120)

    return {
        "Regime": regime,
        "SlopeMA20": _trend_label_from_slope(s20),
        "SlopeMA50": _trend_label_from_slope(s50),
        "SlopeMA200": _trend_label_from_slope(s200),
        "DistToMA50Pct": dist50,
        "DistToMA200Pct": dist200,
        "Cross": {
            "PriceVsMA50": cross_price_ma50,
            "PriceVsMA200": cross_price_ma200,
            "MA20VsMA50": cross_ma20_ma50,
            "MA50VsMA200": cross_ma50_ma200,
        },
        "Structure": {
            "PriceAboveMA50": bool(pd.notna(close) and pd.notna(ma50) and close >= ma50),
            "PriceAboveMA200": bool(pd.notna(close) and pd.notna(ma200) and close >= ma200),
            "MA20AboveMA50": bool(pd.notna(ma20) and pd.notna(ma50) and ma20 >= ma50),
            "MA50AboveMA200": bool(pd.notna(ma50) and pd.notna(ma200) and ma50 >= ma200),
        },
    }

def compute_rsi_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    rsi = df["RSI"].dropna()
    if rsi.empty:
        return {}

    last_rsi = _safe_float(rsi.iloc[-1])
    prev_rsi = _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else np.nan

    direction = "N/A"
    if pd.notna(last_rsi) and pd.notna(prev_rsi):
        d = last_rsi - prev_rsi
        direction = "Rising" if d > 1.0 else ("Falling" if d < -1.0 else "Flat")

    state = "Neutral"
    if pd.notna(last_rsi):
        if last_rsi >= 70:
            state = "Overbought"
        elif last_rsi >= 60:
            state = "Bull"
        elif last_rsi >= 50:
            state = "Neutral+"
        elif last_rsi >= 40:
            state = "Neutral-"
        elif last_rsi >= 30:
            state = "Bear"
        else:
            state = "Oversold"

    div = _detect_divergence_simple(df["Close"], df["RSI"], lookback=60)
    return {"Value": last_rsi, "State": state, "Direction": direction, "Divergence": div}

def compute_macd_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    macd_v = df["MACD"].dropna()
    sig = df["MACDSignal"].dropna()
    hist = df["MACDHist"].dropna()
    if macd_v.empty or sig.empty:
        return {}

    last_m = _safe_float(macd_v.iloc[-1])
    last_s = _safe_float(sig.iloc[-1])
    last_h = _safe_float(hist.iloc[-1]) if not hist.empty else np.nan

    state = "Neutral"
    if pd.notna(last_m) and pd.notna(last_s):
        state = "Bull" if last_m > last_s else ("Bear" if last_m < last_s else "Neutral")

    cross = _find_last_cross(df["MACD"], df["MACDSignal"], lookback=30)

    zero = "N/A"
    if pd.notna(last_m):
        zero = "Above" if last_m > 0 else ("Below" if last_m < 0 else "Near")

    hist_state = "N/A"
    if len(hist) >= 4:
        h0, h1, h2 = _safe_float(hist.iloc[-1]), _safe_float(hist.iloc[-2]), _safe_float(hist.iloc[-3])
        if pd.notna(h0) and pd.notna(h1) and pd.notna(h2):
            if h0 >= 0 and h1 >= 0:
                hist_state = "ExpandingUp" if (h0 > h1 > h2) else ("ContractingUp" if (h0 < h1 < h2) else "MixedUp")
            elif h0 < 0 and h1 < 0:
                hist_state = "ExpandingDown" if (h0 < h1 < h2) else ("ContractingDown" if (h0 > h1 > h2) else "MixedDown")
            else:
                hist_state = "Flip"

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
    }

def compute_volume_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    last = df.iloc[-1]
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    ratio = (vol / avg) if (pd.notna(vol) and pd.notna(avg) and avg != 0) else np.nan

    regime = "N/A"
    if pd.notna(ratio):
        if ratio >= 1.8:
            regime = "Spike"
        elif ratio >= 1.2:
            regime = "High"
        elif ratio >= 0.8:
            regime = "Normal"
        else:
            regime = "Low"

    return {"Vol": vol, "Avg20Vol": avg, "Ratio": ratio, "Regime": regime}

# ============================================================
# 6C. PRICE ACTION FEATURES PACK (PYTHON-ONLY)
# ============================================================

def compute_price_action_features(df: pd.DataFrame, dual_fib: Dict[str, Any], vol_feat: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty or len(df) < 2:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # candle fields (fallback: use Close-only if missing Open/High/Low)
    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
    o = _safe_float(last.get("Open")) if has_ohlc else _safe_float(prev.get("Close"))
    h = _safe_float(last.get("High")) if has_ohlc else _safe_float(last.get("Close"))
    l = _safe_float(last.get("Low")) if has_ohlc else _safe_float(last.get("Close"))
    c = _safe_float(last.get("Close"))

    po = _safe_float(prev.get("Open")) if has_ohlc else _safe_float(df.iloc[-3].get("Close")) if len(df) >= 3 else _safe_float(prev.get("Close"))
    ph = _safe_float(prev.get("High")) if has_ohlc else _safe_float(prev.get("Close"))
    pl = _safe_float(prev.get("Low")) if has_ohlc else _safe_float(prev.get("Close"))
    pc = _safe_float(prev.get("Close"))

    rng = (h - l) if pd.notna(h) and pd.notna(l) else np.nan
    body = abs(c - o) if pd.notna(c) and pd.notna(o) else np.nan
    upper = (h - max(o, c)) if pd.notna(h) and pd.notna(o) and pd.notna(c) else np.nan
    lower = (min(o, c) - l) if pd.notna(l) and pd.notna(o) and pd.notna(c) else np.nan

    body_pct = (body / rng * 100) if pd.notna(body) and pd.notna(rng) and rng > 0 else np.nan
    upper_pct = (upper / rng * 100) if pd.notna(upper) and pd.notna(rng) and rng > 0 else np.nan
    lower_pct = (lower / rng * 100) if pd.notna(lower) and pd.notna(rng) and rng > 0 else np.nan

    # range percentile vs N days
    N = 60
    if has_ohlc:
        ranges = (df["High"] - df["Low"]).dropna().tail(N)
    else:
        ranges = df["Close"].diff().abs().dropna().tail(N)
    range_pctile = np.nan
    if len(ranges) >= 10 and pd.notna(rng):
        range_pctile = float((ranges <= rng).mean() * 100)

    gap_pct = np.nan
    if pd.notna(o) and pd.notna(pc) and pc != 0:
        gap_pct = (o - pc) / pc * 100

    # patterns (1–2 bar)
    patterns: List[str] = []

    # doji
    if pd.notna(body_pct) and body_pct <= 15:
        patterns.append("Doji")

    # hammer / shooting star (pin bar)
    if pd.notna(lower_pct) and pd.notna(body_pct) and pd.notna(upper_pct):
        if (lower_pct >= 55) and (body_pct <= 35) and (upper_pct <= 20):
            patterns.append("Hammer/PinBarBull")
        if (upper_pct >= 55) and (body_pct <= 35) and (lower_pct <= 20):
            patterns.append("ShootingStar/PinBarBear")

    # engulfing
    prev_bear = pd.notna(pc) and pd.notna(po) and (pc < po)
    prev_bull = pd.notna(pc) and pd.notna(po) and (pc > po)
    curr_bull = pd.notna(c) and pd.notna(o) and (c > o)
    curr_bear = pd.notna(c) and pd.notna(o) and (c < o)

    if prev_bear and curr_bull and pd.notna(o) and pd.notna(c) and pd.notna(po) and pd.notna(pc):
        if (o <= pc) and (c >= po):
            patterns.append("BullEngulfing")
    if prev_bull and curr_bear and pd.notna(o) and pd.notna(c) and pd.notna(po) and pd.notna(pc):
        if (o >= pc) and (c <= po):
            patterns.append("BearEngulfing")

    # inside / outside bar
    if has_ohlc and pd.notna(h) and pd.notna(l) and pd.notna(ph) and pd.notna(pl):
        if (h < ph) and (l > pl):
            patterns.append("InsideBar")
        if (h > ph) and (l < pl):
            patterns.append("OutsideBar")

    # context: near MA / near Fib
    near_ma = []
    for name in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(name))
        if pd.notna(v) and pd.notna(c) and c != 0:
            if abs(c - v) / c <= 0.012:  # within ~1.2%
                near_ma.append(name)

    near_fib = []
    fib_short = dual_fib.get("auto_short", {}).get("levels", {}) or {}
    fib_long = dual_fib.get("fixed_long", {}).get("levels", {}) or {}
    for tag, lv in [("S", fib_short), ("L", fib_long)]:
        for k in ["38.2", "50.0", "61.8"]:
            v = _safe_float(lv.get(k))
            if pd.notna(v) and pd.notna(c) and c != 0:
                if abs(c - v) / c <= 0.012:
                    near_fib.append(f"{tag}{k}")

    # PA signal (coarse)
    signal = "Neutral"
    if any(p in patterns for p in ["BullEngulfing", "Hammer/PinBarBull"]) and not any(p in patterns for p in ["BearEngulfing", "ShootingStar/PinBarBear"]):
        signal = "Bull"
    elif any(p in patterns for p in ["BearEngulfing", "ShootingStar/PinBarBear"]) and not any(p in patterns for p in ["BullEngulfing", "Hammer/PinBarBull"]):
        signal = "Bear"

    # candle role vs volume regime
    vol_reg = (vol_feat or {}).get("Regime", "N/A")
    candle_role = "N/A"
    if vol_reg in ["Spike", "High"]:
        if "Doji" in patterns or "ShootingStar/PinBarBear" in patterns:
            candle_role = "ExhaustionRisk"
        elif "BullEngulfing" in patterns or "Hammer/PinBarBull" in patterns:
            candle_role = "Confirmation"
        else:
            candle_role = "Participation"

    return {
        "CandleAnatomy": {
            "BodyPct": body_pct,
            "UpperWickPct": upper_pct,
            "LowerWickPct": lower_pct,
            "RangePctile": range_pctile,
            "GapPct": gap_pct,
        },
        "Patterns": patterns,
        "Signal": signal,
        "Context": {
            "NearMA": near_ma,
            "NearFib": near_fib,
            "VolumeRegime": vol_reg,
            "CandleRole": candle_role,
        },
    }

# ============================================================
# 7. CONVICTION SCORE
# ============================================================

def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if last["Close"] > last["MA200"]:
        score += 2
    if last["RSI"] > 55:
        score += 1
    if last["Volume"] > last["Avg20Vol"]:
        score += 1
    if last["MACD"] > last["MACDSignal"]:
        score += 0.5
    return min(10.0, score)

# ============================================================
# 8. TRADE PLAN LOGIC (TP kỹ thuật + stop động MA/Fibo + buffer)
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str
    meta: Dict[str, Any] = field(default_factory=dict)

def _compute_rr(entry: float, stop: float, tp: float) -> float:
    if any(pd.isna([entry, stop, tp])) or entry <= stop:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def _levels_from_ma_fib(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, float]:
    lv = {}
    for k in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(k))
        if pd.notna(v):
            lv[k] = v
    s = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    l = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}
    for k in ["38.2", "50.0", "61.8"]:
        v = _safe_float(s.get(k))
        if pd.notna(v):
            lv[f"FibS{k}"] = v
    for k in ["38.2", "50.0", "61.8"]:
        v = _safe_float(l.get(k))
        if pd.notna(v):
            lv[f"FibL{k}"] = v
    return lv

def _pick_anchor_below(entry: float, levels: Dict[str, float]) -> Optional[str]:
    below = [(name, v) for name, v in levels.items() if pd.notna(v) and v < entry]
    if not below:
        return None
    return max(below, key=lambda x: x[1])[0]

def _pick_tp_above(entry: float, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer short fib extensions; fallback to swing high; then 1.15x
    s_levels = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    candidates = []
    for k in ["127.2", "161.8"]:
        v = _safe_float(s_levels.get(k))
        if pd.notna(v) and v > entry:
            candidates.append((f"FibS{k}", v))
    if candidates:
        name, v = min(candidates, key=lambda x: x[1])
        return {"TP": v, "Source": name}

    swing_hi = _safe_float((dual_fib.get("auto_short", {}) or {}).get("swing_high"))
    if pd.notna(swing_hi) and swing_hi > entry:
        return {"TP": swing_hi, "Source": "SwingHighShort"}

    return {"TP": entry * 1.15, "Source": "Fallback15%"}

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any], fund_use_in_plan: bool) -> Dict[str, TradeSetup]:
    if df.empty:
        return {}

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))

    fib_short = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    res_zone = _safe_float(fib_short.get("61.8"), close * 1.05)
    sup_zone = _safe_float(fib_short.get("38.2"), close * 0.95)

    # buffer = max(0.5*ATR14, 0.4% entry)
    atr14 = _safe_float(last.get("ATR14"))
    def calc_buffer(entry: float) -> float:
        b1 = (0.5 * atr14) if pd.notna(atr14) else np.nan
        b2 = 0.004 * entry if pd.notna(entry) else np.nan
        vals = [v for v in [b1, b2] if pd.notna(v)]
        return max(vals) if vals else (0.006 * entry if pd.notna(entry) else np.nan)

    levels = _levels_from_ma_fib(last, dual_fib)

    setups: Dict[str, TradeSetup] = {}

    # --- Breakout ---
    entry_b = _round_price(res_zone * 1.01)
    anchor_name_b = _pick_anchor_below(entry_b, levels)
    anchor_b = _safe_float(levels.get(anchor_name_b)) if anchor_name_b else np.nan
    buf_b = calc_buffer(entry_b)
    stop_b = _round_price(anchor_b - buf_b) if pd.notna(anchor_b) and pd.notna(buf_b) else _round_price(entry_b * 0.97)

    # TP kỹ thuật
    tp_pack_b = _pick_tp_above(entry_b, dual_fib)
    tp_b = _round_price(_safe_float(tp_pack_b.get("TP")))
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    if pd.notna(rr_b) and rr_b >= 2.0:
        setups["Breakout"] = TradeSetup(
            "Breakout", entry_b, stop_b, tp_b, rr_b, "Cao",
            meta={"StopAnchor": anchor_name_b or "Fallback", "Buffer": buf_b, "TPSource": tp_pack_b.get("Source", "N/A")}
        )

    # --- Pullback ---
    entry_p = _round_price(sup_zone)
    anchor_name_p = _pick_anchor_below(entry_p, levels)
    anchor_p = _safe_float(levels.get(anchor_name_p)) if anchor_name_p else np.nan
    buf_p = calc_buffer(entry_p)
    stop_p = _round_price(anchor_p - buf_p) if pd.notna(anchor_p) and pd.notna(buf_p) else _round_price(entry_p * 0.97)

    tp_pack_p = _pick_tp_above(entry_p, dual_fib)
    tp_p = _round_price(_safe_float(tp_pack_p.get("TP")))
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    if pd.notna(rr_p) and rr_p >= 2.0:
        setups["Pullback"] = TradeSetup(
            "Pullback", entry_p, stop_p, tp_p, rr_p, "TB",
            meta={"StopAnchor": anchor_name_p or "Fallback", "Buffer": buf_p, "TPSource": tp_pack_p.get("Source", "N/A")}
        )

    # If Upside âm: vẫn giữ plan kỹ thuật; chỉ cấm GPT dùng target trong C/D (xử lý ở prompt/pack)
    return setups

def pick_primary_setup(trade_plans: Dict[str, TradeSetup]) -> Optional[str]:
    if not trade_plans:
        return None
    # choose highest RR
    items = [(k, v) for k, v in trade_plans.items() if pd.notna(v.rr)]
    if not items:
        return list(trade_plans.keys())[0]
    return max(items, key=lambda x: x[1].rr)[0]

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

def classify_scenario12(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    rules_hit = []

    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"; rules_hit.append("Trend=Up")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"; rules_hit.append("Trend=Down")
        else:
            trend = "Neutral"; rules_hit.append("Trend=Neutral")

    mom = "Neutral"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = "Bull"; rules_hit.append("Momentum=Bull")
        elif (rsi <= 45) and (macd_v < sig):
            mom = "Bear"; rules_hit.append("Momentum=Bear")
        elif (rsi >= 70):
            mom = "Exhaust"; rules_hit.append("Momentum=Exhaust")
        else:
            mom = "Neutral"; rules_hit.append("Momentum=Neutral")

    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg}")

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
        "RulesHit": rules_hit,
    }

# ============================================================
# 9C. MASTER SCORE (PYTHON-ONLY) — giữ nguyên logic cũ
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
        s_lv = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
        l_lv = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8")); s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8")); l_382 = _safe_float(l_lv.get("38.2"))

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
        if upside_pct >= 25:
            fcomp = 2.0
        elif upside_pct >= 15:
            fcomp = 1.5
        elif upside_pct >= 5:
            fcomp = 1.0
        else:
            fcomp = 0.5
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

    return {"Components": comps, "Total": round(total, 2), "Tier": tier, "PositionSizing": sizing, "BestRR": best_rr if pd.notna(best_rr) else np.nan}

# ============================================================
# 9D. RISK–REWARD SIMULATION PACK (PYTHON-ONLY)
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
        rows.append({"Setup": k, "Entry": entry, "Stop": stop, "TP": tp, "RR": rr, "RiskPct": risk_pct, "RewardPct": reward_pct, "Probability": s.probability})
        if pd.notna(rr):
            best_rr = rr if pd.isna(best_rr) else max(best_rr, rr)
    return {"Setups": rows, "BestRR": best_rr if pd.notna(best_rr) else np.nan}

# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Error": "Không đọc được dữ liệu Price_Vol.xlsx"}

    df = df_all[df_all["Ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {ticker}"}

    # Indicators
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    df["ATR14"] = atr_wilder(df, 14)

    dual_fib = compute_dual_fibonacci_auto(df, long_window=250)
    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)

    # Fundamental
    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].astype(str).str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    close = _safe_float(last.get("Close"))
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)

    # Normalize for upside calculation (Close often in "thousand" units)
    close_for_calc = close
    target_for_calc = target_vnd
    if pd.notna(close) and pd.notna(target_vnd):
        if (close < 500) and (target_vnd > 1000):
            target_for_calc = target_vnd / 1000.0
        elif (close > 1000) and (target_vnd < 500):
            target_for_calc = target_vnd * 1000.0

    upside_pct = ((target_for_calc - close_for_calc) / close_for_calc * 100) if (pd.notna(target_for_calc) and pd.notna(close_for_calc) and close_for_calc != 0) else np.nan

    fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct
    fund_row["TargetK"] = (target_vnd / 1000.0) if pd.notna(target_vnd) else np.nan

    # Rule: if upside < 0 => do NOT use fundamental target in Trade plan / R:R (GPT must not mention in C/D)
    fund_use_in_plan = bool(pd.notna(upside_pct) and upside_pct >= 0)

    # Trade plan (TP kỹ thuật + stop động)
    trade_plans = build_trade_plan(df, dual_fib, fund_use_in_plan)
    primary_setup = pick_primary_setup(trade_plans)

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

    # Pro features
    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    pa_feat = compute_price_action_features(df, dual_fib, vol_feat)

    analysis_pack = {
        "Ticker": ticker.upper(),
        "PrimarySetup": primary_setup,
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
            "ATR14": _safe_float(last.get("ATR14")),
        },
        "ScenarioBase": scenario,
        "Scenario12": scenario12,
        "Conviction": conviction,
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {}),
            "SelectionReason": dual_fib.get("selection_reason", "N/A"),
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": fund_row.get("TargetK", np.nan),
            "UpsidePct": upside_pct,
            "UseInPlan": fund_use_in_plan,
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(v.entry),
                "Stop": _safe_float(v.stop),
                "TP": _safe_float(v.tp),
                "RR": _safe_float(v.rr),
                "Probability": v.probability,
                "Meta": v.meta,
            }
            for k, v in trade_plans.items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "ProTech": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat,
            "PriceAction": pa_feat,
        },
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
        "AnalysisPack": analysis_pack,
    }

# ============================================================
# 11. GPT REPORT
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick = data["Ticker"]
    scenario = data["Scenario"]
    conviction = data["Conviction"]
    analysis_pack = data.get("AnalysisPack", {}) or {}

    last = data["Last"]
    close = _fmt_price(last.get("Close"))

    header_html = f"<h2 style='margin:0; padding:0; font-size:28px; line-height:1.2;'>{tick} — {close} | Điểm tin cậy: {conviction:.1f}/10 | {_scenario_vi(scenario)}</h2>"

    fund = analysis_pack.get("Fundamental", {}) or {}
    fund_text = (
        f"Khuyến nghị: {fund.get('Recommendation', 'N/A')} | "
        f"Giá mục tiêu: {_fmt_thousand(fund.get('TargetVND'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if fund else "Không có dữ liệu cơ bản"
    )

    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    prompt = f"""
Bạn là chuyên gia phân tích chứng khoán, nói chuyện thân thiện với “bạn”, văn phong mượt mà, rõ ràng.

QUY TẮC BẮT BUỘC:
- Tuyệt đối không bịa số.
- Tuyệt đối không tự tính bất kỳ con số nào.
- Chỉ dùng dữ liệu trong JSON “AnalysisPack”.
- Không dùng emoji, không dùng bullet 1️⃣2️⃣.
- Mỗi câu tối đa 1–2 con số, tránh liệt kê dày đặc.

FORMAT (BẮT BUỘC, KHÔNG THÊM/BỚT MỤC):
### A. Kỹ thuật
(đúng 8 ý, đánh số 1) đến 8))
1) MA Trend
2) RSI
3) MACD
4) RSI + MACD Bias
5) Fibonacci (2 khung Short/Long) — có thể nhắc SelectionReason ngắn gọn
6) Volume & Price Action (phải dùng ProTech.Volume + ProTech.PriceAction: anatomy/patterns/context)
7) Scenario 12
8) Master Integration (MasterScore + Conviction)

### B. Cơ bản
Dùng đúng dòng: {fund_text}

### C. Trade plan
- Chỉ được dùng setup “PrimarySetup” trong AnalysisPack.
- Nếu Fundamental.UseInPlan = false thì trong mục C tuyệt đối KHÔNG nhắc Target/Upside.
- Nêu rõ Entry/Stop/TP và diễn giải “TP kỹ thuật” + “stop động theo MA/Fibo + buffer” dựa trên Meta.

### D. Rủi ro vs lợi nhuận
- Chỉ được dùng RRSim của “PrimarySetup”.
- Nếu Fundamental.UseInPlan = false thì trong mục D tuyệt đối KHÔNG nhắc Target/Upside.
- Phải trích đúng RiskPct/RewardPct/RR/Probability theo setup đó.

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1600,
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return f"{header_html}\n\n{content}"

# ============================================================
# 12. STREAMLIT UI
# ============================================================

st.markdown("<h1 style='color:#A855F7; margin-bottom:6px;'>INCEPTION v4.8</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### Đăng nhập")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB").upper()
    run_btn = st.button("Phân tích", type="primary", use_container_width=True)

# ============================================================
# 13. MAIN EXECUTION
# ============================================================

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng. Vui lòng nhập lại.")
    else:
        with st.spinner(f"Đang xử lý phân tích {ticker_input}..."):
            try:
                result = analyze_ticker(ticker_input)
                report = generate_insight_report(result)
                st.markdown("<hr>", unsafe_allow_html=True)
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
    © 2025 INCEPTION Research Framework<br>
    Phiên bản 4.8 | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True,
)
