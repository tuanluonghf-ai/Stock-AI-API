# ============================================================
# INCEPTION v4.7 | Strategic Investor Edition
# app.py — Streamlit + GPT-4 Turbo
# Author: INCEPTION AI Research Framework
# Purpose: Technical–Fundamental Integrated Research Assistant
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(page_title="INCEPTION v4.7",
                   layout="wide",
                   page_icon="🟣")

st.markdown("""
<style>
body {
    background-color: #0B0E11;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
strong {
    color: #E5E7EB;
    font-weight: 700;
}
h1, h2, h3 {
    color: #E5E7EB;
}
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
    return f"{float(x)/1000:.{ndigits}f}"

def _safe_float(x, default=np.nan) -> float:
    try: return float(x)
    except: return default

def _round_price(x: float, ndigits: int = 2) -> float:
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

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
    """
    Detect last crossover event of series_a vs series_b within lookback.
    Returns: { "Event": "CrossUp"/"CrossDown"/"None", "BarsAgo": int or None }
    """
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
    """
    Simple divergence detector using last two local swing highs/lows within lookback.
    Returns: { "Type": "Bullish"/"Bearish"/"None", "Detail": str }
    """
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

def _scenario_vi(x: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng — Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng — Pha điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm — Yếu",
        "Neutral / Sideways": "Đi ngang / Trung tính",
    }
    return m.get(x, x)

def _pick_nearest_above(levels: List[float], ref: float) -> float:
    vals = [v for v in levels if pd.notna(v) and pd.notna(ref) and v > ref]
    return min(vals) if vals else np.nan

def _pick_nearest_below(levels: List[float], ref: float) -> float:
    vals = [v for v in levels if pd.notna(v) and pd.notna(ref) and v < ref]
    return max(vals) if vals else np.nan

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

    # normalize Target (handle "42,500" string, "42.5", etc.)
    # Rule: parse numeric; if parsed target is small (< 500) we assume it is already in thousand
    # and convert to VND by *1000 for storage consistency.
    tgt = pd.to_numeric(df["Target"], errors="coerce")
    df["Target"] = tgt
    df.loc[df["Target"].notna() & (df["Target"] < 500), "Target"] = df.loc[df["Target"].notna() & (df["Target"] < 500), "Target"] * 1000

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
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series([np.nan] * len(df), index=df.index)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# ============================================================
# 6. FIBONACCI DUAL-FRAME (AUTO SELECT 60 OR 90 + LONG 250)
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
        return {
            "short_window": None,
            "long_window": None,
            "auto_short": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "fixed_long": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "alt_short": {"window": None, "swing_high": np.nan, "swing_low": np.nan, "levels": {}},
            "selection_reason": "N/A"
        }

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

# ============================================================
# 6B. PRO TECH FEATURES (PYTHON-ONLY)
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
        return _safe_float(s.iloc[-1] - s.iloc[-(n+1)])

    s20 = slope(df["MA20"], 10)
    s50 = slope(df["MA50"], 10)
    s200 = slope(df["MA200"], 10)

    regime = "Neutral"
    if pd.notna(close) and pd.notna(ma50) and pd.notna(ma200):
        if close >= ma50 and ma50 >= ma200:
            regime = "Up"
        elif close < ma50 and ma50 < ma200:
            regime = "Down"
        else:
            regime = "Neutral"

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
    if df.empty:
        return {}

    rsi = df["RSI"].dropna()
    if rsi.empty:
        return {}

    last_rsi = _safe_float(rsi.iloc[-1])
    prev_rsi = _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else np.nan
    direction = "N/A"
    if pd.notna(last_rsi) and pd.notna(prev_rsi):
        delta = last_rsi - prev_rsi
        if delta > 1.0:
            direction = "Rising"
        elif delta < -1.0:
            direction = "Falling"
        else:
            direction = "Flat"

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

    return {
        "Value": last_rsi,
        "State": state,
        "Direction": direction,
        "Divergence": div
    }

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
        if last_m > last_s:
            state = "Bull"
        elif last_m < last_s:
            state = "Bear"

    cross = _find_last_cross(df["MACD"], df["MACDSignal"], lookback=30)

    zero = "N/A"
    if pd.notna(last_m):
        if last_m > 0.0:
            zero = "Above"
        elif last_m < 0.0:
            zero = "Below"
        else:
            zero = "Near"

    hist_state = "N/A"
    if len(hist) >= 4:
        h0 = _safe_float(hist.iloc[-1])
        h1 = _safe_float(hist.iloc[-2])
        h2 = _safe_float(hist.iloc[-3])
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
        "Divergence": div
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

def compute_market_context(df_all: pd.DataFrame) -> Dict[str, Any]:
    """
    Uses Price_Vol.xlsx for VNINDEX and VN30 (ticker names expected: VNINDEX, VN30).
    Fallback returns N/A if missing.
    """
    def pack(tick: str) -> Dict[str, Any]:
        d = df_all[df_all["Ticker"].astype(str).str.upper() == tick].copy()
        if d.empty or len(d) < 2:
            return {"Ticker": tick, "Close": np.nan, "ChangePct": np.nan, "Regime": "N/A"}
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
                if c >= ma50 and ma50 >= ma200:
                    regime = "Up"
                elif c < ma50 and ma50 < ma200:
                    regime = "Down"
                else:
                    regime = "Neutral"
        except:
            regime = "N/A"
        return {"Ticker": tick, "Close": c, "ChangePct": chg, "Regime": regime}

    vnindex = pack("VNINDEX")
    vn30 = pack("VN30")
    return {"VNINDEX": vnindex, "VN30": vn30}

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
# 8. TRADE PLAN LOGIC (TECH TP + DYNAMIC STOP BY MA/FIBO + BUFFER)
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
    if any(pd.isna([entry, stop, tp])) or entry <= stop or tp <= entry:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def _buffer_from_atr(last_atr: float, close: float) -> float:
    # Buffer derived from ATR; fallback is small fraction of close.
    if pd.notna(last_atr) and last_atr > 0:
        return float(last_atr) * 0.30
    if pd.notna(close) and close > 0:
        return float(close) * 0.005
    return np.nan

def _collect_fib_levels(dual_fib: Dict[str, Any], keys: List[str]) -> List[float]:
    out = []
    try:
        s_lv = dual_fib.get("auto_short", {}).get("levels", {})
        l_lv = dual_fib.get("fixed_long", {}).get("levels", {})
        for k in keys:
            out.append(_safe_float(s_lv.get(k)))
            out.append(_safe_float(l_lv.get(k)))
    except:
        pass
    return out

def _collect_swing_levels(dual_fib: Dict[str, Any]) -> List[float]:
    out = []
    try:
        out.append(_safe_float(dual_fib.get("auto_short", {}).get("swing_high")))
        out.append(_safe_float(dual_fib.get("auto_short", {}).get("swing_low")))
        out.append(_safe_float(dual_fib.get("fixed_long", {}).get("swing_high")))
        out.append(_safe_float(dual_fib.get("fixed_long", {}).get("swing_low")))
    except:
        pass
    return out

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty: return {}

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    last_atr = _safe_float(last.get("ATR14"))
    buf = _buffer_from_atr(last_atr, close)

    fib_short = dual_fib.get("auto_short", {}).get("levels", {}) or {}

    # Reference zones from short fib (keep original intent)
    res_zone = _safe_float(fib_short.get("61.8"))
    sup_zone = _safe_float(fib_short.get("38.2"))

    if pd.isna(res_zone) and pd.notna(close):
        res_zone = close * 1.05
    if pd.isna(sup_zone) and pd.notna(close):
        sup_zone = close * 0.95

    # Candidate levels
    fib_ret_keys = ["38.2", "50.0", "61.8"]
    fib_ext_keys = ["127.2", "161.8"]

    fib_retr = _collect_fib_levels(dual_fib, fib_ret_keys)
    fib_ext = _collect_fib_levels(dual_fib, fib_ext_keys)
    swings = _collect_swing_levels(dual_fib)

    ma_levels = [ma20, ma50, ma200]
    support_candidates = fib_retr + ma_levels + swings
    resistance_candidates = fib_ext + swings  # extensions + swing highs as tech targets

    setups = {}

    # === Breakout Setup ===
    entry_b = _round_price(res_zone * 1.01) if pd.notna(res_zone) else np.nan

    # stop anchor: nearest support below entry (MA/Fib/Swings), then minus buffer
    stop_anchor_b = _pick_nearest_below(support_candidates, entry_b)
    if pd.isna(stop_anchor_b):
        stop_anchor_b = _safe_float(dual_fib.get("auto_short", {}).get("swing_low"))
    stop_b = _round_price(stop_anchor_b - buf) if (pd.notna(stop_anchor_b) and pd.notna(buf)) else np.nan

    # TP: nearest resistance above entry (prefer fib extensions / swing highs); then minus small buffer not needed
    tp_anchor_b = _pick_nearest_above(resistance_candidates, entry_b)
    tp_b = _round_price(tp_anchor_b) if pd.notna(tp_anchor_b) else _round_price(entry_b * 1.25) if pd.notna(entry_b) else np.nan

    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    if pd.notna(rr_b) and rr_b >= 2.5:
        setups["Breakout"] = TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao")

    # === Pullback Setup ===
    entry_p = _round_price(sup_zone) if pd.notna(sup_zone) else np.nan

    # stop anchor: nearest support below entry; if none, use swing low; then minus buffer
    stop_anchor_p = _pick_nearest_below(support_candidates, entry_p)
    if pd.isna(stop_anchor_p):
        stop_anchor_p = _safe_float(dual_fib.get("auto_short", {}).get("swing_low"))
    stop_p = _round_price(stop_anchor_p - buf) if (pd.notna(stop_anchor_p) and pd.notna(buf)) else np.nan

    # TP: nearest resistance above entry from retracement ceiling + extensions + swing highs
    # For pullback, allow aiming back to a nearer retracement ceiling first (e.g., 50/61.8), otherwise extensions.
    pullback_res_candidates = _collect_fib_levels(dual_fib, ["50.0", "38.2", "61.8"]) + resistance_candidates
    tp_anchor_p = _pick_nearest_above(pullback_res_candidates, entry_p)
    tp_p = _round_price(tp_anchor_p) if pd.notna(tp_anchor_p) else _round_price(entry_p * 1.20) if pd.notna(entry_p) else np.nan

    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    if pd.notna(rr_p) and rr_p >= 2.5:
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
# 9B. 12-SCENARIO CLASSIFICATION (PYTHON-ONLY)
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

    rules_hit = []

    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"
            rules_hit.append("Trend=Up (Close>=MA50>=MA200)")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"
            rules_hit.append("Trend=Down (Close<MA50<MA200)")
        else:
            trend = "Neutral"
            rules_hit.append("Trend=Neutral (mixed MA structure)")

    mom = "Neutral"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = "Bull"
            rules_hit.append("Momentum=Bull (RSI>=55 & MACD>=Signal)")
        elif (rsi <= 45) and (macd_v < sig):
            mom = "Bear"
            rules_hit.append("Momentum=Bear (RSI<=45 & MACD<Signal)")
        elif (rsi >= 70):
            mom = "Exhaust"
            rules_hit.append("Momentum=Exhaust (RSI>=70)")
        else:
            mom = "Neutral"
            rules_hit.append("Momentum=Neutral (between zones)")

    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")

    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Bull": 0, "Neutral": 1, "Bear": 2, "Exhaust": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1  # 1..12

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
        "RulesHit": rules_hit
    }

# ============================================================
# 9C. MASTER INTEGRATION SCORE (PYTHON-ONLY)
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
            if c >= s_618:
                fibc += 1.2
            elif c >= s_382:
                fibc += 0.8
            else:
                fibc += 0.4

        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            if c >= l_618:
                fibc += 0.8
            elif c >= l_382:
                fibc += 0.5
            else:
                fibc += 0.2
    except:
        fibc = 0.0
    comps["Fibonacci"] = fibc

    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(s.rr)]
        best_rr = max(rrs) if rrs else np.nan
    rrcomp = 0.0
    if pd.notna(best_rr):
        if best_rr >= 4.0:
            rrcomp = 2.0
        elif best_rr >= 3.0:
            rrcomp = 1.5
        else:
            rrcomp = 1.0
    comps["RRQuality"] = rrcomp

    # Fundamental component (Upside) stays in MasterScore, but NOT for TradePlan/RR
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
        tier = "A+"
        sizing = "Aggressive (2.0x) if risk control ok"
    elif total >= 7.5:
        tier = "A"
        sizing = "Full size (1.0x) + consider pyramiding"
    elif total >= 6.0:
        tier = "B"
        sizing = "Medium size (0.6–0.8x)"
    elif total >= 4.5:
        tier = "C"
        sizing = "Small / tactical (0.3–0.5x)"
    else:
        tier = "D"
        sizing = "No edge / avoid or hedge"

    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": tier,
        "PositionSizing": sizing,
        "BestRR": best_rr if pd.notna(best_rr) else np.nan
    }

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

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Error": "Không đọc được dữ liệu Price_Vol.xlsx"}

    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {ticker}"}

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
    trade_plans = build_trade_plan(df, dual_fib)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)

    # Normalize for upside calculation:
    # - Price_Vol Close typically in "thousand" units (e.g., 30.5), while Target stored in VND (e.g., 42500).
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

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    market_ctx = compute_market_context(df_all)

    stock_chg = np.nan
    if len(df) >= 2:
        stock_chg = _pct_change(_safe_float(df.iloc[-1].get("Close")), _safe_float(df.iloc[-2].get("Close")))
    mkt_chg = _safe_float(market_ctx.get("VNINDEX", {}).get("ChangePct"))
    rel = "N/A"
    if pd.notna(stock_chg) and pd.notna(mkt_chg):
        if stock_chg > mkt_chg + 0.3:
            rel = "Stronger"
        elif stock_chg < mkt_chg - 0.3:
            rel = "Weaker"
        else:
            rel = "InLine"

    # Fundamental handling rule for TradePlan/RR: never use Target/Upside in TP/RR logic
    use_fundamental_in_trade_plan = False
    is_over_valued_vs_target = bool(pd.notna(upside_pct) and upside_pct < 0)

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
            "ATR14": _safe_float(last.get("ATR14")),
            "Volume": _safe_float(last.get("Volume")),
            "Avg20Vol": _safe_float(last.get("Avg20Vol")),
        },
        "ScenarioBase": scenario,
        "Scenario12": scenario12,
        "Conviction": conviction,
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {}),
            "AltShort": dual_fib.get("alt_short", {}),
            "SelectionReason": dual_fib.get("selection_reason", "N/A")
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": fund_row.get("TargetK", np.nan),
            "UpsidePct": upside_pct,
            "UseInTradePlan": use_fundamental_in_trade_plan,
            "IsOverValuedVsTarget": is_over_valued_vs_target
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
        "ProTech": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat
        },
        "Market": {
            "VNINDEX": market_ctx.get("VNINDEX", {}),
            "VN30": market_ctx.get("VN30", {}),
            "StockChangePct": stock_chg,
            "RelativeStrengthVsVNINDEX": rel
        }
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
# 11. GPT-4 TURBO STRATEGIC INSIGHT GENERATION
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick = data["Ticker"]
    scenario = data["Scenario"]
    conviction = data["Conviction"]
    analysis_pack = data.get("AnalysisPack", {})

    last = data["Last"]
    close = _fmt_price(last.get("Close"))
    header_html = f"<h2 style='margin:0; padding:0; font-size:26px; line-height:1.2;'>{tick} — {close} | Điểm tin cậy: {conviction:.1f}/10 | {_scenario_vi(scenario)}</h2>"

    fund = data.get("Fundamental", {})
    fund_text = (
        f"Khuyến nghị: {fund.get('Recommendation', 'N/A')} | "
        f"Giá mục tiêu: {_fmt_thousand(fund.get('Target'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if fund else "Không có dữ liệu cơ bản"
    )

    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    prompt = f"""
Bạn là chuyên gia phân tích chứng khoán, nói chuyện thân thiện với “bạn”, văn phong mượt mà, rõ ràng.
TUYỆT ĐỐI:
- Không bịa số.
- Không tự tính bất kỳ con số nào.
- Chỉ dùng đúng dữ liệu trong JSON “AnalysisPack”.

QUY TẮC VỀ ĐỊNH GIÁ CƠ BẢN (BẮT BUỘC):
- Target/Upside trong Fundamental chỉ dùng cho mục B (Cơ bản) và có thể nhắc trong mục A(8) như một lớp “rủi ro định giá”.
- TUYỆT ĐỐI KHÔNG dùng Target/Upside để suy luận Trade Plan hoặc R:R.
- Trade Plan + R:R chỉ dùng dữ liệu kỹ thuật trong JSON (đặc biệt TradePlans, RRSim, Fibonacci, ProTech, Last).

YÊU CẦU FORMAT OUTPUT:
- Không dùng emoji.
- Không dùng kiểu bullet 1️⃣2️⃣.
- Trình bày đúng 4 mục chính A–D như sau:

A. Kỹ thuật
1) ...
2) ...
3) ...
4) ...
5) ...
6) ...
7) ...
8) ...

B. Cơ bản
...

C. Trade plan
...

D. Rủi ro vs lợi nhuận
...

Gợi ý nội dung mục A (8 mục):
1) MA Trend (tận dụng ProTech.MA: Regime, Slope, Dist, Cross)
2) RSI (ProTech.RSI: Value, State, Direction, Divergence)
3) MACD (ProTech.MACD: State, Cross, ZeroLine, HistState, Divergence)
4) RSI + MACD Bias (kết hợp trạng thái đã có trong JSON, không tự tính)
5) Fibonacci (Fibonacci.ShortWindow & LongWindow + levels + SelectionReason)
6) Volume & Price Action (ProTech.Volume + Last; có thể nhắc ATR14 như bối cảnh biến động)
7) Scenario 12 (Scenario12)
8) Master Integration (MasterScore + Conviction; có thể nhắc Fundamental.IsOverValuedVsTarget nếu true, nhưng chỉ như “risk layer”, không kéo qua C/D)

Mục B: dùng đúng dòng dữ liệu: {fund_text}
Mục C: dùng TradePlans trong JSON, diễn giải ngắn gọn chiến lược phù hợp.
Mục D: dùng RRSim (RiskPct, RewardPct, RR, Probability).

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1600
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return f"{header_html}\n\n{content}"

# ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

st.markdown("<h1 style='color:#A855F7; margin-bottom:6px;'>INCEPTION v4.7</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB").upper()
    run_btn = st.button("Phân tích", type="primary", use_container_width=True)

col_main, = st.columns([1])

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
    Phiên bản 4.7 | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
)
