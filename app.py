# ============================================================
# INCEPTION v4.9 FULL | Strategic Investor Edition
# app.py — Streamlit + OpenAI
# Author: INCEPTION AI Research Framework
# Purpose: Python computes ALL numbers; GPT only narrates from AnalysisPack
# ============================================================

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(page_title="INCEPTION v4.9", layout="wide", page_icon="🟣")

st.markdown(
    """
<style>
body { background-color: #0B0E11; color: #E5E7EB; font-family: 'Segoe UI', sans-serif; }
strong { color: #E5E7EB; font-weight: 700; }
h1, h2, h3 { color: #E5E7EB; }
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
hr { border-top: 1px solid rgba(255,255,255,0.12); }
.codebox {
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 12px 14px;
}
.small { font-size: 12.5px; opacity: 0.9; }
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

DEFAULT_MODEL = "gpt-4o-mini"  # editable in sidebar

# ============================================================
# 3. HELPERS (SAFE + FORMATTING)
# ============================================================
def _safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return default


def _fmt_price(x, ndigits=2):
    if pd.isna(x):
        return ""
    return f"{float(x):,.{ndigits}f}"


def _fmt_thousand_vnd(x_vnd, ndigits=1):
    if pd.isna(x_vnd):
        return ""
    return f"{float(x_vnd)/1000:,.{ndigits}f}"


def _fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{float(x):.2f}%"


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


def _to_jsonable(obj: Any) -> Any:
    # Convert numpy/pandas types to json-friendly primitives
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


# ============================================================
# 3B. CROSS + DIVERGENCE (SIMPLE)
# ============================================================
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
# 4. LOADERS (ROBUST COLUMN NORMALIZATION)
# ============================================================
@st.cache_data
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"Lỗi khi đọc file {path}: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    colmap = {c: c.strip().lower() for c in df.columns}

    # normalize common VN headers
    rename = {}
    for c, cl in colmap.items():
        if cl in ["ngay", "date", "time", "datetime"]:
            rename[c] = "Date"
        elif cl in ["ma", "ticker", "symbol", "code"]:
            rename[c] = "Ticker"
        elif cl in ["open", "mo", "mở", "gia mo cua", "giá mở cửa"]:
            rename[c] = "Open"
        elif cl in ["high", "cao", "gia cao nhat", "giá cao nhất"]:
            rename[c] = "High"
        elif cl in ["low", "thap", "gia thap nhat", "giá thấp nhất"]:
            rename[c] = "Low"
        elif cl in ["close", "dong", "đóng", "gia dong cua", "giá đóng cửa", "price"]:
            rename[c] = "Close"
        elif cl in ["vol", "volume", "khoi luong", "khối lượng"]:
            rename[c] = "Volume"
    df.rename(columns=rename, inplace=True)

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values(["Ticker", "Date"])
    else:
        df = df.sort_values(["Ticker"]).copy()

    # if OHLC missing, synthesize from Close (minimal viable for indicators + fib proxy)
    if "Close" in df.columns:
        if "Open" not in df.columns:
            df["Open"] = df.groupby("Ticker")["Close"].shift(1)
        if "High" not in df.columns:
            df["High"] = df[["Open", "Close"]].max(axis=1)
        if "Low" not in df.columns:
            df["Low"] = df[["Open", "Close"]].min(axis=1)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        if c1 in ["recommendation", "khuyennghi", "khuyến nghị", "rating"]:
            rename_map[c] = "Recommendation"
    df.rename(columns=rename_map, inplace=True)

    if "Ticker" not in df.columns or "Target" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    if "Recommendation" not in df.columns:
        df["Recommendation"] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    tgt = pd.to_numeric(df["Target"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["Target"] = tgt

    # normalize: if numeric < 500 => treat as "thousand" and convert to VND
    m = df["Target"].notna() & (df["Target"] < 500)
    df.loc[m, "Target"] = df.loc[m, "Target"] * 1000.0

    out = df[["Ticker", "Target", "Recommendation"]].drop_duplicates(subset=["Ticker"], keep="last")
    return out


@st.cache_data
def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name"])

    colmap = {c: c.strip().lower() for c in df.columns}
    rename = {}
    for c, cl in colmap.items():
        if cl in ["ticker", "ma", "symbol", "code"]:
            rename[c] = "Ticker"
        if cl in ["name", "ten", "tên", "company", "companyname", "doanh nghiep", "doanh nghiệp"]:
            rename[c] = "Name"
    df.rename(columns=rename, inplace=True)

    if "Ticker" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Name"])
    if "Name" not in df.columns:
        df["Name"] = ""

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Name"] = df["Name"].astype(str).fillna("")
    return df[["Ticker", "Name"]].drop_duplicates(subset=["Ticker"], keep="last")


# ============================================================
# 5. INDICATORS
# ============================================================
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    else:
        tr = df["Close"].diff().abs()
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)

    df["RSI"] = rsi_wilder(df["Close"], 14)

    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"] = m
    df["MACDSignal"] = s
    df["MACDHist"] = h

    df["Avg20Vol"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else np.nan
    df["ATR14"] = atr_wilder(df, 14)
    return df


# ============================================================
# 6. FIBONACCI DUAL-FRAME (AUTO SHORT 60/90 + FIXED LONG 250)
# ============================================================
def _fib_levels(low: float, high: float) -> Dict[str, float]:
    rng = high - low
    if pd.isna(rng) or rng <= 0:
        return {}
    return {
        "23.6": high - 0.236 * rng,
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng,
        "78.6": high - 0.786 * rng,
        "88.6": high - 0.886 * rng,
        "127.2": high + 0.272 * rng,
        "161.8": high + 0.618 * rng,
    }


def _compute_fib_window(df: pd.DataFrame, w: int) -> Dict[str, Any]:
    L = w if len(df) >= w else len(df)
    win = df.tail(L)
    hi = win["High"].max() if "High" in win.columns else win["Close"].max()
    lo = win["Low"].min() if "Low" in win.columns else win["Close"].min()
    return {"window": int(L), "swing_high": float(hi) if pd.notna(hi) else np.nan, "swing_low": float(lo) if pd.notna(lo) else np.nan, "levels": _fib_levels(lo, hi)}


def _score_fib_relevance(close: float, fib: Dict[str, Any]) -> float:
    lv = fib.get("levels", {}) or {}
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

    return float(score)


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
        "short_window": int(chosen.get("window")),
        "long_window": int(L_long),
        "auto_short": {"swing_high": chosen.get("swing_high"), "swing_low": chosen.get("swing_low"), "levels": chosen.get("levels", {})},
        "fixed_long": {"swing_high": float(l_hi) if pd.notna(l_hi) else np.nan, "swing_low": float(l_lo) if pd.notna(l_lo) else np.nan, "levels": _fib_levels(l_lo, l_hi)},
        "alt_short": alt,
        "selection_reason": reason,
    }


# ============================================================
# 6B. FEATURES PACK (MA/RSI/MACD/VOLUME/PRICE ACTION)
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


def compute_price_action_features(df: pd.DataFrame, dual_fib: Dict[str, Any], vol_feat: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty or len(df) < 2:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2]
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

    patterns: List[str] = []
    if pd.notna(body_pct) and body_pct <= 15:
        patterns.append("Doji")

    if pd.notna(lower_pct) and pd.notna(body_pct) and pd.notna(upper_pct):
        if (lower_pct >= 55) and (body_pct <= 35) and (upper_pct <= 20):
            patterns.append("Hammer/PinBarBull")
        if (upper_pct >= 55) and (body_pct <= 35) and (lower_pct <= 20):
            patterns.append("ShootingStar/PinBarBear")

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

    if has_ohlc and pd.notna(h) and pd.notna(l) and pd.notna(ph) and pd.notna(pl):
        if (h < ph) and (l > pl):
            patterns.append("InsideBar")
        if (h > ph) and (l < pl):
            patterns.append("OutsideBar")

    near_ma = []
    for name in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(name))
        if pd.notna(v) and pd.notna(c) and c != 0:
            if abs(c - v) / c <= 0.012:
                near_ma.append(name)

    near_fib = []
    fib_short = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    fib_long = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}
    for tag, lv in [("S", fib_short), ("L", fib_long)]:
        for k in ["38.2", "50.0", "61.8", "78.6"]:
            v = _safe_float(lv.get(k))
            if pd.notna(v) and pd.notna(c) and c != 0:
                if abs(c - v) / c <= 0.012:
                    near_fib.append(f"{tag}{k}")

    signal = "Neutral"
    if any(p in patterns for p in ["BullEngulfing", "Hammer/PinBarBull"]) and not any(
        p in patterns for p in ["BearEngulfing", "ShootingStar/PinBarBear"]
    ):
        signal = "Bull"
    elif any(p in patterns for p in ["BearEngulfing", "ShootingStar/PinBarBear"]) and not any(
        p in patterns for p in ["BullEngulfing", "Hammer/PinBarBull"]
    ):
        signal = "Bear"

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
# 7. CONVICTION SCORE (SIMPLE BASELINE)
# ============================================================
def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if pd.notna(last.get("Close")) and pd.notna(last.get("MA200")) and last["Close"] > last["MA200"]:
        score += 2.0
    if pd.notna(last.get("RSI")) and last["RSI"] > 55:
        score += 1.0
    if pd.notna(last.get("Volume")) and pd.notna(last.get("Avg20Vol")) and last["Volume"] > last["Avg20Vol"]:
        score += 1.0
    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACDSignal")) and last["MACD"] > last["MACDSignal"]:
        score += 0.5
    return float(min(10.0, score))


# ============================================================
# 8. TRADE PLAN LOGIC (STOP DYNAMIC: MA/FIB + BUFFER)
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
    lv: Dict[str, float] = {}
    for k in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(k))
        if pd.notna(v):
            lv[k] = v

    s = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    l = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}
    for k in ["23.6", "38.2", "50.0", "61.8", "78.6", "88.6"]:
        v = _safe_float(s.get(k))
        if pd.notna(v):
            lv[f"FibS{k}"] = v
    for k in ["23.6", "38.2", "50.0", "61.8", "78.6", "88.6"]:
        v = _safe_float(l.get(k))
        if pd.notna(v):
            lv[f"FibL{k}"] = v

    return lv


def _pick_anchor_below(entry: float, levels: Dict[str, float]) -> Optional[str]:
    below = [(name, v) for name, v in levels.items() if pd.notna(v) and v < entry]
    if not below:
        return None
    return max(below, key=lambda x: x[1])[0]


def _pick_tp_above(entry: float, dual_fib: Dict[str, Any], last_close: float) -> Dict[str, Any]:
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

    # last fallback: modest +10% (avoid aggressive)
    if pd.notna(last_close):
        return {"TP": entry * 1.10, "Source": "Fallback10%"}
    return {"TP": entry * 1.10, "Source": "Fallback10%"}


def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty:
        return {}

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    atr14 = _safe_float(last.get("ATR14"))

    fib_short = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    # prefer 61.8 as "ceiling" for breakout trigger, 38.2 as pullback buy-zone
    res_zone = _safe_float(fib_short.get("61.8"), close * 1.05)
    sup_zone = _safe_float(fib_short.get("38.2"), close * 0.95)

    def calc_buffer(entry: float) -> float:
        b1 = (0.5 * atr14) if pd.notna(atr14) else np.nan
        b2 = 0.004 * entry if pd.notna(entry) else np.nan
        vals = [v for v in [b1, b2] if pd.notna(v)]
        return max(vals) if vals else (0.006 * entry if pd.notna(entry) else np.nan)

    levels = _levels_from_ma_fib(last, dual_fib)
    setups: Dict[str, TradeSetup] = {}

    # --- Breakout setup ---
    entry_b = _round_price(res_zone * 1.01)
    anchor_name_b = _pick_anchor_below(entry_b, levels)
    anchor_b = _safe_float(levels.get(anchor_name_b)) if anchor_name_b else np.nan
    buf_b = calc_buffer(entry_b)
    stop_b = _round_price(anchor_b - buf_b) if pd.notna(anchor_b) and pd.notna(buf_b) else _round_price(entry_b * 0.97)

    tp_pack_b = _pick_tp_above(entry_b, dual_fib, close)
    tp_b = _round_price(_safe_float(tp_pack_b.get("TP")))
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    if pd.notna(rr_b) and rr_b >= 2.0:
        setups["Breakout"] = TradeSetup(
            "Breakout", entry_b, stop_b, tp_b, float(rr_b), "Cao",
            meta={"StopAnchor": anchor_name_b or "Fallback", "Buffer": float(buf_b) if pd.notna(buf_b) else None, "TPSource": tp_pack_b.get("Source", "N/A")}
        )

    # --- Pullback setup ---
    entry_p = _round_price(sup_zone)
    anchor_name_p = _pick_anchor_below(entry_p, levels)
    anchor_p = _safe_float(levels.get(anchor_name_p)) if anchor_name_p else np.nan
    buf_p = calc_buffer(entry_p)
    stop_p = _round_price(anchor_p - buf_p) if pd.notna(anchor_p) and pd.notna(buf_p) else _round_price(entry_p * 0.97)

    tp_pack_p = _pick_tp_above(entry_p, dual_fib, close)
    tp_p = _round_price(_safe_float(tp_pack_p.get("TP")))
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    if pd.notna(rr_p) and rr_p >= 2.0:
        setups["Pullback"] = TradeSetup(
            "Pullback", entry_p, stop_p, tp_p, float(rr_p), "TB",
            meta={"StopAnchor": anchor_name_p or "Fallback", "Buffer": float(buf_p) if pd.notna(buf_p) else None, "TPSource": tp_pack_p.get("Source", "N/A")}
        )

    return setups


def pick_primary_setup(trade_plans: Dict[str, TradeSetup]) -> Optional[str]:
    if not trade_plans:
        return None
    items = [(k, v) for k, v in trade_plans.items() if pd.notna(v.rr)]
    if not items:
        return list(trade_plans.keys())[0]
    return max(items, key=lambda x: x[1].rr)[0]


def compute_rrsim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    out = {}
    for k, v in (trade_plans or {}).items():
        entry, stop, tp = _safe_float(v.entry), _safe_float(v.stop), _safe_float(v.tp)
        risk_pct = ((entry - stop) / entry * 100) if (pd.notna(entry) and pd.notna(stop) and entry != 0) else np.nan
        reward_pct = ((tp - entry) / entry * 100) if (pd.notna(entry) and pd.notna(tp) and entry != 0) else np.nan
        out[k] = {
            "Entry": entry, "Stop": stop, "TP": tp,
            "RiskPct": risk_pct, "RewardPct": reward_pct,
            "RR": _safe_float(v.rr),
            "Probability": v.probability,
            "Meta": v.meta or {},
        }
    return out


# ============================================================
# 9. SCENARIO CLASSIFICATION (4 + 12)
# ============================================================
def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last.get("Close"), last.get("MA20"), last.get("MA50"), last.get("MA200")
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
        ("Up", "Bull"): "S1 – Uptrend + Bullish Momentum",
        ("Up", "Neutral"): "S2 – Uptrend + Neutral Momentum",
        ("Up", "Bear"): "S3 – Uptrend + Bearish Pullback",
        ("Up", "Exhaust"): "S4 – Uptrend + Overbought/Exhaust",
        ("Neutral", "Bull"): "S5 – Range + Bullish Attempt",
        ("Neutral", "Neutral"): "S6 – Range + Balanced",
        ("Neutral", "Bear"): "S7 – Range + Bearish Pressure",
        ("Neutral", "Exhaust"): "S8 – Range + Overbought Risk",
        ("Down", "Bull"): "S9 – Downtrend + Short-covering Bounce",
        ("Down", "Neutral"): "S10 – Downtrend + Weak Stabilization",
        ("Down", "Bear"): "S11 – Downtrend + Bearish Momentum",
        ("Down", "Exhaust"): "S12 – Downtrend + Overbought Rebound Risk",
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
# 10. FUNDAMENTAL PACK FROM LOCAL TARGET FILES (NO WEB)
# ============================================================
def build_fundamental_pack(ticker: str, close_vnd: float, targets_df: pd.DataFrame) -> Dict[str, Any]:
    if targets_df is None or targets_df.empty:
        return {"HasData": False, "TargetVND": np.nan, "UpsidePct": np.nan, "Recommendation": ""}

    row = targets_df[targets_df["Ticker"] == ticker]
    if row.empty:
        return {"HasData": False, "TargetVND": np.nan, "UpsidePct": np.nan, "Recommendation": ""}

    tgt = _safe_float(row.iloc[-1].get("Target"))
    rec = str(row.iloc[-1].get("Recommendation") or "").strip()

    upside = _pct_change(tgt, close_vnd) if (pd.notna(tgt) and pd.notna(close_vnd) and close_vnd != 0) else np.nan
    return {"HasData": True, "TargetVND": tgt, "UpsidePct": upside, "Recommendation": rec}


# ============================================================
# 11. NARRATIVE HINTS (PYTHON PRE-DIGEST)
# ============================================================
def build_narratives(ticker: str, name: str, last: pd.Series, ma_feat: Dict[str, Any], rsi_feat: Dict[str, Any],
                     macd_feat: Dict[str, Any], vol_feat: Dict[str, Any], pa_feat: Dict[str, Any],
                     scenario4: str, scenario12: Dict[str, Any], dual_fib: Dict[str, Any],
                     fund: Dict[str, Any], primary_key: Optional[str], rrsim: Dict[str, Any]) -> Dict[str, str]:
    close = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi_v = _safe_float(rsi_feat.get("Value"))
    vol_ratio = _safe_float(vol_feat.get("Ratio"))
    macd_state = (macd_feat.get("State") or "N/A")

    trend_hint = f"TrendRegime={ma_feat.get('Regime','N/A')}; PriceVsMA200={'Above' if (pd.notna(close) and pd.notna(ma200) and close>=ma200) else 'Below/NA'}; MA20/50/200 slopes={ma_feat.get('SlopeMA20','N/A')}/{ma_feat.get('SlopeMA50','N/A')}/{ma_feat.get('SlopeMA200','N/A')}."
    rsi_hint = f"RSIState={rsi_feat.get('State','N/A')}; RSIDirection={rsi_feat.get('Direction','N/A')}; RSIDiv={((rsi_feat.get('Divergence') or {}).get('Type','None'))}."
    macd_hint = f"MACDState={macd_state}; ZeroLine={macd_feat.get('ZeroLine','N/A')}; HistState={macd_feat.get('HistState','N/A')}."
    vol_hint = f"VolumeRegime={vol_feat.get('Regime','N/A')}; Vol/Avg20≈{(round(vol_ratio,2) if pd.notna(vol_ratio) else 'N/A')}x."
    pa_hint = f"PA={pa_feat.get('Signal','Neutral')}; Patterns={', '.join(pa_feat.get('Patterns',[])[:3]) if pa_feat.get('Patterns') else 'None'}; ContextNear={'/'.join((pa_feat.get('Context') or {}).get('NearMA',[])[:2]) or 'None'}."

    fib_short_w = dual_fib.get("short_window")
    fib_reason = dual_fib.get("selection_reason", "N/A")
    fib_hint = f"FibShort={fib_short_w}({fib_reason}); use long(250) as structure reference."

    fund_hint = "Fund=N/A"
    if fund.get("HasData"):
        ups = _safe_float(fund.get("UpsidePct"))
        # 1 number only: upside
        fund_hint = f"FundTargetAvailable; Upside≈{(round(ups,1) if pd.notna(ups) else 'N/A')}%."

    rr_hint = "TradePlan: No eligible RR>=2.0 setups."
    if primary_key and primary_key in rrsim:
        p = rrsim[primary_key]
        rp = _safe_float(p.get("RiskPct"))
        rw = _safe_float(p.get("RewardPct"))
        rr = _safe_float(p.get("RR"))
        # keep 2 numbers: risk and reward; RR optional as 3rd number (avoid)
        rr_hint = f"PrimarySetup={primary_key}; Risk≈{(round(rp,1) if pd.notna(rp) else 'N/A')}%; Reward≈{(round(rw,1) if pd.notna(rw) else 'N/A')}%."

    head = f"{ticker} {('- ' + name) if name else ''} | Scenario4={scenario4} | Scenario12={scenario12.get('Name','N/A')}."
    return {
        "Headline": head,
        "TrendHint": trend_hint,
        "RSIHint": rsi_hint,
        "MACDHint": macd_hint,
        "VolumeHint": vol_hint,
        "PriceActionHint": pa_hint,
        "FibHint": fib_hint,
        "FundHint": fund_hint,
        "TradeHint": rr_hint,
    }


# ============================================================
# 12. MASTER INTEGRATION FRAME (CONVICTION WEIGHTING - LIGHT)
# (kept compact; Python-only; GPT just narrates result)
# ============================================================
def compute_master_integration_score(ma_feat: Dict[str, Any], rsi_feat: Dict[str, Any], macd_feat: Dict[str, Any],
                                     vol_feat: Dict[str, Any], pa_feat: Dict[str, Any]) -> Dict[str, Any]:
    pts = 0.0
    hit = []

    # Trend ×3
    if ma_feat.get("Regime") == "Up" and (ma_feat.get("Structure", {}) or {}).get("MA50AboveMA200", False):
        pts += 3.0; hit.append("Trend×3")
    elif ma_feat.get("Regime") == "Down":
        pts += 0.6; hit.append("TrendWeak")

    # RSI ×1.5
    if rsi_feat.get("State") in ["Bull", "Neutral+"]:
        pts += 1.5; hit.append("RSI×1.5")
    elif rsi_feat.get("State") in ["Bear", "Oversold"]:
        pts += 0.6; hit.append("RSIWeak")

    # MACD ×1.8
    if macd_feat.get("State") == "Bull" and macd_feat.get("ZeroLine") == "Above":
        pts += 1.8; hit.append("MACD×1.8")
    elif macd_feat.get("State") == "Bear":
        pts += 0.7; hit.append("MACDWeak")

    # Volume ×1.2
    if vol_feat.get("Regime") in ["High", "Spike"]:
        pts += 1.2; hit.append("VOL×1.2")
    elif vol_feat.get("Regime") == "Low":
        pts += 0.6; hit.append("VOLLOW")

    # Price action ×1.2
    if pa_feat.get("Signal") == "Bull":
        pts += 1.2; hit.append("PA×1.2")
    elif pa_feat.get("Signal") == "Bear":
        pts += 0.7; hit.append("PAWeak")

    # Map to tier
    tier = "≤2: no edge"
    size = "Flat/tiny"
    if pts >= 6.0:
        tier = "6–7: God-tier"; size = "2–3x size (only if risk rules ok)"
    elif pts >= 5.0:
        tier = "5: very high"; size = "Full size + pyramid"
    elif pts >= 4.0:
        tier = "4: high"; size = "Full size"
    elif pts >= 3.0:
        tier = "3: tradeable"; size = "50–70% size"

    return {"Points": float(round(pts, 2)), "Tier": tier, "SizeGuide": size, "Hits": hit}


# ============================================================
# 13. ANALYSISPACK (SINGLE SOURCE OF TRUTH FOR GPT)
# ============================================================
def build_analysis_pack(ticker: str, name: str, df_ticker: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
    df_ticker = add_indicators(df_ticker)
    last = df_ticker.iloc[-1].copy()

    dual_fib = compute_dual_fibonacci_auto(df_ticker, 250)
    ma_feat = compute_ma_features(df_ticker)
    rsi_feat = compute_rsi_features(df_ticker)
    macd_feat = compute_macd_features(df_ticker)
    vol_feat = compute_volume_features(df_ticker)
    pa_feat = compute_price_action_features(df_ticker, dual_fib, vol_feat)

    scenario4 = classify_scenario(last)
    scenario12 = classify_scenario12(last)

    conviction = compute_conviction(last)
    fund = build_fundamental_pack(ticker, _safe_float(last.get("Close")), targets_df)

    trade_plans = build_trade_plan(df_ticker, dual_fib)
    primary_key = pick_primary_setup(trade_plans)
    rrsim = compute_rrsim(trade_plans)
    primary_rr = rrsim.get(primary_key) if primary_key else None

    master = compute_master_integration_score(ma_feat, rsi_feat, macd_feat, vol_feat, pa_feat)

    narratives = build_narratives(
        ticker=ticker, name=name, last=last, ma_feat=ma_feat, rsi_feat=rsi_feat, macd_feat=macd_feat, vol_feat=vol_feat,
        pa_feat=pa_feat, scenario4=scenario4, scenario12=scenario12, dual_fib=dual_fib, fund=fund,
        primary_key=primary_key, rrsim=rrsim
    )

    # Guard D: only allow these exact values in section D
    guard_d = {}
    if primary_rr:
        rp = _safe_float(primary_rr.get("RiskPct"))
        rw = _safe_float(primary_rr.get("RewardPct"))
        rr = _safe_float(primary_rr.get("RR"))
        guard_d = {
            "PrimarySetup": primary_key,
            "RiskPct": float(round(rp, 2)) if pd.notna(rp) else None,
            "RewardPct": float(round(rw, 2)) if pd.notna(rw) else None,
            "RR": float(round(rr, 2)) if pd.notna(rr) else None,
            "Entry": float(round(_safe_float(primary_rr.get("Entry")), 2)) if pd.notna(_safe_float(primary_rr.get("Entry"))) else None,
            "Stop": float(round(_safe_float(primary_rr.get("Stop")), 2)) if pd.notna(_safe_float(primary_rr.get("Stop"))) else None,
            "TP": float(round(_safe_float(primary_rr.get("TP")), 2)) if pd.notna(_safe_float(primary_rr.get("TP"))) else None,
        }

    pack = {
        "Meta": {
            "AppVersion": "4.9",
            "GeneratedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "Ticker": ticker,
        "Name": name or "",
        "LastBar": {
            "Date": (df_ticker.iloc[-1].get("Date").isoformat() if "Date" in df_ticker.columns and pd.notna(df_ticker.iloc[-1].get("Date")) else ""),
            "Close": _safe_float(last.get("Close")),
            "Volume": _safe_float(last.get("Volume")),
        },
        "IndicatorsLast": {
            "MA20": _safe_float(last.get("MA20")),
            "MA50": _safe_float(last.get("MA50")),
            "MA200": _safe_float(last.get("MA200")),
            "RSI": _safe_float(last.get("RSI")),
            "MACD": _safe_float(last.get("MACD")),
            "MACDSignal": _safe_float(last.get("MACDSignal")),
            "MACDHist": _safe_float(last.get("MACDHist")),
            "Avg20Vol": _safe_float(last.get("Avg20Vol")),
            "ATR14": _safe_float(last.get("ATR14")),
        },
        "DualFib": dual_fib,
        "Features": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat,
            "PriceAction": pa_feat,
        },
        "Scenario": {
            "Scenario4": scenario4,
            "Scenario4VI": _scenario_vi(scenario4),
            "Scenario12": scenario12,
        },
        "Scores": {
            "Conviction10": conviction,
            "MasterIntegration": master,
        },
        "Fundamental": fund,
        "TradePlans": {k: _to_jsonable(v.__dict__) for k, v in (trade_plans or {}).items()},
        "RRSim": rrsim,
        "PrimarySetup": primary_key,
        "PrimaryRR": primary_rr or {},
        "GuardD": guard_d,
        "Narratives": narratives,
    }
    return _to_jsonable(pack)


# ============================================================
# 14. GPT PROMPT (STRICT A–D + GUARD D)
# ============================================================
def build_system_prompt() -> str:
    return (
        "You are INCEPTION Narrative Engine.\n"
        "Python is the single source of truth for ALL calculations.\n"
        "You MUST only narrate based on the provided AnalysisPack JSON.\n"
        "You MUST output exactly 4 main sections A–D with headings:\n"
        "### A. Phân tích Kỹ thuật\n"
        "### B. Phân tích Cơ bản\n"
        "### C. Kịch bản & Hành động\n"
        "### D. Rủi ro vs Lợi nhuận\n"
        "\n"
        "Critical constraints:\n"
        "1) No new calculations, no invented numbers.\n"
        "2) Each sentence max 1–2 numeric values.\n"
        "3) Section D: Only use numbers from AnalysisPack.GuardD and AnalysisPack.PrimaryRR.\n"
        "   Do NOT output any other risk/reward % or stop/TP beyond those.\n"
        "4) If Fundamental.HasData is false, say 'Chưa có dữ liệu mục tiêu (file nội bộ)'.\n"
        "5) Prefer concise, actionable wording.\n"
    )


def build_user_prompt(analysis_pack: Dict[str, Any]) -> str:
    # Force GPT to use narratives first; still has full pack for referencing exact values.
    return (
        "ANALYSISPACK_JSON:\n"
        f"{json.dumps(analysis_pack, ensure_ascii=False)}\n\n"
        "Write the report in Vietnamese, professional tone.\n"
        "Use AnalysisPack.Narratives as the primary language anchors.\n"
        "Do not dump raw tables.\n"
        "In A: cover MA/RSI/MACD/Volume/PriceAction/Fibo in 8 sub-bullets (no numbering needed).\n"
        "In C: include (1) If holding: what to do now; (2) If want to buy: what zones to watch and why.\n"
        "In D: explicitly state PrimarySetup name and provide Entry/Stop/TP + Risk% + Reward% using ONLY GuardD.\n"
        "End D with a single position-sizing suggestion based on MasterIntegration tier.\n"
    )


# ============================================================
# 15. GUARD D VALIDATION + RETRY
# ============================================================
def _extract_section_d(text: str) -> str:
    # grab from "### D." to end
    m = re.search(r"(###\s*D\.[\s\S]*)$", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_percent_numbers(text: str) -> List[float]:
    nums = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
        nums.append(_safe_float(m.group(1)))
    return [x for x in nums if pd.notna(x)]


def _near(a: float, b: float, tol: float = 0.25) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


def validate_guard_d(output_text: str, guard_d: Dict[str, Any]) -> Tuple[bool, str]:
    if not guard_d or not guard_d.get("PrimarySetup"):
        # No plan => allow, but ensure D doesn't fabricate risk/reward
        d = _extract_section_d(output_text)
        pct = _extract_percent_numbers(d)
        if pct:
            return False, "D has % numbers but no PrimarySetup. Must not output risk/reward%."
        return True, ""

    d = _extract_section_d(output_text)
    if not d:
        return False, "Missing section D."

    risk_ok = True
    reward_ok = True

    risk_true = _safe_float(guard_d.get("RiskPct"))
    reward_true = _safe_float(guard_d.get("RewardPct"))

    pcts = _extract_percent_numbers(d)

    # D must contain exactly these (risk and reward) at least once; and must not contain other % values
    found_risk = any(_near(x, risk_true) for x in pcts) if pd.notna(risk_true) else True
    found_reward = any(_near(x, reward_true) for x in pcts) if pd.notna(reward_true) else True

    # Disallow extra percent values that are not near risk/reward
    extra = []
    for x in pcts:
        ok = False
        if pd.notna(risk_true) and _near(x, risk_true):
            ok = True
        if pd.notna(reward_true) and _near(x, reward_true):
            ok = True
        if not ok:
            extra.append(x)

    risk_ok = found_risk
    reward_ok = found_reward

    if not (risk_ok and reward_ok) or extra:
        msg = []
        if not risk_ok:
            msg.append("Risk% mismatch/missing")
        if not reward_ok:
            msg.append("Reward% mismatch/missing")
        if extra:
            msg.append(f"Extra % values in D: {extra[:6]}")
        return False, " | ".join(msg)

    return True, ""


def repair_with_guard(client: OpenAI, model: str, system_prompt: str, analysis_pack: Dict[str, Any], bad_output: str, guard_d: Dict[str, Any]) -> str:
    # Repair only D, keep A–C unchanged
    g = guard_d or {}
    repair_inst = (
        "You must repair ONLY section D.\n"
        "Do NOT change A/B/C.\n"
        "Section D must use ONLY these values:\n"
        f"- PrimarySetup: {g.get('PrimarySetup')}\n"
        f"- Entry: {g.get('Entry')}\n"
        f"- Stop: {g.get('Stop')}\n"
        f"- TP: {g.get('TP')}\n"
        f"- RiskPct: {g.get('RiskPct')}\n"
        f"- RewardPct: {g.get('RewardPct')}\n"
        f"- RR: {g.get('RR')}\n"
        "No other percent values are allowed in D.\n"
        "Keep each sentence max 1–2 numbers.\n"
        "Return the full report A–D.\n"
    )

    user = (
        "ANALYSISPACK_JSON:\n"
        f"{json.dumps(analysis_pack, ensure_ascii=False)}\n\n"
        "BAD_OUTPUT:\n"
        f"{bad_output}\n\n"
        "REPAIR_INSTRUCTION:\n"
        f"{repair_inst}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def run_gpt_report(api_key: str, model: str, analysis_pack: Dict[str, Any]) -> str:
    client = OpenAI(api_key=api_key)
    sys_p = build_system_prompt()
    usr_p = build_user_prompt(analysis_pack)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.25,
        messages=[
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
    )
    text = resp.choices[0].message.content

    ok, reason = validate_guard_d(text, analysis_pack.get("GuardD") or {})
    if ok:
        return text

    # one repair attempt (hard-guard)
    repaired = repair_with_guard(client, model, sys_p, analysis_pack, text, analysis_pack.get("GuardD") or {})
    ok2, _ = validate_guard_d(repaired, analysis_pack.get("GuardD") or {})
    return repaired if ok2 else repaired


# ============================================================
# 16. ACCESS KEY QUOTA (SESSION)
# ============================================================
def _init_quota_state():
    if "quota_state" not in st.session_state:
        st.session_state["quota_state"] = {k: int(v.get("quota", 0)) for k, v in VALID_KEYS.items()}
    if "active_user_key" not in st.session_state:
        st.session_state["active_user_key"] = ""


def _quota_remaining(user_key: str) -> int:
    _init_quota_state()
    return int(st.session_state["quota_state"].get(user_key, 0))


def _consume_quota(user_key: str, n: int = 1) -> bool:
    _init_quota_state()
    rem = _quota_remaining(user_key)
    if rem < n:
        return False
    st.session_state["quota_state"][user_key] = rem - n
    return True


# ============================================================
# 17. UI
# ============================================================
st.title("INCEPTION v4.9 — Technical/Fundamental Analyzer (Python Truth + GPT Narrative)")

_init_quota_state()

with st.sidebar:
    st.subheader("Access")
    user_key = st.text_input("User Key", value=st.session_state.get("active_user_key", ""), type="password")
    if user_key:
        st.session_state["active_user_key"] = user_key

    valid_user = user_key in VALID_KEYS
    if valid_user:
        st.success(f"OK: {VALID_KEYS[user_key]['name']} | Remaining: {_quota_remaining(user_key)}")
    else:
        st.info("Nhập User Key để mở quota.")

    st.divider()
    st.subheader("OpenAI")
    api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption("Model có thể thay đổi theo account/API của bạn.")

    st.divider()
    st.subheader("Input Options")
    use_virtual_bar = st.checkbox("Append virtual latest bar (Price/Volume)", value=False)
    v_price = st.text_input("Virtual Price (VND or 'thousand' style like 42.2)", value="")
    v_volume = st.text_input("Virtual Volume", value="")

    show_pack = st.checkbox("Show AnalysisPack JSON (debug)", value=False)

# Load base data
df_all = load_price_vol(PRICE_VOL_PATH)
targets_df = load_hsc_targets(HSC_TARGET_PATH)
names_df = load_ticker_names(TICKER_NAME_PATH)

if df_all.empty:
    st.error(f"Không đọc được dữ liệu từ {PRICE_VOL_PATH}.")
    st.stop()

tickers = sorted(df_all["Ticker"].dropna().unique().tolist()) if "Ticker" in df_all.columns else []
if not tickers:
    st.error("Không có cột Ticker trong Price_Vol.xlsx.")
    st.stop()

col1, col2 = st.columns([1.2, 1])

with col1:
    ticker = st.selectbox("Chọn mã", options=tickers, index=0)
with col2:
    tname = ""
    if not names_df.empty and "Ticker" in names_df.columns:
        m = names_df[names_df["Ticker"] == ticker]
        if not m.empty:
            tname = str(m.iloc[-1].get("Name") or "")
    st.text_input("Tên doanh nghiệp (auto)", value=tname, disabled=True)

df_t = df_all[df_all["Ticker"] == ticker].copy()
if df_t.empty or len(df_t) < 30:
    st.warning("Dữ liệu quá ít (khuyến nghị >= 60 bars để fib/RSI/MACD ổn định).")

# Append virtual bar if requested
if use_virtual_bar:
    last_row = df_t.iloc[-1].copy()
    # parse price: if < 500 treat as thousand style
    vp = _safe_float(v_price)
    if pd.notna(vp) and vp < 500:
        vp = vp * 1000.0
    vv = _safe_float(v_volume)

    if pd.notna(vp) or pd.notna(vv):
        new = last_row.copy()
        new["Date"] = pd.to_datetime(datetime.now().date())
        if pd.notna(vp):
            new["Close"] = vp
            # keep OHLC coherent
            new["Open"] = _safe_float(last_row.get("Close"))
            new["High"] = max(_safe_float(new.get("Open")), _safe_float(new.get("Close")))
            new["Low"] = min(_safe_float(new.get("Open")), _safe_float(new.get("Close")))
        if pd.notna(vv):
            new["Volume"] = vv

        df_t = pd.concat([df_t, pd.DataFrame([new])], ignore_index=True)
        st.info("Đã append virtual latest bar (không ghi đè lịch sử).")
    else:
        st.warning("Virtual bar bật nhưng Price/Volume trống hoặc không hợp lệ.")

# Preview last bars
st.markdown("#### Data Preview (last 8 rows)")
show_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df_t.columns]
st.dataframe(df_t[show_cols].tail(8), use_container_width=True)

run_btn = st.button("RUN ANALYZE", disabled=(not valid_user or not api_key))

if run_btn:
    if not valid_user:
        st.error("User Key không hợp lệ.")
        st.stop()
    if not api_key:
        st.error("Thiếu OPENAI_API_KEY.")
        st.stop()
    if _quota_remaining(user_key) <= 0:
        st.error("Hết quota.")
        st.stop()

    with st.spinner("Computing AnalysisPack (Python)…"):
        pack = build_analysis_pack(ticker, tname, df_t, targets_df)

    if show_pack:
        st.markdown("#### AnalysisPack JSON")
        st.json(pack)

    # consume quota only when pack ok
    if not _consume_quota(user_key, 1):
        st.error("Hết quota.")
        st.stop()

    with st.spinner("Generating narrative report (GPT)…"):
        try:
            report = run_gpt_report(api_key=api_key, model=model, analysis_pack=pack)
        except Exception as e:
            st.error(f"GPT error: {e}")
            st.stop()

    st.markdown("#### Report")
    st.markdown(report)

    # lightweight KPI panel
    st.markdown("---")
    st.markdown("#### Quick KPI")
    last = pack.get("IndicatorsLast", {})
    fund = pack.get("Fundamental", {})
    g = pack.get("GuardD", {})

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Close (k VND)", _fmt_thousand_vnd(pack["LastBar"]["Close"], 1))
    k2.metric("Conviction (0–10)", f"{pack['Scores']['Conviction10']:.1f}")
    if fund.get("HasData"):
        k3.metric("Target (k VND)", _fmt_thousand_vnd(fund.get("TargetVND"), 1), _fmt_pct(fund.get("UpsidePct")))
    else:
        k3.metric("Target", "N/A", "")
    if g.get("PrimarySetup"):
        k4.metric("Primary Setup", g.get("PrimarySetup"), f"RR {g.get('RR')}")
    else:
        k4.metric("Primary Setup", "None", "")

st.markdown(
    '<div class="codebox small">'
    "<b>Notes</b><br/>"
    "• Python tính toàn bộ chỉ số + TradePlan + RR (AnalysisPack). GPT chỉ diễn giải.<br/>"
    "• Section D có GuardD validator + 1 lần repair nếu GPT lệch số Risk/Reward.<br/>"
    "</div>",
    unsafe_allow_html=True,
)
