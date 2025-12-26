# ============================================================
# INCEPTION v4.6 FINAL | Strategic Investor Edition
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

st.set_page_config(page_title="INCEPTION v4.6",
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
.report-header {
    font-size: 28px;
    font-weight: 800;
    color: #E5E7EB;
    margin: 8px 0 14px 0;
}
.report-body {
    color: #E5E7EB;
    white-space: pre-wrap;
    line-height: 1.6;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(180deg, #111827 0%, #000000 100%);
    color: white;
    font-weight: 800;
    border-radius: 12px;
    height: 44px;
    border: 1px solid #1F2937;
    box-shadow: 0px 6px 14px rgba(0,0,0,0.45);
}
.stButton>button:hover {
    background: linear-gradient(180deg, #0F172A 0%, #000000 100%);
    border: 1px solid #374151;
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
# 6. FIBONACCI DUAL-FRAME (PYTHON AUTO-SELECT 60/90 + LONG 250)
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

def _choose_short_fib_window(df: pd.DataFrame) -> int:
    """
    Python tự chọn 60 hoặc 90 dựa trên biến động gần đây.
    - Nếu biến động 20 phiên cao hơn đáng kể so với 60 phiên -> ưu tiên 60 (nhạy hơn).
    - Ngược lại -> ưu tiên 90 (ổn định hơn).
    """
    if df is None or df.empty:
        return 60
    if len(df) < 90:
        return 60 if len(df) >= 60 else max(20, len(df))

    rng = (df["High"] - df["Low"]).astype(float)
    vol20 = float(rng.tail(20).mean()) if len(rng) >= 20 else float(rng.mean())
    vol60 = float(rng.tail(60).mean()) if len(rng) >= 60 else float(rng.mean())

    if (pd.notna(vol20) and pd.notna(vol60) and vol60 != 0) and (vol20 / vol60 >= 1.12):
        return 60
    return 90

def compute_dual_fibonacci(df: pd.DataFrame, long_window: int = 250) -> Dict[str, Any]:
    short_window = _choose_short_fib_window(df)
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
    ma50 = last["MA50"]

    fib_short = dual_fib["auto_short"]["levels"]

    # Basic reference levels
    res_zone = fib_short.get("61.8", close * 1.05)
    sup_zone = fib_short.get("38.2", close * 0.95)

    # === Breakout Setup ===
    entry_b = _round_price(res_zone * 1.01)
    stop_b = _round_price(max(ma20 * 0.985, sup_zone * 0.99))
    tp_b = _round_price(entry_b * 1.25)
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    # === Pullback Setup ===
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

def _scenario_vi(s: str) -> str:
    mapping = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng – Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng – Pha điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm – Yếu",
        "Neutral / Sideways": "Trung tính / Đi ngang",
    }
    return mapping.get(s, s)

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

    return {
        "Setups": rows,
        "BestRR": best_rr if pd.notna(best_rr) else np.nan
    }

# ============================================================
# 9E. PRICE ACTION FEATURES PACK (PYTHON-ONLY)  <<< ADDED
# ============================================================

def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

def _pct(x, denom):
    if pd.isna(x) or pd.isna(denom) or denom == 0:
        return np.nan
    return float(x) / float(denom) * 100.0

def _near_level(close: float, level: float, atr: float, pct_th: float = 0.8, atr_th: float = 0.25) -> bool:
    if any(pd.isna([close, level])):
        return False
    dist_pct = abs(close - level) / close * 100.0 if close != 0 else np.nan
    if pd.notna(dist_pct) and dist_pct <= pct_th:
        return True
    if pd.notna(atr) and atr != 0:
        if abs(close - level) <= atr_th * atr:
            return True
    return False

def build_price_action_pack(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    """
    3-layer PA pack:
    - Candle anatomy: body%, wick%, range percentile, gap
    - Core patterns: doji/pinbar/engulfing/inside-outside
    - Context: near MA/Fib + volume regime + confirm/exhaust label
    """
    if df is None or df.empty or len(df) < 3:
        return {"Error": "Not enough bars"}

    d = df.copy()

    # Required columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in d.columns:
            return {"Error": f"Missing column: {col}"}

    # Range/ATR
    d["Range"] = (d["High"].astype(float) - d["Low"].astype(float))
    d["ATR14"] = _atr_wilder(d, 14)
    last = d.iloc[-1]
    prev = d.iloc[-2]

    o = _safe_float(last["Open"])
    h = _safe_float(last["High"])
    l = _safe_float(last["Low"])
    c = _safe_float(last["Close"])
    prev_o = _safe_float(prev["Open"])
    prev_h = _safe_float(prev["High"])
    prev_l = _safe_float(prev["Low"])
    prev_c = _safe_float(prev["Close"])

    rng = _safe_float(last["Range"])
    atr = _safe_float(last["ATR14"])
    avg_rng20 = _safe_float(d["Range"].tail(20).mean())

    # Candle anatomy
    body = abs(c - o) if pd.notna(c) and pd.notna(o) else np.nan
    upper_wick = (h - max(o, c)) if pd.notna(h) and pd.notna(o) and pd.notna(c) else np.nan
    lower_wick = (min(o, c) - l) if pd.notna(l) and pd.notna(o) and pd.notna(c) else np.nan

    body_pct = _pct(body, rng)
    upper_wick_pct = _pct(upper_wick, rng)
    lower_wick_pct = _pct(lower_wick, rng)

    # Range percentile vs last N days
    N = 60 if len(d) >= 60 else len(d)
    range_window = d["Range"].tail(N)
    range_pctile = np.nan
    if len(range_window) >= 5 and pd.notna(rng):
        # Percentile rank of last range within window
        rank = range_window.rank(pct=True)
        range_pctile = float(rank.iloc[-1] * 100.0)

    # Gap detection
    gap_type = "None"
    gap_pct = np.nan
    if pd.notna(o) and pd.notna(prev_h) and pd.notna(prev_l) and pd.notna(prev_c) and prev_c != 0:
        if o > prev_h:
            gap_type = "GapUp"
            gap_pct = (o - prev_h) / prev_c * 100.0
        elif o < prev_l:
            gap_type = "GapDown"
            gap_pct = (prev_l - o) / prev_c * 100.0

    # Core patterns (1-2 bar)
    patterns = []

    # Doji
    if pd.notna(body_pct) and body_pct <= 10:
        patterns.append("Doji")

    # Hammer / Shooting Star (Pin bar)
    # Hammer: long lower wick, small body near top
    if pd.notna(lower_wick) and pd.notna(body) and pd.notna(upper_wick):
        if (lower_wick >= 2.0 * body) and (upper_wick <= 0.35 * body):
            patterns.append("Hammer/PinBarBull")
        if (upper_wick >= 2.0 * body) and (lower_wick <= 0.35 * body):
            patterns.append("ShootingStar/PinBarBear")

    # Engulfing
    bull_engulf = False
    bear_engulf = False
    if all(pd.notna([o, c, prev_o, prev_c])):
        bull_engulf = (c > o) and (prev_c < prev_o) and (c >= prev_o) and (o <= prev_c)
        bear_engulf = (c < o) and (prev_c > prev_o) and (c <= prev_o) and (o >= prev_c)
    if bull_engulf:
        patterns.append("BullishEngulfing")
    if bear_engulf:
        patterns.append("BearishEngulfing")

    # Inside / Outside bar
    inside = False
    outside = False
    if all(pd.notna([h, l, prev_h, prev_l])):
        inside = (h <= prev_h) and (l >= prev_l)
        outside = (h >= prev_h) and (l <= prev_l)
    if inside:
        patterns.append("InsideBar")
    if outside:
        patterns.append("OutsideBar")

    # Context: near MA & Fib
    near_ma = []
    near_fib = []
    near_levels = []

    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    if _near_level(c, ma20, atr):
        near_ma.append("MA20")
        near_levels.append("MA20")
    if _near_level(c, ma50, atr):
        near_ma.append("MA50")
        near_levels.append("MA50")
    if _near_level(c, ma200, atr):
        near_ma.append("MA200")
        near_levels.append("MA200")

    s_lv = dual_fib.get("auto_short", {}).get("levels", {})
    for k in ["38.2", "50.0", "61.8"]:
        lv = _safe_float(s_lv.get(k))
        if _near_level(c, lv, atr):
            near_fib.append(f"FiboS{k}")
            near_levels.append(f"FiboS{k}")

    l_lv = dual_fib.get("fixed_long", {}).get("levels", {})
    for k in ["38.2", "50.0", "61.8"]:
        lv = _safe_float(l_lv.get(k))
        if _near_level(c, lv, atr):
            near_fib.append(f"FiboL{k}")
            near_levels.append(f"FiboL{k}")

    # Volume regime
    avg20_vol = _safe_float(last.get("Avg20Vol"))
    vol = _safe_float(last.get("Volume"))
    vol_ratio = (vol / avg20_vol) if (pd.notna(vol) and pd.notna(avg20_vol) and avg20_vol != 0) else np.nan

    vol_regime = "N/A"
    if pd.notna(vol_ratio):
        if vol_ratio >= 1.8:
            vol_regime = "Spike"
        elif vol_ratio >= 1.2:
            vol_regime = "High"
        elif vol_ratio <= 0.8:
            vol_regime = "Low"
        else:
            vol_regime = "Normal"

    # Confirmation vs Exhaustion label (deterministic)
    # Confirmation: High/Spike volume + close near extreme in direction
    # Exhaustion: High/Spike volume + long wick (pin/doji) or over-extended range percentile high
    candle_dir = "Neutral"
    if pd.notna(c) and pd.notna(o):
        candle_dir = "Bull" if c > o else ("Bear" if c < o else "Neutral")

    close_pos = np.nan
    if pd.notna(rng) and rng != 0 and pd.notna(c) and pd.notna(l):
        close_pos = (c - l) / rng  # 0..1

    pa_signal = "Neutral"
    if vol_regime in ["High", "Spike"]:
        # confirmation
        if (candle_dir == "Bull") and pd.notna(close_pos) and (close_pos >= 0.75) and ("BearishEngulfing" not in patterns):
            pa_signal = "ConfirmationBull"
        elif (candle_dir == "Bear") and pd.notna(close_pos) and (close_pos <= 0.25) and ("BullishEngulfing" not in patterns):
            pa_signal = "ConfirmationBear"
        else:
            # exhaustion heuristics
            exhaustion_hit = False
            if ("Doji" in patterns) or ("ShootingStar/PinBarBear" in patterns) or ("Hammer/PinBarBull" in patterns):
                exhaustion_hit = True
            if pd.notna(range_pctile) and range_pctile >= 85:
                exhaustion_hit = True
            pa_signal = "Exhaustion" if exhaustion_hit else "Neutral"

    # Friendly hint labels (so GPT doesn't need to invent)
    pattern_primary = patterns[0] if patterns else "None"
    context_primary = ", ".join(near_levels[:3]) if near_levels else "None"
    vol_label = vol_regime

    return {
        "CandleAnatomy": {
            "BodyPct": body_pct,
            "UpperWickPct": upper_wick_pct,
            "LowerWickPct": lower_wick_pct
        },
        "Range": {
            "Range": rng,
            "ATR14": atr,
            "AvgRange20": avg_rng20,
            "RangePctileN": range_pctile,
            "N": int(N)
        },
        "Gap": {
            "Type": gap_type,
            "GapPct": gap_pct
        },
        "Patterns": patterns,
        "PatternPrimary": pattern_primary,
        "Context": {
            "NearMA": near_ma,
            "NearFibo": near_fib,
            "NearLevels": near_levels,
            "ContextPrimary": context_primary
        },
        "Volume": {
            "Regime": vol_label,
            "VolRatioToAvg20": vol_ratio
        },
        "Signal": {
            "CandleDir": candle_dir,
            "PA_Signal": pa_signal
        }
    }

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

    dual_fib = compute_dual_fibonacci(df)
    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    # --- Target normalization + Upside calc (unit-safe) ---
    close_raw = _safe_float(last.get("Close"), np.nan)
    target_raw = _safe_float(fund_row.get("Target"), np.nan)

    target_vnd = np.nan
    target_k = np.nan
    if pd.notna(target_raw):
        if target_raw < 1000:
            target_k = float(target_raw)
            target_vnd = float(target_raw) * 1000.0
        else:
            target_vnd = float(target_raw)
            target_k = float(target_raw) / 1000.0

    # Align close to same unit as target for upside
    close_for_upside = close_raw
    if pd.notna(close_raw) and pd.notna(target_vnd):
        # If close looks like "30.5" while target is "42,500" -> close likely in thousand
        if close_raw < 1000 and target_vnd >= 1000:
            close_for_upside = close_raw * 1000.0

    upside_pct = np.nan
    if pd.notna(target_vnd) and pd.notna(close_for_upside) and close_for_upside != 0:
        upside_pct = (target_vnd - close_for_upside) / close_for_upside * 100.0

    fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct
    fund_row["Recommendation"] = fund_row.get("Recommendation", "N/A") if fund_row else "N/A"

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

    # --- Price Action pack (ADDED) ---
    pa_pack = build_price_action_pack(df, dual_fib)

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
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {})
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": target_k,
            "UpsidePct": upside_pct
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
        "PriceAction": pa_pack
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

def generate_insight_report(data: Dict[str, Any]) -> Tuple[str, str]:
    if "Error" in data:
        return "❌ Lỗi", f"❌ {data['Error']}"

    tick = data["Ticker"]
    last = data["Last"]
    scenario = data["Scenario"]
    conviction = data["Conviction"]
    analysis_pack = data.get("AnalysisPack", {})

    close = _fmt_price(last.get("Close"))
    header = f"{tick} — {close} | Điểm tin cậy: {conviction:.1f}/10 | {_scenario_vi(scenario)}"

    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    prompt = f"""
Bạn là chuyên gia chứng khoán cao cấp. Hãy viết báo cáo rõ ràng, mạch lạc, thân thiện, nói với "bạn".

QUY TẮC BẮT BUỘC:
- Tuyệt đối KHÔNG bịa số liệu.
- Tuyệt đối KHÔNG tự tính toán bất kỳ con số nào.
- Chỉ dùng đúng dữ liệu trong JSON "AnalysisPack" bên dưới.
- Mỗi câu tối đa 1–2 con số. Ưu tiên nhãn/kết luận đã có trong pack.

OUTPUT FORMAT BẮT BUỘC (không thêm mục khác):
### A. Kỹ thuật
1) MA Trend
2) RSI
3) MACD
4) RSI + MACD Bias
5) Fibonacci (2 khung Short/Long)
6) Volume & Price Action (BẮT BUỘC dùng AnalysisPack.PriceAction: Patterns/Signal/Context/Volume)
7) 12-Scenario (Scenario12)
8) MasterScore (Tier + PositionSizing)

### B. Cơ bản
- Dùng AnalysisPack.Fundamental: Recommendation, TargetK, UpsidePct
- Giá mục tiêu hiển thị dạng số "ngàn" (TargetK), không ghi đơn vị.

### C. Trade plan
- Chỉ dựa trên AnalysisPack.TradePlans, không tự chế.

### D. Rủi ro vs lợi nhuận
- Chỉ dựa trên AnalysisPack.RRSim (RiskPct, RewardPct, RR, Probability)

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1600
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return header, content

# ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.6</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập")
    user_key = st.text_input("Mã VIP:", type="password")
    ticker_input = st.text_input("Mã cổ phiếu:", value="VCB").upper()
    run_btn = st.button("Phân tích")

col_main, = st.columns([1])

# ============================================================
# 13. MAIN EXECUTION
# ============================================================

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng. Vui lòng nhập lại.")
    else:
        with st.spinner(f"Đang phân tích {ticker_input}..."):
            try:
                result = analyze_ticker(ticker_input)
                header_text, report = generate_insight_report(result)
                st.markdown(f"<div class='report-header'>{header_text}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-body'>{report}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Lỗi xử lý: {e}")
