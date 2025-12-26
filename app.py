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

/* Sidebar button style: black glossy, white text, full width */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(180deg, #1f1f1f 0%, #000000 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    height: 42px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    box-shadow: 0 10px 18px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08) !important;
}
.stButton > button:hover {
    background: linear-gradient(180deg, #2a2a2a 0%, #050505 100%) !important;
}
.stButton > button:active {
    transform: translateY(1px);
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

def _scenario_vi(s: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng — Xác nhận breakout",
        "Uptrend – Pullback Phase": "Xu hướng tăng — Pha điều chỉnh (pullback)",
        "Downtrend – Weak Phase": "Xu hướng giảm — Pha yếu",
        "Neutral / Sideways": "Đi ngang / trung tính",
    }
    return m.get(s, s)

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

def compute_dual_fibonacci(df: pd.DataFrame, short_window: int = 60, long_window: int = 250) -> Dict[str, Any]:
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

def select_fib_short_window(df: pd.DataFrame) -> int:
    # Auto choose 60 or 90. High volatility -> 90 (smoother), else 60.
    if df is None or df.empty:
        return 60
    if len(df) < 90:
        return 60
    for col in ["High", "Low", "Close"]:
        if col not in df.columns:
            return 60

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr14 = tr.rolling(14).mean()
    last_close = close.iloc[-1]
    last_atr = atr14.iloc[-1]

    if pd.isna(last_close) or pd.isna(last_atr) or last_close == 0:
        return 60

    atr_pct = (last_atr / last_close) * 100
    return 90 if atr_pct >= 3.5 else 60

def detect_fibo_conflict(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    s_lv = dual_fib.get("auto_short", {}).get("levels", {})
    l_lv = dual_fib.get("fixed_long", {}).get("levels", {})

    s_618 = _safe_float(s_lv.get("61.8"))
    l_618 = _safe_float(l_lv.get("61.8"))
    l_382 = _safe_float(l_lv.get("38.2"))

    conflict = False
    ctype = "N/A"
    rule = "Long=structure (ceiling/floor), Short=tactical (entry)"

    if pd.notna(c) and pd.notna(l_618) and (abs(c - l_618) / l_618 * 100 <= 2.0):
        conflict = True
        ctype = "Near long resistance (Long 61.8 ±2%)"

    if pd.notna(c) and pd.notna(l_382) and (c < l_382):
        conflict = True
        ctype = "Below long support (Long 38.2 breached)"

    if conflict and pd.notna(c) and pd.notna(s_618) and (c >= s_618):
        ctype = f"{ctype} + Short bullish trigger (Close>=Short 61.8)"

    return {
        "FiboConflictFlag": bool(conflict),
        "FiboConflictType": ctype,
        "FiboPriorityRuleApplied": rule
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
    fib_long = dual_fib["fixed_long"]["levels"]
    fib_hi = dual_fib["auto_short"]["swing_high"]
    fib_lo = dual_fib["auto_short"]["swing_low"]

    # Basic reference levels
    res_zone = fib_short.get("61.8", close * 1.05)
    sup_zone = fib_short.get("38.2", close * 0.95)

    # === Breakout Setup ===
    entry_b = _round_price(res_zone * 1.01)
    stop_b = _round_price(max(ma20 * 0.985, sup_zone * 0.99))
    tp_b = _round_price(entry_b * 1.25)  # default 25% upside
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    # === Pullback Setup ===
    entry_p = _round_price(sup_zone)
    stop_p = _round_price(entry_p * 0.94)
    tp_p = _round_price(entry_p * 1.20)
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    # Filter RR < 2.5
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
    rsi, macd_v, sig = last["RSI"], last["MACD"], last["MACDSignal"]

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

    # Trend regime (3)
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

    # Momentum regime (4)
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

    # Volume regime (informational)
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")

    # 12 scenarios = Trend(3) x Momentum(4)
    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Bull": 0, "Neutral": 1, "Bear": 2, "Exhaust": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1  # 1..12

    name_map = {
        ( "Up","Bull" ): "S1 – Uptrend + Bullish Momentum",
        ( "Up","Neutral" ): "S2 – Uptrend + Neutral Momentum",
        ( "Up","Bear" ): "S3 – Uptrend + Bearish Pullback",
        ( "Up","Exhaust" ): "S4 – Uptrend + Overbought/Exhaust",

        ( "Neutral","Bull" ): "S5 – Range + Bullish Attempt",
        ( "Neutral","Neutral" ): "S6 – Range + Balanced",
        ( "Neutral","Bear" ): "S7 – Range + Bearish Pressure",
        ( "Neutral","Exhaust" ): "S8 – Range + Overbought Risk",

        ( "Down","Bull" ): "S9 – Downtrend + Short-covering Bounce",
        ( "Down","Neutral" ): "S10 – Downtrend + Weak Stabilization",
        ( "Down","Bear" ): "S11 – Downtrend + Bearish Momentum",
        ( "Down","Exhaust" ): "S12 – Downtrend + Overbought Rebound Risk",
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

    # Components (0..2 each), total 0..10-ish
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

    fib_short_window = select_fib_short_window(df)
    dual_fib = compute_dual_fibonacci(df, short_window=fib_short_window, long_window=250)
    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    # IMPORTANT: Close in Price_Vol.xlsx is typically in "thousand VND" (e.g., 30.5 == 30,500 VND)
    close_k = float(last["Close"]) if pd.notna(last["Close"]) else np.nan

    # Target in file is VND (e.g., 42,500 VND). Convert to thousand for upside calculation/display.
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)
    target_k = (target_vnd / 1000) if pd.notna(target_vnd) else np.nan
    upside_pct = ((target_k - close_k) / close_k * 100) if (pd.notna(target_k) and pd.notna(close_k) and close_k != 0) else np.nan

    fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

    fibo_conf = detect_fibo_conflict(last, dual_fib)

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
        # kept for internal logic only (do not ask GPT to present)
        "FiboConflictFlag": fibo_conf.get("FiboConflictFlag"),
        "FiboConflictType": fibo_conf.get("FiboConflictType"),
        "FiboPriorityRuleApplied": fibo_conf.get("FiboPriorityRuleApplied"),
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
        "MasterScore": master
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
    } # ============================================================
# 11. GPT-4 TURBO STRATEGIC INSIGHT GENERATION
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    """
    Hàm này gửi dữ liệu kỹ thuật và cơ bản sang GPT-4 Turbo
    để tạo báo cáo phân tích theo chuẩn Strategic Commentary.
    """
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick = data["Ticker"]
    last = data["Last"]
    trade_plans = data["TradePlans"]
    fund = data["Fundamental"]
    conviction = data["Conviction"]
    scenario = data["Scenario"]
    analysis_pack = data.get("AnalysisPack", {})

    close = _fmt_price(last.get("Close"))
    scenario_vi = _scenario_vi(scenario)

    header = f"## {tick} — {close} | Điểm tin cậy: {conviction:.1f}/10 | {scenario_vi}"

    # Trade Plan summary
    tp_text = []
    for k, s in trade_plans.items():
        tp_text.append(f"{k}: Entry {s.entry}, Stop {s.stop}, TP {s.tp}, R:R {s.rr:.2f}")
    tp_summary = " | ".join(tp_text) if tp_text else "Chưa có chiến lược đạt chuẩn R:R ≥ 2.5"

    # Fundamental (Target shown in thousand number without unit label)
    target_k = analysis_pack.get("Fundamental", {}).get("TargetK", np.nan)
    upside_pct = analysis_pack.get("Fundamental", {}).get("UpsidePct", np.nan)
    rec = fund.get("Recommendation", "N/A") if fund else "N/A"

    fund_text = f"Khuyến nghị: {rec} | Giá mục tiêu: {_fmt_price(target_k, 1)} | Upside: {_fmt_pct(upside_pct)}"

        # === Prompt ===
    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    OUTPUT_FORMAT = """
OUTPUT FORMAT (BẮT BUỘC):
- Chỉ được xuất đúng 4 mục, đúng thứ tự, đúng tiêu đề Markdown (không thêm/bớt):
### A. Kỹ thuật
### B. Cơ bản
### C. Trade plan
### D. Rủi ro vs lợi nhuận

- Trong mục A. Kỹ thuật: bắt buộc có đúng 8 ý, đánh số từ 1 đến 8, đúng thứ tự sau (mỗi ý 2–4 câu là đủ):
1. MA Trend:
2. RSI:
3. MACD:
4. RSI + MACD Bias:
5. Fibonacci (2 khung Short/Long):
6. Volume & Price Action:
7. 12-Scenario:
8. MasterScore:

- Tuyệt đối KHÔNG dùng emoji bullets (1️⃣2️⃣3️⃣…), KHÔNG ghi “Executive Summary”, KHÔNG viết dồn thành 1 đoạn văn dài.
- Văn phong thân thiện, nói với “bạn”, ngắn gọn, mượt, dễ hiểu.
- Mỗi câu chỉ dùng tối đa 1–2 con số; không liệt kê dày đặc.
"""

    prompt = f"""
Bạn là chuyên gia phân tích chiến lược của một công ty chứng khoán cao cấp.

QUY TẮC BẮT BUỘC (FRAME-LOCK):
- Tuyệt đối KHÔNG bịa số liệu.
- Tuyệt đối KHÔNG tự tính toán bất kỳ con số nào (kể cả cộng/trừ/nhân/chia).
- Chỉ được phép sử dụng đúng dữ liệu trong JSON "AnalysisPack" bên dưới.
- Nếu thiếu dữ liệu thì ghi rõ "N/A" và không suy diễn.

{OUTPUT_FORMAT}

Gợi ý nội dung theo dữ liệu có sẵn:
- Kỹ thuật: dùng Last, Fibonacci (Short/Long), Scenario12, MasterScore.
- Cơ bản: dùng Fundamental (Recommendation, TargetK, UpsidePct).
- Trade plan: dùng TradePlans (Entry/Stop/TP/RR/Probability).
- Rủi ro vs lợi nhuận: dùng RRSim (RiskPct/RewardPct/RR/Probability).

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""

"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, trợ lý phân tích đầu tư, nói chuyện tự nhiên và rõ ràng."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1600
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return f"{header}\n\n{content}"

# ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.6</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB").upper()
    run_btn = st.button("Phân tích")

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
                st.markdown(report)
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
    Phiên bản 4.6 | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
)

