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

st.set_page_config(page_title="INCEPTION v4.6 – Strategic Investor Edition",
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

def compute_fib_for_window(df: pd.DataFrame, window: int) -> Dict[str, Any]:
    L = window if len(df) >= window else len(df)
    win = df.tail(L)
    hi, lo = win["High"].max(), win["Low"].min()
    return {
        "window": L,
        "swing_high": hi,
        "swing_low": lo,
        "levels": _fib_levels(lo, hi)
    }

def _count_touches(series: pd.Series, level: float, tol_pct: float = 0.007) -> int:
    if pd.isna(level) or level == 0:
        return 0
    tol = abs(level) * tol_pct
    return int(((series - level).abs() <= tol).sum())

def choose_best_short_fibo_window(df: pd.DataFrame, candidates=(60, 90), tol_pct: float = 0.007) -> int:
    best_w = candidates[0]
    best_score = -1

    close_series = df["Close"]
    high_series = df["High"]
    low_series = df["Low"]
    last_close = float(df.iloc[-1]["Close"])

    for w in candidates:
        fib = compute_fib_for_window(df, w)
        levels = fib["levels"]

        s_hi = fib["swing_high"]
        s_lo = fib["swing_low"]
        score_swing = _count_touches(high_series.tail(fib["window"]), s_hi, tol_pct) + _count_touches(low_series.tail(fib["window"]), s_lo, tol_pct)

        score_levels = 0
        for key in ["38.2", "50.0", "61.8"]:
            lv = levels.get(key, np.nan)
            if pd.isna(lv):
                continue
            touches = (
                _count_touches(close_series.tail(fib["window"]), lv, tol_pct) +
                _count_touches(high_series.tail(fib["window"]), lv, tol_pct) +
                _count_touches(low_series.tail(fib["window"]), lv, tol_pct)
            )
            dist = abs(lv - last_close) / last_close if last_close != 0 else 1
            weight = 1 / (1 + dist * 10)
            score_levels += touches * weight

        score = score_swing * 2 + score_levels

        if score > best_score:
            best_score = score
            best_w = w

    return int(best_w)

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

    target = _safe_float(fund_row.get("Target"))
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

    fib_short_window = choose_best_short_fibo_window(df, candidates=(60, 90))
    dual_fib = compute_dual_fibonacci(df, short_window=fib_short_window, long_window=250)

    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    target = _safe_float(fund_row.get("Target"), np.nan)
    upside_pct = ((target - close) / close * 100) if (pd.notna(target) and pd.notna(close) and close != 0) else np.nan
    fund_row["Target"] = target
    fund_row["UpsidePct"] = upside_pct

    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)

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
            "TargetVND": target,
            "TargetK": (target / 1000) if pd.notna(target) else np.nan,
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

    # Chuẩn bị dữ liệu
    tick = data["Ticker"]
    last = data["Last"]
    trade_plans = data["TradePlans"]
    fund = data["Fundamental"]
    conviction = data["Conviction"]
    scenario = data["Scenario"]
    analysis_pack = data.get("AnalysisPack", {})

    close = _fmt_price(last.get("Close"))
    rsi = _fmt_price(last.get("RSI"))
    macd_v = _fmt_price(last.get("MACD"))
    ma20 = _fmt_price(last.get("MA20"))
    ma50 = _fmt_price(last.get("MA50"))
    ma200 = _fmt_price(last.get("MA200"))
    vol = _fmt_int(last.get("Volume"))
    avg_vol = _fmt_int(last.get("Avg20Vol"))

    header = f"**{tick} — {close} | Conviction: {conviction:.1f}/10 | {scenario}**"

    # Trade Plan summary
    tp_text = []
    for k, s in trade_plans.items():
        tp_text.append(f"{k}: Entry {s.entry}, Stop {s.stop}, TP {s.tp}, R:R {s.rr:.2f}")
    tp_summary = " | ".join(tp_text) if tp_text else "Chưa có chiến lược đạt chuẩn R:R ≥ 2.5"

    # Fundamental (Target displayed in thousand, without unit label)
    fund_text = (
        f"Khuyến nghị: {fund.get('Recommendation', 'N/A')} | "
        f"Giá mục tiêu: {_fmt_thousand(fund.get('Target'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if fund else "Không có dữ liệu fundamental"
    )

    # === Prompt ===
    pack_json = json.dumps(analysis_pack, ensure_ascii=False)

    prompt = f"""
    Bạn là chuyên gia phân tích chiến lược của một công ty chứng khoán cao cấp.
    Hãy viết báo cáo ngắn gọn (~700-900 từ) theo cấu trúc chuẩn sau, bằng tiếng Việt,
    văn phong chuyên nghiệp, gần gũi và có chiều sâu.

    QUY TẮC BẮT BUỘC (FRAME-LOCK):
    - Tuyệt đối KHÔNG bịa số liệu.
    - Tuyệt đối KHÔNG tự tính toán bất kỳ con số nào (kể cả cộng/trừ/nhân/chia).
    - Chỉ được phép sử dụng đúng dữ liệu trong JSON "AnalysisPack" bên dưới.
    - Nếu thiếu dữ liệu thì ghi rõ "N/A" và không suy diễn.

    1️⃣ **Executive Summary (3–4 câu)**
    - Nhận định tổng thể xu hướng hiện tại của {tick}, dòng tiền, động lượng.
    - Tác động lên chiến lược hành động của nhà đầu tư trung–dài hạn.

    2️⃣ **A. Phân tích Kỹ thuật**
    Bao gồm (chỉ dùng số trong JSON):
    - MA Trend (MA20, MA50, MA200)
    - RSI Analysis
    - MACD Analysis
    - RSI + MACD Bias
    - Fibonacci (2 khung: ShortWindow & LongWindow): hỗ trợ – kháng cự – vùng chiến lược
    - Volume & Price Action
    - 12-Scenario Classification (Scenario12)
    - Master Integration + MasterScore

    3️⃣ **B. Fundamental Analysis Summary**
    - Dữ liệu: {fund_text}

    4️⃣ **C. Trade Plan**
    - {tp_summary}

    5️⃣ **D. Risk–Reward Simulation**
    - Diễn giải dựa trên RRSim trong JSON (RiskPct, RewardPct, RR, Probability).

    Dữ liệu (AnalysisPack JSON):
    {pack_json}
    """

    # ============================================================
    # ẨN API KEY KHI KHỞI TẠO CLIENT
    # ============================================================
    try:
        client = OpenAI()  # Key lấy tự động từ môi trường
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

    return f"{header}\n\n{content}" # ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

# --- Header section ---
st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.6 — Strategic Investor Edition</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;'>Công cụ phân tích chiến lược cho nhà đầu tư trung–dài hạn (Lợi nhuận 15–100%, Rủi ro 5–8%).</p>", unsafe_allow_html=True)
st.divider()

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("🚀 Phân tích ngay", type="primary")

# --- Layout containers ---
col_main, = st.columns([1])  # Chỉ hiển thị phần Report (ẩn Chart column tạm thời)

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
    Phiên bản 4.6 – Strategic Investor Edition | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
) # ============================================================
# 15. FINAL TOUCHES – MARKDOWN OPTIMIZATION & SAFETY CHECKS
# ============================================================

def render_markdown_safe(text: str):
    """Đảm bảo hiển thị báo cáo Markdown có xuống dòng và format rõ ràng."""
    text = text.replace("\n\n", "<br><br>")
    st.markdown(f"<div style='white-space:pre-wrap; color:#E5E7EB;'>{text}</div>", unsafe_allow_html=True)

# Kiểm tra file dữ liệu
missing_files = []
for f in [PRICE_VOL_PATH, HSC_TARGET_PATH, TICKER_NAME_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    st.warning(f"⚠️ Thiếu file dữ liệu: {', '.join(missing_files)}. Hãy kiểm tra lại thư mục trước khi chạy.")
else:
    st.info("✅ Tất cả file dữ liệu đã sẵn sàng. Bạn có thể tiến hành phân tích.")

# ============================================================
# 16. RUNNING GUIDE
# ============================================================

st.divider()
st.markdown(
    """
    <div style='color:#9CA3AF; font-size:14px; line-height:1.6;'>
    <strong>📘 Hướng dẫn sử dụng:</strong><br>
    1️⃣ Mở Terminal hoặc Command Prompt.<br>
    2️⃣ Di chuyển đến thư mục chứa file <code>app.py</code> và các file Excel dữ liệu.<br>
    3️⃣ Gõ lệnh: <code>streamlit run app.py</code><br>
    4️⃣ Nhập Mã VIP và Mã Cổ Phiếu (VD: HPG, FPT, VNM).<br>
    5️⃣ Hệ thống sẽ tự động tạo báo cáo phân tích chiến lược.<br><br>
    <em>Lưu ý:</em> INCEPTION v4.6 dành cho nhà đầu tư chiến lược (Target 15–100%, Risk 5–8%).<br>
    Không sử dụng cho mục đích giao dịch ngắn hạn hoặc lướt sóng trong ngày.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 17. SAFETY EXIT (FOR EMPTY RUNS)
# ============================================================

if not run_btn:
    st.markdown(
        """
        <br><br>
        <div style='text-align:center; color:#A855F7;'>
        🔍 <strong>Nhập mã cổ phiếu và nhấn “Phân tích ngay” để bắt đầu.</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
