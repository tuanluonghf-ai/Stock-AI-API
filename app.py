# ============================================================
# INCEPTION v5.2 | Strategic Investor Edition
# app.py — Streamlit + GPT-4 Turbo
# Author: INCEPTION AI Research Framework
# Purpose: Technical–Fundamental Integrated Research Assistant
# CHANGELOG:
# v5.0:   Core Refactoring (Facts over Conclusions)
# v5.1:   TradePlan v2 (Active/Watch Status + ATR Buffer)
# v5.2:   "B" Series Update (Total Unlocking):
#         - MasterScore: Remove Fundamentls, Tier=N/A (Step 1B)
#         - Scenario12: Neutral Naming (Step 2B)
#         - Bias: Facts + Tags only (Step 3B)
#         - Conviction: Added ConvictionPack breakdown (Step 4B)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(page_title="INCEPTION v5.2",
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

def _scenario_vi(x: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng — Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng — Pha điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm — Yếu",
        "Neutral / Sideways": "Đi ngang / Trung tính",
    }
    return m.get(x, x)

# ============================================================
# 3B. GUARD-D: PRIMARYSETUP + VALIDATION + RETRY
# ============================================================
def _extract_d_block(text: str) -> str:
    m = re.search(r"(^|\n)\s*D\.\s*Rủi\s*ro.*$", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return m.group(0) if m else text

def _grab_number(block: str, label_patterns: List[str]) -> float:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            s = s.replace(",", "")
            s = s.replace("%", "")
            return _safe_float(s)
    return np.nan

def _grab_text(block: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            return s
    return None

def _close_enough(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol

def validate_section_d(text: str, primary: Dict[str, Any]) -> bool:
    block = _extract_d_block(text)
    got_risk = _grab_number(block, [
        r"Risk%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"Rủi\s*ro\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"RiskPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
    ])
    got_reward = _grab_number(block, [
        r"Reward%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"Lợi\s*nhuận\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"RewardPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
    ])
    got_rr = _grab_number(block, [
        r"\bRR\b\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"R\s*[:/]\s*R\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    got_prob = _grab_text(block, [
        r"Probability\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
        r"Xác\s*suất\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
    ]) or ""
    
    exp_risk = _safe_float(primary.get("RiskPct"))
    exp_reward = _safe_float(primary.get("RewardPct"))
    exp_rr = _safe_float(primary.get("RR"))
    exp_prob = (primary.get("Probability") or "").strip().lower()
    
    ok = True
    if pd.notna(exp_risk):
        ok &= _close_enough(got_risk, exp_risk, tol=0.05)
    if pd.notna(exp_reward):
        ok &= _close_enough(got_reward, exp_reward, tol=0.05)
    if pd.notna(exp_rr):
        ok &= _close_enough(got_rr, exp_rr, tol=0.05)
        
    gp = got_prob.strip().lower()
    if exp_prob:
        ok &= (exp_prob in gp) if gp else False
        
    return bool(ok)

def _call_openai(prompt: str, temperature: float) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=1600
    )
    return response.choices[0].message.content

def call_gpt_with_guard(prompt: str, analysis_pack: Dict[str, Any], max_retry: int = 2) -> str:
    primary = (analysis_pack.get("PrimarySetup") or {})
    temps = [0.7, 0.2, 0.0]
    last_text = ""
    
    for i in range(min(max_retry + 1, len(temps))):
        temp = temps[i]
        extra = ""
        if i > 0:
            extra = f"""
SỬA LỖI BẮT BUỘC (chỉ sửa mục D, giữ nguyên các mục khác):
Mục D đang sai số. Hãy sửa lại mục D bằng cách COPY ĐÚNG các số sau (không được tự tính/ước lượng):
Risk%={primary.get('RiskPct')}, Reward%={primary.get('RewardPct')}, RR={primary.get('RR')}, Probability={primary.get('Probability')}.

Mục D bắt buộc đúng format 4 dòng:
Risk%: <...>
Reward%: <...>
RR: <...>
Probability: <...>
"""
        text = _call_openai(prompt + extra, temperature=temp)
        last_text = text
        
        if validate_section_d(text, primary):
            return text
            
    return last_text

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

# ============================================================
# 6. FIBONACCI & CONTEXT
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

# --- STEP 3B: RSI+MACD BIAS (FACTS ONLY) ---
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
    
    tags = []
    
    # Neutral RSI tags
    if rsi_zone in ["Zone70Plus", "Zone60_70"]: tags.append("RSI_UpperZone")
    if rsi_zone in ["Zone30_40", "ZoneBelow30"]: tags.append("RSI_LowerZone")
    if rsi_dir == "Rising": tags.append("RSI_Rising")
    if rsi_dir == "Falling": tags.append("RSI_Falling")
    if rsi_div_type == "Bullish": tags.append("RSI_BullDiv")
    if rsi_div_type == "Bearish": tags.append("RSI_BearDiv")
    
    # Neutral MACD tags
    if macd_rel == "MACD_Above_Signal": tags.append("MACD_AboveSig")
    if macd_rel == "MACD_Below_Signal": tags.append("MACD_BelowSig")
    if zero == "Above": tags.append("MACD_AboveZero")
    if zero == "Below": tags.append("MACD_BelowZero")
    if "Positive" in hist_state: tags.append("Hist_Pos")
    if "Negative" in hist_state: tags.append("Hist_Neg")
    if "Expanding" in hist_state: tags.append("Hist_Expanding")
    if "Contracting" in hist_state: tags.append("Hist_Contracting")
    if cross_event == "CrossUp" and cross_bars is not None and cross_bars <= 3: tags.append("MACD_CrossUp_Recent")
    if cross_event == "CrossDown" and cross_bars is not None and cross_bars <= 3: tags.append("MACD_CrossDown_Recent")
    
    # Alignment Logic (Structure, not sentiment)
    alignment = "Mixed"
    bias_code = "Neutral"
    
    is_rsi_up = ("RSI_Rising" in tags) or ("RSI_UpperZone" in tags)
    is_macd_up = ("MACD_AboveSig" in tags) or ("Hist_Pos" in tags)
    is_rsi_down = ("RSI_Falling" in tags) or ("RSI_LowerZone" in tags)
    is_macd_down = ("MACD_BelowSig" in tags) or ("Hist_Neg" in tags)
    
    if is_rsi_up and is_macd_up:
        alignment = "Aligned_Positive"
        bias_code = "Momentum_Up"
    elif is_rsi_down and is_macd_down:
        alignment = "Aligned_Negative"
        bias_code = "Momentum_Down"
    elif is_rsi_up and is_macd_down:
        alignment = "Mixed_RSI_Lead"
        bias_code = "Divergent"
    elif is_rsi_down and is_macd_up:
        alignment = "Mixed_MACD_Lead"
        bias_code = "Divergent"
        
    if "Zone70Plus" in rsi_zone: bias_code = "Extended_High"
    if "ZoneBelow30" in rsi_zone: bias_code = "Extended_Low"
    
    notes = []
    if "RSI_BullDiv" in tags: notes.append("Có phân kỳ dương RSI (Bullish Divergence).")
    if "RSI_BearDiv" in tags: notes.append("Có phân kỳ âm RSI (Bearish Divergence).")
    if cross_event != "None" and cross_bars is not None:
        notes.append(f"MACD vừa có tín hiệu {cross_event} (cách {cross_bars} bar).")
        
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
# 7. CONVICTION SCORE (Step 4B Update)
# ============================================================
def compute_conviction_pack(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    
    score = 5.0
    comps = {"Base": 5.0}
    notes = []
    
    # Rule 1: Above MA200 (+2)
    if pd.notna(c) and pd.notna(ma200) and c > ma200:
        score += 2.0
        comps["AboveMA200"] = 2.0
        notes.append("Giá nằm trên MA200 (+2).")
    
    # Rule 2: RSI > 55 (+1)
    if pd.notna(rsi) and rsi > 55:
        score += 1.0
        comps["RSI_Strong"] = 1.0
        notes.append("RSI > 55 thể hiện sức mạnh (+1).")
        
    # Rule 3: Vol > Avg (+1)
    vol_ratio = np.nan
    if pd.notna(vol) and pd.notna(avg) and avg != 0:
        vol_ratio = vol / avg
        if vol > avg:
            score += 1.0
            comps["Vol_Breakout"] = 1.0
            notes.append("Volume vượt trung bình (+1).")
            
    # Rule 4: MACD > Signal (+0.5)
    if pd.notna(macd_v) and pd.notna(sig) and macd_v > sig:
        score += 0.5
        comps["MACD_Bull"] = 0.5
        notes.append("MACD > Signal (fact).")
        
    score = float(min(10.0, score))
    return {
        "Score": round(score, 2),
        "Components": comps,
        "Facts": {
            "Close": c,
            "MA200": ma200,
            "RSI": rsi,
            "Volume": vol,
            "Avg20Vol": avg,
            "VolRatio": vol_ratio,
            "MACD": macd_v,
            "MACDSignal": sig
        },
        "Notes": notes,
        "Policy": "Python cung cấp facts + breakdown; GPT tự diễn giải conviction theo bối cảnh."
    }

def compute_conviction(last: pd.Series) -> float:
    pack = compute_conviction_pack(last)
    return _safe_float(pack.get("Score"), default=5.0)

# ============================================================
# 8. TRADE PLAN LOGIC (Step 9+10 / v5.1)
# ============================================================
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
    if any(pd.isna([entry, stop, tp])) or entry <= stop: return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _buffer_price(df: pd.DataFrame, entry: float) -> float:
    if pd.isna(entry) or entry == 0: return np.nan
    a = atr_wilder(df, 14)
    atr_last = _safe_float(a.dropna().iloc[-1]) if not a.dropna().empty else np.nan
    b1 = 0.5 * atr_last if pd.notna(atr_last) else np.nan
    b2 = entry * 0.004
    if pd.notna(b1): return max(b1, b2)
    return b2

def _nearest_above(levels: Dict[str, Any], x: float) -> Tuple[Optional[str], float]:
    best_k, best_v = None, np.nan
    for k, v in (levels or {}).items():
        fv = _safe_float(v)
        if pd.isna(fv) or pd.isna(x): continue
        if fv > x:
            if (best_k is None) or (fv < best_v): best_k, best_v = str(k), fv
    return best_k, best_v

def _vol_ratio(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns: return np.nan
    vol = _safe_float(df.iloc[-1].get("Volume"))
    avg = _safe_float(df.iloc[-1].get("Avg20Vol")) if "Avg20Vol" in df.columns else np.nan
    if pd.notna(vol) and pd.notna(avg) and avg != 0: return vol / avg
    return np.nan

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty: return {}
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    fib_short = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
    fib_long  = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
    r618 = _safe_float(fib_short.get("61.8"))
    r500 = _safe_float(fib_short.get("50.0"))
    s382 = _safe_float(fib_short.get("38.2"))
    levels_tp = {}
    levels_tp.update(fib_short)
    levels_tp.update(fib_long)
    vr = _vol_ratio(df)
    
    # 1) BREAKOUT PLAN
    base_res = r618 if pd.notna(r618) else close
    entry_b = _round_price(base_res * 1.01) if pd.notna(base_res) else np.nan
    buf_b = _buffer_price(df, entry_b) if pd.notna(entry_b) else np.nan
    stop_candidates = []
    if pd.notna(r618): stop_candidates.append(r618)
    if pd.notna(ma20): stop_candidates.append(ma20)
    stop_base_b = min(stop_candidates) if stop_candidates else np.nan
    stop_b = _round_price(stop_base_b - buf_b) if (pd.notna(stop_base_b) and pd.notna(buf_b)) else np.nan
    tp_label_b, tp_val_b = _nearest_above(levels_tp, entry_b) if pd.notna(entry_b) else (None, np.nan)
    if pd.notna(tp_val_b): tp_b = _round_price(tp_val_b)
    else:
        if pd.notna(entry_b) and pd.notna(stop_b) and entry_b > stop_b: tp_b = _round_price(entry_b + 3.0 * (entry_b - stop_b))
        else: tp_b = np.nan
    rr_b = _compute_rr(entry_b, stop_b, tp_b)
    tags_b = []
    if pd.notna(r618): tags_b.append("Anchor=Fib61.8")
    if pd.notna(ma20): tags_b.append("StopRef=MA20/Fib")
    if pd.notna(vr): tags_b.append(f"VolRatio={round(vr,2)}")
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
            if pd.notna(vr) and vr >= 1.1: tags_b.append("Trigger=VolumeSupport")
    breakout = TradeSetup(name="Breakout", entry=entry_b, stop=stop_b, tp=tp_b, rr=rr_b, probability="N/A", status=status_b, reason_tags=tags_b)
    
    # 2) PULLBACK PLAN
    sup_candidates = []
    if pd.notna(r500) and pd.notna(close) and r500 < close: sup_candidates.append(("Fib50.0", r500))
    if pd.notna(s382) and pd.notna(close) and s382 < close: sup_candidates.append(("Fib38.2", s382))
    if pd.notna(ma50) and pd.notna(close) and ma50 < close: sup_candidates.append(("MA50", ma50))
    if sup_candidates:
        sup_candidates.sort(key=lambda kv: abs(close - kv[1]))
        sup_tag, sup_val = sup_candidates[0]
        entry_p = _round_price(sup_val)
    else:
        sup_tag, sup_val = ("Fallback", close)
        entry_p = _round_price(close) if pd.notna(close) else np.nan
    buf_p = _buffer_price(df, entry_p) if pd.notna(entry_p) else np.nan
    stop_cands_p = []
    for nm, v in [("Fib38.2", s382), ("MA50", ma50), ("MA200", ma200)]:
        fv = _safe_float(v)
        if pd.notna(fv) and pd.notna(entry_p) and fv < entry_p: stop_cands_p.append((nm, fv))
    if stop_cands_p:
        stop_cands_p.sort(key=lambda kv: abs(entry_p - kv[1]))
        stop_base_p = stop_cands_p[0][1]
        tags_p_stop = f"StopRef={stop_cands_p[0][0]}"
    else:
        stop_base_p = entry_p
        tags_p_stop = "StopRef=Fallback"
    stop_p = _round_price(stop_base_p - buf_p) if (pd.notna(stop_base_p) and pd.notna(buf_p)) else np.nan
    tp_label_p, tp_val_p = _nearest_above(levels_tp, entry_p) if pd.notna(entry_p) else (None, np.nan)
    if pd.notna(tp_val_p): tp_p = _round_price(tp_val_p)
    else:
        if pd.notna(entry_p) and pd.notna(stop_p) and entry_p > stop_p: tp_p = _round_price(entry_p + 2.6 * (entry_p - stop_p))
        else: tp_p = np.nan
    rr_p = _compute_rr(entry_p, stop_p, tp_p)
    tags_p = [f"EntryAnchor={sup_tag}", tags_p_stop]
    if pd.notna(vr): tags_p.append(f"VolRatio={round(vr,2)}")
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
    pullback = TradeSetup(name="Pullback", entry=entry_p, stop=stop_p, tp=tp_p, rr=rr_p, probability="N/A", status=status_p, reason_tags=tags_p)
    
    return {"Breakout": breakout, "Pullback": pullback}

# ============================================================
# 9. SCENARIO CLASSIFICATION
# ============================================================
def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20: return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200: return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200: return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# --- STEP 2B: SCENARIO 12 NEUTRAL ---
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
    
    # 1. Trend
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
    else:
        rules_hit.append("Trend=N/A (missing MA50/MA200/Close)")
        
    # 2. Momentum (Alignment)
    mom = "Mixed"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        rsi_up = (rsi >= 55)
        rsi_down = (rsi <= 45)
        rsi_ext = (rsi >= 70)
        macd_up = (macd_v >= sig)
        macd_down = (macd_v < sig)
        
        if rsi_ext:
            mom = "Extended"
            rules_hit.append("Momentum=Extended (RSI>=70)")
        elif rsi_up and macd_up:
            mom = "Aligned"
            rules_hit.append("Momentum=Aligned_Up (RSI>=55 & MACD>=Sig)")
        elif rsi_down and macd_down:
            mom = "Aligned" # Aligned Down
            rules_hit.append("Momentum=Aligned_Down (RSI<=45 & MACD<Sig)")
        elif rsi_up and macd_down:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bull but MACD Bear)")
        elif rsi_down and macd_up:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bear but MACD Bull)")
        else:
            mom = "Mixed"
            rules_hit.append("Momentum=Mixed (Between zones)")
    else:
        rules_hit.append("Momentum=N/A")
        
    # 3. Volume
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")
    else:
        rules_hit.append("Volume=N/A (missing Volume/Avg20Vol)")
        
    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Aligned": 0, "Mixed": 1, "Counter": 2, "Extended": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1
    
    name_map = {
        ("Up", "Aligned"):   "S1 – Uptrend + Momentum Aligned",
        ("Up", "Mixed"):     "S2 – Uptrend + Momentum Mixed",
        ("Up", "Counter"):   "S3 – Uptrend + Momentum Counter",
        ("Up", "Extended"):  "S4 – Uptrend + RSI Extended",
        ("Neutral", "Aligned"):  "S5 – Range + Momentum Aligned",
        ("Neutral", "Mixed"):    "S6 – Range + Balanced/Mixed",
        ("Neutral", "Counter"):  "S7 – Range + Momentum Counter",
        ("Neutral", "Extended"): "S8 – Range + RSI Extended",
        ("Down", "Aligned"):  "S9 – Downtrend + Momentum Aligned",
        ("Down", "Mixed"):    "S10 – Downtrend + Momentum Mixed",
        ("Down", "Counter"):  "S11 – Downtrend + Momentum Counter",
        ("Down", "Extended"): "S12 – Downtrend + RSI Extended",
    }
    
    return {
        "Code": int(code),
        "TrendRegime": trend,
        "MomentumRegime": mom,
        "VolumeRegime": vol_reg,
        "Name": name_map.get((trend, mom), "Scenario – N/A"),
        "RulesHit": rules_hit,
        "Flags": {
            "TrendUp": (trend=="Up"),
            "TrendDown": (trend=="Down"),
            "MomAligned": (mom=="Aligned"),
            "MomExtended": (mom=="Extended"),
            "VolHigh": (vol_reg=="High")
        }
    }

# --- STEP 1B: MASTER SCORE (FACTS ONLY) ---
def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup], fund_row: Dict[str, Any]) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))
    
    comps = {}
    facts = {}
    notes = []
    
    # 1. Trend
    trend = 0.0
    trend_sig = "N/A"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            trend = 2.0
            trend_sig = "StrongUp"
            notes.append("Trend: Cấu trúc tăng mạnh (Close > MA50 > MA200).")
        elif (c >= ma200):
            trend = 1.2
            trend_sig = "ModUp"
            notes.append("Trend: Tăng trung hạn (Trên MA200).")
        else:
            trend = 0.4
            trend_sig = "Down/Weak"
            notes.append("Trend: Yếu (Dưới MA200).")
    comps["Trend"] = trend
    facts["TrendSignal"] = trend_sig
    
    # 2. Momentum
    mom = 0.0
    mom_sig = "N/A"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = 2.0
            mom_sig = "RSI>=55 & MACD>=Signal"
            notes.append("Momentum: RSI > 55 và MACD nằm trên Signal.")
        elif (rsi <= 45) and (macd_v < sig):
            mom = 0.4
            mom_sig = "RSI<=45 & MACD<Signal"
            notes.append("Momentum: RSI <=45 và MACD nằm dưới Signal.")
        else:
            mom = 1.1
            mom_sig = "Mixed"
            notes.append("Momentum: RSI/MACD đang ở trạng thái pha trộn, cần đọc theo bối cảnh.")
    comps["Momentum"] = mom
    facts["MomentumSignal"] = mom_sig
    
    # 3. Volume
    vcomp = 0.0
    vol_sig = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol) and avg_vol != 0:
        if vol > avg_vol:
            vcomp = 1.6
            vol_sig = "Vol>Avg20"
            notes.append("Volume: khối lượng lớn hơn trung bình 20 phiên.")
        else:
            vcomp = 0.9
            vol_sig = "Vol<=Avg20"
            notes.append("Volume: khối lượng không vượt trung bình 20 phiên.")
    comps["Volume"] = vcomp
    facts["VolumeSignal"] = vol_sig
    
    # 4. Fibonacci
    fibc = 0.0
    try:
        s_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
        l_lv = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))
        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            if c >= s_618: fibc += 1.2
            elif c >= s_382: fibc += 0.8
            else: fibc += 0.4
        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            if c >= l_618: fibc += 0.8
            elif c >= l_382: fibc += 0.5
            else: fibc += 0.2
    except: fibc = 0.0
    comps["Fibonacci"] = fibc
    
    # 5. RR Quality
    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(s.rr)]
        best_rr = max(rrs) if rrs else np.nan
    rrcomp = 0.0
    rr_sig = "N/A"
    if pd.notna(best_rr):
        if best_rr >= 4.0:
            rrcomp = 2.0
            rr_sig = "RR>=4.0"
        elif best_rr >= 3.0:
            rrcomp = 1.5
            rr_sig = "RR>=3.0"
        else:
            rrcomp = 1.0
            rr_sig = "RR<3.0"
    comps["RRQuality"] = rrcomp
    facts["RRSignal"] = rr_sig
    facts["BestRR"] = best_rr if pd.notna(best_rr) else np.nan
    
    total = float(sum(comps.values()))
    
    notes.append("MasterScore: Technical-only (Fundamental excluded).")
    notes.append("MasterScore: Tier/Sizing returned as N/A (GPT must decide).")
    
    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": "N/A",           # Unlocked
        "PositionSizing": "N/A", # Unlocked
        "Facts": facts,
        "Notes": notes
    }

# ============================================================
# 9D. RISK–REWARD SIMULATION PACK
# ============================================================
def build_rr_sim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    rows = []
    best_rr = np.nan
    for k, s in (trade_plans or {}).items():
        status = getattr(s, "status", "Watch") or "Watch"
        if status == "Invalid": continue
        entry = _safe_float(s.entry)
        stop = _safe_float(s.stop)
        tp = _safe_float(s.tp)
        rr = _safe_float(s.rr)
        risk_pct = ((entry - stop) / entry * 100) if (pd.notna(entry) and pd.notna(stop) and entry != 0) else np.nan
        reward_pct = ((tp - entry) / entry * 100) if (pd.notna(tp) and pd.notna(entry) and entry != 0) else np.nan
        rows.append({
            "Setup": k, "Entry": entry, "Stop": stop, "TP": tp, "RR": rr,
            "RiskPct": risk_pct, "RewardPct": reward_pct, "Probability": s.probability,
            "Status": status, "ReasonTags": list(getattr(s, "reason_tags", []) or [])
        })
        if pd.notna(rr):
            best_rr = rr if pd.isna(best_rr) else max(best_rr, rr)
    return {"Setups": rows, "BestRR": best_rr if pd.notna(best_rr) else np.nan}

# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================
def analyze_ticker(ticker: str) -> Dict[str, Any]:
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty: return {"Error": "Không đọc được dữ liệu Price_Vol.xlsx"}
    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty: return {"Error": f"Không tìm thấy mã {ticker}"}
    
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    
    dual_fib = compute_dual_fibonacci_auto(df, long_window=250)
    last = df.iloc[-1]
    
    # --- Step 5 Context ---
    fib_ctx = compute_fibonacci_context_pack(last, dual_fib)
    
    conviction = compute_conviction(last)
    conviction_pack = compute_conviction_pack(last) # Step 4B Add
    
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)
    
    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}
    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)
    close_for_calc = close
    target_for_calc = target_vnd
    if pd.notna(close) and pd.notna(target_vnd):
        if (close < 500) and (target_vnd > 1000): target_for_calc = target_vnd / 1000.0
        elif (close > 1000) and (target_vnd < 500): target_for_calc = target_vnd * 1000.0
    upside_pct = ((target_for_calc - close_for_calc) / close_for_calc * 100) if (pd.notna(target_for_calc) and pd.notna(close_for_calc) and close_for_calc != 0) else np.nan
    fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct
    fund_row["TargetK"] = (target_vnd / 1000.0) if pd.notna(target_vnd) else np.nan
    
    # --- Step 2B Scenario Neutral ---
    scenario12 = classify_scenario12(last)
    
    rrsim = build_rr_sim(trade_plans)
    
    # --- Step 1B Master Score (Fact Only) ---
    master = compute_master_score(last, dual_fib, trade_plans, fund_row)
    
    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    
    # --- Step 3B Bias (Fact Only) ---
    bias_feat = compute_rsi_macd_bias_features(rsi_feat, macd_feat)
    
    # --- Step 6 Price Action ---
    pa_feat = compute_price_action_features(df, fib_ctx=fib_ctx, vol_feat=vol_feat, tol_pct=0.8)
    
    market_ctx = compute_market_context(df_all)
    stock_chg = np.nan
    if len(df) >= 2:
        stock_chg = _pct_change(_safe_float(df.iloc[-1].get("Close")), _safe_float(df.iloc[-2].get("Close")))
    mkt_chg = _safe_float(market_ctx.get("VNINDEX", {}).get("ChangePct"))
    rel = "N/A"
    if pd.notna(stock_chg) and pd.notna(mkt_chg):
        if stock_chg > mkt_chg + 0.3: rel = "Stronger"
        elif stock_chg < mkt_chg - 0.3: rel = "Weaker"
        else: rel = "InLine"
        
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
        "ConvictionPack": conviction_pack, # Step 4B
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {}),
            "AltShort": dual_fib.get("alt_short", {}),
            "SelectionReason": dual_fib.get("selection_reason", "N/A"),
            "Context": fib_ctx
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": fund_row.get("TargetK", np.nan),
            "UpsidePct": upside_pct
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(v.entry),
                "Stop": _safe_float(v.stop),
                "TP": _safe_float(v.tp),
                "RR": _safe_float(v.rr),
                "Probability": v.probability,
                "Status": getattr(v, "status", "Watch"),
                "ReasonTags": list(getattr(v, "reason_tags", []) or [])
            } for k, v in trade_plans.items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "ProTech": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat,
            "Bias": bias_feat,
            "PriceAction": pa_feat
        },
        "Market": {
            "VNINDEX": market_ctx.get("VNINDEX", {}),
            "VN30": market_ctx.get("VN30", {}),
            "StockChangePct": stock_chg,
            "RelativeStrengthVsVNINDEX": rel
        }
    }
    
    # Primary Picker
    def pick_primary_setup_v2(rrsim: Dict[str, Any]) -> Dict[str, Any]:
        setups = rrsim.get("Setups", []) or []
        if not setups: return {"Name": "N/A", "RiskPct": np.nan, "RewardPct": np.nan, "RR": np.nan, "Probability": "N/A"}
        def status_rank(s):
            stt = (s.get("Status") or "Watch").strip().lower()
            if stt == "active": return 0
            if stt == "watch": return 1
            return 2
        best = None
        for s in setups:
            rr = _safe_float(s.get("RR"))
            if pd.isna(rr): continue
            if best is None:
                best = s
                continue
            if status_rank(s) < status_rank(best):
                best = s
                continue
            if status_rank(s) > status_rank(best):
                continue
            if rr > _safe_float(best.get("RR")):
                best = s
        best = best or setups[0]
        return {
            "Name": best.get("Setup") or best.get("Name") or "N/A",
            "RiskPct": _safe_float(best.get("RiskPct")),
            "RewardPct": _safe_float(best.get("RewardPct")),
            "RR": _safe_float(best.get("RR")),
            "Probability": best.get("Probability", "N/A")
        }
    
    primary = pick_primary_setup_v2(rrsim)
    analysis_pack["PrimarySetup"] = primary
    
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
    if "Error" in data: return f"❌ {data['Error']}"
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
    primary = (analysis_pack.get("PrimarySetup") or {})
    must_risk = primary.get("RiskPct")
    must_reward = primary.get("RewardPct")
    must_rr = primary.get("RR")
    must_prob = primary.get("Probability")
    prompt = f"""
Bạn là chuyên gia phân tích chứng khoán, nói chuyện thân thiện với “bạn”, văn phong mượt mà, rõ ràng.
TUYỆT ĐỐI:
- Không bịa số.
- Không tự tính bất kỳ con số nào.
- Chỉ dùng đúng dữ liệu trong JSON “AnalysisPack”.

YÊU CẦU FORMAT OUTPUT:
- Không dùng emoji.
- Không dùng kiểu bullet 1️⃣ 2️⃣ .
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
4) RSI + MACD Bias (dùng ProTech.Bias: BiasCode/Tags/Alignment/Facts)
5) Fibonacci (Fibonacci.ShortWindow & LongWindow + levels + SelectionReason)
6) Volume & Price Action (ProTech.Volume + ProTech.PriceAction + Last; không tự tính)
7) Scenario 12 (Scenario12: Name/Flags/RulesHit)
8) Master Integration (MasterScore + ConvictionPack)

Mục B: dùng đúng dòng dữ liệu: {fund_text}

Mục C: dùng TradePlans trong JSON, diễn giải ngắn gọn chiến lược phù hợp.

YÊU CẦU TRÌNH BÀY: C và D phải liền mạch như một khối; kết thúc mục C xong in ngay mục D (4 dòng) không chèn thêm đoạn giải thích xen kẽ.

KHÓA CỨNG MỤC D (copy đúng số, không được tự tính/ước lượng):
- Risk% = {must_risk}
- Reward% = {must_reward}
- RR = {must_rr}
- Probability = {must_prob}

Trong mục D, bắt buộc hiển thị đúng 4 dòng theo format:
Risk%: <...>
Reward%: <...>
RR: <...>
Probability: <...>

Dữ liệu (AnalysisPack JSON):
{pack_json}
"""
    try:
        content = call_gpt_with_guard(prompt, analysis_pack, max_retry=2)
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"
    return f"{header_html}\n\n{content}"

# ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================
st.markdown("<h1 style='color:#A855F7; margin-bottom:6px;'>INCEPTION v5.2</h1>", unsafe_allow_html=True)
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
    Phiên bản 5.2 | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
)
