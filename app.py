# ============================================================
# INCEPTION v5.0 FINAL | Strategic Investor Edition
# app.py — Streamlit + GPT-4 Turbo
# Key fixes (v5.0):
# - Guard-D: normalize accents for Probability + hard overwrite D if validation fails
# - Output postprocess: enforce A items "1) .. 8)" (convert "1." -> "1)")
# - Prompt: stop copying Digest raw strings; require narrative based on Digest labels
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import unicodedata
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(page_title="INCEPTION v5.0", layout="wide")

st.markdown(
    """
<style>
body { background-color:#0B0E11; color:#E5E7EB; font-family:'Segoe UI', sans-serif; }
strong { color:#E5E7EB; font-weight:700; }
h1,h2,h3 { color:#E5E7EB; }
.stButton>button{
    width:100%;
    background: linear-gradient(180deg, #111827 0%, #000000 100%);
    color:#FFFFFF !important;
    font-weight:700;
    border-radius:10px;
    height:44px;
    border:1px solid rgba(255,255,255,0.12);
    box-shadow:0 6px 14px rgba(0,0,0,0.45);
}
.stButton>button:hover{
    background: linear-gradient(180deg, #0B1220 0%, #000000 100%);
    border:1px solid rgba(255,255,255,0.18);
}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# 2. PATHS & CONSTANTS
# ============================================================

PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
TICKER_NAME_PATH = "Ticker name.xlsx"

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuan", "quota": 999},
    "KH01": {"name": "Khach moi 01", "quota": 5},
    "KH02": {"name": "Khach moi 02", "quota": 5},
    "KH03": {"name": "Khach moi 03", "quota": 5},
    "KH04": {"name": "Khach moi 04", "quota": 5},
    "KH05": {"name": "Khach moi 05", "quota": 5},
}

MIN_RR_KEEP = 2.5
PREF_RR_TARGET = 3.0


# ============================================================
# 3. HELPERS
# ============================================================

def _fmt_price(x, ndigits=2):
    if pd.isna(x): return ""
    return f"{float(x):.{ndigits}f}"

def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"{float(x):.1f}%"

def _fmt_thousand(x, ndigits=1):
    if pd.isna(x): return ""
    return f"{float(x)/1000:.{ndigits}f}"

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _round_price(x: float, ndigits: int = 2) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return round(float(x), ndigits)

def _sgn(x: float) -> int:
    if pd.isna(x): return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a - b) / b * 100

def _trend_label_from_slope(slope: float, eps: float = 1e-9) -> str:
    if pd.isna(slope): return "N/A"
    if slope > eps: return "Up"
    if slope < -eps: return "Down"
    return "Flat"

def _strip_accents(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def _norm_prob(s: str) -> str:
    # normalize case + strip accents + squeeze spaces
    s0 = _strip_accents(s).lower()
    s0 = re.sub(r"\s+", " ", s0).strip()
    return s0

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
        if s[i] == 0 or s[i - 1] == 0:
            continue
        if s[i] != s[i - 1]:
            last_event = "CrossUp" if s[i - 1] < s[i] else "CrossDown"
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

    lows, highs = [], []
    for i in range(2, n - 2):
        if c[i] < c[i - 1] and c[i] < c[i + 1]:
            lows.append(i)
        if c[i] > c[i - 1] and c[i] > c[i + 1]:
            highs.append(i)

    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if (c[i2] < c[i1]) and (o[i2] > o[i1]):
            return {"Type": "Bullish", "Detail": f"Price LL vs Osc HL (swings {i1}->{i2})"}

    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if (c[i2] > c[i1]) and (o[i2] < o[i1]):
            return {"Type": "Bearish", "Detail": f"Price HH vs Osc LH (swings {i1}->{i2})"}

    return {"Type": "None", "Detail": "N/A"}

def _scenario_vi(x: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu huong tang — Xac nhan but pha",
        "Uptrend – Pullback Phase": "Xu huong tang — Pha dieu chinh",
        "Downtrend – Weak Phase": "Xu huong giam — Yeu",
        "Neutral / Sideways": "Di ngang / Trung tinh",
    }
    return m.get(x, x)


# ============================================================
# 3B. GUARD-D
# ============================================================

def pick_primary_setup(rrsim: Dict[str, Any]) -> Dict[str, Any]:
    setups = rrsim.get("Setups", []) or []
    if not setups:
        return {"Name": "N/A", "RiskPct": np.nan, "RewardPct": np.nan, "RR": np.nan, "Probability": "N/A"}

    best = None
    for s in setups:
        rr = _safe_float(s.get("RR"))
        if pd.isna(rr):
            continue
        if (best is None) or (rr > _safe_float(best.get("RR"))):
            best = s

    best = best or setups[0]
    return {
        "Name": best.get("Setup") or best.get("Name") or "N/A",
        "RiskPct": _safe_float(best.get("RiskPct")),
        "RewardPct": _safe_float(best.get("RewardPct")),
        "RR": _safe_float(best.get("RR")),
        "Probability": best.get("Probability", "N/A")
    }

def _extract_d_block(text: str) -> str:
    m = re.search(r"(^|\n)\s*D\.\s*Rui\s*ro.*$", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not m:
        m = re.search(r"(^|\n)\s*D\.\s*Rủi\s*ro.*$", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return m.group(0) if m else text

def _grab_number(block: str, label_patterns: List[str]) -> float:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip().replace(",", "").replace("%", "")
            return _safe_float(s)
    return np.nan

def _grab_text(block: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            return (m.group(1) or "").strip()
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
    exp_prob = _norm_prob(primary.get("Probability") or "")

    ok = True
    if pd.notna(exp_risk):
        ok &= _close_enough(got_risk, exp_risk, tol=0.05)
    if pd.notna(exp_reward):
        ok &= _close_enough(got_reward, exp_reward, tol=0.05)
    if pd.notna(exp_rr):
        ok &= _close_enough(got_rr, exp_rr, tol=0.05)

    gp = _norm_prob(got_prob)
    if exp_prob:
        ok &= (exp_prob in gp) if gp else False

    return bool(ok)

def _call_openai(prompt: str, temperature: float) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Ban la INCEPTION AI, chuyen gia phan tich dau tu."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=1700
    )
    return response.choices[0].message.content

def _hard_overwrite_section_d(text: str, primary: Dict[str, Any]) -> str:
    # Always use Python truth
    rp = _safe_float(primary.get("RiskPct"))
    wp = _safe_float(primary.get("RewardPct"))
    rr = _safe_float(primary.get("RR"))
    prob = str(primary.get("Probability", "N/A"))

    d_block = (
        "D. Rui ro vs loi nhuan\n"
        f"Risk%: {rp:.2f}\n"
        f"Reward%: {wp:.2f}\n"
        f"RR: {rr:.2f}\n"
        f"Probability: {prob}\n"
    )

    # Replace existing D block if exists; else append
    m = re.search(r"(^|\n)\s*D\.\s*.*$", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if m:
        # cut from D to end, replace with our D only (keep earlier sections)
        pre = text[:m.start()].rstrip()
        return pre + "\n\n" + d_block
    return text.rstrip() + "\n\n" + d_block

def _postprocess_format(text: str) -> str:
    # Convert "1." to "1)" at line start
    text = re.sub(r"(?m)^\s*([1-8])\.\s+", r"\1) ", text)
    # Ensure headings are present as A/B/C/D (best-effort; do not invent content)
    return text

def call_gpt_with_guard(prompt: str, analysis_pack: Dict[str, Any], max_retry: int = 2) -> str:
    primary = (analysis_pack.get("PrimarySetup") or {})
    temps = [0.7, 0.2, 0.0]

    last_text = ""
    for i in range(min(max_retry + 1, len(temps))):
        temp = temps[i]
        extra = ""
        if i > 0:
            extra = f"""

SUA LOI BAT BUOC (chi sua muc D, giu nguyen cac muc khac):
COPY DUNG (khong them/bot ky tu) 4 dong sau:
Risk%: {primary.get('RiskPct')}
Reward%: {primary.get('RewardPct')}
RR: {primary.get('RR')}
Probability: {primary.get('Probability')}
"""

        text = _call_openai(prompt + extra, temperature=temp)
        last_text = text
        if validate_section_d(text, primary):
            out = _postprocess_format(text)
            return out

    # If still fail -> hard overwrite D in Python
    forced = _hard_overwrite_section_d(last_text, primary)
    forced = _postprocess_format(forced)
    return forced


# ============================================================
# 4. LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception as e:
        st.error(f"Loi khi doc file {path}: {e}")
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Ngay": "Date", "Ngày": "Date",
        "Ma": "Ticker", "Mã": "Ticker",
        "Dong cua": "Close", "Đóng cửa": "Close", "Gia dong cua": "Close", "Giá đóng cửa": "Close",
        "Mo cua": "Open", "Mở cửa": "Open",
        "Cao nhat": "High", "Cao nhất": "High",
        "Thap nhat": "Low", "Thấp nhất": "Low",
        "Vol": "Volume", "Khoi luong": "Volume", "Khối lượng": "Volume",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "Date" not in df.columns or "Ticker" not in df.columns:
        st.error("Thieu cot Date/Ticker trong Price_Vol.xlsx")
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
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
    hi = df["High"]
    lo = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ============================================================
# 6. FIBONACCI DUAL-FRAME
# ============================================================

def _fib_levels(low, high):
    rng = high - low
    if pd.isna(rng) or rng <= 0:
        return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng,
        "78.6": high - 0.786 * rng,
        "127.2": high + 0.272 * rng,
        "161.8": high + 0.618 * rng
    }

def _compute_fib_window(df: pd.DataFrame, w: int) -> Dict[str, Any]:
    L = w if len(df) >= w else len(df)
    win = df.tail(L)
    hi = win["High"].max()
    lo = win["Low"].min()
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
            "conflict_flag": False,
            "priority_rule_applied": "N/A"
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
    l_hi = win_long["High"].max()
    l_lo = win_long["Low"].min()
    fixed_long = {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)}

    s61 = _safe_float(chosen.get("levels", {}).get("61.8"))
    l61 = _safe_float(fixed_long.get("levels", {}).get("61.8"))
    conflict = False
    if pd.notna(last_close) and pd.notna(s61) and pd.notna(l61) and last_close != 0:
        conflict = (abs(s61 - l61) / last_close) >= 0.03

    return {
        "short_window": chosen.get("window"),
        "long_window": L_long,
        "auto_short": {"swing_high": chosen.get("swing_high"), "swing_low": chosen.get("swing_low"), "levels": chosen.get("levels", {})},
        "fixed_long": fixed_long,
        "alt_short": alt,
        "selection_reason": reason,
        "conflict_flag": bool(conflict),
        "priority_rule_applied": "LongStructure_ShortTactical"
    }


# ============================================================
# 6B. PRO FEATURES
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
        else:
            regime = "Neutral"

    dist50 = ((close - ma50) / ma50 * 100) if (pd.notna(close) and pd.notna(ma50) and ma50 != 0) else np.nan
    dist200 = ((close - ma200) / ma200 * 100) if (pd.notna(close) and pd.notna(ma200) and ma200 != 0) else np.nan

    return {
        "Regime": regime,
        "SlopeMA20": _trend_label_from_slope(s20),
        "SlopeMA50": _trend_label_from_slope(s50),
        "SlopeMA200": _trend_label_from_slope(s200),
        "DistToMA50Pct": dist50,
        "DistToMA200Pct": dist200,
        "Cross": {
            "PriceVsMA50": _find_last_cross(df["Close"], df["MA50"], lookback=20),
            "PriceVsMA200": _find_last_cross(df["Close"], df["MA200"], lookback=60),
            "MA20VsMA50": _find_last_cross(df["MA20"], df["MA50"], lookback=60),
            "MA50VsMA200": _find_last_cross(df["MA50"], df["MA200"], lookback=120)
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
        direction = "Rising" if delta > 1.0 else ("Falling" if delta < -1.0 else "Flat")

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

    return {"Value": last_rsi, "State": state, "Direction": direction, "Divergence": _detect_divergence_simple(df["Close"], df["RSI"], 60)}

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

    return {
        "Value": last_m,
        "Signal": last_s,
        "Hist": last_h,
        "State": state,
        "Cross": _find_last_cross(df["MACD"], df["MACDSignal"], 30),
        "ZeroLine": zero,
        "HistState": hist_state,
        "Divergence": _detect_divergence_simple(df["Close"], df["MACD"], 60)
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
    def pack(tick: str) -> Dict[str, Any]:
        d = df_all[df_all["Ticker"].astype(str).str.upper() == tick].copy()
        if d.empty or len(d) < 2:
            return {"Ticker": tick, "Close": np.nan, "ChangePct": np.nan, "Regime": "N/A"}
        d = d.sort_values("Date")
        c = _safe_float(d.iloc[-1].get("Close"))
        p = _safe_float(d.iloc[-2].get("Close"))
        chg = _pct_change(c, p)
        d["MA50"] = sma(d["Close"], 50)
        d["MA200"] = sma(d["Close"], 200)
        ma50 = _safe_float(d.iloc[-1].get("MA50"))
        ma200 = _safe_float(d.iloc[-1].get("MA200"))
        regime = "N/A"
        if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
            if c >= ma50 and ma50 >= ma200:
                regime = "Up"
            elif c < ma50 and ma50 < ma200:
                regime = "Down"
            else:
                regime = "Neutral"
        return {"Ticker": tick, "Close": c, "ChangePct": chg, "Regime": regime}

    return {"VNINDEX": pack("VNINDEX"), "VN30": pack("VN30")}


# ============================================================
# 7. CONVICTION
# ============================================================

def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if _safe_float(last.get("Close")) > _safe_float(last.get("MA200")): score += 2
    if _safe_float(last.get("RSI")) > 55: score += 1
    if _safe_float(last.get("Volume")) > _safe_float(last.get("Avg20Vol")): score += 1
    if _safe_float(last.get("MACD")) > _safe_float(last.get("MACDSignal")): score += 0.5
    return float(min(10.0, score))


# ============================================================
# 8. TRADE PLAN (DYNAMIC STOP/TP)
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str
    notes: str = ""

def _compute_rr(entry: float, stop: float, tp: float) -> float:
    if any(pd.isna([entry, stop, tp])) or entry <= stop:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def _compute_buffer(last: pd.Series) -> float:
    c = _safe_float(last.get("Close"))
    atr = _safe_float(last.get("ATR14"))
    hi = _safe_float(last.get("High"))
    lo = _safe_float(last.get("Low"))
    rng = hi - lo if pd.notna(hi) and pd.notna(lo) else np.nan

    if pd.notna(atr) and atr > 0:
        return float(0.6 * atr)
    if pd.notna(rng) and rng > 0:
        return float(0.6 * rng)
    if pd.notna(c) and c > 0:
        return float(0.01 * c)
    return np.nan

def _levels_below(price: float, levels: List[float]) -> List[float]:
    out = []
    for lv in levels:
        if pd.notna(lv) and pd.notna(price) and lv < price:
            out.append(float(lv))
    return out

def _pick_support_below(price: float, candidates: List[float]) -> float:
    below = _levels_below(price, candidates)
    return max(below) if below else np.nan

def _pick_target_from_candidates(entry: float, stop: float, candidates: List[float], pref_rr: float = PREF_RR_TARGET) -> float:
    if pd.isna(entry) or pd.isna(stop) or entry <= stop:
        return np.nan
    risk = entry - stop
    if risk <= 0:
        return np.nan

    good = []
    for tp in candidates:
        if pd.notna(tp) and tp > entry:
            rr = (tp - entry) / risk
            if 2.0 <= rr <= 5.0:
                good.append((abs(rr - pref_rr), rr, tp))
    if good:
        good.sort(key=lambda x: x[0])
        return float(good[0][2])

    return float(entry + pref_rr * risk)

def _estimate_probability(setup_name: str, ma_feat: Dict[str, Any], rsi_feat: Dict[str, Any], macd_feat: Dict[str, Any], vol_feat: Dict[str, Any]) -> str:
    # keep ASCII labels to reduce Guard mismatch
    score = 0
    regime = (ma_feat or {}).get("Regime", "Neutral")
    if regime == "Up": score += 2
    if regime == "Neutral": score += 1

    rsi_state = (rsi_feat or {}).get("State", "Neutral")
    if rsi_state in ["Bull", "Neutral+"]: score += 1

    macd_state = (macd_feat or {}).get("State", "Neutral")
    if macd_state == "Bull": score += 1

    vol_reg = (vol_feat or {}).get("Regime", "N/A")
    if vol_reg in ["High", "Spike"]: score += 1

    if setup_name.lower() == "breakout" and vol_reg in ["High", "Spike"]:
        score += 1
    if setup_name.lower() == "pullback" and rsi_state in ["Bear", "Neutral-"]:
        score += 1

    if score >= 6: return "Cao"
    if score >= 4: return "TB"
    return "Thap"

def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any], ma_feat: Dict[str, Any], rsi_feat: Dict[str, Any], macd_feat: Dict[str, Any], vol_feat: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df.empty:
        return {}

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))

    fib_short = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    fib_long = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}

    swing_hi_s = _safe_float((dual_fib.get("auto_short", {}) or {}).get("swing_high"))
    swing_hi_l = _safe_float((dual_fib.get("fixed_long", {}) or {}).get("swing_high"))

    res_zone = _safe_float(fib_short.get("61.8", np.nan))
    sup_zone = _safe_float(fib_short.get("38.2", np.nan))
    if pd.isna(res_zone) and pd.notna(close): res_zone = close
    if pd.isna(sup_zone) and pd.notna(close): sup_zone = close

    buffer = _compute_buffer(last)

    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    fib_support_keys = ["38.2", "50.0", "61.8", "78.6"]
    fib_ext_keys = ["127.2", "161.8"]

    supports = [ma20, ma50, ma200]
    for k in fib_support_keys:
        supports.append(_safe_float(fib_short.get(k)))
        supports.append(_safe_float(fib_long.get(k)))

    targets = []
    for k in fib_ext_keys:
        targets.append(_safe_float(fib_short.get(k)))
        targets.append(_safe_float(fib_long.get(k)))
    targets += [swing_hi_s, swing_hi_l]

    setups: Dict[str, TradeSetup] = {}

    # Breakout
    entry_b = _round_price(res_zone * 1.005) if pd.notna(res_zone) else np.nan
    if pd.notna(entry_b) and pd.notna(buffer):
        sup_b = _pick_support_below(entry_b, supports)
        if pd.isna(sup_b) and pd.notna(close):
            sup_b = close
        stop_b = _round_price(sup_b - buffer) if pd.notna(sup_b) else np.nan
        if pd.notna(stop_b) and stop_b >= entry_b:
            stop_b = _round_price(entry_b - buffer)
        tp_b = _round_price(_pick_target_from_candidates(entry_b, stop_b, targets, PREF_RR_TARGET))
        rr_b = _compute_rr(entry_b, stop_b, tp_b)
        if pd.notna(rr_b) and rr_b >= MIN_RR_KEEP:
            prob_b = _estimate_probability("Breakout", ma_feat, rsi_feat, macd_feat, vol_feat)
            setups["Breakout"] = TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, prob_b, "Dynamic stop (MA/Fib + buffer).")

    # Pullback
    entry_p = _round_price(sup_zone) if pd.notna(sup_zone) else np.nan
    if pd.notna(entry_p) and pd.notna(buffer):
        sup_p = _pick_support_below(entry_p, supports)
        swing_lo_s = _safe_float((dual_fib.get("auto_short", {}) or {}).get("swing_low"))
        if pd.isna(sup_p):
            sup_p = swing_lo_s
        stop_p = _round_price(sup_p - buffer) if pd.notna(sup_p) else np.nan
        if pd.notna(stop_p) and stop_p >= entry_p:
            stop_p = _round_price(entry_p - buffer)
        tp_p = _round_price(_pick_target_from_candidates(entry_p, stop_p, targets, PREF_RR_TARGET))
        rr_p = _compute_rr(entry_p, stop_p, tp_p)
        if pd.notna(rr_p) and rr_p >= MIN_RR_KEEP:
            prob_p = _estimate_probability("Pullback", ma_feat, rsi_feat, macd_feat, vol_feat)
            setups["Pullback"] = TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, prob_p, "Dynamic stop (MA/Fib + buffer).")

    return setups


# ============================================================
# 9. SCENARIOS + MASTER
# ============================================================

def classify_scenario(last: pd.Series) -> str:
    c = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20:
            return "Uptrend – Breakout Confirmation"
        if c > ma200 and ma20 > ma200:
            return "Uptrend – Pullback Phase"
        if c < ma200 and ma50 < ma200:
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
            trend = "Up"; rules_hit.append("Trend=Up (Close>=MA50>=MA200)")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"; rules_hit.append("Trend=Down (Close<MA50<MA200)")
        else:
            trend = "Neutral"; rules_hit.append("Trend=Neutral (mixed MA structure)")

    mom = "Neutral"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = "Bull"; rules_hit.append("Momentum=Bull (RSI>=55 & MACD>=Signal)")
        elif (rsi <= 45) and (macd_v < sig):
            mom = "Bear"; rules_hit.append("Momentum=Bear (RSI<=45 & MACD<Signal)")
        elif (rsi >= 70):
            mom = "Exhaust"; rules_hit.append("Momentum=Exhaust (RSI>=70)")
        else:
            mom = "Neutral"; rules_hit.append("Momentum=Neutral (between zones)")

    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")

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

    return {"Code": int(code), "TrendRegime": trend, "MomentumRegime": mom, "VolumeRegime": vol_reg, "Name": name_map.get((trend, mom), "Scenario – N/A"), "RulesHit": rules_hit}

def build_rr_sim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    rows = []
    best_rr = np.nan
    for k, s in (trade_plans or {}).items():
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

def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup], upside_pct: float) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    comps: Dict[str, float] = {}

    trend = 0.0
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        trend = 2.0 if (c >= ma50 and ma50 >= ma200) else (1.2 if c >= ma200 else 0.4)
    comps["Trend"] = trend

    mom = 0.0
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        mom = 2.0 if (rsi >= 55 and macd_v >= sig) else (0.4 if (rsi <= 45 and macd_v < sig) else 1.1)
    comps["Momentum"] = mom

    vcomp = 0.0
    if pd.notna(vol) and pd.notna(avg_vol):
        vcomp = 1.6 if vol > avg_vol else 0.9
    comps["Volume"] = vcomp

    fibc = 0.0
    try:
        s_lv = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
        l_lv = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))

        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            fibc += 1.2 if c >= s_618 else (0.8 if c >= s_382 else 0.4)
        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            fibc += 0.8 if c >= l_618 else (0.5 if c >= l_382 else 0.2)
    except Exception:
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
    if pd.notna(upside_pct) and upside_pct > 0:
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

    return {"Components": comps, "Total": round(total, 2), "Tier": tier, "PositionSizing": sizing, "BestRR": best_rr if pd.notna(best_rr) else np.nan}


# ============================================================
# 10. MAIN ANALYSIS
# ============================================================

def analyze_ticker(ticker: str) -> Dict[str, Any]:
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Error": "Khong doc duoc du lieu Price_Vol.xlsx"}

    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Khong tim thay ma {ticker}"}

    if df["Close"].dropna().empty:
        return {"Error": "Thieu du lieu Close"}

    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    df["ATR14"] = atr_wilder(df, 14)

    dual_fib = compute_dual_fibonacci_auto(df, 250)
    last = df.iloc[-1]

    conviction = compute_conviction(last)
    scenario = classify_scenario(last)

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    close = _safe_float(last.get("Close"))
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)

    close_for_calc = close
    target_for_calc = target_vnd
    if pd.notna(close) and pd.notna(target_vnd):
        if (close < 500) and (target_vnd > 1000):
            target_for_calc = target_vnd / 1000.0

    upside_pct = ((target_for_calc - close_for_calc) / close_for_calc * 100) if (pd.notna(target_for_calc) and pd.notna(close_for_calc) and close_for_calc != 0) else np.nan

    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    market_ctx = compute_market_context(df_all)

    trade_plans = build_trade_plan(df, dual_fib, ma_feat, rsi_feat, macd_feat, vol_feat)
    scenario12 = classify_scenario12(last)
    rrsim = build_rr_sim(trade_plans)
    master = compute_master_score(last, dual_fib, trade_plans, upside_pct)

    stock_chg = np.nan
    if len(df) >= 2:
        stock_chg = _pct_change(_safe_float(df.iloc[-1].get("Close")), _safe_float(df.iloc[-2].get("Close")))
    mkt_chg = _safe_float(market_ctx.get("VNINDEX", {}).get("ChangePct"))
    rel = "N/A"
    if pd.notna(stock_chg) and pd.notna(mkt_chg):
        rel = "Stronger" if stock_chg > mkt_chg + 0.3 else ("Weaker" if stock_chg < mkt_chg - 0.3 else "InLine")

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
            "ATR14": _safe_float(last.get("ATR14")),
        },
        "ScenarioBase": scenario,
        "Scenario12": scenario12,
        "Conviction": conviction,
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "SelectionReason": dual_fib.get("selection_reason", "N/A"),
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": (target_vnd / 1000.0) if pd.notna(target_vnd) else np.nan,
            "UpsidePct": upside_pct,
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(v.entry),
                "Stop": _safe_float(v.stop),
                "TP": _safe_float(v.tp),
                "RR": _safe_float(v.rr),
                "Probability": v.probability,
                "Notes": v.notes
            } for k, v in (trade_plans or {}).items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "ProTech": {"MA": ma_feat, "RSI": rsi_feat, "MACD": macd_feat, "Volume": vol_feat},
        "Market": {"VNINDEX": market_ctx.get("VNINDEX", {}), "VN30": market_ctx.get("VN30", {}), "StockChangePct": stock_chg, "RelativeStrengthVsVNINDEX": rel},
    }

    analysis_pack["PrimarySetup"] = pick_primary_setup(rrsim)

    return {"Ticker": ticker.upper(), "Last": last.to_dict(), "Scenario": scenario, "Conviction": conviction, "Fundamental": fund_row, "AnalysisPack": analysis_pack}


# ============================================================
# 11. GPT REPORT
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data:
        return f"Loi: {data['Error']}"

    tick = data["Ticker"]
    scenario = data["Scenario"]
    conviction = data["Conviction"]
    analysis_pack = data.get("AnalysisPack", {})

    last = data["Last"]
    close = _fmt_price(last.get("Close"))
    header_html = f"<h2 style='margin:0; padding:0; font-size:26px; line-height:1.2;'>{tick} — {close} | Diem tin cay: {conviction:.1f}/10 | {_scenario_vi(scenario)}</h2>"

    fund = analysis_pack.get("Fundamental", {}) or {}
    fund_text = (
        f"Khuyen nghi: {fund.get('Recommendation', 'N/A')} | "
        f"Gia muc tieu: {_fmt_thousand(fund.get('TargetVND'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if fund else "Khong co du lieu co ban"
    )

    pack_json = json.dumps(analysis_pack, ensure_ascii=False)
    primary = (analysis_pack.get("PrimarySetup") or {})

    prompt = f"""
Ban la chuyen gia phan tich chung khoan, van phong muot ma, ro rang.
TUYET DOI:
- Khong bia so.
- Khong tu tinh bat ky con so nao.
- Khong copy/paste nguyen chuoi ky thuat dang "label=...; label=...".
- Chi dung du lieu trong JSON "AnalysisPack".

YEU CAU FORMAT OUTPUT (bat buoc dung dung):
A. Ky thuat
1) ...
2) ...
3) ...
4) ...
5) ...
6) ...
7) ...
8) ...

B. Co ban
...

C. Trade plan
...

D. Rui ro vs loi nhuan
Risk%: ...
Reward%: ...
RR: ...
Probability: ...

Goi y noi dung A:
- Dien giai tu ProTech (MA/RSI/MACD/Volume) va Scenario12 + MasterScore.
- Moi dong 1–2 con so toi da, uu tien nhan dinh.

Muc B: dung dung dong: {fund_text}

KHOA CUNG MUC D (copy dung gia tri, khong them/bot):
Risk%: {primary.get('RiskPct')}
Reward%: {primary.get('RewardPct')}
RR: {primary.get('RR')}
Probability: {primary.get('Probability')}

Du lieu JSON:
{pack_json}
"""

    try:
        content = call_gpt_with_guard(prompt, analysis_pack, max_retry=2)
    except Exception as e:
        content = f"Loi khi goi GPT: {e}"

    return f"{header_html}\n\n{content}"


# ============================================================
# 12. UI
# ============================================================

st.markdown("<h1 style='color:#A855F7; margin-bottom:6px;'>INCEPTION v5.0</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### Dang nhap nguoi dung")
    user_key = st.text_input("Nhap Ma VIP:", type="password")
    ticker_input = st.text_input("Ma Co Phieu:", value="VCB").upper()
    run_btn = st.button("Phan tich", type="primary", use_container_width=True)

# ============================================================
# 13. MAIN
# ============================================================

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("Ma VIP khong dung. Vui long nhap lai.")
    else:
        with st.spinner(f"Dang xu ly phan tich {ticker_input}..."):
            try:
                result = analyze_ticker(ticker_input)
                report = generate_insight_report(result)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(report, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Loi xu ly: {e}")

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <p style='text-align:center; color:#6B7280; font-size:13px;'>
    INCEPTION v5.0
    </p>
    """,
    unsafe_allow_html=True
)
