# ============================================================
# INCEPTION v5.0 FINAL | FRAME ENFORCED Edition
# app.py — Streamlit + GPT-4 Turbo
# Author: INCEPTION AI Research Framework
# Purpose: Locked Technical–Fundamental Insight Generator
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any

# ============================================================
# 1. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="INCEPTION v5.0 – FRAME ENFORCED Edition",
    layout="wide",
    page_icon="🟣"
)

st.markdown("""
<style>
body {
    background-color: #FFFFFF;
    color: #000000;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, strong, b {
    color: #000000;
    font-weight: 700;
}
table, th, td {
    border: 1px solid #000000;
    color: #000000 !important;
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

def _round_price(x: float, ndigits: int = 2) -> float:
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

# ============================================================
# 4. LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path=PRICE_VOL_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().title() for c in df.columns]
        df.rename(columns={"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_hsc_targets(path=HSC_TARGET_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns={"TP (VND)": "Target"}, inplace=True)
        df["Upside"] = pd.to_numeric(df.get("Upside/Downside", 0), errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Target", "Upside"])

# ============================================================
# 5. INDICATORS
# ============================================================

def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(close, period=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
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
# 6. FIBONACCI
# ============================================================

def _fib_levels(low, high):
    rng = high - low
    if rng <= 0: return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng
    }

def compute_dual_fibonacci(df):
    L_short = min(len(df), 90)
    L_long = min(len(df), 250)
    s_hi, s_lo = df["High"].tail(L_short).max(), df["Low"].tail(L_short).min()
    l_hi, l_lo = df["High"].tail(L_long).max(), df["Low"].tail(L_long).min()
    return {
        "short": _fib_levels(s_lo, s_hi),
        "long": _fib_levels(l_lo, l_hi)
    }

# ============================================================
# 7. CONVICTION & TRADE PLAN
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    prob: str
    stop_pct: float
    tp_pct: float

def compute_conviction(last):
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)

def _compute_rr(entry, stop, tp):
    if entry <= 0 or stop <= 0 or tp <= 0: return np.nan
    risk, reward = abs(entry - stop), abs(tp - entry)
    return round(reward / risk, 2) if risk > 0 else np.nan

def build_trade_plan(df, fib):
    last = df.iloc[-1]
    close = last["Close"]
    res = fib["short"].get("61.8", close * 1.05)
    sup = fib["short"].get("38.2", close * 0.95)

    entry_b = max(res * 1.01, close * 1.02)
    stop_b = entry_b * 0.94
    tp_b = entry_b * 1.25
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    entry_p = min(sup, close * 0.98)
    stop_p = entry_p * 0.94
    tp_p = entry_p * 1.20
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    return {
        "Breakout": TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao",
                               round((stop_b - entry_b)/entry_b*100,1), round((tp_b - entry_b)/entry_b*100,1)),
        "Pullback": TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, "Trung bình",
                               round((stop_p - entry_p)/entry_p*100,1), round((tp_p - entry_p)/entry_p*100,1))
    }

def classify_scenario(last):
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if ma20 > ma50 > ma200 and c > ma20:
        return "Uptrend – Breakout Confirmation"
    elif c > ma200 and ma20 > ma200:
        return "Uptrend – Pullback Phase"
    elif c < ma200 and ma50 < ma200:
        return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# ============================================================
# 8. ANALYZE
# ============================================================

def analyze_ticker(ticker):
    df_all = load_price_vol()
    if df_all.empty: return {"Error": "Không có dữ liệu"}

    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty: return {"Error": f"Không tìm thấy mã {ticker}"}

    df["MA20"], df["MA50"], df["MA200"] = sma(df["Close"],20), sma(df["Close"],50), sma(df["Close"],200)
    df["Avg20Vol"], df["RSI"] = sma(df["Volume"],20), rsi_wilder(df["Close"])
    m, s, _ = macd(df["Close"])
    df["MACD"], df["MACDSignal"] = m, s

    fib = compute_dual_fibonacci(df)
    last = df.iloc[-1]
    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trades = build_trade_plan(df, fib)
    fund = load_hsc_targets()
    fund = fund[fund["Ticker"].str.upper()==ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    return {
        "Ticker": ticker,
        "Last": last.to_dict(),
        "Scenario": scenario,
        "Conviction": conviction,
        "Fib": fib,
        "TradePlans": trades,
        "Fundamental": fund_row
    }

# ============================================================
# 9. GPT REPORT
# ============================================================

def generate_report(data):
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick, last, fib, trades = data["Ticker"], data["Last"], data["Fib"], data["TradePlans"]
    fund, conviction, scenario = data["Fundamental"], data["Conviction"], data["Scenario"]

    close = _fmt_price(last["Close"])
    ma20, ma50, ma200 = _fmt_price(last["MA20"]), _fmt_price(last["MA50"]), _fmt_price(last["MA200"])
    rsi, macd_v = _fmt_price(last["RSI"]), _fmt_price(last["MACD"])
    vol, avg_vol = _fmt_int(last["Volume"]), _fmt_int(last["Avg20Vol"])

    fund_text = f"Target: {_fmt_price(fund.get('Target'))}, Upside: {_fmt_pct(fund.get('Upside',0)*100)}" if fund else "Không có dữ liệu"
    trade_table = "| Chiến lược | Entry (ưu tiên) | Stop-loss | Take-profit | Xác suất | R:R ước tính |\n|-------------|----------------|------------|--------------|-----------|--------------|\n"
    for v in trades.values():
        trade_table += f"| {v.name} | {v.entry:.2f} | {v.stop:.2f} ({v.stop_pct}%) | {v.tp:.2f} (+{v.tp_pct}%) | {v.prob} | {v.rr:.2f} |\n"

    prompt = f"""
Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược.
Hãy viết báo cáo hoàn chỉnh theo cấu trúc cố định dưới đây (không thay đổi bố cục, không tự tạo mục khác):

A. Phân tích Kỹ thuật
1. MA Trend Analysis
2. RSI Analysis
3. MACD Analysis
4. RSI + MACD Bias Matrix
5. Fibonacci Levels
6. Volume & Price Action
7. Kịch bản tiềm năng
8. Độ tin cậy (⭐ {conviction:.1f}/10 → Xu hướng nghiêng ...)

B. Fundamental Summary
- {fund_text}

C. Trade Plan & Risk–Reward Simulation
{trade_table}

Chỉ phân tích dựa trên dữ liệu:
Close={close}, MA20={ma20}, MA50={ma50}, MA200={ma200}, RSI={rsi}, MACD={macd_v}, Volume={vol}, Avg20Vol={avg_vol}.
Văn phong chuyên nghiệp, gần gũi, dễ hiểu, khuyến nghị mang tính chiến lược.
"""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role":"system","content":"Bạn là INCEPTION AI, chuyên gia phân tích chiến lược."},
                      {"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=1800
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = f"Lỗi GPT: {e}"

    header = f"### {tick} — {close} ⭐ {conviction:.1f}/10<br><small>{scenario}</small>"
    return f"{header}\n\n{text}"

# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.markdown("<h1>INCEPTION v5.0 — FRAME ENFORCED Edition</h1>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### Đăng nhập người dùng")
    key = st.text_input("Nhập Mã VIP:", type="password")
    ticker = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run = st.button("Phân tích ngay")

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

with col2:
    if run:
        if key not in VALID_KEYS:
            st.error("Sai mã VIP.")
        else:
            with st.spinner(f"Đang phân tích {ticker}..."):
                result = analyze_ticker(ticker)
                report = generate_report(result)
                st.markdown(report, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;'>Nhập mã cổ phiếu và nhấn “Phân tích ngay”.</div>", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;font-size:13px;'>© 2025 INCEPTION Research Framework | Version 5.0</p>", unsafe_allow_html=True)
