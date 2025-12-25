# ============================================================
# INCEPTION v4.9.1 FINAL | Frame Locked Edition
# app.py — Streamlit + GPT-4 Turbo
# Author: INCEPTION AI Research Framework
# Purpose: Locked-Frame Technical–Fundamental Research Assistant
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="INCEPTION v4.9.1 – Frame Locked Edition",
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
strong, h1, h2, h3 {
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
# 4. DATA LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().title() for c in df.columns]
        rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
        df.rename(columns=rename, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
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
# 6. FIBONACCI DUAL FRAME
# ============================================================

def _fib_levels(low, high):
    rng = high - low
    if rng <= 0: return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng
    }

def compute_dual_fibonacci(df: pd.DataFrame):
    L_short = min(len(df), 90)
    L_long = min(len(df), 250)
    win_short = df.tail(L_short)
    win_long = df.tail(L_long)
    s_hi, s_lo = win_short["High"].max(), win_short["Low"].min()
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()
    return {
        "short": {"levels": _fib_levels(s_lo, s_hi)},
        "long": {"levels": _fib_levels(l_lo, l_hi)}
    }

# ============================================================
# 7. CONVICTION + TRADE LOGIC
# ============================================================

def compute_conviction(last):
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)

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

def _compute_rr(entry, stop, tp):
    if any(pd.isna([entry, stop, tp])) or entry <= 0:
        return np.nan
    risk = abs(entry - stop)
    reward = abs(tp - entry)
    return round(reward / risk, 2) if risk > 0 else np.nan

def build_trade_plan(df, dual_fib):
    last = df.iloc[-1]
    close = last["Close"]
    fib_short = dual_fib["short"]["levels"]
    res = fib_short.get("61.8", close * 1.05)
    sup = fib_short.get("38.2", close * 0.95)

    entry_b = max(res * 1.01, close * 1.02)
    stop_b = entry_b * 0.94
    tp_b = entry_b * 1.25
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    entry_p = min(sup, close * 0.98)
    stop_p = entry_p * 0.94
    tp_p = entry_p * 1.20
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    stop_pct_b = round((stop_b - entry_b) / entry_b * 100, 1)
    tp_pct_b = round((tp_b - entry_b) / entry_b * 100, 1)
    stop_pct_p = round((stop_p - entry_p) / entry_p * 100, 1)
    tp_pct_p = round((tp_p - entry_p) / entry_p * 100, 1)

    return {
        "Breakout": TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao", stop_pct_b, tp_pct_b),
        "Pullback": TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, "TB", stop_pct_p, tp_pct_p)
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
# 8. ANALYSIS FUNCTION
# ============================================================

def analyze_ticker(ticker: str):
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Error": "Không có dữ liệu"}

    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {ticker}"}

    df["MA20"], df["MA50"], df["MA200"] = sma(df["Close"], 20), sma(df["Close"], 50), sma(df["Close"], 200)
    df["Avg20Vol"], df["RSI"] = sma(df["Volume"], 20), rsi_wilder(df["Close"])
    m, s, h = macd(df["Close"])
    df["MACD"], df["MACDSignal"] = m, s

    dual_fib = compute_dual_fibonacci(df)
    last = df.iloc[-1]
    conviction = compute_conviction(last)
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)
    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}

    return {
        "Ticker": ticker.upper(),
        "Last": last.to_dict(),
        "Scenario": scenario,
        "Conviction": conviction,
        "TradePlans": trade_plans,
        "Fundamental": fund_row
    }

# ============================================================
# 9. GPT-4 REPORT GENERATION
# ============================================================

def generate_insight_report(data: Dict[str, Any]):
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick, last = data["Ticker"], data["Last"]
    trade_plans, fund, conviction, scenario = data["TradePlans"], data["Fundamental"], data["Conviction"], data["Scenario"]

    close = _fmt_price(last.get("Close"))
    ma20, ma50, ma200 = _fmt_price(last.get("MA20")), _fmt_price(last.get("MA50")), _fmt_price(last.get("MA200"))
    rsi, macd_v = _fmt_price(last.get("RSI")), _fmt_price(last.get("MACD"))
    vol, avg_vol = _fmt_int(last.get("Volume")), _fmt_int(last.get("Avg20Vol"))

    trade_table = "| Chiến lược | Entry (ưu tiên) | Stop-loss | Take-profit | Xác suất | R:R ước tính |\n"
    trade_table += "|------------|----------------|------------|--------------|-----------|--------------|\n"
    for v in trade_plans.values():
        trade_table += f"| {v.name} | {v.entry:.2f} | {v.stop:.2f} ({v.stop_pct}%) | {v.tp:.2f} (+{v.tp_pct}%) | {v.prob} | {v.rr:.2f} |\n"

    fund_text = f"Target: {_fmt_price(fund.get('Target'))}, Upside: {_fmt_pct(fund.get('Upside', 0)*100)}" if fund else "Không có dữ liệu định giá"

    prompt = f"""
Bạn là INCEPTION AI, chuyên gia phân tích chiến lược.
Hãy viết báo cáo ngắn gọn, định dạng chuẩn như sau, không chèn emoji, không đổi khung:

A. Phân tích Kỹ thuật
- Close: {close}
- Volume: {vol} | Avg20 Vol: {avg_vol}
- MA20 / MA50 / MA200: {ma20} / {ma50} / {ma200}
- RSI (14): {rsi}
- MACD: {macd_v}
- Kịch bản: {scenario}
Hãy nêu nhận định xu hướng, vùng giá hỗ trợ/kháng cự (dựa trên Fibo), và chiến lược phù hợp.

B. Fundamental Summary
- {fund_text}

C. Trade Plan & Risk–Reward Simulation
{trade_table}

Phân tích: Tóm tắt R:R trung bình, chiến lược ưu tiên và điều kiện hành động phù hợp.
Ngữ điệu chuyên nghiệp, thân thiện, không tự tạo cấu trúc khác.
"""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1800
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"Lỗi gọi GPT: {e}"

    header = f"### {tick} — {close}  ⭐ {conviction:.1f}/10<br><small>{scenario}</small>"
    return f"{header}\n\n{content}"

# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.markdown("<h1>INCEPTION v4.9.1 — Frame Locked Edition</h1>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("Phân tích ngay")

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])

with col2:
    if run_btn:
        if user_key not in VALID_KEYS:
            st.error("Sai mã VIP.")
        else:
            with st.spinner(f"Đang xử lý {ticker_input}..."):
                data = analyze_ticker(ticker_input)
                report = generate_insight_report(data)
                st.markdown(report, unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;'>Nhập mã cổ phiếu và nhấn “Phân tích ngay”.</div>", unsafe_allow_html=True)

st.divider()
st.markdown(
    "<p style='text-align:center; font-size:13px;'>© 2025 INCEPTION Research Framework | Version 4.9.1</p>",
    unsafe_allow_html=True
)
