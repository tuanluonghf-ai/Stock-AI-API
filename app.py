# ============================================================
# INCEPTION v5.1 | Full Integration Edition
# Author: INCEPTION AI Research Framework
# Purpose: Technical–Fundamental Integrated Research Assistant
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

st.set_page_config(page_title="INCEPTION v5.1", layout="wide", page_icon="🟣")

# ==== FIXED THEME & FONT (DARK MODE IMMUNITY) ====
st.markdown("""
<style>
:root { color-scheme: only dark; }
body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #0B0E11 !important;
    color: #E5E7EB !important;
    font-family: 'Segoe UI', sans-serif !important;
}
h1, h2, h3, strong { color: #FFFFFF !important; font-weight: 700 !important; }
ul { list-style-type: disc; padding-left: 1.5rem; }
a, a:visited { color: #A855F7 !important; }
table {
    color: #E5E7EB !important;
    background-color: #111418 !important;
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #272B30;
    padding: 6px 10px;
    text-align: center;
}
th { font-weight: 700; background-color: #1A1E22; }
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

def _safe_float(x, default=np.nan):
    try: return float(x)
    except: return default

def _round_price(x: float, ndigits: int = 2):
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

# ============================================================
# 4. LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path: str = PRICE_VOL_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().title() for c in df.columns]
        rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
        df.rename(columns=rename, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
        return df
    except Exception as e:
        st.error(f"Lỗi đọc {path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_hsc_targets(path: str = HSC_TARGET_PATH):
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
# 6. FIBONACCI + CONVICTION
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

def compute_dual_fibonacci(df):
    L_short = 60 if len(df) >= 60 else len(df)
    L_long = 250 if len(df) >= 250 else len(df)
    win_short = df.tail(L_short)
    win_long = df.tail(L_long)
    s_hi, s_lo = win_short["High"].max(), win_short["Low"].min()
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()
    return {
        "auto_short": {"swing_high": s_hi, "swing_low": s_lo, "levels": _fib_levels(s_lo, s_hi)},
        "fixed_long": {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)}
    }

def compute_conviction(last):
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)

# ============================================================
# 7. TRADE PLAN & SCENARIO
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str

def _compute_rr(entry, stop, tp):
    if any(pd.isna([entry, stop, tp])) or entry <= stop: return np.nan
    risk = entry - stop
    reward = tp - entry
    return reward / risk if risk > 0 else np.nan

def build_trade_plan(df, dual_fib):
    if df.empty: return {}
    last = df.iloc[-1]
    close, ma20 = last["Close"], last["MA20"]
    fib_short = dual_fib["auto_short"]["levels"]
    res_zone, sup_zone = fib_short.get("61.8", close * 1.05), fib_short.get("38.2", close * 0.95)
    entry_b, stop_b, tp_b = res_zone * 1.01, ma20 * 0.985, res_zone * 1.25
    rr_b = _compute_rr(entry_b, stop_b, tp_b)
    entry_p, stop_p, tp_p = sup_zone, sup_zone * 0.94, sup_zone * 1.20
    rr_p = _compute_rr(entry_p, stop_p, tp_p)
    setups = {}
    if rr_b >= 2.0: setups["Breakout"] = TradeSetup("Breakout", _round_price(entry_b), _round_price(stop_b), _round_price(tp_b), rr_b, "Cao")
    if rr_p >= 2.0: setups["Pullback"] = TradeSetup("Pullback", _round_price(entry_p), _round_price(stop_p), _round_price(tp_p), rr_p, "TB")
    return setups

def classify_scenario(last):
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if ma20 > ma50 > ma200 and c > ma20: return "Xu hướng tăng – Bứt phá"
    elif c > ma200 and ma20 > ma200: return "Xu hướng tăng – Nhịp điều chỉnh"
    elif c < ma200 and ma50 < ma200: return "Xu hướng giảm – Yếu"
    else: return "Trung tính / Đi ngang"

# ============================================================
# 8. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_ticker(ticker):
    df_all = load_price_vol()
    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    df["MA20"], df["MA50"], df["MA200"] = sma(df["Close"], 20), sma(df["Close"], 50), sma(df["Close"], 200)
    df["Avg20Vol"], df["RSI"] = sma(df["Volume"], 20), rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"] = m, s
    dual_fib, last, conviction = compute_dual_fibonacci(df), df.iloc[-1], compute_conviction(df.iloc[-1])
    scenario, trade_plans = classify_scenario(last), build_trade_plan(df, dual_fib)
    hsc = load_hsc_targets()
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}
    return {"Ticker": ticker, "Last": last.to_dict(), "Scenario": scenario, "Conviction": conviction,
            "DualFibo": dual_fib, "TradePlans": trade_plans, "Fundamental": fund_row}

# ============================================================
# 9. GPT STRATEGIC INSIGHT GENERATION
# ============================================================

def generate_insight_report(data):
    if "Error" in data: return f"❌ {data['Error']}"
    tick, last, trade_plans, fund, conviction, scenario = data["Ticker"], data["Last"], data["TradePlans"], data["Fundamental"], data["Conviction"], data["Scenario"]
    close, vol, avg_vol, ma20, ma50, ma200, rsi, macd_v, sig = _fmt_price(last["Close"]), _fmt_int(last["Volume"]/1_000_000), _fmt_int(last["Avg20Vol"]/1_000_000), _fmt_price(last["MA20"]), _fmt_price(last["MA50"]), _fmt_price(last["MA200"]), _fmt_price(last["RSI"]), _fmt_price(last["MACD"]), _fmt_price(last["MACDSignal"])

    header = f"**{tick} — {close} | ⭐ {conviction:.1f}/10 | {scenario}**"
    snap = f"""
• Close: {close}  
• Volume: {vol} tr | so với Avg20 Vol: {avg_vol} tr  
• MA20 / MA50 / MA200: {ma20} / {ma50} / {ma200}  
• RSI (14): ~{rsi}  
• MACD / Signal: {macd_v} / {sig}
"""
    tp_summary = []
    for k, s in trade_plans.items():
        tp_summary.append(f"{k}: Entry {s.entry}, Stop {s.stop}, TP {s.tp}, R:R {s.rr:.2f}")
    tp_text = " | ".join(tp_summary) if tp_summary else "Chưa có chiến lược đạt chuẩn."

    fund_text = (f"Giá mục tiêu: {_fmt_price(fund.get('Target'))} | Upside: {_fmt_pct(fund.get('Upside',0)*100)}" if fund else "Không có dữ liệu cơ bản")

    prompt = f"""
Bạn là chuyên gia phân tích chiến lược. Viết báo cáo (~700-900 từ) theo cấu trúc:
- Đoạn mở đầu: đánh giá tổng thể xu hướng & vị thế cổ phiếu {tick} so với thị trường (VNINDEX, VN30), ảnh hưởng tới chiến lược hành động.
- A. Phân tích kỹ thuật (sau đoạn Technical Snapshot bên dưới):
{snap}
  1. MA Trend
  2. RSI
  3. MACD
  4. RSI + MACD Bias Matrix
  5. Fibonacci
  6. Volume & Price Action
  7. Kịch bản tiềm năng
  8. Độ tin cậy (⭐ {conviction:.1f}/10)
- B. Phân tích cơ bản: {fund_text}
- C. Trade Plan & Risk–Reward: {tp_text}
Giọng văn: gần gũi, chuyên nghiệp, tự nhiên như đang tư vấn cho khách hàng tổ chức. Không bịa số.
"""
    try:
        client = OpenAI()
        r = client.chat.completions.create(model="gpt-4-turbo",
            messages=[{"role":"system","content":"Bạn là INCEPTION AI, chuyên gia phân tích chứng khoán."},
                      {"role":"user","content":prompt}],
            temperature=0.7, max_tokens=1600)
        return f"{header}\n\n{r.choices[0].message.content}"
    except Exception as e:
        return f"⚠️ Lỗi khi gọi GPT: {e}"

# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v5.1</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;'>Công cụ phân tích chiến lược cho nhà đầu tư trung–dài hạn (15–100% lợi nhuận, rủi ro 5–8%).</p>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] button {
        width: 75% !important; margin-left: 12%; margin-top: 6px;
        border-radius: 10px; background-color: #7C3AED !important;
        color: white !important; font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #A855F7 !important; transform: scale(1.03);
    }
    </style>
    """, unsafe_allow_html=True)
    run_btn = st.button("🚀 Phân tích ngay")
    base_btn = st.button("📊 Phân tích cơ bản")
    news_btn = st.button("📰 Tin tức")

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng.")
    else:
        with st.spinner(f"Đang xử lý phân tích {ticker_input}..."):
            data = analyze_ticker(ticker_input)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(generate_insight_report(data))
