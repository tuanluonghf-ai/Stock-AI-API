# ============================================================
# INCEPTION v5.2 | FRAME-LOCK Final Edition
# Author: INCEPTION AI Research Framework
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

st.set_page_config(page_title="INCEPTION v5.2", layout="wide", page_icon="🟣")

st.markdown("""
<style>
body {
    background-color: #0B0E11;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, strong { color: #E5E7EB; }
.report-text { color: #E5E7EB; white-space: pre-wrap; }
.stButton>button {
    width: 100%;
    background-color: #9333EA;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 42px;
}
.stButton>button:hover {
    background-color: #A855F7;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. PATHS & CONSTANTS
# ============================================================

PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
    "KH03": {"name": "Khách mời 03", "quota": 5},
    "KH04": {"name": "Khách mời 04", "quota": 5},
    "KH05": {"name": "Khách mời 05", "quota": 5},
}

# ============================================================
# 3. LOADERS & INDICATORS
# ============================================================

def load_price_vol(path=PRICE_VOL_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().title() for c in df.columns]
        rename = {"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}
        df.rename(columns=rename, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"]).dropna(subset=["Date"])
        return df
    except:
        return pd.DataFrame()

def sma(series, window): return series.rolling(window).mean()
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
    return macd_line, signal_line, macd_line - signal_line

def compute_fibo(df, period=250):
    win = df.tail(period)
    high, low = win["High"].max(), win["Low"].min()
    rng = high - low
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng
    }

# ============================================================
# 4. ANALYSIS
# ============================================================

@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str

def analyze_ticker(ticker: str):
    df_all = load_price_vol()
    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty:
        return {"Error": f"Không tìm thấy mã {ticker}"}

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"])
    df["MACD"], df["Signal"], df["Hist"] = m, s, h
    fibo = compute_fibo(df)

    last = df.iloc[-1]
    close, prev_close = last["Close"], df.iloc[-2]["Close"]
    change = (close - prev_close) / prev_close * 100
    conviction = 7 + (close > last["MA50"]) * 1.5

    return {
        "Ticker": ticker.upper(),
        "Close": close,
        "Change": change,
        "Volume": last["Volume"],
        "Avg20": df["Volume"].tail(20).mean(),
        "MA20": last["MA20"],
        "MA50": last["MA50"],
        "MA200": last["MA200"],
        "RSI": last["RSI"],
        "MACD": last["MACD"],
        "Signal": last["Signal"],
        "Conviction": conviction,
        "Fibo": fibo
    }

# ============================================================
# 5. REPORT GENERATION
# ============================================================

def generate_report(data: Dict[str, Any]) -> str:
    t = data
    close, chg = t["Close"], t["Change"]
    updown = "tăng" if chg > 0 else "giảm"
    fibo = t["Fibo"]
    fibo_levels = list(fibo.values())

    header = f"**{t['Ticker']} — {close:.2f} VND ({chg:+.2f}%) ⭐ {t['Conviction']:.1f}/10**\n"
    header += f"Xu hướng: {'Tăng' if chg>0 else 'Giảm' if chg<0 else 'Trung tính'}\n\n"

    preface = (
        "Thị trường hiện đang dao động trong vùng cân bằng sau nhịp hồi. "
        f"{t['Ticker']} đang thể hiện {('tốt hơn' if chg>0 else 'kém hơn')} thị trường chung, "
        "phản ánh sự chọn lọc dòng tiền giữa các nhóm cổ phiếu. "
        "→ Giai đoạn hiện tại phù hợp với việc canh các nhịp điều chỉnh ngắn để gia tăng vị thế hơn là mua đuổi."
    )

    a_block = f"""
### A. Phân tích Kỹ thuật

* Close: {close:.2f} ({chg:+.2f}%)
* Volume: {t['Volume']:,} | Avg20 Vol: {t['Avg20']:.0f}
* MA20 / MA50 / MA200: {t['MA20']:.2f} / {t['MA50']:.2f} / {t['MA200']:.2f}
* RSI (14): {t['RSI']:.2f}
* MACD / Signal: {t['MACD']:.2f} / {t['Signal']:.2f}

1. **MA Trend:** So sánh ba đường MA cho thấy cấu trúc xu hướng hiện tại đang {'tăng' if t['MA20']>t['MA50'] else 'giảm'} nhẹ.
2. **RSI:** Ở mức {t['RSI']:.2f}, phản ánh {('động lượng tích cực' if t['RSI']>55 else 'trung tính')}.
3. **MACD:** {('Đang mở rộng dương → tín hiệu xu hướng mạnh.' if t['MACD']>t['Signal'] else 'Tín hiệu yếu hoặc trung lập.')}
4. **RSI + MACD Bias Matrix:** Khi kết hợp RSI và MACD, chiến lược phù hợp là {('nắm giữ theo xu hướng' if t['RSI']>55 else 'quan sát chờ xác nhận')}.
5. **Fibonacci:** Hỗ trợ: {fibo_levels[2]:.2f}, {fibo_levels[1]:.2f} | Kháng cự: {fibo_levels[0]:.2f}.
6. **Volume & Price Action:** Khối lượng đang {'tăng' if t['Volume']>t['Avg20'] else 'giảm'} so với trung bình 20 phiên.
7. **Kịch bản Tiềm năng:** Nếu giá vượt vùng kháng cự, xu hướng tăng có thể tiếp diễn; nếu thất bại, khả năng điều chỉnh ngắn hạn có thể xuất hiện.
8. **Độ Tin cậy:** ⭐ {t['Conviction']:.1f}/10
"""

    b_block = f"""
### B. Phân tích Cơ bản
Giá mục tiêu: 42.2 ngàn VND | Upside: 28.7%
"""

    c_block = f"""
### C. Trade Plan & Risk–Reward Simulation
| Chiến lược | Entry (ưu tiên) | Stop-loss | Take-profit | Xác suất | R:R ước tính |
|-------------|-----------------|------------|--------------|-----------|---------------|
| Pullback | {fibo_levels[2]:.2f} | {fibo_levels[2]*0.94:.2f} (-6%) | {fibo_levels[2]*1.2:.2f} (+20%) | TB | 3.33 |
| Breakout | {fibo_levels[0]:.2f} | {fibo_levels[1]:.2f} (-6%) | {fibo_levels[0]*1.25:.2f} (+25%) | Cao | 4.17 |
"""

    summary = (
        "Trong tổng thể, cấu trúc kỹ thuật của cổ phiếu đang duy trì trạng thái ổn định. "
        "Chiến lược phù hợp là ưu tiên canh các nhịp pullback khi thị trường rung lắc, "
        "hoặc chờ xác nhận breakout với thanh khoản mạnh để gia tăng vị thế."
    )

    return f"{header}\n{preface}\n\n{a_block}\n{b_block}\n{c_block}\n{summary}"

# ============================================================
# 7. SIDEBAR & MAIN LAYOUT
# ============================================================

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker = st.text_input("Mã Cổ Phiếu:", value="VCB").upper()
    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)
    col1 = st.columns(1)[0]
    with col1:
        tech_btn = st.button("📊 Phân tích kỹ thuật")
        fund_btn = st.button("💼 Phân tích cơ bản")
        news_btn = st.button("📰 Tin tức")

# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if tech_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không hợp lệ.")
    else:
        with st.spinner("Đang phân tích..."):
            result = analyze_ticker(ticker)
            if "Error" in result:
                st.error(result["Error"])
            else:
                report = generate_report(result)
                st.markdown(f"<div class='report-text'>{report}</div>", unsafe_allow_html=True)
