# ============================================================
# INCEPTION v4.8 FINAL | Analyst Precision Edition
# app.py — Streamlit + GPT-4 Turbo
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
from typing import Dict, Any, Tuple, List, Optional

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(page_title="INCEPTION v4.8 — Analyst Precision Edition",
                   layout="wide",
                   page_icon="🟣")

st.markdown("""
<style>
body {
    background-color: #0B0E11;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}
strong {
    color: #FFFFFF;
    font-weight: 700;
}
h1, h2, h3 {
    color: #FFFFFF;
}
hr {border: 1px solid #333;}
table, th, td {
    border: 1px solid #555;
    padding: 6px;
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

def _safe_float(x, default=np.nan) -> float:
    try: return float(x)
    except: return default

def _round_price(x: float, ndigits: int = 2) -> float:
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

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
def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Target", "Upside"])
    df.rename(columns={"TP (VND)": "Target"}, inplace=True)
    df["Upside"] = pd.to_numeric(df.get("Upside/Downside", 0), errors="coerce")
    return df

# ============================================================
# 5. INDICATORS & CALCULATIONS
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
# 6. FIBONACCI (DUAL FRAME)
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

def compute_dual_fibonacci(df: pd.DataFrame) -> Dict[str, Any]:
    L_short = min(90, len(df))
    L_long = min(250, len(df))
    win_short = df.tail(L_short)
    win_long = df.tail(L_long)
    s_hi, s_lo = win_short["High"].max(), win_short["Low"].min()
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()
    return {
        "auto_short": {"frame": f"AUTO_{L_short}D", "swing_high": s_hi, "swing_low": s_lo, "levels": _fib_levels(s_lo, s_hi)},
        "fixed_long": {"frame": f"FIXED_{L_long}D", "swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)}
    }

# ============================================================
# 7. CONVICTION SCORE & SCENARIO CLASSIFICATION
# ============================================================

def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)

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

# ============================================================
# 8. TRADE PLAN LOGIC (PYTHON-BASED)
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
    res_zone = fib_short.get("61.8", close * 1.05)
    sup_zone = fib_short.get("38.2", close * 0.95)

    entry_b = _round_price(res_zone * 1.01)
    stop_b = _round_price(max(ma20 * 0.985, sup_zone * 0.99))
    tp_b = _round_price(entry_b * 1.25)
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    entry_p = _round_price(sup_zone)
    stop_p = _round_price(entry_p * 0.94)
    tp_p = _round_price(entry_p * 1.20)
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    setups = {
        "Breakout": TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao"),
        "Pullback": TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, "Trung bình")
    }

    return setups
# ============================================================
# 9. MAIN ANALYSIS PIPELINE
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
    m, s, h = macd(df["Close"])
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h

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
        "DualFibo": dual_fib,
        "TradePlans": trade_plans,
        "Fundamental": fund_row
    }

# ============================================================
# 10. GPT-4 TURBO INSIGHT GENERATION
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data:
        return f"❌ {data['Error']}"

    tick = data["Ticker"]
    last = data["Last"]
    trade_plans = data["TradePlans"]
    fund = data["Fundamental"]
    conviction = data["Conviction"]
    scenario = data["Scenario"]
    dual_fib = data["DualFibo"]

    close = _fmt_price(last.get("Close"))
    ma20, ma50, ma200 = _fmt_price(last.get("MA20")), _fmt_price(last.get("MA50")), _fmt_price(last.get("MA200"))
    rsi, macd_v = _fmt_price(last.get("RSI")), _fmt_price(last.get("MACD"))
    vol, avg_vol = _fmt_int(last.get("Volume")), _fmt_int(last.get("Avg20Vol"))

    # Trade Plan summary for GPT context
    tp_summary = "\n".join([
        f"- {v.name}: Entry {v.entry}, Stop {v.stop}, TP {v.tp}, R:R {v.rr:.2f}, Prob {v.probability}"
        for v in trade_plans.values()
    ]) if trade_plans else "Chưa có setup đủ điều kiện R:R."

    fund_text = (
        f"Target: {_fmt_price(fund.get('Target'))}, Upside: {_fmt_pct(fund.get('Upside', 0)*100)}"
        if fund else "Không có dữ liệu định giá cơ bản"
    )

    # GPT PROMPT (frame chuẩn, Python-driven)
    prompt = f"""
    Bạn là chuyên gia phân tích của INCEPTION Research.
    Hãy viết báo cáo phân tích (~700–900 từ) theo format sau, hoàn toàn bằng tiếng Việt.

    ⚠️ QUAN TRỌNG:
    - KHÔNG được bịa ra số liệu.
    - Chỉ sử dụng dữ liệu do Python cung cấp.
    - Phân tích hướng đến nhà đầu tư chiến lược (mục tiêu 15–100%, risk 5–8%).
    - Gọi người đọc là “bạn” thay vì “nhà đầu tư”.

    =========================================================
    HEADER:
    {tick} — {close} | ⭐ {conviction:.1f}/10 | {scenario}
    =========================================================

    1️⃣ Executive Summary
    - Tổng quan xu hướng hiện tại, dòng tiền, tâm lý, và hành động nên cân nhắc.

    2️⃣ A. Phân tích Kỹ thuật
    Trình bày theo 8 phần sau:
    - Snapshot dữ liệu:
      • Close: {close}
      • Volume: {vol} | Avg20 Vol: {avg_vol}
      • MA20 / MA50 / MA200: {ma20} / {ma50} / {ma200}
      • RSI (14): {rsi}
      • MACD: {macd_v}
      • FIBO (Python xác định 2 khung): hỗ trợ – kháng cự

    - MA Trend
    - RSI Analysis
    - MACD Signal
    - RSI + MACD Bias Matrix → mô tả tổ hợp và chiến lược phù hợp
    - Fibonacci Dual-Frame → nêu rõ vùng hỗ trợ, kháng cự, so sánh với giá hiện tại
    - Volume & Price Action → thêm nhận định về mẫu hình nến (Price Action)
    - Kịch bản tiềm năng → dựa trên {scenario}, mô tả chiến lược hành động phù hợp
    - Độ tin cậy → Conviction {conviction:.1f}/10, xu hướng, đề xuất hành động cụ thể.

    3️⃣ B. Fundamental Summary
    - {fund_text}

    4️⃣ C. Trade Plan & Risk–Reward Simulation
    - Dữ liệu Python:
    {tp_summary}

    Trình bày bảng 6 cột:
    Chiến lược | Entry (ưu tiên) | Stop-loss | Take-profit | Xác suất | R:R ước tính
    Dòng 1: Pullback
    Dòng 2: Breakout

    Sau bảng:
    - Tổng hợp: R:R trung bình có trọng số.
    - Kịch bản ưu tiên và vùng giá phù hợp trong điều kiện thị trường cụ thể.
    """

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI – chuyên gia phân tích chiến lược đầu tư trung–dài hạn."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1800
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi gọi GPT: {e}"

    header = f"### {tick} — {close}  ⭐ {conviction:.1f}/10<br><small>{scenario}</small>"
    return f"{header}\n\n{content}"

# ============================================================
# 11. STREAMLIT UI
# ============================================================

st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.8 — Analyst Precision Edition</h1>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("🚀 Phân tích ngay", type="primary")

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])  # Sidebar / Report / Chart Placeholder

# ============================================================
# 12. MAIN EXECUTION
# ============================================================

with col2:
    if run_btn:
        if user_key not in VALID_KEYS:
            st.error("❌ Mã VIP không đúng. Vui lòng nhập lại.")
        else:
            with st.spinner(f"Đang xử lý dữ liệu cho {ticker_input}..."):
                result = analyze_ticker(ticker_input)
                report = generate_insight_report(result)
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(report, unsafe_allow_html=True)
    else:
        st.markdown(
            "<br><div style='text-align:center; color:#A855F7;'>🔍 <strong>Nhập mã cổ phiếu và nhấn “Phân tích ngay” để bắt đầu.</strong></div>",
            unsafe_allow_html=True
        )

# ============================================================
# 13. FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <p style='text-align:center; color:#888; font-size:13px;'>
    © 2025 INCEPTION Research Framework<br>
    Phiên bản 4.8 — Analyst Precision Edition | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
)
