# ============================================================
# INCEPTION v5.0.1 | FRAME-ALIGNED Natural Output Edition
# Author: INCEPTION AI Research Framework
# Purpose: Strategic Investment Assistant (Professional Insight FRAME)
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

st.set_page_config(page_title="INCEPTION v5.0.1 – Strategic FRAME Edition", layout="wide", page_icon="🟣")

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
# 2. FILE PATHS & ACCESS CONTROL
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
# 3. UTILITIES
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

def _round(x, d=2):
    if np.isnan(x): return np.nan
    return round(float(x), d)

# ============================================================
# 4. DATA LOADERS
# ============================================================

@st.cache_data
def load_price_vol(path=PRICE_VOL_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip().title() for c in df.columns]
        df.rename(columns={"Ngay": "Date", "Ma": "Ticker", "Vol": "Volume"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]).sort_values(["Ticker", "Date"])
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

def sma(series, w): return series.rolling(w).mean()
def ema(series, s): return series.ewm(span=s, adjust=False).mean()

def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

# ============================================================
# 6. FIBONACCI
# ============================================================

def _fib_levels(low, high):
    rng = high - low
    if rng <= 0: return {}
    return {
        "38.2": high - 0.382*rng,
        "50.0": high - 0.5*rng,
        "61.8": high - 0.618*rng
    }

def compute_dual_fib(df):
    short, long = df.tail(90), df.tail(250)
    s_hi, s_lo = short["High"].max(), short["Low"].min()
    l_hi, l_lo = long["High"].max(), long["Low"].min()
    return {"short": _fib_levels(s_lo, s_hi), "long": _fib_levels(l_lo, l_hi)}

# ============================================================
# 7. TRADE PLAN / CONVICTION
# ============================================================

@dataclass
class TradeSetup:
    name: str; entry: float; stop: float; tp: float; rr: float
    prob: str; stop_pct: float; tp_pct: float

def _rr(e, s, t):
    if e<=0 or s<=0 or t<=0: return np.nan
    r, w = abs(e-s), abs(t-e)
    return round(w/r,2) if r>0 else np.nan

def compute_conviction(last):
    score=5
    if last["Close"]>last["MA200"]: score+=2
    if last["RSI"]>55: score+=1
    if last["MACD"]>last["MACDSignal"]: score+=1
    if last["Volume"]>last["Avg20Vol"]: score+=1
    return min(10,score)

def build_trade_plan(df,fib):
    last=df.iloc[-1]; c=last["Close"]
    res=fib["short"].get("61.8",c*1.05); sup=fib["short"].get("38.2",c*0.95)
    entry_b=max(res*1.01,c*1.02); stop_b=entry_b*0.94; tp_b=entry_b*1.25
    entry_p=min(sup,c*0.98); stop_p=entry_p*0.94; tp_p=entry_p*1.20
    return {
        "Breakout":TradeSetup("Breakout",_round(entry_b),_round(stop_b),_round(tp_b),
                              _rr(entry_b,stop_b,tp_b),"Cao",
                              round((stop_b-entry_b)/entry_b*100,1),
                              round((tp_b-entry_b)/entry_b*100,1)),
        "Pullback":TradeSetup("Pullback",_round(entry_p),_round(stop_p),_round(tp_p),
                              _rr(entry_p,stop_p,tp_p),"TB",
                              round((stop_p-entry_p)/entry_p*100,1),
                              round((tp_p-entry_p)/entry_p*100,1))
    }

def classify_scenario(last):
    c,ma20,ma50,ma200=last["Close"],last["MA20"],last["MA50"],last["MA200"]
    if ma20>ma50>ma200 and c>ma20: return "Uptrend – Breakout Confirmation"
    elif c>ma200 and ma20>ma200: return "Uptrend – Pullback Phase"
    elif c<ma200 and ma50<ma200: return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# ============================================================
# 8. MAIN ANALYSIS
# ============================================================

def analyze_ticker(t):
    df_all=load_price_vol()
    if df_all.empty: return {"Error":"Không có dữ liệu"}
    df=df_all[df_all["Ticker"].str.upper()==t.upper()]
    if df.empty: return {"Error":f"Không tìm thấy mã {t}"}

    df["MA20"],df["MA50"],df["MA200"]=sma(df["Close"],20),sma(df["Close"],50),sma(df["Close"],200)
    df["Avg20Vol"],df["RSI"]=sma(df["Volume"],20),rsi_wilder(df["Close"])
    m,s,_=macd(df["Close"]); df["MACD"],df["MACDSignal"]=m,s

    fib=compute_dual_fib(df); last=df.iloc[-1]
    conviction=compute_conviction(last)
    trades=build_trade_plan(df,fib)
    scenario=classify_scenario(last)
    fund=load_hsc_targets(); fund=fund[fund["Ticker"].str.upper()==t.upper()]
    fund_row=fund.iloc[0].to_dict() if not fund.empty else {}

    return {"Ticker":t,"Last":last.to_dict(),"Fib":fib,"TradePlans":trades,
            "Conviction":conviction,"Scenario":scenario,"Fundamental":fund_row}

# ============================================================
# 9. GPT REPORT
# ============================================================

def generate_report(data):
    if "Error" in data: return data["Error"]
    t,last,fib,trades=data["Ticker"],data["Last"],data["Fib"],data["TradePlans"]
    fund,conv,sc=data["Fundamental"],data["Conviction"],data["Scenario"]

    c=_fmt_price(last["Close"]); ma20=_fmt_price(last["MA20"]); ma50=_fmt_price(last["MA50"]); ma200=_fmt_price(last["MA200"])
    rsi=_fmt_price(last["RSI"]); macd=_fmt_price(last["MACD"])
    v=_fmt_int(last["Volume"]); av=_fmt_int(last["Avg20Vol"])
    fund_text=f"Target: {_fmt_price(fund.get('Target'))}, Upside: {_fmt_pct(fund.get('Upside',0)*100)}" if fund else "Không có dữ liệu"
    trade_table="| Chiến lược | Entry | Stop-loss | Take-profit | Xác suất | R:R |\n|-------------|--------|-----------|--------------|-----------|-------|\n"
    for s in trades.values(): trade_table+=f"| {s.name} | {s.entry} | {s.stop} ({s.stop_pct}%) | {s.tp} (+{s.tp_pct}%) | {s.prob} | {s.rr} |\n"

    prompt=f"""
Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược.
Hãy tạo báo cáo phân tích chuyên sâu (~800-900 từ) theo format:
A. Phân tích Kỹ thuật
1. MA Trend
2. RSI
3. MACD
4. RSI + MACD Bias Matrix
5. Fibonacci Levels
6. Volume & Price Action
7. Kịch bản tiềm năng
8. Độ tin cậy (⭐ {conv:.1f}/10 → Xu hướng nghiêng ...)

B. Fundamental Summary
- {fund_text}

C. Trade Plan & Risk–Reward Simulation
{trade_table}

Phải đảm bảo đủ 8 mục trong phần A.
GPT được phép nhận định xu hướng, vùng hỗ trợ/kháng cự, và đề xuất chiến lược.
Dữ liệu: Close={c}, MA20={ma20}, MA50={ma50}, MA200={ma200}, RSI={rsi}, MACD={macd}, Volume={v}, AvgVol={av}.
"""

    try:
        client=OpenAI()
        r=client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role":"system","content":"Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược."},
                      {"role":"user","content":prompt}],
            temperature=0.4,max_tokens=1800)
        text=r.choices[0].message.content
    except Exception as e: text=f"Lỗi GPT: {e}"

    header=f"### {t} — {c} ⭐ {conv:.1f}/10<br><small>{sc}</small>"
    return f"{header}\n\n{text}"

# ============================================================
# 10. STREAMLIT UI
# ============================================================

st.markdown("<h1>INCEPTION v5.0.1 — FRAME-ALIGNED Natural Output Edition</h1>", unsafe_allow_html=True)
with st.sidebar:
    key=st.text_input("Mã VIP:",type="password")
    t=st.text_input("Mã cổ phiếu:",value="HPG").upper()
    run=st.button("Phân tích")

col1,col2,col3=st.columns([0.2,0.6,0.2])
with col2:
    if run:
        if key not in VALID_KEYS:
            st.error("Sai mã VIP.")
        else:
            with st.spinner(f"Đang xử lý {t}..."):
                d=analyze_ticker(t)
                rep=generate_report(d)
                st.markdown(rep,unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;'>Nhập mã cổ phiếu và nhấn “Phân tích”.</div>", unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align:center;font-size:13px;'>© 2025 INCEPTION Research Framework | Version 5.0.1</p>", unsafe_allow_html=True)
