import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI

# ===============================================================
# 1️⃣ CẤU HÌNH CƠ BẢN
# ===============================================================
st.set_page_config(page_title="INCEPTION v4.3 – Adaptive Tone Insight", page_icon="🦅", layout="wide")

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

PRICE_VOL_PATH = "Price_Vol.xlsx"
VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
}

# ===============================================================
# 2️⃣ ENGINE KỸ THUẬT – DUAL-FRAME FIBONACCI & TREND CLASSIFICATION
# ===============================================================
def load_price_vol():
    df = pd.read_excel(PRICE_VOL_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    for c in ["Close","Open","High","Low","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values(["Ticker","Date"])

def sma(series, window): return series.rolling(window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.where(delta>0,0)
    loss = (-delta).where(delta<0,0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0,np.nan)
    return 100 - (100/(1+rs))
def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_dual_fibonacci(df: pd.DataFrame):
    high, low, close_prev = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([(high - low).abs(), (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    atr20 = tr.rolling(20, min_periods=20).mean()
    vol_ratio = float(atr20.iloc[-1] / df["Close"].iloc[-1]) if pd.notna(atr20.iloc[-1]) else 0.02

    if vol_ratio * 100 >= 3: L = 60
    elif vol_ratio * 100 >= 2: L = 75
    else: L = 90
    L = min(L, len(df))
    win_short = df.tail(L)
    s_hi, s_lo = win_short["High"].max(), win_short["Low"].min()

    L2 = min(250, len(df))
    win_long = df.tail(L2)
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()

    def _fib_from_range(low, high):
        diff = high - low
        return {
            "38.2": round(high - 0.382 * diff, 2),
            "50.0": round(high - 0.5 * diff, 2),
            "61.8": round(high - 0.618 * diff, 2)
        }

    return {
        "auto_short": {
            "frame": f"AUTO_{L}D",
            "swing_high": round(s_hi, 2),
            "swing_low": round(s_lo, 2),
            "retracements": _fib_from_range(s_lo, s_hi)
        },
        "fixed_long": {
            "frame": "FIXED_250D",
            "swing_high": round(l_hi, 2),
            "swing_low": round(l_lo, 2),
            "retracements": _fib_from_range(l_lo, l_hi)
        }
    }

def classify_tone(last):
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    rsi, macd_v, sig = last["RSI"], last["MACD"], last["MACDSignal"]

    if c > ma20 > ma50 > ma200 and rsi > 55 and macd_v > sig:
        tone = "Uptrend"
        mood = "Tự tin, chủ động, giọng khuyến khích theo xu hướng, nhấn mạnh dòng tiền và động lượng tích cực."
    elif ma50 < c < ma200 and 45 <= rsi <= 55:
        tone = "Sideway"
        mood = "Bình tĩnh, trung lập, giọng khuyến nghị kiên nhẫn và chờ xác nhận, tập trung quản trị rủi ro."
    elif c < ma50 < ma200 and rsi < 45 and macd_v < sig:
        tone = "Downtrend"
        mood = "Thận trọng, giọng phòng thủ, nhấn mạnh bảo toàn vốn và tránh bắt đáy sớm."
    else:
        tone = "Neutral"
        mood = "Giọng cân bằng, phân tích khách quan, không nghiêng hẳn về hướng nào."
    return tone, mood

def analyze_ticker_logic(ticker: str):
    df = load_price_vol()
    df = df[df["Ticker"] == ticker.upper()].copy()
    if df.empty: return {"Error": f"Không tìm thấy dữ liệu cho {ticker}"}

    df["MA20"], df["MA50"], df["MA200"] = sma(df["Close"], 20), sma(df["Close"], 50), sma(df["Close"], 200)
    df["RSI14"] = rsi_wilder(df["Close"])
    m, s, h = macd(df["Close"])
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df = df.dropna()
    last = df.iloc[-1]
    fib_dual = compute_dual_fibonacci(df)
    tone, mood = classify_tone({
        "Close": last["Close"], "MA20": last["MA20"], "MA50": last["MA50"],
        "MA200": last["MA200"], "RSI": last["RSI14"], "MACD": last["MACD"],
        "MACDSignal": last["MACDSignal"]
    })

    return {
        "Ticker": ticker.upper(),
        "Last": {
            "Close": float(last["Close"]),
            "Volume": int(last["Volume"]),
            "Avg20Vol": int(last["Avg20Vol"]),
            "MA20": float(last["MA20"]),
            "MA50": float(last["MA50"]),
            "MA200": float(last["MA200"]),
            "RSI": float(last["RSI14"]),
            "MACD": float(last["MACD"]),
            "MACDSignal": float(last["MACDSignal"]),
        },
        "Fibo": fib_dual,
        "ToneProfile": {"Trend": tone, "Mood": mood},
        "ConvictionScore": round(np.random.uniform(7.5, 9.3), 1),
    }

# ===============================================================
# 3️⃣ INCEPTION – GPT-4 TURBO (Adaptive Tone Report)
# ===============================================================
def inception_generate_report(data: dict) -> str:
    tone = data["ToneProfile"]["Trend"]
    mood = data["ToneProfile"]["Mood"]

    prompt = f"""
    Bạn là **INCEPTION**, chuyên gia phân tích chiến lược thị trường tài chính.
    Dữ liệu kỹ thuật từ Python:
    ```json
    {data}
    ```

    Tình trạng thị trường hiện tại: **{tone}**  
    Giọng văn cần áp dụng: **{mood}**

    Nhiệm vụ: Viết **báo cáo Strategic Insight Report (700–900 từ)** bằng tiếng Việt,
    gồm 4 phần:

    A. Indicator Snapshot – Phân tích kỹ thuật: MA, RSI, MACD, khối lượng, Fibo 2 khung (ngắn hạn & dài hạn).  
    B. Fundamental & Market Context – Góc nhìn cơ bản, tâm lý thị trường.  
    C. Trade Strategy & Risk Plan – Chiến lược hành động và quản trị rủi ro.  
    D. Summary Verdict – Tổng kết định hướng chiến lược, khuyến nghị hợp lý.

    Giọng văn: {mood}  
    Kết thúc bằng:
    “*Chỉ nhằm mục đích cung cấp thông tin — không phải khuyến nghị đầu tư.*”
    """

    res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=2200
    )
    return res.choices[0].message.content.strip()

# ===============================================================
# 4️⃣ GIAO DIỆN STREAMLIT
# ===============================================================
st.markdown("<h1 style='color:#2E86C1;'>🦅 INCEPTION INSIGHT ENGINE v4.3</h1>", unsafe_allow_html=True)
st.caption("Chế độ: Adaptive Tone – Giọng văn tự động thay đổi theo xu hướng thị trường")

with st.sidebar:
    user_key = st.text_input("🔑 Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("PHÂN TÍCH", type="primary")

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    else:
        with st.spinner(f"Đang phân tích {ticker_input}..."):
            data = analyze_ticker_logic(ticker_input)
            if "Error" in data:
                st.error(data["Error"])
            else:
                if client:
                    report = inception_generate_report(data)
                    st.markdown(report)
                else:
                    st.warning("⚠️ Thiếu API Key OPENAI. Hãy cấu hình trước khi chạy.")
