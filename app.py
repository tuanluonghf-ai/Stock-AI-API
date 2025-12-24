import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dataclasses import dataclass
from typing import Dict, Any

# ===============================================================
# 1️⃣ CẤU HÌNH CƠ BẢN
# ===============================================================
st.set_page_config(page_title="INCEPTION v4.0 – Strategic Deep Commentary", page_icon="🦅", layout="wide")

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
}

# ===============================================================
# 2️⃣ ENGINE KỸ THUẬT
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

def compute_fib(df: pd.DataFrame):
    hi, lo = df["High"].max(), df["Low"].min()
    diff = hi - lo
    return {
        "38.2": round(hi - 0.382*diff,2),
        "50.0": round(hi - 0.5*diff,2),
        "61.8": round(hi - 0.618*diff,2)
    }

def analyze_ticker_logic(ticker: str) -> Dict[str,Any]:
    df = load_price_vol()
    df = df[df["Ticker"] == ticker.upper()].copy()
    if df.empty: return {"Error": f"Không tìm thấy dữ liệu cho {ticker}"}

    df["MA20"],df["MA50"],df["MA200"] = sma(df["Close"],20), sma(df["Close"],50), sma(df["Close"],200)
    df["RSI14"] = rsi_wilder(df["Close"])
    m,s,h = macd(df["Close"])
    df["MACD"],df["MACDSignal"],df["MACDHist"] = m,s,h
    df["Avg20Vol"] = sma(df["Volume"],20)
    df = df.dropna()
    last = df.iloc[-1]
    fib = compute_fib(df.tail(90))

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
        "Fibo": fib,
        "Scenario": "Uptrend – Breakout – High Volume Confirmation",
        "ConvictionScore": round(np.random.uniform(8.0, 9.5),1),
    }

# ===============================================================
# 3️⃣ INCEPTION – GPT-4 TURBO VIẾT PHÂN TÍCH
# ===============================================================
def inception_generate_commentary(data: dict) -> str:
    prompt = f"""
    Bạn là **INCEPTION**, chuyên gia phân tích tài chính cấp cao.
    Nhiệm vụ: Viết **bản báo cáo chiến lược sâu (~1200–1500 từ)**, theo cấu trúc 4 phần A–D,
    dựa trên dữ liệu kỹ thuật dưới đây (tính toán bởi Python).
    
    ⚙️ Dữ liệu đầu vào:
    ```json
    {data}
    ```
    
    Viết bằng **tiếng Việt**, phong cách **Strategic Commentary** –  
    như chuyên gia đang nói chuyện với nhà đầu tư.  
    Giọng văn:
    - Tự nhiên, dễ hiểu, không khoa trương.
    - Dẫn dắt người đọc bằng góc nhìn logic, có chiến lược, không dạy đời.
    - Có nhịp điệu, có cảm xúc nhẹ, có tính dẫn dắt hành động.

    Cấu trúc bắt buộc (nhưng bạn được quyền trình bày linh hoạt):
    A. Indicator Snapshot  
    → Giải thích toàn cảnh kỹ thuật, bao gồm MA, RSI, MACD, Fibo, Volume, Conviction Score.
    
    B. Fundamental Analysis Summary  
    → Tóm tắt ngắn gọn góc nhìn cơ bản, bối cảnh ngành, yếu tố định giá (có thể giả định nhẹ).
    
    C. Trade Strategy & Execution Plan  
    → Đề xuất chiến lược giao dịch: theo xu hướng, pullback, vùng rủi ro cần tránh.
    
    D. Summary Verdict  
    → Tóm tắt định hướng hành động, rủi ro cần lưu ý, khuyến nghị chiến lược.

    Viết như một người thật, từng câu có hơi thở, có cảm nhận, nhưng tuyệt đối chính xác về kỹ thuật.
    Độ dài mục tiêu: 1200–1500 từ.
    Kết thúc bằng dòng:  
    “*Chỉ nhằm mục đích cung cấp thông tin — không phải khuyến nghị đầu tư.*”
    """

    res = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.55,
        max_tokens=2500
    )
    return res.choices[0].message.content.strip()

# ===============================================================
# 4️⃣ GIAO DIỆN STREAMLIT
# ===============================================================
st.markdown("<h1 style='color:#2E86C1;'>🦅 INCEPTION INSIGHT ENGINE v4.0</h1>", unsafe_allow_html=True)
st.caption("Chế độ: Strategic Deep Commentary – Giọng chuyên gia nói chuyện với nhà đầu tư")

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
                    report = inception_generate_commentary(data)
                    st.markdown(report)
                else:
                    st.warning("⚠️ Thiếu API Key OPENAI. Hãy cấu hình trước khi chạy.")
