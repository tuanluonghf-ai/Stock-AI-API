# ============================================================
# INCEPTION v4.6 FINAL | Strategic Investor Edition
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

st.set_page_config(page_title="INCEPTION v4.6 – Strategic Investor Edition",
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

def _isnan(x) -> bool:
    try: return x is None or (isinstance(x, float) and np.isnan(x))
    except: return True

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
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Target", "Upside"])
    df.rename(columns={"TP (VND)": "Target"}, inplace=True)
    df["Upside"] = pd.to_numeric(df.get("Upside/Downside", 0), errors="coerce")
    return df

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
# 6. FIBONACCI DUAL-FRAME
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

# ============================================================
# 7. CONVICTION SCORE
# ============================================================

def compute_conviction(last: pd.Series) -> float:
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI"] > 55: score += 1
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    return min(10.0, score)
# ============================================================
# 8. TRADE PLAN LOGIC
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
    fib_long = dual_fib["fixed_long"]["levels"]
    fib_hi = dual_fib["auto_short"]["swing_high"]
    fib_lo = dual_fib["auto_short"]["swing_low"]

    # Basic reference levels
    res_zone = fib_short.get("61.8", close * 1.05)
    sup_zone = fib_short.get("38.2", close * 0.95)

    # === Breakout Setup ===
    entry_b = _round_price(res_zone * 1.01)
    stop_b = _round_price(max(ma20 * 0.985, sup_zone * 0.99))
    tp_b = _round_price(entry_b * 1.25)  # default 25% upside
    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    # === Pullback Setup ===
    entry_p = _round_price(sup_zone)
    stop_p = _round_price(entry_p * 0.94)
    tp_p = _round_price(entry_p * 1.20)
    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    # Filter RR < 2.5
    setups = {}
    if rr_b >= 2.5:
        setups["Breakout"] = TradeSetup("Breakout", entry_b, stop_b, tp_b, rr_b, "Cao")
    if rr_p >= 2.5:
        setups["Pullback"] = TradeSetup("Pullback", entry_p, stop_p, tp_p, rr_p, "TB")

    return setups

# ============================================================
# 9. SCENARIO CLASSIFICATION
# ============================================================

def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    rsi, macd_v, sig = last["RSI"], last["MACD"], last["MACDSignal"]

    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20:
            return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200:
            return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200:
            return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# ============================================================
# 10. MAIN ANALYSIS FUNCTION
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
    m, s, h = macd(df["Close"], 12, 26, 9)
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
    } # ============================================================
# 11. GPT-4 TURBO STRATEGIC INSIGHT GENERATION
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    """
    Hàm này gửi dữ liệu kỹ thuật và cơ bản sang GPT-4 Turbo
    để tạo báo cáo phân tích theo chuẩn Strategic Commentary.
    """
    if "Error" in data:
        return f"❌ {data['Error']}"

    # Chuẩn bị dữ liệu
    tick = data["Ticker"]
    last = data["Last"]
    trade_plans = data["TradePlans"]
    fund = data["Fundamental"]
    conviction = data["Conviction"]
    scenario = data["Scenario"]

    close = _fmt_price(last.get("Close"))
    rsi = _fmt_price(last.get("RSI"))
    macd_v = _fmt_price(last.get("MACD"))
    ma20 = _fmt_price(last.get("MA20"))
    ma50 = _fmt_price(last.get("MA50"))
    ma200 = _fmt_price(last.get("MA200"))
    vol = _fmt_int(last.get("Volume"))
    avg_vol = _fmt_int(last.get("Avg20Vol"))

    header = f"**{tick} — {close} | Conviction: {conviction:.1f}/10 | {scenario}**"

    # Trade Plan summary
    tp_text = []
    for k, s in trade_plans.items():
        tp_text.append(f"{k}: Entry {s.entry}, Stop {s.stop}, TP {s.tp}, R:R {s.rr:.2f}")
    tp_summary = " | ".join(tp_text) if tp_text else "Chưa có chiến lược đạt chuẩn R:R ≥ 2.5"

    # Fundamental
    fund_text = (
        f"Khuyến nghị: {fund.get('Recommendation', 'N/A')} | "
        f"Giá mục tiêu: {_fmt_price(fund.get('Target'))} | "
        f"Upside: {_fmt_pct(fund.get('Upside', 0)*100)}"
        if fund else "Không có dữ liệu fundamental"
    )

    # === Prompt ===
    prompt = f"""
    Bạn là chuyên gia phân tích chiến lược của một công ty chứng khoán cao cấp.
    Hãy viết báo cáo ngắn gọn (~700-900 từ) theo cấu trúc chuẩn sau, bằng tiếng Việt, 
    văn phong chuyên nghiệp, gần gũi và có chiều sâu:

    1️⃣ **Executive Summary (3–4 câu)**
    - Nhận định tổng thể xu hướng hiện tại của {tick}, dòng tiền, động lượng.
    - Tác động lên chiến lược hành động của nhà đầu tư trung–dài hạn.

    2️⃣ **A. Phân tích Kỹ thuật**
    Bao gồm:
    - MA Trend (MA20, MA50, MA200)
    - RSI Analysis (động lượng, vùng quá mua/bán)
    - MACD Analysis (tín hiệu xu hướng)
    - RSI + MACD Bias
    - Fibonacci (2 khung 60–90 & 250 ngày): hỗ trợ – kháng cự – vùng chiến lược
    - Volume & Price Action
    - 12-Scenario Classification
    - Master Integration + Conviction Score

    3️⃣ **B. Fundamental Analysis Summary**
    - Dữ liệu: {fund_text}

    4️⃣ **C. Trade Plan**
    - {tp_summary}

    5️⃣ **D. Risk–Reward Simulation**
    - Diễn giải R:R, xác suất, và chiến lược phù hợp khẩu vị lợi nhuận 15–100%, rủi ro 5–8%.

    Ngữ điệu cần tự nhiên, chuyên nghiệp, kiểu như chuyên gia phân tích trình bày trước khách hàng tổ chức.
    Phải đảm bảo:
    - Không tự bịa số liệu.
    - Chỉ phân tích dựa trên các giá trị thực sau:
      MA20={ma20}, MA50={ma50}, MA200={ma200}, RSI={rsi}, MACD={macd_v},
      Volume={vol}, AvgVol={avg_vol}, Conviction={conviction:.1f}.
    """

    # ============================================================
    # ẨN API KEY KHI KHỞI TẠO CLIENT
    # ============================================================
    try:
        client = OpenAI()  # Key lấy tự động từ môi trường
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư chiến lược."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1600
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"⚠️ Lỗi khi gọi GPT: {e}"

    return f"{header}\n\n{content}" # ============================================================
# 12. STREAMLIT UI & APP LAYOUT
# ============================================================

# --- Header section ---
st.markdown("<h1 style='color:#A855F7;'>🟣 INCEPTION v4.6 — Strategic Investor Edition</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#9CA3AF;'>Công cụ phân tích chiến lược cho nhà đầu tư trung–dài hạn (Lợi nhuận 15–100%, Rủi ro 5–8%).</p>", unsafe_allow_html=True)
st.divider()

# --- Sidebar controls ---
with st.sidebar:
    st.markdown("### 🔐 Đăng nhập người dùng")
    user_key = st.text_input("Nhập Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("🚀 Phân tích ngay", type="primary")

# --- Layout containers ---
col_main, = st.columns([1])  # Chỉ hiển thị phần Report (ẩn Chart column tạm thời)

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
                st.markdown(report)
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
    Phiên bản 4.6 – Strategic Investor Edition | Engine GPT-4 Turbo
    </p>
    """,
    unsafe_allow_html=True
) # ============================================================
# 15. FINAL TOUCHES – MARKDOWN OPTIMIZATION & SAFETY CHECKS
# ============================================================

def render_markdown_safe(text: str):
    """Đảm bảo hiển thị báo cáo Markdown có xuống dòng và format rõ ràng."""
    text = text.replace("\n\n", "<br><br>")
    st.markdown(f"<div style='white-space:pre-wrap; color:#E5E7EB;'>{text}</div>", unsafe_allow_html=True)

# Kiểm tra file dữ liệu
missing_files = []
for f in [PRICE_VOL_PATH, HSC_TARGET_PATH, TICKER_NAME_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    st.warning(f"⚠️ Thiếu file dữ liệu: {', '.join(missing_files)}. Hãy kiểm tra lại thư mục trước khi chạy.")
else:
    st.info("✅ Tất cả file dữ liệu đã sẵn sàng. Bạn có thể tiến hành phân tích.")

# ============================================================
# 16. RUNNING GUIDE
# ============================================================

st.divider()
st.markdown(
    """
    <div style='color:#9CA3AF; font-size:14px; line-height:1.6;'>
    <strong>📘 Hướng dẫn sử dụng:</strong><br>
    1️⃣ Mở Terminal hoặc Command Prompt.<br>
    2️⃣ Di chuyển đến thư mục chứa file <code>app.py</code> và các file Excel dữ liệu.<br>
    3️⃣ Gõ lệnh: <code>streamlit run app.py</code><br>
    4️⃣ Nhập Mã VIP và Mã Cổ Phiếu (VD: HPG, FPT, VNM).<br>
    5️⃣ Hệ thống sẽ tự động tạo báo cáo phân tích chiến lược.<br><br>
    <em>Lưu ý:</em> INCEPTION v4.6 dành cho nhà đầu tư chiến lược (Target 15–100%, Risk 5–8%).<br>
    Không sử dụng cho mục đích giao dịch ngắn hạn hoặc lướt sóng trong ngày.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 17. SAFETY EXIT (FOR EMPTY RUNS)
# ============================================================

if not run_btn:
    st.markdown(
        """
        <br><br>
        <div style='text-align:center; color:#A855F7;'>
        🔍 <strong>Nhập mã cổ phiếu và nhấn “Phân tích ngay” để bắt đầu.</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
