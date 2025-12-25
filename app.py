# ============================================================
# INCEPTION v5.2 | FRAME-LOCK Final Edition
# Author: INCEPTION AI Research Framework
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import unicodedata
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

def _norm_col(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return "".join(s.lower().split())

def load_targets(path=HSC_TARGET_PATH):
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        norm_map = {c: _norm_col(c) for c in df.columns}

        ticker_col = None
        for c, n in norm_map.items():
            if n in ("ticker", "ma", "symbol", "code"):
                ticker_col = c
                break
        if ticker_col is None:
            for c, n in norm_map.items():
                if "ticker" in n or "symbol" in n or n == "ma":
                    ticker_col = c
                    break

        target_col = None
        for c, n in norm_map.items():
            if n in ("target", "targetprice", "giacmuctieu", "giamuctieu", "muctieu", "giatarget"):
                target_col = c
                break
        if target_col is None:
            for c, n in norm_map.items():
                if "target" in n or "muctieu" in n:
                    target_col = c
                    break

        if ticker_col is None or target_col is None:
            return pd.DataFrame(columns=["Ticker", "Target"])

        out = df[[ticker_col, target_col]].copy()
        out.rename(columns={ticker_col: "Ticker", target_col: "Target"}, inplace=True)
        out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
        out["Target"] = pd.to_numeric(out["Target"], errors="coerce")
        out = out.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"], keep="last")
        return out
    except:
        return pd.DataFrame(columns=["Ticker", "Target"])

def get_target_price(ticker: str) -> float:
    tdf = load_targets()
    if tdf.empty:
        return np.nan
    row = tdf[tdf["Ticker"] == ticker.upper()]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["Target"]) if pd.notna(row.iloc[0]["Target"]) else np.nan

def compute_market_context(df_all: pd.DataFrame, stock_change_pct: float) -> Dict[str, Any]:
    def _get_index_metrics(symbol: str) -> Dict[str, Any]:
        dfi = df_all[df_all["Ticker"].str.upper() == symbol.upper()].copy()
        if dfi.empty or len(dfi) < 2 or "Close" not in dfi.columns:
            return {"Ticker": symbol.upper(), "Available": False}

        dfi["MA20"] = sma(dfi["Close"], 20)
        dfi["MA50"] = sma(dfi["Close"], 50)

        last = dfi.iloc[-1]
        prev = dfi.iloc[-2]

        close = float(last["Close"])
        prev_close = float(prev["Close"])
        chg = (close - prev_close) / prev_close * 100 if prev_close != 0 else np.nan

        ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan
        ma50 = float(last["MA50"]) if pd.notna(last["MA50"]) else np.nan

        return {
            "Ticker": symbol.upper(),
            "Available": True,
            "Close": close,
            "ChangePct": chg,
            "MA20": ma20,
            "MA50": ma50,
            "AboveMA20": (close > ma20) if pd.notna(ma20) else np.nan,
            "AboveMA50": (close > ma50) if pd.notna(ma50) else np.nan,
        }

    vnindex = _get_index_metrics("VNINDEX")
    vn30 = _get_index_metrics("VN30")

    rel = {}
    for idx in [vnindex, vn30]:
        if idx.get("Available") and pd.notna(idx.get("ChangePct")) and pd.notna(stock_change_pct):
            rel[idx["Ticker"]] = float(stock_change_pct - float(idx["ChangePct"]))
        else:
            rel[idx["Ticker"]] = np.nan

    return {
        "VNINDEX": vnindex,
        "VN30": vn30,
        "RelPerfPctPoint": rel
    }

def format_market_brief(market: Dict[str, Any]) -> Dict[str, str]:
    def _fmt_idx(idx: Dict[str, Any]) -> str:
        if not idx.get("Available"):
            return f"{idx.get('Ticker','N/A')}: N/A"
        chg = idx.get("ChangePct")
        close = idx.get("Close")
        above50 = idx.get("AboveMA50")
        trend = "trên MA50" if above50 is True else ("dưới MA50" if above50 is False else "MA50 N/A")
        return f"{idx['Ticker']} {close:.2f} ({chg:+.2f}%), {trend}"

    vnindex_s = _fmt_idx(market.get("VNINDEX", {"Ticker": "VNINDEX", "Available": False}))
    vn30_s = _fmt_idx(market.get("VN30", {"Ticker": "VN30", "Available": False}))

    rel = market.get("RelPerfPctPoint", {})
    rel_vni = rel.get("VNINDEX", np.nan)
    rel_vn30 = rel.get("VN30", np.nan)

    rel_s = []
    if pd.notna(rel_vni):
        rel_s.append(f"So với VNINDEX: {rel_vni:+.2f} điểm %")
    else:
        rel_s.append("So với VNINDEX: N/A")

    if pd.notna(rel_vn30):
        rel_s.append(f"So với VN30: {rel_vn30:+.2f} điểm %")
    else:
        rel_s.append("So với VN30: N/A")

    return {
        "VNINDEX_LINE": vnindex_s,
        "VN30_LINE": vn30_s,
        "REL_LINE": " | ".join(rel_s)
    }

def gpt_preface_expert(t: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            "Thị trường hiện đang dao động trong vùng cân bằng sau nhịp hồi. "
            f"{t['Ticker']} đang thể hiện {('tốt hơn' if t['Change']>0 else 'kém hơn')} thị trường chung, "
            "phản ánh sự chọn lọc dòng tiền giữa các nhóm cổ phiếu. "
            "→ Giai đoạn hiện tại phù hợp với việc canh các nhịp điều chỉnh ngắn để gia tăng vị thế hơn là mua đuổi."
        )

    market = t.get("Market", {})
    mbrief = format_market_brief(market)

    target = t.get("Target", np.nan)
    upside = t.get("Upside", np.nan)

    target_str = f"{target/1000:.1f} ngàn VND" if pd.notna(target) else "N/A"
    upside_str = f"{upside:.1f}%" if pd.notna(upside) else "N/A"

    payload = {
        "ticker": t.get("Ticker"),
        "stock_close": float(t.get("Close")) if pd.notna(t.get("Close")) else None,
        "stock_change_pct": float(t.get("Change")) if pd.notna(t.get("Change")) else None,
        "ma20": float(t.get("MA20")) if pd.notna(t.get("MA20")) else None,
        "ma50": float(t.get("MA50")) if pd.notna(t.get("MA50")) else None,
        "ma200": float(t.get("MA200")) if pd.notna(t.get("MA200")) else None,
        "rsi": float(t.get("RSI")) if pd.notna(t.get("RSI")) else None,
        "macd": float(t.get("MACD")) if pd.notna(t.get("MACD")) else None,
        "signal": float(t.get("Signal")) if pd.notna(t.get("Signal")) else None,
        "target_price_display": target_str,
        "upside_display": upside_str,
        "vnindex_summary": mbrief["VNINDEX_LINE"],
        "vn30_summary": mbrief["VN30_LINE"],
        "relative_perf_summary": mbrief["REL_LINE"],
    }

    system_msg = (
        "Bạn là chuyên gia chiến lược chứng khoán cao cấp. "
        "Chỉ được phép sử dụng đúng các số liệu đã được cung cấp trong JSON. "
        "Tuyệt đối không bịa số, không tự tính toán, không suy diễn thêm con số. "
        "Nếu thiếu dữ liệu thì ghi rõ N/A. "
        "Viết 3-5 câu tiếng Việt, văn phong chuyên gia, tóm tắt: "
        "(1) trạng thái kỹ thuật (MA/RSI/MACD) "
        "(2) upside cơ bản "
        "(3) tương quan với thị trường (VNINDEX, VN30) dựa trên các dòng summary đã cho."
    )

    user_msg = f"Dữ liệu (JSON): {payload}\nYêu cầu: viết đoạn 'preface' ngắn gọn theo đúng nguyên tắc trên."

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except:
        return (
            "Thị trường hiện đang dao động trong vùng cân bằng sau nhịp hồi. "
            f"{t['Ticker']} đang thể hiện {('tốt hơn' if t['Change']>0 else 'kém hơn')} thị trường chung, "
            "phản ánh sự chọn lọc dòng tiền giữa các nhóm cổ phiếu. "
            "→ Giai đoạn hiện tại phù hợp với việc canh các nhịp điều chỉnh ngắn để gia tăng vị thế hơn là mua đuổi."
        )

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

def analyze_ticker(ticker: str, fibo_period: int = 250):
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
    fibo = compute_fibo(df, period=fibo_period)

    last = df.iloc[-1]
    close, prev_close = last["Close"], df.iloc[-2]["Close"]
    change = (close - prev_close) / prev_close * 100
    conviction = 7 + (close > last["MA50"]) * 1.5

    target = get_target_price(ticker)
    upside = (target - close) / close * 100 if pd.notna(target) and close != 0 else np.nan

    market = compute_market_context(df_all, change)

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
        "Fibo": fibo,
        "Target": target,
        "Upside": upside,
        "Market": market
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

    preface = gpt_preface_expert(t)

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

    if pd.notna(t.get("Target")) and pd.notna(t.get("Upside")):
        b_block = f"""
### B. Phân tích Cơ bản
Giá mục tiêu: {t['Target']/1000:.1f} ngàn VND | Upside: {t['Upside']:.1f}%
Nhận định: Upside {t['Upside']:.1f}% → ưu tiên chiến lược theo xu hướng, chỉ gia tăng khi kỹ thuật xác nhận.
"""
    else:
        b_block = f"""
### B. Phân tích Cơ bản
Giá mục tiêu: N/A | Upside: N/A
Nhận định: Chưa đọc được target từ file Tickers target price.xlsx cho mã này.
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
    fibo_period = st.selectbox("Fibo Window (phiên)", [60, 90, 250], index=2)
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
            result = analyze_ticker(ticker, fibo_period)
            if "Error" in result:
                st.error(result["Error"])
            else:
                report = generate_report(result)
                st.markdown(f"<div class='report-text'>{report}</div>", unsafe_allow_html=True)
