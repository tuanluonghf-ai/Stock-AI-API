import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import argparse

# ==========================================
# 1. CẤU HÌNH WEB APP
# ==========================================
st.set_page_config(
    page_title="Tuan Finance",
    page_icon="🦅",
    layout="wide"
)

# Cấu hình đường dẫn file
PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
TICKER_NAME_PATH = "Ticker name.xlsx"

api_key = os.environ.get("OPENAI_API_KEY")

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời 01", "quota": 5},
    "KH02":   {"name": "Khách mời 02", "quota": 5},
    "KH03":   {"name": "Khách mời 03", "quota": 5},
    "KH04":   {"name": "Khách mời 04", "quota": 5},
    "KH05":   {"name": "Khách mời 05", "quota": 5},
}
# ==============================================================================
# 2. KHU VỰC ENGINE LOGIC (GIỮ NGUYÊN BẢN 100% LOGIC TÍNH TOÁN)
# ==============================================================================

# --- Formatting helpers ---
def fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%d/%m/%Y")

def _fmt_price(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{float(x):.2f}"

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{float(x):.1f}%"

def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{int(round(float(x))):,}"

def _safe(x, default=""):
    if x is None: return default
    if isinstance(x, float) and np.isnan(x): return default
    return x

# --- Loaders ---
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["date", "ngay", "day", "datetime"]: col_map[c] = "Date"
        elif lc in ["ticker", "symbol", "ma", "mã"]: col_map[c] = "Ticker"
        elif lc in ["close", "closeprice", "close price", "giá đóng cửa"]: col_map[c] = "Close"
        elif lc in ["volume", "vol", "khối lượng"]: col_map[c] = "Volume"
        elif lc == "open": col_map[c] = "Open"
        elif lc == "high": col_map[c] = "High"
        elif lc == "low": col_map[c] = "Low"
        elif lc in ["vma 20", "vma20", "avg20vol"]: col_map[c] = "VMA20_File"

    df = df.rename(columns=col_map)
    required = ["Date", "Ticker", "Close", "Volume"]
    for r in required:
        if r not in df.columns: df[r] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Close", "Volume", "Open", "High", "Low", "VMA20_File"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name"])
    df.columns = [str(c).strip() for c in df.columns]
    if "Ticker" not in df.columns: df["Ticker"] = ""
    name_col = "Stock Name" if "Stock Name" in df.columns else ("Name" if "Name" in df.columns else None)
    if name_col is None: df["Name"] = ""
    else: df = df.rename(columns={name_col: "Name"})
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    return df[["Ticker", "Name"]].drop_duplicates()

def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=["Date", "Ticker", "CTCK", "Recommendation", "Target", "Link", "Company", "ClosePrice"])
    df.columns = [str(c).strip() for c in df.columns]
    if "TP (VND)" not in df.columns and "Target" in df.columns:
        df.rename(columns={"Target": "TP (VND)"}, inplace=True)
    for c in ["Ticker", "Company", "Recommendation", "TP (VND)", "Close Price (VND)"]:
        if c not in df.columns: df[c] = np.nan
    out = pd.DataFrame()
    out["Date"] = ""
    out["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    out["CTCK"] = "HSC"
    out["Recommendation"] = df["Recommendation"].astype(str).str.strip()
    out["Target"] = pd.to_numeric(df["TP (VND)"], errors="coerce")
    out["Link"] = ""
    out["Company"] = df["Company"].astype(str).str.strip()
    out["ClosePrice"] = pd.to_numeric(df["Close Price (VND)"], errors="coerce")
    if "Upside/Downside" in df.columns: out["Upside"] = pd.to_numeric(df["Upside/Downside"], errors="coerce")
    else: out["Upside"] = 0
    if "2025F P/E" in df.columns: out["PE_2025"] = pd.to_numeric(df["2025F P/E"], errors="coerce")
    else: out["PE_2025"] = 0
    return out

# --- Indicators ---
def sma(series, window): return series.rolling(window=window, min_periods=window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False, min_periods=span).mean()
def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --- Fib Logic ---
def fib_60d_levels(df, lookback=60):
    tail = df.tail(lookback)
    if "High" in tail.columns and "Low" in tail.columns:
        swing_hi = float(tail["High"].max())
        swing_lo = float(tail["Low"].min())
    else:
        swing_hi = float(tail["Close"].max())
        swing_lo = float(tail["Close"].min())
    rng = swing_hi - swing_lo
    if rng <= 0: return {"hi": swing_hi, "lo": swing_lo}
    ratios = {"23.6%": 0.236, "38.2%": 0.382, "50.0%": 0.500, "61.8%": 0.618, "78.6%": 0.786}
    levels = {"hi": swing_hi, "lo": swing_lo}
    for k, r in ratios.items(): levels[k] = swing_lo + r * rng
    return levels

def fib_support_resistance(fib, close):
    levels = []
    for k, v in (fib or {}).items():
        if v is None or np.isnan(float(v)) or float(v) <= 0: continue
        levels.append((k, float(v)))
    resist = [x for x in levels if x[1] > close]
    supp   = [x for x in levels if x[1] < close]
    resist.sort(key=lambda x: x[1])
    supp.sort(key=lambda x: x[1], reverse=True)
    return resist, supp

def fib_zones(levels_list, close, merge_threshold_pct=0.30):
    if not levels_list or close <= 0: return []
    thr = (merge_threshold_pct / 100.0) * close
    zones = []
    cur = {"low": levels_list[0][1], "high": levels_list[0][1], "labels": [levels_list[0][0]]}
    for (lab, lv) in levels_list[1:]:
        if abs(lv - cur["low"]) <= thr or abs(lv - cur["high"]) <= thr:
            cur["low"] = min(cur["low"], lv)
            cur["high"] = max(cur["high"], lv)
            cur["labels"].append(lab)
        else:
            zones.append(cur)
            cur = {"low": lv, "high": lv, "labels": [lab]}
    zones.append(cur)
    out = []
    for z in zones:
        center = (z["low"] + z["high"]) / 2.0
        out.append({"low": z["low"], "high": z["high"], "center": center, "labels": z["labels"]})
    return out

def nearest_zone_above(zones): return sorted(zones, key=lambda z: z["center"])[0] if zones else None
def nearest_zone_below(zones): return sorted(zones, key=lambda z: z["center"], reverse=True)[0] if zones else None

# --- 12 Scenarios ---
def classify_12_scenarios(last):
    c = last.get("Close", np.nan)
    ma20, ma50, ma200 = last.get("MA20", np.nan), last.get("MA50", np.nan), last.get("MA200", np.nan)
    rsi, macd_v, macd_s = last.get("RSI14", np.nan), last.get("MACD", np.nan), last.get("MACDSignal", np.nan)
    if any(np.isnan(x) for x in [c, ma20, ma50, ma200, rsi]): return "Insufficient data"
    above20, above50, above200 = c > ma20, c > ma50, c > ma200
    bull_stack = (ma20 > ma50) and (ma50 > ma200)
    bear_stack = (ma20 < ma50) and (ma50 < ma200)
    macd_bull = macd_v > macd_s
    macd_above0 = macd_v > 0
    rsi_bull = rsi >= 55
    rsi_bear = rsi <= 45

    if bull_stack and above20 and above50 and above200 and rsi_bull and macd_bull and macd_above0: return "1) Strong uptrend (trend continuation)"
    if bull_stack and above200 and (not above20) and rsi >= 45 and macd_bull: return "2) Uptrend pullback (buy-the-dip zone)"
    if above200 and (not bull_stack) and above20 and rsi_bull and macd_bull: return "3) Early uptrend / re-accumulation"
    if above200 and (not above50) and rsi_bull and macd_bull: return "4) Trend transition (reclaiming MA50)"
    if above200 and above50 and (not above20) and rsi >= 50: return "5) Shallow pullback in bullish regime"
    if above200 and abs(c - ma50)/ma50 < 0.01 and 45 <= rsi <= 55: return "6) Sideways above MA200 (range/accumulation)"
    if (not above200) and bear_stack and rsi_bear and (not macd_bull): return "7) Strong downtrend (avoid)"
    if (not above200) and bear_stack and rsi >= 35 and macd_bull: return "8) Downtrend relief rally (sell into strength)"
    if (not above200) and (not bear_stack) and macd_bull and rsi >= 50: return "9) Bottoming attempt (speculative)"
    if (not above200) and rsi_bear and macd_v < 0 and macd_bull: return "10) Divergence bounce (tight risk)"
    if above200 and (not above50) and rsi < 50 and (not macd_bull): return "11) Failed reclaim (caution)"
    return "12) Neutral / mixed signals (wait confirmation)"

# --- Scoring ---
def score_trend(last):
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    score = 0.0
    score += 8 if c > ma20 else 0
    score += 8 if c > ma50 else 0
    score += 8 if c > ma200 else 0
    if (ma20 > ma50) and (ma50 > ma200): score += 11
    elif (ma20 < ma50) and (ma50 < ma200): score += 0
    else: score += 5
    return float(min(35.0, max(0.0, score)))

def score_momentum(last):
    rsi, macd_v, macd_s = last["RSI14"], last["MACD"], last["MACDSignal"]
    score = 0.0
    if rsi >= 70: score += 18
    elif rsi >= 55: score += 14
    elif rsi >= 45: score += 10
    elif rsi >= 30: score += 6
    else: score += 2
    score += 4 if macd_v > macd_s else 1
    score += 3 if macd_v > 0 else 0
    return float(min(25.0, max(0.0, score)))

def score_volume(last):
    vol, avg20 = last["Volume"], last["Avg20Vol"]
    if np.isnan(avg20) or avg20 <= 0: return 0.0
    ratio = vol / avg20
    if ratio >= 2.0: return 20.0
    if ratio >= 1.5: return 16.0
    if ratio >= 1.1: return 12.0
    if ratio >= 0.9: return 9.0
    if ratio >= 0.7: return 6.0
    return 3.0

def score_structure(df):
    if len(df) < 60: return 0.0
    close = df["Close"]
    c = float(close.iloc[-1])
    hi20, lo20 = float(close.tail(20).max()), float(close.tail(20).min())
    hi60, lo60 = float(close.tail(60).max()), float(close.tail(60).min())
    def pos(x, lo, hi):
        if hi <= lo: return 0.5
        return (x - lo) / (hi - lo)
    p20, p60 = pos(c, lo20, hi20), pos(c, lo60, hi60)
    score = 10.0 * p20 + 10.0 * p60
    return float(min(20.0, max(0.0, score)))

# --- Trade Plan ---
@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str

def build_trade_plan(df, fib):
    close_s = df["Close"]
    last_close = float(close_s.iloc[-1])
    ma20 = float(df["MA20"].iloc[-1]) if pd.notna(df["MA20"].iloc[-1]) else np.nan
    ma50 = float(df["MA50"].iloc[-1]) if pd.notna(df["MA50"].iloc[-1]) else np.nan
    resist, supp = fib_support_resistance(fib, last_close)
    res_z = fib_zones(resist, last_close)
    sup_z = fib_zones(supp, last_close)
    near_res = nearest_zone_above(res_z)
    near_sup = nearest_zone_below(sup_z)
    hi20 = float(close_s.tail(20).max())

    # Breakout
    breakout_base = near_res["high"] if near_res else hi20
    breakout_entry = round(float(breakout_base) * 1.002, 2)
    if near_sup: breakout_stop = min(near_sup["low"] * 0.998, ma20 * 0.995 if not np.isnan(ma20) else near_sup["low"] * 0.998)
    else: breakout_stop = (ma20 * 0.992) if not np.isnan(ma20) else (breakout_entry * 0.98)
    breakout_stop = round(float(breakout_stop), 2)
    next_res = res_z[1] if res_z and len(res_z) >= 2 else None
    fib_hi = fib.get("hi", np.nan)
    if next_res: breakout_tp = round(float(next_res["center"]), 2)
    else: breakout_tp = round(float(fib_hi), 2) if not np.isnan(fib_hi) else round(breakout_entry * 1.06, 2)
    breakout_rr = (breakout_tp - breakout_entry) / max(1e-9, (breakout_entry - breakout_stop))

    # Pullback
    pullback_entry = near_sup["center"] if near_sup else (ma20 if not np.isnan(ma20) else last_close)
    pullback_entry = round(float(pullback_entry), 2)
    pullback_stop = (near_sup["low"] * 0.997) if near_sup else ((ma50 * 0.985) if not np.isnan(ma50) else (pullback_entry * 0.97))
    if not np.isnan(ma50): pullback_stop = min(pullback_stop, ma50 * 0.99)
    pullback_stop = round(float(pullback_stop), 2)
    pullback_tp = near_res["center"] if near_res else hi20
    pullback_tp = round(float(max(pullback_tp, pullback_entry * 1.02)), 2)
    pullback_rr = (pullback_tp - pullback_entry) / max(1e-9, (pullback_entry - pullback_stop))

    rsi = float(df["RSI14"].iloc[-1]) if pd.notna(df["RSI14"].iloc[-1]) else np.nan
    macd_v = float(df["MACD"].iloc[-1]) if pd.notna(df["MACD"].iloc[-1]) else np.nan
    macd_s = float(df["MACDSignal"].iloc[-1]) if pd.notna(df["MACDSignal"].iloc[-1]) else np.nan
    align = (not np.isnan(rsi)) and (not np.isnan(macd_v)) and (not np.isnan(macd_s)) and (rsi >= 55) and (macd_v > macd_s)
    breakout_prob = "Cao" if align else "TB"
    pullback_prob = "Trung-Cao" if align and (not np.isnan(ma50)) and last_close > ma50 else "TB"

    return {
        "Breakout": TradeSetup("Breakout", breakout_entry, breakout_stop, breakout_tp, float(breakout_rr), breakout_prob),
        "Pullback": TradeSetup("Pullback", pullback_entry, pullback_stop, pullback_tp, float(pullback_rr), pullback_prob),
    }

def weighted_rr(setups):
    weights = {"Cao": 0.55, "Trung-Cao": 0.45, "TB": 0.35, "Thấp": 0.20}
    total_w, total = 0.0, 0.0
    best, best_metric = "", -1.0
    for k, s in setups.items():
        w = weights.get(s.probability, 0.30)
        total_w += w
        total += w * s.rr
        metric = s.rr * w
        if metric > best_metric: best_metric, best = metric, k
    if total_w <= 0: return float("nan"), ""
    return total / total_w, best

# --- Core Analysis ---
def analyze_ticker(ticker: str):
    ticker = ticker.upper().strip()
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty: return {"Ticker": ticker, "Error": "Không đọc được file Price_Vol.xlsx trên Server"}
    df = df_all[df_all["Ticker"] == ticker].copy()
    if df.empty: return {"Ticker": ticker, "Error": f"Mã {ticker} không tìm thấy trong dữ liệu Price_Vol."}
    df = df.sort_values("Date").reset_index(drop=True)

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI14"] = rsi_wilder(df["Close"], 14)
    macd_line, signal_line, hist = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = macd_line, signal_line, hist
    df["PrevClose"] = df["Close"].shift(1)
    df["ChgPct"] = (df["Close"] / df["PrevClose"] - 1.0) * 100.0

    fib = fib_60d_levels(df, 60)
    last_row = df.iloc[-1]
    last = last_row.to_dict()

    scenario = classify_12_scenarios(last)

    tscore = score_trend(last)
    mscore = score_momentum(last)
    vscore = score_volume(last)
    sscore = score_structure(df)
    conviction = tscore + mscore + vscore + sscore
    conv_bd = {"Trend_35": tscore, "Momentum_25": mscore, "Volume_20": vscore, "Structure_20": sscore}

    setups = build_trade_plan(df, fib)
    avg_rr, preferred = weighted_rr(setups)

    names = load_ticker_names(TICKER_NAME_PATH)
    company_name = ""
    m = names[names["Ticker"] == ticker]
    if not m.empty: company_name = str(m.iloc[0]["Name"])

    hsc = load_hsc_targets(HSC_TARGET_PATH)
    hsc_t = hsc[hsc["Ticker"] == ticker].copy()
    hsc_row = {}
    if not hsc_t.empty:
        r = hsc_t.iloc[0]
        hsc_row = {
            "Date": r.get("Date", ""),
            "CTCK": r.get("CTCK", "HSC"),
            "Recommendation": r.get("Recommendation", ""),
            "Target": float(r["Target"]) if pd.notna(r.get("Target", np.nan)) else None,
            "Link": r.get("Link", ""),
            "Upside": float(r["Upside"]) if "Upside" in r and pd.notna(r["Upside"]) else 0,
            "PE_2025": float(r["PE_2025"]) if "PE_2025" in r and pd.notna(r["PE_2025"]) else 0
        }
    else: hsc_row = {"Date": "", "CTCK": "HSC", "Recommendation": "", "Target": None, "Link": "", "Upside": 0, "PE_2025": 0}

    return {
        "Header": {"Ticker": ticker, "CompanyName": company_name, "LastPrice": last["Close"], "ChgPct": last["ChgPct"], "Date": fmt_date(pd.Timestamp(last["Date"]))},
        "Indicators": {**last, "Fib60D": fib, "Scenario": scenario, "ConvictionScore": conviction, "ConvictionBreakdown": conv_bd},
        "HSC": hsc_row,
        "TradePlan": setups,
        "RRSimulation": {"WeightedAvgRR": avg_rr, "Preferred": preferred},
    }

# ==========================================
# 3. MARKDOWN RENDERER (VIỆT HÓA THEO YÊU CẦU)
# ==========================================
def render_markdown(res: dict) -> str:
    h = res.get("Header", {})
    ind = res.get("Indicators", {})
    fib = ind.get("Fib60D", {}) or {}
    hsc = res.get("HSC", {}) or {}
    tp = res.get("TradePlan", {}) or {}
    rr = res.get("RRSimulation", {}) or {}

    ticker = _safe(h.get("Ticker", ""))
    cname = _safe(h.get("CompanyName", ""))
    last_price = h.get("LastPrice", None)
    chg = h.get("ChgPct", None)
    dt = _safe(h.get("Date", ""))

    header = f"**{ticker}** ({cname})\nGiá đóng cửa: **{_fmt_price(last_price)}** ({_fmt_pct(chg)}) | Ngày: **{dt}**"

    close = ind.get("Close", None)
    vol = ind.get("Volume", None)
    avg20 = ind.get("Avg20Vol", None)
    ma20 = ind.get("MA20", None)
    ma50 = ind.get("MA50", None)
    ma200 = ind.get("MA200", None)
    rsi14 = ind.get("RSI14", None)
    macd_v = ind.get("MACD", None)
    
    # 7. Việt hóa Scenario
    scenario_en = _safe(ind.get("Scenario", ""))
    scenario_map = {
        "1) Strong uptrend (trend continuation)": "1) Xu hướng Tăng mạnh (Tiếp diễn)",
        "2) Uptrend pullback (buy-the-dip zone)": "2) Điều chỉnh trong xu hướng Tăng (Vùng mua)",
        "3) Early uptrend / re-accumulation": "3) Chớm tăng / Tái tích lũy",
        "4) Trend transition (reclaiming MA50)": "4) Chuyển pha (Vượt lại MA50)",
        "5) Shallow pullback in bullish regime": "5) Điều chỉnh nhẹ trong pha Tăng",
        "6) Sideways above MA200 (range/accumulation)": "6) Đi ngang trên MA200 (Tích lũy)",
        "7) Strong downtrend (avoid)": "7) Xu hướng Giảm mạnh (Nên tránh)",
        "8) Downtrend relief rally (sell into strength)": "8) Nhịp hồi trong xu hướng Giảm (Bán khi hồi)",
        "9) Bottoming attempt (speculative)": "9) Nỗ lực tạo đáy (Rủi ro cao)",
        "10) Divergence bounce (tight risk)": "10) Bật hồi phân kỳ (Dừng lỗ chặt)",
        "11) Failed reclaim (caution)": "11) Thất bại khi vượt kháng cự (Cẩn trọng)",
        "12) Neutral / mixed signals (wait confirmation)": "12) Tín hiệu trung tính (Chờ xác nhận)",
        "Insufficient data": "Dữ liệu không đủ"
    }
    scenario_vn = scenario_map.get(scenario_en, scenario_en)

    conv = ind.get("ConvictionScore", 0)
    conv_bd = ind.get("ConvictionBreakdown", {})

    # MA Trend logic
    ma_trend = []
    if all(not np.isnan(x) for x in [close, ma20, ma50, ma200]):
        if close > ma20 and close > ma50 and close > ma200 and (ma20 > ma50 > ma200):
            ma_trend.append("Giá trên MA20/50/200 và MA20>MA50>MA200 → **Uptrend mạnh**.")
        elif close > ma200 and (close < ma20 or close < ma50):
            ma_trend.append("Giá trên MA200 nhưng giằng co MA20/50 → **Tích lũy / Điều chỉnh**.")
        elif close < ma200 and (ma20 < ma50 < ma200):
            ma_trend.append("Giá dưới MA200 và các dây MA dốc xuống → **Downtrend**.")
        else:
            ma_trend.append("Cấu trúc MA hỗn hợp → **Trung tính**.")

    # RSI logic
    rsi_note = []
    if not np.isnan(rsi14):
        if rsi14 >= 70: rsi_note.append("RSI >= 70: Vùng **Quá mua** (Cẩn trọng).")
        elif rsi14 >= 55: rsi_note.append("RSI >= 55: Động lượng **Tích cực**.")
        elif rsi14 >= 45: rsi_note.append("RSI 45-55: Trạng thái **Cân bằng**.")
        else: rsi_note.append("RSI <= 45: Động lượng **Yếu**.")

    # Vol logic (Mục 6)
    vol_note = []
    if vol and avg20:
        ratio = vol / avg20
        if ratio >= 1.5: vol_note.append(f"Khối lượng **Đột biến** ({ratio:.1f}x TB20).")
        elif ratio >= 0.9: vol_note.append("Khối lượng **Trung bình**.")
        else: vol_note.append("Khối lượng **Thấp** (Tiết cung hoặc thiếu cầu).")

    # Xây dựng nội dung Markdown
    md = []
    md.append(header)
    md.append("\n---\n")
    md.append("### A. Chỉ số Kỹ thuật Nhanh (Indicator Snapshot)")
    md.append(f"- **Giá:** {_fmt_price(close)}")
    md.append(f"- **Vol:** {_fmt_int(vol)} | **TB 20 phiên:** {_fmt_int(avg20)}")
    md.append(f"- **MA20 / MA50 / MA200:** {_fmt_price(ma20)} / {_fmt_price(ma50)} / {_fmt_price(ma200)}")
    md.append(f"- **RSI(14):** {_fmt_price(rsi14)}")
    md.append(f"- **MACD:** {_fmt_price(macd_v)}")

    md.append("\n#### 1. Xu hướng MA")
    md.extend([f"- {x}" for x in ma_trend])

    md.append("\n#### 2. Phân tích RSI")
    md.extend([f"- {x}" for x in rsi_note])
    
    # 5. Fib
    md.append("\n#### 5. Các mức Fibonacci (60 phiên)")
    resist, supp = fib_support_resistance(fib, float(close))
    res_z = fib_zones(resist, float(close))
    sup_z = fib_zones(supp, float(close))
    
    if sup_z:
        top_sup = sorted(sup_z, key=lambda x: x["center"], reverse=True)[0]
        md.append(f"- **Hỗ trợ gần nhất:** {top_sup['low']:.2f}-{top_sup['high']:.2f}")
    if res_z:
        top_res = sorted(res_z, key=lambda x: x["center"])[0]
        md.append(f"- **Kháng cự gần nhất:** {top_res['low']:.2f}-{top_res['high']:.2f}")

    # 6. Việt hóa Volume
    md.append("\n#### 6. Phân tích Khối lượng & Hành động giá")
    md.extend([f"- {x}" for x in vol_note])

    # 7. Việt hóa Scenario
    md.append("\n#### 7. Phân loại Kịch bản (12 Scenario)")
    md.append(f"- **Trạng thái:** {scenario_vn}")

    # 8. Việt hóa Conviction
    md.append("\n#### 8. Điểm tin cậy tổng hợp (Conviction Score)")
    md.append(f"- **Điểm số:** {conv:.0f}/100")
    md.append(f"  *(Xu hướng: {conv_bd.get('Trend_35',0)} | Động lượng: {conv_bd.get('Momentum_25',0)} | Volume: {conv_bd.get('Volume_20',0)} | Cấu trúc: {conv_bd.get('Structure_20',0)})*")

    md.append("\n---\n")
    # B. Việt hóa Fundamental
    md.append("### B. Tổng hợp Phân tích Cơ bản")
    md.append(f"- **Khuyến nghị gốc:** {hsc.get('Recommendation', 'N/A')}")
    md.append(f"- **Giá mục tiêu:** {_fmt_price(hsc.get('Target'))} (Upside: {hsc.get('Upside',0)*100:.1f}%)")
    md.append(f"- **P/E 2025F:** {hsc.get('PE_2025', 'N/A')}")

    # C. Tin tức (Placeholder - Giữ nguyên theo yêu cầu)
    md.append("\n---\n")
    md.append("### C. Tin tức & Sự kiện")
    md.append("- *(Chức năng đang phát triển - Chưa kết nối dữ liệu thực tế)*")

    md.append("\n---\n")
    md.append("### D. Kế hoạch Giao dịch (Gợi ý)")
    md.append("| Chiến lược | Vào lệnh (Entry) | Cắt lỗ (Stop) | Chốt lời (TP) | Xác suất |")
    md.append("|---|---|---|---|---|")
    
    prob_map = {"Cao": "Cao", "Trung-Cao": "Khá", "TB": "TB", "Thấp": "Thấp"}
    
    for k, v in tp.items():
        k_vn = "Breakout (Phá vỡ)" if k == "Breakout" else "Pullback (Điều chỉnh)"
        prob_vn = prob_map.get(v.probability, v.probability)
        md.append(f"| {k_vn} | {_fmt_price(v.entry)} | {_fmt_price(v.stop)} | {_fmt_price(v.tp)} | {prob_vn} |")

    md.append(f"\n**R:R Trung bình:** {_fmt_price(rr.get('WeightedAvgRR'))}")
    
    return "\n".join(md)

# ==========================================
# 4. GIAO DIỆN WEB STREAMLIT
# ==========================================
# 1. Thay đổi tiêu đề theo ý anh
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
    color: #2E86C1;
}
.quote-font {
    font-size:20px !important;
    font-style: italic;
    color: #555;
}
.sub-text {
    font-size:16px !important;
    color: #333;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Đơn Giản là đỉnh cao của Phức tạp-    Leonardo da Vinci </p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Tôi là sự phức tạp, còn bạn ..?</p>', unsafe_allow_html=True)
st.divider()

# 2. Xóa chữ "Control Panel" trong Sidebar
with st.sidebar:
    # st.header("Control Panel") -> Đã xóa
    user_key = st.text_input("🔑 Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    
    # 3. Đổi nút Run thành XEM
    run_btn = st.button("XEM", type="primary")

# Main Execution
if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    else:
        with st.spinner(f"Đang phân tích {ticker_input}..."):
            result = analyze_ticker(ticker_input)
            
            if "Error" in result:
                st.error(f"❌ {result['Error']}")
            else:
                engine_report = render_markdown(result)
                
                # Hiển thị báo cáo
                st.markdown(engine_report)
                
                # Gửi cho AI (GPT)
                if api_key:
                    st.divider()
                    st.info("🤖 **Góc nhìn Chuyên gia (AI Synthesis):**")
                    try:
                        client = OpenAI(api_key=api_key)
                        prompt = f"""
                        Bạn là Chuyên gia Tài chính cấp cao. Dưới đây là báo cáo kỹ thuật chi tiết:
                        {engine_report}
                        
                        Hãy viết một đoạn nhận định ngắn (khoảng 150 từ) bằng tiếng Việt cho nhà đầu tư cá nhân.
                        Tập trung vào:
                        1. Xu hướng chính (Dựa trên Scenario).
                        2. Hành động cụ thể (Mua/Bán/Chờ) dựa trên Trade Plan.
                        3. Rủi ro cần lưu ý.
                        """
                        res = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.write(res.choices[0].message.content)
                    except: pass
