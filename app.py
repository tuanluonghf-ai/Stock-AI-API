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
    page_title="Tuan Finance - AI Engine",
    page_icon="🦅",
    layout="wide"
)

# Cấu hình đường dẫn file (Trên Render sẽ nằm cùng thư mục root)
PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
TICKER_NAME_PATH = "Ticker name.xlsx"

api_key = os.environ.get("OPENAI_API_KEY")

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# ==============================================================================
# 2. KHU VỰC ENGINE LOGIC (DÁN NGUYÊN VĂN CODE CỦA ANH TỪ ĐÂY)
# ==============================================================================

# --- Formatting helpers ---
def fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%a, %d-%b-%Y")

def _fmt_price(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.2f}"

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{float(x):.1f}%"

def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{int(round(float(x))):,}"

def _safe(x, default=""):
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    return x


# --- Loaders ---
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        # Trả về DataFrame rỗng thay vì raise Error để App không bị crash
        return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["date", "ngay", "day", "datetime"]:
            col_map[c] = "Date"
        elif lc in ["ticker", "symbol", "ma", "mã"]:
            col_map[c] = "Ticker"
        elif lc in ["close", "closeprice", "close price", "giá đóng cửa", "gia dong cua"]:
            col_map[c] = "Close"
        elif lc in ["volume", "vol", "khối lượng", "khoi luong"]:
            col_map[c] = "Volume"
        elif lc == "open":
            col_map[c] = "Open"
        elif lc == "high":
            col_map[c] = "High"
        elif lc == "low":
            col_map[c] = "Low"
        elif lc in ["vma 20", "vma20", "avg20vol"]:
            col_map[c] = "VMA20_File"

    df = df.rename(columns=col_map)

    required = ["Date", "Ticker", "Close", "Volume"]
    for r in required:
        if r not in df.columns:
            df[r] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in ["Close", "Volume", "Open", "High", "Low", "VMA20_File"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name"])

    df.columns = [str(c).strip() for c in df.columns]

    if "Ticker" not in df.columns:
        df["Ticker"] = ""

    name_col = "Stock Name" if "Stock Name" in df.columns else ("Name" if "Name" in df.columns else None)
    if name_col is None:
        df["Name"] = ""
    else:
        df = df.rename(columns={name_col: "Name"})

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    return df[["Ticker", "Name"]].drop_duplicates()


def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception:
        return pd.DataFrame(columns=["Date", "Ticker", "CTCK", "Recommendation", "Target", "Link", "Company", "ClosePrice"])

    df.columns = [str(c).strip() for c in df.columns]
    
    # Chuẩn hóa tên cột linh hoạt hơn một chút
    # Nếu file Excel có cột 'TP (VND)' thì giữ, không thì tìm cột tương tự
    if "TP (VND)" not in df.columns and "Target" in df.columns:
        df.rename(columns={"Target": "TP (VND)"}, inplace=True)
        
    for c in ["Ticker", "Company", "Recommendation", "TP (VND)", "Close Price (VND)"]:
        if c not in df.columns:
            df[c] = np.nan

    out = pd.DataFrame()
    out["Date"] = ""
    out["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    out["CTCK"] = "HSC"
    out["Recommendation"] = df["Recommendation"].astype(str).str.strip()
    out["Target"] = pd.to_numeric(df["TP (VND)"], errors="coerce")
    out["Link"] = ""
    out["Company"] = df["Company"].astype(str).str.strip()
    out["ClosePrice"] = pd.to_numeric(df["Close Price (VND)"], errors="coerce")
    
    # Lấy thêm các chỉ số cơ bản quan trọng nếu có
    if "Upside/Downside" in df.columns:
        out["Upside"] = pd.to_numeric(df["Upside/Downside"], errors="coerce")
    else:
        out["Upside"] = 0
        
    if "2025F P/E" in df.columns:
        out["PE_2025"] = pd.to_numeric(df["2025F P/E"], errors="coerce")
    else:
        out["PE_2025"] = 0
        
    return out


# --- Indicators ---
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# --- Fibonacci (60-day swing) ---
def fib_60d_levels(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    tail = df.tail(lookback)
    if "High" in tail.columns and "Low" in tail.columns and tail["High"].notna().any() and tail["Low"].notna().any():
        swing_hi = float(tail["High"].max())
        swing_lo = float(tail["Low"].min())
    else:
        swing_hi = float(tail["Close"].max())
        swing_lo = float(tail["Close"].min())

    rng = swing_hi - swing_lo
    if rng <= 0:
        return {"hi": swing_hi, "lo": swing_lo}

    ratios = {"23.6%": 0.236, "38.2%": 0.382, "50.0%": 0.500, "61.8%": 0.618, "78.6%": 0.786}
    levels = {"hi": swing_hi, "lo": swing_lo}
    for k, r in ratios.items():
        levels[k] = swing_lo + r * rng
    return levels


# --- Fib: Support/Resistance + Zones ---
def fib_support_resistance(fib: dict, close: float):
    levels = []
    for k, v in (fib or {}).items():
        if v is None:
            continue
        try:
            lv = float(v)
        except Exception:
            continue
        if np.isnan(lv) or lv <= 0:
            continue
        dist_pct = (lv / close - 1.0) * 100.0 if close and close > 0 else np.nan
        levels.append((k, lv, dist_pct))

    resist = [x for x in levels if x[1] > close]
    supp   = [x for x in levels if x[1] < close]

    resist.sort(key=lambda x: x[1])                 # nearest above close
    supp.sort(key=lambda x: x[1], reverse=True)     # nearest below close
    return resist, supp


def fib_zones(levels: list, close: float, merge_threshold_pct: float = 0.30):
    if not levels or close <= 0:
        return []

    thr = (merge_threshold_pct / 100.0) * close

    zones = []
    cur = {"low": levels[0][1], "high": levels[0][1], "labels": [levels[0][0]]}

    def within(a, b):
        return abs(a - b) <= thr

    for (lab, lv, dist) in levels[1:]:
        if within(lv, cur["low"]) or within(lv, cur["high"]):
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
        dist_center_pct = (center / close - 1.0) * 100.0
        out.append({
            "low": float(z["low"]),
            "high": float(z["high"]),
            "center": float(center),
            "labels": z["labels"],
            "dist_center_pct": float(dist_center_pct),
        })
    return out


def nearest_zone_above(zones: list):
    if not zones:
        return None
    return sorted(zones, key=lambda z: z["center"])[0]

def nearest_zone_below(zones: list):
    if not zones:
        return None
    return sorted(zones, key=lambda z: z["center"], reverse=True)[0]


# --- 12-scenario classification ---
def classify_12_scenarios(last: Dict[str, float]) -> str:
    c = last.get("Close", np.nan)
    ma20 = last.get("MA20", np.nan)
    ma50 = last.get("MA50", np.nan)
    ma200 = last.get("MA200", np.nan)
    rsi = last.get("RSI14", np.nan)
    macd_v = last.get("MACD", np.nan)
    macd_s = last.get("MACDSignal", np.nan)

    if any(np.isnan(x) for x in [c, ma20, ma50, ma200, rsi, macd_v, macd_s]):
        return "Insufficient data"

    above20, above50, above200 = c > ma20, c > ma50, c > ma200
    bull_stack = (ma20 > ma50) and (ma50 > ma200)
    bear_stack = (ma20 < ma50) and (ma50 < ma200)
    macd_bull = macd_v > macd_s
    macd_above0 = macd_v > 0
    rsi_bull = rsi >= 55
    rsi_bear = rsi <= 45

    if bull_stack and above20 and above50 and above200 and rsi_bull and macd_bull and macd_above0:
        return "1) Strong uptrend (trend continuation)"
    if bull_stack and above200 and (not above20) and rsi >= 45 and macd_bull:
        return "2) Uptrend pullback (buy-the-dip zone)"
    if above200 and (not bull_stack) and above20 and rsi_bull and macd_bull:
        return "3) Early uptrend / re-accumulation"
    if above200 and (not above50) and rsi_bull and macd_bull:
        return "4) Trend transition (reclaiming MA50)"
    if above200 and above50 and (not above20) and rsi >= 50:
        return "5) Shallow pullback in bullish regime"
    if above200 and abs(c - ma50) / ma50 < 0.01 and 45 <= rsi <= 55:
        return "6) Sideways above MA200 (range/accumulation)"
    if (not above200) and bear_stack and rsi_bear and (not macd_bull):
        return "7) Strong downtrend (avoid)"
    if (not above200) and bear_stack and rsi >= 35 and macd_bull:
        return "8) Downtrend relief rally (sell into strength)"
    if (not above200) and (not bear_stack) and macd_bull and rsi >= 50:
        return "9) Bottoming attempt (speculative)"
    if (not above200) and rsi_bear and macd_v < 0 and macd_bull:
        return "10) Divergence bounce (tight risk)"
    if above200 and (not above50) and rsi < 50 and (not macd_bull):
        return "11) Failed reclaim (caution)"
    return "12) Neutral / mixed signals (wait confirmation)"


# --- Conviction score ---
def score_trend(last: Dict[str, float]) -> float:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    score = 0.0
    score += 8 if c > ma20 else 0
    score += 8 if c > ma50 else 0
    score += 8 if c > ma200 else 0
    if (ma20 > ma50) and (ma50 > ma200):
        score += 11
    elif (ma20 < ma50) and (ma50 < ma200):
        score += 0
    else:
        score += 5
    return float(min(35.0, max(0.0, score)))

def score_momentum(last: Dict[str, float]) -> float:
    rsi, macd_v, macd_s = last["RSI14"], last["MACD"], last["MACDSignal"]
    score = 0.0
    if rsi >= 70:
        score += 18
    elif rsi >= 55:
        score += 14
    elif rsi >= 45:
        score += 10
    elif rsi >= 30:
        score += 6
    else:
        score += 2
    score += 4 if macd_v > macd_s else 1
    score += 3 if macd_v > 0 else 0
    return float(min(25.0, max(0.0, score)))

def score_volume(last: Dict[str, float]) -> float:
    vol, avg20 = last["Volume"], last["Avg20Vol"]
    if np.isnan(avg20) or avg20 <= 0:
        return 0.0
    ratio = vol / avg20
    if ratio >= 2.0: return 20.0
    if ratio >= 1.5: return 16.0
    if ratio >= 1.1: return 12.0
    if ratio >= 0.9: return 9.0
    if ratio >= 0.7: return 6.0
    return 3.0

def score_structure(df: pd.DataFrame) -> float:
    if len(df) < 60:
        return 0.0
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


# --- Trade plan + R:R ---
@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str


def build_trade_plan(df: pd.DataFrame, fib: Dict[str, float]) -> Dict[str, TradeSetup]:
    close_s = df["Close"]
    last_close = float(close_s.iloc[-1])

    ma20 = float(df["MA20"].iloc[-1]) if pd.notna(df["MA20"].iloc[-1]) else np.nan
    ma50 = float(df["MA50"].iloc[-1]) if pd.notna(df["MA50"].iloc[-1]) else np.nan

    # Fib -> S/R -> zones
    resist, supp = fib_support_resistance(fib, last_close)
    res_z = fib_zones(resist, last_close, merge_threshold_pct=0.30)
    sup_z = fib_zones(supp, last_close, merge_threshold_pct=0.30)

    near_res = nearest_zone_above(res_z)   # nearest resistance zone
    near_sup = nearest_zone_below(sup_z)   # nearest support zone

    hi20 = float(close_s.tail(20).max())
    lo20 = float(close_s.tail(20).min())

    # --- Breakout ---
    breakout_base = near_res["high"] if near_res else hi20
    breakout_entry = round(float(breakout_base) * 1.002, 2)  # +0.2%

    if near_sup:
        breakout_stop = min(near_sup["low"] * 0.998, ma20 * 0.995 if not np.isnan(ma20) else near_sup["low"] * 0.998)
    else:
        breakout_stop = (ma20 * 0.992) if not np.isnan(ma20) else (breakout_entry * 0.98)
    breakout_stop = round(float(breakout_stop), 2)

    # TP = next resistance zone center if exists else fib hi else RR target
    next_res = None
    if res_z and len(res_z) >= 2:
        res_sorted = sorted(res_z, key=lambda z: z["center"])
        next_res = res_sorted[1]
    if next_res:
        breakout_tp = round(float(next_res["center"]), 2)
    else:
        fib_hi = fib.get("hi", np.nan)
        breakout_tp = round(float(fib_hi), 2) if fib_hi is not None and not (isinstance(fib_hi, float) and np.isnan(fib_hi)) else round(breakout_entry * 1.06, 2)

    breakout_rr = (breakout_tp - breakout_entry) / max(1e-9, (breakout_entry - breakout_stop))

    # --- Pullback ---
    pullback_entry = near_sup["center"] if near_sup else (ma20 if not np.isnan(ma20) else last_close)
    pullback_entry = round(float(pullback_entry), 2)

    pullback_stop = (near_sup["low"] * 0.997) if near_sup else ((ma50 * 0.985) if not np.isnan(ma50) else (pullback_entry * 0.97))
    if not np.isnan(ma50):
        pullback_stop = min(pullback_stop, ma50 * 0.99)
    pullback_stop = round(float(pullback_stop), 2)

    pullback_tp = near_res["center"] if near_res else hi20
    pullback_tp = round(float(max(pullback_tp, pullback_entry * 1.02)), 2)

    pullback_rr = (pullback_tp - pullback_entry) / max(1e-9, (pullback_entry - pullback_stop))

    # Probability (deterministic): RSI+MACD alignment
    rsi = float(df["RSI14"].iloc[-1]) if pd.notna(df["RSI14"].iloc[-1]) else np.nan
    macd_v = float(df["MACD"].iloc[-1]) if pd.notna(df["MACD"].iloc[-1]) else np.nan
    macd_s = float(df["MACDSignal"].iloc[-1]) if pd.notna(df["MACDSignal"].iloc[-1]) else np.nan
    align = (not np.isnan(rsi)) and (not np.isnan(macd_v)) and (not np.isnan(macd_s)) and (rsi >= 55) and (macd_v > macd_s)

    breakout_prob = "Cao" if align else "Trung bình"
    pullback_prob = "Trung–cao" if align and (not np.isnan(ma50)) and last_close > ma50 else "Trung bình"

    return {
        "Breakout": TradeSetup("Breakout", breakout_entry, breakout_stop, breakout_tp, float(breakout_rr), breakout_prob),
        "Pullback": TradeSetup("Pullback", pullback_entry, pullback_stop, pullback_tp, float(pullback_rr), pullback_prob),
    }


def weighted_rr(setups: Dict[str, TradeSetup]) -> Tuple[float, str]:
    weights = {"Cao": 0.55, "Trung–cao": 0.45, "Trung bình": 0.35, "Thấp": 0.20}
    total_w, total = 0.0, 0.0
    best, best_metric = "", -1.0
    for k, s in setups.items():
        w = weights.get(s.probability, 0.30)
        total_w += w
        total += w * s.rr
        metric = s.rr * w
        if metric > best_metric:
            best_metric, best = metric, k
    if total_w <= 0:
        return float("nan"), ""
    return total / total_w, best


# =============================
# Core analysis
# =============================
def analyze_ticker(ticker: str) -> Dict[str, Any]:
    ticker = ticker.upper().strip()

    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty:
        return {"Ticker": ticker, "Error": "Không đọc được file Price_Vol.xlsx trên Server"}
        
    df = df_all[df_all["Ticker"] == ticker].copy()
    if df.empty:
        return {"Ticker": ticker, "Error": f"Mã {ticker} không tìm thấy trong dữ liệu Price_Vol."}

    df = df.sort_values("Date").reset_index(drop=True)

    # Indicators
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
    last = {
        "Date": last_row["Date"],
        "Close": float(last_row["Close"]),
        "Volume": float(last_row["Volume"]),
        "Avg20Vol": float(last_row["Avg20Vol"]) if pd.notna(last_row["Avg20Vol"]) else np.nan,
        "MA20": float(last_row["MA20"]) if pd.notna(last_row["MA20"]) else np.nan,
        "MA50": float(last_row["MA50"]) if pd.notna(last_row["MA50"]) else np.nan,
        "MA200": float(last_row["MA200"]) if pd.notna(last_row["MA200"]) else np.nan,
        "RSI14": float(last_row["RSI14"]) if pd.notna(last_row["RSI14"]) else np.nan,
        "MACD": float(last_row["MACD"]) if pd.notna(last_row["MACD"]) else np.nan,
        "MACDSignal": float(last_row["MACDSignal"]) if pd.notna(last_row["MACDSignal"]) else np.nan,
        "MACDHist": float(last_row["MACDHist"]) if pd.notna(last_row["MACDHist"]) else np.nan,
        "ChgPct": float(last_row["ChgPct"]) if pd.notna(last_row["ChgPct"]) else np.nan,
    }

    scenario = classify_12_scenarios(last)

    # Conviction score
    req = ["Close", "MA20", "MA50", "MA200", "RSI14", "MACD", "MACDSignal", "Volume", "Avg20Vol"]
    if all(pd.notna(last.get(k, np.nan)) for k in req):
        tscore = score_trend(last)
        mscore = score_momentum(last)
        vscore = score_volume(last)
        sscore = score_structure(df)
        conviction = tscore + mscore + vscore + sscore
    else:
        tscore = mscore = vscore = sscore = np.nan
        conviction = np.nan
        
    conv_bd = {"Trend_35": tscore, "Momentum_25": mscore, "Volume_20": vscore, "Structure_20": sscore}

    setups = build_trade_plan(df, fib)
    avg_rr, preferred = weighted_rr(setups)

    # Names + HSC
    names = load_ticker_names(TICKER_NAME_PATH)
    company_name = ""
    m = names[names["Ticker"] == ticker]
    if not m.empty:
        company_name = str(m.iloc[0]["Name"])

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
            # Lấy thêm Upside từ file HSC
            "Upside": float(r["Upside"]) if "Upside" in r and pd.notna(r["Upside"]) else 0,
            "PE_2025": float(r["PE_2025"]) if "PE_2025" in r and pd.notna(r["PE_2025"]) else 0
        }
    else:
        hsc_row = {"Date": "", "CTCK": "HSC", "Recommendation": "", "Target": None, "Link": "", "Upside": 0, "PE_2025": 0}

    return {
        "Header": {
            "Ticker": ticker,
            "CompanyName": company_name,
            "LastPrice": round(last["Close"], 2),
            "ChgPct": round(last["ChgPct"], 1) if pd.notna(last["ChgPct"]) else None,
            "Date": fmt_date(pd.Timestamp(last["Date"])),
        },
        "Indicators": {
            **last,
            "Fib60D": {k: float(v) for k, v in fib.items()},
            "Scenario": scenario,
            "ConvictionScore": float(conviction) if pd.notna(conviction) else None,
            "ConvictionBreakdown": conv_bd,
        },
        "HSC": hsc_row,
        "TradePlan": setups,
        "RRSimulation": {"WeightedAvgRR": round(avg_rr, 2) if not np.isnan(avg_rr) else None, "Preferred": preferred},
    }

# ==========================================
# 3. MARKDOWN RENDERER (Hàm tạo văn bản báo cáo)
# ==========================================
def render_markdown(res: dict) -> str:
    # (Đoạn này vẫn giữ nguyên từ code gốc của anh, tôi chỉ đóng gói lại thành hàm để dùng cho Streamlit)
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

    header_name = f"{ticker} ({cname})" if cname else ticker
    header = f"**{header_name}**\n**Giá đóng cửa:** {_fmt_price(last_price)} ({_fmt_pct(chg)}) — **Ngày:** {dt}"

    close = ind.get("Close", None)
    vol = ind.get("Volume", None)
    avg20 = ind.get("Avg20Vol", None)
    ma20 = ind.get("MA20", None)
    ma50 = ind.get("MA50", None)
    ma200 = ind.get("MA200", None)
    rsi14 = ind.get("RSI14", None)
    macd_v = ind.get("MACD", None)
    macd_s = ind.get("MACDSignal", None)

    scenario = _safe(ind.get("Scenario", ""))
    conv = ind.get("ConvictionScore", None)
    conv_bd = ind.get("ConvictionBreakdown", {}) or {}

    # Logic diễn giải xu hướng (Interpretations)
    ma_trend = []
    if all(not (isinstance(x, float) and np.isnan(x)) for x in [close, ma20, ma50, ma200]):
        if close > ma20 and close > ma50 and close > ma200 and (ma20 > ma50 > ma200):
            ma_trend.append("Giá trên MA20/50/200 và MA20>MA50>MA200 → **Uptrend**.")
        elif close > ma200 and (close < ma20 or close < ma50):
            ma_trend.append("Giá trên MA200 nhưng giằng co với MA20/MA50 → **Pullback / tích lũy**.")
        elif close < ma200 and (ma20 < ma50 < ma200):
            ma_trend.append("Giá dưới MA200 và MA xếp giảm dần → **Downtrend**.")
        else:
            ma_trend.append("Cấu trúc MA trộn lẫn → **Trung tính**, chờ xác nhận.")

    rsi_note = []
    if not (isinstance(rsi14, float) and np.isnan(rsi14)):
        if rsi14 >= 70: rsi_note.append("RSI vào vùng quá mua (>=70) → ưu tiên quản trị rủi ro.")
        elif rsi14 >= 55: rsi_note.append("RSI tích cực (>=55) → động lượng tăng chiếm ưu thế.")
        elif rsi14 >= 45: rsi_note.append("RSI trung tính (45–55) → thị trường cân bằng.")
        else: rsi_note.append("RSI yếu (<=45) → rủi ro điều chỉnh cao hơn.")

    macd_note = []
    if not (isinstance(macd_v, float) and np.isnan(macd_v)):
        if macd_v > macd_s and macd_v > 0: macd_note.append("MACD trên Signal & trên 0 → xu hướng tăng được xác nhận.")
        elif macd_v > macd_s and macd_v <= 0: macd_note.append("MACD trên Signal nhưng dưới 0 → hồi phục sớm.")
        else: macd_note.append("MACD dưới Signal → động lượng suy yếu.")

    bias = "Mixed"
    if not (isinstance(rsi14, float) and np.isnan(rsi14)):
        if rsi14 >= 55 and macd_v > macd_s: bias = "Bullish"
        elif rsi14 <= 45 and macd_v < macd_s: bias = "Bearish"
        else: bias = "Neutral"

    # Xây dựng nội dung Markdown
    md = []
    md.append(header)
    md.append("\n---\n")
    md.append("### A. Indicator Snapshot")
    md.append(f"- **Close:** {_fmt_price(close)}")
    md.append(f"- **Volume:** {_fmt_int(vol)} | **Avg20 Vol:** {_fmt_int(avg20)}")
    md.append(f"- **MA20/50/200:** {_fmt_price(ma20)} / {_fmt_price(ma50)} / {_fmt_price(ma200)}")
    md.append(f"- **RSI(14):** {_fmt_price(rsi14)}")
    md.append(f"- **MACD:** {_fmt_price(macd_v)}")

    md.append("\n#### 1. MA Trend Analysis")
    md.extend([f"- {x}" for x in ma_trend])
    
    md.append("\n#### 2. RSI Analysis")
    md.extend([f"- {x}" for x in rsi_note])
    
    md.append("\n#### 3. MACD Analysis")
    md.extend([f"- {x}" for x in macd_note])
    
    md.append(f"\n#### 4. RSI + MACD Bias Matrix\n- Bias: **{bias}**")

    md.append("\n#### 5. Fibonacci Levels (60 phiên gần nhất)")
    # Logic hiển thị Fib Lines
    if fib and (close is not None):
        resist, supp = fib_support_resistance(fib, float(close))
        res_z = fib_zones(resist, float(close))
        sup_z = fib_zones(supp, float(close))
        
        md.append("- **Vùng hỗ trợ quan trọng:**")
        if sup_z:
            top_sup = sorted(sup_z, key=lambda x: x["center"], reverse=True)[0]
            md.append(f"  {top_sup['low']:.2f}-{top_sup['high']:.2f} (Center: {top_sup['center']:.2f})")
        else: md.append("  (Không có hỗ trợ gần)")
            
        md.append("- **Kháng cự gần:**")
        if res_z:
            top_res = sorted(res_z, key=lambda x: x["center"])[0]
            md.append(f"  {top_res['low']:.2f}-{top_res['high']:.2f} (Center: {top_res['center']:.2f})")
        else: md.append("  (Không có kháng cự gần)")

    md.append("\n#### 6. Volume & Price Action")
    vol_str = "Vượt trung bình" if vol > avg20 else "Thấp hơn trung bình"
    md.append(f"- Volume {vol_str} (gấp {vol/avg20:.1f}x Avg20)")

    md.append("\n#### 7. 12-Scenario Classification")
    md.append(f"- **Scenario:** {scenario}")

    md.append("\n#### 8. Master Integration + Conviction Score")
    md.append(f"- **Conviction Score:** {conv:.0f}/100")
    md.append(f"  *(Trend: {conv_bd.get('Trend_35',0)} | Momentum: {conv_bd.get('Momentum_25',0)} | Volume: {conv_bd.get('Volume_20',0)} | Structure: {conv_bd.get('Structure_20',0)})*")

    md.append("\n---\n")
    md.append("### B. Fundamental Analysis Summary")
    md.append(f"- **Recommendation:** {hsc.get('Recommendation', 'N/A')}")
    md.append(f"- **Target Price:** {_fmt_price(hsc.get('Target'))} (Upside: {hsc.get('Upside',0)*100:.1f}%)")
    md.append(f"- **P/E 2025F:** {hsc.get('PE_2025', 'N/A')}")
    
    md.append("\n---\n")
    md.append("### D. Suggestions (Trade Plan)")
    
    # Table Trade Plan
    md.append("| Chiến lược | Entry | Stop-loss | Take-profit | Xác suất |")
    md.append("|---|---|---|---|---|")
    for k, v in tp.items():
        md.append(f"| {k} | {_fmt_price(v.entry)} | {_fmt_price(v.stop)} | {_fmt_price(v.tp)} | {v.probability} |")

    md.append(f"\n**Tổng hợp:** R:R trung bình trọng số ≈ **{_fmt_price(rr.get('WeightedAvgRR'))}**.")
    md.append(f"Kịch bản ưu tiên: **{rr.get('Preferred')}**.")
    
    return "\n".join(md)


# ==========================================
# 4. GIAO DIỆN WEB (STREAMLIT)
# ==========================================
st.title("🦅 PRO STOCK ENGINE (NATIVE LOGIC)")
st.caption("Engine tính toán gốc - Kết hợp AI Lập luận")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    user_key = st.text_input("🔑 Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("🚀 RUN ENGINE", type="primary")

# Main Execution
if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    else:
        with st.spinner(f"Đang chạy Engine cho mã {ticker_input}..."):
            # 1. Gọi hàm phân tích (Engine gốc)
            result = analyze_ticker(ticker_input)
            
            # 2. Kiểm tra lỗi
            if "Error" in result:
                st.error(f"❌ {result['Error']}")
            else:
                # 3. Tạo báo cáo từ Engine (Văn bản thô)
                engine_report = render_markdown(result)
                
                # 4. Hiển thị báo cáo thô của Engine trước
                st.success("✅ Engine chạy thành công!")
                
                # Chia cột hiển thị chỉ số quan trọng
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Giá", _fmt_price(result["Header"]["LastPrice"]))
                c2.metric("Target (HSC)", _fmt_price(result["HSC"]["Target"]), 
                          f"{result['HSC']['Upside']*100:.1f}%" if result['HSC']['Upside'] else None)
                c3.metric("Conviction", f"{result['Indicators']['ConvictionScore']:.0f}/100")
                c4.metric("Kịch bản", result["RRSimulation"]["Preferred"])
                
                with st.expander("📄 Xem dữ liệu thô từ Engine (Click để mở)"):
                    st.markdown(engine_report)
                
                # 5. Gửi cho AI (GPT) để viết lại cho hay hơn
                if api_key:
                    st.divider()
                    st.subheader("🤖 AI Executive Report")
                    with st.spinner("AI đang đọc báo cáo của Engine và viết nhận định..."):
                        try:
                            client = OpenAI(api_key=api_key)
                            
                            # Prompt: Đưa toàn bộ báo cáo thô của Engine vào và bảo AI viết lại
                            prompt = f"""
                            Bạn là Giám đốc Phân tích Đầu tư. Dưới đây là dữ liệu tính toán chi tiết từ hệ thống Engine của tôi:
                            
                            --- BEGIN ENGINE REPORT ---
                            {engine_report}
                            --- END ENGINE REPORT ---
                            
                            YÊU CẦU:
                            Dựa trên báo cáo thô ở trên, hãy viết một bản nhận định đầu tư chuyên nghiệp, ngắn gọn (khoảng 500 từ).
                            Tập trung vào:
                            1. Sức khỏe xu hướng (Dựa trên Scenario và Conviction Score).
                            2. Sự đồng thuận giữa Kỹ thuật (Engine) và Cơ bản (HSC Target).
                            3. Hành động khuyến nghị (Dựa trên Trade Plan: Entry/Stop/TP).
                            
                            Giọng văn dứt khoát, khách quan.
                            """
                            
                            res = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": prompt}]
                            )
                            st.markdown(res.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Lỗi kết nối AI: {str(e)}")