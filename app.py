import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# ==========================================
# 1. CẤU HÌNH WEB APP & DATA LOADER
# ==========================================
st.set_page_config(
    page_title="Tuan Finance - Insight",
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

# --- Formatting helpers ---
def fmt_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%d/%m/%Y")

def _fmt_price(x, ndigits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{float(x):.{ndigits}f}"

def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{int(round(float(x))):,}"

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
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

# --- Loaders ---
def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    try: df = pd.read_excel(path)
    except: return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["date", "ngay"]: col_map[c] = "Date"
        elif lc in ["ticker", "ma"]: col_map[c] = "Ticker"
        elif lc in ["close", "closeprice"]: col_map[c] = "Close"
        elif lc in ["volume", "vol"]: col_map[c] = "Volume"
        elif lc == "open": col_map[c] = "Open"
        elif lc == "high": col_map[c] = "High"
        elif lc == "low": col_map[c] = "Low"
        elif lc in ["vma 20", "avg20vol"]: col_map[c] = "VMA20_File"
    df = df.rename(columns=col_map)
    required = ["Date", "Ticker", "Close", "Volume"]
    for r in required:
        if r not in df.columns: df[r] = np.nan
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Close", "Volume", "Open", "High", "Low", "VMA20_File"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df

def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    try: df = pd.read_excel(path)
    except: return pd.DataFrame(columns=["Ticker", "Name"])
    df.columns = [str(c).strip() for c in df.columns]
    if "Ticker" not in df.columns: df["Ticker"] = ""
    name_col = "Stock Name" if "Stock Name" in df.columns else ("Name" if "Name" in df.columns else None)
    if name_col is None: df["Name"] = ""
    else: df = df.rename(columns={name_col: "Name"})
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df[["Ticker", "Name"]].drop_duplicates()

def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    try: df = pd.read_excel(path)
    except: return pd.DataFrame(columns=["Ticker", "Target"])
    df.columns = [str(c).strip() for c in df.columns]
    if "TP (VND)" not in df.columns and "Target" in df.columns: df.rename(columns={"Target": "TP (VND)"}, inplace=True)
    out = pd.DataFrame()
    out["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip() if "Ticker" in df.columns else ""
    out["Recommendation"] = df["Recommendation"].astype(str).str.strip() if "Recommendation" in df.columns else ""
    out["Target"] = pd.to_numeric(df["TP (VND)"], errors="coerce") if "TP (VND)" in df.columns else np.nan
    out["Upside"] = pd.to_numeric(df["Upside/Downside"], errors="coerce") if "Upside/Downside" in df.columns else 0
    out["PE_2025"] = pd.to_numeric(df["2025F P/E"], errors="coerce") if "2025F P/E" in df.columns else 0
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

# ==============================================================================
# 2. FIBONACCI & TRADE PLAN ENGINE (Python Pure Logic)
# ==============================================================================

# --- Helpers from Step 2 ---
@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str
    risk_per_share: float
    reward_per_share: float

def flatten_fib_for_tradeplan(dual_fib: Dict[str, Any]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    short = (dual_fib or {}).get("auto_short", {}) or {}
    for k, v in (short.get("retracements_from_low", {}) or {}).items(): res[f"Retr_L_{k}"] = float(v)
    for k, v in (short.get("retracements_from_high", {}) or {}).items(): res[f"Retr_H_{k}"] = float(v)
    for k, v in (short.get("extensions_from_low", {}) or {}).items(): res[f"Ext_L_{k}"] = float(v)
    for k, v in (short.get("extensions_from_high", {}) or {}).items(): res[f"Ext_H_{k}"] = float(v)
    res["hi"] = _safe_float(short.get("swing_high", np.nan))
    res["lo"] = _safe_float(short.get("swing_low", np.nan))
    return res

def fib_support_resistance(fib_flat: Dict[str, float], close: float) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    levels: List[Tuple[str, float]] = []
    if not fib_flat or _isnan(close) or close <= 0: return [], []
    for k, v in fib_flat.items():
        if k in ("hi", "lo"): continue
        if _isnan(v) or float(v) <= 0: continue
        levels.append((k, float(v)))
    resist = [x for x in levels if x[1] > close]
    supp   = [x for x in levels if x[1] < close]
    resist.sort(key=lambda x: x[1])
    supp.sort(key=lambda x: x[1], reverse=True)
    return resist, supp

def fib_zones(levels_list: List[Tuple[str, float]], close: float, merge_threshold_pct: float = 0.30) -> List[Dict[str, Any]]:
    if not levels_list or _isnan(close) or close <= 0: return []
    thr = (merge_threshold_pct / 100.0) * close
    zones: List[Dict[str, Any]] = []
    cur = {"low": levels_list[0][1], "high": levels_list[0][1], "labels": [levels_list[0][0]]}
    for lab, lv in levels_list[1:]:
        if abs(lv - cur["low"]) <= thr or abs(lv - cur["high"]) <= thr:
            cur["low"] = min(cur["low"], lv)
            cur["high"] = max(cur["high"], lv)
            cur["labels"].append(lab)
        else:
            zones.append(cur)
            cur = {"low": lv, "high": lv, "labels": [lab]}
    zones.append(cur)
    out: List[Dict[str, Any]] = []
    for z in zones:
        center = (z["low"] + z["high"]) / 2.0
        out.append({"low": float(z["low"]), "high": float(z["high"]), "center": float(center), "labels": z["labels"]})
    return out

def nearest_zone_above(zones: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return sorted(zones, key=lambda z: z["center"])[0] if zones else None

def nearest_zone_below(zones: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return sorted(zones, key=lambda z: z["center"], reverse=True)[0] if zones else None

def _compute_rr_val(entry: float, stop: float, tp: float) -> float:
    if any(_isnan(x) for x in [entry, stop, tp]): return np.nan
    risk = entry - stop
    reward = tp - entry
    if risk <= 0: return np.nan
    return reward / risk

def _probability_label(df: pd.DataFrame) -> Tuple[str, str]:
    last = df.iloc[-1]
    rsi = _safe_float(last.get("RSI14", np.nan))
    macd_v = _safe_float(last.get("MACD", np.nan))
    sig  = _safe_float(last.get("MACDSignal", np.nan))
    ma50 = _safe_float(last.get("MA50", np.nan))
    close = _safe_float(last.get("Close", np.nan))
    align = (not _isnan(rsi)) and (not _isnan(macd_v)) and (not _isnan(sig)) and (rsi >= 55.0) and (macd_v > sig)
    breakout_prob = "Cao" if align else "TB"
    pullback_prob = "Trung-Cao" if (align and (not _isnan(ma50)) and (not _isnan(close)) and close > ma50) else "TB"
    return breakout_prob, pullback_prob

def build_trade_plan_breakout_pullback(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    if df is None or df.empty: return {}
    df = df.sort_values("Date").reset_index(drop=True)
    last = df.iloc[-1]
    close_s = df["Close"].astype(float)
    last_close = _safe_float(last.get("Close", np.nan))
    if _isnan(last_close): return {}
    ma20 = _safe_float(last.get("MA20", np.nan))
    ma50 = _safe_float(last.get("MA50", np.nan))
    
    fib_flat = flatten_fib_for_tradeplan(dual_fib)
    resist, supp = fib_support_resistance(fib_flat, last_close)
    res_z = fib_zones(resist, last_close)
    sup_z = fib_zones(supp, last_close)
    near_res = nearest_zone_above(res_z)
    near_sup = nearest_zone_below(sup_z)
    hi20 = _safe_float(close_s.tail(20).max(), np.nan)
    breakout_prob, pullback_prob = _probability_label(df)

    # 1. Breakout
    breakout_base = near_res["high"] if near_res else hi20
    breakout_entry = _round_price(_safe_float(breakout_base) * 1.002, 2)
    if near_sup:
        cand1 = near_sup["low"] * 0.998
        cand2 = (ma20 * 0.995) if (not _isnan(ma20)) else cand1
        breakout_stop = _round_price(min(cand1, cand2), 2)
    else:
        breakout_stop = _round_price((ma20 * 0.992) if (not _isnan(ma20)) else (breakout_entry * 0.98), 2)
    
    next_res = res_z[1] if (res_z and len(res_z) >= 2) else None
    fib_hi = _safe_float(fib_flat.get("hi", np.nan))
    if next_res: breakout_tp = _round_price(_safe_float(next_res["center"]), 2)
    elif (not _isnan(fib_hi)) and fib_hi > breakout_entry: breakout_tp = _round_price(fib_hi, 2)
    else: breakout_tp = _round_price(breakout_entry * 1.06, 2)
    
    breakout_rr = _compute_rr_val(breakout_entry, breakout_stop, breakout_tp)
    
    # 2. Pullback
    pullback_entry = near_sup["center"] if near_sup else (ma20 if not _isnan(ma20) else last_close)
    pullback_entry = _round_price(_safe_float(pullback_entry), 2)
    if near_sup: pullback_stop = near_sup["low"] * 0.997
    else: pullback_stop = (ma50 * 0.985) if (not _isnan(ma50)) else (pullback_entry * 0.97)
    if not _isnan(ma50): pullback_stop = min(pullback_stop, ma50 * 0.99)
    pullback_stop = _round_price(_safe_float(pullback_stop), 2)
    
    pullback_tp = near_res["center"] if near_res else hi20
    pullback_tp = _round_price(_safe_float(max(pullback_tp, pullback_entry * 1.02)), 2)
    pullback_rr = _compute_rr_val(pullback_entry, pullback_stop, pullback_tp)

    return {
        "Breakout": TradeSetup("Breakout", breakout_entry, breakout_stop, breakout_tp, breakout_rr, breakout_prob, 0, 0),
        "Pullback": TradeSetup("Pullback", pullback_entry, pullback_stop, pullback_tp, pullback_rr, pullback_prob, 0, 0)
    }

def weighted_rr(setups: Dict[str, TradeSetup], weights=None) -> Tuple[float, str]:
    if weights is None: weights = {"Cao": 0.60, "Trung-Cao": 0.40, "TB": 0.25, "Thấp": 0.15}
    total_w, total = 0.0, 0.0
    best, best_metric = "", -1.0
    for name, s in (setups or {}).items():
        rr = _safe_float(s.rr, np.nan)
        if _isnan(rr): continue
        w = float(weights.get(s.probability, 0.20))
        total_w += w
        total += w * rr
        metric = w * rr
        if metric > best_metric: best_metric, best = metric, name
    if total_w <= 0: return np.nan, ""
    return total / total_w, best

# --- Fibo Core Calculation ---
def _compute_atr20(df):
    high, low, cp = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([(high-low).abs(), (high-cp).abs(), (low-cp).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=20).mean()

def _fib_levels_from_range(low, high):
    rng = high - low
    if rng <= 0: return {}
    return dict(
        retracements_from_low={0.236: high-0.236*rng, 0.382: high-0.382*rng, 0.5: high-0.5*rng, 0.618: high-0.618*rng},
        retracements_from_high={0.236: low+0.236*rng, 0.382: low+0.382*rng, 0.5: low+0.5*rng, 0.618: low+0.618*rng},
        extensions_from_low={1.272: high+0.272*rng, 1.618: high+0.618*rng},
        extensions_from_high={1.272: low-0.272*rng, 1.618: low-0.618*rng}
    )

def compute_dual_fibonacci(df: pd.DataFrame) -> Dict[str, Any]:
    # Auto Short
    atr20 = _compute_atr20(df)
    vol = float(atr20.iloc[-1] / df['Close'].iloc[-1]) if pd.notna(atr20.iloc[-1]) else 0.02
    L = 60 if vol*100 >= 3 else (75 if vol*100 >= 2 else 90)
    L = min(L, len(df))
    win = df.tail(L)
    s_hi, s_lo = float(win['High'].max()), float(win['Low'].min())
    auto_short = {'frame': f'AUTO_{L}D', 'vol': vol, 'window_L': L, 'swing_high': s_hi, 'swing_low': s_lo, **_fib_levels_from_range(s_lo, s_hi)}
    
    # Fixed Long
    L2 = min(250, len(df))
    win2 = df.tail(L2)
    l_hi, l_lo = float(win2['High'].max()), float(win2['Low'].min())
    fixed_long = {'frame': 'FIXED_250D', 'window_L': L2, 'swing_high': l_hi, 'swing_low': l_lo, **_fib_levels_from_range(l_lo, l_hi)}
    
    return {'auto_short': auto_short, 'fixed_long': fixed_long}

# --- 12 Scenarios Logic ---
def classify_12_scenarios(last):
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    rsi, macd_v, sig = last["RSI14"], last["MACD"], last["MACDSignal"]
    if any(np.isnan(x) for x in [c, ma20, ma50, ma200, rsi]): return "Insufficient data"
    
    bull_stack = ma20 > ma50 > ma200
    bear_stack = ma20 < ma50 < ma200
    macd_bull = macd_v > sig
    
    if bull_stack and c > ma20 and rsi >= 55 and macd_bull: return "1) Uptrend – Breakout – High Volume Confirmation"
    if bull_stack and c > ma200 and c < ma20: return "2) Uptrend Pullback"
    if c > ma200 and not bull_stack: return "3) Early Uptrend / Accumulation"
    if not c > ma200 and bear_stack: return "7) Strong Downtrend"
    return "12) Neutral / Sideways"

# ==============================================================================
# 3. CORE ANALYSIS & OUTPUT GENERATION
# ==============================================================================

def analyze_ticker_logic(ticker: str):
    ticker = ticker.upper().strip()
    df_all = load_price_vol(PRICE_VOL_PATH)
    if df_all.empty: return {"Error": "Lỗi file Price_Vol.xlsx"}
    df = df_all[df_all["Ticker"] == ticker].copy().sort_values("Date").reset_index(drop=True)
    if df.empty: return {"Error": f"Mã {ticker} không tìm thấy"}

    # Calculate Indicators
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI14"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    
    dual_fib = compute_dual_fibonacci(df)
    last = df.iloc[-1].to_dict()
    
    # Calculate Scores & Scenarios
    scenario = classify_12_scenarios(last)
    
    # Conviction Score (Simplified logic to match example)
    score = 5.0
    if last["Close"] > last["MA200"]: score += 2
    if last["RSI14"] > 50: score += 1.5
    if last["Volume"] > last["Avg20Vol"]: score += 1
    if last["MACD"] > last["MACDSignal"]: score += 0.5
    score = min(10.0, score)

    # Fundamental (Excel)
    hsc = load_hsc_targets(HSC_TARGET_PATH)
    fund_row = hsc[hsc["Ticker"] == ticker].iloc[0].to_dict() if not hsc[hsc["Ticker"] == ticker].empty else {}

    # Trade Plan (Python Pure)
    setups = build_trade_plan_breakout_pullback(df, dual_fib)
    w_rr, pref_scen = weighted_rr(setups)

    # Prepare Data for View
    return {
        "Ticker": ticker,
        "Last": last,
        "DualFib": dual_fib,
        "Scenario": scenario,
        "Score": score,
        "Fundamental": fund_row,
        "TradePlan": setups,
        "RR": {"Weighted": w_rr, "Preferred": pref_scen},
        "DataFrame": df # Pass df for extra checks if needed
    }

def render_inception_markdown(data: dict) -> str:
    if "Error" in data: return f"❌ {data['Error']}"
    
    last = data["Last"]
    tick = data["Ticker"]
    fib = data["DualFib"]["auto_short"]
    fund = data["Fundamental"]
    
    # A. Indicator Snapshot Logic
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    rsi, macd_v, sig = last["RSI14"], last["MACD"], last["MACDSignal"]
    vol, avg_vol = last["Volume"], last["Avg20Vol"]
    
    # A1. MA Trend Text
    if c > ma20 and c > ma50 and c > ma200 and ma20 > ma50 > ma200:
        ma_text = "Giá nằm trên toàn bộ MA20/50/200, đồng thời MA20 > MA50 > MA200 → cấu trúc uptrend chuẩn, xu hướng trung–dài hạn rất tích cực."
    elif c > ma200:
        ma_text = "Giá nằm trên MA200 nhưng các đường ngắn hạn đang giằng co → Xu hướng dài hạn Tăng, ngắn hạn Tích lũy."
    else:
        ma_text = "Giá nằm dưới MA200 → Xu hướng dài hạn tiêu cực."

    # A2. RSI Text
    if rsi > 65: rsi_text = f"RSI ~{rsi:.0f} cho thấy động lượng mạnh, tiệm cận vùng quá mua."
    elif rsi < 35: rsi_text = f"RSI ~{rsi:.0f} ở vùng quá bán, áp lực bán có thể sớm suy yếu."
    else: rsi_text = f"RSI ~{rsi:.0f} ở vùng trung tính, cân bằng cung cầu."

    # A3. MACD Text
    macd_text = "MACD > Signal, histogram mở rộng → xác nhận xung lực tăng." if macd_v > sig else "MACD < Signal → động lượng tăng suy yếu hoặc đảo chiều giảm."

    # A4. Bias Matrix
    bias_text = "Bullish continuation" if (rsi > 50 and macd_v > sig) else ("Bearish pressure" if (rsi < 50 and macd_v < sig) else "Neutral / Mixed")

    # A5. Fibo Text
    retr_l = fib.get("retracements_from_low", {})
    f382 = _fmt_price(retr_l.get(0.382))
    f50 = _fmt_price(retr_l.get(0.5))
    f618 = _fmt_price(retr_l.get(0.618))
    fibo_text = f"Fibo 38.2%: ~{f382}\nFibo 50%: ~{f50}\nFibo 61.8%: ~{f618}\n→ Các nhịp điều chỉnh nếu có, {f50}–{f382} là vùng hỗ trợ."

    # A6. Vol Text
    vol_text = "Khối lượng cao hơn trung bình 20 phiên → dòng tiền chủ động." if vol > avg_vol else "Khối lượng thấp hơn trung bình → áp lực cung cầu yếu."

    # Markdown Construction
    md = []
    md.append(f"# INSIGHT")
    md.append(f"## A. Indicator Snapshot")
    md.append(f"Close: {_fmt_price(c)}\nVolume: ~{_fmt_int(vol)}\nAvg20 Vol: ~{_fmt_int(avg_vol)}")
    md.append(f"MA20 / MA50 / MA200: {_fmt_price(ma20)} / {_fmt_price(ma50)} / {_fmt_price(ma200)}")
    md.append(f"RSI (14): ~{rsi:.0f}\nMACD: {'Dương' if macd_v > 0 else 'Âm'}, {'MACD > Sig' if macd_v > sig else 'MACD < Sig'}")

    md.append(f"### 1. MA Trend Analysis\n{ma_text}")
    md.append(f"### 2. RSI Analysis\n{rsi_text}")
    md.append(f"### 3. MACD Analysis\n{macd_text}")
    md.append(f"### 4. RSI + MACD Bias Matrix\n{bias_text}")
    md.append(f"### 5. Fibonacci Levels\n(Đo từ đáy trung hạn gần nhất lên đỉnh mới)\n{fibo_text}")
    md.append(f"### 6. Volume & Price Action\n{vol_text}")
    md.append(f"### 7. 12-Scenario Classification\n{data['Scenario']}")
    md.append(f"### 8. Master Integration + Conviction Score\nConviction score: {data['Score']:.1f} / 10")

    md.append(f"## B. Fundamental Analysis Summary")
    if fund:
        md.append(f"Dữ liệu từ HSC/Target Excel:")
        md.append(f"- **Khuyến nghị:** {fund.get('Recommendation', 'N/A')}")
        md.append(f"- **Giá mục tiêu:** {_fmt_price(fund.get('Target'))} (Upside: {fund.get('Upside',0)*100:.1f}%)")
        md.append(f"- **P/E 2025F:** {fund.get('PE_2025', 'N/A')}")
    else:
        md.append("Chưa có dữ liệu Fundamental trong file Excel.")

    # C. Trade Plan Table
    md.append(f"## C. Suggestions (Trade Plan)")
    md.append("| Chiến lược | Entry đề xuất | Stop-loss | Take-profit |")
    md.append("|---|---|---|---|")
    for k, s in data["TradePlan"].items():
        md.append(f"| {s.name} | {_fmt_price(s.entry)} | {_fmt_price(s.stop)} | {_fmt_price(s.tp)} |")

    # D. RR Table
    md.append(f"## D. R:R (Risk–Reward) Simulation")
    md.append("| Kịch bản | Entry | Stop-loss | Take-profit | Xác suất | R:R ước tính |")
    md.append("|---|---|---|---|---|---|")
    for k, s in data["TradePlan"].items():
        md.append(f"| {s.name} | {_fmt_price(s.entry)} | {_fmt_price(s.stop)} | {_fmt_price(s.tp)} | {s.probability} | ~{s.rr:.1f} |")
    md.append(f"| Invalidation | — | — | — | 10% | — |")
    
    w_rr = data["RR"]["Weighted"]
    pref = data["RR"]["Preferred"]
    md.append(f"\n**Tổng hợp:** R:R trung bình có trọng số ≈ {w_rr:.1f} lần")
    md.append(f"**Kịch bản ưu tiên:** {pref}")
    md.append("> *Chỉ nhằm mục đích cung cấp thông tin — không phải khuyến nghị đầu tư.*")

    return "\n".join(md)

# ==========================================
# 4. GIAO DIỆN WEB STREAMLIT
# ==========================================
st.markdown("""
<style>
.big-font {font-size:30px !important; font-weight: bold; color: #2E86C1;}
.sub-text {font-size:16px !important; color: #333; margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">INSIGHT ENGINE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Đơn Giản là đỉnh cao của Phức tạp</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    user_key = st.text_input("🔑 Mã VIP:", type="password")
    ticker_input = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    run_btn = st.button("PHÂN TÍCH", type="primary")

if run_btn:
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    else:
        if VALID_KEYS[user_key]["quota"] <= 0:
             st.error("⛔ Hết lượt sử dụng.")
        else:
            VALID_KEYS[user_key]["quota"] -= 1
            with st.spinner(f"Đang xử lý {ticker_input}..."):
                # 1. Chạy Logic Python
                data_result = analyze_ticker_logic(ticker_input)
                
                # 2. Render Markdown (Inception Style)
                final_md = render_inception_markdown(data_result)
                
                # 3. Hiển thị
                st.markdown(final_md)
                
                # 4. AI Summary (Optional - Chỉ tóm tắt, không tính toán)
                if api_key and "Error" not in data_result:
                    st.divider()
                    st.info("🤖 **AI Executive Summary:**")
                    try:
                        client = OpenAI(api_key=api_key)
                        prompt = f"""
                        Tóm tắt ngắn gọn (dưới 200 từ) báo cáo sau cho nhà đầu tư:
                        {final_md}
                        """
                        res = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}]
                        )
                        st.write(res.choices[0].message.content)
                    except: pass