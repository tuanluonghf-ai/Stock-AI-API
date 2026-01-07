"""Scenario, conviction, master score, RR simulation.

No Streamlit dependency.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .helpers import _safe_float
from .tradeplan import TradeSetup

def compute_conviction_pack(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    
    score = 5.0
    components = []
    notes = []
    
    # Rule 1: Above MA200 (+2)
    hit1 = bool(pd.notna(c) and pd.notna(ma200) and c > ma200)
    if hit1:
        score += 2.0
    components.append({"Key": "PriceAboveMA200", "Hit": hit1, "Weight": 2.0})
    notes.append(f"PriceAboveMA200={hit1} (+2.0)")
    
    # Rule 2: RSI > 55 (+1)
    hit2 = bool(pd.notna(rsi) and rsi > 55)
    if hit2:
        score += 1.0
    components.append({"Key": "RSIAbove55", "Hit": hit2, "Weight": 1.0})
    notes.append(f"RSIAbove55={hit2} (+1.0)")
        
    # Rule 3: Volume > Avg20Vol (+1)
    vol_ratio = (vol / avg) if (pd.notna(vol) and pd.notna(avg) and avg != 0) else np.nan
    hit3 = bool(pd.notna(vol_ratio) and vol_ratio > 1.0)
    if hit3:
        score += 1.0
    components.append({"Key": "VolumeAboveAvg20", "Hit": hit3, "Weight": 1.0})
    notes.append(f"VolumeAboveAvg20={hit3} (+1.0)")
            
    # Rule 4: MACD > Signal (+0.5)
    hit4 = bool(pd.notna(macd_v) and pd.notna(sig) and macd_v > sig)
    if hit4:
        score += 0.5
    components.append({"Key": "MACDAboveSignal", "Hit": hit4, "Weight": 0.5})
    notes.append(f"MACDAboveSignal={hit4} (+0.5)")
        
    score = float(min(10.0, score))
    return {
        "Score": round(score, 2),
        "Components": components,
        "Notes": notes, # Now fully neutral tags
        "Facts": {
            "Close": c, "MA200": ma200, "RSI": rsi,
            "VolRatio": vol_ratio, "MACD": macd_v, "Signal": sig
        }
    }

def compute_conviction(last: pd.Series) -> float:
    pack = compute_conviction_pack(last)
    return _safe_float(pack.get("Score"), default=5.0)

# ============================================================
# 8. TRADE PLAN LOGIC (Step 9+10 / v5.6 patch for Step 8)
# - Remove fixed % buffer

def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20: return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200: return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200: return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# --- STEP 6B (v5.4): SCENARIO 12 NEUTRAL ("Extended" -> "RSI_70Plus") ---
def classify_scenario12(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))
    
    rules_hit = []
    
    # 1. Trend
    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"
            rules_hit.append("Trend=Up (Close>=MA50>=MA200)")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"
            rules_hit.append("Trend=Down (Close<MA50<MA200)")
        else:
            trend = "Neutral"
            rules_hit.append("Trend=Neutral (mixed MA structure)")
    else:
        rules_hit.append("Trend=N/A (missing MA50/MA200/Close)")
        
    # 2. Momentum (Alignment)
    mom = "Mixed"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        rsi_up = (rsi >= 55)
        rsi_down = (rsi <= 45)
        rsi_pos_ext = (rsi >= 70)
        macd_up = (macd_v >= sig)
        macd_down = (macd_v < sig)
        
        if rsi_pos_ext:
            mom = "RSI_70Plus" # 6B Change
            rules_hit.append("Momentum=RSI_70Plus (RSI>=70)")
        elif rsi_up and macd_up:
            mom = "Aligned"
            rules_hit.append("Momentum=Aligned_Up (RSI>=55 & MACD>=Sig)")
        elif rsi_down and macd_down:
            mom = "Aligned" # Aligned Down
            rules_hit.append("Momentum=Aligned_Down (RSI<=45 & MACD<Sig)")
        elif rsi_up and macd_down:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bull but MACD Bear)")
        elif rsi_down and macd_up:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bear but MACD Bull)")
        else:
            mom = "Mixed"
            rules_hit.append("Momentum=Mixed (Between zones)")
    else:
        rules_hit.append("Momentum=N/A")
        
    # 3. Volume
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")
    else:
        rules_hit.append("Volume=N/A (missing Volume/Avg20Vol)")
        
    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Aligned": 0, "Mixed": 1, "Counter": 2, "RSI_70Plus": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1
    
    name_map = {
        ("Up", "Aligned"):   "S1 – Uptrend + Momentum Aligned",
        ("Up", "Mixed"):     "S2 – Uptrend + Momentum Mixed",
        ("Up", "Counter"):   "S3 – Uptrend + Momentum Counter",
        ("Up", "RSI_70Plus"):  "S4 – Uptrend + RSI 70+",
        ("Neutral", "Aligned"):  "S5 – Range + Momentum Aligned",
        ("Neutral", "Mixed"):    "S6 – Range + Balanced/Mixed",
        ("Neutral", "Counter"):  "S7 – Range + Momentum Counter",
        ("Neutral", "RSI_70Plus"): "S8 – Range + RSI 70+",
        ("Down", "Aligned"):  "S9 – Downtrend + Momentum Aligned",
        ("Down", "Mixed"):    "S10 – Downtrend + Momentum Mixed",
        ("Down", "Counter"):  "S11 – Downtrend + Momentum Counter",
        ("Down", "RSI_70Plus"): "S12 – Downtrend + RSI 70+",
    }
    
    return {
        "Code": int(code),
        "TrendRegime": trend,
        "MomentumRegime": mom,
        "VolumeRegime": vol_reg,
        "Name": name_map.get((trend, mom), "Scenario – N/A"),
        "RulesHit": rules_hit,
        "Flags": {
            "TrendUp": (trend=="Up"),
            "TrendDown": (trend=="Down"),
            "MomAligned": (mom=="Aligned"),
            "Mom70Plus": (mom=="RSI_70Plus"),
            "VolHigh": (vol_reg=="High")
        }
    }

# --- STEP 5B: MASTER SCORE (FACT ONLY) ---
# --- STEP 5B: MASTER SCORE (FACT ONLY | FUNDAMENTAL-FREE) ---

def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Fundamental (Target/Upside/Recommendation) MUST NOT affect any score computation.
    - This MasterScore is 100% technical + tradeplan RR only.
    """

    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    comps = {}
    notes = []  # neutral condition tags only

    # 1) Trend (pure MA structure)
    trend = 0.0
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            trend = 2.0
            notes.append("TrendTag=Structure_Up_Strong")
        elif (c >= ma200):
            trend = 1.2
            notes.append("TrendTag=Structure_Up_Moderate")
        else:
            trend = 0.4
            notes.append("TrendTag=Structure_Down_Weak")
    comps["Trend"] = trend

    # 2) Momentum (RSI + MACD relation only)
    mom = 0.0
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = 2.0
            notes.append("MomTag=Aligned_Bullish")
        elif (rsi <= 45) and (macd_v < sig):
            mom = 0.4
            notes.append("MomTag=Aligned_Bearish")
        else:
            mom = 1.1
            notes.append("MomTag=Mixed")
    comps["Momentum"] = mom

    # 3) Volume (relative to Avg20Vol only)
    vcomp = 0.0
    if pd.notna(vol) and pd.notna(avg_vol) and avg_vol != 0:
        if vol > avg_vol:
            vcomp = 1.6
            notes.append("VolTag=Above_Avg")
        else:
            vcomp = 0.9
            notes.append("VolTag=Below_Avg")
    comps["Volume"] = vcomp

    # 4) Fibonacci (position vs key bands, no fundamental)
    fibc = 0.0
    try:
        s_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
        l_lv = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))

        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            if c >= s_618:
                fibc += 1.2
            elif c >= s_382:
                fibc += 0.8
            else:
                fibc += 0.4

        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            if c >= l_618:
                fibc += 0.8
            elif c >= l_382:
                fibc += 0.5
            else:
                fibc += 0.2
    except:
        fibc = 0.0
    comps["Fibonacci"] = fibc

    # 5) RR Quality (from trade plans only)
    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(getattr(s, "rr", np.nan))]
        best_rr = max(rrs) if rrs else np.nan

    rrcomp = 0.0
    if pd.notna(best_rr):
        if best_rr >= 4.0:
            rrcomp = 2.0
            notes.append("RRTag=Excellent_Gt4")
        elif best_rr >= 3.0:
            rrcomp = 1.5
            notes.append("RRTag=Good_Gt3")
        else:
            rrcomp = 1.0
            notes.append("RRTag=Normal_Lt3")
    comps["RRQuality"] = rrcomp

    total = float(sum(comps.values()))

    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": "N/A",            # Unlocked
        "PositionSizing": "N/A",  # Unlocked
        "Notes": notes
    }


# ============================================================
# 9D. RISK–REWARD SIMULATION PACK
# (moved from app to core to keep a single source of truth)
# ============================================================
def build_rr_sim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    """Build a JSON-safe RR simulation pack from computed trade plans.

    Notes
    -----
    - Only uses technical/tradeplan fields; no fundamental dependency.
    - Skips setups marked as Invalid.
    """
    rows = []
    best_rr = np.nan
    for k, s in (trade_plans or {}).items():
        status = getattr(s, "status", "Watch") or "Watch"
        if status == "Invalid":
            continue

        entry = _safe_float(getattr(s, "entry", np.nan))
        stop = _safe_float(getattr(s, "stop", np.nan))
        tp = _safe_float(getattr(s, "tp", np.nan))
        rr = _safe_float(getattr(s, "rr", np.nan))

        risk_pct = (
            ((entry - stop) / entry * 100)
            if (pd.notna(entry) and pd.notna(stop) and entry != 0)
            else np.nan
        )
        reward_pct = (
            ((tp - entry) / entry * 100)
            if (pd.notna(tp) and pd.notna(entry) and entry != 0)
            else np.nan
        )

        rows.append(
            {
                "Setup": k,
                "Entry": entry,
                "Stop": stop,
                "TP": tp,
                "RR": rr,
                "RiskPct": risk_pct,
                "RewardPct": reward_pct,
                "Confidence (Tech)": getattr(s, "probability", np.nan),
                "Status": status,
                "ReasonTags": list(getattr(s, "reason_tags", []) or []),
            }
        )

        if pd.notna(rr):
            best_rr = rr if pd.isna(best_rr) else max(best_rr, rr)

    return {"Setups": rows, "BestRR": best_rr if pd.notna(best_rr) else np.nan}
# ============================================================
# 9D. RISK–REWARD SIMULATION PACK
# ============================================================
def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20: return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200: return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200: return "Downtrend – Weak Phase"
    return "Neutral / Sideways"

# --- STEP 6B (v5.4): SCENARIO 12 NEUTRAL ("Extended" -> "RSI_70Plus") ---
def classify_scenario12(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))
    
    rules_hit = []
    
    # 1. Trend
    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"
            rules_hit.append("Trend=Up (Close>=MA50>=MA200)")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"
            rules_hit.append("Trend=Down (Close<MA50<MA200)")
        else:
            trend = "Neutral"
            rules_hit.append("Trend=Neutral (mixed MA structure)")
    else:
        rules_hit.append("Trend=N/A (missing MA50/MA200/Close)")
        
    # 2. Momentum (Alignment)
    mom = "Mixed"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        rsi_up = (rsi >= 55)
        rsi_down = (rsi <= 45)
        rsi_pos_ext = (rsi >= 70)
        macd_up = (macd_v >= sig)
        macd_down = (macd_v < sig)
        
        if rsi_pos_ext:
            mom = "RSI_70Plus" # 6B Change
            rules_hit.append("Momentum=RSI_70Plus (RSI>=70)")
        elif rsi_up and macd_up:
            mom = "Aligned"
            rules_hit.append("Momentum=Aligned_Up (RSI>=55 & MACD>=Sig)")
        elif rsi_down and macd_down:
            mom = "Aligned" # Aligned Down
            rules_hit.append("Momentum=Aligned_Down (RSI<=45 & MACD<Sig)")
        elif rsi_up and macd_down:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bull but MACD Bear)")
        elif rsi_down and macd_up:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bear but MACD Bull)")
        else:
            mom = "Mixed"
            rules_hit.append("Momentum=Mixed (Between zones)")
    else:
        rules_hit.append("Momentum=N/A")
        
    # 3. Volume
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")
    else:
        rules_hit.append("Volume=N/A (missing Volume/Avg20Vol)")
        
    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Aligned": 0, "Mixed": 1, "Counter": 2, "RSI_70Plus": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1
    
    name_map = {
        ("Up", "Aligned"):   "S1 – Uptrend + Momentum Aligned",
        ("Up", "Mixed"):     "S2 – Uptrend + Momentum Mixed",
        ("Up", "Counter"):   "S3 – Uptrend + Momentum Counter",
        ("Up", "RSI_70Plus"):  "S4 – Uptrend + RSI 70+",
        ("Neutral", "Aligned"):  "S5 – Range + Momentum Aligned",
        ("Neutral", "Mixed"):    "S6 – Range + Balanced/Mixed",
        ("Neutral", "Counter"):  "S7 – Range + Momentum Counter",
        ("Neutral", "RSI_70Plus"): "S8 – Range + RSI 70+",
        ("Down", "Aligned"):  "S9 – Downtrend + Momentum Aligned",
        ("Down", "Mixed"):    "S10 – Downtrend + Momentum Mixed",
        ("Down", "Counter"):  "S11 – Downtrend + Momentum Counter",
        ("Down", "RSI_70Plus"): "S12 – Downtrend + RSI 70+",
    }
    
    return {
        "Code": int(code),
        "TrendRegime": trend,
        "MomentumRegime": mom,
        "VolumeRegime": vol_reg,
        "Name": name_map.get((trend, mom), "Scenario – N/A"),
        "RulesHit": rules_hit,
        "Flags": {
            "TrendUp": (trend=="Up"),
            "TrendDown": (trend=="Down"),
            "MomAligned": (mom=="Aligned"),
            "Mom70Plus": (mom=="RSI_70Plus"),
            "VolHigh": (vol_reg=="High")
        }
    }

# --- STEP 5B: MASTER SCORE (FACT ONLY) ---
# --- STEP 5B: MASTER SCORE (FACT ONLY | FUNDAMENTAL-FREE) ---

def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Fundamental (Target/Upside/Recommendation) MUST NOT affect any score computation.
    - This MasterScore is 100% technical + tradeplan RR only.
    """

    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    comps = {}
    notes = []  # neutral condition tags only

    # 1) Trend (pure MA structure)
    trend = 0.0
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            trend = 2.0
            notes.append("TrendTag=Structure_Up_Strong")
        elif (c >= ma200):
            trend = 1.2
            notes.append("TrendTag=Structure_Up_Moderate")
        else:
            trend = 0.4
            notes.append("TrendTag=Structure_Down_Weak")
    comps["Trend"] = trend

    # 2) Momentum (RSI + MACD relation only)
    mom = 0.0
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = 2.0
            notes.append("MomTag=Aligned_Bullish")
        elif (rsi <= 45) and (macd_v < sig):
            mom = 0.4
            notes.append("MomTag=Aligned_Bearish")
        else:
            mom = 1.1
            notes.append("MomTag=Mixed")
    comps["Momentum"] = mom

    # 3) Volume (relative to Avg20Vol only)
    vcomp = 0.0
    if pd.notna(vol) and pd.notna(avg_vol) and avg_vol != 0:
        if vol > avg_vol:
            vcomp = 1.6
            notes.append("VolTag=Above_Avg")
        else:
            vcomp = 0.9
            notes.append("VolTag=Below_Avg")
    comps["Volume"] = vcomp

    # 4) Fibonacci (position vs key bands, no fundamental)
    fibc = 0.0
    try:
        s_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
        l_lv = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))

        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            if c >= s_618:
                fibc += 1.2
            elif c >= s_382:
                fibc += 0.8
            else:
                fibc += 0.4

        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            if c >= l_618:
                fibc += 0.8
            elif c >= l_382:
                fibc += 0.5
            else:
                fibc += 0.2
    except:
        fibc = 0.0
    comps["Fibonacci"] = fibc

    # 5) RR Quality (from trade plans only)
    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(getattr(s, "rr", np.nan))]
        best_rr = max(rrs) if rrs else np.nan

    rrcomp = 0.0
    if pd.notna(best_rr):
        if best_rr >= 4.0:
            rrcomp = 2.0
            notes.append("RRTag=Excellent_Gt4")
        elif best_rr >= 3.0:
            rrcomp = 1.5
            notes.append("RRTag=Good_Gt3")
        else:
            rrcomp = 1.0
            notes.append("RRTag=Normal_Lt3")
    comps["RRQuality"] = rrcomp

    total = float(sum(comps.values()))

    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": "N/A",            # Unlocked
        "PositionSizing": "N/A",  # Unlocked
        "Notes": notes
    }
# ============================================================
# 9D. RISK–REWARD SIMULATION PACK
# ============================================================
