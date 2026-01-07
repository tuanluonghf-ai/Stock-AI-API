from __future__ import annotations

"""Dashboard summary pack builder.

This module produces a compact, renderer-friendly summary contract that
prevents drift between the Executive Snapshot dashboard and the detailed
sections below.

Design constraints:
- Streamlit-free (core only).
- Must NOT modify engine math/thresholds; it only summarizes existing packs.
- Output is JSON-safe and stable.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .helpers import _safe_text, _safe_float
from .policy import get_class_policy_hint_line


def _pretty_level_label(level_type: str, side: str = "overhead") -> str:
    """Human-friendly label for a structural level type.

    Kept minimal and deterministic so core packs can reference structural levels
    without importing UI renderers.
    """
    t = (level_type or "").strip().upper()
    s = (side or "overhead").strip().lower()

    # Moving averages
    if t == "MA200":
        return "MA200 (structural)"
    if t == "MA50":
        return "MA50 (structural-lite)"
    if t == "MA20":
        return "MA20 (tactical)"

    # Weekly/Daily pivots:
    # A prior LOW can become resistance after breakdown; a prior HIGH can become support after reclaim.
    if t == "WEEKLY_SWING_LOW":
        return "Weekly pivot (prior LOW — now resistance)" if s == "overhead" else "Weekly pivot (prior LOW)"
    if t == "WEEKLY_SWING_HIGH":
        return "Weekly pivot (prior HIGH)" if s == "support" else "Weekly pivot (prior HIGH — resistance)"
    if t == "DAILY_SWING_LOW":
        return "Daily pivot (prior LOW — now resistance)" if s == "overhead" else "Daily pivot (prior LOW)"
    if t == "DAILY_SWING_HIGH":
        return "Daily pivot (prior HIGH)" if s == "support" else "Daily pivot (prior HIGH — resistance)"

    # Fibonacci (keep concise)
    if t.startswith("FIB_"):
        tt = t.replace("FIB_FIB_", "FIB_")
        tt = tt.replace("FIB_SHORT_", "Fib short ").replace("FIB_LONG_", "Fib long ")
        tt = tt.replace("_", " ")
        return tt

    return t or "N/A"


def compute_dashboard_summary_pack_v1(
    analysis_pack: Dict[str, Any],
    character_pack: Dict[str, Any],
    gate_status: str = "",
) -> Dict[str, Any]:
    """Compute a compact DashboardSummaryPack for Executive Snapshot rendering.

    Purpose:
      - Avoid Dashboard vs Detail drift by reading Python pre-digested packs (TradePlan/Decision/PositionManager).
      - Eliminate renderer-side branching errors by producing one stable summary contract.

    IMPORTANT:
      - This is a *summary pack* only. It must NOT change engine scoring, gating thresholds, or trade-plan math.
    """
    ap = analysis_pack or {}
    cp = character_pack or {}

    # --- core inputs ---
    primary = ap.get("PrimarySetup") or {}
    primary = primary if isinstance(primary, dict) else {}
    setup_name = _safe_text(primary.get("Name") or "N/A").strip()

    master_total = (ap.get("MasterScore") or {}).get("Total", np.nan)
    conviction = ap.get("Conviction", np.nan)

    class_name = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "N/A").strip()
    policy_hint = get_class_policy_hint_line(class_name)

    # ProTech (Now)
    protech = ap.get("ProTech") or {}
    protech = protech if isinstance(protech, dict) else {}
    ma = protech.get("MA") or {}
    ma = ma if isinstance(ma, dict) else {}
    volp = protech.get("Volume") or {}
    volp = volp if isinstance(volp, dict) else {}

    ma_reg = _safe_text(ma.get("Regime") or "N/A").strip()
    vol_reg = _safe_text(volp.get("Regime") or "N/A").strip()
    vol_ratio = (protech.get("Volume") or {}).get("Ratio")

    # Flags
    flags_list = list(cp.get("Flags") or [])
    has_near_res = any(isinstance(f, dict) and _safe_text(f.get("code")).strip() == "NearMajorResistance" for f in flags_list)
    has_near_sup = any(isinstance(f, dict) and _safe_text(f.get("code")).strip() == "NearMajorSupport" for f in flags_list)

    # Location tag (quick)
    loc_tag = "Neutral"
    if has_near_res:
        loc_tag = "Near Resistance"
    elif has_near_sup:
        loc_tag = "Near Support"
    else:
        ma_struct = ma.get("Structure") or {}
        ma_struct = ma_struct if isinstance(ma_struct, dict) else {}
        above_200 = ma_struct.get("PriceAboveMA200")
        if above_200 is True:
            loc_tag = "Above MA200"
        elif above_200 is False:
            loc_tag = "Below MA200"

    # State label (setup intent first)
    sn_l = (setup_name or "").lower()
    state_label = "Neutral"
    if "breakout" in sn_l:
        state_label = "Breakout Attempt"
    elif "pullback" in sn_l:
        state_label = "Pullback"
    else:
        if ma_reg == "Close>=MA50>=MA200":
            state_label = "Uptrend"
        elif ma_reg == "Close<MA50<MA200":
            state_label = "Downtrend"
        elif ma_reg == "MixedStructure":
            state_label = "Mixed/Choppy"

    reg_tag = f"Vol: {vol_reg}" if (vol_reg and vol_reg != "N/A") else ""
    state_capsule_line = " | ".join([x for x in [state_label, loc_tag, reg_tag] if x])

    # One-line score interpretation (dashboard-short)
    def _bucket(v: float) -> str:
        if v < 4.0:
            return "low"
        if v < 6.0:
            return "mid"
        if v < 8.0:
            return "good"
        return "high"

    insight_line = ""
    ms_v = _safe_float(master_total, default=np.nan)
    cs_v = _safe_float(conviction, default=np.nan)
    if pd.notna(ms_v) and pd.notna(cs_v):
        ms_b = _bucket(float(ms_v))
        cs_b = _bucket(float(cs_v))
        if ms_b in ("low",):
            insight_line = "Cơ hội kém hấp dẫn; ưu tiên quan sát và chờ cấu trúc/điểm vào tốt hơn."
        elif ms_b == "mid" and cs_b in ("good", "high"):
            insight_line = "Cơ hội trung tính nhưng độ tin cậy khá tốt; ưu tiên plan kỷ luật, tránh FOMO."
        elif ms_b in ("good", "high") and cs_b in ("low", "mid"):
            insight_line = "Cơ hội khá hấp dẫn nhưng độ tin cậy chưa cao; chỉ triển khai chọn lọc và chờ trigger đồng pha."
        elif ms_b in ("good", "high") and cs_b in ("good", "high"):
            insight_line = "Cơ hội hấp dẫn và độ tin cậy cao; có thể triển khai theo plan, tập trung quản trị rủi ro."
        else:
            insight_line = "Theo dõi nghiêm túc; ưu tiên đúng nhịp/điều kiện thay vì vào vội."
    gtxt = (gate_status or "").strip().upper()
    if gtxt not in ("", "N/A", "ACTIVE") and insight_line:
        insight_line = f"{insight_line} (Gate: {gtxt})"

    # --- resolve gates (prefer TradePlanPack.v1) ---
    def _status(v: Any, good: float, warn: float) -> str:
        x = _safe_float(v, default=np.nan)
        if pd.isna(x):
            return "N/A"
        if float(x) >= good:
            return "PASS"
        if float(x) >= warn:
            return "WAIT"
        return "FAIL"

    plan_status = "N/A"
    rr_plan = _safe_float(primary.get("RR"), default=np.nan)

    st_break = st_vol = st_rr = st_struct = "N/A"

    # StructureQuality (needed for next-step labels & ceiling gate)
    sq: Dict[str, Any] = {}
    try:
        sq = (
            (cp or {}).get("StructureQuality")
            or (ap or {}).get("StructureQuality")
            or (ap or {}).get("StructureQualityPack")
            or {}
        )
    except Exception:
        sq = {}
    sq = sq if isinstance(sq, dict) else {}

    tpp = ap.get("TradePlanPack") or {}
    tpp = tpp if isinstance(tpp, dict) else {}
    if _safe_text(tpp.get("schema") or "").strip() == "TradePlanPack.v1":
        pp = tpp.get("plan_primary") or {}
        pp = pp if isinstance(pp, dict) else {}
        gpack = pp.get("gates") or {}
        gpack = gpack if isinstance(gpack, dict) else {}
        plan_status = _safe_text(pp.get("state") or "N/A").strip().upper()
        rr_plan = _safe_float(pp.get("rr_actual"), default=rr_plan)
        st_break = _safe_text(gpack.get("trigger") or "N/A").strip().upper()
        st_vol = _safe_text(gpack.get("volume") or "N/A").strip().upper()
        st_rr = _safe_text(gpack.get("rr") or "N/A").strip().upper()
        st_struct = _safe_text(gpack.get("structure") or "N/A").strip().upper()
    else:
        # legacy fallback: derive statuses from metrics and structure ceiling gate
        combat = cp.get("CombatStats") or {}
        combat = combat if isinstance(combat, dict) else {}
        st_break = _status(combat.get("BreakoutForce"), good=6.8, warn=5.5)
        st_vol = _status(vol_ratio, good=1.20, warn=0.95)
        st_rr = _status(rr_plan, good=1.80, warn=1.30)
        cg = ((sq.get("Gates") or {}).get("CeilingGate") or {}) if isinstance((sq.get("Gates") or {}), dict) else {}
        st_struct = _safe_text(cg.get("Status") or "N/A").strip().upper()

    def _norm_st(s: str) -> str:
        s = (s or "").strip().upper()
        return s if s in ("PASS", "WAIT", "FAIL") else "N/A"

    st_break = _norm_st(st_break)
    st_vol = _norm_st(st_vol)
    st_rr = _norm_st(st_rr)
    st_struct = _norm_st(st_struct)

    gate_line = f"Gate: {gtxt or 'N/A'} | Plan: {setup_name} ({plan_status or 'N/A'})"

    # Next step (single line, long-only language)
    next_step = "Theo dõi và chờ thêm dữ liệu."
    if st_struct in ("FAIL", "WAIT"):
        _ov = ((sq or {}).get("OverheadResistance", {}) or {}).get("Nearest", {}) or {}
        _comps = _ov.get("ComponentsTop") if isinstance(_ov.get("ComponentsTop"), list) else []
        _t = _safe_text(((_comps[0] or {}).get("Type")) if (len(_comps) > 0 and isinstance(_comps[0], dict)) else "").strip()
        _t_pretty = _pretty_level_label(_t, side="overhead") if _t else ""
        next_step = "Chờ reclaim mốc cấu trúc phía trên trước khi tăng xác suất vào lệnh."
        if _t_pretty and _t_pretty != "N/A":
            next_step = f"Chờ reclaim mốc cấu trúc phía trên ({_t_pretty}) trước khi tăng xác suất vào lệnh."
    elif st_break in ("FAIL", "WAIT"):
        next_step = "Chờ breakout xác nhận/follow-through; tránh vào sớm khi lực chưa rõ."
    elif st_vol in ("FAIL", "WAIT"):
        next_step = "Chờ volume xác nhận (≥ 1.2×20D) để giảm false-break."
    elif st_rr in ("FAIL", "WAIT"):
        next_step = "Chờ điểm vào tốt hơn để RR ≥ 1.8 (hoặc giảm risk/stop hợp lý)."
    else:
        next_step = "Có thể triển khai theo plan; ưu tiên kỷ luật stop, tránh FOMO."

    # Risk flags (top 2, severity>=2)
    risk_lines: List[str] = []
    for f in flags_list:
        if not isinstance(f, dict):
            continue
        try:
            sev = int(f.get("severity", 0))
        except Exception:
            sev = 0
        if sev < 2:
            continue
        code = _safe_text(f.get("code") or "").strip()
        note = _safe_text(f.get("note") or "").strip()
        if code and note:
            risk_lines.append(f"[{code}] {note}")
        elif code:
            risk_lines.append(f"[{code}]")
        elif note:
            risk_lines.append(note)
        if len(risk_lines) >= 2:
            break
    if not risk_lines:
        risk_lines = ["None"]

    # Decision/Position (summary values only; renderer formats)
    decision_summary: Dict[str, Any] = {}
    try:
        dp = ap.get("DecisionPack") or {}
        pmp = ap.get("PositionManagerPack") or {}
        posp = ap.get("PositionStatePack") or {}
        dp = dp if isinstance(dp, dict) else {}
        pmp = pmp if isinstance(pmp, dict) else {}
        posp = posp if isinstance(posp, dict) else {}

        if _safe_text(dp.get("schema") or "").strip() == "DecisionPack.v1":
            decision_summary = {
                "mode": _safe_text(dp.get("mode") or posp.get("mode") or "N/A").strip().upper(),
                "action": _safe_text(dp.get("action") or "N/A").strip().upper(),
                "urgency": _safe_text(dp.get("urgency") or "").strip().upper(),
                "constraint0": _safe_text((dp.get("constraints") or [""])[0] if isinstance(dp.get("constraints"), list) and dp.get("constraints") else "").strip(),
                "position_size_pct_nav": _safe_float(pmp.get("position_size_pct_nav"), default=_safe_float(posp.get("position_size_pct_nav"), default=np.nan)),
                "pnl_pct": _safe_float(dp.get("pnl_pct"), default=np.nan),
                "trim_pct_of_position": _safe_float(pmp.get("trim_pct_of_position"), default=np.nan),
                "stop_suggest": _safe_float(pmp.get("stop_suggest"), default=np.nan),
            }
    except Exception:
        decision_summary = {}

    out = {
        "schema": "DashboardSummaryPack.v1",
        "CurrentStatusCard": {
            "state_capsule_line": state_capsule_line,
            "master_total": master_total,
            "conviction": conviction,
            "insight_line": insight_line,
            "policy_hint_line": policy_hint,
            "gate_line": gate_line,
            "next_step": next_step,
            "plan_status": plan_status,
            "triggers": {
                "breakout": st_break,
                "volume": st_vol,
                "rr": st_rr,
                "structure": st_struct,
            },
            "risk_flags": risk_lines,
            "decision": decision_summary,
        },
    }
    return out
