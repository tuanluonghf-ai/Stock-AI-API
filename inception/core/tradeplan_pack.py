"""Trade plan pack builder (v1).

Builds a standardized TradePlanPack using only Python-computed facts.
Long-only and portfolio-ready.

Key design constraints (confirmed):
- TradePlanBuilder emits *plans* (levels + gates + triggers), NOT BUY/SELL actions.
- DecisionLayer is the only module allowed to output actionable BUY/WAIT/EXIT/TRIM.

Display policy (confirmed):
- top1_if_clear_winner, else top2.
- Clear winner thresholds: 8.0 / 1.2 / 15% with guardrails.

PlanTypes (confirmed):
- ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .helpers import _safe_float, _safe_text
from .policy import CLASS_POLICY_HINTS, get_class_policy_hint_line
from .gates import trade_plan_gate
from .fail_reasons import make_reason, rank_reasons
from .validate import collect_data_quality_pack


_PLAN_TYPES: List[str] = ["PULLBACK", "BREAKOUT", "MEAN_REV", "RECLAIM", "DEFENSIVE"]
_PLAN_NAME_MAP: Dict[str, str] = {
    "PULLBACK": "Pullback",
    "BREAKOUT": "Breakout",
    "MEAN_REV": "MeanRev",
    "RECLAIM": "Reclaim",
    "DEFENSIVE": "Defensive",
}

_CLEAR_WINNER = {
    "score1_min": 8.0,
    "gap_min": 1.2,
    "rel_gap_min": 0.15,
    "abs_gap_floor": 1.0,
}


def compute_trade_plan_pack_v1(analysis_pack: Dict[str, Any], character_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ap = analysis_pack or {}
    cp = character_ctx or {}

    # -----------------------------
    # Mode context (FLAT vs HOLDING)
    # -----------------------------
    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}
    mode = _safe_text(pos.get("mode") or "FLAT").strip().upper()
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    is_holding = bool(pos.get("is_holding")) if mode == "HOLDING" else False

    # Holding distress overlay (uses avg_cost/current_price if provided)
    pnl_pct = _safe_float(pos.get("pnl_pct"), default=np.nan)
    current_px = _safe_float(pos.get("current_price"), default=np.nan)
    # Backward-friendly alias (older code blocks in this file referenced `current_price`).
    current_price = current_px
    distress_level = _safe_text(pos.get("distress_level") or "").strip().upper()
    if distress_level in ("", "-", "N/A"):
        # Derive deterministically from pnl_pct when missing
        if pd.notna(pnl_pct):
            if pnl_pct <= -25.0:
                distress_level = "SEVERE"
            elif pnl_pct <= -15.0:
                distress_level = "MEDIUM"
            elif pnl_pct <= -7.0:
                distress_level = "MILD"
            else:
                distress_level = "OK"
        else:
            distress_level = "-"

    # Deterministic underwater flag (do not rely on optional distress_level key)
    underwater = bool(is_holding and pd.notna(pnl_pct) and float(pnl_pct) <= -15.0)

    # -----------------------------
    # DNA / Status context
    # -----------------------------
    dna = ap.get("DNAPack") or cp.get("DNAPack") or {}
    dna = dna if isinstance(dna, dict) else {}

    sp = ap.get("StatusPack") or cp.get("StatusPack") or {}
    sp = sp if isinstance(sp, dict) else {}
    sp_t = sp.get("technicals") or {}
    sp_t = sp_t if isinstance(sp_t, dict) else {}

    scen = ap.get("Scenario12") or {}
    scen = scen if isinstance(scen, dict) else {}
    scen_name = _safe_text(scen.get("Name") or "-").strip() or "-"

    final_class = _safe_text(cp.get("CharacterClass") or dna.get("class_primary") or ap.get("CharacterClass") or "").strip()
    policy_hint_line = get_class_policy_hint_line(final_class) if final_class else ""

    # Class policy RR minimum (fallback 1.5)
    rr_min = 1.5
    try:
        pol = CLASS_POLICY_HINTS.get(final_class or "", {}) if isinstance(CLASS_POLICY_HINTS, dict) else {}
        rr_min = float(pol.get("RRMin", rr_min)) if isinstance(pol, dict) else rr_min
    except Exception:
        rr_min = 1.5

    # -----------------------------
    # Legacy metrics used for gating (system personality preserved)
    # -----------------------------
    combat = (cp.get("CombatStats") or ap.get("CombatStats") or {})
    combat = combat if isinstance(combat, dict) else {}
    breakout_force = _safe_float(combat.get("BreakoutForce"), default=np.nan)

    protech = ap.get("ProTech") or {}
    protech = protech if isinstance(protech, dict) else {}
    vol = protech.get("Volume") or {}
    vol = vol if isinstance(vol, dict) else {}
    vol_ratio = _safe_float(vol.get("Ratio"), default=np.nan)

    sq = (cp.get("StructureQuality") or ap.get("StructureQuality") or ap.get("StructureQualityPack") or {})
    sq = sq if isinstance(sq, dict) else {}
    gates_pack = sq.get("Gates") or {}
    gates_pack = gates_pack if isinstance(gates_pack, dict) else {}
    cg = gates_pack.get("CeilingGate") or {}
    cg = cg if isinstance(cg, dict) else {}
    st_struct = _safe_text(cg.get("Status") or "WAIT").strip().upper()
    if st_struct not in ("PASS", "WAIT", "FAIL"):
        st_struct = "WAIT"

    gate_status, _meta = trade_plan_gate(ap, cp)
    gs = _safe_text(gate_status).strip().upper()
    if gs == "LOCK":
        st_exec = "FAIL"
    elif gs == "ACTIVE":
        st_exec = "PASS"
    else:
        st_exec = "WAIT"

    # -----------------------------
    # Resolve plan setups by type (from AnalysisPack.TradePlans)
    # -----------------------------
    trade_plans = ap.get("TradePlans") or []
    trade_plans = trade_plans if isinstance(trade_plans, list) else []

    plan_by_name: Dict[str, Dict[str, Any]] = {}
    for row in trade_plans:
        if not isinstance(row, dict):
            continue
        nm = _safe_text(row.get("Name") or "").strip()
        if nm:
            plan_by_name[nm.upper()] = row

    def _get_plan_row(ptype: str) -> Dict[str, Any]:
        nm = _PLAN_NAME_MAP.get(ptype, "")
        if not nm:
            return {}
        return plan_by_name.get(nm.upper(), {})

    # -----------------------------
    # Helpers
    # -----------------------------
    def _vol_gate(ptype: str) -> str:
        # Breakout/Reclaim require confirmation; Pullback/MeanRev tolerate lower volume; Defensive always PASS.
        if ptype == "DEFENSIVE":
            return "PASS"
        vr = vol_ratio
        if ptype in ("BREAKOUT", "RECLAIM"):
            if pd.isna(vr):
                return "WAIT"
            if vr >= 1.20:
                return "PASS"
            if vr >= 0.95:
                return "WAIT"
            return "FAIL"
        if ptype == "MEAN_REV":
            if pd.isna(vr):
                return "WAIT"
            return "PASS" if vr <= 1.15 else "WAIT"
        # PULLBACK default
        if pd.isna(vr):
            return "WAIT"
        return "PASS" if vr >= 0.85 else "WAIT"

    def _rr_gate(ptype: str, rr: float) -> str:
        if ptype == "DEFENSIVE":
            return "PASS"
        if pd.isna(rr):
            return "FAIL"
        # Policy minimum is class-dependent; keep a hard floor consistent with system guardrails.
        hard_floor = 1.2
        if rr >= max(rr_min, hard_floor):
            return "PASS"
        if rr >= max(hard_floor, 0.85 * rr_min):
            return "WAIT"
        return "FAIL"

    def _trigger_gate(ptype: str, setup_status: str) -> str:
        if ptype == "DEFENSIVE":
            return "PASS"
        s = (setup_status or "").strip().upper()
        if s == "ACTIVE":
            return "PASS"
        if s == "INVALID":
            return "FAIL"
        return "WAIT"

    def _breakout_gate(ptype: str) -> str:
        # Preserve system personality: BreakoutForce is still a structural quality proxy.
        if ptype in ("BREAKOUT", "RECLAIM"):
            x = breakout_force
            if pd.isna(x):
                return "WAIT"
            if x >= 6.8:
                return "PASS"
            if x >= 5.5:
                return "WAIT"
            return "FAIL"
        return "PASS"

    def _entry_zone(entry: float, stop: float) -> Tuple[float, float]:
        if pd.isna(entry):
            return (np.nan, np.nan)
        # Prefer risk-derived buffer (no fixed %). Bound to keep UI readable.
        buf = np.nan
        if pd.notna(stop) and entry > stop:
            risk = float(entry - stop)
            buf = 0.20 * risk
        if pd.isna(buf) or buf <= 0:
            # fallback: small adaptive range, not exceeding 1.2%
            buf = max(0.003 * float(entry), min(0.012 * float(entry), 0.006 * float(entry)))
        lo = float(entry) - float(buf)
        hi = float(entry) + float(buf)
        return (lo, hi)

    def _plan_completeness(ptype: str, entry: float, stop: float, tp1: float) -> Dict[str, Any]:
        if ptype == "DEFENSIVE":
            return {"status": "PASS", "missing": [], "message": "-"}

        missing: List[str] = []
        if pd.isna(entry):
            missing.append("Entry")
        if pd.isna(stop):
            missing.append("Stop")
        if pd.isna(tp1):
            missing.append("TP1")

        # Entry zone requires entry and stop.
        if pd.isna(entry) or pd.isna(stop) or not (entry > stop):
            missing.append("EntryZone")

        if missing:
            msg = "Trade plan thiếu: " + ", ".join(sorted(set(missing)))
            return {"status": "FAIL", "missing": sorted(set(missing)), "message": msg}
        return {"status": "PASS", "missing": [], "message": "-"}

    def _fit_score(ptype: str) -> float:
        # DNA priors (stable): favor Trend/Momentum/Range alignment.
        style = _safe_text(dna.get("style_primary") or "").strip().upper()
        tilt_to = _safe_text(dna.get("style_tilt_to") or "-").strip().upper()
        tilt_strength = _safe_text(dna.get("style_tilt_strength") or "-").strip().upper()

        score = 5.0
        if style == "TREND":
            if ptype in ("PULLBACK", "RECLAIM"):
                score += 2.0
            elif ptype == "BREAKOUT":
                score += 1.2
            elif ptype == "MEAN_REV":
                score -= 0.8
        elif style == "MOMENTUM":
            if ptype in ("BREAKOUT", "RECLAIM"):
                score += 2.0
            elif ptype == "PULLBACK":
                score += 0.8
            elif ptype == "MEAN_REV":
                score -= 0.8
        elif style == "RANGE":
            if ptype == "MEAN_REV":
                score += 2.2
            elif ptype == "PULLBACK":
                score += 0.6
            elif ptype in ("BREAKOUT", "RECLAIM"):
                score -= 1.2

        # Tilt (only if NearBoundary set in DNAPack)
        if tilt_to not in ("", "-", "N/A") and tilt_strength in ("WEAK", "MEDIUM", "STRONG"):
            if tilt_to == "TREND" and ptype in ("PULLBACK", "RECLAIM"):
                score += 0.8
            if tilt_to == "MOMENTUM" and ptype in ("BREAKOUT", "RECLAIM"):
                score += 0.8
            if tilt_to == "RANGE" and ptype == "MEAN_REV":
                score += 0.8
            if tilt_strength == "STRONG":
                score += 0.4

        # Defensive is a valid posture under high risk.
        risk_ctx = _safe_text(sp_t.get("risk_context") or "").strip().upper()
        if ptype == "DEFENSIVE":
            score = 7.2 if risk_ctx == "HIGH" else 5.2

        # HOLDING: when underwater is medium/severe, bias away from add-style plans
        if is_holding and distress_level in ("MEDIUM", "SEVERE"):
            if ptype == "DEFENSIVE":
                score += 1.6
            elif ptype == "RECLAIM":
                score += 1.0
            elif ptype in ("BREAKOUT", "PULLBACK", "MEAN_REV"):
                score -= 1.4

        return float(max(0.0, min(10.0, score)))

    def _status_support_score(ptype: str) -> float:
        # Contextualized technicals from StatusPack (no action output).
        trend = _safe_text(sp_t.get("trend_sentiment") or "").strip().upper()
        mom = _safe_text(sp_t.get("momentum_sentiment") or "").strip().upper()
        risk = _safe_text(sp_t.get("risk_context") or "").strip().upper()

        score = 5.0
        if trend in ("BULLISH", "POSITIVE", "UP"):
            if ptype in ("PULLBACK", "RECLAIM"):
                score += 1.2
            if ptype == "BREAKOUT":
                score += 0.6
        if trend in ("BEARISH", "NEGATIVE", "DOWN"):
            if ptype in ("BREAKOUT", "RECLAIM"):
                score -= 1.2

        if mom in ("BULLISH", "POSITIVE"):
            if ptype in ("BREAKOUT", "RECLAIM"):
                score += 1.0
        if mom in ("BEARISH", "NEGATIVE"):
            if ptype == "MEAN_REV":
                score += 0.6  # mean-rev can work better when momentum is exhausted
            if ptype in ("BREAKOUT", "RECLAIM"):
                score -= 0.8

        if risk == "HIGH":
            if ptype == "DEFENSIVE":
                score += 1.6
            else:
                score -= 0.6

        return float(max(0.0, min(10.0, score)))

    def _rr_score(rr: float) -> float:
        if pd.isna(rr):
            return 0.0
        # Normalize RR to 0..10 with diminishing returns.
        if rr >= 4.0:
            return 10.0
        if rr >= 3.0:
            return 8.5
        if rr >= 2.0:
            return 7.0
        if rr >= 1.5:
            return 5.5
        if rr >= 1.2:
            return 4.0
        return 2.0

    def _exec_score() -> float:
        if st_exec == "PASS":
            return 8.0
        if st_exec == "WAIT":
            return 6.0
        return 2.0

    def _management_rules(ptype: str) -> List[str]:
        if ptype == "BREAKOUT":
            return [
                "Entry chỉ khi giá vượt vùng kháng cự và có follow-through.",
                "Stop: dưới vùng breakout + buffer (động theo ATR/vol proxy).",
                "Nếu breakout fail (đóng cửa lại dưới vùng), ưu tiên thoát/giảm.",
            ]
        if ptype == "PULLBACK":
            return [
                "Ưu tiên hồi về hỗ trợ (MA/Fib) rồi mới vào; tránh mua đuổi.",
                "Stop: dưới hỗ trợ neo + buffer (động).",
                "Khi hồi lên kháng cự gần, ưu tiên chốt một phần/siết stop.",
            ]
        if ptype == "MEAN_REV":
            return [
                "Chỉ triển khai khi có range rõ; ưu tiên vào gần hỗ trợ và thoát gần kháng cự.",
                "Stop: dưới hỗ trợ neo + buffer (động).",
                "Không đuổi theo breakout trong regime range nếu chưa có xác nhận.",
            ]
        if ptype == "RECLAIM":
            return [
                "Chờ reclaim mốc cấu trúc (giá lấy lại kháng cự và giữ được).",
                "Stop: dưới mốc reclaim + buffer (động).",
                "Nếu reclaim fail → giảm/thoát nhanh.",
            ]
        # DEFENSIVE
        return [
            "Ưu tiên đứng ngoài hoặc giảm rủi ro khi cấu trúc/volume chưa xác nhận.",
            "Chỉ chuyển sang plan chủ động khi Structure/Volume/Trigger cải thiện.",
        ]

    def _collect_fail_reasons(ptype: str, g: Dict[str, Any], pc: Dict[str, Any], rr_val: float) -> List[Dict[str, Any]]:
        """Return ranked reasons explaining why the plan is not ACTIVE / not deployable.

        Includes both FAIL and WAIT blockers to support:
          - dashboard gate labels
          - "PASS triggers but plan empty" prevention
        """
        reasons: List[Dict[str, Any]] = []

        # Data quality: high-signal only
        try:
            dq = collect_data_quality_pack(ap, cp, extra={"TradePlanPack": ap.get("TradePlanPack")}, max_issues=6)
            if isinstance(dq, dict) and int(dq.get("error_count", 0)) > 0:
                reasons.append(make_reason("DATA_QUALITY_LOW"))
        except Exception:
            pass

        # Plan completeness
        pc_s = _safe_text(pc.get("status") or "").strip().upper()
        if pc_s == "FAIL":
            missing = pc.get("missing")
            miss = ", ".join([_safe_text(x).strip() for x in (missing if isinstance(missing, list) else []) if _safe_text(x).strip()])
            reasons.append(make_reason("PLAN_MISSING_LEVELS", detail=miss))

        # Execution lock
        exec_s = _safe_text(g.get("exec") or "").strip().upper()
        if exec_s == "FAIL":
            reasons.append(make_reason("EXEC_LOCKED"))

        # Structure
        struct_s = _safe_text(g.get("structure") or "").strip().upper()
        if struct_s == "FAIL":
            reasons.append(make_reason("STRUCTURE_BROKEN"))
        elif struct_s == "WAIT":
            reasons.append(make_reason("STRUCTURE_CEILING"))

        # RR
        rr_s = _safe_text(g.get("rr") or "").strip().upper()
        if rr_s == "FAIL":
            reasons.append(make_reason("RR_BELOW_MIN"))
        elif rr_s == "WAIT" and ptype != "DEFENSIVE":
            # borderline RR is a soft blocker
            reasons.append(make_reason("RR_BORDERLINE"))

        # Volume
        vol_s = _safe_text(g.get("volume") or "").strip().upper()
        if vol_s == "FAIL":
            reasons.append(make_reason("VOLUME_NOT_CONFIRM"))
        elif vol_s == "WAIT" and ptype != "DEFENSIVE":
            reasons.append(make_reason("VOLUME_NEED_CONFIRM"))

        # Trigger
        trig_s = _safe_text(g.get("trigger") or "").strip().upper()
        if trig_s == "FAIL":
            reasons.append(make_reason("TRIGGER_INVALID"))
        elif trig_s == "WAIT" and ptype != "DEFENSIVE":
            reasons.append(make_reason("TRIGGER_NOT_READY"))

        # Breakout quality proxy (only relevant for breakout/reclaim)
        br_s = _safe_text(g.get("breakout") or "").strip().upper()
        if ptype in ("BREAKOUT", "RECLAIM") and br_s == "FAIL":
            reasons.append(make_reason("BREAKOUT_QUALITY_LOW"))



        # HOLDING distress: do not encourage adds when position is deep underwater.
        if is_holding and distress_level in ("MEDIUM", "SEVERE") and ptype in ("BREAKOUT", "PULLBACK", "MEAN_REV"):
            reasons.append(make_reason("HOLD_UNDERWATER_SEVERE" if distress_level == "SEVERE" else "HOLD_UNDERWATER_MEDIUM"))
        # Regime conflict (soft): when contextual status is negative for the plan type
        try:
            ss = _status_support_score(ptype)
            if ptype not in ("DEFENSIVE",) and float(ss) <= 3.4:
                reasons.append(make_reason("REGIME_CONFLICT"))
        except Exception:
            pass

        ranked = rank_reasons(reasons)
        return ranked[:3]

    def _rr_display(rr_status: str, rr_value: float) -> str:
        rs = (rr_status or "").strip().upper()
        if rs != "PASS" or pd.isna(rr_value):
            return "-"
        try:
            return f"{float(rr_value):.1f}"
        except Exception:
            return "-"

    # -----------------------------
    # Build candidates
    # -----------------------------
    candidates: List[Dict[str, Any]] = []

    for ptype in _PLAN_TYPES:
        row = _get_plan_row(ptype)

        entry = _safe_float(row.get("Entry"), default=np.nan)
        stop = _safe_float(row.get("Stop"), default=np.nan)
        tp1 = _safe_float(row.get("TP"), default=np.nan)
        rr_act = _safe_float(row.get("RR"), default=np.nan)
        setup_status = _safe_text(row.get("Status") or "Watch").strip().upper()

        # Secondary TP: deterministic extension from TP1
        tp2 = np.nan
        try:
            if pd.notna(tp1) and pd.notna(entry):
                tp2 = float(entry) + 1.6 * (float(tp1) - float(entry))
        except Exception:
            tp2 = np.nan

        # Entry zone derived from risk distance (no fixed %)
        lo, hi = _entry_zone(entry, stop)

        pc = _plan_completeness(ptype, entry, stop, tp1)
        plan_gate = "PASS" if _safe_text(pc.get("status") or "").strip().upper() == "PASS" else "FAIL"

        g = {
            "plan": plan_gate,
            "trigger": _trigger_gate(ptype, setup_status),
            "volume": _vol_gate(ptype),
            "rr": _rr_gate(ptype, rr_act),
            "exec": st_exec,
            "structure": st_struct,
            # Extra gate that preserves legacy breakout quality
            "breakout": _breakout_gate(ptype),
        }

        # State: blueprint only (no action)
        state = "WATCH"
        if g["exec"] == "FAIL":
            state = "LOCK"
        elif ptype == "DEFENSIVE":
            state = "WATCH"
        elif g["plan"] == "FAIL":
            state = "INVALID" if (not is_holding) else "WATCH"
        elif setup_status == "ACTIVE" and all(_safe_text(g[k]).strip().upper() != "FAIL" for k in ("structure", "volume", "rr", "breakout")):
            state = "ACTIVE"
        else:
            state = "WATCH"

        # Ranking score (0..10)
        rank = 0.42 * _fit_score(ptype) + 0.28 * _status_support_score(ptype) + 0.20 * _rr_score(rr_act) + 0.10 * _exec_score()
        rank = float(max(0.0, min(10.0, rank)))

        candidates.append(
            {
                "type": ptype,
                "rank_score": rank,
                "state": state,
                "gates": {k: _safe_text(v).strip().upper() for k, v in g.items()},
                "plan_completeness": pc,
                "fail_reasons": _collect_fail_reasons(ptype, g, pc, rr_act),
                "entry_zone": {"Low": float(lo) if pd.notna(lo) else np.nan, "High": float(hi) if pd.notna(hi) else np.nan},
                "stop": float(stop) if pd.notna(stop) else np.nan,
                "tp1": float(tp1) if pd.notna(tp1) else np.nan,
                "tp2": float(tp2) if pd.notna(tp2) else np.nan,
                "rr_actual": float(rr_act) if pd.notna(rr_act) else np.nan,
                "rr_display": _rr_display(g.get("rr"), rr_act),
                "rr_min": float(rr_min),
                "management_rules": _management_rules(ptype),
                "invalidation": "Invalidation: thiếu stop/entry zone." if _safe_text(pc.get("status") or "").strip().upper() == "FAIL" else "Invalidation: thủng stop/đóng cửa dưới vùng cấu trúc neo stop.",
                "notes_short": f"Scenario: {scen_name} | Setup: {_PLAN_NAME_MAP.get(ptype, ptype)}",
            }
        )

    # Defensive overlay levels (HOLDING de-risk): pick nearest valid stop below current price
    hard_stop = np.nan
    reclaim_level = np.nan
    reclaim_zone_low = np.nan
    reclaim_zone_high = np.nan
    try:
        if pd.notna(current_px):
            stops = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                if _safe_text(c.get("type") or "").strip().upper() == "DEFENSIVE":
                    continue
                s = _safe_float(c.get("stop"), default=np.nan)
                if pd.notna(s) and s < float(current_px):
                    stops.append(float(s))
            if stops:
                hard_stop = max(stops)
        # Reclaim confirmation level: use RECLAIM entry when available
        rrow = _get_plan_row("RECLAIM")
        reclaim_level = _safe_float(rrow.get("Entry"), default=np.nan)
    except Exception:
        pass


    # Reclaim trigger zone: prefer RECLAIM entry zone (Low/High) for clearer execution guidance.
    try:
        for _c in candidates:
            if not isinstance(_c, dict):
                continue
            if _safe_text(_c.get("type") or "").strip().upper() == "RECLAIM":
                _ez = _c.get("entry_zone") or {}
                if isinstance(_ez, dict):
                    reclaim_zone_low = _safe_float(_ez.get("Low"), default=np.nan)
                    reclaim_zone_high = _safe_float(_ez.get("High"), default=np.nan)
                break
    except Exception:
        pass
    if pd.notna(reclaim_level) and (pd.isna(reclaim_zone_low) or pd.isna(reclaim_zone_high)):
        reclaim_zone_low = float(reclaim_level)
        reclaim_zone_high = float(reclaim_level)

    # Attach overlay to DEFENSIVE candidate (for UI/Decision consistency)
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if _safe_text(c.get("type") or "").strip().upper() != "DEFENSIVE":
            continue
        c["defensive_hard_stop"] = float(hard_stop) if pd.notna(hard_stop) else np.nan
        c["defensive_reclaim_level"] = float(reclaim_level) if pd.notna(reclaim_level) else np.nan
        c["defensive_reclaim_zone"] = {"Low": float(reclaim_zone_low) if pd.notna(reclaim_zone_low) else np.nan, "High": float(reclaim_zone_high) if pd.notna(reclaim_zone_high) else np.nan}
        # Backward-friendly alias for UI/renderers
        c["defensive_reclaim"] = c.get("defensive_reclaim_level")
        # Stronger, explicit de-risk language when underwater is medium/severe
        rules = [
            "Ưu tiên đứng ngoài hoặc giảm rủi ro khi cấu trúc/volume chưa xác nhận.",
            "Không add khi đang lỗ sâu; chỉ cân nhắc lại khi reclaim cấu trúc." if (is_holding and distress_level in ("MEDIUM", "SEVERE")) else "Chỉ chuyển sang plan chủ động khi Structure/Volume/Trigger cải thiện.",
        ]
        if pd.notna(hard_stop):
            rules.append(f"Hard stop: nếu đóng cửa < {float(hard_stop):.2f} → ưu tiên giảm mạnh/thoát theo policy.")
        if pd.notna(reclaim_level):
            rules.append(f"Reclaim trigger: khi giá giữ lại trên ~{float(reclaim_level):.2f} và volume/structure đồng pha.")
        c["management_rules"] = rules



    # -----------------------------
    # HOLDING Defensive overlay: derive a practical hard stop & reclaim trigger
    # -----------------------------
    try:
        cur_px = float(current_price) if (current_price is not None) else np.nan
    except Exception:
        cur_px = np.nan

    # Choose the nearest meaningful stop below current price (reduce tail risk).
    hard_stop = np.nan
    try:
        stops = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if _safe_text(c.get("type") or "").strip().upper() == "DEFENSIVE":
                continue
            s = _safe_float(c.get("stop"), default=np.nan)
            if pd.notna(s):
                stops.append(float(s))
        below = [s for s in stops if (pd.notna(cur_px) and pd.notna(s) and s < cur_px)]
        if below:
            hard_stop = max(below)
    except Exception:
        hard_stop = np.nan

    # Reclaim trigger: use RECLAIM plan entry (if any) as the confirmation level.
    reclaim_level = np.nan
    reclaim_zone_low = np.nan
    reclaim_zone_high = np.nan
    try:
        rrow = _get_plan_row("RECLAIM")
        reclaim_level = _safe_float(rrow.get("Entry"), default=np.nan)
    except Exception:
        reclaim_level = np.nan


    # Reclaim trigger zone: prefer RECLAIM entry zone (Low/High) when available.
    if pd.isna(reclaim_zone_low) or pd.isna(reclaim_zone_high):
        try:
            for _c in candidates:
                if not isinstance(_c, dict):
                    continue
                if _safe_text(_c.get("type") or "").strip().upper() == "RECLAIM":
                    _ez = _c.get("entry_zone") or {}
                    if isinstance(_ez, dict):
                        reclaim_zone_low = _safe_float(_ez.get("Low"), default=np.nan)
                        reclaim_zone_high = _safe_float(_ez.get("High"), default=np.nan)
                    break
        except Exception:
            pass
    if pd.notna(reclaim_level) and (pd.isna(reclaim_zone_low) or pd.isna(reclaim_zone_high)):
        reclaim_zone_low = float(reclaim_level)
        reclaim_zone_high = float(reclaim_level)

    # Update defensive candidate fields + rules (so UI surfaces de-risk / cut-loss posture)
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if _safe_text(c.get("type") or "").strip().upper() != "DEFENSIVE":
            continue
        c["defensive_hard_stop"] = float(hard_stop) if pd.notna(hard_stop) else np.nan
        c["defensive_reclaim_level"] = float(reclaim_level) if pd.notna(reclaim_level) else np.nan
        c["defensive_reclaim_zone"] = {"Low": float(reclaim_zone_low) if pd.notna(reclaim_zone_low) else np.nan, "High": float(reclaim_zone_high) if pd.notna(reclaim_zone_high) else np.nan}
        # Backward-friendly alias for UI/renderers
        c["defensive_reclaim"] = c.get("defensive_reclaim_level")

        rules = [
            "Ưu tiên đứng ngoài hoặc giảm rủi ro khi cấu trúc/volume chưa xác nhận.",
        ]
        if is_holding and distress_level in ("MEDIUM", "SEVERE"):
            rules.append("Vị thế đang underwater: ưu tiên giảm tỷ trọng về mức an toàn; tuyệt đối không add khi chưa reclaim.")
            if pd.notna(hard_stop):
                rules.append(f"Hard stop: nếu đóng cửa < {float(hard_stop):.2f} → ưu tiên cắt mạnh/thoát để chặn lỗ tiếp.")
            if pd.notna(reclaim_level):
                rules.append(f"Chỉ xem xét phục hồi/giữ chủ động khi reclaim lại > {float(reclaim_level):.2f} và giữ được.")
        else:
            rules.append("Chỉ chuyển sang plan chủ động khi Structure/Volume/Trigger cải thiện.")

        c["management_rules"] = rules[:5]
        break
    # Sort by rank_score desc
    candidates.sort(key=lambda x: float(x.get("rank_score") or 0.0), reverse=True)

    # -----------------------------
    # Display logic: top1_if_clear_winner, else top2
    # -----------------------------
    best = candidates[0] if candidates else {}
    second = candidates[1] if len(candidates) > 1 else None

    score1 = float(best.get("rank_score") or 0.0)
    score2 = float(second.get("rank_score") or 0.0) if isinstance(second, dict) else 0.0
    gap = score1 - score2

    clear_winner = False
    clear_winner_reason = "CW_NONE"

    if (score1 >= _CLEAR_WINNER["score1_min"]) and (gap >= _CLEAR_WINNER["gap_min"]):
        clear_winner = True
        clear_winner_reason = "CW_ABSOLUTE"
    elif gap >= max(_CLEAR_WINNER["abs_gap_floor"], _CLEAR_WINNER["rel_gap_min"] * score1):
        clear_winner = True
        clear_winner_reason = "CW_RELATIVE"

    # Guardrails: show top2 when tilt is strong, runner-up is strong, or risk is HIGH
    tilt_strength = _safe_text(dna.get("style_tilt_strength") or "-").strip().upper()
    risk_ctx = _safe_text(sp_t.get("risk_context") or "").strip().upper()

    force_top2 = False
    force_top2_reason = "-"

    if tilt_strength == "STRONG":
        force_top2 = True
        force_top2_reason = "CW_BLOCK_TILT_STRONG"
    elif score2 >= 7.0:
        force_top2 = True
        force_top2_reason = "CW_BLOCK_RUNNERUP_STRONG"
    elif risk_ctx == "HIGH":
        force_top2 = True
        force_top2_reason = "CW_BLOCK_RISK_HIGH"

    display_mode = "TOP1" if (clear_winner and not force_top2) else "TOP2"

    # HOLDING: prefer showing runner-up unless winner is very dominant
    if is_holding and display_mode == "TOP1" and gap < 1.5:
        display_mode = "TOP2"
        force_top2 = True
        force_top2_reason = "CW_BLOCK_HOLDING"

    plans_display = [best]
    if display_mode == "TOP2" and isinstance(second, dict):
        plans_display.append(second)


    # HOLDING: if underwater is medium/severe, always surface a DEFENSIVE de-risk plan as the alternative.
    # This prevents the UX issue: "đang lỗ sâu" nhưng chỉ thấy các plan theo hướng mua.
    if is_holding and (distress_level in ("MEDIUM", "SEVERE") or (pd.notna(pnl_pct) and float(pnl_pct) <= -15.0)):
        # Ensure at least TOP2 and force show defensive.
        if display_mode == "TOP1":
            display_mode = "TOP2"
            force_top2 = True
            force_top2_reason = "CW_BLOCK_UNDERWATER"
        defensive = None
        for c in candidates:
            if isinstance(c, dict) and _safe_text(c.get("type") or "").strip().upper() == "DEFENSIVE":
                defensive = c
                break
        if isinstance(defensive, dict):
            # If defensive not already displayed, put it as alternative.
            if not any(isinstance(p, dict) and _safe_text(p.get("type") or "").strip().upper() == "DEFENSIVE" for p in plans_display):
                if len(plans_display) == 1:
                    plans_display.append(defensive)
                else:
                    plans_display[1] = defensive

    # HOLDING distress: always show a DEFENSIVE (de-risk) alternative when underwater is medium/severe
    if is_holding and (distress_level in ("MEDIUM", "SEVERE") or (pd.notna(pnl_pct) and float(pnl_pct) <= -15.0)):
        defensive = None
        for c in candidates:
            if isinstance(c, dict) and _safe_text(c.get("type") or "").strip().upper() == "DEFENSIVE":
                defensive = c
                break
        if isinstance(defensive, dict):
            if len(plans_display) == 1:
                display_mode = "TOP2"
                force_top2 = True
                force_top2_reason = "CW_BLOCK_UNDERWATER"
                plans_display.append(defensive)
            elif len(plans_display) >= 2:
                has_def = any(isinstance(p, dict) and _safe_text(p.get("type") or "").strip().upper() == "DEFENSIVE" for p in plans_display)
                if not has_def:
                    display_mode = "TOP2"
                    force_top2 = True
                    force_top2_reason = "CW_BLOCK_UNDERWATER"
                    plans_display[1] = defensive

    explain = "TradePlanBuilder: candidate plans are scored by (DNA priors + contextual status + RR + execution)."
    if policy_hint_line:
        explain += " Policy hint applied."

    plan_primary = plans_display[0] if plans_display else {}
    plan_alt = plans_display[1] if len(plans_display) >= 2 else None

    # Pack-level completeness mirrors the chosen primary plan for dashboard stability.
    pack_plan_comp = plan_primary.get("plan_completeness") if isinstance(plan_primary, dict) else {"status": "FAIL", "missing": ["Primary"], "message": "Trade plan missing."}

    pack = {
        "schema": "TradePlanPack.v1",
        "mode": "HOLDING" if is_holding else "FLAT",
        "policy_hint_line": policy_hint_line,
        "plan_completeness": pack_plan_comp,
        "plan_primary": plan_primary,
        "plan_alt": plan_alt,
        "plans_all": candidates,
        "plans_display": plans_display,
        "display_mode": display_mode,
        "clear_winner": bool(clear_winner),
        "clear_winner_reason": clear_winner_reason,
        "force_top2": bool(force_top2),
        "force_top2_reason": force_top2_reason,
        "clear_winner_thresholds": dict(_CLEAR_WINNER),
        "explain": explain,
        "holding_overlay": {
            "pnl_pct": float(pnl_pct) if pd.notna(pnl_pct) else np.nan,
            "distress_level": distress_level or "-",
        },
    }

    # fail-safe normalization
    try:
        from inception.core.contracts import normalize_tradeplan_pack
        pack = normalize_tradeplan_pack(pack)
    except Exception:
        pass

    return pack
