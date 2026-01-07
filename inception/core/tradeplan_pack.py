"""Trade plan pack builder (v1).

Builds a standardized TradePlanPack using only Python-computed facts.
Long-only and portfolio-ready.

Notes:
- Decision layer/portfolio can override action and sizing.
- This pack provides the execution blueprint (entry zone, stop, targets, gates).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .helpers import _safe_float, _safe_text
from .policy import CLASS_POLICY_HINTS, get_class_policy_hint_line
from .gates import trade_plan_gate



def compute_trade_plan_pack_v1(analysis_pack: Dict[str, Any], character_ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ap = analysis_pack or {}
    cp = character_ctx or {}

    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}
    mode = _safe_text(pos.get("mode") or "FLAT").strip().upper()
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    is_holding = bool(pos.get("is_holding")) if mode == "HOLDING" else False

    scen = ap.get("Scenario12") or {}
    spec = (scen.get("Spec") or {}) if isinstance(scen, dict) else {}
    default_plan = _safe_text(spec.get("DefaultPlan") or "").strip() or "Pullback"
    scen_name = _safe_text((scen.get("Name") if isinstance(scen, dict) else None) or "N/A").strip()

    def _normalize_plan(p: str) -> str:
        s = (p or "").strip().lower()
        if "break" in s:
            return "BREAKOUT"
        if "range" in s:
            return "RANGE_EDGE"
        if "reclaim" in s:
            return "RECLAIM"
        if "avoid" in s or "invalid" in s:
            return "DEFENSIVE"
        return "PULLBACK"

    setup_type = _normalize_plan(default_plan)

    final_class = _safe_text(cp.get("CharacterClass") or ap.get("CharacterClass") or "").strip()
    policy_hint_line = get_class_policy_hint_line(final_class) if final_class else ""

    rr_min = 1.5
    try:
        pol = CLASS_POLICY_HINTS.get(final_class or "", {}) if isinstance(CLASS_POLICY_HINTS, dict) else {}
        rr_min = float(pol.get("RRMin", rr_min)) if isinstance(pol, dict) else rr_min
    except Exception:
        rr_min = 1.5

    primary_setup = ap.get("PrimarySetup") or {}
    primary_setup = primary_setup if isinstance(primary_setup, dict) else {}
    entry = _safe_float(primary_setup.get("Entry"), default=np.nan)
    stop = _safe_float(primary_setup.get("Stop"), default=np.nan)
    tp1 = _safe_float(primary_setup.get("TP1"), default=np.nan)
    tp2 = _safe_float(primary_setup.get("TP2"), default=np.nan)
    rr_act = _safe_float(primary_setup.get("RR"), default=np.nan)
    setup_name = _safe_text(primary_setup.get("Name") or "").strip() or "Primary"

    try:
        if pd.isna(tp2) and pd.notna(tp1) and pd.notna(entry):
            tp2 = float(entry) + 1.6 * (float(tp1) - float(entry))
    except Exception:
        pass

    vol_pct = _safe_float((ap.get("VolPct") or ap.get("VolPct_ATRProxy")), default=np.nan)
    if not pd.notna(vol_pct):
        vol_pct = 1.2
    bufpct = max(0.003, min(0.012, 0.006 * (float(vol_pct) / 1.5)))

    try:
        if pd.notna(entry):
            lo = float(entry) * (1.0 - bufpct)
            hi = float(entry) * (1.0 + bufpct)
        else:
            lo, hi = np.nan, np.nan
    except Exception:
        lo, hi = np.nan, np.nan

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
    st_struct = _safe_text(cg.get("Status") or "N/A").strip().upper()
    if st_struct not in ("PASS", "WAIT", "FAIL"):
        st_struct = "N/A"

    gate_status, _meta = trade_plan_gate(ap, cp)
    gs = _safe_text(gate_status).strip().upper()
    if gs == "LOCK":
        st_exec = "FAIL"
    elif gs == "ACTIVE":
        st_exec = "PASS"
    else:
        st_exec = "WAIT"

    def _trig_status(ptype: str) -> str:
        if not pd.notna(breakout_force):
            return "WAIT"
        if ptype == "BREAKOUT":
            return "PASS" if breakout_force >= 7 else ("WAIT" if breakout_force >= 5 else "FAIL")
        return "PASS" if breakout_force >= 6 else ("WAIT" if breakout_force >= 4 else "FAIL")

    def _vol_status(ptype: str) -> str:
        if not pd.notna(vol_ratio):
            return "WAIT"
        if ptype == "BREAKOUT":
            return "PASS" if vol_ratio >= 1.25 else ("WAIT" if vol_ratio >= 1.05 else "FAIL")
        return "PASS" if vol_ratio >= 1.10 else ("WAIT" if vol_ratio >= 0.95 else "FAIL")

    def _rr_status(rr: float) -> str:
        if not pd.notna(rr):
            return "WAIT"
        return "PASS" if rr >= rr_min else ("WAIT" if rr >= (rr_min - 0.25) else "FAIL")

    gates = {
        "trigger": _trig_status(setup_type),
        "volume": _vol_status(setup_type),
        "rr": _rr_status(rr_act),
        "structure": st_struct if st_struct != "N/A" else "WAIT",
        "exec": st_exec,
    }

    def _state_flat(g: Dict[str, str]) -> str:
        if any(str(v).upper() == "FAIL" for v in g.values()):
            return "INVALID"
        if any(str(v).upper() == "WAIT" for v in g.values()):
            return "WATCH"
        return "ACTIVE"

    def _state_holding(g: Dict[str, str]) -> str:
        if str(g.get("structure", "")).upper() == "FAIL":
            return "ACTIVE"
        if str(g.get("structure", "")).upper() == "WAIT":
            return "ACTIVE"
        return "WATCH"

    plan_state = _state_holding(gates) if is_holding else _state_flat(gates)

    def _entry_rules(ptype: str) -> List[str]:
        base: List[str] = []
        if ptype == "BREAKOUT":
            base.append("Mua mới chỉ khi Breakout + Volume cùng PASS; tránh mua khi Volume FAIL.")
            base.append("Nếu phía trên có trần cấu trúc (Structure=WAIT), ưu tiên kịch bản RECLAIM thay vì đuổi giá.")
        elif ptype in ("PULLBACK", "RANGE_EDGE"):
            base.append("Ưu tiên mua ở vùng hồi về hỗ trợ/MA/Fib; tránh mua sát kháng cự cấu trúc.")
            base.append("Chỉ triển khai khi RR đủ và stop được neo vào cấu trúc rõ ràng.")
        else:
            base.append("Chờ reclaim mốc cấu trúc phía trên (đóng cửa/giữ được) rồi mới mua mới.")
            base.append("Volume cần tối thiểu WAIT→PASS để tránh bull-trap.")
        base.append("Không dời stop xuống; chỉ dời stop lên theo cấu trúc khi giá đi đúng hướng.")
        if policy_hint_line:
            base.append(f"Policy hint: {policy_hint_line}.")
        return base[:5]

    def _holding_rules() -> List[str]:
        base: List[str] = []
        base.append("Ưu tiên bảo toàn vốn/lợi nhuận: dời stop lên theo cấu trúc khi có lợi thế.")
        base.append("Nếu Structure=WAIT (đụng trần cấu trúc), cân nhắc chốt một phần khi đã có lãi; giữ phần còn lại nếu reclaim thành công.")
        base.append("Nếu Structure=FAIL hoặc mất mốc hỗ trợ chính, ưu tiên giảm rủi ro/thoát vị thế theo stop.")
        base.append("Không averaging-down khi cấu trúc xấu; chỉ tăng khi reclaim + volume xác nhận.")
        if policy_hint_line:
            base.append(f"Policy hint: {policy_hint_line}.")
        return base[:5]

    invalidation = "Invalidation: thiếu stop." if pd.isna(stop) else "Invalidation: thủng stop/đóng cửa dưới vùng cấu trúc neo stop."

    plan_primary = {
        "type": setup_type,
        "state": plan_state,
        "gates": gates,
        "entry_zone": {"Low": float(lo) if pd.notna(lo) else np.nan, "High": float(hi) if pd.notna(hi) else np.nan},
        "stop": float(stop) if pd.notna(stop) else np.nan,
        "tp1": float(tp1) if pd.notna(tp1) else np.nan,
        "tp2": float(tp2) if pd.notna(tp2) else np.nan,
        "rr_actual": float(rr_act) if pd.notna(rr_act) else np.nan,
        "rr_min": float(rr_min),
        "management_rules": _holding_rules() if is_holding else _entry_rules(setup_type),
        "invalidation": invalidation,
        "notes_short": f"Scenario: {scen_name} | Setup: {setup_name}",
    }

    plan_alt = None
    if not is_holding and str(gates.get("structure", "")).upper() in ("WAIT", "FAIL"):
        alt_g = dict(gates)
        alt_g["trigger"] = "WAIT"
        alt_g["rr"] = "WAIT" if alt_g.get("rr") == "FAIL" else alt_g.get("rr")
        plan_alt = {
            "type": "RECLAIM",
            "state": "WATCH",
            "gates": alt_g,
            "entry_zone": {"Low": float(lo) if pd.notna(lo) else np.nan, "High": float(hi) if pd.notna(hi) else np.nan},
            "stop": float(stop) if pd.notna(stop) else np.nan,
            "tp1": float(tp1) if pd.notna(tp1) else np.nan,
            "tp2": float(tp2) if pd.notna(tp2) else np.nan,
            "rr_actual": float(rr_act) if pd.notna(rr_act) else np.nan,
            "rr_min": float(rr_min),
            "management_rules": _entry_rules("RECLAIM"),
            "invalidation": invalidation,
            "notes_short": "Alt plan: wait for reclaim structural level; avoid bull-trap.",
        }

    explain = "Plan bám Scenario + Gates; StructureGate được ưu tiên để tránh bull-trap trước trần cấu trúc."
    if is_holding:
        explain = "HOLDING mode: ưu tiên bảo toàn vốn/lợi nhuận; quyết định TRIM/EXIT do Decision Layer dựa trên PnL & gates."

    return {
        "schema": "TradePlanPack.v1",
        "mode": "HOLDING" if is_holding else "FLAT",
        "policy_hint_line": policy_hint_line,
        "plan_primary": plan_primary,
        "plan_alt": plan_alt,
        "explain": explain,
    }
