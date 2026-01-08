"""Decision pack builder (v1).

Long-only, portfolio-ready decision intent.

Policy (confirmed Option B):
- HOLDING + Structure=WAIT: TRIM if in profit; else HOLD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .helpers import _safe_text


def compute_decision_pack_v1(analysis_pack: Dict[str, Any], trade_plan_pack: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ap = analysis_pack or {}
    tpp = trade_plan_pack or (ap.get("TradePlanPack") or {})
    tpp = tpp if isinstance(tpp, dict) else {}

    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}

    mode = _safe_text(pos.get("mode") or tpp.get("mode") or "FLAT").strip().upper()
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    is_holding = bool(pos.get("is_holding")) if mode == "HOLDING" else False

    in_profit = pos.get("in_profit", None)
    pnl_pct = pos.get("pnl_pct", None)

    pp = tpp.get("plan_primary") or {}
    pp = pp if isinstance(pp, dict) else {}
    gates = pp.get("gates") or {}
    gates = gates if isinstance(gates, dict) else {}

    structure = _safe_text(gates.get("structure") or "").strip().upper()
    if structure not in ("PASS", "WAIT", "FAIL"):
        structure = "WAIT"
    rr = _safe_text(gates.get("rr") or "").strip().upper()
    trig = _safe_text(gates.get("trigger") or "").strip().upper()
    vol = _safe_text(gates.get("volume") or "").strip().upper()

    # Step 11: plan completeness gate (execution safety)
    plan_gate = _safe_text(gates.get("plan") or "").strip().upper()
    if plan_gate not in ("PASS", "WAIT", "FAIL"):
        plan_gate = "WAIT"

    plan_comp = pp.get("plan_completeness") or {}
    plan_comp = plan_comp if isinstance(plan_comp, dict) else {}
    missing = plan_comp.get("missing") or []
    missing = missing if isinstance(missing, list) else []

    constraints: List[str] = []
    if structure in ("WAIT", "FAIL"):
        constraints.append("No add while StructureGate is WAIT/FAIL; prioritize reclaim/confirm.")
    if vol == "FAIL":
        constraints.append("Volume not confirmed; avoid aggressive buys/adds.")
    if rr == "FAIL":
        constraints.append("R:R below policy minimum; wait for better location or confirmation.")
    if trig == "FAIL":
        constraints.append("Trigger weak; avoid chasing; require clearer setup.")
    if plan_gate == "FAIL":
        constraints.append("Plan incomplete (missing Stop/EntryZone); do not execute until completed.")
    elif plan_gate == "WAIT" and missing:
        constraints.append("Plan missing critical fields (Stop/EntryZone); prioritize risk controls.")

    # Default decision
    action = "WAIT"
    urgency = "MED"
    rationale = "Theo dõi thêm tín hiệu xác nhận trước khi hành động."

    if not is_holding:
        state = _safe_text(pp.get("state") or "").strip().upper()
        any_fail = any(_safe_text(v).strip().upper() == "FAIL" for v in gates.values())
        # Step 11: even if all technical gates PASS, an incomplete plan disables BUY
        if plan_gate == "FAIL":
            action = "WAIT"
            urgency = "LOW"
            rationale = "Trade plan chưa hoàn chỉnh (thiếu Stop/Entry zone) → KHÔNG vào lệnh; hoàn thiện plan trước."
        elif state == "ACTIVE" and not any_fail:
            action = "BUY"
            urgency = "HIGH"
            rationale = "Các điều kiện chính đã PASS/đủ xác nhận; có thể triển khai kế hoạch mua mới."
        else:
            action = "WAIT"
            urgency = "LOW" if any_fail else "MED"
            rationale = "Chưa đủ điều kiện cho mua mới; ưu tiên chờ thêm xác nhận (đặc biệt là Structure/Volume)."
    else:
        # HOLDING (Option B)
        if structure == "FAIL":
            action = "EXIT"
            urgency = "HIGH"
            rationale = "Cấu trúc bị phá vỡ (Structure FAIL) → ưu tiên thoát/giảm mạnh để bảo toàn vốn."
        elif structure == "WAIT":
            if in_profit is True:
                action = "TRIM"
                urgency = "MED"
                rationale = "Đụng trần cấu trúc gần (Structure WAIT) trong khi đang có lãi → chốt một phần để bảo toàn lợi nhuận."
            else:
                action = "HOLD"
                urgency = "MED"
                rationale = "Structure WAIT nhưng chưa có lợi nhuận rõ ràng → giữ và quản trị rủi ro theo stop/cấu trúc."
        else:
            action = "HOLD"
            urgency = "LOW"
            rationale = "Cấu trúc ổn (Structure PASS) → ưu tiên giữ và dời stop theo cấu trúc."

        # Step 11: HOLDING but missing stop/entry zone -> raise urgency & force risk review language
        if plan_gate == "WAIT" and missing:
            urgency = "HIGH" if urgency != "HIGH" else urgency
            rationale = "Đang nắm giữ nhưng trade plan thiếu Stop/Entry zone → ưu tiên bổ sung stop/giảm rủi ro trước khi hành động thêm."

    constraints = constraints[:3]

    pack = {
        "schema": "DecisionPack.v1",
        "mode": "HOLDING" if is_holding else "FLAT",
        "action": action,
        "urgency": urgency,
        "rationale": rationale,
        "constraints": constraints,
        "in_profit": in_profit,
        "pnl_pct": pnl_pct,
    }

    # Step 8: normalize pack contract (fail-safe)
    try:
        from inception.core.contracts import normalize_decision_pack
        pack = normalize_decision_pack(pack)
    except Exception:
        pass

    return pack

