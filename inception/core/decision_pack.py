"""Decision pack builder (v1).

Long-only, portfolio-ready decision intent.

Policy (confirmed Option B):
- HOLDING + Structure=WAIT: TRIM if in profit; else HOLD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .helpers import _safe_text, _safe_float


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

    # Holding distress overlay (position management)
    holding_horizon = _safe_text(pos.get("holding_horizon") or "SWING").strip().upper()
    pnl_val = _safe_float(pnl_pct, default=float("nan"))
    distress_level = _safe_text(pos.get("distress_level") or "").strip().upper()
    if not distress_level or distress_level in ("-", "N/A"):
        if pnl_val == pnl_val:  # not NaN
            if pnl_val <= -25.0:
                distress_level = "SEVERE"
            elif pnl_val <= -15.0:
                distress_level = "MEDIUM"
            elif pnl_val <= -7.0:
                distress_level = "MILD"
            else:
                distress_level = "OK"
        else:
            distress_level = "-"

    # Optional hard-stop derived from Defensive overlay (if present)
    defensive_hard_stop = None
    defensive_reclaim_level = None
    defensive_reclaim_zone_lo = None
    defensive_reclaim_zone_hi = None
    try:
        alt = tpp.get("plan_alt") if isinstance(tpp, dict) else None
        if isinstance(alt, dict) and _safe_text(alt.get("type") or "").strip().upper() == "DEFENSIVE":
            hs = alt.get("defensive_hard_stop")
            defensive_hard_stop = _safe_float(hs, default=float("nan"))
            defensive_reclaim_level = _safe_float(
                alt.get("defensive_reclaim_level") if alt.get("defensive_reclaim_level") is not None else alt.get("defensive_reclaim"),
                default=float("nan"),
            )
            z = alt.get("defensive_reclaim_zone") or {}
            if isinstance(z, dict):
                defensive_reclaim_zone_lo = _safe_float(z.get("Low"), default=float("nan"))
                defensive_reclaim_zone_hi = _safe_float(z.get("High"), default=float("nan"))
        else:
            # Fallback: search in plans_all for a DEFENSIVE candidate
            for c in (tpp.get("plans_all") or []):
                if not isinstance(c, dict):
                    continue
                if _safe_text(c.get("type") or "").strip().upper() != "DEFENSIVE":
                    continue
                defensive_hard_stop = _safe_float(c.get("defensive_hard_stop"), default=float("nan"))
                defensive_reclaim_level = _safe_float(
                    c.get("defensive_reclaim_level") if c.get("defensive_reclaim_level") is not None else c.get("defensive_reclaim"),
                    default=float("nan"),
                )
                z = c.get("defensive_reclaim_zone") or {}
                if isinstance(z, dict):
                    defensive_reclaim_zone_lo = _safe_float(z.get("Low"), default=float("nan"))
                    defensive_reclaim_zone_hi = _safe_float(z.get("High"), default=float("nan"))
                break
    except Exception:
        defensive_hard_stop = None
        defensive_reclaim_level = None

    if defensive_hard_stop is not None:
        try:
            if defensive_hard_stop != defensive_hard_stop:  # NaN
                defensive_hard_stop = None
        except Exception:
            defensive_hard_stop = None
    if defensive_reclaim_level is not None:
        try:
            if defensive_reclaim_level != defensive_reclaim_level:
                defensive_reclaim_level = None
        except Exception:
            defensive_reclaim_level = None

    # Normalize defensive reclaim zone values (NaN -> None)
    if defensive_reclaim_zone_lo is not None:
        try:
            if defensive_reclaim_zone_lo != defensive_reclaim_zone_lo:
                defensive_reclaim_zone_lo = None
        except Exception:
            defensive_reclaim_zone_lo = None
    if defensive_reclaim_zone_hi is not None:
        try:
            if defensive_reclaim_zone_hi != defensive_reclaim_zone_hi:
                defensive_reclaim_zone_hi = None
        except Exception:
            defensive_reclaim_zone_hi = None


    try:
        current_price = float(pos.get("current_price"))
    except Exception:
        current_price = None

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

        # 1) Underwater overlay (capital protection) — decisive and deterministic
        if distress_level in ("MEDIUM", "SEVERE") and holding_horizon not in ("LONG", "INVEST", "POSITION"):
            if defensive_hard_stop is not None and current_price is not None and current_price <= defensive_hard_stop:
                action = "EXIT"
                urgency = "HIGH"
                rationale = "Vị thế đang underwater và đã chạm/vi phạm Hard Stop phòng thủ → ưu tiên thoát để chặn tail-risk."
                if defensive_hard_stop is not None:
                    rationale += f" (Hard stop ~ {defensive_hard_stop:.2f})"
            else:
                action = "TRIM"
                urgency = "HIGH" if distress_level == "SEVERE" else "MED"
                rationale = "Vị thế đang underwater đáng kể → ưu tiên giảm tỷ trọng; không add cho tới khi reclaim cấu trúc."
                if (defensive_reclaim_zone_lo is not None) and (defensive_reclaim_zone_hi is not None) and (defensive_reclaim_zone_lo != defensive_reclaim_zone_hi):
                    rationale += f" (Reclaim zone: {defensive_reclaim_zone_lo:.2f} – {defensive_reclaim_zone_hi:.2f})"
                elif defensive_reclaim_level is not None:
                    rationale += f" (Reclaim > ~{defensive_reclaim_level:.2f})"

        # 2) Normal holding logic when not in distress
        if action == "WAIT":
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
        # Slot reserved for future Portfolio/Allocation module.
        # Keep deterministic placeholders (no sizing logic here).
        "allocation": {
            "target_size_pct_nav": None,
            "max_size_pct_nav": None,
            "notes": "-",
        },
    }

    # Step 8: normalize pack contract (fail-safe)
    try:
        from inception.core.contracts import normalize_decision_pack
        pack = normalize_decision_pack(pack)
    except Exception:
        pass

    return pack

