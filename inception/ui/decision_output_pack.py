from __future__ import annotations

"""DecisionOutputPack builder (UI-side only).

Goal:
- Normalize analysis_pack['DecisionPack'] (core output) into a stable, UI-friendly structure.
- Do NOT compute new signals.
- Do NOT override or reinterpret Decision Layer; this is a display adapter.
"""

from typing import Any, Dict


def _safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _safe_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _pick_first(d: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) not in (None, ""):
            return d.get(k)
    return None


def _normalize_action(raw: Any) -> str:
    s = _safe_text(raw).upper()
    if s in {"MUA", "BUY", "LONG"}:
        return "BUY"
    if s in {"GIU", "GIỮ", "HOLD"}:
        return "HOLD"
    if s in {"CHO", "WAIT", "OBSERVE", "QUAN SÁT"}:
        return "WAIT"
    if s in {"GIAM", "TRIM", "REDUCE", "HẠ TỶ TRỌNG"}:
        return "TRIM"
    if s in {"BAN", "BÁN", "EXIT", "CUT"}:
        return "EXIT"
    if s in {"AVOID", "NO_TRADE", "STAY_OUT"}:
        return "AVOID"
    if s in {"HEDGE"}:
        return "HEDGE"

    if "TRIM" in s or "GIẢM" in s:
        return "TRIM"
    if "EXIT" in s or "BÁN" in s:
        return "EXIT"
    if "BUY" in s or "MUA" in s:
        return "BUY"
    if "HOLD" in s or "GIỮ" in s:
        return "HOLD"
    if "WAIT" in s or "QUAN" in s:
        return "WAIT"
    return "UNKNOWN"


def _conviction_tier_from_score(score: Any) -> str:
    v = _safe_float(score, default=None)
    if v is None:
        return "UNKNOWN"
    # score is usually 0–10
    if v >= 6.0:
        return "GOD"
    if v >= 5.0:
        return "VERY_HIGH"
    if v >= 4.0:
        return "HIGH"
    if v >= 3.0:
        return "TRADEABLE"
    if v >= 2.0:
        return "LOW"
    return "NONE"


def _risk_posture_from_action(action: str) -> str:
    if action in {"EXIT", "TRIM", "AVOID"}:
        return "DEFENSIVE"
    if action in {"BUY"}:
        return "OFFENSIVE"
    return "NEUTRAL"


def compute_decision_output_pack(analysis_pack: Dict[str, Any]) -> Dict[str, Any]:
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    dp = ap.get("DecisionPack") if isinstance(ap.get("DecisionPack"), dict) else {}

    # Mode (prefer DashboardSummaryPack if present)
    dsum = ap.get("DashboardSummaryPack") if isinstance(ap.get("DashboardSummaryPack"), dict) else {}
    mode = _safe_text(dsum.get("mode") or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        mode = _safe_text(_pick_first(dp, ["Mode", "mode", "PositionMode"]) or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        mode = "UNKNOWN"

    # Primary action
    raw_action = _pick_first(dp, ["PrimaryAction", "Action", "Decision", "Recommendation", "FinalAction"])
    primary_action = _normalize_action(raw_action)

    # Conviction tier
    raw_conv = _pick_first(dp, ["ConvictionTier", "ConvictionScore", "Conviction", "FinalConviction", "Score"])
    tier = _safe_text(raw_conv).upper()
    known = {"GOD", "VERY_HIGH", "HIGH", "TRADEABLE", "LOW", "NONE"}
    conviction_tier = tier if tier in known else _conviction_tier_from_score(raw_conv)

    risk_posture = _risk_posture_from_action(primary_action)

    # Key levels (best-effort)
    key_levels = {
        "reclaim": _safe_text(_pick_first(dp, ["ReclaimZone", "Reclaim", "ReclaimLine", "ReclaimRange"]) or ""),
        "risk": _safe_text(_pick_first(dp, ["RiskZone", "Risk", "RiskLine", "Invalidation", "CutLine"]) or ""),
        "trigger": _safe_text(_pick_first(dp, ["Trigger", "TriggerLine", "Confirm", "Confirmation"]) or ""),
    }

    # Reason keys: if core provides, keep; else coarse-map from action
    reason_keys = dp.get("ReasonKeys") if isinstance(dp.get("ReasonKeys"), list) else []
    if not reason_keys:
        reason_keys = [f"DECISION_{primary_action}"] if primary_action != "UNKNOWN" else ["DECISION_UNKNOWN"]

    return {
        "version": "1.0",
        "mode": mode,
        "primary_action": primary_action,
        "conviction_tier": conviction_tier,
        "risk_posture": risk_posture,
        "key_levels": key_levels,
        "reason_keys": reason_keys,
    }
