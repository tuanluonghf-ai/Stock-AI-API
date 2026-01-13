from __future__ import annotations

from typing import Any, Dict, List

from inception.core.helpers import _safe_bool, _safe_text


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def build_stability_diagnostics(*, analysis_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Build StabilityDiagnosticsPack (audit & QA).

    Scope:
      - No indicator changes.
      - No mutation of raw packs.
      - Produces a compact diff between raw vs stable outputs for UI/QA.
    """
    ap = _ensure_dict(analysis_pack)

    dp = _ensure_dict(ap.get("DecisionPack"))
    dsp = _ensure_dict(ap.get("DecisionStabilityPack"))

    tpp = _ensure_dict(ap.get("TradePlanPack"))
    psp = _ensure_dict(ap.get("PlanStabilityPack"))

    nap = _ensure_dict(ap.get("NarrativeAnchorPack"))

    raw_action = _safe_text(dp.get("action") or dp.get("Action") or "").strip().upper()
    stable_action = _safe_text(dsp.get("stable_action") or "").strip().upper()

    raw_plan_present = True
    try:
        # Heuristic: if no primary setup name and no triggers, treat as absent
        primary = tpp.get("plan_primary") if isinstance(tpp.get("plan_primary"), dict) else {}
        name = _safe_text(primary.get("name") or primary.get("Name") or "").strip()
        triggers = primary.get("triggers") if isinstance(primary.get("triggers"), dict) else {}
        raw_plan_present = bool(name) or bool(triggers)
    except Exception:
        raw_plan_present = True

    plan_state = _safe_text(psp.get("plan_state") or psp.get("stable_plan_state") or "").strip().upper()
    if not plan_state:
        # Backward compatibility if older key names exist
        plan_state = _safe_text(psp.get("stable_plan_state") or "").strip().upper()

    regime = _safe_text(nap.get("regime") or "").strip()
    anchor_phrase = _safe_text(nap.get("anchor_phrase") or "").strip()

    decision_changed = bool(raw_action and stable_action and raw_action != stable_action)

    # Plan state changed is best-effort; compare previous state if present in psp
    prev_plan_state = _safe_text(psp.get("prev_plan_state") or "").strip().upper()
    plan_state_changed = bool(prev_plan_state and plan_state and prev_plan_state != plan_state)

    prev_regime = _safe_text(nap.get("prev_regime") or "").strip()
    narrative_anchor_changed = bool(prev_regime and regime and prev_regime != regime)

    notes: List[str] = []
    if decision_changed:
        r = _safe_text(dsp.get("reason") or "").strip()
        if r:
            notes.append(f"decision: {r}")
    if plan_state and plan_state != "ACTIVE":
        r = _safe_text(psp.get("reason") or "").strip()
        if r:
            notes.append(f"plan: {r}")
    if narrative_anchor_changed:
        r = _safe_text(nap.get("reason") or "").strip()
        if r:
            notes.append(f"narrative: {r}")

    # UI explain line (short, controlled)
    explain_line = ""
    if decision_changed:
        explain_line = "Hành động được giữ ổn định để tránh nhiễu ngắn hạn; ưu tiên chờ thêm xác nhận."
    elif plan_state and plan_state in ("PAUSED", "INVALIDATED"):
        if plan_state == "PAUSED":
            explain_line = "Kế hoạch tạm gián đoạn do thiếu xác nhận; ưu tiên quan sát và chờ điều kiện quay lại."
        else:
            explain_line = "Kế hoạch bị vô hiệu do điều kiện cốt lõi không còn phù hợp; ưu tiên bảo toàn và đợi tái lập cấu trúc."
    elif narrative_anchor_changed:
        explain_line = "Bối cảnh vận động đã đổi chế độ; nội dung được neo lại theo regime mới để giữ mạch nhất quán."

    return {
        "schema": "StabilityDiagnosticsPack.v1",
        "decision_changed": bool(decision_changed),
        "plan_state_changed": bool(plan_state_changed),
        "narrative_anchor_changed": bool(narrative_anchor_changed),
        "notes": notes[:8],
        "raw_vs_stable": {
            "raw_action": raw_action or "",
            "stable_action": stable_action or "",
            "raw_plan_present": bool(raw_plan_present),
            "plan_state": plan_state or "",
            "regime": regime or "",
        },
        "ui_explain_line": explain_line,
        "anchor_phrase": anchor_phrase,
    }
