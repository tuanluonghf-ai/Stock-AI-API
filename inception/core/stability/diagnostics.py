from __future__ import annotations

from typing import Any, Dict, List

from inception.core.helpers import _safe_bool, _safe_text
from inception.core.stability.phrase_bank import pick_phrase


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

    # Phrase bank rotation with distress-aware tone
    ticker = _safe_text(ap.get("Ticker") or ap.get("ticker") or "").strip().upper()

    if decision_changed:
        variants = [
            "Hành động được giữ ổn định để tránh nhiễu ngắn hạn; ưu tiên chờ thêm xác nhận trước khi đổi nhịp.",
            "Tạm thời giữ quyết định để giảm rủi ro “giật cục”; chờ tín hiệu xác nhận rõ hơn.",
            "Giữ nhịp thận trọng trong ngắn hạn; khi điều kiện xác nhận đầy đủ, hệ thống sẽ cập nhật hành động.",
            "Hôm nay ưu tiên ổn định hơn là phản ứng vội; cần thêm xác nhận để chuyển pha an toàn.",
        ]
        distress_variants = [
            "Ưu tiên bảo toàn trong giai đoạn bất lợi; quyết định được giữ ổn định để tránh phát sinh rủi ro thêm.",
            "Khi vị thế đang chịu áp lực, hệ thống chọn giữ nhịp phòng thủ thay vì phản ứng vội.",
            "Giai đoạn này nên đi chậm và chắc; quyết định được giữ để tránh sai lệch do nhiễu ngắn hạn.",
        ]
        k = f"{ticker}|decision|{raw_action}->{stable_action}|{_safe_text(dsp.get('reason'))}"
        explain_line = pick_phrase(
            key=k,
            variants=distress_variants if distress in ("MEDIUM", "SEVERE") else variants
        )

    elif plan_state and plan_state in ("PAUSED", "INVALIDATED"):
        if plan_state == "PAUSED":
            variants = [
                "Kế hoạch tạm gián đoạn do thiếu xác nhận; ưu tiên quan sát và chờ điều kiện quay lại.",
                "Setup vẫn còn, nhưng xác nhận chưa đủ; tạm dừng kế hoạch để tránh vào lệnh khi xác suất chưa đẹp.",
                "Tạm thời đứng ngoài kế hoạch để giảm nhiễu; khi điều kiện xác nhận phục hồi, kế hoạch sẽ kích hoạt lại.",
                "Kế hoạch được giữ ở trạng thái tạm dừng; chờ thêm xác nhận để quay lại đúng kỷ luật.",
            ]
            distress_variants = [
                "Trong bối cảnh vị thế đang chịu áp lực, kế hoạch được tạm dừng để ưu tiên bảo toàn.",
                "Khi thị trường đi ngược, việc tạm dừng kế hoạch giúp hạn chế rủi ro phát sinh thêm.",
            ]
            k = f"{ticker}|plan|PAUSED|{_safe_text(psp.get('reason'))}"
            explain_line = pick_phrase(
                key=k,
                variants=distress_variants if distress in ("MEDIUM", "SEVERE") else variants
            )
        else:
            variants = [
                "Kế hoạch bị vô hiệu do điều kiện cốt lõi không còn phù hợp; ưu tiên bảo toàn và đợi tái lập cấu trúc.",
                "Điều kiện nền tảng đã thay đổi; kế hoạch cũ không còn hợp lệ. Tạm thời ưu tiên quản trị rủi ro.",
                "Kế hoạch hiện tại không còn đáp ứng rule-of-engagement; chờ cấu trúc thiết lập lại rồi mới hành động.",
                "Trong bối cảnh này, ưu tiên phòng thủ: kế hoạch cũ bị vô hiệu và cần một cấu trúc mới để tái kích hoạt.",
            ]
            distress_variants = [
                "Khi vị thế đang bất lợi, kế hoạch cũ được vô hiệu để tránh kéo dài rủi ro.",
                "Ưu tiên phòng thủ trong giai đoạn áp lực; kế hoạch sẽ chỉ tái kích hoạt khi cấu trúc mới hình thành.",
            ]
            k = f"{ticker}|plan|INVALIDATED|{_safe_text(psp.get('reason'))}"
            explain_line = pick_phrase(
                key=k,
                variants=distress_variants if distress in ("MEDIUM", "SEVERE") else variants
            )

    # NOTE: do not show explain line for regime changes alone (UI hardening).

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
