from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from inception.core.helpers import _safe_bool, _safe_float, _safe_text
from inception.core.stability.state_store import load_state, save_state


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _extract_primary_plan(tpp: Dict[str, Any]) -> Dict[str, Any]:
    # Common shapes we have seen across INCEPTION versions
    for k in ("plan_primary", "primary_setup", "primary", "PrimarySetup", "plan", "Plan"):
        v = tpp.get(k)
        if isinstance(v, dict):
            return v
    return {}


def _raw_plan_present(tpp: Dict[str, Any]) -> bool:
    primary = _extract_primary_plan(tpp)
    name = _safe_text(primary.get("name") or primary.get("Name") or tpp.get("name") or tpp.get("Name")).strip()
    triggers = primary.get("triggers") if isinstance(primary.get("triggers"), dict) else {}
    # If legacy packs have triggers directly on TradePlanPack
    if not triggers and isinstance(tpp.get("triggers"), dict):
        triggers = tpp.get("triggers")  # type: ignore[assignment]
    return bool(name) or bool(triggers)


def _structure_breakdown(analysis_pack: Dict[str, Any], tpp: Dict[str, Any]) -> bool:
    # Best-effort detection. If unknown, return False (do not invalidate by accident).
    # 1) explicit flags
    for k in ("structure_breakdown", "StructureBreakdown", "StructureBreakdownFlag", "roe_broken", "ROEBroken"):
        if k in analysis_pack:
            return _safe_bool(analysis_pack.get(k))
    # 2) gates / guardrails inside TradePlanPack
    gates = _ensure_dict(tpp.get("gates") or tpp.get("guardrails") or {})
    for k in ("structure_ok", "StructureOK", "roe_ok", "ROE_OK"):
        if k in gates:
            ok = _safe_bool(gates.get(k))
            return not ok
    # 3) plan completeness hard fail status (if present)
    pc = _ensure_dict(tpp.get("plan_completeness") or tpp.get("PlanCompleteness") or {})
    st = _safe_text(pc.get("status") or pc.get("Status") or "").strip().upper()
    if st in ("FAIL_HARD", "INVALID", "INVALIDATED"):
        return True
    return False


def _trigger_soft_fail(tpp: Dict[str, Any]) -> Tuple[bool, str]:
    primary = _extract_primary_plan(tpp)
    triggers = primary.get("triggers") if isinstance(primary.get("triggers"), dict) else {}
    if not triggers and isinstance(tpp.get("triggers"), dict):
        triggers = tpp.get("triggers")  # type: ignore[assignment]

    if not triggers:
        return False, ""

    # If any trigger is explicitly False → soft fail
    for k, v in triggers.items():
        if isinstance(v, bool) and v is False:
            return True, f"trigger_fail:{k}"
        # Some legacy shapes might store "PASS"/"FAIL" strings
        if isinstance(v, str) and v.strip().upper() in ("FAIL", "NO", "FALSE"):
            return True, f"trigger_fail:{k}"
    return False, ""


def _load_prev_plan_state(*, ticker: str) -> str:
    st = load_state(scope="plan", ticker=ticker, default={})
    v = st.get("prev_plan_state") if isinstance(st, dict) else ""
    return _safe_text(v).strip().upper()


def _save_prev_plan_state(*, ticker: str, prev_plan_state: str) -> None:
    save_state(scope="plan", ticker=ticker, state={"prev_plan_state": _safe_text(prev_plan_state).strip().upper()})


def apply_plan_stability(*, ticker: str, analysis_pack: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Plan Persistence governor (scope-locked).

    - Does NOT modify indicators.
    - Does NOT mutate raw TradePlanPack.
    - Computes stable plan_state: ACTIVE | PAUSED | INVALIDATED.
    - Uses State Store for prev_plan_state persistence.
    """
    ap = _ensure_dict(analysis_pack)
    tpp = _ensure_dict(ap.get("TradePlanPack"))

    prev_state = _load_prev_plan_state(ticker=ticker) or ""
    raw_present = _raw_plan_present(tpp)
    hard_invalid = _structure_breakdown(ap, tpp)

    # Default inference if no prior state
    inferred_prev = prev_state or ("ACTIVE" if raw_present else "PAUSED")

    plan_state = inferred_prev
    reason = "carryover"

    if hard_invalid:
        plan_state = "INVALIDATED"
        reason = "structure_breakdown"
    else:
        if raw_present:
            soft_fail, why = _trigger_soft_fail(tpp)
            if soft_fail:
                plan_state = "PAUSED"
                reason = why or "trigger_fail"
            else:
                plan_state = "ACTIVE"
                reason = "revalidated"
        else:
            # No plan generated today: never show "–" as disappearance; pause if previously active.
            if inferred_prev == "ACTIVE":
                plan_state = "PAUSED"
                reason = "no_plan_generated"
            else:
                plan_state = inferred_prev or "PAUSED"
                reason = "still_no_plan"

    # Persist
    _save_prev_plan_state(ticker=ticker, prev_plan_state=plan_state)

    pack = {
        "schema": "PlanStabilityPack.v1",
        "prev_plan_state": prev_state or "",
        "raw_plan_present": bool(raw_present),
        "plan_state": plan_state,
        "reason": reason,
        "notes": [],
        "carryover_plan_id": "",
    }
    return pack
