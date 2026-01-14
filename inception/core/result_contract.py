"""ResultPack contract finalizer (v1).

This module enforces a stable Engine → UI boundary.

Principles
  - Facts-first: compute packs using Python when possible.
  - Non-raising: never crash the app; fall back to safe stubs.
  - UI-safe: ensure expected packs exist with predictable shapes.

This is intentionally conservative: it tries to *preserve* existing values
whenever they look valid, and only repairs when packs are missing or drifted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_text(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _schema_is(pack: Any, schema: str) -> bool:
    p = _as_dict(pack)
    return _safe_text(p.get("schema") or "").strip() == schema


def _replace_na(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("N/A", "–")
    if isinstance(value, list):
        return [_replace_na(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_replace_na(v) for v in value)
    if isinstance(value, dict):
        return {k: _replace_na(v) for k, v in value.items()}
    return value


def _stub_tradeplan_pack_v1(ap: Dict[str, Any]) -> Dict[str, Any]:
    """Safe stub that keeps UI rendering stable.

    If we cannot compute TradePlanPack, we populate from PrimarySetup so the
    dashboard does not go blank.
    """
    primary = _as_dict(ap.get("PrimarySetup"))
    mode = _safe_text((_as_dict(ap.get("PositionStatePack")).get("mode") or "FLAT")).strip().upper() or "FLAT"
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    plan_primary = {
        "type": _safe_text(primary.get("Name") or "-").strip() or "-",
        "name": _safe_text(primary.get("Name") or "-").strip() or "-",
        "state": "WAIT",
        "entry": primary.get("Entry"),
        "stop": primary.get("Stop"),
        "tp1": primary.get("TP1") if primary.get("TP1") is not None else primary.get("TP"),
        "rr_actual": primary.get("RR"),
    }
    return {
        "schema": "TradePlanPack.v1",
        "mode": mode,
        "plan_primary": plan_primary,
        "plan_alt": {},
        "plans_all": [],
        "gates": {"breakout": "WAIT", "volume": "WAIT", "rr": "WAIT", "structure": "WAIT"},
        "triggers": {"breakout": "WAIT", "volume": "WAIT", "rr": "WAIT", "structure": "WAIT"},
        "notes": ["TradePlanPack stub (fallback)."],
    }


def _stub_decision_pack_v1(ap: Dict[str, Any]) -> Dict[str, Any]:
    pos = _as_dict(ap.get("PositionStatePack"))
    mode = _safe_text(pos.get("mode") or "FLAT").strip().upper() or "FLAT"
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    return {
        "schema": "DecisionPack.v1",
        "mode": mode,
        "action": "WAIT" if mode == "FLAT" else "HOLD",
        "urgency": "MED",
        "constraints": ["-"],
        "flags": {},
        "weakness": [],
    }


def _stub_position_manager_pack_v1(ap: Dict[str, Any]) -> Dict[str, Any]:
    dp = _as_dict(ap.get("DecisionPack"))
    pos = _as_dict(ap.get("PositionStatePack"))
    mode = _safe_text(pos.get("mode") or dp.get("mode") or "FLAT").strip().upper() or "FLAT"
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    action = _safe_text(dp.get("action") or ("WAIT" if mode == "FLAT" else "HOLD")).strip().upper() or "WAIT"
    return {
        "schema": "PositionManagerPack.v1",
        "mode": mode,
        "action": action,
        "guidance": "-",
        "trim_pct_of_position": None,
        "stop_suggest": None,
        "size_cap_pct_nav": None,
        "position_size_pct_nav": pos.get("position_size_pct_nav"),
        "Levels": {},
        "Notes": [],
    }


def _stub_dashboard_summary_pack_v1() -> Dict[str, Any]:
    return {
        "schema": "DashboardSummaryPack.v1",
        "CurrentStatusCard": {
            "state_capsule_line": "-",
            "master_total": None,
            "conviction": None,
            "insight_line": "-",
            "policy_hint_line": "-",
            "gate_line": "-",
            "next_step": "Theo dõi và chờ thêm dữ liệu.",
            "plan_status": "WAIT",
            "plan_reason": "-",
            "triggers": {"breakout": "WAIT", "volume": "WAIT", "rr": "WAIT", "structure": "WAIT"},
            "risk_flags": ["Không có"],
            "decision": {},
            "data_quality": {"schema": "DataQualityPack.v1", "error_count": 0, "warn_count": 0, "issues": []},
        },
    }


def finalize_result_pack_v1(result_in: Any) -> Dict[str, Any]:
    """Finalize ResultPack to a UI-safe contract.

    Returns a dict (possibly the same object) with repairs applied.
    """
    result = _as_dict(result_in)
    if not result:
        return {"Error": "Invalid ResultPack"}

    issues: List[str] = []

    # --- Normalize AnalysisPack ---
    ap = _as_dict(result.get("AnalysisPack"))
    try:
        from inception.core.contracts import normalize_analysis_pack

        ap = normalize_analysis_pack(ap)
    except Exception:
        ap = ap if isinstance(ap, dict) else {}
        issues.append("AnalysisPack normalization failed")
    result["AnalysisPack"] = ap

    # --- Normalize CharacterPack (Modules.character) ---
    mods = _as_dict(result.get("Modules"))
    cp = _as_dict(mods.get("character"))
    try:
        from inception.core.contracts import normalize_character_pack

        cp = normalize_character_pack(cp)
    except Exception:
        cp = cp if isinstance(cp, dict) else {}
        issues.append("CharacterPack normalization failed")
    if mods:
        mods["character"] = cp
        result["Modules"] = mods

    # --- Ensure TradePlanPack.v1 ---
    tpp = ap.get("TradePlanPack")
    if not _schema_is(tpp, "TradePlanPack.v1"):
        computed = None
        try:
            from inception.core.tradeplan_pack import compute_trade_plan_pack_v1

            computed = compute_trade_plan_pack_v1(ap, cp)
        except Exception:
            computed = None
        if _schema_is(computed, "TradePlanPack.v1"):
            tpp = computed
            issues.append("TradePlanPack repaired (recomputed)")
        else:
            tpp = _stub_tradeplan_pack_v1(ap)
            issues.append("TradePlanPack repaired (stub)")
        ap["TradePlanPack"] = tpp

    # Normalize for downstream consumers
    try:
        from inception.core.contracts import normalize_tradeplan_pack

        ap["TradePlanPack"] = normalize_tradeplan_pack(ap.get("TradePlanPack"))
    except Exception:
        pass

    # --- Ensure DecisionPack.v1 ---
    dp = ap.get("DecisionPack")
    if not _schema_is(dp, "DecisionPack.v1"):
        computed = None
        try:
            from inception.core.decision_pack import compute_decision_pack_v1

            computed = compute_decision_pack_v1(ap, ap.get("TradePlanPack"))
        except Exception:
            computed = None
        if _schema_is(computed, "DecisionPack.v1"):
            dp = computed
            issues.append("DecisionPack repaired (recomputed)")
        else:
            dp = _stub_decision_pack_v1(ap)
            issues.append("DecisionPack repaired (stub)")
        ap["DecisionPack"] = dp

    try:
        from inception.core.contracts import normalize_decision_pack

        ap["DecisionPack"] = normalize_decision_pack(ap.get("DecisionPack"))
    except Exception:
        pass

    # --- Ensure PositionManagerPack.v1 ---
    pmp = ap.get("PositionManagerPack")
    if not _schema_is(pmp, "PositionManagerPack.v1"):
        computed = None
        try:
            from inception.core.position_manager_pack import compute_position_manager_pack_v1

            computed = compute_position_manager_pack_v1(ap, ap.get("TradePlanPack"), ap.get("DecisionPack"))
        except Exception:
            computed = None
        if _schema_is(computed, "PositionManagerPack.v1"):
            pmp = computed
            issues.append("PositionManagerPack repaired (recomputed)")
        else:
            pmp = _stub_position_manager_pack_v1(ap)
            issues.append("PositionManagerPack repaired (stub)")
        ap["PositionManagerPack"] = pmp

    try:
        from inception.core.contracts import normalize_position_manager_pack

        ap["PositionManagerPack"] = normalize_position_manager_pack(ap.get("PositionManagerPack"))
    except Exception:
        pass

    # --- Ensure DashboardSummaryPack.v1 ---
    dsp = result.get("DashboardSummaryPack")
    if not _schema_is(dsp, "DashboardSummaryPack.v1"):
        computed = None
        try:
            from inception.core.dashboard_pack import compute_dashboard_summary_pack_v1

            computed = compute_dashboard_summary_pack_v1(ap, cp)
        except Exception:
            computed = None
        if _schema_is(computed, "DashboardSummaryPack.v1"):
            dsp = computed
            issues.append("DashboardSummaryPack repaired (recomputed)")
        else:
            dsp = _stub_dashboard_summary_pack_v1()
            issues.append("DashboardSummaryPack repaired (stub)")
        result["DashboardSummaryPack"] = dsp

    # --- Diagnostics: ContractPack (optional; UI may ignore) ---
    try:
        ok = True
        # Mark not-ok if we had to stub anything
        if any("(stub)" in x for x in issues):
            ok = False
        result["ContractPack"] = {
            "schema": "ResultContractPack.v1",
            "ok": bool(ok),
            "issues": issues[:12],
        }
        # Also attach under AnalysisPack for convenience
        ap["ContractPack"] = result["ContractPack"]
    except Exception:
        pass

    result["AnalysisPack"] = ap

    # Final UI-safe cleanup: remove any "N/A" tokens from UI-facing strings.
    try:
        result = _replace_na(result)
    except Exception:
        pass

    return result
