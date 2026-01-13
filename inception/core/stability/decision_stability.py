# ============================================================
# FILE: inception/core/stability/decision_stability.py
# ============================================================
"""Decision Stability (Hysteresis governor).

Purpose
-------
Reduce action "flip-flop" caused by small input drifts, WITHOUT modifying core
indicator logic or the raw DecisionPack.

Design
------
- Read: AnalysisPack + (raw) DecisionPack + (raw) TradePlanPack gates
- Output: DecisionStabilityPack (stable_action + reasons)
- Optional persistence: store last stable action per ticker for cross-run stability.

This module is intentionally conservative and fail-safe.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from inception.core.helpers import _safe_bool, _safe_text

# ----------------------------
# Persistence (single source of truth via state_store)
# ----------------------------

def _load_prev_action(*, ticker: str) -> Optional[str]:
    st = load_state(scope="decision", ticker=ticker, default={})
    v = st.get("prev_action") if isinstance(st, dict) else ""
    v = _safe_text(v).strip().upper()
    return v or None


def _save_prev_action(*, ticker: str, action: str) -> None:
    save_state(scope="decision", ticker=ticker, state={"prev_action": _safe_text(action).strip().upper()})


# ----------------------------
# Extraction helpers
# ----------------------------

def _pick_pack(analysis_pack: Dict[str, Any], name: str) -> Dict[str, Any]:
    p = analysis_pack.get(name) if isinstance(analysis_pack, dict) else None
    return p if isinstance(p, dict) else {}


def _extract_tradeplan_gates(analysis_pack: Dict[str, Any]) -> Dict[str, str]:
    tpp = _pick_pack(analysis_pack, "TradePlanPack")
    pp = tpp.get("plan_primary") if isinstance(tpp.get("plan_primary"), dict) else {}
    gates = pp.get("gates") if isinstance(pp.get("gates"), dict) else {}

    def _g(k: str) -> str:
        return _safe_text(gates.get(k) or "").strip().upper()

    # Normalize with conservative defaults
    return {
        "structure": _g("structure") or "WAIT",
        "volume": _g("volume") or "WAIT",
        "trigger": _g("trigger") or "WAIT",
        "rr": _g("rr") or "WAIT",
        "plan": _g("plan") or "WAIT",
    }


def _extract_raw_action(analysis_pack: Dict[str, Any]) -> str:
    dp = _pick_pack(analysis_pack, "DecisionPack")
    a = dp.get("action") if isinstance(dp, dict) else None
    if a is None:
        a = dp.get("Action") if isinstance(dp, dict) else None
    s = _safe_text(a).strip().upper()
    return s or "WAIT"


def _extract_distress(analysis_pack: Dict[str, Any]) -> str:
    pos = _pick_pack(analysis_pack, "PositionStatePack")
    d = _safe_text(pos.get("distress_level") or pos.get("DistressLevel") or "").strip().upper()
    if d:
        return d
    # fallback: tradeplan holding_overlay
    tpp = _pick_pack(analysis_pack, "TradePlanPack")
    ho = tpp.get("holding_overlay") if isinstance(tpp.get("holding_overlay"), dict) else {}
    d2 = _safe_text(ho.get("distress_level") or ho.get("DistressLevel") or "").strip().upper()
    return d2 or "-"


def _extract_risk_shock(analysis_pack: Dict[str, Any]) -> bool:
    dna = _pick_pack(analysis_pack, "DNAPack")
    flags = dna.get("flags") if isinstance(dna.get("flags"), dict) else {}
    if _safe_bool(flags.get("gap_prone")):
        return True
    mods = dna.get("modifiers") if isinstance(dna.get("modifiers"), list) else []
    return any(_safe_text(x).strip().upper() == "GAP" for x in mods)


def _count_negatives(*, gates: Dict[str, str], distress: str, risk_shock: bool) -> Tuple[int, Dict[str, bool]]:
    structure_fail = gates.get("structure") == "FAIL"
    volume_fail = gates.get("volume") == "FAIL"
    trigger_fail = gates.get("trigger") == "FAIL"
    rr_fail = gates.get("rr") == "FAIL"

    distress_u = _safe_text(distress).strip().upper()
    distress_bad = distress_u in ("MEDIUM", "SEVERE")

    # Risk shock counts only when execution is already fragile.
    risk_bad = bool(risk_shock and (gates.get("structure") in ("WAIT", "FAIL") or gates.get("volume") == "FAIL"))

    flags = {
        "structure_fail": bool(structure_fail),
        "volume_fail": bool(volume_fail),
        "trigger_fail": bool(trigger_fail),
        "rr_fail": bool(rr_fail),
        "distress_bad": bool(distress_bad),
        "risk_shock": bool(risk_bad),
    }
    n = sum(1 for v in flags.values() if v)
    return n, flags


# ----------------------------
# Public API
# ----------------------------

def apply_decision_stability(
    *,
    ticker: str,
    analysis_pack: Dict[str, Any]= None,
) -> Dict[str, Any]:
    """Compute DecisionStabilityPack.

    Notes:
    - Uses last stable action from disk (optional) to create hysteresis.
    - Never mutates DecisionPack.
    - Safe-by-default: if anything missing, stable_action == raw_action.
    """

    t = _safe_text(ticker).strip().upper() or ""
    raw_action = _extract_raw_action(analysis_pack)

    prev_action = _load_prev_action(ticker=t)
    if prev_action is None:
        # Cold start: align prev_action with raw to avoid artificial hysteresis.
        prev_action = raw_action

    gates = _extract_tradeplan_gates(analysis_pack)
    distress = _extract_distress(analysis_pack)
    risk_shock = _extract_risk_shock(analysis_pack)
    negatives_count, neg_flags = _count_negatives(gates=gates, distress=distress, risk_shock=risk_shock)

    # BUY confirmation (conservative): require no FAIL on core execution gates.
    buy_confirmation = (
        gates.get("structure") == "PASS"
        and gates.get("volume") != "FAIL"
        and gates.get("trigger") != "FAIL"
        and gates.get("rr") != "FAIL"
        and gates.get("plan") != "FAIL"
    )

    stable_action = raw_action
    reason = "raw"
    confidence_delta = 0.0

    # Hard safety: EXIT always passes through.
    if raw_action == "EXIT":
        stable_action = "EXIT"
        reason = "hard_exit"
    else:
        # Fast de-escalation: if previously BUY, allow quick downgrade.
        if prev_action == "BUY" and raw_action in ("WAIT", "HOLD", "TRIM"):
            stable_action = raw_action
            reason = "fast_deescalate"

        # HOLD -> TRIM requires meaningful deterioration (>=2 negatives) OR structure FAIL.
        elif prev_action == "HOLD" and raw_action == "TRIM":
            if negatives_count >= 2 or neg_flags.get("structure_fail"):
                stable_action = "TRIM"
                reason = "confirmed_shift"
            else:
                stable_action = "HOLD"
                reason = "hysteresis_hold"
                confidence_delta = -0.3

        # WAIT/HOLD -> BUY requires setup + confirmation. Raw BUY implies setup; we enforce confirmation.
        elif prev_action in ("WAIT", "HOLD") and raw_action == "BUY":
            if buy_confirmation:
                stable_action = "BUY"
                reason = "confirmed_shift"
            else:
                stable_action = prev_action
                reason = "hysteresis_wait"
                confidence_delta = -0.3

        # TRIM -> HOLD requires negatives cleared (strict).
        elif prev_action == "TRIM" and raw_action == "HOLD":
            if negatives_count == 0 and gates.get("structure") == "PASS":
                stable_action = "HOLD"
                reason = "confirmed_shift"
            else:
                stable_action = "TRIM"
                reason = "hysteresis_hold"
                confidence_delta = -0.2

    pack: Dict[str, Any] = {
        "schema": "DecisionStabilityPack.v1",
        "prev_action": prev_action,
        "raw_action": raw_action,
        "stable_action": stable_action,
        "reason": reason,
        "gates": {
            "negatives_count": int(negatives_count),
            "required": 2,
            "buy_confirmation": bool(buy_confirmation),
            "neg_flags": neg_flags,
            "tradeplan_gates": gates,
            "distress_level": _safe_text(distress).strip().upper() or "-",
            "risk_shock": bool(risk_shock),
        },
        "confidence_delta": float(confidence_delta),
    }

    # Persist last stable action for cross-run hysteresis.
    try:
        if t:
            _save_prev_action(ticker=t, action=stable_action)
    except Exception:
        pass

    return pack