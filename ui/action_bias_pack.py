from __future__ import annotations

"""ActionBiasPack builder (UI-side only).

Purpose
- Bridge DNA bucket × ZonePack.zone_now → Action Bias + Position Size Guidance.
- MUST NOT compute new signals; only transforms existing packs/labels.
- MUST NOT override Decision Layer; bias/size are guidance only.

Notes
- Designed for INCEPTION UI/Narrative refactor (Phase F).
"""

from typing import Any, Dict


def _safe_text(x: Any) -> str:
    try:
        return str(x).strip()
    except Exception:
        return ""


def _bucket_dna(class_name: str) -> str:
    s = _safe_text(class_name).upper()
    if "ILL" in s or "THANH KHOAN" in s:
        return "Illiquid/Noisy"
    if "EVENT" in s or "GAP" in s:
        return "Event/Gap-Prone"
    if "GLASS" in s or "CANNON" in s:
        return "Glass Cannon"
    if "DEF" in s or "SAFE" in s:
        return "Defensive"
    return "Balanced"


def compute_action_bias_pack(
    analysis_pack: Dict[str, Any],
    zone_pack: Dict[str, Any],
    character_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    zp = zone_pack if isinstance(zone_pack, dict) else {}
    cp = character_pack if isinstance(character_pack, dict) else {}

    # Mode (prefer DashboardSummaryPack if present)
    dsum = ap.get("DashboardSummaryPack") if isinstance(ap.get("DashboardSummaryPack"), dict) else {}
    mode = _safe_text(dsum.get("mode") or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        pos = ap.get("Position") if isinstance(ap.get("Position"), dict) else {}
        mode = _safe_text(pos.get("mode") or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        mode = "UNKNOWN"

    # DNA bucket
    class_name = _safe_text(
        (cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "")
        if isinstance(cp, dict) else ""
    )
    dna_bucket = _bucket_dna(class_name)

    # Zone now
    zone_now = _safe_text(zp.get("zone_now") or "NEUTRAL").upper()
    if zone_now not in {"POSITIVE", "RECLAIM", "RISK", "NEUTRAL"}:
        zone_now = "NEUTRAL"

    # Bias (stable rules)
    if zone_now == "RISK":
        bias = "DEFENSIVE"
        rationale_keys = ["ZONE_RISK"]
    elif zone_now == "RECLAIM":
        bias = "CAUTIOUS"
        rationale_keys = ["ZONE_RECLAIM"]
    elif zone_now == "POSITIVE":
        bias = "AGGRESSIVE"
        rationale_keys = ["ZONE_POSITIVE"]
    else:
        bias = "CAUTIOUS"
        rationale_keys = ["ZONE_NEUTRAL"]

    # Size hint (mode-aware, no override)
    riskier_bucket = dna_bucket in {"Glass Cannon", "Event/Gap-Prone", "Illiquid/Noisy"}

    if mode != "HOLDING":
        # FLAT / entering
        if bias == "AGGRESSIVE":
            size_hint = "PARTIAL" if riskier_bucket else "FULL"
            rationale_keys += ["SIZE_ENTER"]
        elif bias == "CAUTIOUS":
            size_hint = "PROBE"
            rationale_keys += ["SIZE_PROBE"]
        else:
            size_hint = "FLAT"
            rationale_keys += ["SIZE_FLAT"]
    else:
        # HOLDING
        if bias == "AGGRESSIVE":
            size_hint = "PARTIAL"   # hold core; add only on confirmation
            rationale_keys += ["SIZE_HOLD_ADD_COND"]
        elif bias == "CAUTIOUS":
            size_hint = "PROBE"     # hold/observe; avoid increasing commitment
            rationale_keys += ["SIZE_HOLD_OBSERVE"]
        else:
            size_hint = "FLAT"      # reduce commitment posture
            rationale_keys += ["SIZE_REDUCE_RISK"]

    return {
        "version": "1.0",
        "mode": mode,
        "dna_bucket": dna_bucket,
        "zone_now": zone_now,
        "bias": bias,
        "size_hint": size_hint,
        "one_liner": "",
        "rationale_keys": rationale_keys,
    }
