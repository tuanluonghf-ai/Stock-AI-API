from __future__ import annotations

"""NarrativeOutputPack builder (UI-side only).

Build deterministic phrase keys from normalized packs.
"""

from typing import Any, Dict, List


def _safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _dna_key(dna_bucket: str) -> str:
    b = _safe_text(dna_bucket).strip().lower()
    if "illiquid" in b or "nhieu" in b or "noisy" in b:
        return "DNA_ILLIQUID"
    if "event" in b or "gap" in b:
        return "DNA_EVENT"
    if "glass" in b or "cannon" in b:
        return "DNA_GLASS"
    if "defensive" in b:
        return "DNA_DEFENSIVE"
    return "DNA_BALANCED"


def _zone_key(zone_now: str) -> str:
    z = _safe_text(zone_now).upper()
    if z in {"POSITIVE", "RECLAIM", "RISK", "NEUTRAL"}:
        return f"ZONE_{z}"
    return "ZONE_NEUTRAL"


def _bias_key(bias: str) -> str:
    b = _safe_text(bias).upper()
    if b in {"AGGRESSIVE", "CAUTIOUS", "DEFENSIVE"}:
        return f"BIAS_{b}"
    return "BIAS_CAUTIOUS"


def _size_key(size_hint: str) -> str:
    s = _safe_text(size_hint).upper()
    if s in {"FULL", "PARTIAL", "PROBE", "FLAT"}:
        return f"SIZE_{s}"
    return "SIZE_PROBE"


def _decision_key(primary_action: str) -> str:
    a = _safe_text(primary_action).upper()
    if a in {"BUY", "HOLD", "WAIT", "TRIM", "EXIT", "AVOID"}:
        return f"DECISION_{a}"
    return "DECISION_UNKNOWN"


def build_narrative_output_pack(
    ticker: str,
    analysis_pack: Dict[str, Any],
) -> Dict[str, Any]:
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    zp = ap.get("_ZonePack") if isinstance(ap.get("_ZonePack"), dict) else {}
    ab = ap.get("_ActionBiasPack") if isinstance(ap.get("_ActionBiasPack"), dict) else {}
    dop = ap.get("_DecisionOutputPack") if isinstance(ap.get("_DecisionOutputPack"), dict) else {}

    mode = _safe_text(dop.get("mode") or ab.get("mode") or "UNKNOWN").upper()
    if mode not in {"HOLDING", "FLAT", "UNKNOWN"}:
        mode = "UNKNOWN"

    dna_bucket = _safe_text(ab.get("dna_bucket") or "Balanced")
    zone_now = _safe_text(zp.get("zone_now") or "NEUTRAL").upper()
    bias = _safe_text(ab.get("bias") or "CAUTIOUS").upper()
    size_hint = _safe_text(ab.get("size_hint") or "PROBE").upper()
    primary_action = _safe_text(dop.get("primary_action") or "UNKNOWN").upper()

    keys: List[str] = [
        _dna_key(dna_bucket),
        _zone_key(zone_now),
        _bias_key(bias),
        _size_key(size_hint),
        _decision_key(primary_action),
        "SAFETY_LINE",  # always include an uncertainty/safety sentence
    ]

    ctx = {
        "ticker": _safe_text(ticker).upper(),
        "mode": mode,
        "dna_bucket": dna_bucket,
        "zone_now": zone_now,
        "bias": bias,
        "size_hint": size_hint,
        "primary_action": primary_action,
    }

    # Optional levels for future templates (may be empty)
    levels = dop.get("key_levels") if isinstance(dop.get("key_levels"), dict) else {}
    ctx["reclaim"] = _safe_text(levels.get("reclaim") or "")
    ctx["risk"] = _safe_text(levels.get("risk") or "")
    ctx["trigger"] = _safe_text(levels.get("trigger") or "")

    return {
        "version": "1.0",
        "context": ctx,
        "keys": keys,
        "render_policy": {
            "max_sentences": 5,
            "must_include_uncertainty": True,
        },
    }
