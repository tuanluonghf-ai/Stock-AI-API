"""Investor mapping core (Inception Pentagon + Personas).

Builds a UI-ready InvestorMappingPack from existing CharacterPack outputs.
No UI or LLM dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math

from inception.core.helpers import _safe_bool, _safe_float, _safe_text


PENTAGON_KEYS = ["TrendPower", "Explosive", "SafetyShield", "TradingFlow", "Adrenaline"]


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clip_0_10(x: float, default: float = 5.0) -> float:
    try:
        if not _is_finite(x):
            return float(default)
        return float(max(0.0, min(10.0, float(x))))
    except Exception:
        return float(default)


def _weighted_avg(values: Dict[str, float], weights: Dict[str, float], default: float = 5.0) -> float:
    total_w = 0.0
    acc = 0.0
    for k, w in weights.items():
        v = values.get(k)
        if not _is_finite(v):
            continue
        w_f = float(w)
        acc += float(v) * w_f
        total_w += w_f
    if total_w <= 0:
        return float(default)
    return float(acc / total_w)


def compute_pentagon_scores(
    stock_traits: Dict[str, Any],
    core_stats: Dict[str, Any],
    combat_stats: Dict[str, Any],
) -> Dict[str, float]:
    """Compute Inception Pentagon scores (0-10). Robust to missing fields."""
    st = stock_traits if isinstance(stock_traits, dict) else {}
    cs = core_stats if isinstance(core_stats, dict) else {}
    cb = combat_stats if isinstance(combat_stats, dict) else {}

    raw = st.get("Raw") if isinstance(st.get("Raw"), dict) else {}
    dna = st.get("DNA") if isinstance(st.get("DNA"), dict) else {}
    dna_params = dna.get("Params") if isinstance(dna.get("Params"), dict) else {}

    trend_integrity = _safe_float(st.get("TrendIntegrity"), default=math.nan)
    ma_stack = _safe_float(raw.get("MAStackConsistency"), default=math.nan)
    trend_core = _safe_float(cs.get("Trend"), default=math.nan)

    momentum = _safe_float(cs.get("Momentum"), default=math.nan)
    breakout_force = _safe_float(cb.get("BreakoutForce"), default=math.nan)
    upside_power = _safe_float(cb.get("UpsidePower"), default=math.nan)

    stability = _safe_float(cs.get("Stability"), default=math.nan)
    vol_risk = _safe_float(st.get("VolRisk"), default=math.nan)
    mdd_risk = _safe_float(dna_params.get("MaxDrawdownRisk"), default=math.nan)

    liq_trad = _safe_float(st.get("LiquidityTradability"), default=math.nan)
    rr_eff = _safe_float(cb.get("RREfficiency"), default=math.nan)
    support_res = _safe_float(cb.get("SupportResilience"), default=math.nan)

    tail_gap = _safe_float(st.get("TailGapRisk"), default=math.nan)
    mean_rev = _safe_float(st.get("MeanReversionWhipsaw"), default=math.nan)

    trend_power = _weighted_avg(
        {
            "TrendIntegrity": trend_integrity,
            "CoreTrend": trend_core,
            "MAStackConsistency": ma_stack,
        },
        {"TrendIntegrity": 0.55, "CoreTrend": 0.30, "MAStackConsistency": 0.15},
    )

    explosive = _weighted_avg(
        {"Momentum": momentum, "BreakoutForce": breakout_force, "UpsidePower": upside_power},
        {"Momentum": 0.45, "BreakoutForce": 0.35, "UpsidePower": 0.20},
    )

    safety = _weighted_avg(
        {
            "Stability": stability,
            "VolRiskInv": (10.0 - vol_risk) if _is_finite(vol_risk) else math.nan,
            "MaxDrawdownInv": (10.0 - mdd_risk) if _is_finite(mdd_risk) else math.nan,
        },
        {"Stability": 0.45, "VolRiskInv": 0.30, "MaxDrawdownInv": 0.25},
    )

    flow = _weighted_avg(
        {"LiquidityTradability": liq_trad, "RREfficiency": rr_eff, "SupportResilience": support_res},
        {"LiquidityTradability": 0.50, "RREfficiency": 0.30, "SupportResilience": 0.20},
    )

    adrenaline = _weighted_avg(
        {"VolRisk": vol_risk, "TailGapRisk": tail_gap, "MeanReversionWhipsaw": mean_rev},
        {"VolRisk": 0.45, "TailGapRisk": 0.35, "MeanReversionWhipsaw": 0.20},
    )

    scores = {
        "TrendPower": _clip_0_10(trend_power),
        "Explosive": _clip_0_10(explosive),
        "SafetyShield": _clip_0_10(safety),
        "TradingFlow": _clip_0_10(flow),
        "Adrenaline": _clip_0_10(adrenaline),
    }
    return scores


PERSONAS: Dict[str, Dict[str, Any]] = {
    "Compounder": {
        "name": "Compounder",
        "ideal_profile": {
            "TrendPower": 8.5,
            "Explosive": 5.5,
            "SafetyShield": 8.5,
            "TradingFlow": 7.5,
            "Adrenaline": 3.5,
        },
        "weights": {
            "TrendPower": 0.25,
            "Explosive": 0.10,
            "SafetyShield": 0.30,
            "TradingFlow": 0.20,
            "Adrenaline": 0.15,
        },
        "brief": "Prioritizes durability, stability, and clear trends.",
    },
    "AlphaHunter": {
        "name": "AlphaHunter",
        "ideal_profile": {
            "TrendPower": 8.0,
            "Explosive": 8.5,
            "SafetyShield": 5.5,
            "TradingFlow": 6.5,
            "Adrenaline": 6.5,
        },
        "weights": {
            "TrendPower": 0.25,
            "Explosive": 0.30,
            "SafetyShield": 0.15,
            "TradingFlow": 0.15,
            "Adrenaline": 0.15,
        },
        "brief": "Seeks breakout energy and alpha within strong trends.",
    },
    "CashFlowTrader": {
        "name": "CashFlowTrader",
        "ideal_profile": {
            "TrendPower": 6.5,
            "Explosive": 4.5,
            "SafetyShield": 7.5,
            "TradingFlow": 8.5,
            "Adrenaline": 3.5,
        },
        "weights": {
            "TrendPower": 0.20,
            "Explosive": 0.10,
            "SafetyShield": 0.25,
            "TradingFlow": 0.30,
            "Adrenaline": 0.15,
        },
        "brief": "Prefers liquidity, execution consistency, and cash flow.",
    },
    "Speculator": {
        "name": "Speculator",
        "ideal_profile": {
            "TrendPower": 6.0,
            "Explosive": 9.0,
            "SafetyShield": 3.5,
            "TradingFlow": 5.5,
            "Adrenaline": 8.5,
        },
        "weights": {
            "TrendPower": 0.15,
            "Explosive": 0.35,
            "SafetyShield": 0.10,
            "TradingFlow": 0.10,
            "Adrenaline": 0.30,
        },
        "brief": "Accepts higher risk for fast, explosive moves.",
    },
}


def _weighted_distance(values: Dict[str, float], ideal: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = 0.0
    acc = 0.0
    for k, w in weights.items():
        v = values.get(k)
        i = ideal.get(k)
        if not _is_finite(v) or not _is_finite(i):
            continue
        w_f = float(w)
        acc += w_f * abs(float(v) - float(i))
        total_w += w_f
    if total_w <= 0:
        return 5.0
    return float(acc / total_w)


def _score_100_from_distance(dist: float) -> float:
    if not _is_finite(dist):
        return 50.0
    return float(max(0.0, min(100.0, 100.0 - (float(dist) / 10.0 * 100.0))))


def _label_from_score(score_10: float) -> str:
    if score_10 >= 7.5:
        return "Match"
    if score_10 >= 4.5:
        return "Partial Match"
    return "Mismatch"


def _extract_primary_plan_text(dna_pack: Dict[str, Any]) -> str:
    dna = dna_pack if isinstance(dna_pack, dict) else {}
    ppm = dna.get("plan_prior_map") if isinstance(dna.get("plan_prior_map"), dict) else {}
    priors = ppm.get("priors") if isinstance(ppm.get("priors"), dict) else {}
    best_key = ""
    best_score = -1e9
    for k, v in priors.items():
        if not isinstance(v, dict):
            continue
        sc = _safe_float(v.get("fit_score"), default=math.nan)
        if _is_finite(sc) and float(sc) > best_score:
            best_score = float(sc)
            best_key = str(k)
    if best_key:
        return best_key
    # Fallbacks: style/class text
    for key in ("style_primary", "class_primary"):
        s = _safe_text(dna.get(key)).strip()
        if s:
            return s
    return ""


def get_investor_mapping(pentagon_scores: Dict[str, float], dna_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Return persona mapping with DNA veto/downgrades applied."""
    scores = pentagon_scores if isinstance(pentagon_scores, dict) else {}
    dna = dna_pack if isinstance(dna_pack, dict) else {}

    hard_gates = dna.get("hard_gates") if isinstance(dna.get("hard_gates"), dict) else {}
    illiquid = _safe_bool(hard_gates.get("illiquid"))
    gap_prone = _safe_bool(hard_gates.get("gap_prone"))

    plan_text = _extract_primary_plan_text(dna).upper()

    personas_out: Dict[str, Any] = {}
    for key, persona in PERSONAS.items():
        ideal = persona.get("ideal_profile") if isinstance(persona.get("ideal_profile"), dict) else {}
        weights = persona.get("weights") if isinstance(persona.get("weights"), dict) else {}

        dist = _weighted_distance(scores, ideal, weights)
        score_100 = _score_100_from_distance(dist)
        score_10 = round(score_100 / 10.0, 1)

        reasons = []
        vetoed = False

        if illiquid:
            if key in ("AlphaHunter", "CashFlowTrader"):
                vetoed = True
                score_10 = min(score_10, 2.0)
                reasons.append("DNA_HARD_ILLIQUID_VETO")
            else:
                reasons.append("DNA_ILLIQUID_CONTEXT")

        if gap_prone and key == "Compounder":
            score_10 = score_10 - 1.2
            reasons.append("DNA_SOFT_GAP_PRONE")

        if plan_text:
            if ("RANGE" in plan_text) or ("MEAN" in plan_text) or ("REVERSION" in plan_text):
                if key == "AlphaHunter":
                    score_10 = score_10 - 1.0
                    reasons.append("PLAN_STYLE_RANGE_DOWNGRADE")
            if "MOMENTUM" in plan_text:
                if key == "Compounder":
                    score_10 = score_10 - 0.8
                    reasons.append("PLAN_STYLE_MOMENTUM_DOWNGRADE")

        score_10 = _clip_0_10(score_10, default=score_10)
        label = "Mismatch" if vetoed else _label_from_score(score_10)

        # De-dup reasons while preserving order
        seen = set()
        reasons_dedup = []
        for r in reasons:
            if r in seen:
                continue
            reasons_dedup.append(r)
            seen.add(r)

        personas_out[key] = {
            "name": str(persona.get("name") or key),
            "score_10": float(score_10),
            "label": label,
            "reasons": reasons_dedup,
            "vetoed": bool(vetoed),
            "ideal": ideal,
            "brief": str(persona.get("brief") or ""),
        }

    return personas_out


def _primary_persona_name(personas: Dict[str, Any]) -> str:
    best_score = -1e9
    best_name = ""
    for key, item in personas.items():
        name = _safe_text(item.get("name") or key).strip()
        score = _safe_float(item.get("score_10"), default=math.nan)
        if not _is_finite(score):
            continue
        if float(score) > best_score or (float(score) == best_score and name < best_name):
            best_score = float(score)
            best_name = name
    return best_name


def _apply_payoff_persona_cap(personas: Dict[str, Any], payoff_tier: Optional[str]) -> Tuple[bool, str]:
    tier = _safe_text(payoff_tier).strip().upper()
    if not tier or tier not in {"LOW", "MEDIUM"}:
        return False, ""

    primary = _primary_persona_name(personas)
    primary_u = primary.upper()
    reason = ""
    if tier == "MEDIUM" and primary_u == "COMPOUNDER":
        reason = "MEDIUM_blocks_Compounder"
    elif tier == "LOW" and primary_u in {"SPECULATOR", "COMPOUNDER"}:
        reason = "LOW_blocks_Speculator_Compounder"
    else:
        return False, ""

    target = "CashFlowTrader"
    if target not in personas:
        return False, ""

    personas[target]["score_10"] = 10.0
    personas[target]["label"] = _label_from_score(10.0)
    for key, item in personas.items():
        if key == target:
            continue
        score = _safe_float(item.get("score_10"), default=math.nan)
        if _is_finite(score) and float(score) >= 10.0:
            item["score_10"] = 9.9
            item["label"] = _label_from_score(9.9)

    return True, reason


def build_investor_mapping_pack(
    stock_traits: Dict[str, Any],
    core_stats: Dict[str, Any],
    combat_stats: Dict[str, Any],
    dna_pack: Dict[str, Any],
    *,
    payoff_tier: Optional[str] = None,
) -> Dict[str, Any]:
    """Build InvestorMappingPack (UI-ready)."""
    pentagon = compute_pentagon_scores(stock_traits, core_stats, combat_stats)
    personas = get_investor_mapping(pentagon, dna_pack)
    cap_applied, cap_reason = _apply_payoff_persona_cap(personas, payoff_tier)
    return {
        "Pentagon": {k: float(pentagon.get(k, 5.0)) for k in PENTAGON_KEYS},
        "Personas": personas,
        "Meta": {
            "scale": "0-10",
            "labels": {"match_min": 7.5, "partial_min": 4.5},
            "version": "v1",
            "persona_payoff_cap_applied": bool(cap_applied),
            "persona_payoff_cap_reason": cap_reason,
        },
    }
