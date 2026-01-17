"""Current Status pack builder (v1).

Purpose
  - Provide contextualized technical states WITHOUT emitting any BUY/SELL/ACTION.
  - Acts as Module 2 in the linear pipeline: Stock DNA -> Current Status -> TradePlan Builder -> Decision.

Design
  - Deterministic, Python-only.
  - Emits labels (sentiment/condition) that later modules can consume.
  - Prefer "-" placeholders for missing text fields.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .helpers import _safe_bool, _safe_float, _safe_text


def _pct_diff(a: float, b: float) -> float:
    try:
        if b is None:
            return np.nan
        bb = float(b)
        if not np.isfinite(bb) or bb == 0:
            return np.nan
        aa = float(a)
        if not np.isfinite(aa):
            return np.nan
        return (aa / bb - 1.0) * 100.0
    except Exception:
        return np.nan


def compute_status_pack_v1(
    analysis_pack: Dict[str, Any],
    dna_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute StatusPack.v1.

    Inputs
      - analysis_pack: canonical AnalysisPack dict
      - dna_pack: optional DNAPack (class/style/hard gates)
    """
    ap = analysis_pack or {}
    dna = dna_pack if isinstance(dna_pack, dict) else {}

    last = ap.get("Last") or {}
    last = last if isinstance(last, dict) else {}

    protech = ap.get("ProTech") or {}
    protech = protech if isinstance(protech, dict) else {}
    ma = protech.get("MA") or {}
    ma = ma if isinstance(ma, dict) else {}
    vol = protech.get("Volume") or {}
    vol = vol if isinstance(vol, dict) else {}
    pa = protech.get("PriceAction") or {}
    pa = pa if isinstance(pa, dict) else {}

    sq = ap.get("StructureQuality") or {}
    sq = sq if isinstance(sq, dict) else {}
    gates_pack = sq.get("Gates") or {}
    gates_pack = gates_pack if isinstance(gates_pack, dict) else {}
    cg = gates_pack.get("CeilingGate") or {}
    cg = cg if isinstance(cg, dict) else {}
    st_struct = _safe_text(cg.get("Status") or "").strip().upper()
    if st_struct not in ("PASS", "WAIT", "FAIL"):
        st_struct = "WAIT"

    close = _safe_float(last.get("Close"), default=np.nan)
    ma20 = _safe_float(last.get("MA20"), default=np.nan)
    ma50 = _safe_float(last.get("MA50"), default=np.nan)
    ma200 = _safe_float(last.get("MA200"), default=np.nan)

    slope200 = _safe_text((ma.get("SlopeMA200") or {}).get("Label") if isinstance(ma.get("SlopeMA200"), dict) else ma.get("SlopeMA200")).strip().upper()
    slope50 = _safe_text((ma.get("SlopeMA50") or {}).get("Label") if isinstance(ma.get("SlopeMA50"), dict) else ma.get("SlopeMA50")).strip().upper()

    vol_ratio = _safe_float(vol.get("Ratio"), default=np.nan)
    climax = _safe_bool(pa.get("ClimaxFlag"))
    gap_flag = _safe_bool(pa.get("GapFlag"))

    # DNA context
    class_name = _safe_text(dna.get("class_primary") or dna.get("Class") or ap.get("CharacterClass") or "-").strip() or "-"
    style_primary = _safe_text(dna.get("style_primary") or dna.get("Style") or "-").strip() or "-"
    hard_ill = bool((dna.get("hard_gates") or {}).get("illiquid"))
    hard_gap = bool((dna.get("hard_gates") or {}).get("gap_prone"))
    risk_regime = _safe_text(dna.get("risk_regime") or "-").strip() or "-"

    payoff = ap.get("payoff") if isinstance(ap.get("payoff"), dict) else {}
    payoff_tier = _safe_text(payoff.get("payoff_tier")).strip().upper()
    payoff_span_norm = payoff.get("payoff_span_norm")
    payoff_note_map = {
        "LOW": "Biên độ khai thác hạn chế; ưu tiên kỷ luật chọn điểm và cân nhắc chi phí cơ hội.",
        "MEDIUM": "Biên độ khai thác vừa phải; cần chọn kèo có lợi thế rõ để bù nhiễu/chi phí.",
        "HIGH": "Biên độ khai thác rộng; phù hợp cho chiến lược chủ động nếu quản trị rủi ro tốt.",
    }
    payoff_note = payoff_note_map.get(payoff_tier)

    # MA200 contextual label
    d200 = _pct_diff(close, ma200)
    if not np.isfinite(d200):
        ma200_ctx = "-"
    else:
        if d200 >= 0:
            # Above MA200
            if np.isfinite(_pct_diff(close, ma50)) and close < ma50 and style_primary in ("Trend", "Hybrid"):
                ma200_ctx = "HEALTHY_PULLBACK"
            else:
                ma200_ctx = "ABOVE_MA200"
        else:
            # Below MA200
            if (-1.2 <= d200 <= 0) and ("Trend" in class_name or style_primary == "Trend"):
                ma200_ctx = "POSITIVE_PULLBACK_ZONE"
            else:
                ma200_ctx = "NEGATIVE_TREND_BROKEN"

    # Structure context
    if st_struct == "PASS":
        struct_ctx = "INTACT"
    elif st_struct == "WAIT":
        struct_ctx = "CEILING"
    else:
        struct_ctx = "BROKEN"

    # Momentum context (no action)
    combat = ap.get("CombatStats") or {}
    combat = combat if isinstance(combat, dict) else {}
    breakout_force = _safe_float(combat.get("BreakoutForce"), default=np.nan)
    if not np.isfinite(breakout_force):
        momo_ctx = "-"
    else:
        if breakout_force >= 7 and (not np.isfinite(vol_ratio) or vol_ratio >= 1.15):
            momo_ctx = "BREAKOUT_READY"
        elif breakout_force >= 5:
            momo_ctx = "MOMO_BUILDING"
        else:
            momo_ctx = "MOMO_WEAK"

    # Range/mean-reversion context (lightweight)
    lvl = protech.get("LevelContext") or ap.get("LevelContext") or {}
    lvl = lvl if isinstance(lvl, dict) else {}
    ns = lvl.get("NearestSupport")
    nr = lvl.get("NearestResistance")
    ns_v = _safe_float(ns.get("Value") if isinstance(ns, dict) else ns, default=np.nan)
    nr_v = _safe_float(nr.get("Value") if isinstance(nr, dict) else nr, default=np.nan)
    denom = _safe_float(ap.get("VolProxy"), default=np.nan)
    if not np.isfinite(denom) or denom <= 0:
        denom = np.nan

    range_ctx = "-"
    if style_primary == "Range" or ("Range" in class_name):
        if np.isfinite(close) and np.isfinite(ns_v) and close >= ns_v:
            dist = (close - ns_v) / denom if np.isfinite(denom) else np.nan
            if np.isfinite(dist) and dist <= 0.6:
                range_ctx = "RANGE_SUPPORT_ZONE"
        if range_ctx == "-" and np.isfinite(close) and np.isfinite(nr_v) and close <= nr_v:
            dist = (nr_v - close) / denom if np.isfinite(denom) else np.nan
            if np.isfinite(dist) and dist <= 0.6:
                range_ctx = "RANGE_RESIST_ZONE"
        if range_ctx == "-":
            range_ctx = "RANGE_MID"

    # Volume context
    if not np.isfinite(vol_ratio):
        vol_ctx = "-"
    else:
        if vol_ratio >= 1.25:
            vol_ctx = "CONFIRM"
        elif vol_ratio <= 0.85:
            vol_ctx = "FADE"
        else:
            vol_ctx = "NEUTRAL"
    if climax is True:
        vol_ctx = "CLIMAX_RISK" if vol_ctx in ("CONFIRM", "NEUTRAL") else vol_ctx

    # Risk context (high-level)
    vol_pct = _safe_float(ap.get("VolPct") or ap.get("VolPct_ATRProxy"), default=np.nan)
    if hard_ill or hard_gap or gap_flag:
        risk_ctx = "HIGH"
    elif np.isfinite(vol_pct) and vol_pct >= 2.6:
        risk_ctx = "HIGH"
    elif np.isfinite(vol_pct) and vol_pct >= 1.7:
        risk_ctx = "MID"
    else:
        risk_ctx = "LOW"

    return {
        "schema": "StatusPack.v1",
        "class_context": {
            "class_primary": class_name,
            "style_primary": style_primary,
            "risk_regime": risk_regime,
            "hard_gates": {"illiquid": bool(hard_ill), "gap_prone": bool(hard_gap)},
        },
        "payoff": (
            {
                "tier": payoff_tier,
                "span_norm": payoff_span_norm,
                "label": f"Payoff (Tier): {payoff_tier}",
                "note": payoff_note,
            }
            if payoff_tier in payoff_note_map
            else {}
        ),
        "technicals": {
            "ma200_context": ma200_ctx,
            "structure_context": struct_ctx,
            "momentum_context": momo_ctx,
            "range_context": range_ctx,
            "volume_context": vol_ctx,
            "risk_context": risk_ctx,
            "slope200": slope200 or "-",
            "slope50": slope50 or "-",
            "gap_flag": bool(gap_flag),
            "climax_flag": bool(climax),
        },
    }
