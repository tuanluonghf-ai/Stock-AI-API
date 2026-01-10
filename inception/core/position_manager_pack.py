"""Position manager pack builder (v1).

Provides execution guidance for a single asset/position.

Design goals:
- Long-only, portfolio-ready.
- "Sizing-lite": suggests trim %, policy size cap, and a protective stop suggestion.
- Does NOT place orders; does not override portfolio sizing logic.

This is intentionally conservative and meant to reduce user mistakes when the
market is close to structural ceilings/floors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .helpers import _safe_float, _safe_text
from .policy import CLASS_POLICY_HINTS


def _pct_to_float(p: str) -> Optional[float]:
    s = _safe_text(p).strip().replace(" ", "")
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def compute_position_manager_pack_v1(
    analysis_pack: Dict[str, Any],
    trade_plan_pack: Optional[Dict[str, Any]] = None,
    decision_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build PositionManagerPack.v1.

    Expected keys used by UI:
      - schema
      - action
      - guidance
      - trim_pct_of_position
      - stop_suggest
      - size_cap_pct_nav
      - position_size_pct_nav
    """
    ap = analysis_pack or {}
    tpp = trade_plan_pack or (ap.get("TradePlanPack") or {})
    tpp = tpp if isinstance(tpp, dict) else {}
    dp = decision_pack or (ap.get("DecisionPack") or {})
    dp = dp if isinstance(dp, dict) else {}

    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}

    mode = _safe_text(pos.get("mode") or tpp.get("mode") or dp.get("mode") or "FLAT").strip().upper()
    if mode not in ("FLAT", "HOLDING"):
        mode = "FLAT"
    is_holding = bool(pos.get("is_holding")) if mode == "HOLDING" else False

    action = _safe_text(dp.get("action") or "WAIT").strip().upper()
    urgency = _safe_text(dp.get("urgency") or "MED").strip().upper()

    # --- Policy cap (display-only) ---
    final_class = _safe_text(ap.get("CharacterClass") or "").strip()
    pol = CLASS_POLICY_HINTS.get(final_class) if isinstance(CLASS_POLICY_HINTS, dict) else None
    pol = pol if isinstance(pol, dict) else {}
    size_cap_txt = _safe_text(pol.get("size_cap") or "").strip()
    size_cap_frac = _pct_to_float(size_cap_txt)
    size_cap_pct_nav = (size_cap_frac * 100.0) if isinstance(size_cap_frac, float) else None

    # Position state (optional)
    position_size_pct_nav = _safe_float(pos.get("position_size_pct_nav"), default=np.nan)
    if not pd.notna(position_size_pct_nav):
        position_size_pct_nav = _safe_float(pos.get("position_size_pct_nav_nav"), default=np.nan)

    in_profit = pos.get("in_profit", None)

    # Distress overlay (robust): used to prevent nonsensical stop suggestions when underwater.
    pnl_val = _safe_float(pos.get("pnl_pct"), default=np.nan)
    distress_level = _safe_text(pos.get("distress_level") or "").strip().upper()
    if not distress_level or distress_level in ("-", "N/A"):
        if pd.notna(pnl_val):
            if pnl_val <= -25.0:
                distress_level = "SEVERE"
            elif pnl_val <= -15.0:
                distress_level = "MEDIUM"
            elif pnl_val <= -7.0:
                distress_level = "MILD"
            else:
                distress_level = "OK"
        else:
            distress_level = "-"

    cur_px = _safe_float(pos.get("current_price"), default=np.nan)

    # --- Trade plan stop suggestion (default = plan stop) ---
    pp = tpp.get("plan_primary") or {}
    pp = pp if isinstance(pp, dict) else {}
    stop_plan = _safe_float(pp.get("stop"), default=np.nan)

    # Try to tighten stop with UnderlyingSupport low + buffer when holding / in profit
    sq = ap.get("StructureQuality") or ap.get("StructureQualityPack") or {}
    sq = sq if isinstance(sq, dict) else {}
    us = sq.get("UnderlyingSupport") or {}
    us = us if isinstance(us, dict) else {}
    us_n = us.get("Nearest") or {}
    us_n = us_n if isinstance(us_n, dict) else {}
    us_zone = us_n.get("Zone") or {}
    us_zone = us_zone if isinstance(us_zone, dict) else {}

    vol_pct = _safe_float((sq.get("Meta") or {}).get("VolPct_ATRProxy"), default=np.nan)
    if not pd.notna(vol_pct):
        vol_pct = _safe_float(ap.get("VolPct") or ap.get("VolPct_ATRProxy"), default=np.nan)
    if not pd.notna(vol_pct):
        vol_pct = 1.2

    # 0.3%–1.2% buffer; slightly larger in high vol
    base_buf = max(0.003, min(0.012, 0.006 * (float(vol_pct) / 1.5)))

    stop_suggest = stop_plan if pd.notna(stop_plan) else np.nan

    # If HOLDING and underwater, prioritize a defensive hard stop below current price.
    defensive_hard_stop = np.nan
    try:
        alt = tpp.get("plan_alt") if isinstance(tpp, dict) else None
        if isinstance(alt, dict) and _safe_text(alt.get("type") or "").strip().upper() == "DEFENSIVE":
            defensive_hard_stop = _safe_float(alt.get("defensive_hard_stop"), default=np.nan)
        if pd.isna(defensive_hard_stop):
            for c in (tpp.get("plans_all") or []):
                if not isinstance(c, dict):
                    continue
                if _safe_text(c.get("type") or "").strip().upper() != "DEFENSIVE":
                    continue
                defensive_hard_stop = _safe_float(c.get("defensive_hard_stop"), default=np.nan)
                break
    except Exception:
        defensive_hard_stop = np.nan

    if is_holding and distress_level in ("MEDIUM", "SEVERE") and pd.notna(cur_px):
        # Choose the closest meaningful stop below current price.
        candidates = []
        if pd.notna(defensive_hard_stop) and defensive_hard_stop < cur_px:
            candidates.append(float(defensive_hard_stop))
        try:
            for c in (tpp.get("plans_all") or []):
                if not isinstance(c, dict):
                    continue
                s = _safe_float(c.get("stop"), default=np.nan)
                if pd.notna(s) and s < cur_px:
                    candidates.append(float(s))
        except Exception:
            pass
        if candidates:
            stop_suggest = max(candidates)  # nearest below current price
        # Safety: never suggest a stop ABOVE current price when underwater.
        if pd.notna(stop_suggest) and stop_suggest >= cur_px:
            stop_suggest = np.nan
    try:
        zl = _safe_float(us_zone.get("Low"), default=np.nan)
        if pd.notna(zl):
            stop_candidate = float(zl) * (1.0 - float(base_buf))
            # Only tighten stop when holding and either profitable or action is HOLD/TRIM
            if is_holding and ((in_profit is True) or (action in ("HOLD", "TRIM"))):
                if pd.notna(stop_suggest):
                    stop_suggest = max(float(stop_suggest), float(stop_candidate))
                else:
                    stop_suggest = float(stop_candidate)
    except Exception:
        pass

    # Final safety: never suggest a stop ABOVE current price when underwater.
    if is_holding and distress_level in ("MEDIUM", "SEVERE") and pd.notna(cur_px) and pd.notna(stop_suggest):
        if float(stop_suggest) >= float(cur_px):
            stop_suggest = np.nan

    # --- Trim suggestion (only meaningful for TRIM) ---
    trim_pct_of_position = None
    if action == "TRIM":
        oh = sq.get("OverheadResistance") or {}
        oh = oh if isinstance(oh, dict) else {}
        oh_n = oh.get("Nearest") or {}
        oh_n = oh_n if isinstance(oh_n, dict) else {}
        tier = _safe_text(oh_n.get("Tier") or "").strip().upper()
        if tier in ("CONFLUENCE",):
            trim_pct_of_position = 0.50
        elif tier in ("HEAVY",):
            trim_pct_of_position = 0.35
        else:
            trim_pct_of_position = 0.25

    # --- Guidance text (keep short; UI will display) ---
    guidance = ""
    if action == "BUY" and not is_holding:
        guidance = "Triển khai theo Trade Plan; ưu tiên đúng vùng Entry và tuân thủ Stop." 
    elif action == "HOLD":
        if is_holding and distress_level in ("MEDIUM", "SEVERE"):
            guidance = "Vị thế đang underwater: ưu tiên giảm rủi ro; chỉ giữ nếu cấu trúc không breakdown và đã đặt hard stop."
        else:
            guidance = "Giữ vị thế; dời stop lên theo cấu trúc khi có lợi thế."
    elif action == "TRIM":
        if is_holding and distress_level in ("MEDIUM", "SEVERE"):
            guidance = "Underwater: ưu tiên giảm tỷ trọng về mức an toàn; không add cho tới khi reclaim cấu trúc."
        else:
            guidance = "Chốt một phần gần trần cấu trúc để bảo toàn lợi nhuận; giữ phần còn lại nếu reclaim thành công."
    elif action == "EXIT":
        guidance = "Cấu trúc xấu; ưu tiên giảm rủi ro/thoát theo stop hoặc breakdown."
    else:
        guidance = "Quan sát thêm; tránh hành động khi chưa có xác nhận cấu trúc/volume."

    # Normalize NaN -> None for UI friendliness
    pos_sz = float(position_size_pct_nav) if pd.notna(position_size_pct_nav) else None
    stop_out = float(stop_suggest) if pd.notna(stop_suggest) else None

    pack = {
        "schema": "PositionManagerPack.v1",
        "mode": "HOLDING" if is_holding else "FLAT",
        "action": action,
        "urgency": urgency,
        "guidance": guidance,
        "trim_pct_of_position": trim_pct_of_position,
        "stop_suggest": stop_out,
        "size_cap_pct_nav": size_cap_pct_nav,
        "position_size_pct_nav": pos_sz,
    }

    # Step 8: normalize pack contract (fail-safe)
    try:
        from inception.core.contracts import normalize_position_manager_pack
        pack = normalize_position_manager_pack(pack)
    except Exception:
        pass

    return pack

