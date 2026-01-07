"""Execution policy hints by CharacterClass.

This module centralizes *display-only* execution constraints that are shared by
core packs and UI renderers.

Principles:
- No Streamlit dependency.
- Policy hints MUST NOT modify signals/scores; they only define conservative
  execution thresholds/caps.
"""

from __future__ import annotations

from typing import Any, Dict

from .helpers import _safe_text

# Display-only policy: does NOT modify scores, triggers, or signal logic.
CLASS_POLICY_HINTS: Dict[str, Dict[str, Any]] = {
    "Smooth Trend": {"rr_min": 1.8, "size_cap": "100%", "overnight": "Normal"},
    "Momentum Trend": {"rr_min": 2.0, "size_cap": "85%", "overnight": "Caution"},
    "Aggressive Trend": {"rr_min": 2.2, "size_cap": "70%", "overnight": "Limit"},
    "Range / Mean-Reversion (Stable)": {"rr_min": 1.6, "size_cap": "100%", "overnight": "Normal"},
    "Volatile Range": {"rr_min": 2.0, "size_cap": "70%", "overnight": "Caution"},
    "Mixed / Choppy Trader": {"rr_min": 2.2, "size_cap": "60%", "overnight": "Limit"},
    "Event / Gap-Prone": {"rr_min": 2.5, "size_cap": "50%", "overnight": "Avoid"},
    "Illiquid / Noisy": {"rr_min": 2.5, "size_cap": "40%", "overnight": "Caution"},
}


def get_class_policy(final_class: str) -> Dict[str, Any]:
    """Return the policy dict for a class, or empty dict if unknown."""
    cn = _safe_text(final_class).strip()
    p = CLASS_POLICY_HINTS.get(cn)
    return dict(p) if isinstance(p, dict) else {}


def get_rr_min(final_class: str, default: float = 1.5) -> float:
    p = get_class_policy(final_class)
    rr = p.get("rr_min", default)
    try:
        return float(rr)
    except Exception:
        return float(default)


def get_class_policy_hint_line(final_class: str) -> str:
    """Render a concise one-line policy hint for UI/notes."""
    p = get_class_policy(final_class)
    if not p:
        return ""
    rr = p.get("rr_min")
    size_cap = _safe_text(p.get("size_cap")).strip()
    overnight = _safe_text(p.get("overnight")).strip()
    rr_txt = f"RR>={float(rr):.1f}" if isinstance(rr, (int, float)) else ""
    size_txt = f"Size<={size_cap}" if size_cap else ""
    on_txt = f"Overnight: {overnight}" if overnight else ""
    parts = [x for x in (rr_txt, size_txt, on_txt) if x]
    return " | ".join(parts)
