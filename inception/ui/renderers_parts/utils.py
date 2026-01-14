from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from inception.core.policy import CLASS_POLICY_HINTS
from inception.ui.formatters import _ensure_dict
from inception.ui.ui_constants import UI_PLACEHOLDER

try:
    from inception.core.helpers import _safe_text, _safe_float, _clip, _safe_bool, _as_scalar  # type: ignore
except Exception:  # pragma: no cover
    from inception.core.helpers import _safe_text, _safe_float, _safe_bool, _as_scalar  # type: ignore

    def _clip(x: float, lo: float, hi: float) -> float:
        try:
            if pd.isna(x):
                return x
            return float(max(lo, min(hi, float(x))))
        except Exception:
            return x



PLAYSTYLE_TAG_TRANSLATIONS = {
    "Pullback-buy zone (confluence)": "Vùng pullback mua (hội tụ)",
    "Breakout attempt (needs follow-through)": "Nỗ lực breakout (cần follow-through)",
    "Wait for volume confirmation": "Chờ xác nhận dòng tiền",
    "Tight risk control near resistance": "Siết rủi ro gần kháng cự",
    "Use LongStructure_ShortTactical rule": "Ưu tiên cấu trúc dài hạn; tactical dùng để vào lệnh",
}

def _val_or_na(v: Any) -> str:
    """UI-friendly stringify with dash placeholder (system-wide)."""
    try:
        if v is None:
            return UI_PLACEHOLDER
        if isinstance(v, float) and pd.isna(v):
            return UI_PLACEHOLDER
        s = str(v).strip()
        return s if s else UI_PLACEHOLDER
    except Exception:
        return UI_PLACEHOLDER

def _pick_character_narrative(cp: Dict[str, Any]) -> str:
    """Return narrative text from CharacterPack, or placeholder."""
    for key in ("Narrative", "DNA_LINE"):
        val = cp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            for k in ("line_final", "line_draft", "line", "text"):
                v = val.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return UI_PLACEHOLDER

def build_dna_line_v1(
    ticker: str,
    final_class: str,
    style_axis: str | None = None,
    risk_regime: str | None = None,
) -> str:
    """UI narrative generation is disabled. Return placeholder only."""
    return UI_PLACEHOLDER

def get_class_policy_hint_line(final_class: str) -> str:
    cn = _safe_text(final_class).strip()
    p = CLASS_POLICY_HINTS.get(cn)
    if not p:
        return ""
    rr = p.get("rr_min")
    size_cap = _safe_text(p.get("size_cap")).strip()
    overnight = _safe_text(p.get("overnight")).strip()
    rr_txt = f"RR≥{float(rr):.1f}" if isinstance(rr, (int, float)) else ""
    size_txt = f"Size≤{size_cap}" if size_cap else ""
    on_txt = f"Overnight: {overnight}" if overnight else ""
    parts = [x for x in (rr_txt, size_txt, on_txt) if x]
    return " | ".join(parts)
