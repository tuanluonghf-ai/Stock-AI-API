from __future__ import annotations

from typing import Any, Dict


def _ensure_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _safe_text(obj: Any, default: str = "") -> str:
    try:
        if obj is None:
            return default
        s = str(obj)
        return s if s.strip() != "" else default
    except Exception:
        return default


def _safe_float(obj: Any, default: float = 0.0) -> float:
    try:
        if obj is None:
            return default
        if isinstance(obj, (int, float)):
            return float(obj)
        s = str(obj).strip().replace(",", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default
