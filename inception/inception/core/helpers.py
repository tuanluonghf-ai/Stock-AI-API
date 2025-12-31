"""Core helper utilities for INCEPTION.

These functions are shared across core calculations and modules.

Guarantees:
- No Streamlit dependency.
- Coerce potentially ambiguous pandas objects to scalars.
- Provide JSON-safe serialization utilities.
"""

from __future__ import annotations

from typing import Any
import json

import numpy as np
import pandas as pd


class DataError(Exception):
    """Raised when required local data files cannot be loaded."""
    pass


def assert_not_pandas_bool(x: Any, where: str = "") -> Any:
    """Fail fast if a pandas object leaks into boolean logic."""
    if isinstance(x, (pd.Series, pd.DataFrame, pd.Index)):
        raise ValueError(f"Ambiguous pandas object in boolean context at {where}")
    return x


def _safe_str(obj: Any) -> str:
    """Coerce any object (str/dict/None/number) into safe text."""
    try:
        if obj is None:
            return ""
        if isinstance(obj, dict):
            for k in ("Label", "Name", "Value", "Text", "State", "Zone", "Event"):
                v = obj.get(k)
                if v is not None:
                    return str(v)
            return ""
        return str(obj)
    except Exception:
        return ""


def _safe_text(obj: Any) -> str:
    return _safe_str(obj)


def _as_scalar(x: Any) -> Any:
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return None
            return x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        if isinstance(x, (list, tuple, np.ndarray)):
            return x[-1] if len(x) else None
    except Exception:
        pass
    return x


def _coalesce(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        v2 = _as_scalar(v)
        if v2 is None:
            continue
        try:
            if isinstance(v2, float) and pd.isna(v2):
                continue
        except Exception:
            pass
        if isinstance(v2, str) and not v2.strip():
            continue
        return v2
    return None


def _safe_bool(x: Any) -> bool:
    v = _as_scalar(x)
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    if isinstance(v, (float, np.floating)):
        return bool(v) if pd.notna(v) else False
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "yes", "y", "1", "ok"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return False


def _safe_float(x: Any, default: float = np.nan) -> float:
    v = _as_scalar(x)
    if v is None:
        return float(default)
    try:
        if isinstance(v, (int, np.integer, float, np.floating)):
            return float(v) if pd.notna(v) else float(default)
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s == "":
                return float(default)
            return float(s)
        return float(v)
    except Exception:
        return float(default)


def _clip(x: float, lo: float, hi: float) -> float:
    try:
        if pd.isna(x):
            return x
        return float(max(lo, min(hi, x)))
    except Exception:
        return x


def _json_default(o: Any) -> Any:
    try:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, (pd.Series, pd.DataFrame, pd.Index)):
            return str(o)
    except Exception:
        pass
    return str(o)


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """JSON serialization helper used for snapshots / prompt payloads."""
    return json.dumps(obj, default=_json_default, ensure_ascii=False, **kwargs)
