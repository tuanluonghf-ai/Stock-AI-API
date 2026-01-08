from typing import Any, Dict, List, Optional, Tuple, Union
import math
from inception.ui.renderers import render_appendix_e, render_report_pretty
from inception.core.tradeplan_pack import compute_trade_plan_pack_v1
from inception.core.decision_pack import compute_decision_pack_v1
from inception.core.position_manager_pack import compute_position_manager_pack_v1
from inception.core.report_ad_builder import generate_insight_report
from inception.core.policy import get_class_policy_hint_line




# ============================================================
# SAFE TEXT HELPERS (GLOBAL)
# ============================================================

def _val_or_na(v: Any) -> str:
    """Return 'N/A' for None/NaN/empty, else str(v). Safe for renderers."""
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and pd.isna(v):
            return "N/A"
        s = str(v).strip()
        return s if s else "N/A"
    except Exception:
        return "N/A"

def _safe_str(obj) -> str:
    """Coerce any object (str/dict/None/number) into safe text for .strip()/.lower() usage."""
    try:
        if obj is None:
            return ""
        if isinstance(obj, dict):
            for k in ("Label","Name","Value","Text","State","Zone","Event"):
                v = obj.get(k)
                if v is not None:
                    return str(v)
            return ""
        return str(obj)
    except Exception:
        return ""

def _safe_text(obj) -> str:
    # Backward-compatible alias used in validators / prompt builders
    return _safe_str(obj)


# --------------------------
# Scalar coercions (global)
# Avoid pandas Series truth-value ambiguity across modules
# --------------------------
def _as_scalar(x: Any) -> Any:
    try:
        # pandas Series / Index
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return None
            # Prefer last value (consistent with "latest bar" semantics)
            return x.iloc[-1] if hasattr(x, "iloc") else x[-1]
        # numpy array / list-like
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


def _clip(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]. Defensive helper used by packs in app.

    Note: Defined here to avoid NameError when legacy pack builders still
    run inside app_INCEPTION during refactors.
    """
    try:
        # pd may be imported later; resolve at call-time
        if "pd" in globals():
            import pandas as pd  # type: ignore
            if pd.isna(x):
                return x
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return x
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


# ============================================================
# INCEPTION v6.0 | Strategic Investor Edition
# app.py — Streamlit + GPT-4o
# Author: INCEPTION AI Research Framework
# Purpose: Technical–Fundamental Integrated Research Assistant
# CHANGELOG:
# v5.0-5.1: Core Refactoring + TradePlan v2
# v5.2-5.3: Neutralization Phase 1 & 2 (Tags, Facts only)
# v5.4:     Neutralization Phase 3 (Step 6B + 7B)
#           - Scenario12: "Extended" -> "RSI_70Plus" (Neutral).
#           - Bias: Tags updated to match new neutral logic.
#           - ContextPacks: Added RSIContext, VolumeContext, LevelContext
#             (Streaks, Deltas, Nearest S/R Distance) to help GPT judge context.
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np

APP_VERSION = "14.9"
APP_TITLE = "INCEPTION"

# ------------------------------------------------------------
# Playstyle tag translations (UI-only).
# Keep minimal and safe: unknown tags fall back to raw text.
class DataError(Exception):
    """Raised when required local data files cannot be loaded."""
    pass

def assert_not_pandas_bool(x: Any, where: str = "") -> Any:
    """Fail fast if a pandas object leaks into boolean logic."""
    if isinstance(x, (pd.Series, pd.DataFrame, pd.Index)):
        raise ValueError(f"Ambiguous pandas object in boolean context at {where}")
    return x

from pathlib import Path
import os
import json
import re
import html
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


# ------------------------------------------------------------
# Import hardening (Phase 5 prerequisite)
# Ensure repo root is on sys.path (Render/Streamlit safe)
# + Fail-fast required path checks to avoid silent import drift
# ------------------------------------------------------------
import sys
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _require_path(p: Path, hint: str):
    if not p.exists():
        raise RuntimeError(f"Missing required path: {p}. {hint}")

_require_path(REPO_ROOT / "inception" / "__init__.py",
              "Ensure package inception/ exists at repo root and has __init__.py")
_require_path(REPO_ROOT / "inception" / "infra" / "datahub.py",
              "Expected inception/infra/datahub.py")
_require_path(REPO_ROOT / "inception" / "modules" / "base.py",
              "Expected inception/modules/base.py")

# Phase 3: modular execution (registry) + DataHub (infra)
from inception.infra.datahub import DataHub, DataError as HubDataError
from inception.modules.base import run_modules
from inception.modules import load_default_modules
load_default_modules()

# Step 12 (v14.7): single pipeline entry (app only wires inputs -> pipeline -> UI)
from inception.core.pipeline import build_result as build_result_pipeline

# ------------------------------
# JSON safe-serialization helpers
# ------------------------------
def _json_default(obj):
    """Convert non-JSON-serializable objects (Timestamp, numpy types, etc.) to JSON-safe forms."""
    try:
        # pandas NaT / missing
        if obj is pd.NaT:
            return None
    except Exception:
        pass

    if isinstance(obj, (pd.Timestamp, datetime, date)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)

    return str(obj)

def safe_json_dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=_json_default)


# ============================================================
# PHASE 5 HARDENING — PACK SANITIZE & VALIDATE (v6.1)
# - Prevent NaN/Inf leakage to UI/PDF/GPT prompts
# - Prevent pandas objects from entering boolean/render context
# ============================================================

def _is_finite_number(x: Any) -> bool:
    try:
        if isinstance(x, (bool, np.bool_)):
            return True
        if isinstance(x, (int, np.integer)):
            return True
        if isinstance(x, (float, np.floating)):
            return math.isfinite(float(x))
    except Exception:
        return False
    return False

def sanitize_pack(obj: Any) -> Any:
    """Recursively convert obj to JSON-safe / render-safe python primitives.
    - pandas Series/Index -> scalar (latest)
    - pandas DataFrame -> list[dict] (records) (only if needed)
    - numpy scalar -> python scalar
    - NaN/Inf -> None
    """
    try:
        if obj is pd.NaT:
            return None
    except Exception:
        pass

    # pandas containers
    if isinstance(obj, pd.Series):
        return sanitize_pack(_as_scalar(obj))
    if isinstance(obj, pd.Index):
        return [sanitize_pack(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        # Avoid embedding big dataframes by default; keep minimal representation.
        try:
            return [sanitize_pack(r) for r in obj.to_dict(orient="records")]
        except Exception:
            return None

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): sanitize_pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_pack(v) for v in obj]

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        fx = float(obj)
        return fx if math.isfinite(fx) else None
    if isinstance(obj, (np.ndarray,)):
        return [sanitize_pack(v) for v in obj.tolist()]

    # python scalars
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    return obj

def validate_pack_no_pandas(obj: Any, path: str = "root") -> None:
    """Fail-fast if pandas objects leak into the output payload."""
    if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
        raise RuntimeError(f"Phase5 validation failed: pandas object leaked at {path}: {type(obj).__name__}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            validate_pack_no_pandas(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            validate_pack_no_pandas(v, f"{path}[{i}]")

def safe_json_dumps_strict(x: Any) -> str:
    """Strict JSON dump: sanitize first, then disallow NaN."""
    sx = sanitize_pack(x)
    validate_pack_no_pandas(sx)
    return json.dumps(sx, ensure_ascii=False, default=_json_default, allow_nan=False)

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================
# ============================================================
# 2. PATHS & CONSTANTS
# ============================================================
PRICE_VOL_PATH = "Price_Vol.xlsx"
HSC_TARGET_PATH = "Tickers target price.xlsx"
TICKER_NAME_PATH = "Ticker name.xlsx"

# Data directory resolution (robust for Streamlit/VSCode working-directory differences)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("INCEPTION_DATA_DIR", str(BASE_DIR))).resolve()

def resolve_data_path(path: str) -> str:
    """Resolve data file paths reliably.
    Priority:
      1) absolute path if exists
      2) DATA_DIR / path
      3) cwd / path
      4) case-insensitive / fuzzy match in DATA_DIR for common files
    """
    if not path:
        return path
    try:
        p = Path(path)
    except Exception:
        return path

    # Absolute path
    if p.is_absolute() and p.exists():
        return str(p)

    # Direct candidates
    cand1 = (DATA_DIR / p).resolve()
    if cand1.exists():
        return str(cand1)

    cand2 = (Path.cwd() / p).resolve()
    if cand2.exists():
        return str(cand2)

    # Case-insensitive exact name match inside DATA_DIR
    try:
        target = p.name.lower()
        for f in DATA_DIR.glob("*"):
            if f.is_file() and f.name.lower() == target:
                return str(f.resolve())
    except Exception:
        pass

    # Fuzzy match for common dataset names
    name_l = p.name.lower()
    try:
        if ("target" in name_l and "price" in name_l) or ("target" in name_l and name_l.endswith(".xlsx")):
            for f in DATA_DIR.glob("*.xlsx"):
                nl = f.name.lower()
                if "target" in nl and "price" in nl:
                    return str(f.resolve())
        if ("price" in name_l and "vol" in name_l) or ("price_vol" in name_l):
            for f in DATA_DIR.glob("*.xlsx"):
                nl = f.name.lower()
                if "price" in nl and "vol" in nl:
                    return str(f.resolve())
        if ("ticker" in name_l and "name" in name_l):
            for f in DATA_DIR.glob("*.xlsx"):
                nl = f.name.lower()
                if "ticker" in nl and "name" in nl:
                    return str(f.resolve())
    except Exception:
        pass

    # Default expected location
    return str(cand1)


VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
    "KH03": {"name": "Khách mời 03", "quota": 5},
    "KH04": {"name": "Khách mời 04", "quota": 5},
    "KH05": {"name": "Khách mời 05", "quota": 5},
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================
def _fmt_price(x, ndigits=2):
    if pd.isna(x): return ""
    return f"{float(x):.{ndigits}f}"

def _fmt_int(x):
    if pd.isna(x): return ""
    return f"{int(round(float(x))):,}"

def _fmt_pct(x):
    if pd.isna(x): return ""
    return f"{float(x):.1f}%"

def _fmt_thousand(x, ndigits=1):
    if pd.isna(x): return ""
    return f"{float(x)/1000:.{ndigits}f}"

def _safe_float(x, default=np.nan) -> float:
    try: return float(x)
    except: return default

def _round_price(x: float, ndigits: int = 2) -> float:
    if np.isnan(x): return np.nan
    return round(float(x), ndigits)

def _isnan(x) -> bool:
    try: return x is None or (isinstance(x, float) and np.isnan(x))
    except: return True

def _sgn(x: float) -> int:
    if pd.isna(x): return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _pct_change(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0: return np.nan
    return (a - b) / b * 100

def _trend_label_from_slope(slope: float, eps: float = 1e-9) -> str:
    if pd.isna(slope): return "N/A"
    if slope > eps: return "Up"
    if slope < -eps: return "Down"
    return "Flat"

def _find_last_cross(series_a: pd.Series, series_b: pd.Series, lookback: int = 20) -> Dict[str, Any]:
    a = series_a.dropna()
    b = series_b.dropna()
    if a.empty or b.empty:
        return {"Event": "None", "BarsAgo": None}
    
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if df.empty:
        return {"Event": "None", "BarsAgo": None}
    
    df = df.tail(lookback + 2)
    diff = df["a"] - df["b"]
    sign = diff.apply(_sgn)
    
    last_event = None
    last_bars_ago = None
    
    s = sign.values
    for i in range(len(s) - 1, 0, -1):
        if s[i] == 0 or s[i-1] == 0:
            continue
        if s[i] != s[i-1]:
            if s[i-1] < s[i]:
                last_event = "CrossUp"
            else:
                last_event = "CrossDown"
            last_bars_ago = (len(s) - 1) - i
            break
            
    return {"Event": last_event or "None", "BarsAgo": int(last_bars_ago) if last_bars_ago is not None else None}

def _detect_divergence_simple(close: pd.Series, osc: pd.Series, lookback: int = 60) -> Dict[str, Any]:
    c = close.dropna().tail(lookback).reset_index(drop=True)
    o = osc.dropna().tail(lookback).reset_index(drop=True)
    n = min(len(c), len(o))
    if n < 10:
        return {"Type": "None", "Detail": "N/A"}
    
    c = c.tail(n).reset_index(drop=True)
    o = o.tail(n).reset_index(drop=True)
    
    lows = []
    highs = []
    
    for i in range(2, n-2):
        if c[i] < c[i-1] and c[i] < c[i+1]:
            lows.append(i)
        if c[i] > c[i-1] and c[i] > c[i+1]:
            highs.append(i)
            
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        price_ll = c[i2] < c[i1]
        osc_hl = o[i2] > o[i1]
        if price_ll and osc_hl:
            return {"Type": "Bullish", "Detail": f"Price LL vs Osc HL (swings {i1}->{i2})"}
            
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        price_hh = c[i2] > c[i1]
        osc_lh = o[i2] < o[i1]
        if price_hh and osc_lh:
            return {"Type": "Bearish", "Detail": f"Price HH vs Osc LH (swings {i1}->{i2})"}
            
    return {"Type": "None", "Detail": "N/A"}

# ============================================================
# 4. LOADERS
# ============================================================

def _file_mtime(path: str) -> int:
    try:
        return int(Path(path).stat().st_mtime)
    except Exception:
        return 0

@st.cache_data
def _read_excel_cached(resolved_path: str, mtime: int) -> pd.DataFrame:
    # mtime is intentionally part of the cache key
    return pd.read_excel(resolved_path)

def load_price_vol(path: str = PRICE_VOL_PATH) -> pd.DataFrame:
    """Load price/volume data via DataHub (infra)."""
    try:
        hub = DataHub.from_env(default_dir=DATA_DIR)
        return hub.load_price_vol(path)
    except HubDataError as e:
        raise DataError(str(e)) from e

def load_ticker_names(path: str = TICKER_NAME_PATH) -> pd.DataFrame:
    """Load ticker names via DataHub (infra)."""
    try:
        hub = DataHub.from_env(default_dir=DATA_DIR)
        df = hub.load_ticker_names(path)
        if "Name" not in df.columns and "Stock Name" in df.columns:
            df = df.rename(columns={"Stock Name": "Name"})
        return df[["Ticker","Name"]].drop_duplicates(subset=["Ticker"], keep="last") if "Ticker" in df.columns and "Name" in df.columns else pd.DataFrame(columns=["Ticker","Name"])
    except HubDataError:
        return pd.DataFrame(columns=["Ticker","Name"])

def load_hsc_targets(path: str = HSC_TARGET_PATH) -> pd.DataFrame:
    """
    Load bảng target của CTCK (ví dụ Tickers Target Price.xlsx) và chuẩn hóa
    về 3 cột: Ticker, Target, Recommendation.

    Lưu ý:
    - Một số file dùng header 'TP (VND)' hoặc tương tự thay vì 'Target',
      nên cần rename về 'Target' để pipeline Fundamental đọc được.
    """
    try:
        hub = DataHub.from_env(default_dir=DATA_DIR)
        df = hub.load_hsc_targets(path)
    except HubDataError:
        # Fallback: trả về DataFrame rỗng với đúng schema tối thiểu
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    # Chuẩn hóa tên cột Target: ưu tiên cột 'Target', nếu không có thì map từ các tên quen dùng
    if "Target" not in df.columns:
        rename_map = {
            "TP (VND)": "Target",
            "TP": "Target",
            "Target price": "Target",
            "Target Price": "Target",
            "Target Price (VND)": "Target",
            "Giá mục tiêu": "Target",
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
                break  # chỉ cần map một cột phù hợp

    # Chuẩn hóa giá trị Target (nếu đã có cột Target)
    if "Target" in df.columns:
        df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
        # Nếu Target < 500 thì coi như đơn vị là nghìn, nhân lại cho 1,000
        df.loc[df["Target"].notna() & (df["Target"] < 500), "Target"] = df["Target"] * 1000

    # Bảo đảm có cột Recommendation
    if "Recommendation" not in df.columns:
        df["Recommendation"] = ""

    # Giữ lại các cột cần dùng
    cols = [c for c in ["Ticker", "Target", "Recommendation"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["Ticker", "Target", "Recommendation"])

    # Mỗi ticker chỉ giữ dòng cuối (mới nhất)
    return df[cols].drop_duplicates(subset=["Ticker"], keep="last")

def sma(series, window): return series.rolling(window=window).mean()
def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def rsi_wilder(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ============================================================
# 6. FIBONACCI & CONTEXT
# ============================================================
def _fib_levels(low, high):
    rng = high - low
    if rng <= 0: return {}
    return {
        "38.2": high - 0.382 * rng,
        "50.0": high - 0.5 * rng,
        "61.8": high - 0.618 * rng,
        "127.2": high + 0.272 * rng,
        "161.8": high + 0.618 * rng
    }

def _compute_fib_window(df: pd.DataFrame, w: int) -> Dict[str, Any]:
    L = w if len(df) >= w else len(df)
    win = df.tail(L)
    hi, lo = win["High"].max(), win["Low"].min()
    return {"window": L, "swing_high": hi, "swing_low": lo, "levels": _fib_levels(lo, hi)}

def _score_fib_relevance(close: float, fib: Dict[str, Any]) -> float:
    lv = fib.get("levels", {})
    hi = _safe_float(fib.get("swing_high"))
    lo = _safe_float(fib.get("swing_low"))
    if pd.isna(close) or pd.isna(hi) or pd.isna(lo) or hi <= lo:
        return -1e9
    rng = hi - lo
    range_pct = rng / close if close != 0 else np.nan
    s382 = _safe_float(lv.get("38.2"))
    s618 = _safe_float(lv.get("61.8"))
    score = 0.0
    if pd.notna(range_pct):
        score += max(0.0, 1.2 - abs(range_pct - 0.25) * 3.0)
    if range_pct < 0.08:
        score -= 0.8
    if pd.notna(s382) and pd.notna(s618):
        loz = min(s382, s618)
        hiz = max(s382, s618)
        if loz <= close <= hiz:
            score += 2.0
        elif close > hiz:
            score += 1.0
        else:
            score += 0.5
    return score

def compute_dual_fibonacci_auto(df: pd.DataFrame, long_window: int = 250) -> Dict[str, Any]:
    if df.empty:
        return {"short_window": None, "long_window": None, "auto_short": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "fixed_long": {"swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "alt_short": {"window": None, "swing_high": np.nan, "swing_low": np.nan, "levels": {}}, "selection_reason": "N/A"}
    last_close = _safe_float(df.iloc[-1].get("Close"))
    fib60 = _compute_fib_window(df, 60)
    fib90 = _compute_fib_window(df, 90)
    s60 = _score_fib_relevance(last_close, fib60)
    s90 = _score_fib_relevance(last_close, fib90)
    if s90 > s60:
        chosen = fib90
        alt = fib60
        reason = "AutoSelect=90 (higher relevance score)"
    else:
        chosen = fib60
        alt = fib90
        reason = "AutoSelect=60 (higher relevance score)"
    L_long = long_window if len(df) >= long_window else len(df)
    win_long = df.tail(L_long)
    l_hi, l_lo = win_long["High"].max(), win_long["Low"].min()
    return {
        "short_window": chosen.get("window"),
        "long_window": L_long,
        "auto_short": {"swing_high": chosen.get("swing_high"), "swing_low": chosen.get("swing_low"), "levels": chosen.get("levels", {})},
        "fixed_long": {"swing_high": l_hi, "swing_low": l_lo, "levels": _fib_levels(l_lo, l_hi)},
        "alt_short": alt,
        "selection_reason": reason
    }

def _sorted_levels(levels: Dict[str, Any]) -> List[Tuple[str, float]]:
    out = []
    for k, v in (levels or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            out.append((str(k), fv))
    return sorted(out, key=lambda x: x[1])

def _nearest_levels_around_price(levels: Dict[str, Any], close: float, n_each: int = 2) -> Dict[str, Any]:
    if pd.isna(close):
        return {"Supports": [], "Resistances": []}
    lv = _sorted_levels(levels)
    supports = [(k, v) for (k, v) in lv if v <= close]
    resistances = [(k, v) for (k, v) in lv if v >= close]
    supports = sorted(supports, key=lambda kv: abs(close - kv[1]))[:n_each]
    resistances = sorted(resistances, key=lambda kv: abs(close - kv[1]))[:n_each]
    def pack(items):
        arr = []
        for k, v in items:
            dist_pct = ((v - close) / close * 100) if close != 0 else np.nan
            arr.append({"Level": k, "Value": v, "DistPct": dist_pct})
        return arr
    return {"Supports": pack(supports), "Resistances": pack(resistances)}

def _position_band(close: float, levels: Dict[str, Any]) -> str:
    if pd.isna(close): return "N/A"
    l382 = _safe_float((levels or {}).get("38.2"))
    l618 = _safe_float((levels or {}).get("61.8"))
    if pd.isna(l382) or pd.isna(l618): return "N/A"
    lo = min(l382, l618)
    hi = max(l382, l618)
    if close >= hi: return "Above61.8"
    if close <= lo: return "Below38.2"
    return "Between38.2_61.8"

def _confluence_with_ma(levels: Dict[str, Any], ma_values: Dict[str, float], tol_pct: float = 0.6) -> Dict[str, Any]:
    res = {}
    lv = _sorted_levels(levels)
    for ma_key, ma_val in (ma_values or {}).items():
        if pd.isna(ma_val) or ma_val == 0:
            res[ma_key] = []
            continue
        hits = []
        for k, v in lv:
            dist_pct = abs(v - ma_val) / ma_val * 100
            if dist_pct <= tol_pct:
                hits.append({"FibLevel": k, "FibValue": v, "DistPct": dist_pct})
        res[ma_key] = hits
    return res

def compute_fibonacci_context_pack(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    close = _safe_float(last.get("Close"))
    ma_vals = {
        "MA20": _safe_float(last.get("MA20")),
        "MA50": _safe_float(last.get("MA50")),
        "MA200": _safe_float(last.get("MA200")),
    }
    short_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
    long_lv  = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
    short_near = _nearest_levels_around_price(short_lv, close, n_each=2)
    long_near  = _nearest_levels_around_price(long_lv, close, n_each=2)
    short_band = _position_band(close, short_lv)
    long_band  = _position_band(close, long_lv)
    conf_short = _confluence_with_ma(short_lv, ma_vals, tol_pct=0.6)
    conf_long  = _confluence_with_ma(long_lv, ma_vals, tol_pct=0.6)
    fib_conflict = (short_band != "N/A" and long_band != "N/A" and short_band != long_band)
    priority_rule = "LongStructure_ShortTactical" if fib_conflict else "None"
    return {
        "Close": close,
        "ShortBand": short_band,
        "LongBand": long_band,
        "NearestShort": short_near,
        "NearestLong": long_near,
        "ConfluenceShortWithMA": conf_short,
        "ConfluenceLongWithMA": conf_long,
        "FiboConflictFlag": bool(fib_conflict),
        "FiboPriorityRuleApplied": priority_rule
    }

# ============================================================
# 6B. PRO TECH FEATURES (PYTHON-ONLY)
# ============================================================
def compute_ma_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    # --------------------------
    # Helper coercions (avoid Series truth-value ambiguity)
    # --------------------------
    def _as_scalar(x: Any) -> Any:
        try:
            if isinstance(x, pd.Series):
                s = x.dropna()
                if not s.empty:
                    return s.iloc[-1]
                return x.iloc[-1] if len(x) else None
            if isinstance(x, (list, tuple, np.ndarray)):
                return x[-1] if len(x) else None
        except Exception:
            return None
        return x

    def _coalesce(*vals: Any) -> Any:
        for v in vals:
            if v is None:
                continue
            v2 = _as_scalar(v)
            if v2 is None:
                continue
            if isinstance(v2, float) and pd.isna(v2):
                continue
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
            if s in ("false", "f", "no", "n", "0", ""):
                return False
        return False
    
    def slope_value(series: pd.Series, n: int = 10) -> float:
        s = series.dropna()
        if len(s) < n + 1: return np.nan
        return _safe_float(s.iloc[-1] - s.iloc[-(n+1)])
    
    def slope_label(val: float, eps: float = 1e-9) -> str:
        if pd.isna(val): return "N/A"
        if val > eps: return "Positive"
        if val < -eps: return "Negative"
        return "Flat"
        
    s20_v = slope_value(df["MA20"], 10)
    s50_v = slope_value(df["MA50"], 10)
    s200_v = slope_value(df["MA200"], 10)
    
    structure_snapshot = "N/A"
    if pd.notna(close) and pd.notna(ma50) and pd.notna(ma200):
        if close >= ma50 and ma50 >= ma200:
            structure_snapshot = "Close>=MA50>=MA200"
        elif close < ma50 and ma50 < ma200:
            structure_snapshot = "Close<MA50<MA200"
        else:
            structure_snapshot = "MixedStructure"
            
    dist50 = ((close - ma50) / ma50 * 100) if (pd.notna(close) and pd.notna(ma50) and ma50 != 0) else np.nan
    dist200 = ((close - ma200) / ma200 * 100) if (pd.notna(close) and pd.notna(ma200) and ma200 != 0) else np.nan
    cross_price_ma50 = _find_last_cross(df["Close"], df["MA50"], lookback=20)
    cross_price_ma200 = _find_last_cross(df["Close"], df["MA200"], lookback=60)
    cross_ma20_ma50 = _find_last_cross(df["MA20"], df["MA50"], lookback=60)
    cross_ma50_ma200 = _find_last_cross(df["MA50"], df["MA200"], lookback=120)
    
    return {
        "Regime": structure_snapshot,
        "SlopeMA20": slope_label(s20_v),
        "SlopeMA50": slope_label(s50_v),
        "SlopeMA200": slope_label(s200_v),
        "SlopeMA20Value": s20_v,
        "SlopeMA50Value": s50_v,
        "SlopeMA200Value": s200_v,
        "DistToMA50Pct": dist50,
        "DistToMA200Pct": dist200,
        "Cross": {
            "PriceVsMA50": cross_price_ma50,
            "PriceVsMA200": cross_price_ma200,
            "MA20VsMA50": cross_ma20_ma50,
            "MA50VsMA200": cross_ma50_ma200
        },
        "Structure": {
            "PriceAboveMA50": bool(pd.notna(close) and pd.notna(ma50) and close >= ma50),
            "PriceAboveMA200": bool(pd.notna(close) and pd.notna(ma200) and close >= ma200),
            "MA20AboveMA50": bool(pd.notna(ma20) and pd.notna(ma50) and ma20 >= ma50),
            "MA50AboveMA200": bool(pd.notna(ma50) and pd.notna(ma200) and ma50 >= ma200),
        }
    }

def compute_rsi_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    rsi = df["RSI"].dropna()
    if rsi.empty: return {}
    last_rsi = _safe_float(rsi.iloc[-1])
    prev5 = _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else (_safe_float(rsi.iloc[0]) if len(rsi) >= 1 else np.nan)
    direction = "N/A"
    if pd.notna(last_rsi) and pd.notna(prev5):
        delta5 = last_rsi - prev5
        if delta5 > 1.0: direction = "Rising"
        elif delta5 < -1.0: direction = "Falling"
        else: direction = "Flat"
    else:
        delta5 = np.nan
    zone = "N/A"
    if pd.notna(last_rsi):
        if last_rsi >= 70: zone = "Zone70Plus"
        elif last_rsi >= 60: zone = "Zone60_70"
        elif last_rsi >= 50: zone = "Zone50_60"
        elif last_rsi >= 40: zone = "Zone40_50"
        elif last_rsi >= 30: zone = "Zone30_40"
        else: zone = "ZoneBelow30"
    tail6 = rsi.tail(6).tolist()
    tail6 = [(_safe_float(x) if pd.notna(x) else np.nan) for x in tail6]
    tail20 = rsi.tail(20)
    rsi_max20 = _safe_float(tail20.max()) if not tail20.empty else np.nan
    rsi_min20 = _safe_float(tail20.min()) if not tail20.empty else np.nan
    def _streak(cond_series: pd.Series) -> int:
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    streak_above70 = _streak((rsi >= 70).tail(60))
    streak_below30 = _streak((rsi <= 30).tail(60))
    div = _detect_divergence_simple(df["Close"], df["RSI"], lookback=60)
    return {
        "Value": last_rsi,
        "State": zone,
        "Direction": direction,
        "Divergence": div,
        "Delta5": delta5,
        "RSI_Series_6": tail6,
        "Max20": rsi_max20,
        "Min20": rsi_min20,
        "StreakAbove70": streak_above70,
        "StreakBelow30": streak_below30
    }

def compute_macd_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    macd_v = df["MACD"].dropna()
    sig = df["MACDSignal"].dropna()
    hist = df["MACDHist"].dropna()
    if macd_v.empty or sig.empty: return {}
    last_m = _safe_float(macd_v.iloc[-1])
    last_s = _safe_float(sig.iloc[-1])
    last_h = _safe_float(hist.iloc[-1]) if not hist.empty else np.nan
    state = "N/A"
    diff_ms = np.nan
    if pd.notna(last_m) and pd.notna(last_s):
        diff_ms = last_m - last_s
        if diff_ms > 0: state = "MACD_Above_Signal"
        elif diff_ms < 0: state = "MACD_Below_Signal"
        else: state = "MACD_Near_Signal"
    cross = _find_last_cross(df["MACD"], df["MACDSignal"], lookback=30)
    zero = "N/A"
    if pd.notna(last_m):
        if last_m > 0.0: zero = "Above"
        elif last_m < 0.0: zero = "Below"
        else: zero = "Near"
    h_tail6 = []
    if not hist.empty:
        h_tail6 = hist.tail(6).tolist()
        h_tail6 = [(_safe_float(x) if pd.notna(x) else np.nan) for x in h_tail6]
    hist_state = "N/A"
    if len(hist) >= 4:
        h0 = _safe_float(hist.iloc[-1])
        h1 = _safe_float(hist.iloc[-2])
        h2 = _safe_float(hist.iloc[-3])
        if pd.notna(h0) and pd.notna(h1) and pd.notna(h2):
            if h0 >= 0 and h1 >= 0:
                if (h0 > h1 > h2): hist_state = "Expanding_Positive"
                elif (h0 < h1 < h2): hist_state = "Contracting_Positive"
                else: hist_state = "Mixed_Positive"
            elif h0 < 0 and h1 < 0:
                if (h0 < h1 < h2): hist_state = "Expanding_Negative"
                elif (h0 > h1 > h2): hist_state = "Contracting_Negative"
                else: hist_state = "Mixed_Negative"
            else:
                hist_state = "Sign_Flip"
    def _streak(cond_series: pd.Series) -> int:
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    streak_pos = _streak((hist >= 0).tail(60)) if not hist.empty else 0
    streak_neg = _streak((hist < 0).tail(60)) if not hist.empty else 0
    delta5 = np.nan
    if len(macd_v) >= 6:
        delta5 = _safe_float(macd_v.iloc[-1] - macd_v.iloc[-6])
    div = _detect_divergence_simple(df["Close"], df["MACD"], lookback=60)
    return {
        "Value": last_m,
        "Signal": last_s,
        "Hist": last_h,
        "State": state,
        "Cross": cross,
        "ZeroLine": zero,
        "HistState": hist_state,
        "Divergence": div,
        "Diff_MACD_Signal": diff_ms,
        "MACD_Delta5": delta5,
        "Hist_Series_6": h_tail6,
        "Hist_Streak_Pos": streak_pos,
        "Hist_Streak_Neg": streak_neg
    }

def compute_volume_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    ratio = (vol / avg) if (pd.notna(vol) and pd.notna(avg) and avg != 0) else np.nan
    regime = "N/A"
    if pd.notna(ratio):
        if ratio >= 1.8: regime = "Spike"
        elif ratio >= 1.2: regime = "High"
        elif ratio >= 0.8: regime = "Normal"
        else: regime = "Low"
    return {"Vol": vol, "Avg20Vol": avg, "Ratio": ratio, "Regime": regime}

# --- STEP 6B: RSI+MACD BIAS (UPDATED) ---
def compute_rsi_macd_bias_features(rsi_feat: Dict[str, Any], macd_feat: Dict[str, Any]) -> Dict[str, Any]:
    if not rsi_feat or not macd_feat:
        return {"BiasCode": "N/A", "Alignment": "N/A", "Tags": ["MissingData"], "Facts": {}, "Notes": []}
    
    rsi_v = _safe_float(rsi_feat.get("Value"))
    rsi_zone = rsi_feat.get("State", "N/A")
    rsi_dir = rsi_feat.get("Direction", "N/A")
    rsi_div = rsi_feat.get("Divergence", {})
    rsi_div_type = rsi_div.get("Type", "None") if rsi_div else "None"
    
    macd_rel = macd_feat.get("State", "N/A")
    zero = macd_feat.get("ZeroLine", "N/A")
    hist_state = macd_feat.get("HistState", "N/A")
    cross = macd_feat.get("Cross", {})
    cross_event = cross.get("Event", "None") if cross else "None"
    cross_bars = cross.get("BarsAgo") if cross else None
    
    # 6B: Neutralized Bias Logic (Condition-based)
    alignment = "Mixed"
    rsi_pos = (rsi_zone in ["Zone60_70", "Zone70Plus"]) or (rsi_dir == "Rising" and rsi_v > 50)
    rsi_neg = (rsi_zone in ["Zone30_40", "ZoneBelow30"]) or (rsi_dir == "Falling" and rsi_v < 50)
    
    if rsi_zone == "Zone70Plus": alignment = "RSI_70Plus"
    elif rsi_zone == "ZoneBelow30": alignment = "RSI_Below30"
    else:
        # Check alignment in middle zones
        if rsi_pos and macd_rel == "MACD_Above_Signal":
            alignment = "Aligned_Positive"
        elif rsi_neg and macd_rel == "MACD_Below_Signal":
            alignment = "Aligned_Negative"
        else:
            alignment = "Mixed"
            
    tags: List[str] = [
        rsi_zone,
        macd_rel,
        f"MACD_ZeroLine={zero}",
        f"MACD_HistState={hist_state}",
        f"MACD_Cross={cross_event}",
        f"RSI_Direction={rsi_dir}",
        f"RSI_Divergence={rsi_div_type}",
        f"Alignment={alignment}",
    ]
    
    bias_code = "__".join([
        rsi_zone,
        macd_rel,
        f"Zero={zero}",
        f"Hist={hist_state}",
        f"Cross={cross_event}",
        f"Align={alignment}",
    ])
    
    notes: List[str] = []
    notes.append("Bias mô tả bằng điều kiện (facts), không kết luận tốt/xấu.")
    if cross_event != "None" and cross_bars is not None:
        notes.append(f"MACD_CrossEvent={cross_event}; BarsAgo={cross_bars}")
        
    return {
        "BiasCode": bias_code,
        "Alignment": alignment,
        "Tags": tags,
        "Facts": {
            "RSIZone": rsi_zone,
            "RSIValue": rsi_v if pd.notna(rsi_v) else np.nan,
            "RSIDirection": rsi_dir,
            "RSIDivergenceType": rsi_div_type,
            "MACDRelation": macd_rel,
            "MACDZeroLine": zero,
            "MACDHistState": hist_state,
            "MACDCrossEvent": cross_event,
            "MACDCrossBarsAgo": cross_bars
        },
        "Notes": notes
    }

# --- STEP 6: PRICE ACTION & PATTERNS ---
def _pct_dist(a: float, b: float, base: float) -> float:
    if pd.isna(a) or pd.isna(b) or pd.isna(base) or base == 0:
        return np.nan
    return abs(a - b) / abs(base) * 100

def _range_percentile(series: pd.Series, value: float) -> float:
    s = series.dropna()
    if s.empty or pd.isna(value): return np.nan
    return float((s <= value).mean() * 100)

def compute_price_action_features(df: pd.DataFrame, fib_ctx: Optional[Dict[str, Any]] = None, vol_feat: Optional[Dict[str, Any]] = None, tol_pct: float = 0.8) -> Dict[str, Any]:
    if df.empty: return {}
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None
    o = _safe_float(last.get("Open"), _safe_float(last.get("Close")))
    h = _safe_float(last.get("High"))
    l = _safe_float(last.get("Low"))
    c = _safe_float(last.get("Close"))
    if pd.isna(h) or pd.isna(l) or pd.isna(c): return {"Error": "Missing OHLC"}
    rng = h - l
    body = abs(c - o) if (pd.notna(o) and pd.notna(c)) else np.nan
    upper = (h - max(o, c)) if (pd.notna(o) and pd.notna(c)) else np.nan
    lower = (min(o, c) - l) if (pd.notna(o) and pd.notna(c)) else np.nan
    body_pct = (body / rng * 100) if (pd.notna(body) and rng > 0) else np.nan
    upper_pct = (upper / rng * 100) if (pd.notna(upper) and rng > 0) else np.nan
    lower_pct = (lower / rng * 100) if (pd.notna(lower) and rng > 0) else np.nan
    range_pct_close = (rng / c * 100) if (pd.notna(c) and c != 0 and rng >= 0) else np.nan
    direction = "N/A"
    if pd.notna(o) and pd.notna(c):
        direction = "Bull" if c > o else ("Bear" if c < o else "Flat")
    gap_pct = np.nan
    if prev is not None:
        pc = _safe_float(prev.get("Close"))
        if pd.notna(o) and pd.notna(pc) and pc != 0:
            gap_pct = (o - pc) / pc * 100
    ranges = (df["High"] - df["Low"]).tail(60)
    range_pctl_60 = _range_percentile(ranges, rng)
    doji = bool(pd.notna(body_pct) and body_pct <= 10)
    close_from_high_pct = ((h - c) / rng * 100) if (rng > 0 and pd.notna(c)) else np.nan
    close_from_low_pct = ((c - l) / rng * 100) if (rng > 0 and pd.notna(c)) else np.nan
    hammer = False
    shooting_star = False
    if (rng > 0) and pd.notna(body) and pd.notna(upper) and pd.notna(lower):
        hammer = (lower >= 2.0 * body) and (upper <= 0.6 * body) and (pd.notna(close_from_high_pct) and close_from_high_pct <= 30)
        shooting_star = (upper >= 2.0 * body) and (lower <= 0.6 * body) and (pd.notna(close_from_low_pct) and close_from_low_pct <= 30)
    bullish_engulf = False
    bearish_engulf = False
    inside_bar = False
    outside_bar = False
    if prev is not None:
        po = _safe_float(prev.get("Open"), _safe_float(prev.get("Close")))
        ph = _safe_float(prev.get("High"))
        pl = _safe_float(prev.get("Low"))
        pc = _safe_float(prev.get("Close"))
        if pd.notna(ph) and pd.notna(pl):
            inside_bar = bool(h < ph and l > pl)
            outside_bar = bool(h > ph and l < pl)
        if pd.notna(po) and pd.notna(pc) and pd.notna(o) and pd.notna(c):
            prev_bear = pc < po
            prev_bull = pc > po
            last_bull = c > o
            last_bear = c < o
            bullish_engulf = bool(last_bull and prev_bear and (c >= po) and (o <= pc))
            bearish_engulf = bool(last_bear and prev_bull and (o >= pc) and (c <= po))
    patterns = {
        "Doji": doji, "Hammer": hammer, "ShootingStar": shooting_star,
        "BullishEngulfing": bullish_engulf, "BearishEngulfing": bearish_engulf,
        "InsideBar": inside_bar, "OutsideBar": outside_bar,
    }
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    near_ma = {
        "MA20": {"Value": ma20, "DistPct": _pct_dist(c, ma20, c), "Near": bool(pd.notna(ma20) and _pct_dist(c, ma20, c) <= tol_pct)},
        "MA50": {"Value": ma50, "DistPct": _pct_dist(c, ma50, c), "Near": bool(pd.notna(ma50) and _pct_dist(c, ma50, c) <= tol_pct)},
        "MA200": {"Value": ma200, "DistPct": _pct_dist(c, ma200, c), "Near": bool(pd.notna(ma200) and _pct_dist(c, ma200, c) <= tol_pct)},
    }
    near_fib_short = []
    near_fib_long = []
    if fib_ctx:
        for side in ["Supports", "Resistances"]:
            for it in (fib_ctx.get("NearestShort", {}) or {}).get(side, []) or []:
                d = _safe_float(it.get("DistPct"))
                if pd.notna(d) and abs(d) <= tol_pct:
                    near_fib_short.append({"Side": side, **it})
            for it in (fib_ctx.get("NearestLong", {}) or {}).get(side, []) or []:
                d = _safe_float(it.get("DistPct"))
                if pd.notna(d) and abs(d) <= tol_pct:
                    near_fib_long.append({"Side": side, **it})
    vol_regime = (vol_feat.get("Regime") if vol_feat else "N/A")
    notes = []
    notes.append(f"Dir={direction}, BodyPct={body_pct}, WickU={upper_pct}, WickL={lower_pct}, RangePct={range_pct_close}")
    if pd.notna(range_pctl_60): notes.append(f"RangePercentile60={range_pctl_60}")
    if pd.notna(gap_pct): notes.append(f"GapPct={gap_pct}")
    notes.append(f"VolRegime={vol_regime}")
    return {
        "Candle": {
            "Open": o, "High": h, "Low": l, "Close": c, "Direction": direction,
            "BodyPct": body_pct, "UpperWickPct": upper_pct, "LowerWickPct": lower_pct,
            "RangePctOfClose": range_pct_close, "GapPct": gap_pct,
            "RangePercentile60": range_pctl_60, "CloseFromHighPct": close_from_high_pct, "CloseFromLowPct": close_from_low_pct,
        },
        "Patterns": patterns,
        "Context": {
            "NearMA": near_ma, "NearFibShort": near_fib_short, "NearFibLong": near_fib_long, "VolumeRegime": vol_regime,
            "FiboConflictFlag": bool(fib_ctx.get("FiboConflictFlag")) if fib_ctx else False,
            "FiboPriorityRuleApplied": (fib_ctx.get("FiboPriorityRuleApplied") if fib_ctx else "None"),
        },
        "Notes": notes
    }

# --- STEP 7B: NEW CONTEXT FEATURES ---
def compute_rsi_context_features(df: pd.DataFrame, rsi_col: str = "RSI") -> Dict[str, Any]:
    if df.empty or rsi_col not in df.columns:
        return {"Streak70": 0, "Cross70BarsAgo": None, "Delta3": np.nan, "Delta5": np.nan}
    
    rsi = df[rsi_col].dropna()
    if rsi.empty: return {}
    
    last_r = _safe_float(rsi.iloc[-1])
    
    def _streak(cond_series):
        cnt = 0
        for v in reversed(cond_series.tolist()):
            if bool(v): cnt += 1
            else: break
        return int(cnt)
    
    streak70 = _streak((rsi >= 70).tail(60))
    
    # Cross 70 up check
    # We look back to find when it crossed 70 from below
    cross_70_idx = None
    vals = rsi.values
    for i in range(len(vals)-2, -1, -1):
        if vals[i] < 70 and vals[i+1] >= 70:
            cross_70_idx = len(vals) - 1 - (i+1)
            break
            
    delta3 = last_r - _safe_float(rsi.iloc[-4]) if len(rsi) >= 4 else np.nan
    delta5 = last_r - _safe_float(rsi.iloc[-6]) if len(rsi) >= 6 else np.nan
    
    return {
        "Streak70": streak70,
        "Cross70BarsAgo": int(cross_70_idx) if cross_70_idx is not None else None,
        "Delta3": delta3,
        "Delta5": delta5,
        "Turning": "Falling" if pd.notna(delta3) and delta3 < -2 else ("Rising" if pd.notna(delta3) and delta3 > 2 else "Flat")
    }

def compute_volume_context_features(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or "Volume" not in df.columns:
        return {"VolStreakUp": 0, "VolTrend": "N/A"}
        
    vol = df["Volume"].dropna()
    if len(vol) < 5: return {"VolStreakUp": 0, "VolTrend": "N/A"}
    
    # Streak of rising volume
    cnt = 0
    vals = vol.values
    for i in range(len(vals)-1, 0, -1):
        if vals[i] > vals[i-1]: cnt += 1
        else: break
        
    # Simple slope of Vol MA
    avg20 = df["Avg20Vol"] if "Avg20Vol" in df.columns else sma(vol, 20)
    slope = "Flat"
    if len(avg20) >= 5:
        a = avg20.dropna()
        if len(a) >= 5:
            delta = a.iloc[-1] - a.iloc[-5]
            if delta > 0: slope = "Rising"
            elif delta < 0: slope = "Falling"
            
    return {"VolStreakUp": int(cnt), "VolTrend": slope}

def compute_level_context_features(last: pd.Series, dual_fib: Dict[str, Any]) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    if pd.isna(c): return {}
    
    # Merge levels
    levels = {}
    levels.update(dual_fib.get("auto_short", {}).get("levels", {}))
    levels.update(dual_fib.get("fixed_long", {}).get("levels", {}))
    
    sorted_lv = sorted([(k, v) for k,v in levels.items() if pd.notna(v)], key=lambda x: x[1])
    
    sup = [(k,v, (v-c)/c*100) for k,v in sorted_lv if v <= c]
    res = [(k,v, (v-c)/c*100) for k,v in sorted_lv if v >= c]
    
    sup = sorted(sup, key=lambda x: abs(x[2]))[:1] # Nearest
    res = sorted(res, key=lambda x: abs(x[2]))[:1] # Nearest
    
    sup_pack = {"Label": sup[0][0], "Level": sup[0][1], "DistPct": sup[0][2]} if sup else {"Label": "N/A", "Level": np.nan, "DistPct": np.nan}
    res_pack = {"Label": res[0][0], "Level": res[0][1], "DistPct": res[0][2]} if res else {"Label": "N/A", "Level": np.nan, "DistPct": np.nan}
    
    return {
        "NearestSupport": sup_pack,
        "NearestResistance": res_pack
    }

# ============================================================
# 7C. STRUCTURE QUALITY PACK (Support/Resistance Quality-Aware)
# ============================================================
def compute_structure_quality_pack(
    df: pd.DataFrame,
    last: pd.Series,
    dual_fib: Optional[Dict[str, Any]] = None,
    fib_ctx: Optional[Dict[str, Any]] = None,
    *,
    daily_lookback: int = 60,
    weekly_lookback_weeks: int = 78
) -> Dict[str, Any]:
    """
    Computes a quality-aware Support/Resistance pack to prevent structural blind spots:
    - Distinguishes Tactical (60–90D/daily) vs Structural (~250D/weekly) levels
    - Distinguishes strength tiers (LIGHT/MED/HEAVY/CONFLUENCE)
    - Provides a CeilingGate for bull-trap control (breakout over weak tactical vs structural ceiling)

    Output is facts-only; GPT/renderer should only interpret.
    """

    dual_fib = dual_fib if isinstance(dual_fib, dict) else {}
    fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}

    c = _safe_float(last.get("Close"))
    if pd.isna(c) or c == 0:
        return {
            "Meta": {"Price": np.nan, "VolPct_ATRProxy": np.nan, "NearThresholdPct": np.nan, "ZoneWidthPct": np.nan},
            "OverheadResistance": {"Nearest": {}},
            "UnderlyingSupport": {"Nearest": {}},
            "Gates": {"CeilingGate": {"Status": "N/A", "Reason": "Missing price", "Horizon": "N/A", "Tier": "N/A",
                                      "NearestLevelPrice": None, "DistancePct": None}}
        }

    denom = _dynamic_vol_proxy(df, 20)
    vol_pct = (denom / c * 100.0) if (pd.notna(denom) and pd.notna(c) and c != 0) else np.nan

    near_th = float(_clip(max(1.0, 0.8 * vol_pct) if pd.notna(vol_pct) else 1.5, 1.0, 2.5))
    zone_w = float(_clip(max(0.8, 0.6 * vol_pct) if pd.notna(vol_pct) else 1.0, 0.8, 2.0))

    def _cand(type_: str, horizon: str, price: float, weight: float) -> Dict[str, Any]:
        dist_pct = ((price - c) / c * 100.0) if (pd.notna(price) and pd.notna(c) and c != 0) else np.nan
        return {"Type": type_, "Horizon": horizon, "Price": float(price), "Weight": float(weight), "DistPct": float(dist_pct)}

    W = {
        "MA200": 4.0,
        "WEEKLY_SWING": 4.0,
        "FIB_LONG_61.8": 3.5,
        "FIB_LONG_50.0": 2.5,
        "FIB_LONG_38.2": 2.0,
        "MA50": 3.0,
        "DAILY_SWING": 2.5,
        "FIB_SHORT_61.8": 2.5,
        "FIB_SHORT_50.0": 2.0,
        "FIB_SHORT_38.2": 1.5,
        "MA20": 1.5,
        "FIB_EXT": 1.0,
    }

    candidates: List[Dict[str, Any]] = []

    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    if pd.notna(ma20) and ma20 > 0:
        candidates.append(_cand("MA20", "TACTICAL", ma20, W["MA20"]))
    if pd.notna(ma50) and ma50 > 0:
        candidates.append(_cand("MA50", "STRUCTURAL", ma50, W["MA50"]))
    if pd.notna(ma200) and ma200 > 0:
        candidates.append(_cand("MA200", "STRUCTURAL", ma200, W["MA200"]))

    short_lv = (dual_fib.get("auto_short", {}) or {}).get("levels", {}) or {}
    long_lv = (dual_fib.get("fixed_long", {}) or {}).get("levels", {}) or {}

    def _add_fib(levels: Dict[str, Any], prefix: str, horizon: str):
        for k, v in (levels or {}).items():
            lv = _safe_float(v)
            if pd.isna(lv) or lv <= 0:
                continue
            kk = str(k).strip()
            if kk in ("61.8", "61.80"):
                typ = f"{prefix}_61.8"; w = W.get(f"FIB_{prefix}_61.8", 2.0)
            elif kk in ("50.0", "50", "50.00"):
                typ = f"{prefix}_50.0"; w = W.get(f"FIB_{prefix}_50.0", 1.5)
            elif kk in ("38.2", "38.20"):
                typ = f"{prefix}_38.2"; w = W.get(f"FIB_{prefix}_38.2", 1.2)
            else:
                typ = f"{prefix}_EXT"; w = W.get("FIB_EXT", 1.0)
            # Type example: FIB_SHORT_61.8 / FIB_LONG_61.8
            candidates.append(_cand(f"FIB_{typ}", horizon, lv, float(w)))

    _add_fib(short_lv, "SHORT", "TACTICAL")
    _add_fib(long_lv, "LONG", "STRUCTURAL")

    def _extract_daily_pivots(_df: pd.DataFrame, lb: int = 60) -> Tuple[List[float], List[float]]:
        if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty:
            return ([], [])
        if "High" not in _df.columns or "Low" not in _df.columns:
            return ([], [])
        d = _df.tail(int(lb)).copy()
        hi = d["High"]
        lo = d["Low"]
        piv_hi, piv_lo = [], []
        for i in range(1, len(d) - 1):
            h0, h1, h2 = _safe_float(hi.iloc[i-1]), _safe_float(hi.iloc[i]), _safe_float(hi.iloc[i+1])
            l0, l1, l2 = _safe_float(lo.iloc[i-1]), _safe_float(lo.iloc[i]), _safe_float(lo.iloc[i+1])
            if pd.notna(h1) and pd.notna(h0) and pd.notna(h2) and h1 > h0 and h1 > h2:
                piv_hi.append(float(h1))
            if pd.notna(l1) and pd.notna(l0) and pd.notna(l2) and l1 < l0 and l1 < l2:
                piv_lo.append(float(l1))
        return (piv_hi, piv_lo)

    dh, dl = _extract_daily_pivots(df, daily_lookback)
    for p in dh:
        candidates.append(_cand("DAILY_SWING_HIGH", "TACTICAL", p, W["DAILY_SWING"]))
    for p in dl:
        candidates.append(_cand("DAILY_SWING_LOW", "TACTICAL", p, W["DAILY_SWING"]))

    def _extract_weekly_pivots(_df: pd.DataFrame, lb_weeks: int = 78) -> Tuple[List[float], List[float]]:
        if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty:
            return ([], [])
        if "High" not in _df.columns or "Low" not in _df.columns:
            return ([], [])
        d = _df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            for col in ("Date", "Datetime", "date", "datetime"):
                if col in d.columns:
                    try:
                        d[col] = pd.to_datetime(d[col], errors="coerce")
                        d = d.set_index(col)
                        break
                    except Exception:
                        pass
        if not isinstance(d.index, pd.DatetimeIndex):
            return ([], [])
        d = d.sort_index()
        w = d[["High", "Low"]].resample("W").agg({"High": "max", "Low": "min"}).dropna()
        if w.empty:
            return ([], [])
        w = w.tail(int(lb_weeks)).copy()
        wh = w["High"]
        wl = w["Low"]
        piv_hi, piv_lo = [], []
        for i in range(1, len(w) - 1):
            h0, h1, h2 = _safe_float(wh.iloc[i-1]), _safe_float(wh.iloc[i]), _safe_float(wh.iloc[i+1])
            l0, l1, l2 = _safe_float(wl.iloc[i-1]), _safe_float(wl.iloc[i]), _safe_float(wl.iloc[i+1])
            if pd.notna(h1) and pd.notna(h0) and pd.notna(h2) and h1 > h0 and h1 > h2:
                piv_hi.append(float(h1))
            if pd.notna(l1) and pd.notna(l0) and pd.notna(l2) and l1 < l0 and l1 < l2:
                piv_lo.append(float(l1))
        return (piv_hi, piv_lo)

    wh, wl = _extract_weekly_pivots(df, weekly_lookback_weeks)
    for p in wh:
        candidates.append(_cand("WEEKLY_SWING_HIGH", "STRUCTURAL", p, W["WEEKLY_SWING"]))
    for p in wl:
        candidates.append(_cand("WEEKLY_SWING_LOW", "STRUCTURAL", p, W["WEEKLY_SWING"]))

    def _cluster_nearest(side: str) -> Dict[str, Any]:
        if side == "overhead":
            pool = [x for x in candidates if pd.notna(x.get("Price")) and x["Price"] >= c]
            pool = sorted(pool, key=lambda x: x["Price"])
        else:
            pool = [x for x in candidates if pd.notna(x.get("Price")) and x["Price"] <= c]
            pool = sorted(pool, key=lambda x: x["Price"], reverse=True)

        if not pool:
            return {}

        clusters: List[List[Dict[str, Any]]] = []
        for it in pool:
            placed = False
            for cl in clusters:
                center = float(np.mean([z["Price"] for z in cl]))
                if abs(it["Price"] - center) / c * 100.0 <= zone_w:
                    cl.append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([it])

        def _center_dist(cl: List[Dict[str, Any]]) -> float:
            center = float(np.mean([z["Price"] for z in cl]))
            return abs((center - c) / c * 100.0)

        clusters = sorted(clusters, key=_center_dist)
        chosen = clusters[0]
        center = float(np.mean([z["Price"] for z in chosen]))
        dist_pct = abs((center - c) / c * 100.0)

        horizons = {str(z.get("Horizon", "N/A")).upper() for z in chosen}
        has_struct = "STRUCTURAL" in horizons
        has_tact = "TACTICAL" in horizons
        horizon = "BOTH" if (has_struct and has_tact) else ("STRUCTURAL" if has_struct else "TACTICAL")

        confluence_count = int(len(chosen))
        confluence_mult = float(_clip(1.0 + 0.25 * max(0, confluence_count - 1), 1.0, 1.75))

        if dist_pct <= near_th:
            near_factor = 1.0
        else:
            near_factor = max(0.6, near_th / dist_pct) if dist_pct > 0 else 1.0

        raw = sum(float(z.get("Weight", 0.0)) for z in chosen) * confluence_mult * float(near_factor)
        quality = float(_clip(raw * 1.2, 0.0, 10.0))

        if quality >= 8.5 or (confluence_count >= 3 and has_struct):
            tier = "CONFLUENCE"
        elif quality >= 6.5:
            tier = "HEAVY"
        elif quality >= 4.0:
            tier = "MED"
        else:
            tier = "LIGHT"

        low = float(min(z["Price"] for z in chosen))
        high = float(max(z["Price"] for z in chosen))
        comps = sorted(chosen, key=lambda x: (-float(x.get("Weight", 0.0)), abs(float(x.get("DistPct", 0.0)))))[:3]

        return {
            "Zone": {"Low": low, "High": high},
            "Center": center,
            "DistancePct": float(dist_pct),
            "Horizon": horizon,
            "Tier": tier,
            "QualityScore": quality,
            "ComponentsTop": comps,
            "ConfluenceCount": confluence_count
        }

    overhead = _cluster_nearest("overhead")
    support = _cluster_nearest("support")

    gate = {"Status": "N/A", "Reason": "N/A", "Horizon": "N/A", "Tier": "N/A", "NearestLevelPrice": None, "DistancePct": None}
    if overhead:
        dist = _safe_float(overhead.get("DistancePct"), default=np.nan)
        tier = str(overhead.get("Tier", "N/A")).upper()
        hz = str(overhead.get("Horizon", "N/A")).upper()
        gate.update({"Horizon": hz, "Tier": tier, "NearestLevelPrice": float(overhead.get("Center", np.nan)), "DistancePct": float(dist)})

        if pd.isna(dist) or dist > near_th:
            gate.update({"Status": "PASS", "Reason": "No near ceiling"})
        else:
            if hz in ("STRUCTURAL", "BOTH") and tier in ("HEAVY", "CONFLUENCE"):
                status = "WAIT"
                reason = "Structural ceiling near"
                if hz == "BOTH" and tier == "CONFLUENCE" and dist <= 0.5 * near_th:
                    status = "FAIL"
                    reason = "Confluence ceiling too close"
                gate.update({"Status": status, "Reason": reason})
            else:
                gate.update({"Status": "PASS", "Reason": "Ceiling manageable"})

    return {
        "Meta": {
            "Price": float(c),
            "VolPct_ATRProxy": float(vol_pct) if pd.notna(vol_pct) else np.nan,
            "NearThresholdPct": float(near_th),
            "ZoneWidthPct": float(zone_w),
        },
        "OverheadResistance": {"Nearest": overhead or {}},
        "UnderlyingSupport": {"Nearest": support or {}},
        "Gates": {"CeilingGate": gate}
    }

def compute_market_context(df_all: pd.DataFrame) -> Dict[str, Any]:
    def pack(tick: str) -> Dict[str, Any]:
        d = df_all[df_all["Ticker"].astype(str).str.upper() == tick].copy()
        if d.empty or len(d) < 2: return {"Ticker": tick, "Close": np.nan, "ChangePct": np.nan, "Regime": "N/A"}
        d = d.sort_values("Date")
        c = _safe_float(d.iloc[-1].get("Close"))
        p = _safe_float(d.iloc[-2].get("Close"))
        chg = _pct_change(c, p)
        regime = "N/A"
        try:
            d["MA50"] = sma(d["Close"], 50)
            d["MA200"] = sma(d["Close"], 200)
            ma50 = _safe_float(d.iloc[-1].get("MA50"))
            ma200 = _safe_float(d.iloc[-1].get("MA200"))
            if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
                if c >= ma50 and ma50 >= ma200: regime = "Up"
                elif c < ma50 and ma50 < ma200: regime = "Down"
                else: regime = "Neutral"
        except: regime = "N/A"
        return {"Ticker": tick, "Close": c, "ChangePct": chg, "Regime": regime}
    vnindex = pack("VNINDEX")
    vn30 = pack("VN30")
    return {"VNINDEX": vnindex, "VN30": vn30}

# ============================================================
# 7. CONVICTION SCORE (Step 5B Update: Key/Hit/Weight Only)
# ============================================================
def compute_conviction_pack(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    vol = _safe_float(last.get("Volume"))
    avg = _safe_float(last.get("Avg20Vol"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    
    score = 5.0
    components = []
    notes = []
    
    # Rule 1: Above MA200 (+2)
    hit1 = bool(pd.notna(c) and pd.notna(ma200) and c > ma200)
    if hit1:
        score += 2.0
    components.append({"Key": "PriceAboveMA200", "Hit": hit1, "Weight": 2.0})
    notes.append(f"PriceAboveMA200={hit1} (+2.0)")
    
    # Rule 2: RSI > 55 (+1)
    hit2 = bool(pd.notna(rsi) and rsi > 55)
    if hit2:
        score += 1.0
    components.append({"Key": "RSIAbove55", "Hit": hit2, "Weight": 1.0})
    notes.append(f"RSIAbove55={hit2} (+1.0)")
        
    # Rule 3: Volume > Avg20Vol (+1)
    vol_ratio = (vol / avg) if (pd.notna(vol) and pd.notna(avg) and avg != 0) else np.nan
    hit3 = bool(pd.notna(vol_ratio) and vol_ratio > 1.0)
    if hit3:
        score += 1.0
    components.append({"Key": "VolumeAboveAvg20", "Hit": hit3, "Weight": 1.0})
    notes.append(f"VolumeAboveAvg20={hit3} (+1.0)")
            
    # Rule 4: MACD > Signal (+0.5)
    hit4 = bool(pd.notna(macd_v) and pd.notna(sig) and macd_v > sig)
    if hit4:
        score += 0.5
    components.append({"Key": "MACDAboveSignal", "Hit": hit4, "Weight": 0.5})
    notes.append(f"MACDAboveSignal={hit4} (+0.5)")
        
    score = float(min(10.0, score))
    return {
        "Score": round(score, 2),
        "Components": components,
        "Notes": notes, # Now fully neutral tags
        "Facts": {
            "Close": c, "MA200": ma200, "RSI": rsi,
            "VolRatio": vol_ratio, "MACD": macd_v, "Signal": sig
        }
    }

def compute_conviction(last: pd.Series) -> float:
    pack = compute_conviction_pack(last)
    return _safe_float(pack.get("Score"), default=5.0)

# ============================================================
# 8. TRADE PLAN LOGIC (Step 9+10 / v5.6 patch for Step 8)
# - Remove fixed % buffer
# - Stop/Entry/TP anchored to MA+Fib (short+long) + dynamic buffer
# - Add Probability label computed in Python (no GPT math)
# ============================================================
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional


# ------------------------------
# JSON safe-serialization helpers
# ------------------------------
@dataclass
class TradeSetup:
    name: str
    entry: float
    stop: float
    tp: float
    rr: float
    probability: str
    status: str = "Watch"
    reason_tags: List[str] = field(default_factory=list)

def _compute_rr(entry: float, stop: float, tp: float) -> float:
    if any(pd.isna([entry, stop, tp])) or entry <= stop:
        return np.nan
    risk = entry - stop
    reward = tp - entry
    return (reward / risk) if risk > 0 else np.nan

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if df.empty or not all(c in df.columns for c in ["High", "Low", "Close"]):
        return pd.Series(dtype=float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def _dynamic_vol_proxy(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Fallback volatility proxy when ATR is unavailable.
    Data-driven (no fixed %):
    - median(|Close.diff|) + 0.25 * median(High-Low)
    """
    if df.empty or "Close" not in df.columns:
        return np.nan
    d = df.tail(max(lookback, 10)).copy()
    if d.empty:
        return np.nan
    cd = d["Close"].astype(float).diff().abs().dropna()
    rng = (d["High"].astype(float) - d["Low"].astype(float)).abs().dropna() if ("High" in d.columns and "Low" in d.columns) else pd.Series(dtype=float)
    m1 = _safe_float(cd.median()) if not cd.empty else np.nan
    m2 = _safe_float(rng.median()) if not rng.empty else np.nan
    if pd.notna(m1) and pd.notna(m2):
        return float(m1 + 0.25 * m2)
    if pd.notna(m1):
        return float(m1)
    if pd.notna(m2):
        return float(0.25 * m2)
    return np.nan

def _buffer_price_dynamic(df: pd.DataFrame, entry: float) -> float:
    """
    Buffer = max(0.5*ATR14, vol_proxy)
    - No fixed percent
    - Entirely derived from price/volatility series
    """
    if pd.isna(entry) or entry == 0 or df.empty:
        return np.nan

    a = atr_wilder(df, 14)
    atr_last = _safe_float(a.dropna().iloc[-1]) if not a.dropna().empty else np.nan
    b_atr = (0.5 * atr_last) if pd.notna(atr_last) else np.nan

    b_proxy = _dynamic_vol_proxy(df, lookback=20)

    cands = [x for x in [b_atr, b_proxy] if pd.notna(x) and x > 0]
    if not cands:
        return np.nan
    return float(max(cands))

def _collect_levels(levels: Dict[str, Any]) -> List[Tuple[str, float]]:
    out = []
    for k, v in (levels or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            out.append((str(k), float(fv)))
    out.sort(key=lambda x: x[1])
    return out

def _nearest_above(levels: Dict[str, Any], x: float) -> Tuple[Optional[str], float]:
    best_k, best_v = None, np.nan
    for k, v in _collect_levels(levels):
        if pd.isna(x): 
            continue
        if v > x:
            if best_k is None or v < best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _nearest_below(levels: Dict[str, Any], x: float) -> Tuple[Optional[str], float]:
    best_k, best_v = None, np.nan
    for k, v in _collect_levels(levels):
        if pd.isna(x):
            continue
        if v < x:
            if best_k is None or v > best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _vol_ratio(df: pd.DataFrame) -> float:
    if df.empty or "Volume" not in df.columns:
        return np.nan
    vol = _safe_float(df.iloc[-1].get("Volume"))
    avg = _safe_float(df.iloc[-1].get("Avg20Vol")) if "Avg20Vol" in df.columns else np.nan
    if pd.notna(vol) and pd.notna(avg) and avg != 0:
        return float(vol / avg)
    return np.nan

def _probability_label_from_facts(df: pd.DataFrame, rr: float, status: str, vr: float) -> str:
    """
    Neutral probability tag derived only from technical facts already computed in df:
    - Trend (Close/MA50/MA200)
    - Momentum (RSI + MACD vs Signal)
    - Volume ratio
    - RR quality
    """
    if df.empty or pd.isna(rr) or status == "Invalid":
        return "N/A"

    last = df.iloc[-1]
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))

    pts = 0.0

    # Status weight
    st = (status or "Watch").strip().lower()
    if st == "active":
        pts += 0.8
    elif st == "watch":
        pts += 0.3

    # RR weight
    if rr >= 4.0:
        pts += 1.4
    elif rr >= 3.0:
        pts += 1.0
    elif rr >= 1.8:
        pts += 0.5
    else:
        pts += 0.1

    # Trend (structure)
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            pts += 0.9
        elif (c < ma50) and (ma50 < ma200):
            pts -= 0.7
        else:
            pts += 0.2

    # Momentum (RSI + MACD relation)
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            pts += 0.7
        elif (rsi <= 45) and (macd_v < sig):
            pts -= 0.5
        else:
            pts += 0.15

        # RSI>=70 is treated as neutral regime (no extra bullish pts)
        if rsi >= 70:
            pts += 0.0

    # Volume confirmation
    if pd.notna(vr):
        if vr >= 1.1:
            pts += 0.35
        elif vr < 0.8:
            pts -= 0.25

    if pts >= 2.4:
        return "High"
    if pts >= 1.3:
        return "Medium"
    return "Low"

def _build_anchor_level_map(df: pd.DataFrame, fib_short: Dict[str, Any], fib_long: Dict[str, Any]) -> Dict[str, float]:
    """
    Merge MA + fib(short+long) into a single anchor map.
    Keys are tags; values are prices.
    """
    last = df.iloc[-1] if not df.empty else pd.Series(dtype=object)
    anchors: Dict[str, float] = {}

    for ma_key in ["MA20", "MA50", "MA200"]:
        v = _safe_float(last.get(ma_key))
        if pd.notna(v):
            anchors[ma_key] = float(v)

    for k, v in (fib_short or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            anchors[f"FibS_{k}"] = float(fv)

    for k, v in (fib_long or {}).items():
        fv = _safe_float(v)
        if pd.notna(fv):
            anchors[f"FibL_{k}"] = float(fv)

    return anchors

def _nearest_support_below(anchors: Dict[str, float], x: float, exclude_vals: Optional[List[float]] = None) -> Tuple[str, float]:
    """
    Choose the nearest support strictly below x (i.e., the HIGHEST level < x).
    """
    exclude_vals = exclude_vals or []
    best_k, best_v = "N/A", np.nan
    for k, v in (anchors or {}).items():
        if pd.isna(x) or pd.isna(v):
            continue
        if any(pd.notna(ev) and abs(v - ev) <= 1e-9 for ev in exclude_vals):
            continue
        if v < x:
            if pd.isna(best_v) or v > best_v:
                best_k, best_v = k, v
    return best_k, best_v

def _nearest_resistance_above(anchors: Dict[str, float], x: float, exclude_vals: Optional[List[float]] = None) -> Tuple[str, float]:
    """
    Choose the nearest resistance strictly above x (i.e., the LOWEST level > x).
    """
    exclude_vals = exclude_vals or []
    best_k, best_v = "N/A", np.nan
    for k, v in (anchors or {}).items():
        if pd.isna(x) or pd.isna(v):
            continue
        if any(pd.notna(ev) and abs(v - ev) <= 1e-9 for ev in exclude_vals):
            continue
        if v > x:
            if pd.isna(best_v) or v < best_v:
                best_k, best_v = k, v
    return best_k, best_v


# ============================================================
# 8. TRADE PLAN (STRENGTH-AWARE) — v9.9
# - Adds FibStrength + MAStrength + Confluence scoring for Stop/TP selection
# - Uses tol = 0.25*ATR (fallback: 0.3%*Close) for zone matching
# - Replaces fixed TP fallback (3.0R/2.6R) with Class/Regime-configured k·R
# ============================================================

def _atr14_last(df: pd.DataFrame) -> float:
    a = atr_wilder(df, 14)
    try:
        v = _safe_float(a.dropna().iloc[-1]) if not a.dropna().empty else np.nan
    except Exception:
        v = np.nan
    return v


def _tol_price(df: pd.DataFrame, close: float) -> float:
    """Tolerance used for confluence/zone matching.

    Rule: tol = 0.25*ATR14 (fallback: 0.3%*Close when ATR unavailable).
    """
    atr14 = _atr14_last(df)
    if pd.notna(atr14) and atr14 > 0:
        return float(0.25 * atr14)
    # ATR missing → fallback by % of price
    if pd.notna(close) and close > 0:
        return float(0.003 * close)
    return np.nan


def _fib_strength_from_key(k: str) -> int:
    """Discrete strength for the fib keys used in this project (38.2/50.0/61.8/127.2/161.8)."""
    s = (str(k) or '').strip()
    # normalize
    s = s.replace('%', '').replace(' ', '')
    # Retracements
    if s in ('61.8', '61.80'):
        return 3
    if s in ('50.0', '50', '50.00', '38.2', '38.20'):
        return 2
    # Extensions
    if s in ('161.8', '161.80'):
        return 3
    if s in ('127.2', '127.20'):
        return 2
    return 1


def _infer_character_class_quick(df: pd.DataFrame) -> str:
    """Quick, deterministic class inference used ONLY for k·R fallback in Trade Plan.

    This is a lightweight proxy that uses only local price/volume/ATR context.
    Returns one of the new 8-class taxonomy (subset, when signals are insufficient):
      - Smooth Trend
      - Momentum Trend
      - Aggressive Trend
      - Range / Mean-Reversion (Stable)
      - Volatile Range
      - Mixed / Choppy Trader
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "Mixed / Choppy Trader"

    last = df.iloc[-1]
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))

    vr = _vol_ratio(df)
    atr14 = _atr14_last(df)
    atr_pct = (atr14 / close * 100) if (pd.notna(atr14) and pd.notna(close) and close != 0) else np.nan

    trend_stack = (pd.notna(ma20) and pd.notna(ma50) and pd.notna(ma200) and ma20 > ma50 > ma200)
    price_above = (pd.notna(close) and pd.notna(ma20) and close > ma20)

    very_wild = (pd.notna(atr_pct) and atr_pct >= 6.0) or (pd.notna(vr) and vr >= 2.5)
    high_vol = (pd.notna(atr_pct) and atr_pct >= 4.5) or (pd.notna(vr) and vr >= 1.8)

    if trend_stack and price_above and (not high_vol) and (pd.isna(rsi) or rsi >= 50):
        return "Smooth Trend"

    if trend_stack and (pd.notna(rsi) and rsi >= 55) and (pd.notna(vr) and vr >= 1.1):
        return "Momentum Trend"

    if very_wild:
        return "Aggressive Trend" if trend_stack else "Volatile Range"

    # Range proxy: price around MA50 and RSI mid
    if pd.notna(close) and pd.notna(ma50) and ma50 != 0 and abs(close - ma50) / ma50 * 100 <= 2.0 and (pd.notna(rsi) and 40 <= rsi <= 60):
        return "Range / Mean-Reversion (Stable)"

    return "Mixed / Choppy Trader"


def _infer_risk_regime_quick(df: pd.DataFrame) -> str:
    """Quick risk regime tag used ONLY for k·R fallback."""
    if df.empty:
        return 'Normal'
    last = df.iloc[-1]
    close = _safe_float(last.get('Close'))
    avg20 = _safe_float(last.get('Avg20Vol'))
    vr = _vol_ratio(df)
    atr14 = _atr14_last(df)
    atr_pct = (atr14 / close * 100) if (pd.notna(atr14) and pd.notna(close) and close != 0) else np.nan

    if pd.notna(avg20) and avg20 > 0 and avg20 < 120_000:
        return 'LowLiquidity'
    if (pd.notna(atr_pct) and atr_pct >= 6.0) or (pd.notna(vr) and vr >= 2.8):
        return 'EventRisk'
    if (pd.notna(atr_pct) and atr_pct >= 4.5) or (pd.notna(vr) and vr >= 2.0):
        return 'HighVol'
    return 'Normal'


def _kr_fallback_mult(setup: str, cclass: str, regime: str) -> float:
    """k·R fallback multiplier per CharacterClass/Regime. Deterministic config.

    IMPORTANT:
    - This is a fallback only; it must be stable and consistent with the new 8-class taxonomy.
    - It does NOT use any 'Now/Opportunity' inputs.
    """
    setup_k = (setup or "").strip().lower()
    cc = (cclass or "Mixed / Choppy Trader").strip()
    rg = (regime or "Normal").strip()

    # Base by class (keeps prior behavior: breakout > pullback, momentum/aggressive higher)
    base = {
        "Smooth Trend": {"breakout": 2.4, "pullback": 2.2},
        "Momentum Trend": {"breakout": 3.0, "pullback": 2.6},
        "Aggressive Trend": {"breakout": 3.2, "pullback": 2.8},
        "Range / Mean-Reversion (Stable)": {"breakout": 2.0, "pullback": 1.7},
        "Volatile Range": {"breakout": 2.4, "pullback": 2.0},
        "Mixed / Choppy Trader": {"breakout": 2.2, "pullback": 2.0},
        "Event / Gap-Prone": {"breakout": 2.6, "pullback": 2.3},
        "Illiquid / Noisy": {"breakout": 2.0, "pullback": 1.8},
    }
    by_setup = base.get(cc, base["Mixed / Choppy Trader"])
    k = by_setup.get("breakout" if "break" in setup_k else "pullback", 2.2)

    # Regime adjustment (risk-aware, modest)
    adj = {
        "Normal": 1.0,
        "HighVol": 0.90,
        "EventRisk": 0.85,
        "LowLiquidity": 0.85,
    }.get(rg, 1.0)

    return float(k * adj)


def _ma_slope_sign(series: pd.Series, lookback: int = 5) -> float:
    """Return slope sign proxy: last - prev(lookback)."""
    try:
        s = series.dropna()
        if len(s) <= lookback:
            return np.nan
        return float(s.iloc[-1] - s.iloc[-1 - lookback])
    except Exception:
        return np.nan


def _ma_respect_count(df: pd.DataFrame, ma_col: str, tol: float, lookback: int = 60) -> int:
    """Count how often price comes within tol of MA and then rejects in next 1–3 bars.
    A lightweight proxy for 'respect'.
    """
    if df.empty or ma_col not in df.columns or 'Close' not in df.columns or pd.isna(tol) or tol <= 0:
        return 0
    d = df.tail(max(lookback + 5, 20)).copy()
    c = d['Close'].astype(float)
    m = d[ma_col].astype(float)
    if c.isna().all() or m.isna().all():
        return 0

    # touch when within tol
    touch = (c - m).abs() <= tol
    cnt = 0
    idxs = list(d.index)
    for i in range(len(d) - 3):
        if not bool(touch.iloc[i]):
            continue
        # rejection: next bars move away at least 0.5*tol
        base = float(c.iloc[i])
        away = False
        for j in (1, 2, 3):
            if i + j >= len(d):
                break
            if abs(float(c.iloc[i + j]) - float(m.iloc[i + j])) >= 0.5 * tol:
                away = True
                break
        if away:
            cnt += 1
    return int(cnt)


def _ma_strength(df: pd.DataFrame, ma_name: str, ma_val: float, close: float, tol: float) -> int:
    """MAStrength: base(MA200=3, MA50=2, MA20=1) + boosts - penalties; clamp 1..5."""
    name = (ma_name or '').upper().strip()
    base = 1
    if name == 'MA200':
        base = 3
    elif name == 'MA50':
        base = 2
    elif name == 'MA20':
        base = 1

    boosts = 0
    penalties = 0

    # slope boost
    if name in df.columns:
        slope = _ma_slope_sign(df[name].astype(float), lookback=5)
        if pd.notna(slope):
            if slope > 0:
                boosts += 1
            elif slope < 0:
                penalties += 1

    # respect count boost
    rc = _ma_respect_count(df, name, tol=tol, lookback=60)
    if rc >= 2:
        boosts += 1

    # distance penalty (too far from price)
    atr14 = _atr14_last(df)
    if pd.notna(atr14) and atr14 > 0 and pd.notna(close) and close > 0 and pd.notna(ma_val):
        dist = abs(ma_val - close)
        if dist > 5 * atr14:
            penalties += 2
        elif dist > 3 * atr14:
            penalties += 1

    score = base + boosts - penalties
    if score < 1:
        score = 1
    if score > 5:
        score = 5
    return int(score)


def _confluence_bonus(level_price: float, fib_strong_prices: List[float], ma_major_prices: List[float], tol: float) -> int:
    """Confluence: +1 if close to one other strong set, +2 if close to both; cap 2."""
    if pd.isna(level_price) or pd.isna(tol) or tol <= 0:
        return 0
    hit_f = any((pd.notna(p) and abs(p - level_price) <= tol) for p in (fib_strong_prices or []))
    hit_m = any((pd.notna(p) and abs(p - level_price) <= tol) for p in (ma_major_prices or []))
    if hit_f and hit_m:
        return 2
    if hit_f or hit_m:
        return 1
    return 0


def _pick_best_anchor(cands: List[Dict[str, Any]], entry: float, atr14: float) -> Optional[Dict[str, Any]]:
    """Pick candidate with highest score; tie-breaker: closer to entry (tradeable)."""
    if not cands:
        return None
    def dist_penalty(price: float) -> int:
        if pd.isna(price) or pd.isna(entry) or pd.isna(atr14) or atr14 <= 0:
            return 0
        d = abs(entry - price)
        if d > 5 * atr14:
            return 2
        if d > 3 * atr14:
            return 1
        return 0

    best = None
    for c in cands:
        price = _safe_float(c.get('price'))
        score = _safe_float(c.get('score'))
        if pd.isna(price) or pd.isna(score):
            continue
        # add distance penalty at selection time
        sc = float(score) - dist_penalty(price)
        c['_score_adj'] = sc
        if best is None:
            best = c
            continue
        if sc > best.get('_score_adj', -1e9):
            best = c
            continue
        if sc == best.get('_score_adj', -1e9):
            # closer wins
            if abs(entry - price) < abs(entry - _safe_float(best.get('price'))):
                best = c

    return best


def build_trade_plan(df: pd.DataFrame, dual_fib: Dict[str, Any]) -> Dict[str, TradeSetup]:
    """Strength-aware Trade Plan.

    Notes:
    - Keeps existing output shape (Breakout/Pullback with single TP)
    - Stop/TP selection uses strength scoring.
    - Fallback TP uses k·R by class/regime (quick inference, non-invasive).
    """
    if df.empty:
        return {}

    last = df.iloc[-1]
    close = _safe_float(last.get('Close'))

    fib_short = (dual_fib or {}).get('auto_short', {}).get('levels', {}) or {}
    fib_long  = (dual_fib or {}).get('fixed_long', {}).get('levels', {}) or {}

    # Build anchors for supports/resistances (existing behavior)
    anchors = _build_anchor_level_map(df, fib_short, fib_long)
    vr = _vol_ratio(df)

    atr14 = _atr14_last(df)
    tol = _tol_price(df, close)

    # Precompute strong fib prices & major MA prices for confluence scoring
    fib_prices_strong = []
    for src in (fib_short, fib_long):
        for k, v in (src or {}).items():
            if _fib_strength_from_key(k) >= 3:
                fv = _safe_float(v)
                if pd.notna(fv):
                    fib_prices_strong.append(float(fv))

    ma_vals = {
        'MA20': _safe_float(last.get('MA20')),
        'MA50': _safe_float(last.get('MA50')),
        'MA200': _safe_float(last.get('MA200')),
    }
    ma_major_prices = [float(ma_vals['MA50'])] if pd.notna(ma_vals.get('MA50')) else []
    if pd.notna(ma_vals.get('MA200')):
        ma_major_prices.append(float(ma_vals['MA200']))

    # k·R profile (used only when fib/MA targets unavailable)
    cclass = _infer_character_class_quick(df)
    regime = _infer_risk_regime_quick(df)

    plans: Dict[str, TradeSetup] = {}

    # ----------------------------
    # 1) BREAKOUT PLAN
    # Entry: same as v9.8 (base resistance * 1.01)
    # Stop: strongest support anchor below entry (Fib/MA/Anchors) - dynamic buffer
    # TP: strongest target above entry (Fib/MA). If none → k·R fallback.
    # ----------------------------
    base_res = np.nan
    base_res_tag = ''
    if pd.notna(close):
        s618 = _safe_float(fib_short.get('61.8'))
        if pd.notna(s618) and s618 >= close:
            base_res = s618
            base_res_tag = 'EntryAnchor=FibS_61.8'
        else:
            k_res, v_res = _nearest_resistance_above(anchors, close)
            base_res = v_res
            base_res_tag = f'EntryAnchor={k_res}' if k_res != 'N/A' else 'EntryAnchor=Fallback_Close'
            if pd.isna(base_res):
                base_res = close

    entry_b = _round_price(base_res * 1.01) if pd.notna(base_res) else np.nan
    buf_b = _buffer_price_dynamic(df, entry_b) if pd.notna(entry_b) else np.nan

    # ---- STOP candidates (below entry)
    stop_cands: List[Dict[str, Any]] = []

    # fib supports below entry
    def _push_fib(src_name: str, lv: Dict[str, Any]):
        for k, v in (lv or {}).items():
            pv = _safe_float(v)
            if pd.isna(pv) or pd.isna(entry_b):
                continue
            if pv >= entry_b:
                continue
            fs = _fib_strength_from_key(k)
            bonus = _confluence_bonus(pv, fib_prices_strong, ma_major_prices, tol)
            stop_cands.append({
                'type': 'Fib',
                'name': f'{src_name}_{k}',
                'price': float(pv),
                'strength': fs,
                'bonus': bonus,
                'score': float(fs + bonus),
            })

    _push_fib('FibS', fib_short)
    _push_fib('FibL', fib_long)

    # MA supports below entry
    for mn, mv in ma_vals.items():
        if pd.notna(mv) and pd.notna(entry_b) and mv < entry_b:
            ms = _ma_strength(df, mn, float(mv), close, tol)
            bonus = _confluence_bonus(float(mv), fib_prices_strong, ma_major_prices, tol)
            stop_cands.append({
                'type': 'MA',
                'name': mn,
                'price': float(mv),
                'strength': ms,
                'bonus': bonus,
                'score': float(ms + bonus),
            })

    # fallback anchors (from existing merged anchors)
    try:
        k_sup, v_sup = _nearest_support_below(anchors, entry_b)
    except Exception:
        k_sup, v_sup = ('N/A', np.nan)
    if pd.notna(v_sup):
        bonus = _confluence_bonus(float(v_sup), fib_prices_strong, ma_major_prices, tol)
        stop_cands.append({
            'type': 'Anchor',
            'name': f'{k_sup}',
            'price': float(v_sup),
            'strength': 2,
            'bonus': bonus,
            'score': float(2 + bonus),
        })

    # ---- Breakout stop constraint (Rule 3): keep tactical stop tradeable
    stop_constraint_tag = "StopConstraint=2.5ATR"
    stop_cands_trade = stop_cands
    try:
        if pd.notna(atr14) and float(atr14) > 0 and pd.notna(entry_b):
            _tmp = []
            for c in stop_cands:
                pv = _safe_float(c.get('price'))
                if pd.isna(pv):
                    continue
                # distance from entry to anchor for LONG stop
                if (float(entry_b) - float(pv)) <= 2.5 * float(atr14):
                    _tmp.append(c)
            if _tmp:
                stop_cands_trade = _tmp
            else:
                stop_constraint_tag = "StopConstraint=Relaxed"
    except Exception:
        stop_constraint_tag = "StopConstraint=Relaxed"

    best_stop = _pick_best_anchor(stop_cands_trade, entry=entry_b, atr14=atr14)
    stop_ref_tag_b = best_stop.get('name') if best_stop else 'N/A'
    stop_ref_val_b = _safe_float(best_stop.get('price')) if best_stop else np.nan

    stop_b = _round_price(stop_ref_val_b - buf_b) if (pd.notna(stop_ref_val_b) and pd.notna(buf_b)) else np.nan

    # ---- TARGET candidates (above entry)
    tgt_cands: List[Dict[str, Any]] = []

    def _push_fib_above(src_name: str, lv: Dict[str, Any]):
        for k, v in (lv or {}).items():
            pv = _safe_float(v)
            if pd.isna(pv) or pd.isna(entry_b):
                continue
            if pv <= entry_b:
                continue
            fs = _fib_strength_from_key(k)
            bonus = _confluence_bonus(pv, fib_prices_strong, ma_major_prices, tol)
            # proximity score for TP1
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = pv - entry_b
                if 1.0 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            tgt_cands.append({
                'type': 'Fib',
                'name': f'{src_name}_{k}',
                'price': float(pv),
                'strength': fs,
                'bonus': bonus,
                'score': float(fs + bonus + prox),
            })

    _push_fib_above('FibS', fib_short)
    _push_fib_above('FibL', fib_long)

    # MA overhead as resistance/TP
    for mn, mv in ma_vals.items():
        if pd.notna(mv) and pd.notna(entry_b) and mv > entry_b:
            ms = _ma_strength(df, mn, float(mv), close, tol)
            bonus = _confluence_bonus(float(mv), fib_prices_strong, ma_major_prices, tol)
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = float(mv) - entry_b
                if 1.0 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            tgt_cands.append({
                'type': 'MA',
                'name': mn,
                'price': float(mv),
                'strength': ms,
                'bonus': bonus,
                'score': float(ms + bonus + prox),
            })

    # ---- TARGET selection (Rule 1–2)
    # TP1: nearest strong "first trouble area" (structure/MA)
    # TP2: extension-biased payoff target (used for gate/validity)
    def _fib_key_num(_k: Any) -> float:
        try:
            return float(str(_k).strip())
        except Exception:
            return float('nan')

    for c in tgt_cands:
        pv = _safe_float(c.get('price'))
        if pd.isna(pv) or pd.isna(entry_b):
            c['score_tp1'] = c.get('score', 0.0)
            c['score_tp2'] = c.get('score', 0.0)
            continue
        d = float(pv) - float(entry_b)
        c['dist'] = d

        # Base score from strength + confluence bonus
        s_base = float(_safe_float(c.get('strength')) or 0.0) + float(_safe_float(c.get('bonus')) or 0.0)

        # Proximity bonus for TP1 (realistic first take-profit)
        prox = 0.0
        if pd.notna(atr14) and float(atr14) > 0:
            if 1.0 * float(atr14) <= d <= 3.0 * float(atr14):
                prox = 1.0

        # Payoff bonus for TP2 (room for breakout payoff)
        payoff = 0.0
        if pd.notna(atr14) and float(atr14) > 0:
            if d >= 2.5 * float(atr14):
                payoff = 1.0

        # Breakout bias: prefer fib extensions (>=127.2) for TP2
        ext_bias = 0.0
        near100_pen = 0.0
        if c.get('type') == 'Fib':
            name = str(c.get('name', ''))
            fib_k = name.split('_')[-1] if '_' in name else name
            fk = _fib_key_num(fib_k)

            is_100 = (pd.notna(fk) and (99.5 <= fk <= 100.5))
            is_ext = (pd.notna(fk) and (fk >= 127.0))
            if is_ext:
                ext_bias = 1.0

            # Penalize 100% if too close to entry (insufficient upside)
            if is_100 and pd.notna(atr14) and float(atr14) > 0 and d < 1.0 * float(atr14):
                near100_pen = -1.5

        # Score for TP1 vs TP2
        c['score_tp1'] = s_base + prox + 0.5 * ext_bias + near100_pen
        c['score_tp2'] = s_base + payoff + 1.0 * ext_bias + near100_pen

    best_tp1 = None
    best_tp2 = None
    try:
        if tgt_cands:
            best_tp1 = max(tgt_cands, key=lambda x: float(_safe_float(x.get('score_tp1')) or 0.0))
            tp1_val_tmp = _safe_float(best_tp1.get('price')) if best_tp1 else np.nan

            if pd.notna(tp1_val_tmp):
                pool2 = [c for c in tgt_cands
                         if pd.notna(_safe_float(c.get('price')))
                         and float(_safe_float(c.get('price'))) > float(tp1_val_tmp) + (float(tol) if pd.notna(tol) else 0.0)]
            else:
                pool2 = list(tgt_cands)

            best_tp2 = max(pool2, key=lambda x: float(_safe_float(x.get('score_tp2')) or 0.0)) if pool2 else None
    except Exception:
        best_tp1 = None
        best_tp2 = None

    tp1_label_b = best_tp1.get('name') if best_tp1 else None
    tp1_val_b = _safe_float(best_tp1.get('price')) if best_tp1 else np.nan

    tp2_label_b = best_tp2.get('name') if best_tp2 else None
    tp2_val_b = _safe_float(best_tp2.get('price')) if best_tp2 else np.nan

    # Breakout shows TP2 by default (Rule 1). TP1 saved in tags.
    tp_label_b = None
    if pd.notna(tp2_val_b):
        tp_b = _round_price(tp2_val_b)
        tp_label_b = tp2_label_b
    elif pd.notna(tp1_val_b):
        tp_b = _round_price(tp1_val_b)
        tp_label_b = tp1_label_b
        tp2_label_b = None
    else:
        # fallback k·R by Class/Regime
        if pd.notna(entry_b) and pd.notna(stop_b) and entry_b > stop_b:
            k = _kr_fallback_mult('breakout', cclass=cclass, regime=regime)
            tp_b = _round_price(entry_b + k * (entry_b - stop_b))
            tp_label_b = f'Fallback_kR({cclass}/{regime},k={round(k,2)})'
            tp1_label_b = None
            tp2_label_b = None
        else:
            tp_b = np.nan
            tp_label_b = None
            tp1_label_b = None
            tp2_label_b = None

    rr_b = _compute_rr(entry_b, stop_b, tp_b)

    tags_b: List[str] = []
    if base_res_tag:
        tags_b.append(base_res_tag)
    if stop_ref_tag_b and stop_ref_tag_b != 'N/A':
        tags_b.append(f'StopRef={stop_ref_tag_b}')
    if pd.notna(vr):
        tags_b.append(f'VolRatio={round(vr,2)}')
    if pd.notna(buf_b):
        tags_b.append('Buffer=Dynamic(ATR/Proxy)')
    # Stop constraint debug tag (Rule 3)
    if 'stop_constraint_tag' in locals():
        tags_b.append(stop_constraint_tag)
    if tp_label_b:
        tags_b.append(f'TP={tp_label_b}')
    if 'tp1_label_b' in locals() and tp1_label_b:
        tags_b.append(f'TP1={tp1_label_b}')
    if 'tp2_label_b' in locals() and tp2_label_b:
        tags_b.append(f'TP2={tp2_label_b}')
    tags_b.append(f'KRProfile={cclass}/{regime}')

    status_b = 'Watch'
    if any(pd.isna([entry_b, stop_b, tp_b, rr_b])) or (entry_b <= stop_b) or (rr_b < 1.2):
        status_b = 'Invalid'
        tags_b.append('Invalid=GeometryOrRR')
    else:
        near_entry = (abs(close - entry_b) / close * 100) <= 1.2 if (pd.notna(close) and close != 0) else False
        vol_ok = (vr >= 1.1) if pd.notna(vr) else True
        if near_entry and vol_ok:
            status_b = 'Active'
            tags_b.append('Trigger=NearEntry')
            if pd.notna(vr) and vr >= 1.1:
                tags_b.append('Trigger=VolumeSupport')

    prob_b = _probability_label_from_facts(df, rr_b, status_b, vr)
    plans['Breakout'] = TradeSetup(
        name='Breakout',
        entry=entry_b, stop=stop_b, tp=tp_b, rr=rr_b,
        probability=prob_b,
        status=status_b,
        reason_tags=tags_b
    )

    # ----------------------------
    # 2) PULLBACK PLAN
    # Entry: strongest nearby support below close (Fib/MA/Anchors)
    # Stop: strongest support below entry (next level down) - dynamic buffer
    # TP: strongest target above entry (Fib/MA). If none → k·R fallback.
    # ----------------------------
    entry_anchor_tag = 'EntryAnchor=Fallback_Close'
    entry_anchor_val = close

    entry_cands: List[Dict[str, Any]] = []

    # fib/ma supports below close
    for src_name, lv in [('FibS', fib_short), ('FibL', fib_long)]:
        for k, v in (lv or {}).items():
            pv = _safe_float(v)
            if pd.isna(pv) or pd.isna(close):
                continue
            if pv >= close:
                continue
            fs = _fib_strength_from_key(k)
            bonus = _confluence_bonus(pv, fib_prices_strong, ma_major_prices, tol)
            # proximity bonus: closer to close preferred
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = close - pv
                if 0.5 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            entry_cands.append({'type': 'Fib', 'name': f'{src_name}_{k}', 'price': float(pv), 'score': float(fs + bonus + prox)})

    for mn, mv in ma_vals.items():
        if pd.notna(mv) and pd.notna(close) and mv < close:
            ms = _ma_strength(df, mn, float(mv), close, tol)
            bonus = _confluence_bonus(float(mv), fib_prices_strong, ma_major_prices, tol)
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = close - float(mv)
                if 0.5 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            entry_cands.append({'type': 'MA', 'name': mn, 'price': float(mv), 'score': float(ms + bonus + prox)})

    # anchor fallback below close
    k_sup_close, v_sup_close = _nearest_support_below(anchors, close)
    if pd.notna(v_sup_close):
        bonus = _confluence_bonus(float(v_sup_close), fib_prices_strong, ma_major_prices, tol)
        entry_cands.append({'type': 'Anchor', 'name': f'{k_sup_close}', 'price': float(v_sup_close), 'score': float(2 + bonus)})

    best_entry = _pick_best_anchor(entry_cands, entry=close, atr14=atr14)
    if best_entry and pd.notna(best_entry.get('price')):
        entry_anchor_val = float(best_entry.get('price'))
        entry_anchor_tag = f"EntryAnchor={best_entry.get('name')}"

    entry_p = _round_price(entry_anchor_val) if pd.notna(entry_anchor_val) else np.nan
    buf_p = _buffer_price_dynamic(df, entry_p) if pd.notna(entry_p) else np.nan

    # Stop candidates below entry, excluding the chosen entry anchor price
    stop_cands_p: List[Dict[str, Any]] = []
    ex = float(entry_anchor_val) if pd.notna(entry_anchor_val) else None

    def _push_fib_below_ex(src_name: str, lv: Dict[str, Any]):
        for k, v in (lv or {}).items():
            pv = _safe_float(v)
            if pd.isna(pv) or pd.isna(entry_p):
                continue
            if pv >= entry_p:
                continue
            if ex is not None and abs(pv - ex) <= 1e-9:
                continue
            fs = _fib_strength_from_key(k)
            bonus = _confluence_bonus(pv, fib_prices_strong, ma_major_prices, tol)
            stop_cands_p.append({'type': 'Fib', 'name': f'{src_name}_{k}', 'price': float(pv), 'score': float(fs + bonus)})

    _push_fib_below_ex('FibS', fib_short)
    _push_fib_below_ex('FibL', fib_long)

    for mn, mv in ma_vals.items():
        if pd.notna(mv) and pd.notna(entry_p) and mv < entry_p:
            if ex is not None and abs(float(mv) - ex) <= 1e-9:
                continue
            ms = _ma_strength(df, mn, float(mv), close, tol)
            bonus = _confluence_bonus(float(mv), fib_prices_strong, ma_major_prices, tol)
            stop_cands_p.append({'type': 'MA', 'name': mn, 'price': float(mv), 'score': float(ms + bonus)})

    # fallback anchors below entry (exclude exact match)
    k_sup_p, v_sup_p = _nearest_support_below(anchors, entry_p)
    if pd.notna(v_sup_p) and (ex is None or abs(float(v_sup_p) - ex) > 1e-9):
        bonus = _confluence_bonus(float(v_sup_p), fib_prices_strong, ma_major_prices, tol)
        stop_cands_p.append({'type': 'Anchor', 'name': f'{k_sup_p}', 'price': float(v_sup_p), 'score': float(2 + bonus)})

    best_stop_p = _pick_best_anchor(stop_cands_p, entry=entry_p, atr14=atr14)
    stop_ref_tag_p = best_stop_p.get('name') if best_stop_p else 'N/A'
    stop_ref_val_p = _safe_float(best_stop_p.get('price')) if best_stop_p else np.nan

    stop_p = _round_price(stop_ref_val_p - buf_p) if (pd.notna(stop_ref_val_p) and pd.notna(buf_p)) else np.nan

    # Targets above entry
    tgt_cands_p: List[Dict[str, Any]] = []

    def _push_fib_above_p(src_name: str, lv: Dict[str, Any]):
        for k, v in (lv or {}).items():
            pv = _safe_float(v)
            if pd.isna(pv) or pd.isna(entry_p):
                continue
            if pv <= entry_p:
                continue
            fs = _fib_strength_from_key(k)
            bonus = _confluence_bonus(pv, fib_prices_strong, ma_major_prices, tol)
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = pv - entry_p
                if 1.0 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            tgt_cands_p.append({'type': 'Fib', 'name': f'{src_name}_{k}', 'price': float(pv), 'score': float(fs + bonus + prox)})

    _push_fib_above_p('FibS', fib_short)
    _push_fib_above_p('FibL', fib_long)

    for mn, mv in ma_vals.items():
        if pd.notna(mv) and pd.notna(entry_p) and mv > entry_p:
            ms = _ma_strength(df, mn, float(mv), close, tol)
            bonus = _confluence_bonus(float(mv), fib_prices_strong, ma_major_prices, tol)
            prox = 0
            if pd.notna(atr14) and atr14 > 0:
                d = float(mv) - entry_p
                if 1.0 * atr14 <= d <= 3.0 * atr14:
                    prox = 1
            tgt_cands_p.append({'type': 'MA', 'name': mn, 'price': float(mv), 'score': float(ms + bonus + prox)})

    best_tp_p = _pick_best_anchor(tgt_cands_p, entry=entry_p, atr14=atr14)
    tp_label_p = best_tp_p.get('name') if best_tp_p else None
    tp_val_p = _safe_float(best_tp_p.get('price')) if best_tp_p else np.nan

    if pd.notna(tp_val_p):
        tp_p = _round_price(tp_val_p)
    else:
        if pd.notna(entry_p) and pd.notna(stop_p) and entry_p > stop_p:
            k = _kr_fallback_mult('pullback', cclass=cclass, regime=regime)
            tp_p = _round_price(entry_p + k * (entry_p - stop_p))
            tp_label_p = f'Fallback_kR({cclass}/{regime},k={round(k,2)})'
        else:
            tp_p = np.nan

    rr_p = _compute_rr(entry_p, stop_p, tp_p)

    tags_p: List[str] = [entry_anchor_tag]
    if stop_ref_tag_p and stop_ref_tag_p != 'N/A':
        tags_p.append(f'StopRef={stop_ref_tag_p}')
    if pd.notna(vr):
        tags_p.append(f'VolRatio={round(vr,2)}')
    if pd.notna(buf_p):
        tags_p.append('Buffer=Dynamic(ATR/Proxy)')
    if tp_label_p:
        tags_p.append(f'TP={tp_label_p}')
    tags_p.append(f'KRProfile={cclass}/{regime}')

    status_p = 'Watch'
    if any(pd.isna([entry_p, stop_p, tp_p, rr_p])) or (entry_p <= stop_p) or (rr_p < 1.2):
        status_p = 'Invalid'
        tags_p.append('Invalid=GeometryOrRR')
    else:
        near_entry = (abs(close - entry_p) / close * 100) <= 1.2 if (pd.notna(close) and close != 0) else False
        if near_entry:
            status_p = 'Active'
            tags_p.append('Trigger=NearEntry')

    prob_p = _probability_label_from_facts(df, rr_p, status_p, vr)
    plans['Pullback'] = TradeSetup(
        name='Pullback',
        entry=entry_p, stop=stop_p, tp=tp_p, rr=rr_p,
        probability=prob_p,
        status=status_p,
        reason_tags=tags_p
    )

    return plans
# ============================================================
# 9. SCENARIO CLASSIFICATION
# ============================================================
def classify_scenario(last: pd.Series) -> str:
    c, ma20, ma50, ma200 = last["Close"], last["MA20"], last["MA50"], last["MA200"]
    if all(pd.notna([c, ma20, ma50, ma200])):
        if ma20 > ma50 > ma200 and c > ma20: return "Uptrend – Breakout Confirmation"
        elif c > ma200 and ma20 > ma200: return "Uptrend – Pullback Phase"
        elif c < ma200 and ma50 < ma200: return "Downtrend – Weak Phase"
    return "Neutral / Sideways"


# --- Scenario 12 Spec Table (Canonical) ---
# This table is the single source of truth for Scenario12 names and intended plan semantics.
# It must remain "NOW" only (Current Status), and must NOT leak into Stock DNA.
SCENARIO12_SPECS: Dict[int, Dict[str, Any]] = {
    1:  {"Name": "S1 – Uptrend + Momentum Aligned",   "TrendRegime": "Up",      "MomentumRegime": "Aligned",
         "DefaultPlan": "Pullback", "GateHint": "Prefer pullbacks; breakout only with Volume PASS + RR PASS.",
         "Notes": ["Trend up + momentum aligned", "Best risk-adjusted entries on pullbacks/holds above MA50."]},
    2:  {"Name": "S2 – Uptrend + Momentum Mixed",     "TrendRegime": "Up",      "MomentumRegime": "Mixed",
         "DefaultPlan": "Pullback", "GateHint": "Wait for momentum confirmation; avoid chasing highs.",
         "Notes": ["Trend ok but momentum not clean", "Trade smaller; require triggers."]},
    3:  {"Name": "S3 – Uptrend + Momentum Counter",   "TrendRegime": "Up",      "MomentumRegime": "Counter",
         "DefaultPlan": "Pullback", "GateHint": "Treat as pullback correction; wait for MACD/RSI re-alignment.",
         "Notes": ["Trend up but momentum counter", "Higher whipsaw risk."]},
    4:  {"Name": "S4 – Uptrend + RSI 70+",            "TrendRegime": "Up",      "MomentumRegime": "RSI_70Plus",
         "DefaultPlan": "Breakout", "GateHint": "Overheat risk: only breakout with strong volume; otherwise wait.",
         "Notes": ["Late-stage push possible", "Do not FOMO near resistance."]},

    5:  {"Name": "S5 – Range + Momentum Aligned",     "TrendRegime": "Neutral", "MomentumRegime": "Aligned",
         "DefaultPlan": "Range", "GateHint": "Range trade only near edges; confirm levels.",
         "Notes": ["Momentum aligned inside range", "Avoid entries mid-range."]},
    6:  {"Name": "S6 – Range + Balanced/Mixed",       "TrendRegime": "Neutral", "MomentumRegime": "Mixed",
         "DefaultPlan": "Range", "GateHint": "No edge unless clear support/resistance + RR gate.",
         "Notes": ["Balanced/indecisive", "Prefer WAIT unless strong location."]},
    7:  {"Name": "S7 – Range + Momentum Counter",     "TrendRegime": "Neutral", "MomentumRegime": "Counter",
         "DefaultPlan": "Range", "GateHint": "High noise: require strict triggers and small size.",
         "Notes": ["Counter signals in range", "False breaks likely."]},
    8:  {"Name": "S8 – Range + RSI 70+",              "TrendRegime": "Neutral", "MomentumRegime": "RSI_70Plus",
         "DefaultPlan": "Breakout", "GateHint": "May be breakout attempt; need follow-through + volume.",
         "Notes": ["Potential transition from range to trend", "Demand confirmation."]},

    9:  {"Name": "S9 – Downtrend + Momentum Aligned", "TrendRegime": "Down",    "MomentumRegime": "Aligned",
         "DefaultPlan": "Avoid", "GateHint": "Avoid longs; only tactical trades with tight risk.",
         "Notes": ["Downtrend confirmed", "Capital preservation first."]},
    10: {"Name": "S10 – Downtrend + Momentum Mixed",  "TrendRegime": "Down",    "MomentumRegime": "Mixed",
         "DefaultPlan": "Avoid", "GateHint": "Wait for base/structure change; do not bottom-fish.",
         "Notes": ["Weak structure", "Watchlist only unless reversal signals."]},
    11: {"Name": "S11 – Downtrend + Momentum Counter","TrendRegime": "Down",    "MomentumRegime": "Counter",
         "DefaultPlan": "ReversalWatch", "GateHint": "Counter-rally risk: treat as rebound until structure flips.",
         "Notes": ["Countertrend bounce possible", "Require MA structure repair."]},
    12: {"Name": "S12 – Downtrend + RSI 70+",         "TrendRegime": "Down",    "MomentumRegime": "RSI_70Plus",
         "DefaultPlan": "ReversalWatch", "GateHint": "Rare; often short squeeze. Do not assume trend reversal.",
         "Notes": ["Short squeeze / spike risk", "Wait for confirmation before bias flip."]},
}

# --- STEP 6B (v5.4): SCENARIO 12 NEUTRAL ("Extended" -> "RSI_70Plus") ---
def classify_scenario12(last: pd.Series) -> Dict[str, Any]:
    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))
    
    rules_hit = []
    
    # 1. Trend
    trend = "Neutral"
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if c >= ma50 and ma50 >= ma200:
            trend = "Up"
            rules_hit.append("Trend=Up (Close>=MA50>=MA200)")
        elif c < ma50 and ma50 < ma200:
            trend = "Down"
            rules_hit.append("Trend=Down (Close<MA50<MA200)")
        else:
            trend = "Neutral"
            rules_hit.append("Trend=Neutral (mixed MA structure)")
    else:
        rules_hit.append("Trend=N/A (missing MA50/MA200/Close)")
        
    # 2. Momentum (Alignment)
    mom = "Mixed"
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        rsi_up = (rsi >= 55)
        rsi_down = (rsi <= 45)
        rsi_pos_ext = (rsi >= 70)
        macd_up = (macd_v >= sig)
        macd_down = (macd_v < sig)
        
        if rsi_pos_ext:
            mom = "RSI_70Plus" # 6B Change
            rules_hit.append("Momentum=RSI_70Plus (RSI>=70)")
        elif rsi_up and macd_up:
            mom = "Aligned"
            rules_hit.append("Momentum=Aligned_Up (RSI>=55 & MACD>=Sig)")
        elif rsi_down and macd_down:
            mom = "Aligned" # Aligned Down
            rules_hit.append("Momentum=Aligned_Down (RSI<=45 & MACD<Sig)")
        elif rsi_up and macd_down:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bull but MACD Bear)")
        elif rsi_down and macd_up:
            mom = "Counter"
            rules_hit.append("Momentum=Counter (RSI Bear but MACD Bull)")
        else:
            mom = "Mixed"
            rules_hit.append("Momentum=Mixed (Between zones)")
    else:
        rules_hit.append("Momentum=N/A")
        
    # 3. Volume
    vol_reg = "N/A"
    if pd.notna(vol) and pd.notna(avg_vol):
        vol_reg = "High" if vol > avg_vol else "Low"
        rules_hit.append(f"Volume={vol_reg} (Vol {'>' if vol>avg_vol else '<='} Avg20Vol)")
    else:
        rules_hit.append("Volume=N/A (missing Volume/Avg20Vol)")
        
    trend_order = {"Up": 0, "Neutral": 1, "Down": 2}
    mom_order = {"Aligned": 0, "Mixed": 1, "Counter": 2, "RSI_70Plus": 3}
    code = trend_order.get(trend, 1) * 4 + mom_order.get(mom, 1) + 1
    
    name_map = {
        ("Up", "Aligned"):   "S1 – Uptrend + Momentum Aligned",
        ("Up", "Mixed"):     "S2 – Uptrend + Momentum Mixed",
        ("Up", "Counter"):   "S3 – Uptrend + Momentum Counter",
        ("Up", "RSI_70Plus"):  "S4 – Uptrend + RSI 70+",
        ("Neutral", "Aligned"):  "S5 – Range + Momentum Aligned",
        ("Neutral", "Mixed"):    "S6 – Range + Balanced/Mixed",
        ("Neutral", "Counter"):  "S7 – Range + Momentum Counter",
        ("Neutral", "RSI_70Plus"): "S8 – Range + RSI 70+",
        ("Down", "Aligned"):  "S9 – Downtrend + Momentum Aligned",
        ("Down", "Mixed"):    "S10 – Downtrend + Momentum Mixed",
        ("Down", "Counter"):  "S11 – Downtrend + Momentum Counter",
        ("Down", "RSI_70Plus"): "S12 – Downtrend + RSI 70+",
    }
    
    return {
        "Code": int(code),
        "TrendRegime": trend,
        "MomentumRegime": mom,
        "VolumeRegime": vol_reg,
        "Name": (SCENARIO12_SPECS.get(int(code), {}) or {}).get("Name") or name_map.get((trend, mom), "Scenario – N/A"),
        "Spec": (SCENARIO12_SPECS.get(int(code), {}) or {}),
        "RulesHit": rules_hit,
        "Flags": {
            "TrendUp": (trend=="Up"),
            "TrendDown": (trend=="Down"),
            "MomAligned": (mom=="Aligned"),
            "Mom70Plus": (mom=="RSI_70Plus"),
            "VolHigh": (vol_reg=="High")
        }
    }

# --- STEP 5B: MASTER SCORE (FACT ONLY) ---
# --- STEP 5B: MASTER SCORE (FACT ONLY | FUNDAMENTAL-FREE) ---

def compute_master_score(last: pd.Series, dual_fib: Dict[str, Any], trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Fundamental (Target/Upside/Recommendation) MUST NOT affect any score computation.
    - This MasterScore is 100% technical + tradeplan RR only.
    """

    c = _safe_float(last.get("Close"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))
    rsi = _safe_float(last.get("RSI"))
    macd_v = _safe_float(last.get("MACD"))
    sig = _safe_float(last.get("MACDSignal"))
    vol = _safe_float(last.get("Volume"))
    avg_vol = _safe_float(last.get("Avg20Vol"))

    comps = {}
    notes = []  # neutral condition tags only

    # 1) Trend (pure MA structure)
    trend = 0.0
    if pd.notna(c) and pd.notna(ma50) and pd.notna(ma200):
        if (c >= ma50) and (ma50 >= ma200):
            trend = 2.0
            notes.append("TrendTag=Structure_Up_Strong")
        elif (c >= ma200):
            trend = 1.2
            notes.append("TrendTag=Structure_Up_Moderate")
        else:
            trend = 0.4
            notes.append("TrendTag=Structure_Down_Weak")
    comps["Trend"] = trend

    # 2) Momentum (RSI + MACD relation only)
    mom = 0.0
    if pd.notna(rsi) and pd.notna(macd_v) and pd.notna(sig):
        if (rsi >= 55) and (macd_v >= sig):
            mom = 2.0
            notes.append("MomTag=Aligned_Bullish")
        elif (rsi <= 45) and (macd_v < sig):
            mom = 0.4
            notes.append("MomTag=Aligned_Bearish")
        else:
            mom = 1.1
            notes.append("MomTag=Mixed")
    comps["Momentum"] = mom

    # 3) Volume (relative to Avg20Vol only)
    vcomp = 0.0
    if pd.notna(vol) and pd.notna(avg_vol) and avg_vol != 0:
        if vol > avg_vol:
            vcomp = 1.6
            notes.append("VolTag=Above_Avg")
        else:
            vcomp = 0.9
            notes.append("VolTag=Below_Avg")
    comps["Volume"] = vcomp

    # 4) Fibonacci (position vs key bands, no fundamental)
    fibc = 0.0
    try:
        s_lv = (dual_fib or {}).get("auto_short", {}).get("levels", {}) or {}
        l_lv = (dual_fib or {}).get("fixed_long", {}).get("levels", {}) or {}
        s_618 = _safe_float(s_lv.get("61.8"))
        s_382 = _safe_float(s_lv.get("38.2"))
        l_618 = _safe_float(l_lv.get("61.8"))
        l_382 = _safe_float(l_lv.get("38.2"))

        if pd.notna(c) and pd.notna(s_618) and pd.notna(s_382):
            if c >= s_618:
                fibc += 1.2
            elif c >= s_382:
                fibc += 0.8
            else:
                fibc += 0.4

        if pd.notna(c) and pd.notna(l_618) and pd.notna(l_382):
            if c >= l_618:
                fibc += 0.8
            elif c >= l_382:
                fibc += 0.5
            else:
                fibc += 0.2
    except:
        fibc = 0.0
    comps["Fibonacci"] = fibc

    # 5) RR Quality (from trade plans only)
    best_rr = np.nan
    if trade_plans:
        rrs = [s.rr for s in trade_plans.values() if pd.notna(getattr(s, "rr", np.nan))]
        best_rr = max(rrs) if rrs else np.nan

    rrcomp = 0.0
    if pd.notna(best_rr):
        if best_rr >= 4.0:
            rrcomp = 2.0
            notes.append("RRTag=Excellent_Gt4")
        elif best_rr >= 3.0:
            rrcomp = 1.5
            notes.append("RRTag=Good_Gt3")
        else:
            rrcomp = 1.0
            notes.append("RRTag=Normal_Lt3")
    comps["RRQuality"] = rrcomp

    total = float(sum(comps.values()))

    return {
        "Components": comps,
        "Total": round(total, 2),
        "Tier": "N/A",            # Unlocked
        "PositionSizing": "N/A",  # Unlocked
        "Notes": notes
    }
# ============================================================
# 9D. RISK–REWARD SIMULATION PACK
# ============================================================
def build_rr_sim(trade_plans: Dict[str, TradeSetup]) -> Dict[str, Any]:
    rows = []
    best_rr = np.nan
    for k, s in (trade_plans or {}).items():
        status = getattr(s, "status", "Watch") or "Watch"
        if status == "Invalid": continue
        entry = _safe_float(s.entry)
        stop = _safe_float(s.stop)
        tp = _safe_float(s.tp)
        rr = _safe_float(s.rr)
        risk_pct = ((entry - stop) / entry * 100) if (pd.notna(entry) and pd.notna(stop) and entry != 0) else np.nan
        reward_pct = ((tp - entry) / entry * 100) if (pd.notna(tp) and pd.notna(entry) and entry != 0) else np.nan
        rows.append({
            "Setup": k, "Entry": entry, "Stop": stop, "TP": tp, "RR": rr,
            "RiskPct": risk_pct, "RewardPct": reward_pct, "Confidence (Tech)": s.probability,
            "Status": status, "ReasonTags": list(getattr(s, "reason_tags", []) or [])
        })
        if pd.notna(rr):
            best_rr = rr if pd.isna(best_rr) else max(best_rr, rr)
    return {"Setups": rows, "BestRR": best_rr if pd.notna(best_rr) else np.nan}

# ============================================================
# 10. MAIN ANALYSIS FUNCTION
# ============================================================

# ============================================================
# [MISSING FUNCTIONS] CÁC HÀM BỊ THIẾU (v5.9.1 hotfix)
# ============================================================

def compute_game_stats(last: pd.Series, master: Dict[str, Any], ps: Dict[str, Any], tech: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chuyển đổi các chỉ số kỹ thuật thô thành Game Stats (thang 10).
    """
    _ = _safe_float(last.get("Close"))  # giữ lại để mở rộng, không dùng trực tiếp ở v5.9.1
    slope = _safe_float(tech.get("MA", {}).get("SlopeMA50Value"))
    rsi = _safe_float(last.get("RSI"))
    vol_ratio = _safe_float(tech.get("Volume", {}).get("Ratio"))
    master_score = _safe_float(master.get("Total"), 0)

    rr = _safe_float(ps.get("RR"), 0)
    risk_pct = _safe_float(ps.get("RiskPct"), 0)

    # [ATK]
    atk = master_score
    if slope > 0:
        atk += 0.5
    atk = min(10.0, atk)

    # [SPD]
    spd = 5.0
    if rsi > 50:
        spd += 2.0
    if rsi > 70:
        spd += 1.0
    if vol_ratio > 1.2:
        spd += 2.0
    if vol_ratio < 0.8:
        spd -= 1.0
    spd = min(10.0, max(0.0, spd))

    # [DEF]
    def_stat = 5.0
    if risk_pct > 0 and risk_pct < 5.0:
        def_stat += 3.0
    elif risk_pct < 8.0:
        def_stat += 1.0
    else:
        def_stat -= 2.0
    def_stat = min(10.0, max(0.0, def_stat))

    # [CRT]
    if rr >= 3.0:
        crt = 9.5
    elif rr >= 2.0:
        crt = 7.5
    elif rr >= 1.5:
        crt = 5.0
    else:
        crt = 2.0

    # [HP]
    hp = 5.0
    avg_vol = _safe_float(last.get("Avg20Vol"))
    if avg_vol > 1_000_000:
        hp += 3.0
    elif avg_vol > 500_000:
        hp += 1.0
    elif avg_vol < 100_000:
        hp -= 2.0
    if vol_ratio > 1.0:
        hp += 1.0
    hp = min(10.0, max(0.0, hp))

    stats = {"ATK": atk, "SPD": spd, "DEF": def_stat, "CRT": crt, "HP": hp}

    if atk >= 8.0 and spd >= 7.0:
        archetype = "Assassin (Sát thủ)"
    elif def_stat >= 8.0 and hp >= 7.0:
        archetype = "Tanker (Đỡ đòn)"
    elif crt >= 8.0 and risk_pct < 5.0:
        archetype = "Sniper (Thiện xạ)"
    elif atk >= 6.0 and def_stat >= 6.0:
        archetype = "Warrior (Đấu sĩ)"
    elif rsi < 30:
        archetype = "Necromancer (Bắt đáy)"
    else:
        archetype = "Villager"

    return {
        "Stats": stats,
        "Class": archetype,
        "AvgScore": round((atk + spd + def_stat + crt + hp) / 5, 1),
    }


def analyze_ticker(ticker: str) -> Dict[str, Any]:
    try:
        df_all = load_price_vol(PRICE_VOL_PATH)
    except DataError as e:
        return {"Error": str(e)}
    df = df_all[df_all["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty: return {"Error": f"Không tìm thấy mã {ticker}"}
    
    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["MA200"] = sma(df["Close"], 200)
    df["Avg20Vol"] = sma(df["Volume"], 20)
    df["RSI"] = rsi_wilder(df["Close"], 14)
    m, s, h = macd(df["Close"], 12, 26, 9)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = m, s, h
    
    dual_fib = compute_dual_fibonacci_auto(df, long_window=250)
    last = df.iloc[-1]
    
    # --- Step 5 Context ---
    fib_ctx = compute_fibonacci_context_pack(last, dual_fib)
    
    conviction = compute_conviction(last)
    conviction_pack = compute_conviction_pack(last) # Step 4B/5B Add
    
    scenario = classify_scenario(last)
    trade_plans = build_trade_plan(df, dual_fib)
    
    try:
        hsc = load_hsc_targets(HSC_TARGET_PATH)
    except DataError:
        hsc = pd.DataFrame()
    fund = hsc[hsc["Ticker"].str.upper() == ticker.upper()]
    fund_row = fund.iloc[0].to_dict() if not fund.empty else {}
    close = float(last["Close"]) if pd.notna(last["Close"]) else np.nan
    target_vnd = _safe_float(fund_row.get("Target"), np.nan)
    close_for_calc = close
    target_for_calc = target_vnd
    if pd.notna(close) and pd.notna(target_vnd):
        if (close < 500) and (target_vnd > 1000): target_for_calc = target_vnd / 1000.0
        elif (close > 1000) and (target_vnd < 500): target_for_calc = target_vnd * 1000.0
    upside_pct = ((target_for_calc - close_for_calc) / close_for_calc * 100) if (pd.notna(target_for_calc) and pd.notna(close_for_calc) and close_for_calc != 0) else np.nan
    fund_row["Target"] = target_vnd
    fund_row["UpsidePct"] = upside_pct
    fund_row["TargetK"] = (target_vnd / 1000.0) if pd.notna(target_vnd) else np.nan
    
    # --- Step 2B+6B Scenario Neutral ---
    scenario12 = classify_scenario12(last)
    
    rrsim = build_rr_sim(trade_plans)
    
    # --- Step 1B/5B Master Score (Fact Only) ---
    master = compute_master_score(last, dual_fib, trade_plans)
    
    ma_feat = compute_ma_features(df)
    rsi_feat = compute_rsi_features(df)
    macd_feat = compute_macd_features(df)
    vol_feat = compute_volume_features(df)
    
    # --- Step 3B+6B Bias (Fact Only) ---
    bias_feat = compute_rsi_macd_bias_features(rsi_feat, macd_feat)
    
    # --- Step 6 Price Action ---
    pa_feat = compute_price_action_features(df, fib_ctx=fib_ctx, vol_feat=vol_feat, tol_pct=0.8)
    
    # --- STEP 7B (New Context Features) ---
    rsi_ctx = compute_rsi_context_features(df)
    vol_ctx = compute_volume_context_features(df)
    lvl_ctx = compute_level_context_features(last, dual_fib)

    # --- STEP 7C: Structure Quality (Support/Resistance Quality-Aware) ---
    struct_q = compute_structure_quality_pack(df, last, dual_fib=dual_fib, fib_ctx=fib_ctx)

    market_ctx = compute_market_context(df_all)
    stock_chg = np.nan
    if len(df) >= 2:
        stock_chg = _pct_change(_safe_float(df.iloc[-1].get("Close")), _safe_float(df.iloc[-2].get("Close")))
    mkt_chg = _safe_float(market_ctx.get("VNINDEX", {}).get("ChangePct"))
    rel = "N/A"
    if pd.notna(stock_chg) and pd.notna(mkt_chg):
        if stock_chg > mkt_chg + 0.3: rel = "Stronger"
        elif stock_chg < mkt_chg - 0.3: rel = "Weaker"
        else: rel = "InLine"
        
    analysis_pack = {
        "_schema_version": "1.0",
        "Ticker": ticker.upper(),
        "Last": {
            "Close": _safe_float(last.get("Close")),
            "MA20": _safe_float(last.get("MA20")),
            "MA50": _safe_float(last.get("MA50")),
            "MA200": _safe_float(last.get("MA200")),
            "RSI": _safe_float(last.get("RSI")),
            "MACD": _safe_float(last.get("MACD")),
            "MACDSignal": _safe_float(last.get("MACDSignal")),
            "MACDHist": _safe_float(last.get("MACDHist")),
            "Volume": _safe_float(last.get("Volume")),
            "Avg20Vol": _safe_float(last.get("Avg20Vol")),
        },
        "ScenarioBase": scenario,
        "Scenario12": scenario12,
        "Conviction": conviction,
        "ConvictionPack": conviction_pack, # Step 4B/5B
        "Market": {
            "VNINDEX": market_ctx.get("VNINDEX", {}),
            "VN30": market_ctx.get("VN30", {}),
            "RelativeStrengthVsVNINDEX": rel,
        },
        "Fibonacci": {
            "ShortWindow": dual_fib.get("short_window"),
            "LongWindow": dual_fib.get("long_window"),
            "Short": dual_fib.get("auto_short", {}),
            "Long": dual_fib.get("fixed_long", {}),
            "AltShort": dual_fib.get("alt_short", {}),
            "SelectionReason": dual_fib.get("selection_reason", "N/A"),
            "Context": fib_ctx
        },
        "Fundamental": {
            "Recommendation": fund_row.get("Recommendation", "N/A") if fund_row else "N/A",
            "TargetVND": target_vnd,
            "TargetK": fund_row.get("TargetK", np.nan),
            "UpsidePct": upside_pct
        },
        "TradePlans": [
            {
                "Name": k,
                "Entry": _safe_float(v.entry),
                "Stop": _safe_float(v.stop),
                "TP": _safe_float(v.tp),
                "RR": _safe_float(v.rr),
                "Confidence (Tech)": v.probability,
                "Status": getattr(v, "status", "Watch"),
                "ReasonTags": list(getattr(v, "reason_tags", []) or [])
            } for k, v in trade_plans.items()
        ],
        "RRSim": rrsim,
        "MasterScore": master,
        "ProTech": {
            "MA": ma_feat,
            "RSI": rsi_feat,
            "MACD": macd_feat,
            "Volume": vol_feat,
            "Bias": bias_feat,
            "PriceAction": pa_feat,
            # 7B: New Context Packs
            "RSIContext": rsi_ctx,
            "VolumeContext": vol_ctx,
            "LevelContext": lvl_ctx
        },
        "Market": {
            "VNINDEX": market_ctx.get("VNINDEX", {}),
            "VN30": market_ctx.get("VN30", {}),
            "StockChangePct": stock_chg,
            "RelativeStrengthVsVNINDEX": rel
        }
    }
    
    # Primary Picker (moved to module for maintainability)
    try:
        from inception.core.primary_setup import pick_primary_setup_v3
        primary = pick_primary_setup_v3(rrsim)
    except Exception:
        # Safe fallback (rare)
        primary = {"Name": "N/A", "Status": "N/A", "Entry": np.nan, "Stop": np.nan, "TP1": np.nan, "TP2": np.nan, "RiskPct": np.nan, "RewardPct": np.nan, "RR": np.nan, "Confidence (Tech)": "N/A", "ReasonTags": []}
    # Attach StructureQuality to AnalysisPack (single source of truth for flags/gates/plan anchoring)
    try:
        analysis_pack["StructureQuality"] = sanitize_pack(struct_q) if isinstance(struct_q, dict) else {}
    except Exception:
        analysis_pack["StructureQuality"] = struct_q if isinstance(struct_q, dict) else {}

    
    # Phase 5: prevent NaN leakage in PrimarySetup
    primary = sanitize_pack(primary)
    analysis_pack["PrimarySetup"] = primary
    # Step 8: normalize AnalysisPack contract (fail-safe, prevents type drift)
    try:
        from inception.core.contracts import normalize_analysis_pack
        analysis_pack = normalize_analysis_pack(analysis_pack)
    except Exception:
        pass

    # ------------------------------------------------------------
    # Step 11: attach TradePlanPack + DecisionPack into AnalysisPack
    # Purpose: if plan thiếu Stop/Entry zone, hệ thống tự hạ state và
    # Decision Layer/Report A–D giải thích rõ.
    # ------------------------------------------------------------
    try:
        from inception.core.tradeplan_pack import compute_trade_plan_pack_v1
        from inception.core.decision_pack import compute_decision_pack_v1
        tpp = compute_trade_plan_pack_v1(analysis_pack, character_ctx={})
        analysis_pack["TradePlanPack"] = tpp
        analysis_pack["DecisionPack"] = compute_decision_pack_v1(analysis_pack, tpp)
    except Exception:
        # Never fail analysis due to optional packs
        pass


    last_dict = {k: _as_scalar(v) for k, v in (last.to_dict() if hasattr(last, "to_dict") else dict(last)).items()}
    return {
        "Ticker": ticker.upper(),
        "Last": last_dict,
        "Scenario": scenario,
        "Conviction": conviction,
        "DualFibo": dual_fib,
        "TradePlans": trade_plans,
        "Fundamental": fund_row,
        "Scenario12": scenario12,
        "MasterScore": master,
        "RRSim": rrsim,
        "AnalysisPack": analysis_pack,
        "_DF": df
    }


# ============================================================
# 10B. GAME CHARACTER MODULE (PYTHON-ONLY, NON-BREAKING)
# ============================================================
from typing import Tuple

# ============================================================
# 11B. UI HELPERS (PRESENTATION ONLY)
# ============================================================
def run_module_contract_smoke_test(sample_tickers: List[str]) -> List[Tuple[str, str]]:
    """Quick regression test: ensures AnalysisPack exists and module payloads are JSON-safe."""
    errors: List[Tuple[str, str]] = []
    for t in sample_tickers:
        try:
            result = analyze_ticker(t)
        except Exception as e:
            errors.append((t, f"analyze_ticker crash: {e}"))
            continue
        if not isinstance(result, dict):
            errors.append((t, "result is not dict"))
            continue
        ap = result.get("AnalysisPack", {})
        if not isinstance(ap, dict) or not ap:
            errors.append((t, "AnalysisPack empty or not dict"))
        mods = result.get("Modules", {})
        if isinstance(mods, dict):
            for mname, mpayload in mods.items():
                if not isinstance(mpayload, dict):
                    errors.append((t, f"Module {mname} not dict"))
                    continue
                for k, v in mpayload.items():
                    if isinstance(v, (pd.Series, pd.DataFrame, pd.Index)):
                        errors.append((t, f"Module {mname} field {k} is pandas object"))
        else:
            errors.append((t, "Modules not dict"))
    return errors





def main():
    st.set_page_config(page_title=APP_TITLE,
                       layout="wide",
                       page_icon="🟣")

    st.markdown("""
    <style>
        body {
            background-color: #FFFFFF;
            color: #0F172A;
            font-family: 'Segoe UI', sans-serif;
        }

        
        /* Layout: full width (dashboard + details) */
        .block-container{
            max-width: 100% !important;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
        }
        @media (max-width: 900px){
            .block-container{padding-left: 1rem; padding-right: 1rem;}
        }

/* Increase default text size across the app (Report + modules) */
        .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {
            font-size: 17px !important;
            line-height: 1.55 !important;
        }
        .stMarkdown h2 { font-size: 26px !important; }
        .stMarkdown h3 { font-size: 22px !important; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .main, .block-container {
            background: #FFFFFF !important;
            color: #0F172A !important;
        }
        [data-testid="stSidebar"] * {
            color: #0F172A !important; /* labels/text default dark */
        }
        strong {
            color: #0F172A;
            font-weight: 700;
        }
        h1, h2, h3 {
            color: #0F172A;
        }
        /* Full-width glossy black button */
        .stButton>button {
            width: 100%;
            background: linear-gradient(180deg, #111827 0%, #000000 100%);
            color: #FFFFFF !important;
            font-weight: 700;
            border-radius: 10px;
            height: 44px;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 6px 14px rgba(0,0,0,0.45);
        }
        .stButton>button:hover {
            background: linear-gradient(180deg, #0B1220 0%, #000000 100%);
            border: 1px solid rgba(255,255,255,0.18);
        }

    /* =========================
       GAME CHARACTER CARD
       ========================= */
    .gc-card{border:1px solid #E5E7EB;border-radius:16px;padding:14px 14px 10px;background:#ffffff;}
    .gc-head{display:block;margin-bottom:10px;}
    .gc-title{font-weight:800;letter-spacing:.6px;font-size:12px;color:#6B7280;}
    .gc-class{font-weight:800;font-size:18px;color:#111827;}
    .gc-h1{font-weight:900;font-size:32px;color:#0F172A;line-height:1.2;}
    .gc-blurb{margin-top:8px;font-size:18px;line-height:1.6;color:#334155;}
    .gc-bline{margin-top:6px;font-size:18px;line-height:1.5;color:#334155;}
    .gc-bline b{color:#0F172A;}
    .gc-radar-wrap{display:flex;gap:14px;align-items:center;}
    .gc-radar-svg{width:220px;height:220px;flex:0 0 auto;}
    .gc-radar-metrics{flex:1;min-width:220px;}
    .gc-radar-item{display:flex;justify-content:space-between;gap:10px;margin:4px 0;font-size:16px;color:#334155;}
    .gc-radar-lab{font-weight:700;color:#334155;}
    .gc-radar-val{font-weight:800;color:#0F172A;}

    .gc-sec{margin-top:10px;padding-top:10px;border-top:1px dashed #E5E7EB;}
    .gc-sec-t{font-weight:900;font-size:20px;color:#374151;margin-bottom:10px;}
    .gc-row{display:flex;gap:10px;align-items:center;margin:6px 0;}
    .gc-k{width:190px;font-size:20px;color:#374151;}
    .gc-bar{flex:1;height:16px;background:#F3F4F6;border-radius:99px;overflow:hidden;}
    .gc-fill{height:16px;background:linear-gradient(90deg,#2563EB 0%,#7C3AED 100%);border-radius:99px;}
    .gc-v{width:96px;text-align:right;font-size:20px;color:#111827;font-weight:800;}
    .gc-flag{display:flex;gap:8px;align-items:center;margin:6px 0;padding:6px 8px;background:#F9FAFB;border-radius:10px;border:1px solid #EEF2F7;}
    .gc-sev{font-size:14px;font-weight:800;color:#111827;background:#E5E7EB;border-radius:8px;padding:2px 6px;}
    .gc-code{font-size:14px;font-weight:800;color:#374151;}
    .gc-note{font-size:17px;color:#6B7280;}
    .gc-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;}
    .gc-tag{font-size:15px;background:#111827;color:#fff;border-radius:999px;padding:4px 10px;}
    .gc-conv{display:grid;gap:6px;}
    .gc-conv-tier,.gc-conv-pts{font-size:24px;color:#111827;font-weight:600;}
    .gc-conv-guide{font-size:20px;color:#6B7280;line-height:1.35;}
    

    
/* =========================
   EXECUTIVE SNAPSHOT (DASHBOARD) — DARK THEME
   ========================= */
.es-card{
  border:1px solid rgba(255,255,255,0.16);
  border-radius:18px;
  padding:14px;
  background:#081A33;
  margin-top:8px;
  width:100%;
  box-sizing:border-box;
  box-shadow: 0 2px 10px rgba(0,0,0,0.20);
}
.es-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;}
.es-left{display:flex;flex-direction:column;gap:4px;}
.es-tline{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.es-ticker{font-weight:900;font-size:26px;color:#FFFFFF;letter-spacing:0.4px;}
.es-price{font-weight:900;font-size:26px;color:#FFFFFF;}
.es-chg{
  font-weight:900;font-size:14px;padding:4px 10px;border-radius:999px;
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.16);
  color:#FFFFFF;
}
.es-sub{font-size:16px;color:rgba(255,255,255,0.82);font-weight:800;}
.es-right{text-align:right;display:flex;flex-direction:column;gap:6px;}
.es-badge{
  font-weight:900;font-size:13px;padding:6px 10px;border-radius:999px;
  background:#0B1426;color:#FFFFFF;display:inline-block;
  border:1px solid rgba(255,255,255,0.14);
}
.es-kelly{
  font-weight:900;font-size:13px;padding:6px 10px;border-radius:999px;
  background:#0F2A44;color:#FFFFFF;border:1px solid rgba(255,255,255,0.18);
  display:inline-block;
}
.es-meta{font-size:13px;color:rgba(255,255,255,0.70);font-weight:800;}
.es-body{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:14px;}
@media(max-width: 980px){.es-body{grid-template-columns:1fr;}.es-right{text-align:left;}}
.es-panel{
  border:1px solid rgba(255,255,255,0.14);
  border-radius:16px;
  padding:12px;
  background:#0F2A44;
}
.es-pt{font-weight:950;font-size:14px;color:rgba(255,255,255,0.70);letter-spacing:0.7px;margin-bottom:8px;}
.es-metric{display:flex;justify-content:space-between;gap:10px;font-size:16px;margin:6px 0;}
.es-metric .k{color:rgba(255,255,255,0.78);font-weight:850;}
.es-metric .v{color:#FFFFFF;font-weight:950;}
.es-mini{height:10px;background:rgba(255,255,255,0.12);border-radius:99px;overflow:hidden;margin-top:6px;}
.es-mini>div{height:10px;background:linear-gradient(90deg,#2563EB 0%,#7C3AED 100%);border-radius:99px;}
.es-bline-wrap{margin-top:6px;}
.es-bline{font-size:13px;color:rgba(255,255,255,0.82);line-height:1.35;margin:2px 0;}
.es-sig-wrap{display:flex;justify-content:center;align-items:center;margin-top:10px;}
.es-sig-radar{flex:0 0 220px;}
.es-radar-svg{width:220px;height:220px;display:block;}
.es-sig-metrics{flex:1;}
.es-sig-row{display:flex;justify-content:space-between;gap:10px;font-size:14px;margin:4px 0;}
.es-sig-row .k{color:rgba(255,255,255,0.78);font-weight:850;}
.es-sig-row .v{color:#FFFFFF;font-weight:950;}
.es-note{font-size:14px;color:rgba(255,255,255,0.82);line-height:1.45;}
.es-bul{margin:6px 0 0 16px;padding:0;}
.es-bul li{margin:2px 0;font-size:14px;color:rgba(255,255,255,0.86);font-weight:650;}
.es-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px;}
.es-dot.g{background:#22C55E;}
.es-dot.y{background:#F59E0B;}
.es-dot.r{background:#EF4444;}


/* Make expander summary look like an action button */
div[data-testid="stExpander"] > details{border:0 !important; background:transparent !important;}
div[data-testid="stExpander"] > details > summary{
    background: linear-gradient(180deg, #111827 0%, #000000 100%);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    padding: 10px 14px;
}
div[data-testid="stExpander"] > details > summary:hover{opacity:0.98;}


/* =========================
       MAJOR SECTION HEADERS
       ========================= */
    .major-sec{
        background:#0b1426;
        border:3px solid #FFFFFF;
        border-radius:18px;
        padding:15px 20px;
        margin:30px 0px 20px 0px;
        color:#FFFFFF;
        font-weight:900;
        font-size:26px;
        letter-spacing:0.8px;
        text-transform:uppercase;
    }
    
</style>
    """, unsafe_allow_html=True)


    st.markdown("""
    <style>
      /* App container */
      .incept-wrap { max-width: 1200px; margin: 0 auto; }
      .incept-header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding: 6px 0 8px 0; }
      .incept-brand { font-size: 60px; font-weight: 900; letter-spacing: 0.5px; }
      .stMarkdown .incept-brand { font-size: 60px !important; font-weight: 900; letter-spacing: 0.5px; }
      .incept-nav { display:flex; gap:18px; align-items:center; }
      .incept-nav a { text-decoration:none; font-weight:800; color:#0F172A; font-size:14px; letter-spacing:0.6px; }
      .incept-nav a:hover { opacity: 0.75; }

      /* Streamlit top header */
      header[data-testid="stHeader"] { background: #ffffff !important; }
      /* Remove any dark tint behind header */
      div[data-testid="stDecoration"] { background: #ffffff !important; }

      /* Sidebar inputs/button text color */
      section[data-testid="stSidebar"] { color: #0F172A !important; }
      section[data-testid="stSidebar"] * { color: #0F172A !important; }
      section[data-testid="stSidebar"] input {
        color: #0F172A !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
      }
      section[data-testid="stSidebar"] input::placeholder { color: #475569 !important; }
      section[data-testid="stSidebar"] textarea { color: #ffffff !important; }
      section[data-testid="stSidebar"] button { color: #ffffff !important; }

      /* Sidebar background + border */
      section[data-testid="stSidebar"]{
        background: #F4F6F8 !important;
        border-right: 1px solid #E2E8F0 !important;
      }
      section[data-testid="stSidebar"] > div{
        background: transparent !important;
      }

      /* If your inputs are created via st.text_input, the visible text is inside this */
      .stTextInput input {
        color: #0F172A !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
      }
      .stTextInput input::placeholder { color: #475569 !important; }

      /* Button label (Streamlit uses nested div/span) */
      .stButton > button { color: #ffffff !important; }
      /* Sidebar button text white (normal + hover) and dark background */
      section[data-testid="stSidebar"] .stButton > button,
      section[data-testid="stSidebar"] .stButton > button:hover{
        color: #FFFFFF !important;
        background: #0B0F17 !important;
        border: 1px solid #0B0F17 !important;
      }

      /* Sidebar "Phân tích" button: white bg, black border, black text */
      section[data-testid="stSidebar"] .stButton > button{
        background: #FFFFFF !important;
        border: 1px solid #0F172A !important;
        color: #0F172A !important;
        box-shadow: none !important;
      }
      /* Hover state */
      section[data-testid="stSidebar"] .stButton > button:hover{
        background: #F8FAFC !important;
        border: 1px solid #0F172A !important;
        color: #0F172A !important;
      }
      /* Active/pressed state */
      section[data-testid="stSidebar"] .stButton > button:active{
        background: #F1F5F9 !important;
        border: 1px solid #0F172A !important;
        color: #0F172A !important;
      }
      /* Disabled state (keep readable) */
      section[data-testid="stSidebar"] .stButton > button:disabled{
        background: #E2E8F0 !important;
        border: 1px solid #94A3B8 !important;
        color: #334155 !important;
        opacity: 1 !important;
      }

      /* Report section titles (match header line 1 size) */
      .sec-title { font-size: 34px; font-weight: 900; letter-spacing: 0.6px; margin: 18px 0 10px 0; }

      /* Cards / callouts / metrics */
      .incept-card {
        background: #F7F7F9;
        border: 1px solid #DDDEE2;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0;
      }
      .incept-callout {
        background: #FFF7ED;
        border: 1px solid #FDBA74;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 10px 0;
      }
      .incept-divider { height: 1px; background: #E2E8F0; margin: 16px 0; }
      .incept-metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin-top: 10px;
      }
      .incept-metric {
        background: #0F172A;
        color: white;
        border-radius: 14px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }
      .incept-metric .k { font-size: 14px; color: #CBD5E1; margin: 0; font-weight: 800; }
      .incept-metric .v { font-size: 20px; font-weight: 900; line-height: 1.2; }


      /* =========================
         TRADE PLAN (SETUP OVERVIEW)
         ========================= */
      .tp-sec-h{background:#FFEDD5;border:1px solid #FDBA74;border-radius:14px;padding:12px 14px;margin:10px 0 8px;}
      .tp-sec-title{display:flex;align-items:center;justify-content:space-between;gap:10px;font-weight:900;font-size:18px;color:#9A3412;}
      .tp-badge{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:900;font-size:12px;letter-spacing:.6px;}
      .tp-badge.active{background:#FB923C;color:#111827;}
      .tp-badge.watch{background:#FFF7ED;color:#9A3412;border:1px solid #FDBA74;}
      .tp-note{background:#FFF7ED;border-left:6px solid #FB923C;padding:10px 12px;border-radius:12px;color:#7C2D12;font-weight:700;margin:8px 0 10px;}
      .tp-card{border:1px solid #E5E7EB;border-radius:16px;padding:12px 12px;background:#ffffff;margin:10px 0;}
      .tp-title{font-size:16px;color:#0F172A;}
      .tp-status{font-weight:900;color:#64748B;font-size:12px;margin-left:6px;}
      .tp-meta{color:#475569;font-size:13px;margin-top:6px;}
      .tp-levels{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;color:#0F172A;font-size:14px;}
      .tp-levels span{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:6px 10px;}
      .tp-card.dim{opacity:.55;}
      .tp-ref{display:inline-block;margin-left:8px;padding:3px 8px;border-radius:999px;background:#F1F5F9;border:1px dashed #CBD5E1;color:#475569;font-size:11px;font-weight:950;letter-spacing:.6px;text-transform:uppercase;}
      .tp-expl{background:#ffffff;border:1px solid #E2E8F0;border-left:5px solid #FDBA74;border-radius:14px;padding:10px 12px;margin:-2px 0 10px;color:#334155;font-size:14px;line-height:1.55;}

      /* =========================
         DECISION LAYER (CENTRAL SWITCH)
         ========================= */
      .dl-wrap{border:2px solid #CBD5E1;border-radius:18px;padding:14px;background:#ffffff;margin-top:10px;}
      .dl-header{font-weight:950;font-size:26px;color:#0F172A;line-height:1.2;background:#FFEDD5;border:1px solid #FDBA74;border-radius:16px;padding:12px 14px;margin-bottom:12px;}
      .dl-hero{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;}
      .dl-card{flex:1;min-width:260px;border-radius:16px;padding:14px;border:1px solid #E5E7EB;}
      .dl-card.low{background:#F8FAFC;}
      .dl-card.med{background:#FFF7ED;border-color:#FDBA74;}
      .dl-card.high{background:#ECFDF5;border-color:#A7F3D0;}
      .dl-k{font-size:12px;font-weight:900;letter-spacing:.8px;color:#64748B;}
      .dl-v{font-size:22px;font-weight:950;color:#0F172A;margin-top:2px;}
      .dl-sub{color:#334155;font-size:14px;margin-top:6px;line-height:1.5;}
      .dl-sec{margin-top:10px;padding-top:10px;border-top:1px dashed #E5E7EB;}
      .dl-sec-t{font-weight:900;font-size:16px;color:#374151;margin-bottom:8px;}
      .dl-flag{display:flex;gap:10px;align-items:flex-start;background:#ffffff;border:1px solid #E2E8F0;border-radius:12px;padding:10px 10px;margin:8px 0;}
      .dl-sev{background:#FB923C;color:#111827;font-weight:950;border-radius:10px;padding:4px 8px;font-size:12px;min-width:34px;text-align:center;}
      .dl-code{font-weight:900;color:#0F172A;font-size:12px;min-width:120px;}
      .dl-note{color:#334155;font-size:13px;line-height:1.45;}
      .dl-taghint{ color: rgba(255,255,255,0.78) !important; }
      .dl-tags{margin-top:4px;}
      .dl-pill{display:inline-block;margin:6px 8px 0 0;padding:6px 10px;border-radius:999px;background:#F1F5F9;border:1px solid #E2E8F0;color:#0F172A;font-size:12px;font-weight:850;}
      .dl-bias-tags{margin-top:8px;line-height:1.2;}
      .dl-pill-mini{font-size:11px;padding:5px 9px;margin:6px 6px 0 0;}

      /* Conviction tier label (Decision Layer) */
      .conv-tag{display:block;font-weight:900;text-transform:uppercase;margin-top:8px;font-size:14px;letter-spacing:0.6px;}
      .conv-noedge{color:#EF4444;}
      .conv-weak{color:#FACC15;}
      .conv-tradeable{color:#22C55E;}
      .conv-strong{color:#16A34A;}
      .conv-high{color:#3B82F6;}
      .conv-god{color:#A855F7;}
      .conv-unknown{color:#E5E7EB;}

      /* Playstyle tag explanations */
      .dl-tags{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;}
      .dl-tagitem{display:inline-flex;flex-direction:column;gap:6px;max-width:520px;}
      .dl-taghint{font-size:12px;color:rgba(255,255,255,0.78);line-height:1.45;}

      /* Right placeholder panel */
      .right-panel {
        border: 1px dashed #CBD5E1;
        border-radius: 14px;
        padding: 14px 14px;
        min-height: 520px;
        background: #ffffff;
      }
      .right-panel .t { font-weight: 900; color:#0F172A; margin-bottom: 8px; }
      .right-panel .d { color:#64748B; font-size: 13px; line-height: 1.5; }

      /* Sidebar toggle button always black */
      button[aria-label="Toggle sidebar"],
      button[title="Collapse sidebar"],
      [data-testid="collapsedControl"] button {
        background: #000000 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
      }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("""
    <style>
      /* =========================
         DARK BLUE THEME OVERRIDES
         ========================= */
      :root{
        --incept-bg:#0B1F3A;         /* Dark Blue */
        --incept-panel:#0F2A44;      /* Slightly lighter for inputs */
        --incept-text:#FFFFFF;
      }

      /* Main background */
      body, .stApp, [data-testid="stAppViewContainer"], .main, .block-container{
        background: var(--incept-bg) !important;
        color: var(--incept-text) !important;
      }

      /* Streamlit top header */
      header[data-testid="stHeader"],
      div[data-testid="stDecoration"]{
        background: var(--incept-bg) !important;
      }

      /* Sidebar background + text */
      section[data-testid="stSidebar"]{
        background: var(--incept-bg) !important;
        border-right: 1px solid rgba(255,255,255,0.10) !important;
      }
      section[data-testid="stSidebar"] > div{ background: transparent !important; }
      section[data-testid="stSidebar"] *{
        color: var(--incept-text) !important;
      }

      /* Global typography */
      .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div,
      label, .stTextInput label, .stSelectbox label, .stRadio label{
        color: var(--incept-text) !important;
      }
      h1,h2,h3,h4,h5,h6{ color: var(--incept-text) !important; }

      /* Top nav links */
      .incept-nav a{ color: var(--incept-text) !important; }

      /* Inputs */
      .stTextInput input,
      section[data-testid="stSidebar"] input{
        color: var(--incept-text) !important;
        background: var(--incept-panel) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
      }
      .stTextInput input::placeholder,
      section[data-testid="stSidebar"] input::placeholder{
        color: rgba(255,255,255,0.60) !important;
      }

      /* Sidebar button */
      section[data-testid="stSidebar"] .stButton > button{
        background: #000000 !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        color: var(--incept-text) !important;
        box-shadow: none !important;
      }
      section[data-testid="stSidebar"] .stButton > button:hover{
        background: #111827 !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        color: var(--incept-text) !important;
      }

      /* Major section headers: BIG, BOLD, HIGH-CONTRAST (no button/badge look) */
      .major-sec{
        font-size: 26px !important;
        font-weight: 900 !important;
        border: 3px solid #FFFFFF !important;
        background: #0b1426 !important;
        padding: 15px 20px !important;
        text-transform: UPPERCASE !important;
        margin: 30px 0px 20px 0px !important;
        border-radius: 18px !important;
        color: #FFFFFF !important;
        letter-spacing: 0.8px !important;
      }

/* Force strong text to stay readable on dark background */
      strong, b{ color: var(--incept-text) !important; }

      /* Generic light cards -> dark cards */
      .incept-card,
      .incept-callout,
      .right-panel,
      .gc-card,
      .dl-wrap,
      .tp-card,
      .tp-expl{
        background: #081A33 !important;
        border-color: rgba(255,255,255,0.16) !important;
        color: var(--incept-text) !important;
      }
      .incept-callout{ border-color: rgba(255,255,255,0.20) !important; }

      /* =========================
         STOCK DNA (gc-*) overrides
         ========================= */
      .gc-title{ color: rgba(255,255,255,0.70) !important; }
      .gc-class,.gc-h1,.gc-sec-t,.gc-k,.gc-v,.gc-conv-tier,.gc-conv-pts{ color: var(--incept-text) !important; }
      .gc-blurb{ color: rgba(255,255,255,0.82) !important; }
      .gc-row{ color: var(--incept-text) !important; }
      .gc-bar{ background: rgba(255,255,255,0.10) !important; }
      .gc-fill{ background: linear-gradient(90deg,#2563EB 0%,#7C3AED 100%) !important; }
      .gc-flag{ background: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.14) !important; }
      .gc-sev{ background: rgba(255,255,255,0.18) !important; color: var(--incept-text) !important; }
      .gc-code{ color: var(--incept-text) !important; }
      .gc-note{ color: rgba(255,255,255,0.72) !important; }

      /* Current Status text that used to render dark */
      .stMarkdown p, .stMarkdown span, .stMarkdown div{ color: var(--incept-text) !important; }

      /* =========================
         TRADE PLAN + R:R (tp-*) theme: indigo/dark blue
         Replace orange/white blocks with blue/purple tones
         ========================= */
      .tp-sec-h{ background:#0F2A44 !important; border:1px solid rgba(255,255,255,0.18) !important; }
      .tp-sec-title{ color: var(--incept-text) !important; }
      .tp-badge.active{ background:#4F46E5 !important; color:#FFFFFF !important; border:1px solid rgba(255,255,255,0.18) !important; }
      .tp-badge.watch{ background:#0B1F3A !important; color:#FFFFFF !important; border:1px solid rgba(255,255,255,0.18) !important; }
      .tp-note{ background:#081A33 !important; border-left:6px solid #4F46E5 !important; color:#FFFFFF !important; }
      .tp-card{ background:#081A33 !important; border:1px solid rgba(255,255,255,0.16) !important; }
      .tp-title{ color:#FFFFFF !important; }
      .tp-status{ color: rgba(255,255,255,0.70) !important; }
      .tp-meta{ color: rgba(255,255,255,0.78) !important; }
      .tp-levels{ color:#FFFFFF !important; }
      .tp-levels span{ background:#0F2A44 !important; border:1px solid rgba(255,255,255,0.18) !important; color:#FFFFFF !important; }
      .tp-ref{ background:#0F2A44 !important; border:1px dashed rgba(255,255,255,0.28) !important; color:#FFFFFF !important; }
      .tp-expl{ background:#081A33 !important; border:1px solid rgba(255,255,255,0.16) !important; border-left:5px solid #4F46E5 !important; color: rgba(255,255,255,0.86) !important; }

      /* =========================
         DECISION LAYER (dl-*) theme: blue/purple + white borders
         ========================= */
      .dl-wrap{ background:#081A33 !important; border:2px solid rgba(255,255,255,0.18) !important; }
      .dl-header{ background:#0F2A44 !important; border:1px solid rgba(255,255,255,0.18) !important; color:#FFFFFF !important; }
      .dl-card{ background:#0B1F3A !important; border:1px solid rgba(255,255,255,0.16) !important; }
      .dl-k{ color: rgba(255,255,255,0.72) !important; }
      .dl-v{ color:#FFFFFF !important; }
      .dl-sub{ color: rgba(255,255,255,0.82) !important; }
      .dl-sec{ border-top:1px dashed rgba(255,255,255,0.16) !important; }
      .dl-sec-t{ color:#FFFFFF !important; }
      .dl-flag{ background:#081A33 !important; border:1px solid rgba(255,255,255,0.16) !important; }
      .dl-sev{ background:#4F46E5 !important; color:#FFFFFF !important; }
      .dl-code{ color:#FFFFFF !important; }
      .dl-note{ color: rgba(255,255,255,0.82) !important; }
      .dl-pill{ background:#0F2A44 !important; border:1px solid rgba(255,255,255,0.18) !important; color:#FFFFFF !important; }
      .dl-taghint{ color: rgba(255,255,255,0.78) !important; }
      .dl-tags{display:flex !important;flex-wrap:wrap !important;gap:12px !important;margin-top:6px !important;}
      .dl-tagitem{display:inline-flex !important;flex-direction:column !important;gap:6px !important;max-width:520px !important;}
      .dl-taghint{font-size:12px !important;color: rgba(255,255,255,0.78) !important;line-height:1.45 !important;}


      /* Risk/Reward metric cards: keep dark + white border accents */
      .incept-metric{ border:1px solid rgba(255,255,255,0.14) !important; }
      .incept-metric .k{ color: rgba(255,255,255,0.72) !important; }
      .right-panel{ border: 1px dashed rgba(255,255,255,0.20) !important; }

      /* GAME CHARACTER CARD (Current Status / Scenario & Scores) - force white text */
      .gc-title,.gc-class,.gc-h1,.gc-blurb,.gc-sec-t,.gc-k,.gc-v,.gc-code,.gc-note{ color: var(--incept-text) !important; }
      .gc-title{ color: rgba(255,255,255,0.70) !important; }
      .gc-blurb,.gc-note{ color: rgba(255,255,255,0.78) !important; }
      .gc-sec{ border-top: 1px dashed rgba(255,255,255,0.18) !important; }
      .gc-bar{ background: rgba(255,255,255,0.12) !important; }
      .gc-flag{ background:#0F2A44 !important; border:1px solid rgba(255,255,255,0.14) !important; }
      .gc-sev{ background: rgba(255,255,255,0.14) !important; color: var(--incept-text) !important; }

      /* =========================
         TRADE PLAN THEME (replace orange/white with indigo + deep blue)
         ========================= */
      .tp-sec-h{background:#0F2A44 !important;border:1px solid rgba(255,255,255,0.20) !important;}
      .tp-sec-title{color: var(--incept-text) !important;}
      .tp-badge.active{background:#4F46E5 !important;color:#FFFFFF !important;border:1px solid rgba(255,255,255,0.22) !important;}
      .tp-badge.watch{background:#081A33 !important;color:#FFFFFF !important;border:1px solid rgba(255,255,255,0.22) !important;}
      .tp-note{background:#081A33 !important;border-left:6px solid #4F46E5 !important;color:#FFFFFF !important;}
      .tp-title,.tp-meta,.tp-levels{color:#FFFFFF !important;}
      .tp-meta{color: rgba(255,255,255,0.82) !important;}
      .tp-status{color: rgba(255,255,255,0.70) !important;}
      .tp-levels span{background:#0F2A44 !important;border:1px solid rgba(255,255,255,0.16) !important;color:#FFFFFF !important;}
      .tp-ref{background:#0F2A44 !important;border:1px dashed rgba(255,255,255,0.30) !important;color:#FFFFFF !important;}
      .tp-expl{border-left:5px solid #4F46E5 !important;color: rgba(255,255,255,0.86) !important;}
      .tp-card{border:1px solid rgba(255,255,255,0.16) !important;}

      /* =========================
         DECISION LAYER THEME (indigo + deep blue)
         ========================= */
      .dl-header{background:#0F2A44 !important;border:1px solid rgba(255,255,255,0.20) !important;color:#FFFFFF !important;}
      .dl-card{border:1px solid rgba(255,255,255,0.16) !important;background:#0F2A44 !important;color:#FFFFFF !important;}
      .dl-card.low,.dl-card.med,.dl-card.high{background:#0F2A44 !important;border-color: rgba(255,255,255,0.16) !important;}
      .dl-k{color: rgba(255,255,255,0.70) !important;}
      .dl-v{color:#FFFFFF !important;}
      .dl-sub{color: rgba(255,255,255,0.82) !important;}
      .dl-sec{border-top:1px dashed rgba(255,255,255,0.18) !important;}
      .dl-sec-t{color:#FFFFFF !important;}
      .dl-flag{background:#081A33 !important;border:1px solid rgba(255,255,255,0.16) !important;}
      .dl-sev{background:#4F46E5 !important;color:#FFFFFF !important;}
      .dl-code{color:#FFFFFF !important;}
      .dl-note{color: rgba(255,255,255,0.85) !important;}
      .dl-pill{background:#0F2A44 !important;border:1px solid rgba(255,255,255,0.16) !important;color:#FFFFFF !important;}
    </style>
    """, unsafe_allow_html=True)


    # 12. STREAMLIT UI & APP LAYOUT
    # ============================================================
    st.markdown(f"""
    <div class="incept-wrap">
      <div class="incept-header">
        <div class="incept-brand">{APP_TITLE}</div>
        <div class="incept-nav">
          <a href="javascript:void(0)">CỔ PHIẾU</a>
          <a href="javascript:void(0)">DANH MỤC</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    with st.sidebar:
        user_key = st.text_input("Client Code", type="password", placeholder="Client Code")
        ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB", key="ticker_input").upper()

        # Reset holding-mode inputs when switching ticker to avoid carrying stale avg_cost/PnL into a new symbol
        _prev_ticker = st.session_state.get("__last_ticker")
        if _prev_ticker is None:
            st.session_state["__last_ticker"] = ticker_input
        elif _safe_text(_prev_ticker).strip().upper() != ticker_input:
            st.session_state["position_mode_radio"] = "Mua mới (FLAT)"
            st.session_state["avg_cost_inp"] = 0.0
            st.session_state["position_size_pct_inp"] = 0.0
            st.session_state["risk_budget_pct_inp"] = 1.0
            st.session_state["holding_horizon_inp"] = "Swing"
            st.session_state["timeframe_inp"] = "D"
            st.session_state["__last_ticker"] = ticker_input

        # ------------------------------
        # Position Mode (Portfolio-ready; Long-only)
        # ------------------------------
        _pm_label = st.radio(
            "Tình trạng vị thế",
            ["Mua mới (FLAT)", "Đang nắm giữ (HOLDING)"],
            index=0,
            key="position_mode_radio"
        )
        position_mode = "HOLDING" if str(_pm_label).startswith("Đang") else "FLAT"

        with st.expander("Thông tin vị thế (tuỳ chọn)", expanded=(position_mode == "HOLDING")):
            avg_cost = st.number_input("Giá vốn (avg cost)", min_value=0.0, value=0.0, step=0.1, key="avg_cost_inp")
            position_size_pct = st.number_input("Tỷ trọng đang nắm (% NAV)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="position_size_pct_inp")
            risk_budget_pct = st.number_input("Ngân sách rủi ro (% NAV / trade)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="risk_budget_pct_inp")
            holding_horizon = st.selectbox("Holding horizon", ["Swing", "Position"], index=0, key="holding_horizon_inp")
            timeframe = st.selectbox("Timeframe", ["D", "W"], index=0, key="timeframe_inp")

        run_btn = st.button("Phân tích", type="primary", use_container_width=True)

        # Mặc định sử dụng layout Appendix E (Character-style); bỏ lựa chọn chế độ hiển thị
        output_mode = "Character"

    # ============================================================
    # 13. MAIN EXECUTION
    # ============================================================
    if run_btn:
        if user_key not in VALID_KEYS:
            st.error("❌ Client Code không đúng. Vui lòng nhập lại.")
        else:
            with st.spinner(f"Đang xử lý phân tích {ticker_input}..."):
                try:
                    avg_cost_f = None
                    try:
                        avg_cost_f = float(avg_cost) if float(avg_cost) > 0 else None
                    except Exception:
                        avg_cost_f = None

                    # In FLAT mode, ignore holding inputs entirely (do not compute/show PnL)
                    if position_mode != "HOLDING":
                        avg_cost_f = None
                        position_size_pct = 0.0

                    pnl_pct = None
                    in_profit = None
                    try:
                        pnl_pct = None
                        in_profit = None
                    except Exception:
                        pnl_pct = None
                        in_profit = None

                    # ------------------------------
                    # Step 12 (v14.7): single pipeline entry
                    # - loads data
                    # - builds AnalysisPack
                    # - injects PositionStatePack
                    # - runs modules (character -> report_ad)
                    # - builds DashboardSummaryPack
                    # ------------------------------
                    result = build_result_pipeline(
                        ticker=ticker_input,
                        data_dir=str(DATA_DIR),
                        price_vol_path=str(PRICE_VOL_PATH),
                        position_mode=position_mode,
                        avg_cost=avg_cost_f,
                        position_size_pct_nav=float(position_size_pct) if isinstance(position_size_pct, (int, float)) else 0.0,
                        risk_budget_pct_nav=float(risk_budget_pct) if isinstance(risk_budget_pct, (int, float)) else 1.0,
                        holding_horizon=holding_horizon,
                        timeframe=timeframe,
                        enabled_modules=["character", "report_ad"],
                    )

                    report = ""
                    try:
                        report = (result or {}).get("Report") if isinstance(result, dict) else ""
                    except Exception:
                        report = ""
                    if not report:
                        report = generate_insight_report(result if isinstance(result, dict) else {})
                    st.markdown("<hr>", unsafe_allow_html=True)
                    analysis_pack = result.get("AnalysisPack", {}) if isinstance(result, dict) else {}
                    if not isinstance(analysis_pack, dict):
                        analysis_pack = {}
                    if 'output_mode' in locals() and output_mode == 'Character':
                        # Decision Layer Report (Anti-Anchoring Output Order)
                        render_appendix_e(result, report, analysis_pack)
                    else:
                        # Legacy report A–D
                        render_report_pretty(report, analysis_pack)
                except Exception as e:
                    st.error(f"⚠️ Lỗi xử lý: {e}")
                    try:
                        import traceback
                        with st.expander('Chi tiết lỗi (traceback)', expanded=False):
                            st.code(traceback.format_exc())
                    except Exception:
                        pass
                    st.exception(e)

    # ============================================================
    # 14. FOOTER
    # ============================================================
    st.divider()
    st.markdown(
        f"""
        <p style='text-align:center; color:#6B7280; font-size:13px;'>
        © 2026 INCEPTION Research Framework<br>
        Phiên bản {APP_VERSION} | Engine GPT-4o
        </p>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
