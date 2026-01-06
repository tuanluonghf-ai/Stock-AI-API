from typing import Any, Dict, List, Optional, Tuple, Union
import math



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

APP_VERSION = "11.8"
APP_TITLE = "INCEPTION"

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
from openai import OpenAI
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
# ============================================================
# LOCAL MODULE REGISTRY OVERRIDE (v6.1)
# - Ensures Phase 4 can run as a single-file app without relying
#   on external `inception.modules.*` versions that may be stale.
# - Prevents pandas Series/DataFrame from leaking into UI boolean contexts.
# ============================================================
def _json_sanitize(obj: Any):
    """Recursively convert pandas/numpy objects into JSON-safe scalars."""
    if isinstance(obj, (pd.Series, pd.Index, pd.DataFrame)):
        return _as_scalar(obj)
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def run_modules(analysis_pack: Dict[str, Any], enabled: List[str], ctx: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Minimal module runner (single-file). Returns (modules_out, errors)."""
    modules_out: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    enabled_set = set([str(x).strip().lower() for x in (enabled or [])])

    # --- Report A–D (GPT narrative) ---
    if "report_ad" in enabled_set:
        try:
            base_result = ctx.get("result") if isinstance(ctx, dict) else None
            if isinstance(base_result, dict):
                modules_out["report_ad"] = {"report": str(generate_insight_report(base_result))}
            else:
                modules_out["report_ad"] = {"report": ""}
        except Exception as e:
            errors.append(f"report_ad: {e}")
            modules_out["report_ad"] = {"report": "", "Error": str(e)}

    # --- Character ---
    if "character" in enabled_set:
        try:
            df_used = ctx.get("df") if isinstance(ctx, dict) else None
            if df_used is None or not hasattr(df_used, "columns"):
                raise ValueError("Missing df in ctx for character module")
            cp = compute_character_pack(df_used, analysis_pack or {})
            if not isinstance(cp, dict):
                raise ValueError("CharacterPack must be a dict")
            cp = _json_sanitize(cp)
            modules_out["character"] = cp
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            errors.append(f"character: {e}")
            # Include traceback for fast diagnosis on localhost
            modules_out["character"] = {"Error": str(e), "Traceback": tb}

    return modules_out, errors

import inception.modules.report_ad  # registers report_ad
import inception.modules.character  # registers character



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

def _scenario_vi(x: str) -> str:
    m = {
        "Uptrend – Breakout Confirmation": "Xu hướng tăng — Xác nhận bứt phá",
        "Uptrend – Pullback Phase": "Xu hướng tăng — Pha điều chỉnh",
        "Downtrend – Weak Phase": "Xu hướng giảm — Yếu",
        "Neutral / Sideways": "Đi ngang / Trung tính",
    }
    return m.get(x, x)

# ============================================================
# 3B. GUARD-D: PRIMARYSETUP + VALIDATION + RETRY
# ============================================================
def _extract_d_block(text: str) -> str:
    m = re.search(r"(^|\n)\s*D\.\s*Rủi\s*ro.*$", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return m.group(0) if m else text

def _grab_number(block: str, label_patterns: List[str]) -> float:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            s = s.replace(",", "")
            s = s.replace("%", "")
            return _safe_float(s)
    return np.nan

def _grab_text(block: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            return s
    return None

def _close_enough(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol

def validate_section_d(text: str, primary: Dict[str, Any]) -> bool:
    block = _extract_d_block(text)
    got_risk = _grab_number(block, [
        r"Risk%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"Rủi\s*ro\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"RiskPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
    ])
    got_reward = _grab_number(block, [
        r"Reward%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"Lợi\s*nhuận\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"RewardPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
    ])
    got_rr = _grab_number(block, [
        r"\bRR\b\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"R\s*[:/]\s*R\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    ])
    got_prob = _grab_text(block, [
        # New canonical label
        r"Confidence\s*\(\s*Tech\s*\)\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
        r"Độ\s*tin\s*cậy\s*\(\s*Kỹ\s*thuật\s*\)\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
        # Backward compatibility
        r"Probability\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
        r"Xác\s*suất\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
    ]) or ""
    
    exp_risk = _safe_float(primary.get("RiskPct"))
    exp_reward = _safe_float(primary.get("RewardPct"))
    exp_rr = _safe_float(primary.get("RR"))
    exp_prob = _safe_text(primary.get("Confidence (Tech)", primary.get("Probability"))).strip().lower()
    
    ok = True
    if pd.notna(exp_risk):
        ok &= _close_enough(got_risk, exp_risk, tol=0.05)
    if pd.notna(exp_reward):
        ok &= _close_enough(got_reward, exp_reward, tol=0.05)
    if pd.notna(exp_rr):
        ok &= _close_enough(got_rr, exp_rr, tol=0.05)
        
    # got_prob already a string or None from _grab_text
    gp = (str(got_prob).strip().lower() if got_prob is not None else "")
    if exp_prob:
        ok &= (exp_prob in gp) if gp else False
        
    return bool(ok)

def _call_openai(prompt: str, temperature: float) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=1600
    )
    return response.choices[0].message.content

def call_gpt_with_guard(prompt: str, analysis_pack: Dict[str, Any], max_retry: int = 2) -> str:
    primary = (analysis_pack.get("PrimarySetup") or {})
    temps = [0.7, 0.2, 0.0]
    last_text = ""
    
    for i in range(min(max_retry + 1, len(temps))):
        temp = temps[i]
        extra = ""
        if i > 0:
            extra = f"""
SỬA LỖI BẮT BUỘC (chỉ sửa mục D, giữ nguyên các mục khác):
Mục D đang sai số. Hãy sửa lại mục D bằng cách COPY ĐÚNG các số sau (không được tự tính/ước lượng):
Risk%={primary.get('RiskPct')}, Reward%={primary.get('RewardPct')}, RR={primary.get('RR')}, Confidence (Tech)={primary.get('Confidence (Tech)', primary.get('Probability'))}.

Mục D bắt buộc đúng format 4 dòng:
Risk%: <...>
Reward%: <...>
RR: <...>
Confidence (Tech): <...>
"""
        text = _call_openai(prompt + extra, temperature=temp)
        last_text = text
        
        if validate_section_d(text, primary):
            return text
            
    return last_text

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


def render_game_card(data: Dict[str, Any]):
    """Vẽ Radar Chart cho Game Character"""
    import plotly.graph_objects as go

    if not data:
        st.warning("Chưa có dữ liệu Game Character")
        return

    stats = data.get("Stats", {})
    archetype = data.get("Class", "N/A")
    avg = data.get("AvgScore", 0)

    categories = ["ATK", "SPD", "DEF", "HP", "CRT"]
    values = [
        stats.get("ATK", 0),
        stats.get("SPD", 0),
        stats.get("DEF", 0),
        stats.get("HP", 0),
        stats.get("CRT", 0),
    ]
    # Khép kín vòng tròn
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=archetype,
            line_color="#00FF00" if avg >= 7 else ("#FFFF00" if avg >= 5 else "#FF0000"),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=30, b=30),
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Class", archetype)
        st.metric("Power Level", f"{avg}/10")
    with c2:
        st.plotly_chart(fig, use_container_width=True)

# ============================================================


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
    
    # Primary Picker
    def pick_primary_setup_v2(rrsim: Dict[str, Any]) -> Dict[str, Any]:
        setups = rrsim.get("Setups", []) or []
        if not setups:
            return {"Name": "N/A", "RiskPct": None, "RewardPct": None, "RR": None, "Confidence (Tech)": "N/A"}
        def status_rank(s):
            stt = (s.get("Status") or "Watch").strip().lower()
            if stt == "active": return 0
            if stt == "watch": return 1
            return 2
        best = None
        for s in setups:
            rr = _safe_float(s.get("RR"))
            if pd.isna(rr): continue
            if best is None:
                best = s
                continue
            if status_rank(s) < status_rank(best):
                best = s
                continue
            if status_rank(s) > status_rank(best):
                continue
            if rr > _safe_float(best.get("RR")):
                best = s
        best = best or setups[0]
        return {
            "Name": best.get("Setup") or best.get("Name") or "N/A",
            "RiskPct": _safe_float(best.get("RiskPct")),
            "RewardPct": _safe_float(best.get("RewardPct")),
            "RR": _safe_float(best.get("RR")),
            "Confidence (Tech)": best.get("Confidence (Tech)", best.get("Probability", "N/A"))
        }
    # Attach StructureQuality to AnalysisPack (single source of truth for flags/gates/plan anchoring)
    try:
        analysis_pack["StructureQuality"] = sanitize_pack(struct_q) if isinstance(struct_q, dict) else {}
    except Exception:
        analysis_pack["StructureQuality"] = struct_q if isinstance(struct_q, dict) else {}

    
    primary = pick_primary_setup_v2(rrsim)
    # Phase 5: prevent NaN leakage in PrimarySetup
    primary = sanitize_pack(primary)
    analysis_pack["PrimarySetup"] = primary
    
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

def _clip(x: float, lo: float, hi: float) -> float:
    if pd.isna(x): return np.nan
    return float(max(lo, min(hi, x)))

def _bucket_score(x: float, bins: list, scores: list) -> float:
    """
    bins: ascending thresholds (len = n)
    scores: corresponding scores (len = n+1)
    """
    if pd.isna(x): return np.nan
    for i, b in enumerate(bins):
        if x < b:
            return float(scores[i])
    return float(scores[-1])

def _avg(*xs: float) -> float:
    vals = [x for x in xs if pd.notna(x)]
    return float(np.mean(vals)) if vals else np.nan

def _atr_last(df: pd.DataFrame, period: int = 14) -> float:
    a = atr_wilder(df, period)
    if a is None or getattr(a, "empty", True):
        return np.nan
    s = a.dropna()
    return _safe_float(s.iloc[-1]) if not s.empty else np.nan

def _candle_strength_from_class(candle_class: str) -> float:
    """
    Returns [0..1] strength multiplier from candle label.
    """
    c = (candle_class or "").strip().lower()
    if c in ("strong_bull", "bull_engulf", "bull_engulfing", "marubozu_bull"):
        return 1.0
    if c in ("bull", "hammer", "bullish_hammer"):
        return 0.7
    if c in ("doji_high_vol", "shooting_star", "gravestone", "spinning_top"):
        return 0.35
    if c in ("strong_bear", "bear_engulf", "bear_engulfing", "marubozu_bear"):
        return 0.0
    if c in ("bear",):
        return 0.2
    return 0.5

def _derive_liquidity_score(vol_ratio: float, liquidity_base: float) -> float:
    """
    Non-invasive fallback:
    - If liquidity_base exists (0..10), use it.
    - Else approximate using vol_ratio (not perfect but stable).
    """
    if pd.notna(liquidity_base):
        return _clip(liquidity_base, 0, 10)
    vr = _safe_float(vol_ratio)
    if pd.isna(vr): return 5.0
    return _bucket_score(vr, bins=[0.6, 0.9, 1.2, 1.6, 2.2], scores=[3, 4.5, 6, 7.5, 8.5, 9.5])

def compute_character_pack(df: pd.DataFrame, analysis_pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces CharacterPack without breaking any existing keys.
    Uses ONLY already-computed features inside AnalysisPack + df series.
    """
    # Defensive normalization: avoid truthiness checks on pandas objects (Series/DataFrame)
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    # Sanitize AnalysisPack values to avoid pandas objects leaking into boolean/math operations
    def _sanitize_obj(obj: Any):
        if isinstance(obj, (pd.Series, pd.Index, pd.DataFrame)):
            return _as_scalar(obj)
        if isinstance(obj, dict):
            return {str(k): _sanitize_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_obj(v) for v in obj]
        return obj

    ap = _sanitize_obj(ap)


    last = ap.get("Last", {})
    last = last if isinstance(last, dict) else {}

    protech = ap.get("ProTech", {})
    protech = protech if isinstance(protech, dict) else {}

    ma = protech.get("MA", {})
    ma = ma if isinstance(ma, dict) else {}
    rsi = protech.get("RSI", {})
    rsi = rsi if isinstance(rsi, dict) else {}
    macd = protech.get("MACD", {})
    macd = macd if isinstance(macd, dict) else {}
    vol = protech.get("Volume", {})
    vol = vol if isinstance(vol, dict) else {}
    pa = protech.get("PriceAction", {})
    pa = pa if isinstance(pa, dict) else {}
    # Pre-read close/MA levels from Last pack (needed for LevelContext fallback inference)
    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    # ---- Level Context (Support/Resistance distances) ----
    # Depending on pipeline version, LevelContext may live in different locations.
    # We MUST NOT accept an "empty" dict (or a dict with only null values), otherwise UpsidePower becomes N/A.
    def _lvl_num(v: Any) -> float:
        if v is None:
            return np.nan
        if isinstance(v, dict):
            v = v.get("Value")
        return _safe_float(v)

    def _lvl_strength(d: Any) -> int:
        if not isinstance(d, dict) or len(d) == 0:
            return 0
        # count how many expected fields have finite numeric content
        keys = [
            "NearestResistance", "NearestSupport",
            "UpsideToResistance", "DownsideToSupport",
            # common alternates
            "Upside", "Downside",
            "Resistance", "Support",
        ]
        score = 0
        for k in keys:
            if k in d:
                x = _lvl_num(d.get(k))
                if pd.notna(x):
                    score += 1
        return score

    lvl_source = "None"
    lvl: Dict[str, Any] = {}

    _candidates = [
        ("ProTech.LevelContext", protech.get("LevelContext")),
        ("Top.LevelContext", ap.get("LevelContext")),
        ("ProTech.Levels", protech.get("Levels")),
        ("Top.Levels", ap.get("Levels")),
    ]

    best_score = 0
    best_name = "None"
    best_cand = None
    for _name, _cand in _candidates:
        s = _lvl_strength(_cand)
        if s > best_score:
            best_score, best_name, best_cand = s, _name, _cand

    if best_score > 0 and isinstance(best_cand, dict):
        lvl = best_cand
        lvl_source = best_name

    # Final safety net: infer nearest S/R locally (MA levels + recent swing high/low)
    # This guarantees Upside/Downside when upstream packs are absent.
    if not isinstance(lvl, dict) or len(lvl) == 0:
        def _infer_nearest_sr(_df: pd.DataFrame, _close: float,
                              _ma20: float, _ma50: float, _ma200: float,
                              lookback: int = 60) -> Tuple[float, float]:
            if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty or pd.isna(_close):
                return (np.nan, np.nan)

            # Candidates from MA levels (if available)
            ma_vals = [x for x in [_ma20, _ma50, _ma200] if pd.notna(x)]
            res_cands = [x for x in ma_vals if x > _close]
            sup_cands = [x for x in ma_vals if x < _close]

            # Candidates from recent swing extremes (robust fallback)
            try:
                if "High" in _df.columns:
                    hh = float(pd.to_numeric(_df["High"], errors="coerce").tail(lookback).max())
                    # If price is at/near a recent high, treat it as the nearest "resistance" (upside room ~ 0).
                    # This avoids UpsidePower=N/A when the stock is making new highs within lookback.
                    if pd.notna(hh) and hh >= _close:
                        res_cands.append(hh)
                if "Low" in _df.columns:
                    ll = float(pd.to_numeric(_df["Low"], errors="coerce").tail(lookback).min())
                    if pd.notna(ll) and ll < _close:
                        sup_cands.append(ll)
            except Exception:
                pass

            nearest_res = min(res_cands) if res_cands else np.nan
            nearest_sup = max(sup_cands) if sup_cands else np.nan
            return (nearest_res, nearest_sup)

        _nr, _ns = _infer_nearest_sr(df, close, ma20, ma50, ma200, lookback=60)
        if pd.notna(_nr) or pd.notna(_ns):
            lvl = {
                "NearestResistance": {"Value": _nr} if pd.notna(_nr) else None,
                "NearestSupport": {"Value": _ns} if pd.notna(_ns) else None,
                "UpsideToResistance": (max(0.0, _nr - close) if (pd.notna(_nr) and pd.notna(close)) else np.nan),
                "DownsideToSupport": (max(0.0, close - _ns) if (pd.notna(_ns) and pd.notna(close)) else np.nan),
            }
            lvl_source = "Local.Infer"
    fib_ctx = ap.get("FibonacciContext", {})
    fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}
    # prefer nested AnalysisPack["Fibonacci"]["Context"] if available
    if not fib_ctx:
        fib = ap.get("Fibonacci", {})
        if isinstance(fib, dict):
            ctx = fib.get("Context", {})
            if isinstance(ctx, dict):
                fib_ctx = ctx

    # --------------------------
    # STRUCTURE QUALITY (quality-aware support/resistance)
    # Prefer AnalysisPack['StructureQuality'] as single source of truth; fallback if missing.
    # --------------------------
    struct_q = ap.get("StructureQuality", {})
    if not isinstance(struct_q, dict) or not struct_q:
        try:
            fib = ap.get("Fibonacci", {}) if isinstance(ap.get("Fibonacci", {}), dict) else {}
            dual_fib_lite = {
                "auto_short": fib.get("Short", {}) if isinstance(fib.get("Short", {}), dict) else {},
                "fixed_long": fib.get("Long", {}) if isinstance(fib.get("Long", {}), dict) else {},
            }
            last_row = df.iloc[-1] if (df is not None and hasattr(df, "iloc") and len(df) > 0) else last
            struct_q = compute_structure_quality_pack(df, last_row, dual_fib=dual_fib_lite, fib_ctx=fib_ctx)
        except Exception:
            struct_q = {}


    primary = ap.get("PrimarySetup", {})
    primary = primary if isinstance(primary, dict) else {}
    rrsim = ap.get("RRSim", {})
    rrsim = rrsim if isinstance(rrsim, dict) else {}

    close = _safe_float(last.get("Close"))
    ma20 = _safe_float(last.get("MA20"))
    ma50 = _safe_float(last.get("MA50"))
    ma200 = _safe_float(last.get("MA200"))

    def _safe_label(obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            v = obj.get("Label")
            return str(v) if v is not None else None
        # allow plain strings like "Positive"/"Negative"/"Flat"
        if isinstance(obj, (str, int, float)):
            s = str(obj).strip()
            return s if s else None
        return None
    s20 = _coalesce(_safe_label(ma.get("SlopeMA20")), ma.get("SlopeMA20Label"))
    s50 = _coalesce(_safe_label(ma.get("SlopeMA50")), ma.get("SlopeMA50Label"))
    s200 = _coalesce(_safe_label(ma.get("SlopeMA200")), ma.get("SlopeMA200Label"))
    structure = _coalesce(ma.get("Structure"), ma.get("StructureSnapshot"), ma.get("StructureSnapshotV2"))

    rsi14 = _safe_float(rsi.get("RSI"))
    rsi_state = _safe_text(_coalesce(rsi.get("State"), rsi.get("Zone"))).strip()
    rsi_div = _safe_text(rsi.get("Divergence")).strip().lower()

    macd_state = _safe_text(macd.get("State")).strip().lower()
    macd_zero = _safe_text(macd.get("ZeroLine")).strip().lower()
    hist_state = _safe_text(macd.get("HistState")).strip().lower()
    macd_div = _safe_text(macd.get("Divergence")).strip().lower()

    vol_ratio = _safe_float(vol.get("Ratio"))
    vol_regime = _safe_text(vol.get("Regime")).strip().lower()

    candle_class = _safe_text(_coalesce(pa.get("CandleClass"), pa.get("Candle"), "")).strip()
    climax = _safe_bool(_coalesce(pa.get("ClimaxFlag"), pa.get("ClimacticFlag")))
    gap = _safe_bool(_coalesce(pa.get("GapFlag"), pa.get("Gap")))
    atr = _atr_last(df, 14)
    atr = _as_scalar(atr)
    vol_proxy = _safe_float(ap.get("VolProxy")) if pd.notna(_safe_float(ap.get("VolProxy"))) else _dynamic_vol_proxy(df, 20)

    # Levels / distances (prefer LevelContext / FibonacciContext packs)
    ns = lvl.get("NearestSupport"); nearest_sup = (ns.get("Value") if isinstance(ns, dict) else _as_scalar(ns))
    nr = lvl.get("NearestResistance"); nearest_res = (nr.get("Value") if isinstance(nr, dict) else _as_scalar(nr))
    nearest_sup = _safe_float(nearest_sup)
    nearest_res = _safe_float(nearest_res)

    upside = _safe_float(lvl.get("UpsideToResistance"))
    downside = _safe_float(lvl.get("DownsideToSupport"))
    if pd.isna(upside) and pd.notna(nearest_res) and pd.notna(close):
        upside = max(0.0, nearest_res - close)
    if pd.isna(downside) and pd.notna(nearest_sup) and pd.notna(close):
        downside = max(0.0, close - nearest_sup)

    denom = atr if pd.notna(atr) and atr > 0 else (vol_proxy if pd.notna(vol_proxy) and vol_proxy > 0 else np.nan)
    upside_n = upside / denom if pd.notna(denom) and denom > 0 else np.nan
    downside_n = downside / denom if pd.notna(denom) and denom > 0 else np.nan
    rr = (upside / downside) if (pd.notna(upside) and pd.notna(downside) and downside > 0) else _safe_float(primary.get("RR"))

    # If we have a valid downside but no resistance/upside, infer an upside target from RR.
    # This prevents UpsidePower=N/A in "open sky" cases (new highs / no nearby resistance).
    # Conservative approach: only infer when RR is explicitly available and downside>0.
    if pd.isna(upside) and pd.notna(rr) and pd.notna(downside) and downside > 0 and pd.notna(close):
        try:
            upside = float(rr) * float(downside)
            if pd.isna(nearest_res):
                nearest_res = float(close) + float(upside)
            if isinstance(lvl, dict):
                lvl["NearestResistance"] = {"Value": float(nearest_res)}
                lvl["UpsideToResistance"] = float(upside)
                # keep existing support if present
            # recompute normalized upside
            if pd.notna(denom) and denom > 0:
                upside_n = upside / denom
        except Exception:
            pass
    fib_conflict = False
    try:
        fc = fib_ctx if isinstance(fib_ctx, dict) else {}
        fib_conflict = _safe_bool(fc.get("FiboConflictFlag"))
    except Exception:
        fib_conflict = False

    confluence_count = np.nan
    try:
        fc = fib_ctx if isinstance(fib_ctx, dict) else {}
        confluence_count = _safe_float(fc.get("ConfluenceCount"))
    except Exception:
        confluence_count = np.nan

    if pd.isna(confluence_count):
        # robust fallback: infer from any iterable hits inside Confluence*WithMA
        try:
            fc = fib_ctx if isinstance(fib_ctx, dict) else {}
            conf_short = fc.get("ConfluenceShortWithMA")
            conf_long = fc.get("ConfluenceLongWithMA")

            def _count_hits(obj):
                # obj can be dict/list/str/number
                if obj is None:
                    return 0
                if isinstance(obj, dict):
                    return sum(_count_hits(v) for v in obj.values())
                if isinstance(obj, (list, tuple, set)):
                    return len(obj)
                # if it's a scalar/str -> treat as 1 hit only if non-empty string
                if isinstance(obj, str):
                    return 1 if obj.strip() else 0
                return 1

            confluence_count = float(_count_hits(conf_short) + _count_hits(conf_long))
            confluence_count = min(confluence_count, 5.0) if pd.notna(confluence_count) else np.nan
        except Exception:
            confluence_count = np.nan

    # --------------------------
    # CORE STATS (0–10)
    # --------------------------
    # Trend: 4 components
    t1 = 2.5 if (pd.notna(close) and pd.notna(ma200) and close >= ma200) else 0.5
    t2 = 2.5 if (pd.notna(ma20) and pd.notna(ma50) and ma20 >= ma50) else 0.5
    t3 = 2.5 if (str(s50).lower() == "positive" and str(s200).lower() != "negative") else (1.25 if str(s50).lower() == "positive" else 0.5)
    cross_obj = ma.get("Cross", {})
    cross_event = ""
    if isinstance(cross_obj, dict):
        cross_ma = cross_obj.get("MA50VsMA200")
        if isinstance(cross_ma, dict):
            cross_event = str(_as_scalar(cross_ma.get("Event")) if cross_ma.get("Event") is not None else "")
        else:
            cross_event = str(_as_scalar(cross_ma) if cross_ma is not None else "")
        if not cross_event:
            cross_event = str(_coalesce(cross_obj.get("Event"), cross_obj.get("Name"), ""))
    else:
        cross_event = str(_as_scalar(cross_obj) if cross_obj is not None else "")
    cross_l = cross_event.strip().lower()
    # CrossUp = golden cross, CrossDown = death cross
    t4 = 2.5 if ("crossup" in cross_l or "golden" in cross_l) else (0.5 if ("crossdown" in cross_l or "death" in cross_l) else 1.25)
    trend = _clip(_avg(t1, t2, t3, t4) * 4.0, 0, 10)  # NOTE: sum of 4 sub-scores (max 10)

    # Momentum: RSI + MACD + Hist + candle
    # RSI best zone: ~60–70 (bullish but not overheated)
    if pd.isna(rsi14):
        m1 = 1.25
    else:
        if 60 <= rsi14 <= 70: m1 = 2.5
        elif 55 <= rsi14 < 60: m1 = 2.0
        elif 70 < rsi14 <= 78: m1 = 1.8
        elif 45 <= rsi14 < 55: m1 = 1.25
        else: m1 = 0.8
    m2 = 2.5 if ("bull" in macd_state and "above" in macd_zero) else (1.8 if "bull" in macd_state else 0.8)
    m3 = 2.5 if "expanding" in hist_state else (1.6 if "flat" in hist_state or "neutral" in hist_state else 1.0)
    m4 = 2.5 * _candle_strength_from_class(candle_class)
    momentum = _clip(_avg(m1, m2, m3, m4) * 4.0, 0, 10)  # NOTE: sum of 4 sub-scores (max 10)

    # Stability: inverse volatility + whipsaw penalty + shock penalty
    # Use denom as proxy for ATR; higher denom relative to price => more volatile.
    if pd.notna(close) and close > 0 and pd.notna(denom):
        vol_pct = denom / close * 100
        s1 = _bucket_score(vol_pct, bins=[1.2, 2.0, 3.0, 4.5], scores=[9.0, 8.0, 6.5, 5.0, 3.5])
    else:
        s1 = 5.5
    structure_l = _safe_text(structure).strip().lower()
    whipsaw = ("mixed" in structure_l) or ("side" in structure_l)
    s2 = 7.5 if not whipsaw else 4.5
    s3 = 7.0 if (str(s50).lower() != "flat") else 5.0
    s4_pen = 1.5 if (_safe_bool(climax) or _safe_bool(gap)) else 0.0
    stability = _clip(_avg(s1, s2, s3) - s4_pen, 0, 10)

    # Reliability: alignment + volume confirm - divergence - whipsaw
    align = 1.0 if (pd.notna(close) and pd.notna(ma50) and pd.notna(ma200) and ((close >= ma50 >= ma200) or (close < ma50 < ma200))) else 0.5
    r1 = 2.5 if align >= 1.0 else 1.25
    r2 = _bucket_score(vol_ratio, bins=[0.8, 1.0, 1.3, 1.8], scores=[0.8, 1.4, 2.0, 2.3, 2.5])
    div_pen = 2.0 if ("bear" in rsi_div or "bear" in macd_div) else 0.0
    r4_pen = 1.5 if whipsaw else 0.0
    reliability = _clip(_avg(r1, r2, 2.0) * 3.0 - div_pen - r4_pen, 0, 10)  # NOTE: sum of 3 sub-scores

    # Liquidity: base if exists, else from vol_ratio
    liquidity_base = _safe_float(ap.get("LiquidityScoreBase"))
    liquidity = _derive_liquidity_score(vol_ratio, liquidity_base)

    core_stats = {
        "Trend": float(trend),
        "Momentum": float(momentum),
        "Stability": float(stability),
        "Reliability": float(reliability),
        "Liquidity": float(liquidity)
    }

    # --------------------------
    # COMBAT STATS (0–10)
    # --------------------------
    upside_power = _bucket_score(upside_n, bins=[0.8, 1.5, 2.5], scores=[2.5, 5.0, 7.5, 9.5])
    downside_risk = (10.0 - _bucket_score(downside_n, bins=[0.8, 1.5, 2.5], scores=[2.5, 5.0, 7.5, 9.5])) if pd.notna(downside_n) else 5.5
    downside_risk = _clip(downside_risk, 0, 10)

    rr_eff = _bucket_score(rr, bins=[1.2, 1.8, 2.5], scores=[2.5, 5.0, 7.5, 9.5]) if pd.notna(rr) else 5.0

    # Breakout Force: close above key + vol confirm + candle - divergence/overheat
    above_ma200 = (pd.notna(close) and pd.notna(ma200) and close >= ma200)
    b1 = 3.5 if above_ma200 else 1.5
    b2 = _bucket_score(vol_ratio, bins=[0.9, 1.2, 1.6, 2.2], scores=[0.8, 1.6, 2.4, 3.0, 3.5])
    b3 = 3.0 * _candle_strength_from_class(candle_class)
    overheat_pen = 1.5 if (pd.notna(rsi14) and rsi14 >= 75 and "contract" in hist_state) else 0.0
    div_pen2 = 1.5 if ("bear" in rsi_div or "bear" in macd_div) else 0.0
    breakout_force = _clip(b1 + b2 + b3 - overheat_pen - div_pen2, 0, 10)

    # Support Resilience: confluence + absorption + RSI integrity
    conf = 3.5 if (pd.notna(confluence_count) and confluence_count >= 3) else (2.0 if pd.notna(confluence_count) and confluence_count >= 2 else 1.0)
    absorption = 2.5 if (_safe_bool(pa.get("NoSupplyFlag")) or "hammer" in _safe_text(candle_class).lower()) else 1.2
    rsi_ok = 2.5 if (pd.notna(rsi14) and rsi14 >= 50) else 1.2
    support_resilience = _clip(conf + absorption + rsi_ok, 0, 10)

    # Upside quality (0–10): keep UpsidePower as raw "room", then score a quality-adjusted variant.
    # - Confidence (Tech) acts as a stabilizer: High > Medium > Low.
    # - BreakoutForce / VolumeRatio / RR reinforce quality (not room).
    ps = ap.get("PrimarySetup") or {}
    conf_label = _safe_text(ps.get("Confidence (Tech)", ps.get("Probability", ""))).strip().lower()

    conf_mult = 1.0
    if "high" in conf_label:
        conf_mult = 1.15
    elif "med" in conf_label:
        conf_mult = 1.00
    elif "low" in conf_label:
        conf_mult = 0.80

    m_breakout = 0.85 + 0.30 * (float(breakout_force) / 10.0)
    m_vol = 1.0
    if pd.notna(vol_ratio):
        m_vol = 0.90 + 0.20 * _clip((float(vol_ratio) - 0.80) / 1.80, 0, 1)
    m_rr = 1.0
    if pd.notna(rr):
        m_rr = 0.90 + 0.20 * _clip((float(rr) - 1.20) / 3.00, 0, 1)

    total_mult = _clip(conf_mult * m_breakout * m_vol * m_rr, 0.65, 1.35)
    upside_quality = _clip(float(_clip(upside_power, 0, 10)) * float(total_mult), 0, 10)

    combat_stats = {
        # Naming: UpsidePower is kept for backward compatibility; UI should call it "Upside Room".
        "UpsideRoom": float(_clip(upside_power, 0, 10)),
        "UpsideQuality": float(upside_quality),
        "UpsidePower": float(_clip(upside_power, 0, 10)),
        "DownsideRisk": float(downside_risk),
        "RREfficiency": float(_clip(rr_eff, 0, 10)),
        "BreakoutForce": float(breakout_force),
        "SupportResilience": float(support_resilience)
    }

    # --------------------------
    # WEAKNESS FLAGS (severity 1–3)
    # --------------------------
    flags = []
    def add_flag(code: str, severity: int, note: str, meta: Optional[Dict[str, Any]] = None):
        rec = {"code": code, "severity": int(severity), "note": note}
        if isinstance(meta, dict) and meta:
            rec["meta"] = meta
        flags.append(rec)

    # NearMajorResistance: quality-aware (structural vs tactical; tier-aware).
    # Prevents false positives where a breakout clears a weak tactical level but hits a structural ceiling (e.g., MA200).
    try:
        ov = ((struct_q or {}).get("OverheadResistance", {}) or {}).get("Nearest", {}) or {}
        meta_p = (struct_q or {}).get("Meta", {}) if isinstance((struct_q or {}).get("Meta", {}), dict) else {}
        near_th = _safe_float(meta_p.get("NearThresholdPct"), default=np.nan)
        dist = _safe_float(ov.get("DistancePct"), default=np.nan)
        hz = _safe_text(ov.get("Horizon") or "N/A").upper()
        tier = _safe_text(ov.get("Tier") or "N/A").upper()

        comps = ov.get("ComponentsTop") if isinstance(ov.get("ComponentsTop"), list) else []
        type_top = _safe_text((comps[0] or {}).get("Type")) if (len(comps) > 0 and isinstance(comps[0], dict)) else ""
        type_top = type_top.strip()

        is_near = (pd.notna(dist) and pd.notna(near_th) and dist <= near_th)
        if is_near and hz in ("STRUCTURAL", "BOTH") and tier in ("HEAVY", "CONFLUENCE"):
            sev = 3 if tier == "CONFLUENCE" else 2
            note = "Trần cấu trúc gần – upside ngắn bị nén"
            if type_top:
                note = f"{note} ({type_top})"
            add_flag("NearMajorResistance", sev, note, meta={"Horizon": hz, "Tier": tier, "TypeTop": type_top, "DistancePct": dist, "NearThPct": near_th})
        elif is_near and hz == "TACTICAL" and tier in ("LIGHT", "MED"):
            # Minor reminder only (does not block; reduces over-warning on weak tactical levels)
            add_flag("NearMinorResistance", 1, "Cản ngắn hạn gần (tactical)")
    except Exception:
        # Legacy fallback
        if pd.notna(upside_n) and upside_n < 1.0:
            add_flag("NearMajorResistance", 2, "Upside ngắn trước kháng cự gần")
    if pd.notna(vol_ratio) and vol_ratio < 0.9:
        add_flag("NoVolumeConfirm", 2, "Thiếu xác nhận dòng tiền")
    if "bear" in rsi_div:
        add_flag("RSI_BearDiv", 3, "RSI phân kỳ giảm")
    if "bear" in macd_div:
        add_flag("MACD_BearDiv", 3, "MACD phân kỳ giảm")
    if fib_conflict:
        add_flag("TrendConflict", 2, "Xung đột Fib short vs long (ưu tiên luật cấu trúc)")
    if whipsaw:
        add_flag("WhipsawZone", 2, "Vùng nhiễu quanh MA/structure pha trộn")
    if pd.notna(rsi14) and rsi14 >= 75 and "contract" in hist_state:
        add_flag("Overheated", 2, "Đà nóng nhưng histogram co lại")
    if liquidity <= 4.5:
        add_flag("LiquidityLow", 2, "Thanh khoản thấp, dễ trượt giá")
    if (_safe_bool(climax) or _safe_bool(gap)):
        add_flag("VolShockRisk", 2, "Có dấu hiệu shock/gap")

    # --------------------------
    # CONVICTION TIER (0–7)
    # --------------------------
    points = 0.0
    points += 1.0 if trend >= 7 else (0.5 if trend >= 5 else 0.0)
    points += 1.0 if momentum >= 7 else (0.5 if momentum >= 5 else 0.0)
    # Location: confluence or breakout strength
    points += 1.0 if (pd.notna(confluence_count) and confluence_count >= 3) else (0.5 if breakout_force >= 6.5 else 0.0)
    points += 1.0 if (pd.notna(vol_ratio) and vol_ratio >= 1.2) else (0.5 if pd.notna(vol_ratio) and vol_ratio >= 1.0 else 0.0)
    points += 1.0 if (pd.notna(rr) and rr >= 1.8) else (0.5 if pd.notna(rr) and rr >= 1.4 else 0.0)
    points += 1.0 if reliability >= 7 else (0.5 if reliability >= 5 else 0.0)

    # Bonus: killer zone (confluence>=4 + strong candle + vol confirm)
    if (pd.notna(confluence_count) and confluence_count >= 4 and _candle_strength_from_class(candle_class) >= 0.7 and pd.notna(vol_ratio) and vol_ratio >= 1.2):
        points += 1.0

    # Penalties
    for f in flags:
        if f["severity"] == 2: points -= 0.5
        if f["severity"] == 3: points -= 1.0
        if f["code"] == "TrendConflict" and f["severity"] >= 2: points -= 0.5

    points = float(max(0.0, min(7.0, points)))

    # Map points to tier (same thresholds as spec)
    if points <= 1.5: tier = 1
    elif points <= 2.5: tier = 2
    elif points <= 3.5: tier = 3
    elif points <= 4.5: tier = 4
    elif points <= 5.5: tier = 5
    elif points <= 6.5: tier = 6
    else: tier = 7

    # Size guidance
    size_map = {
        1: "No edge — đứng ngoài",
        2: "Edge yếu — quan sát / size nhỏ",
        3: "Trade được — 30–50% size",
        4: "Edge tốt — full size",
        5: "Edge mạnh — full size + có thể add",
        6: "Hiếm — có thể overweight có kiểm soát",
        7: "God-tier — ưu tiên cao nhất, quản trị rủi ro chặt"
    }
    size_guidance = size_map.get(tier, "N/A")
    # --------------------------
    # STOCK TRAITS (5Y OHLCV) — Composite Scores (0–10)
    # Goal: improve class assignment without touching Report A–D logic.
    # --------------------------
    # Notes:
    # - Scores are designed to be stable over time and reflect "stock character".
    # - If OHLCV is missing/insufficient, traits fall back to neutral (5.0).
    def _get_series(col_names):
        # Case-insensitive lookup
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype=float)
        cols = {str(c).strip().lower(): c for c in df.columns}
        for n in col_names:
            key = str(n).strip().lower()
            if key in cols:
                try:
                    return pd.to_numeric(df[cols[key]], errors="coerce")
                except Exception:
                    return pd.Series(dtype=float)
        return pd.Series(dtype=float)

    o = _get_series(["Open", "O"])
    h = _get_series(["High", "H"])
    l = _get_series(["Low", "L"])
    c = _get_series(["Close", "C", "Adj Close", "AdjClose"])
    v = _get_series(["Volume", "Vol", "V"])

    # Ensure consistent length and recent window (up to ~5y daily)
    _n = int(min(len(c), 1260)) if hasattr(c, "__len__") else 0
    o = o.tail(_n) if _n else o
    h = h.tail(_n) if _n else h
    l = l.tail(_n) if _n else l
    c = c.tail(_n) if _n else c
    v = v.tail(_n) if _n else v

    def _roll_median(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).median()

    def _roll_mean(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).mean()

    def _roll_std(s, w):
        return s.rolling(int(w), min_periods=max(3, int(w)//3)).std(ddof=0)

    def _atr14(_h, _l, _c):
        prev_c = _c.shift(1)
        tr = pd.concat([(_h - _l).abs(), (_h - prev_c).abs(), (_l - prev_c).abs()], axis=1).max(axis=1)
        return _roll_mean(tr, 14), tr

    def _adx14(_h, _l, _c):
        # Wilder's ADX14 (robust, simplified smoothing)
        up = _h.diff()
        dn = -_l.diff()
        dm_p = up.where((up > dn) & (up > 0), 0.0)
        dm_m = dn.where((dn > up) & (dn > 0), 0.0)
        atr, tr = _atr14(_h, _l, _c)
        # Wilder smoothing (EMA with alpha=1/14)
        alpha = 1.0 / 14.0
        tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        dm_p_s = dm_p.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        dm_m_s = dm_m.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        di_p = 100.0 * (dm_p_s / tr_s).replace([np.inf, -np.inf], np.nan)
        di_m = 100.0 * (dm_m_s / tr_s).replace([np.inf, -np.inf], np.nan)
        dx = (100.0 * (di_p - di_m).abs() / (di_p + di_m)).replace([np.inf, -np.inf], np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
        return adx

    def _percentile_rank(hist: pd.Series, x: float) -> float:
        # Robust percentile rank of x within hist (0..1)
        try:
            s = pd.to_numeric(hist, errors="coerce").dropna()
            if len(s) < 30 or (x is None) or (not np.isfinite(float(x))):
                return np.nan
            x = float(x)
            less = float((s < x).sum())
            eq = float((s == x).sum())
            return (less + 0.5 * eq) / float(len(s))
        except Exception:
            return np.nan

    def _score_from_bins(x, bins, scores, default=5.0):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        try:
            return float(_bucket_score(float(x), bins=bins, scores=scores))
        except Exception:
            return float(default)

    # ===== Trend Integrity (TI) =====
    if _n >= 260 and pd.notna(c.iloc[-1]):
        ma20_s = c.rolling(20, min_periods=20).mean()
        ma50_s = c.rolling(50, min_periods=50).mean()
        ma200_s = c.rolling(200, min_periods=200).mean()

        # PctAboveMA200 over 1y and 2y (if available)
        w1 = int(min(252, _n))
        w2 = int(min(504, _n))
        pct_above_1y = float((c.tail(w1) > ma200_s.tail(w1)).mean()) if w1 >= 50 else np.nan
        pct_above_2y = float((c.tail(w2) > ma200_s.tail(w2)).mean()) if w2 >= 100 else np.nan
        pct_above = np.nanmean([pct_above_1y, pct_above_2y])

        # MA stack consistency (bull or bear)
        stack_bull = (ma20_s > ma50_s) & (ma50_s > ma200_s)
        stack_bear = (ma20_s < ma50_s) & (ma50_s < ma200_s)
        stack_cons = float((stack_bull.tail(w1) | stack_bear.tail(w1)).mean()) if w1 >= 50 else np.nan

        # Flip rate (churn) on MA200 side + MA20/MA50 cross
        side200 = np.sign((c - ma200_s).dropna())
        flips200 = float((side200.tail(w1).diff().fillna(0) != 0).sum()) if len(side200.tail(w1)) > 5 else np.nan
        spread2050 = (ma20_s - ma50_s).dropna()
        flips2050 = float((np.sign(spread2050.tail(w1)).diff().fillna(0) != 0).sum()) if len(spread2050.tail(w1)) > 5 else np.nan
        flip_rate = np.nanmean([flips200, flips2050]) * (252.0 / max(1.0, float(w1)))

        # ADX strength (median last ~6 months)
        adx = _adx14(h, l, c)
        adx_med = float(adx.dropna().tail(126).median()) if len(adx.dropna()) >= 30 else np.nan
    else:
        pct_above = stack_cons = flip_rate = adx_med = np.nan

    ti_pct_score = _score_from_bins(pct_above, bins=[0.35, 0.50, 0.65, 0.80], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    ti_stack_score = _score_from_bins(stack_cons, bins=[0.15, 0.30, 0.45, 0.60], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    ti_flip_score = _score_from_bins(flip_rate, bins=[3, 7, 12, 18], scores=[10.0, 8.0, 6.0, 4.0, 2.0])  # lower flip is better
    ti_adx_score = _score_from_bins(adx_med, bins=[15, 20, 25, 35], scores=[3.0, 5.0, 7.0, 9.0, 10.0])
    trend_integrity = _clip(0.30*ti_pct_score + 0.20*ti_stack_score + 0.30*ti_flip_score + 0.20*ti_adx_score, 0, 10)

    # ===== Volatility Structure (VS) — we use VolRisk (0–10) =====
    if _n >= 60 and pd.notna(c.iloc[-1]):
        r = np.log(c).diff()
        rv20 = float(_roll_std(r, 20).iloc[-1] * np.sqrt(252.0) * 100.0) if len(r.dropna()) >= 25 else np.nan
        rv60 = float(_roll_std(r, 60).iloc[-1] * np.sqrt(252.0) * 100.0) if len(r.dropna()) >= 65 else np.nan
        rv = np.nanmean([rv20, rv60])

        atr, tr = _atr14(h, l, c)
        atr_pct = float((atr.iloc[-1] / c.iloc[-1]) * 100.0) if pd.notna(atr.iloc[-1]) and c.iloc[-1] != 0 else np.nan

        rv20_series = _roll_std(r, 20) * np.sqrt(252.0) * 100.0
        vol_of_vol = float(_roll_std(rv20_series, 252).iloc[-1]) if len(rv20_series.dropna()) >= 300 else np.nan

        tr_med20 = _roll_median(tr, 20)
        exp_rate = float((tr.tail(60) > tr_med20.tail(60)).mean()) if len(tr.dropna()) >= 80 else np.nan
    else:
        rv = atr_pct = vol_of_vol = exp_rate = np.nan

    # Convert to "risk score" where higher = more volatile / harder to control
    vs_atr_risk = _score_from_bins(atr_pct, bins=[1.2, 2.0, 3.0, 4.5], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_rv_risk  = _score_from_bins(rv,      bins=[18, 25, 35, 50],    scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_vov_risk = _score_from_bins(vol_of_vol, bins=[4, 7, 12, 18],   scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vs_exp_risk = _score_from_bins(exp_rate, bins=[0.35, 0.45, 0.55, 0.65], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    vol_risk = _clip(0.30*vs_atr_risk + 0.25*vs_rv_risk + 0.25*vs_vov_risk + 0.20*vs_exp_risk, 0, 10)

    # ===== Tail & Gap Risk (TGR) — higher = worse =====
    if _n >= 260 and pd.notna(c.iloc[-1]):
        ret = c.pct_change()
        ret_1y = ret.tail(252)
        mu = float(ret_1y.mean()) if len(ret_1y.dropna()) >= 50 else np.nan
        sd = float(ret_1y.std(ddof=0)) if len(ret_1y.dropna()) >= 50 else np.nan
        thr = (mu - 2.0*sd) if (np.isfinite(mu) and np.isfinite(sd)) else np.nan
        left_tail_freq = float((ret_1y < thr).mean()) if np.isfinite(thr) else np.nan

        # ES 5% (absolute magnitude)
        q05 = float(ret_1y.quantile(0.05)) if len(ret_1y.dropna()) >= 50 else np.nan
        es5 = float(ret_1y[ret_1y <= q05].mean()) if np.isfinite(q05) and (ret_1y <= q05).sum() >= 5 else np.nan
        es5_abs = abs(es5) * 100.0 if np.isfinite(es5) else np.nan

        prev_c = c.shift(1)
        gap_pct = (o - prev_c).abs() / prev_c
        gap_freq = float((gap_pct.tail(252) > 0.015).mean()) if len(gap_pct.dropna()) >= 80 else np.nan

        # Crash clusters: count sequences of >=2 consecutive days with ret <= -2%
        crash = (ret <= -0.02).astype(int)
        crash_1y = crash.tail(252).fillna(0).to_numpy()
        clusters = 0
        run = 0
        for x in crash_1y:
            if x == 1:
                run += 1
            else:
                if run >= 2:
                    clusters += 1
                run = 0
        if run >= 2:
            clusters += 1
        crash_clusters = float(clusters) * (252.0 / 252.0)
    else:
        left_tail_freq = es5_abs = gap_freq = crash_clusters = np.nan

    tgr_tail = _score_from_bins(left_tail_freq, bins=[0.01, 0.025, 0.05, 0.08], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_es   = _score_from_bins(es5_abs,       bins=[2.0, 3.5, 5.0, 7.0],     scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_gap  = _score_from_bins(gap_freq,      bins=[0.01, 0.03, 0.06, 0.10],  scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tgr_clu  = _score_from_bins(crash_clusters,bins=[1, 2, 4, 6],             scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    tail_risk = _clip(0.35*tgr_tail + 0.30*tgr_es + 0.20*tgr_gap + 0.15*tgr_clu, 0, 10)

    # ===== Mean-Reversion / Whipsaw Propensity (MRW) — higher = more range/whipsaw =====
    def _variance_ratio(_c, k=5, w=252):
        _c = pd.to_numeric(_c, errors="coerce")
        r1 = _c.pct_change()
        rk = _c.pct_change(k)
        r1w = r1.tail(w).dropna()
        rkw = rk.tail(w).dropna()
        if len(r1w) < max(30, k*6) or len(rkw) < max(30, k*6):
            return np.nan
        v1 = float(r1w.var(ddof=0))
        vk = float(rkw.var(ddof=0))
        if v1 <= 0:
            return np.nan
        return vk / (k * v1)

    def _half_life_spread(spread, w=252):
        s = pd.to_numeric(spread, errors="coerce").dropna().tail(w)
        if len(s) < 60:
            return np.nan
        x = s.shift(1).dropna()
        y = s.loc[x.index]
        if len(x) < 50:
            return np.nan
        # OLS slope b
        b = np.polyfit(x.values, y.values, 1)[0]
        if b <= 0 or b >= 0.999:
            return np.nan
        return float(-np.log(2.0) / np.log(b))

    if _n >= 300 and pd.notna(c.iloc[-1]):
        vr5 = _variance_ratio(c, k=5, w=504)
        vr10 = _variance_ratio(c, k=10, w=504)
        vr = np.nanmean([vr5, vr10])

        r = c.pct_change()
        ac_list = []
        for lag in [1,2,3,4,5]:
            ac = float(r.tail(504).autocorr(lag=lag)) if len(r.dropna()) >= 200 else np.nan
            if np.isfinite(ac):
                ac_list.append(ac)
        ac_mean = float(np.nanmean(ac_list)) if len(ac_list) else np.nan

        atr, tr = _atr14(h, l, c)
        spread = (c - c.rolling(20, min_periods=20).mean()) / (atr.replace(0, np.nan))
        hl = _half_life_spread(spread, w=504)

        ma20_s = c.rolling(20, min_periods=20).mean()
        ma50_s = c.rolling(50, min_periods=50).mean()
        cross = ((ma20_s - ma50_s).apply(np.sign)).diff().fillna(0)
        cross_churn = float((cross.tail(252) != 0).sum())
    else:
        vr = ac_mean = hl = cross_churn = np.nan

    mr_vr = _score_from_bins(vr, bins=[0.85, 0.95, 1.05, 1.15], scores=[10.0, 8.0, 5.0, 3.0, 1.0])  # lower VR => more mean-reversion
    mr_ac = _score_from_bins(ac_mean, bins=[-0.08, -0.03, 0.03, 0.08], scores=[10.0, 8.0, 5.0, 3.0, 1.0])  # negative autocorr => mean-reversion
    mr_hl = _score_from_bins(hl, bins=[5, 10, 20, 40], scores=[10.0, 8.0, 6.0, 4.0, 2.0])  # smaller half-life => faster reversion
    mr_ch = _score_from_bins(cross_churn, bins=[2, 5, 10, 18], scores=[2.0, 4.0, 6.0, 8.0, 10.0])  # more churn => more whipsaw
    meanrev_prop = _clip(0.30*mr_vr + 0.20*mr_ac + 0.20*mr_hl + 0.30*mr_ch, 0, 10)

    # ===== Breakout Quality (BQ) — higher = better follow-through =====
    breakout_quality = 5.0
    bq_conf = 0.0
    ft_rate = fb_rate = retest_rate = np.nan
    if _n >= 300 and pd.notna(c.iloc[-1]) and len(v.dropna()) >= 80:
        atr, tr = _atr14(h, l, c)
        hh55 = h.rolling(55, min_periods=55).max().shift(1)
        level = hh55
        # Breakout day: close > HH55 and prior close <= HH55
        bday = (c > hh55) & (c.shift(1) <= hh55)
        vol_med20 = v.rolling(20, min_periods=20).median().shift(1)
        vconf = v > (vol_med20 * 1.2)
        events = (bday & vconf).tail(252)
        idxs = list(np.where(events.fillna(False).to_numpy())[0])
        # Map idxs to absolute indices in tail window
        base = int(len(c.tail(252)) - len(events))  # usually 0
        # We'll iterate using positions within the tail(252) window for simplicity
        c252 = c.tail(252).reset_index(drop=True)
        h252 = h.tail(252).reset_index(drop=True)
        l252 = l.tail(252).reset_index(drop=True)
        lvl252 = level.tail(252).reset_index(drop=True)
        atr252 = atr.tail(252).reset_index(drop=True)
        evpos = np.where((bday & vconf).tail(252).fillna(False).to_numpy())[0].tolist()

        if len(evpos) >= 3:
            ft = fb = rt = 0
            for p in evpos:
                lvl = float(lvl252.iloc[p]) if pd.notna(lvl252.iloc[p]) else np.nan
                if not np.isfinite(lvl):
                    continue
                # False break: within next 5 days, close falls back below level
                fb_win = c252.iloc[p+1:min(p+6, len(c252))]
                if len(fb_win) > 0 and (fb_win < lvl).any():
                    fb += 1
                # Follow-through: within next 10 days, (a) stays above level for most days OR (b) reaches +1 ATR from breakout close
                ft_win = c252.iloc[p+1:min(p+11, len(c252))]
                h_win = h252.iloc[p+1:min(p+11, len(h252))]
                atr_p = float(atr252.iloc[p]) if pd.notna(atr252.iloc[p]) else np.nan
                c_p = float(c252.iloc[p]) if pd.notna(c252.iloc[p]) else np.nan
                cond_a = (len(ft_win) >= 5 and (ft_win > lvl).mean() >= 0.70)
                cond_b = (np.isfinite(atr_p) and np.isfinite(c_p) and len(h_win) > 0 and (h_win.max() >= c_p + atr_p))
                if cond_a or cond_b:
                    ft += 1
                # Retest success: touches near level then closes meaningfully above within a few days
                lo_win = l252.iloc[p+1:min(p+11, len(l252))]
                if len(lo_win) > 0 and (lo_win <= lvl * 1.005).any():
                    # after first touch, require a close > lvl*1.01 within next 5 days
                    touch_idx = int(np.where((lo_win <= lvl * 1.005).to_numpy())[0][0]) + (p+1)
                    rec_win = c252.iloc[touch_idx:min(touch_idx+6, len(c252))]
                    if len(rec_win) > 0 and (rec_win >= lvl * 1.01).any():
                        rt += 1
            n_ev = max(1, len(evpos))
            ft_rate = ft / n_ev
            fb_rate = fb / n_ev
            retest_rate = rt / n_ev
            breakout_quality = _clip((0.50*ft_rate + 0.30*retest_rate + 0.20*(1.0 - fb_rate)) * 10.0, 0, 10)
            bq_conf = 1.0
        else:
            # Not enough events — neutral score, low confidence
            breakout_quality = 5.0
            bq_conf = 0.3

    # ===== Liquidity & Tradability (LT) — higher = more tradable =====
    liq_tradability = 5.0
    dv20 = amihud20 = vol_cv20 = np.nan
    dv_score = ami_score = cv_score = 5.0
    if _n >= 120 and pd.notna(c.iloc[-1]) and len(v.dropna()) >= 60:
        dollar_vol = (c * v).replace([np.inf, -np.inf], np.nan)
        dv20_s = dollar_vol.rolling(20, min_periods=20).median()
        dv20 = float(dv20_s.iloc[-1]) if pd.notna(dv20_s.iloc[-1]) else np.nan
        dv_pct = _percentile_rank(dv20_s.dropna(), dv20)
        dv_score = _clip((dv_pct * 10.0) if np.isfinite(dv_pct) else 5.0, 0, 10)

        ret = c.pct_change()
        amihud = (ret.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
        amihud_s = amihud.rolling(20, min_periods=20).mean()
        amihud20 = float(amihud_s.iloc[-1]) if pd.notna(amihud_s.iloc[-1]) else np.nan
        ami_pct = _percentile_rank(amihud_s.dropna(), amihud20)
        ami_score = _clip(((1.0 - ami_pct) * 10.0) if np.isfinite(ami_pct) else 5.0, 0, 10)

        vol_cv = (v.rolling(20, min_periods=20).std(ddof=0) / v.rolling(20, min_periods=20).mean()).replace([np.inf, -np.inf], np.nan)
        vol_cv20 = float(vol_cv.iloc[-1]) if pd.notna(vol_cv.iloc[-1]) else np.nan
        cv_pct = _percentile_rank(vol_cv.dropna(), vol_cv20)
        cv_score = _clip(((1.0 - cv_pct) * 10.0) if np.isfinite(cv_pct) else 5.0, 0, 10)

        liq_tradability = _clip(0.50*dv_score + 0.30*ami_score + 0.20*cv_score, 0, 10)

    # ===== Drawdown & Recovery (DR) — higher = worse (riskier) =====
    mdd_abs = dd_freq = rec_days = np.nan
    mdd_risk = dd_freq_risk = rec_risk = 5.0
    if _n >= 260 and len(c.dropna()) >= 260:
        c2 = c.dropna()
        roll_max = c2.cummax()
        dd = (c2 / roll_max) - 1.0
        mdd = float(dd.min()) if len(dd) else np.nan
        mdd_abs = abs(mdd) * 100.0 if np.isfinite(mdd) else np.nan
    
        # Count drawdown "episodes" deeper than -10%
        thresh = -0.10
        below = (dd <= thresh).astype(int)
        dd_events = int(((below.diff() == 1).sum())) if len(below) > 1 else 0
        years = max(1.0, float(len(dd)) / 252.0)
        dd_freq = float(dd_events) / years
    
        # Recovery speed: median days from DD episode start back to prior peak
        rec_days_list: List[int] = []
        in_dd = False
        start_i = None
        peak_val = None
        vals = c2.values
        peaks = roll_max.values
        for i in range(len(vals)):
            ddv = float(dd.iloc[i])
            if (not in_dd) and (ddv <= thresh):
                in_dd = True
                start_i = i
                peak_val = float(peaks[i])
            if in_dd and peak_val and (vals[i] >= peak_val * 0.999):
                rec_days_list.append(int(i - (start_i or 0)))
                in_dd = False
                start_i = None
                peak_val = None
        if rec_days_list:
            rec_days = float(np.median(rec_days_list))
    
        mdd_risk = _score_from_bins(mdd_abs, bins=[12, 20, 30, 45], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
        dd_freq_risk = _score_from_bins(dd_freq, bins=[0.5, 1.2, 2.5, 4.0], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
        rec_risk = _score_from_bins(rec_days, bins=[20, 45, 90, 150], scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    
    # Autocorrelation (lag-1) — momentum vs mean-reversion tendency (higher = more momentum)
    autocorr1 = np.nan
    autocorr_score = 5.0
    try:
        _ret = c.pct_change().dropna()
        if len(_ret) >= 260:
            autocorr1 = float(_ret.tail(756).autocorr(lag=1))
            autocorr_score = _score_from_bins(autocorr1, bins=[-0.15, -0.05, 0.05, 0.15],
                                              scores=[2.0, 4.0, 6.0, 8.0, 10.0])
    except Exception:
        pass


    stock_traits = {
        "TrendIntegrity": float(trend_integrity),
        "VolRisk": float(vol_risk),
        "TailGapRisk": float(tail_risk),
        "MeanReversionWhipsaw": float(meanrev_prop),
        "BreakoutQuality": float(breakout_quality),
        "LiquidityTradability": float(liq_tradability),
        "Confidence": {
            "BreakoutQuality": float(bq_conf)
        },
        "Raw": {
            "PctAboveMA200": float(pct_above) if np.isfinite(pct_above) else np.nan,
            "MAStackConsistency": float(stack_cons) if np.isfinite(stack_cons) else np.nan,
            "TrendFlipRatePerYear": float(flip_rate) if np.isfinite(flip_rate) else np.nan,
            "ADXMedian": float(adx_med) if np.isfinite(adx_med) else np.nan,
            "ATRpct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
            "RealizedVol": float(rv) if np.isfinite(rv) else np.nan,
            "VolOfVol": float(vol_of_vol) if np.isfinite(vol_of_vol) else np.nan,
            "RangeExpansionRate": float(exp_rate) if np.isfinite(exp_rate) else np.nan,
            "LeftTailFreq": float(left_tail_freq) if np.isfinite(left_tail_freq) else np.nan,
            "ES5AbsPct": float(es5_abs) if np.isfinite(es5_abs) else np.nan,
            "GapFreq": float(gap_freq) if np.isfinite(gap_freq) else np.nan,
            "CrashClusters": float(crash_clusters) if np.isfinite(crash_clusters) else np.nan,
            "VarianceRatio": float(vr) if np.isfinite(vr) else np.nan,
            "AutocorrMean": float(ac_mean) if np.isfinite(ac_mean) else np.nan,
            "HalfLife": float(hl) if np.isfinite(hl) else np.nan,
            "CrossChurn": float(cross_churn) if np.isfinite(cross_churn) else np.nan,
            "FollowThroughRate": float(ft_rate) if np.isfinite(ft_rate) else np.nan,
            "FalseBreakRate": float(fb_rate) if np.isfinite(fb_rate) else np.nan,
            "RetestSuccessRate": float(retest_rate) if np.isfinite(retest_rate) else np.nan,
            "DollarVolMedian20": float(dv20) if np.isfinite(dv20) else np.nan,
            "Amihud20": float(amihud20) if np.isfinite(amihud20) else np.nan,
            "VolumeCV20": float(vol_cv20) if np.isfinite(vol_cv20) else np.nan
        }
    }

    # Adjust existing stats using traits (bounded, non-invasive)
    trend_adj = _clip(trend + ((trend_integrity - 5.0) / 5.0) * 1.2, 0, 10)
    momentum_adj = _clip(momentum + ((breakout_quality - 5.0) / 5.0) * 0.8 - ((meanrev_prop - 5.0) / 5.0) * 0.5, 0, 10)
    stability_adj = _clip(stability - ((vol_risk - 5.0) / 5.0) * 1.2 - ((tail_risk - 5.0) / 5.0) * 1.0 - ((meanrev_prop - 5.0) / 5.0) * 0.6, 0, 10)
    reliability_adj = _clip(reliability + ((trend_integrity - 5.0) / 5.0) * 1.0 + ((breakout_quality - 5.0) / 5.0) * 0.8 + ((liq_tradability - 5.0) / 5.0) * 0.6
                           - ((tail_risk - 5.0) / 5.0) * 1.0 - ((meanrev_prop - 5.0) / 5.0) * 0.7, 0, 10)

    adjusted_stats = {
        "TrendAdj": float(trend_adj),
        "MomentumAdj": float(momentum_adj),
        "StabilityAdj": float(stability_adj),
        "ReliabilityAdj": float(reliability_adj)
    }
    # Character class (2-tier DNA taxonomy; stable, long-run oriented)
    # Tier-1: StyleAxis ∈ {"Trend","Momentum","Range","Hybrid"}; RiskRegime ∈ {"Low","Mid","High"}
    # Tier-2: 6–8 classes mapped via thresholds (designed to be stable, not "current snapshot").
    # Notes:
    # - Risk metrics (VolRisk/TailGapRisk/Drawdown*) are "higher = worse".
    # - Style metrics (TrendIntegrity/BreakoutQuality/MeanReversion*) define how the stock typically trades.
    def _tier1_style() -> str:
        # Momentum attempt dominates only when breakout quality + momentum are persistent.
        if (breakout_quality >= 6.8 and momentum_adj >= 6.5 and meanrev_prop <= 6.5):
            return "Momentum"
        if (meanrev_prop >= 6.8 or whipsaw or (autocorr1 is not None and np.isfinite(autocorr1) and autocorr1 <= -0.05)):
            return "Range"
        if (trend_integrity >= 6.7 and meanrev_prop <= 5.7 and (autocorr1 is None or (not np.isfinite(autocorr1)) or autocorr1 >= -0.02)):
            return "Trend"
        return "Hybrid"

    def _tier1_risk() -> Tuple[str, float]:
        risks = []
        for x in [vol_risk, tail_risk, mdd_risk, rec_risk]:
            try:
                xv = float(x)
                if np.isfinite(xv):
                    risks.append(xv)
            except Exception:
                pass
        rscore = float(np.mean(risks)) if risks else 5.0
        if rscore <= 4.5:
            return "Low", rscore
        if rscore >= 6.5:
            return "High", rscore
        return "Mid", rscore

    style_axis = _tier1_style()
    risk_regime, risk_score = _tier1_risk()

    # Tier-2 mapping (8 Classes) — STRICTLY long-run (3–5y) only
    # Modifiers (derived from long-run risks / tradability)
    liq_cons = float(liq_tradability) if np.isfinite(liq_tradability) else 5.0
    liq_level = float(dv_score) if np.isfinite(dv_score) else 5.0
    tail_r = float(tail_risk) if np.isfinite(tail_risk) else 5.0
    vol_r = float(vol_risk) if np.isfinite(vol_risk) else 5.0
    dd_r = float(mdd_risk) if np.isfinite(mdd_risk) else 5.0
    vov_r = float(vs_vov_risk) if np.isfinite(vs_vov_risk) else 5.0

    # Optional raw gap frequency (if available)
    gap_f = stock_traits.get("Raw", {}).get("GapFreq", np.nan)
    try:
        gap_f = float(gap_f)
    except Exception:
        gap_f = np.nan

    modifiers: List[str] = []
    if liq_cons <= 3.0 or liq_level <= 3.0:
        modifiers.append("ILLIQ")
    if (tail_r >= 7.5) or (np.isfinite(gap_f) and gap_f >= 0.08):
        modifiers.append("GAP")
    if (vol_r >= 7.2) or (dd_r >= 7.2):
        modifiers.append("HIVOL")
    if vov_r >= 7.0:
        modifiers.append("CHOPVOL")
    if (vol_r <= 3.8 and dd_r <= 5.2 and tail_r <= 6.0 and liq_tradability >= 6.0):
        modifiers.append("DEF")

    # Priority: execution risk first
    if "ILLIQ" in modifiers:
        cclass = "Illiquid / Noisy"
    elif "GAP" in modifiers:
        cclass = "Event / Gap-Prone"
    elif style_axis == "Trend":
        cclass = "Aggressive Trend" if ("HIVOL" in modifiers or risk_regime == "High") else "Smooth Trend"
    elif style_axis == "Momentum":
        cclass = "Momentum Trend"
    elif style_axis == "Range":
        cclass = "Volatile Range" if ("HIVOL" in modifiers or "CHOPVOL" in modifiers or risk_regime == "High") else "Range / Mean-Reversion (Stable)"
    else:
        cclass = "Mixed / Choppy Trader"


# Enrich StockTraits with stable 2-tier DNA taxonomy + 15-parameter pack (for Python tagging & UI)
    try:
        stock_traits.setdefault("DNA", {})
        # DNAConfidence / ClassLockFlag — stability proxy for the long-run DNA label (0–100)
        try:
            n_bars = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        except Exception:
            n_bars = 0

        vol_stability = 10.0 - float(vs_vov_risk) if np.isfinite(vs_vov_risk) else 5.0  # higher = more stable
        liq_cons_score = float(cv_score) if np.isfinite(cv_score) else 5.0
        liq_level_score = float(dv_score) if np.isfinite(dv_score) else 5.0

        dna_confidence = 60.0
        dna_confidence += 10.0 if n_bars >= 900 else (5.0 if n_bars >= 600 else 0.0)
        dna_confidence += 10.0 if vol_stability >= 6.0 else 0.0
        dna_confidence -= 15.0 if (np.isfinite(tail_risk) and float(tail_risk) >= 8.0) else 0.0
        dna_confidence -= 10.0 if (np.isfinite(mdd_risk) and float(mdd_risk) >= 8.0) else 0.0
        dna_confidence -= 15.0 if (liq_cons_score <= 3.0 or liq_level_score <= 3.0) else 0.0
        dna_confidence = float(max(0.0, min(100.0, dna_confidence)))
        class_lock = bool(dna_confidence < 50.0)


        def _pick_primary_modifier(mods: List[str]) -> str:
            for mm in ["ILLIQ","GAP","HIVOL","CHOPVOL","HBETA","DEF"]:
                if mm in mods:
                    return mm
            return ""

        primary_mod = _pick_primary_modifier(modifiers if isinstance(modifiers, list) else [])
        stock_traits["DNA"]["Tier1"] = {
            "StyleAxis": str(style_axis),
            "RiskRegime": str(risk_regime),
            "RiskScore": float(risk_score) if np.isfinite(risk_score) else np.nan,
            # DNA confidence is a stability/coverage proxy (0–100) — strictly long-run inputs only
            "DNAConfidence": float(dna_confidence),
            "ClassLockFlag": bool(class_lock),
            "Modifiers": list(modifiers) if isinstance(modifiers, list) else [],
            "PrimaryModifier": str(primary_mod),
        }
        stock_traits["DNA"]["Params"] = {
            # Group 1: Trend Structure (higher = better)
            "TrendIntegrity": float(trend_integrity),
            "TrendPersistence": float(ti_pct_score),
            "TrendChurnControl": float(ti_flip_score),
    
            # Group 2: Volatility & Tail (higher = worse)
            "VolRisk": float(vol_risk),
            "TailGapRisk": float(tail_risk),
            "VolOfVolRisk": float(vs_vov_risk),
    
            # Group 3: Drawdown & Recovery (higher = worse)
            "MaxDrawdownRisk": float(mdd_risk),
            "RecoverySlownessRisk": float(rec_risk),
            "DrawdownFrequencyRisk": float(dd_freq_risk),
    
            # Group 4: Liquidity & Tradability (higher = better)
            "LiquidityTradability": float(liq_tradability),
            "LiquidityLevel": float(dv_score),
            "LiquidityConsistency": float(cv_score),
    
            # Group 5: Behavior / Setup Bias
            "BreakoutQuality": float(breakout_quality),
            "MeanReversionWhipsaw": float(meanrev_prop),
            "AutoCorrMomentum": float(autocorr_score),
        }
        stock_traits["DNA"]["Groups"] = {
            "TrendStructure": ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"],
            "VolatilityTail": ["VolRisk", "TailGapRisk", "VolOfVolRisk"],
            "DrawdownRecovery": ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"],
            "LiquidityTradability": ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"],
            "BehaviorSetup": ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"],
        }
        # Store extra raw metrics (optional diagnostics)
        stock_traits.setdefault("Raw", {})
        stock_traits["Raw"].update({
            "RealizedVolPct": float(rv) if np.isfinite(rv) else np.nan,
            "ATRpct": float(atr_pct) if np.isfinite(atr_pct) else np.nan,
            "VolOfVol": float(vol_of_vol) if np.isfinite(vol_of_vol) else np.nan,
            "TRExpansionRate": float(exp_rate) if np.isfinite(exp_rate) else np.nan,
            "LeftTailFreq": float(left_tail_freq) if np.isfinite(left_tail_freq) else np.nan,
            "ES5AbsPct": float(es5_abs) if np.isfinite(es5_abs) else np.nan,
            "GapFreq": float(gap_freq) if np.isfinite(gap_freq) else np.nan,
            "MaxDrawdownAbsPct": float(mdd_abs) if np.isfinite(mdd_abs) else np.nan,
            "DrawdownEpisodesPerYear": float(dd_freq) if np.isfinite(dd_freq) else np.nan,
            "RecoveryDaysMedian": float(rec_days) if np.isfinite(rec_days) else np.nan,
            "AutoCorr1": float(autocorr1) if np.isfinite(autocorr1) else np.nan,
        })
    except Exception:
        # Hard-fail is not allowed; keep backward compatibility.
        pass

    # Action tags (lightweight, for GPT/UI)
    tags = []
    if tier >= 4 and (pd.notna(confluence_count) and confluence_count >= 3):
        tags.append("Pullback-buy zone (confluence)")
    if breakout_force >= 7:
        tags.append("Breakout attempt (needs follow-through)")
    if any(f["code"] == "NoVolumeConfirm" for f in flags):
        tags.append("Wait for volume confirmation")
    if any(f["code"] in ("NearMajorResistance", "Overheated") for f in flags):
        tags.append("Tight risk control near resistance")
    if fib_conflict:
        tags.append("Use LongStructure_ShortTactical rule")


    # --- v5.9.9: ensure ATR is scalar-safe (avoid pandas truthiness) ---
    atr_scalar = _as_scalar(atr)
    atr_f = None
    try:
        if atr_scalar is not None and atr_scalar == atr_scalar:
            atr_f = float(atr_scalar)
    except Exception:
        atr_f = None
    atr_pos = (atr_f is not None and atr_f > 0)

    return {
        "CharacterClass": cclass,
        "CoreStats": core_stats,
        "AdjustedStats": adjusted_stats,
        "StockTraits": stock_traits,
        "CombatStats": combat_stats,
        "StructureQuality": struct_q,
        "Flags": flags,
        "Conviction": {"Points": points, "Tier": tier, "SizeGuidance": size_guidance},
        "ActionTags": tags,
        "Meta": {
            "DenomUsed": "ATR14" if atr_pos else "VolProxy",
            "ConfidenceTech": ps.get("Confidence (Tech)", ps.get("Probability", "N/A")),
            "UpsideQualityMult": float(total_mult) if pd.notna(total_mult) else np.nan,
            "LevelCtxSource": lvl_source,
            "LevelCtxKeys": ",".join(list(lvl.keys())[:8]) if isinstance(lvl, dict) else "",
            "Close": float(close) if pd.notna(close) else np.nan,
            "NearestRes": float(nearest_res) if pd.notna(nearest_res) else np.nan,
            "NearestSup": float(nearest_sup) if pd.notna(nearest_sup) else np.nan,
            "UpsideRaw": float(upside) if pd.notna(upside) else np.nan,
            "DownsideRaw": float(downside) if pd.notna(downside) else np.nan,
            "ATR14": atr_f if (atr_f is not None) else np.nan,
            "VolProxy": float(vol_proxy) if pd.notna(vol_proxy) else np.nan,
            "UpsideNorm": float(upside_n) if pd.notna(upside_n) else np.nan,
            "DownsideNorm": float(downside_n) if pd.notna(downside_n) else np.nan,
            "RR": float(rr) if pd.notna(rr) else np.nan
        }
    }

def _character_blurb_fallback(ticker: str, cclass: str) -> str:
    """
    Deterministic fallback when GPT narrative is disabled.
    Must contain no numbers. Keep short, stable, and aligned with 2-tier DNA taxonomy.
    """
    name = (ticker or "").upper().strip() or "Cổ phiếu"
    cc = (cclass or "N/A").strip()

    # Use dashboard template if available
    lines = CLASS_TEMPLATES_DASHBOARD.get(cc) or []
    if lines:
        # Convert 3-line dashboard template into a compact paragraph
        return f"{name}: {lines[0]} {lines[1]} {lines[2]}"

    # Generic fallback (class unknown)
    return (f"{name} đang được gắn nhãn DNA '{cc}'. Hãy ưu tiên bám cấu trúc giá, "
            f"chỉ triển khai khi trade plan có điều kiện rõ ràng và tuân thủ kỷ luật quản trị rủi ro.")

def get_character_blurb(ticker: str, cclass: str) -> str:
    # GPT paragraph: 100–200 words, no numbers
    cache_key = f"_gc_blurb::{(ticker or '').upper().strip()}::{(cclass or '').strip()}"
    if cache_key in st.session_state:
        return st.session_state.get(cache_key) or ""
    base = _character_blurb_fallback(ticker, cclass)
    try:
        prompt = f"""Bạn là chuyên gia tài chính. Hãy viết một đoạn ngắn tiếng Việt (khoảng 100–200 từ),
văn phong chuyên nghiệp, dễ hiểu. Tuyệt đối KHÔNG nhắc bất kỳ con số nào (không số điểm, không phần trăm, không mốc giá,
không số phiên, không ký hiệu số). Không liệt kê chỉ báo/thuật ngữ theo dạng báo cáo. Hãy mô tả:
- Cổ phiếu {ticker.upper().strip()} thuộc nhóm (class) {cclass}.
- Bản chất hành vi giá thường gặp của nhóm này.
- Phù hợp với kiểu trader/trường phái nào và không phù hợp với kiểu nào.
- Nêu một ví dụ ngắn về hành vi thường gặp (ví dụ: dao động trong biên, bật ở hỗ trợ, thất bại khi vượt cản…).
Kết thúc bằng một câu định hướng hành động theo phong cách quản trị rủi ro.
"""
        txt = _call_openai(prompt, temperature=0.5)
        txt = (txt or "").strip()
        # Safety: remove digits if model violates rule
        txt = re.sub(r"\d", "", txt)
        if len(txt) < 40:
            txt = base
    except Exception:
        txt = base
    st.session_state[cache_key] = txt
    return txt

def render_character_card(character_pack: Dict[str, Any]) -> None:
    """
    Streamlit rendering for Character Card.
    Does not affect existing report A–D.
    """
    cp = character_pack or {}
    core = cp.get("CoreStats") or {}
    combat = cp.get("CombatStats") or {}
    conv = cp.get("Conviction") or {}
    flags = cp.get("Flags") or []
    cclass = cp.get("CharacterClass") or "N/A"
    err = (cp.get("Error") or "")

    ticker = _safe_text(cp.get('_Ticker') or '').strip().upper()
    headline = f"{ticker} - {cclass}" if ticker else str(cclass)
    class_key = str(cclass).strip()
    dash_lines: List[str] = (CLASS_TEMPLATES_DASHBOARD.get(class_key) or []).copy()
    if not dash_lines:
        # Fallback: keep Dashboard readable even if class is unknown
        fallback = get_character_blurb(ticker, class_key)
        if fallback:
            dash_lines = [f"Đặc tính: {fallback}"]
        else:
            dash_lines = [f"Đặc tính: {class_key}"]

    def _fmt_bline(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if ":" in s:
            k, v = s.split(":", 1)
            return f'<div class="gc-bline"><b>{html.escape(k.strip())}:</b> {html.escape(v.strip())}</div>'
        return f'<div class="gc-bline">{html.escape(s)}</div>'

    blurb_html = "".join([_fmt_bline(x) for x in dash_lines if str(x).strip()])
    # Show runtime error (if CharacterPack fallback was used)
    if err:
        st.error(f"Character module error: {err}")
        tb = cp.get("Traceback")
        if tb:
            with st.expander("Character traceback (debug)"):
                st.code(str(tb))


    def _radar_svg(stats: List[Tuple[str, float]], maxv: float = 10.0, size: int = 220) -> str:
        """Return an inline SVG radar chart (0–maxv) for the Character Card."""
        n = len(stats)
        if n < 3:
            return ""
        cx = cy = size / 2.0
        r = size * 0.34
        angles = [(-math.pi / 2.0) + i * (2.0 * math.pi / n) for i in range(n)]

        def pt(angle: float, radius: float) -> Tuple[float, float]:
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        # Normalize values
        vals: List[float] = []
        for _, v in stats:
            vv = 0.0 if pd.isna(v) else float(v)
            vv = float(_clip(vv, 0.0, maxv))
            vals.append(vv)

        # Grid polygons
        grid_levels = [2, 4, 6, 8, 10]
        grid_polys = []
        for lv in grid_levels:
            rr = r * (lv / maxv)
            pts = [pt(a, rr) for a in angles]
            grid_polys.append(" ".join([f"{x:.1f},{y:.1f}" for x, y in pts]))

        # Axis endpoints
        axis_pts = [pt(a, r) for a in angles]

        # Data polygon
        data_pts = [pt(angles[i], r * (vals[i] / maxv)) for i in range(n)]
        data_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in data_pts])

        # Labels
        label_pts = [pt(a, r + 28) for a in angles]

        parts: List[str] = []
        parts.append(f'<svg class="gc-radar-svg" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">')
        # grid
        for poly in grid_polys:
            parts.append(f'<polygon points="{poly}" fill="none" stroke="#E5E7EB" stroke-width="1" />')
        # axes
        for (x, y) in axis_pts:
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" />')
        # data
        parts.append(f'<polygon points="{data_points}" fill="rgba(15,23,42,0.12)" stroke="#0F172A" stroke-width="2" />')
        # points
        for (x, y) in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#0F172A" />')
        # labels
        for i, (lx, ly) in enumerate(label_pts):
            lab = html.escape(str(stats[i][0]))
            # anchor by horizontal position
            anchor = "middle"
            if lx < cx - 10:
                anchor = "end"
            elif lx > cx + 10:
                anchor = "start"
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" font-weight="700" fill="#334155">{lab}</text>')
        parts.append('</svg>')
        return "".join(parts)


    st.markdown(
        f"""
        <div class="gc-card">
          <div class="gc-head">
            <div class="gc-h1">{html.escape(str(headline))}</div>
            <div class="gc-blurb">{blurb_html}</div>
          </div>
        """,
        unsafe_allow_html=True
    )

    
    # show CharacterPack error if present
    if cp.get("Error"):
        st.warning(f"Character module error: {cp.get('Error')}")

    # Dashboard Class Signature (Radar) — 5 long-run DNA anchors (no 'Now/Opportunity' metrics)
    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"]))
    vol_risk_g = _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"]))
    dd_risk_g = _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"]))
    liq_g = _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"]))
    beh_g = _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"]))

    # Convert risk groups into positive-direction scores for the dashboard radar (higher = better control)
    vol_stab = (10.0 - float(vol_risk_g)) if pd.notna(vol_risk_g) else np.nan
    dd_res = (10.0 - float(dd_risk_g)) if pd.notna(dd_risk_g) else np.nan

    radar_stats: List[Tuple[str, float]] = [
        ("Trend", trend_g),
        ("Vol-Stability", vol_stab),
        ("DD-Resilience", dd_res),
        ("Liquidity", liq_g),
        ("Behavior", beh_g),
    ]
    svg = _radar_svg(radar_stats, maxv=10.0, size=220)

    # Side metrics list (keep numbers traceable, do not decide here)
    _metrics_html_parts: List[str] = []
    for lab, val in radar_stats:
        vv = 0.0 if pd.isna(val) else float(val)
        vv = float(_clip(vv, 0.0, 10.0))
        _metrics_html_parts.append(
            f'<div class="gc-radar-item"><span class="gc-radar-lab">{html.escape(str(lab))}</span>'
            f'<span class="gc-radar-val">{vv:.1f}/10</span></div>'
        )
    metrics_html = "".join(_metrics_html_parts)

    st.markdown(
        f'''
        <div class="gc-sec">
          <div class="gc-sec-t">CLASS SIGNATURE</div>
          <div class="gc-radar-wrap">
            {svg}
            <div class="gc-radar-metrics">{metrics_html}</div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


    tier = conv.get("Tier", "N/A")
    pts = conv.get("Points", np.nan)
    guide = conv.get("SizeGuidance", "")
    st.markdown(
        f"""
        <div class="gc-sec">
          <div class="gc-sec-t">CONVICTION</div>
          <div class="gc-conv">
            <div class="gc-conv-tier">Tier: <b>{tier}</b> / 7</div>
            <div class="gc-conv-pts">Points: {pts:.1f}</div>
            <div class="gc-conv-guide">{guide}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if flags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">WEAKNESSES</div>', unsafe_allow_html=True)
        for f in flags[:8]:
            sev = int(f.get("severity", 1))
            note = f.get("note", "")
            code = f.get("code", "")
            st.markdown(
                f"""<div class="gc-flag"><span class="gc-sev">S{sev}</span><span class="gc-code">{code}</span><span class="gc-note">{note}</span></div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    tags = cp.get("ActionTags") or []
    if tags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">PLAYSTYLE TAGS</div>', unsafe_allow_html=True)
        st.markdown("<div class='gc-tags'>" + "".join([f"<span class='gc-tag'>{t}</span>" for t in tags[:8]]) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# 10.9 DECISION LAYER REPORT — OUTPUT ORDER ANTI-ANCHORING BIAS (v6.4)
# ============================================================



# ============================================================
# CLASS TEXT TEMPLATES (STOCK DNA)
# ============================================================

# Fixed 3-paragraph templates per class to keep text stable.
CLASS_TEMPLATES: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Nhóm này có xu hướng dài hạn tương đối rõ và bền. Giá thường bám cấu trúc trend (đỉnh–đáy nâng dần) và tôn trọng các vùng hỗ trợ động, nên hành vi ít bị nhiễu so với nhóm biến động cao.",
        "Nhà đầu tư phù hợp là người đánh theo xu hướng, ưu tiên mua ở nhịp điều chỉnh/pullback thay vì mua đuổi. Có thể nắm giữ trung hạn khi cấu trúc còn nguyên vẹn và chỉ gia tăng khi có xác nhận tiếp diễn.",
        "Điểm lưu ý là tránh phá kỷ luật khi thị trường nhiễu: chỉ giữ vị thế khi trend còn hợp lệ và luôn có mức dừng lỗ theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Nhóm này thiên về động lượng: khi vào pha tăng/giảm, giá thường chạy nhanh theo hướng chính và khó vào lại nếu chậm nhịp. Breakout/continuation có xác suất tốt hơn so với việc bắt đáy.",
        "Phù hợp với trader chủ động, theo dõi sát và chấp nhận ra/vào theo nhịp. Ưu tiên vào khi có follow-through và quản trị vị thế bằng trailing theo cấu trúc.",
        "Cần cảnh giác khi động lượng suy yếu (thiếu follow-through, đà thu hẹp): ưu tiên chốt từng phần và không nới kỷ luật stop.",
    ],
    "Aggressive Trend": [
        "Nhóm này có xu hướng nhưng tải rủi ro cao: biến động lớn, tail risk/gap có thể xuất hiện khi dòng tiền đổi trạng thái. Lợi nhuận tiềm năng cao nhưng sai nhịp sẽ trả giá nhanh.",
        "Phù hợp với trader chịu rung lắc, kỷ luật stop và quản trị size nghiêm ngặt. Chỉ nên tham gia khi plan rõ ràng và điểm vào tối ưu (không FOMO).",
        "Ưu tiên chiến thuật hit-and-run/pyramid có điều kiện sau khi đã giảm rủi ro (free-ride). Tránh giữ vị thế quá lớn qua thời điểm nhạy cảm.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Nhóm này vận động trong biên tương đối ổn định và hay quay về vùng cân bằng. Xác suất mua gần hỗ trợ–bán gần kháng cự tốt hơn kỳ vọng chạy trend dài liên tục.",
        "Phù hợp với trader kỷ luật, kiên nhẫn chờ vùng biên; ưu tiên scale-in ở vùng hỗ trợ và chốt dần ở vùng kháng cự.",
        "Rủi ro chính là phá biên: khi đóng cửa ra khỏi hộp giá, cần cắt/giảm nhanh để tránh bị kéo sang một regime mới.",
    ],
    "Volatile Range": [
        "Nhóm này vẫn có tính chất range/mean-reversion nhưng nhiễu và rung lắc mạnh hơn; dễ false-break và quét stop nếu vào giữa biên.",
        "Phù hợp với trader chọn lọc: chỉ vào khi có xác nhận đảo chiều tại mép biên và chấp nhận tỷ trọng nhỏ hơn bình thường.",
        "Ưu tiên chốt nhanh, không bình quân giá khi cấu trúc yếu và luôn quản trị rủi ro theo volatility.",
    ],
    "Mixed / Choppy Trader": [
        "Nhóm này thiếu xu hướng rõ ràng hoặc hay đổi tính theo thời gian; tín hiệu dễ nhiễu và whipsaw cao. Edge thường chỉ xuất hiện theo từng nhịp ngắn.",
        "Phù hợp với giao dịch chiến thuật: chỉ trade khi setup đạt chất lượng cao và có xác nhận; phần lớn thời gian nên đứng ngoài.",
        "Kỷ luật vào/ra và size nhỏ là bắt buộc; tránh giữ vị thế dài khi cấu trúc không rõ.",
    ],
    "Event / Gap-Prone": [
        "Nhóm này có rủi ro sự kiện/gap cao: giá có thể nhảy mạnh ngoài dự kiến, khiến stop dễ bị trượt. Dù có thể mang lại lợi nhuận lớn, execution risk cũng cao.",
        "Phù hợp với trader chấp nhận rủi ro tail, ưu tiên giao dịch ngắn hạn và giảm nắm giữ qua thời điểm nhạy cảm.",
        "Bắt buộc dùng size nhỏ, stop theo cấu trúc + buffer và chỉ tham gia khi reward đủ lớn để bù rủi ro.",
    ],
    "Illiquid / Noisy": [
        "Nhóm này có rủi ro thực thi: thanh khoản thiếu ổn định, spread/độ trượt có thể làm sai lệch R:R thực tế. Tín hiệu kỹ thuật thường kém tin cậy hơn do nhiễu.",
        "Phù hợp với nhà đầu tư rất kỷ luật và chấp nhận giải ngân nhỏ; ưu tiên lệnh giới hạn và tránh đuổi giá.",
        "Nếu không đạt điều kiện thanh khoản tối thiểu, nên coi đây là nhóm 'NO TRADE' dù kịch bản nhìn đẹp trên giấy.",
    ],
}



# Dashboard (Character Card) short narrative per class — single source of truth for Dashboard narrative
CLASS_TEMPLATES_DASHBOARD: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Đặc tính: Trend bền, hành vi giá tương đối sạch, phù hợp nắm giữ theo cấu trúc.",
        "Phù hợp: Trend-follow / tích lũy theo nhịp điều chỉnh, ưu tiên kỷ luật hơn tốc độ.",
        "Chiến thuật: Pullback & trend continuation, dời stop theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Đặc tính: Động lượng mạnh, chạy nhanh khi có lực, khó vào lại nếu chậm nhịp.",
        "Phù hợp: Trader chủ động, theo dõi sát, chốt lời từng phần theo đà.",
        "Chiến thuật: Breakout/continuation có xác nhận, trailing theo nhịp.",
    ],
    "Aggressive Trend": [
        "Đặc tính: Trend có nhưng rủi ro cao (biến động/tail/gap), sai nhịp trả giá nhanh.",
        "Phù hợp: Trader chịu rung lắc, size nhỏ hơn chuẩn, kỷ luật stop tuyệt đối.",
        "Chiến thuật: Hit & Run / pyramid có điều kiện sau khi giảm rủi ro.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Đặc tính: Sideway ổn định, hay quay về vùng cân bằng, biên hỗ trợ/kháng cự rõ.",
        "Phù hợp: Trader kiên nhẫn, đánh theo biên, ưu tiên xác suất hơn kỳ vọng lớn.",
        "Chiến thuật: Buy near support – sell near resistance, scale-in/out theo vùng.",
    ],
    "Volatile Range": [
        "Đặc tính: Range nhưng nhiễu, dễ false-break và quét stop; rung lắc mạnh.",
        "Phù hợp: Trader chọn lọc, chỉ vào ở mép biên và giảm tỷ trọng.",
        "Chiến thuật: Vào khi có xác nhận đảo chiều, chốt nhanh, quản trị theo vol.",
    ],
    "Mixed / Choppy Trader": [
        "Đặc tính: Không rõ trend/range, whipsaw cao; edge chỉ xuất hiện theo nhịp ngắn.",
        "Phù hợp: Tactical trader, chấp nhận đứng ngoài phần lớn thời gian.",
        "Chiến thuật: Trade khi setup thật rõ + có xác nhận; size nhỏ.",
    ],
    "Event / Gap-Prone": [
        "Đặc tính: Rủi ro sự kiện/gap cao, stop dễ trượt; cần reward lớn để bù rủi ro.",
        "Phù hợp: Trader mạo hiểm nhưng kỷ luật, tránh giữ qua thời điểm nhạy cảm.",
        "Chiến thuật: Size nhỏ, stop + buffer, chỉ tham gia khi RR đủ dày.",
    ],
    "Illiquid / Noisy": [
        "Đặc tính: Rủi ro thực thi (thanh khoản kém/không ổn định), tín hiệu dễ nhiễu.",
        "Phù hợp: Chỉ dành cho người rất kỷ luật, chấp nhận giải ngân nhỏ.",
        "Chiến thuật: Lệnh giới hạn, tránh đuổi giá; không đạt liquidity gate thì NO TRADE.",
    ],
}


# ============================================================
# CLASS POLICY HINTS (CURRENT STATUS) — DISPLAY-ONLY
# ============================================================
# Purpose: provide a one-line execution policy hint based on FINAL CLASS.
# Option C1 (hint only): does NOT modify scores, triggers, or trade plan logic.
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

# Mapping for bilingual playstyle tags (EN → EN + VI).
PLAYSTYLE_TAG_TRANSLATIONS: Dict[str, str] = {
    "Pullback-buy zone (confluence)": "Pullback-buy zone (confluence) - Vùng mua pullback có nhiều yếu tố hội tụ",
    "Breakout attempt (needs follow-through)": "Breakout attempt (needs follow-through) - Nỗ lực breakout, cần phiên xác nhận tiếp theo",
    "Wait for volume confirmation": "Wait for volume confirmation - Chờ xác nhận khối lượng thanh khoản",
    "Tight risk control near resistance": "Tight risk control near resistance - Siết chặt quản trị rủi ro gần vùng kháng cự",
    "Use LongStructure_ShortTactical rule": "Use LongStructure_ShortTactical rule - Áp dụng quy tắc cấu trúc dài hạn, tác chiến ngắn hạn",
}


def render_character_traits(character_pack: Dict[str, Any]) -> None:
    """
    STOCK DNA (Long-run 3–5Y) — STRICT layer.
    Displays ONLY:
      - Class + stable class narrative
      - Tier-1 (StyleAxis, RiskRegime) + DNAConfidence
      - DNA group scores (5 groups)
      - The 15-parameter pack (inside an expander)

    Deliberately excludes legacy "CORE STATS" and any 'Now/Opportunity' metrics.
    """
    cp = character_pack or {}
    cclass = _safe_text(cp.get("CharacterClass") or "N/A").strip()
    ticker = _safe_text(cp.get("_Ticker") or "").strip().upper()

    # ---- Class narrative (stable templates) ----
    class_label = f"CLASS: {cclass}"
    st.markdown(f"**{html.escape(class_label)}**")
    paras = CLASS_TEMPLATES.get(cclass) or []
    if paras:
        for para in paras:
            st.markdown(str(para))
    else:
        # fallback if template is missing
        st.markdown(get_character_blurb(ticker, cclass) or "")

    # ---- DNA pack (15 params / 5 groups) ----
    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
    tier1 = dna.get("Tier1") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    def _bar(label: str, value: Any) -> None:
        v = _safe_float(value, default=np.nan)
        if pd.isna(v):
            pct = 0.0
            v_disp = "N/A"
        else:
            v10 = float(max(0.0, min(10.0, float(v))))
            pct = _clip(v10 / 10.0 * 100.0, 0.0, 100.0)
            v_disp = f"{v10:.1f}/10"
        st.markdown(
            f"""
            <div class="gc-row">
              <div class="gc-k">{html.escape(str(label))}</div>
              <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
              <div class="gc-v">{html.escape(str(v_disp))}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    style_axis = tier1.get("StyleAxis", "N/A")
    risk_regime = tier1.get("RiskRegime", "N/A")
    dna_conf = tier1.get("DNAConfidence", np.nan)
    lock_flag = tier1.get("ClassLockFlag", False)
    modifiers = tier1.get("Modifiers", []) or []

    conf_txt = "N/A" if pd.isna(_safe_float(dna_conf, default=np.nan)) else f"{float(dna_conf):.0f}/100"
    mod_txt = ", ".join([str(x) for x in modifiers]) if isinstance(modifiers, list) and modifiers else "None"
    lock_txt = "LOCKED (low confidence)" if bool(lock_flag) else "OK"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">STOCK DNA (LONG-RUN 3–5Y)</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='gc-muted'>Tier-1: Style {html.escape(str(style_axis))} | Risk {html.escape(str(risk_regime))} | "
        f"DNAConfidence: {html.escape(conf_txt)} | {html.escape(lock_txt)}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='gc-muted'>Modifiers: {html.escape(mod_txt)}</div>",
        unsafe_allow_html=True
    )

    # 5 groups (stable anchors)
    _bar("Trend Structure", _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"])))
    _bar("Volatility & Tail Risk (higher = worse)", _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"])))
    _bar("Drawdown & Recovery Risk (higher = worse)", _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"])))
    _bar("Liquidity & Tradability", _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"])))
    _bar("Behavior / Setup Bias", _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"])))
    st.markdown("</div>", unsafe_allow_html=True)

    # 15 parameters (optional detail)
    label_map = {
        "TrendIntegrity": "Trend Integrity",
        "TrendPersistence": "Trend Persistence",
        "TrendChurnControl": "Trend Churn Control",
        "VolRisk": "Volatility Level (risk)",
        "TailGapRisk": "Tail/Gap Risk",
        "VolOfVolRisk": "Vol Regime Instability",
        "MaxDrawdownRisk": "Max Drawdown Risk",
        "RecoverySlownessRisk": "Recovery Slowness",
        "DrawdownFrequencyRisk": "Drawdown Frequency",
        "LiquidityTradability": "Tradability",
        "LiquidityLevel": "Liquidity Level",
        "LiquidityConsistency": "Liquidity Consistency",
        "BreakoutQuality": "Breakout Quality",
        "MeanReversionWhipsaw": "Mean-Reversion / Whipsaw",
        "AutoCorrMomentum": "Momentum Autocorr",
    }
    group_labels = {
        "TrendStructure": "Group 1 — Trend Structure",
        "VolatilityTail": "Group 2 — Volatility & Tail",
        "DrawdownRecovery": "Group 3 — Drawdown & Recovery",
        "LiquidityTradability": "Group 4 — Liquidity & Tradability",
        "BehaviorSetup": "Group 5 — Behavior / Setup Bias",
    }

    with st.expander("DNA Parameters (15)", expanded=False):
        for gk in ["TrendStructure", "VolatilityTail", "DrawdownRecovery", "LiquidityTradability", "BehaviorSetup"]:
            keys = groups.get(gk) or []
            st.markdown(f"**{group_labels.get(gk, gk)}**")
            for k in keys:
                _bar(label_map.get(k, k), params.get(k))



def render_combat_stats_panel(character_pack: Dict[str, Any]) -> None:
    """Render Combat Stats as 'Now / Opportunity' metrics (0–10), intended to live under CURRENT STATUS."""
    cp = character_pack or {}
    combat = cp.get("CombatStats") or {}

    combat_order = [
        ("Upside Power", combat.get("UpsidePower")),
        ("Downside Risk", combat.get("DownsideRisk")),
        ("RR Efficiency", combat.get("RREfficiency")),
        ("Breakout Force", combat.get("BreakoutForce")),
        ("Support Resilience", combat.get("SupportResilience")),
    ]

    def bar_0_10(label: str, value: Any) -> None:
        v = _safe_float(value, default=np.nan)
        if pd.isna(v):
            pct = 0.0
            v_disp = "N/A"
        else:
            v10 = float(max(0.0, min(10.0, float(v))))
            pct = _clip(v10 / 10.0 * 100.0, 0.0, 100.0)
            v_disp = f"{v10:.1f}/10"

        st.markdown(
            f"""
            <div class="gc-row">
              <div class="gc-k">{html.escape(str(label))}</div>
              <div class="gc-bar"><div class="gc-fill" style="width:{pct:.1f}%"></div></div>
              <div class="gc-v">{html.escape(v_disp)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">COMBAT STATS (NOW)</div>', unsafe_allow_html=True)
    for label, value in combat_order:
        bar_0_10(label, value)
    st.markdown("</div>", unsafe_allow_html=True)

def render_stock_dna_insight(character_pack: Dict[str, Any]) -> None:
    """
    DNA Insight — MUST stay in the long-run layer.
    No 'Now/Opportunity' metrics allowed here.
    """
    cp = character_pack or {}
    cclass = _safe_text(cp.get("CharacterClass") or "N/A").strip()

    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
    tier1 = dna.get("Tier1") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", []))
    vol_g = _avg(groups.get("VolatilityTail", []))
    dd_g = _avg(groups.get("DrawdownRecovery", []))
    liq_g = _avg(groups.get("LiquidityTradability", []))
    beh_g = _avg(groups.get("BehaviorSetup", []))

    style_axis = _safe_text(tier1.get("StyleAxis", "N/A"))
    risk_regime = _safe_text(tier1.get("RiskRegime", "N/A"))
    dna_conf = _safe_float(tier1.get("DNAConfidence", np.nan), default=np.nan)
    modifiers = tier1.get("Modifiers", []) or []

    # Interpretive tags (no extra computation beyond existing scores)
    strengths: List[str] = []
    cautions: List[str] = []

    if pd.notna(trend_g) and trend_g >= 6.8:
        strengths.append("Trend structure tương đối bền")
    if pd.notna(liq_g) and liq_g >= 6.2:
        strengths.append("Tính tradable/khớp lệnh khá ổn")

    if pd.notna(vol_g) and vol_g >= 6.8:
        cautions.append("Rủi ro biến động/tail cao hơn mức trung bình")
    if pd.notna(dd_g) and dd_g >= 6.8:
        cautions.append("Drawdown có thể sâu hoặc hồi phục chậm")
    if "ILLIQ" in modifiers:
        cautions.append("Execution risk cao do thanh khoản/ổn định thanh khoản yếu")
    if "GAP" in modifiers:
        cautions.append("Gap/event risk nổi bật; tránh nắm giữ quá tự tin qua thời điểm nhạy cảm")

    s_txt = "; ".join(strengths) if strengths else "Hành vi dài hạn ở mức trung tính"
    c_txt = "; ".join(cautions) if cautions else "Không có cảnh báo DNA nổi bật"

    conf_txt = "N/A" if pd.isna(dna_conf) else f"{dna_conf:.0f}/100"

    st.markdown("**DNA Insight (Long-run)**")
    st.markdown(f"- Class: **{html.escape(cclass)}** | Tier-1: Style **{html.escape(style_axis)}**, Risk **{html.escape(risk_regime)}** | DNAConfidence: **{html.escape(conf_txt)}**")
    st.markdown(f"- Strengths: {html.escape(s_txt)}")
    st.markdown(f"- Cautions: {html.escape(c_txt)}")

    # Strategy fit (high-level)
    if style_axis == "Trend":
        fit = "Ưu tiên chiến lược theo xu hướng: pullback/continuation; tránh bắt đáy ngược trend."
    elif style_axis == "Range":
        fit = "Ưu tiên chiến lược đánh biên: mua gần hỗ trợ – bán gần kháng cự; tránh mua đuổi giữa biên."
    elif style_axis == "Momentum":
        fit = "Ưu tiên chiến lược theo động lượng: breakout có xác nhận; quản trị vị thế chủ động."
    else:
        fit = "Ưu tiên chiến thuật chọn lọc: chỉ trade khi setup thật rõ và có xác nhận."

    st.markdown(f"- Fit: {html.escape(fit)}")

def render_current_status_insight(master_score_total: Any, conviction_score: Any, gate_status: Optional[str] = None) -> None:
    """Current Status Insight — concise interpretation of MasterScore & Conviction (single block).
    Note: This is intentionally short and belongs directly under the two score bars.
    """
    ms = _safe_float(master_score_total, default=np.nan)
    cs = _safe_float(conviction_score, default=np.nan)
    if pd.isna(ms) or pd.isna(cs) or (not math.isfinite(float(ms))) or (not math.isfinite(float(cs))):
        return
    ms = float(ms)
    cs = float(cs)

    def _bucket(v: float) -> str:
        if v < 4.0:
            return "low"
        if v < 6.0:
            return "mid"
        if v < 8.0:
            return "good"
        return "high"

    ms_b = _bucket(ms)
    cs_b = _bucket(cs)

    ms_meaning = {
        "low": ("Chất lượng cơ hội hiện tại kém hấp dẫn.",
                "Thường phản ánh: xu hướng/structure xấu hoặc R:R không đáng để mạo hiểm."),
        "mid": ("Cơ hội ở mức trung tính – có chất liệu nhưng chưa ‘ngon’ để commit mạnh.",
                "Thường phản ánh: trend chưa đủ sạch hoặc điểm vào chưa tối ưu; cần thêm xác nhận."),
        "good": ("Cơ hội khá hấp dẫn và có thể triển khai nếu trade plan rõ.",
                 "Thường phản ánh: cấu trúc/trend ổn và R:R tương đối tốt khi chọn đúng nhịp."),
        "high": ("Cơ hội rất hấp dẫn, thuộc nhóm ‘đáng ưu tiên’ trong watchlist.",
                 "Thường phản ánh: cấu trúc/trend đẹp và R:R/điểm vào đang ở vùng thuận lợi."),
    }[ms_b]

    cs_meaning = {
        "low": ("‘Độ chắc chắn của nhận định’ thấp (tín hiệu còn nhiễu / dễ bị đảo).",
                "Chỉ quan sát; tránh hành động lớn vì xác suất sai cao."),
        "mid": ("‘Độ chắc chắn của nhận định’ trung tính (đủ để theo dõi nghiêm túc).",
                "Có tín hiệu hợp lý nhưng chưa đủ đồng thuận để trade mạnh."),
        "good": ("‘Độ chắc chắn của nhận định’ khá tốt (đồng thuận tăng).",
                 "Có thể triển khai có kỷ luật; ưu tiên plan rõ ràng, tránh FOMO."),
        "high": ("‘Độ chắc chắn của nhận định’ cao (đồng thuận mạnh, ít nhiễu).",
                 "Phù hợp triển khai theo kế hoạch; tập trung quản trị rủi ro thay vì do dự."),
    }[cs_b]

    block = f"""Điểm tổng hợp {ms:.1f}/10
{ms_meaning[0]}
{ms_meaning[1]}

Điểm tin cậy {cs:.1f}/10
{cs_meaning[0]}
{cs_meaning[1]}"""
    st.markdown(block)

def render_executive_snapshot(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any], gate_status: str) -> None:
    """Executive Snapshot — dashboard-style summary card.

    Renderer-only: must not change engine scoring or trade-plan math.

    Notes:
      - Uses HTML for layout; always escape dynamic strings.
      - Detail sections (Stock DNA / Current Status / Trade Plan / Decision Layer) are rendered separately under an expander.
    """
    ap = analysis_pack or {}
    cp = character_pack or {}

    # --------- helpers ---------
    def _sf(x: Any) -> float:
        v = _safe_float(x, default=np.nan)
        try:
            v = float(v)
        except Exception:
            return np.nan
        return v if (not pd.isna(v) and math.isfinite(v)) else np.nan

    def _fmt_num(x: Any, nd: int = 1) -> str:
        v = _sf(x)
        return "N/A" if pd.isna(v) else f"{v:.{nd}f}"

    def _fmt_px(x: Any) -> str:
        v = _sf(x)
        return "N/A" if pd.isna(v) else f"{v:.2f}"

    def _fmt_pct(x: Any) -> str:
        v = _sf(x)
        return "" if pd.isna(v) else f"{v:+.2f}%"

    def _bar_pct_10(x: Any) -> float:
        v = _sf(x)
        if pd.isna(v):
            return 0.0
        return float(max(0.0, min(100.0, (v / 10.0) * 100.0)))

    def _dot(val: Any, good: float, warn: float) -> str:
        v = _sf(val)
        if pd.isna(v):
            return "y"
        if v >= good:
            return "g"
        if v >= warn:
            return "y"
        return "r"

    def _tier_label(tier: Any) -> str:
        try:
            t = int(float(tier))
        except Exception:
            return "TIER N/A"
        label_map = {
            7: "GOD-TIER",
            6: "VERY STRONG BUY",
            5: "STRONG BUY",
            4: "BUY",
            3: "WATCH",
            2: "CAUTIOUS",
            1: "NO EDGE",
        }
        return f"TIER {t}: {label_map.get(t, 'N/A')}"

    def _kelly_label(tier: Any) -> str:
        try:
            t = int(float(tier))
        except Exception:
            return "KELLY BET: N/A"
        if t >= 5:
            return "KELLY BET: FULL SIZE"
        if t == 4:
            return "KELLY BET: FULL SIZE"
        if t == 3:
            return "KELLY BET: HALF SIZE"
        if t == 2:
            return "KELLY BET: SMALL"
        return "KELLY BET: NO TRADE"

    # --------- data extraction ---------
    ticker = _safe_text(ap.get("Ticker") or cp.get("_Ticker") or "").strip().upper()
    last_pack = ap.get("Last") or {}
    close_px = last_pack.get("Close")

    mkt = ap.get("Market") or {}
    chg_pct = mkt.get("StockChangePct")

    scenario_name = _safe_text((ap.get("Scenario12") or {}).get("Name") or "N/A").strip()

    master_total = (ap.get("MasterScore") or {}).get("Total", np.nan)
    conviction = ap.get("Conviction", np.nan)

    class_name = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "N/A").strip()

    conv = cp.get("Conviction") or {}
    tier = conv.get("Tier", None)

    core = cp.get("CoreStats") or {}
    combat = cp.get("CombatStats") or {}

    # Primary setup (already computed by Python)
    primary = ap.get("PrimarySetup") or {}
    setup_name = _safe_text(primary.get("Name") or "N/A").strip()

    # Try to find matching plan for TP
    entry = stop = tp = rr = None
    for p in (ap.get("TradePlans") or []):
        if _safe_text(p.get("Name") or "").strip() == setup_name:
            entry = p.get("Entry")
            stop = p.get("Stop")
            tp = p.get("TP")
            rr = p.get("RR")
            break

    # Red flags (from CharacterPack weaknesses)
    flags = list(cp.get("Flags") or [])
    red_notes = []
    for f in flags:
        try:
            sev = int(f.get("severity", 0))
        except Exception:
            sev = 0
        if sev >= 2:
            note = _safe_text(f.get("note") or f.get("code") or "").strip()
            if note:
                red_notes.append(note)
        if len(red_notes) >= 2:
            break
    if not red_notes:
        red_notes = ["None"]

    # Triggers
    vol_ratio = (ap.get("ProTech") or {}).get("Volume", {}).get("Ratio")
    rr_val = rr

    dot_breakout = _dot(combat.get("BreakoutForce"), good=6.8, warn=5.5)
    dot_volume = _dot(vol_ratio, good=1.20, warn=0.95)
    dot_rr = _dot(rr_val, good=1.80, warn=1.30)

    # --- DEBUG (auto-show only when Upside Room is N/A) ---
    meta = cp.get("Meta") or {}
    if pd.isna(_sf((combat or {}).get("UpsideRoom", (combat or {}).get("UpsidePower")))):
        st.caption(f"[DEBUG] UpsideRoom=N/A | DenomUsed={meta.get('DenomUsed')} | ATR14={_fmt_num(meta.get('ATR14'),2)} | VolProxy={_fmt_num(meta.get('VolProxy'),2)}")
        st.caption(f"[DEBUG] Close={_fmt_num(meta.get('Close'),2)} | NR={_fmt_num(meta.get('NearestRes'),2)} | NS={_fmt_num(meta.get('NearestSup'),2)} | UpsideRaw={_fmt_num(meta.get('UpsideRaw'),2)} | DownsideRaw={_fmt_num(meta.get('DownsideRaw'),2)} | LvlSrc={meta.get('LevelCtxSource')}")
        st.caption(f"[DEBUG] UpsideNorm={_fmt_num(meta.get('UpsideNorm'),2)} | DownsideNorm={_fmt_num(meta.get('DownsideNorm'),2)} | RR={_fmt_num(meta.get('RR'),2)} | BreakoutForce={_fmt_num((combat or {}).get('BreakoutForce'),2)} | VolRatio={_fmt_num(vol_ratio,2)} | RR_plan={_fmt_num(rr_val,2)} | LvlKeys={meta.get('LevelCtxKeys')}")

    # --------- render ---------
    tier_badge = _tier_label(tier)
    kelly_badge = _kelly_label(tier)
    gate = (gate_status or "N/A").strip().upper()

    # Header strings
    title_left = f"{ticker}"
    if _fmt_px(close_px) != "N/A":
        title_left = f"{title_left} | {_fmt_px(close_px)}"
    chg_str = _fmt_pct(chg_pct)

    sub_1 = " | ".join([x for x in [class_name, scenario_name] if x and x != "N/A"])
    sub_2 = f"Điểm tổng hợp: {_fmt_num(master_total,1)} | Điểm tin cậy: {_fmt_num(conviction,1)} | Gate: {gate}"

    # Pillar metrics
    def _metric_row(k: str, v: Any, nd: int = 1):
        return f"<div class='es-metric'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(_fmt_num(v, nd))}</div></div>"

        # Panel 1 (DNA) — compact class narrative + Class Signature radar (5 metrics)
    dash_lines: List[str] = (CLASS_TEMPLATES_DASHBOARD.get(class_name) or []).copy()
    if not dash_lines:
        # Fallback: keep Dashboard readable even if class is unknown
        fallback = get_character_blurb(ticker, class_name)
        if fallback:
            dash_lines = [f"Đặc tính: {fallback}"]
        else:
            dash_lines = [f"Đặc tính: {class_name}"]

    def _fmt_bline_es(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if ":" in s:
            k, v = s.split(":", 1)
            return f'<div class="es-bline"><b>{html.escape(k.strip())}:</b> {html.escape(v.strip())}</div>'
        return f'<div class="es-bline">{html.escape(s)}</div>'

    narrative_html = "".join([_fmt_bline_es(x) for x in dash_lines if str(x).strip()])

    def _radar_svg_es(stats: List[Tuple[str, float]], maxv: float = 10.0, size: int = 180) -> str:
        """Inline SVG radar chart (0–maxv) for Executive Snapshot (dark background)."""
        n = len(stats)
        if n < 3:
            return ""
        cx = cy = size / 2.0
        r = size * 0.34
        angles = [(-math.pi / 2.0) + i * (2.0 * math.pi / n) for i in range(n)]

        def pt(angle: float, radius: float) -> Tuple[float, float]:
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        vals: List[float] = []
        for _, v in stats:
            vv = 0.0 if pd.isna(v) else float(v)
            vv = float(_clip(vv, 0.0, maxv))
            vals.append(vv)

        grid_levels = [2, 4, 6, 8, 10]
        grid_polys = []
        for lv in grid_levels:
            rr_ = r * (lv / maxv)
            pts = [pt(a, rr_) for a in angles]
            grid_polys.append(" ".join([f"{x:.1f},{y:.1f}" for x, y in pts]))

        axis_pts = [pt(a, r) for a in angles]
        data_pts = [pt(angles[i], r * (vals[i] / maxv)) for i in range(n)]
        data_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in data_pts])
        label_pts = [pt(a, r + 26) for a in angles]

        parts: List[str] = []
        parts.append(f'<svg class="es-radar-svg" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">')
        for poly in grid_polys:
            parts.append(f'<polygon points="{poly}" fill="none" stroke="rgba(255,255,255,0.16)" stroke-width="1" />')
        for (x, y) in axis_pts:
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,0.18)" stroke-width="1" />')
        parts.append(f'<polygon points="{data_points}" fill="rgba(124,58,237,0.20)" stroke="rgba(124,58,237,0.95)" stroke-width="2" />')
        for (x, y) in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="rgba(124,58,237,0.95)" />')
        for i, (lx, ly) in enumerate(label_pts):
            lab = html.escape(str(stats[i][0]))
            raw_v = stats[i][1]
            val_txt = "—" if pd.isna(raw_v) else f"{vals[i]:.1f}"
            anchor = "middle"
            if lx < cx - 10:
                anchor = "end"
            elif lx > cx + 10:
                anchor = "start"
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" font-weight="900" fill="rgba(255,255,255,0.85)">'
                f'<tspan x="{lx:.1f}" dy="0">{lab}</tspan>'
                f'<tspan x="{lx:.1f}" dy="13" font-size="11" font-weight="850" fill="rgba(255,255,255,0.70)">{val_txt}</tspan>'
                f'</text>'
            )
        parts.append('</svg>')
        return "".join(parts)

    # Class Signature (DNA long-run only): 5-group anchors (0–10). No 'Now/Opportunity' metrics here.
    dna_pack = (cp.get("StockTraits") or {}).get("DNA") or {}
    params = dna_pack.get("Params") or {}
    groups = dna_pack.get("Groups") or {}

    tier1 = dna_pack.get("Tier1") or {}
    style_axis_es = _safe_text(tier1.get("StyleAxis") or "").strip()
    primary_mod_es = _safe_text(tier1.get("PrimaryModifier") or "").strip()

    def _mod_label_es(pm: str) -> str:
        pm = (pm or "").strip().upper()
        if pm == "GAP":
            return "Event/Gap-Prone"
        if pm == "ILLIQ":
            return "Illiquid/Noisy"
        if pm == "HIVOL":
            return "High-Vol"
        if pm == "CHOPVOL":
            return "Choppy-Vol"
        if pm == "DEF":
            return "Defensive"
        if pm == "HBETA":
            return "High-Beta"
        return ""

    mod_lab_es = _mod_label_es(primary_mod_es)
    dna_conf_es = tier1.get("DNAConfidence")

    badge_bits_es: List[str] = []
    if style_axis_es:
        badge_bits_es.append(f"Style: {style_axis_es}")
    if mod_lab_es:
        badge_bits_es.append(f"Flag: {mod_lab_es}")
    conf_txt_es = _fmt_num(dna_conf_es, 0)
    if conf_txt_es != "N/A":
        badge_bits_es.append(f"DNA: {conf_txt_es}")

    dna_badges_es = " | ".join(badge_bits_es) if badge_bits_es else ""
    dna_badge_html_es = f'<div class="es-note" style="margin-top:4px;opacity:0.85;">{html.escape(dna_badges_es)}</div>' if dna_badges_es else ""


    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _sf(params.get(k))
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"]))
    vol_risk_g = _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"]))
    dd_risk_g = _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"]))
    liq_g = _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"]))
    beh_g = _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"]))

    # Convert risk groups into positive-direction scores (higher = better control)
    vol_stab = (10.0 - float(vol_risk_g)) if pd.notna(vol_risk_g) else float("nan")
    dd_res = (10.0 - float(dd_risk_g)) if pd.notna(dd_risk_g) else float("nan")

    sig_stats: List[Tuple[str, float]] = [
        ("Trend", trend_g),
        ("Vol-Stability", vol_stab),
        ("DD-Resilience", dd_res),
        ("Liquidity", liq_g),
        ("Behavior", beh_g),
    ]
    radar_svg = _radar_svg_es(sig_stats, maxv=10.0, size=220)

    panel1 = f"""
    <div class="es-panel">
      <div class="es-pt">1) STOCK DNA</div>
      <div class="es-note" style="font-weight:900;">{html.escape(class_name)}</div>{dna_badge_html_es}
      <div class="es-bline-wrap">{narrative_html}</div>
      <div class="es-sig-wrap">
        <div class="es-sig-radar">{radar_svg}</div>
      </div>
    </div>
    """

    # ---------------------------
    # Panel 2 — CURRENT STATUS (Dashboard)
    #   Goal: answer in seconds: state → scores → triggers → next step → risks
    # ---------------------------
    protech = ap.get("ProTech") or {}
    protech = protech if isinstance(protech, dict) else {}
    ma = protech.get("MA") or {}
    ma = ma if isinstance(ma, dict) else {}
    volp = protech.get("Volume") or {}
    volp = volp if isinstance(volp, dict) else {}
    bias = protech.get("Bias") or {}
    bias = bias if isinstance(bias, dict) else {}

    ma_reg = _safe_text(ma.get("Regime") or "N/A").strip()
    vol_reg = _safe_text(volp.get("Regime") or "N/A").strip()

    # Location tag: prioritize explicit risk flags; fallback to MA200 positioning
    flags_list = list(cp.get("Flags") or [])
    has_near_res = any(isinstance(f, dict) and _safe_text(f.get("code")).strip() == "NearMajorResistance" for f in flags_list)
    has_near_sup = any(isinstance(f, dict) and _safe_text(f.get("code")).strip() == "NearMajorSupport" for f in flags_list)
    loc_tag = "Neutral"
    if has_near_res:
        loc_tag = "Near Resistance"
    elif has_near_sup:
        loc_tag = "Near Support"
    else:
        ma_struct = ma.get("Structure") or {}
        ma_struct = ma_struct if isinstance(ma_struct, dict) else {}
        above_200 = ma_struct.get("PriceAboveMA200")
        if above_200 is True:
            loc_tag = "Above MA200"
        elif above_200 is False:
            loc_tag = "Below MA200"

    # State label: prefer setup intent; fallback to MA regime
    sn_l = (setup_name or "").lower()
    state_label = "Neutral"
    if "breakout" in sn_l:
        state_label = "Breakout Attempt"
    elif "pullback" in sn_l:
        state_label = "Pullback"
    else:
        if ma_reg == "Close>=MA50>=MA200":
            state_label = "Uptrend"
        elif ma_reg == "Close<MA50<MA200":
            state_label = "Downtrend"
        elif ma_reg == "MixedStructure":
            state_label = "Mixed/Choppy"

    reg_tag = f"Vol: {vol_reg}" if (vol_reg and vol_reg != "N/A") else ""
    state_capsule_line = " | ".join([x for x in [state_label, loc_tag, reg_tag] if x])

    # Score bars (0–10)
    ms_pct = _bar_pct_10(master_total)
    cs_pct = _bar_pct_10(conviction)

    # One-line score interpretation (keep extremely short for dashboard)
    def _bucket(v: float) -> str:
        if v < 4.0:
            return "low"
        if v < 6.0:
            return "mid"
        if v < 8.0:
            return "good"
        return "high"

    insight_line_es = ""
    ms_v = _safe_float(master_total, default=np.nan)
    cs_v = _safe_float(conviction, default=np.nan)
    if pd.notna(ms_v) and pd.notna(cs_v):
        ms_b = _bucket(float(ms_v))
        cs_b = _bucket(float(cs_v))
        if ms_b in ("low",):
            insight_line_es = "Cơ hội kém hấp dẫn; ưu tiên quan sát và chờ cấu trúc/điểm vào tốt hơn."
        elif ms_b == "mid" and cs_b in ("good", "high"):
            insight_line_es = "Cơ hội trung tính nhưng độ tin cậy khá tốt; ưu tiên plan kỷ luật, tránh FOMO."
        elif ms_b in ("good", "high") and cs_b in ("low", "mid"):
            insight_line_es = "Cơ hội khá hấp dẫn nhưng độ tin cậy chưa cao; chỉ triển khai chọn lọc và chờ trigger đồng pha."
        elif ms_b in ("good", "high") and cs_b in ("good", "high"):
            insight_line_es = "Cơ hội hấp dẫn và độ tin cậy cao; có thể triển khai theo plan, tập trung quản trị rủi ro."
        else:
            insight_line_es = "Theo dõi nghiêm túc; ưu tiên đúng nhịp/điều kiện thay vì vào vội."
    if (gate_status or "").strip().upper() not in ("", "N/A", "ACTIVE") and insight_line_es:
        insight_line_es = f"{insight_line_es} (Gate: {(gate_status or '').strip().upper()})"

    policy_hint_es = get_class_policy_hint_line(class_name)

    # Trigger status (Plan-Gated) — use PASS/WAIT/FAIL text, not just dots
    def _status(v: Any, good: float, warn: float) -> str:
        x = _safe_float(v, default=np.nan)
        if pd.isna(x):
            return "N/A"
        if float(x) >= good:
            return "PASS"
        if float(x) >= warn:
            return "WAIT"
        return "FAIL"

    # Plan status + RR (prefer TradePlans to stay consistent with detail)
    plan_status_es = "N/A"
    rr_plan = rr_val
    for p in (ap.get("TradePlans") or []):
        if _safe_text(p.get("Name") or "").strip() == setup_name and setup_name and setup_name != "N/A":
            plan_status_es = _safe_text(p.get("Status") or "N/A").strip()
            rr_plan = _safe_float(p.get("RR"), default=rr_plan)
            break

    st_break = _status((combat or {}).get("BreakoutForce"), good=6.8, warn=5.5)
    st_vol = _status(vol_ratio, good=1.20, warn=0.95)
    st_rr = _status(rr_plan, good=1.80, warn=1.30)

    # Structure (Ceiling) Gate: quality-aware resistance ceiling control
    sq = {}
    try:
        sq = (character_pack or {}).get("StructureQuality", {}) or (analysis_pack or {}).get("StructureQuality", {})
    except Exception:
        sq = {}
    cg = ((sq or {}).get("Gates", {}) or {}).get("CeilingGate", {}) if isinstance((sq or {}).get("Gates", {}), dict) else {}
    st_struct = _safe_text(cg.get("Status") or "N/A").strip().upper()
    if st_struct not in ("PASS", "WAIT", "FAIL"):
        st_struct = "N/A"

    def _dot_from_status(s: str) -> str:
        s = (s or "").upper()
        if s == "PASS":
            return "g"
        if s == "WAIT":
            return "y"
        if s == "FAIL":
            return "r"
        return "y"

    dot_b2 = _dot_from_status(st_break)
    dot_v2 = _dot_from_status(st_vol)
    dot_r2 = _dot_from_status(st_rr)
    dot_s2 = _dot_from_status(st_struct)

    gate_line = f"Gate: {(gate_status or 'N/A').strip().upper()} | Plan: {setup_name} ({plan_status_es or 'N/A'})"

    # One Next Step (single line)
    next_step = "Theo dõi và chờ thêm dữ liệu."
    if st_struct in ("FAIL", "WAIT"):
        _ov = ((sq or {}).get("OverheadResistance", {}) or {}).get("Nearest", {}) or {}
        _comps = _ov.get("ComponentsTop") if isinstance(_ov.get("ComponentsTop"), list) else []
        _t = _safe_text(((_comps[0] or {}).get("Type")) if (len(_comps) > 0 and isinstance(_comps[0], dict)) else "").strip()
        next_step = "Chờ vượt trần cấu trúc trước khi tăng xác suất vào lệnh."
        if _t:
            next_step = f"Chờ vượt trần cấu trúc ({_t}) trước khi tăng xác suất vào lệnh."
    elif st_break in ("FAIL", "WAIT"):
        next_step = "Chờ breakout xác nhận/follow-through; tránh vào sớm khi lực chưa rõ."
    elif st_vol in ("FAIL", "WAIT"):
        next_step = "Chờ volume xác nhận (≥ 1.2×20D) để giảm false-break."
    elif st_rr in ("FAIL", "WAIT"):
        next_step = "Chờ điểm vào tốt hơn để RR ≥ 1.8 (hoặc giảm risk/stop hợp lý)."
    else:
        next_step = "Có thể triển khai theo plan; ưu tiên kỷ luật stop, tránh FOMO."

    # Risk flags (top 2, severity>=2)
    risk_lines: List[str] = []
    for f in flags_list:
        if not isinstance(f, dict):
            continue
        try:
            sev = int(f.get("severity", 0))
        except Exception:
            sev = 0
        if sev < 2:
            continue
        code = _safe_text(f.get("code") or "").strip()
        note = _safe_text(f.get("note") or "").strip()
        if code and note:
            risk_lines.append(f"[{code}] {note}")
        elif code:
            risk_lines.append(f"[{code}]")
        elif note:
            risk_lines.append(note)
        if len(risk_lines) >= 2:
            break
    if not risk_lines:
        risk_lines = ["None"]

    panel2 = f"""
    <div class=\"es-panel\">
      <div class=\"es-pt\">2) CURRENT STATUS</div>

      <div class=\"es-note\" style=\"font-weight:950;\">{html.escape(state_capsule_line)}</div>

      <div class=\"es-metric\"><div class=\"k\">Điểm tổng hợp</div><div class=\"v\">{html.escape(_fmt_num(master_total,1))}</div></div>
      <div class=\"es-mini\"><div style=\"width:{ms_pct:.0f}%\"></div></div>

      <div class=\"es-metric\" style=\"margin-top:6px;\"><div class=\"k\">Điểm tin cậy</div><div class=\"v\">{html.escape(_fmt_num(conviction,1))}</div></div>
      <div class=\"es-mini\"><div style=\"width:{cs_pct:.0f}%\"></div></div>

      {f'<div class="es-note" style="margin-top:8px;">{html.escape(insight_line_es)}</div>' if insight_line_es else ''}

      {f'<div class="es-note" style="margin-top:6px;"><b>Policy:</b> {html.escape(policy_hint_es)}</div>' if policy_hint_es else ''}

      <div class=\"es-note\" style=\"margin-top:10px;font-weight:950;\">Trigger Status (Plan-Gated)</div>
      <div class=\"es-note\"><span class=\"es-dot {dot_b2}\"></span>Breakout: <b>{html.escape(st_break)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_v2}\"></span>Volume: <b>{html.escape(st_vol)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_r2}\"></span>R:R: <b>{html.escape(st_rr)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_s2}\"></span>Structure: <b>{html.escape(st_struct)}</b></div>

      <div class=\"es-note\" style=\"margin-top:8px;\"><b>{html.escape(gate_line)}</b></div>
      <div class=\"es-note\" style=\"margin-top:6px;\">Next step: {html.escape(next_step)}</div>

      <div class=\"es-note\" style=\"margin-top:10px;font-weight:950;\">Risk Flags</div>
      <ul class=\"es-bul\">{''.join([f'<li>{html.escape(x)}</li>' for x in risk_lines])}</ul>
    </div>
    """


    def _delta_pct(entry_x: Any, level_x: Any) -> Any:
        e = _safe_float(entry_x)
        l = _safe_float(level_x)
        if pd.notna(e) and pd.notna(l) and e != 0:
            return (l - e) / e * 100.0
        return np.nan

    stop_delta = _delta_pct(entry, stop)
    tp_delta = _delta_pct(entry, tp)
    stop_str = _fmt_px(stop) if pd.isna(stop_delta) else f"{_fmt_px(stop)} ({_fmt_pct(stop_delta)})"
    tp_str = _fmt_px(tp) if pd.isna(tp_delta) else f"{_fmt_px(tp)} ({_fmt_pct(tp_delta)})"

    panel3 = f"""
    <div class="es-panel">
      <div class="es-pt">3) SCENARIO</div>
      <div class="es-note"><b>Kịch bản chính:</b> {html.escape(scenario_name)}</div>
      <ul class="es-bul">
        <li>Setup: {html.escape(setup_name)}</li>
        <li>Entry/Stop: {html.escape(_fmt_px(entry))} / {html.escape(stop_str)}</li>
        <li>Target: {html.escape(tp_str)} (RR {html.escape(_fmt_num(rr,1))})</li>
      </ul>
      <div class="es-note" style="margin-top:8px;font-weight:900;">Red Flags</div>
      <ul class="es-bul">{''.join([f'<li>{html.escape(x)}</li>' for x in red_notes])}</ul>
    </div>
    """

    card_html = f"""
    <div class="es-card">
      <div class="es-head">
        <div class="es-left">
          <div class="es-tline">
            <div class="es-ticker">{html.escape(title_left)}</div>
            {f'<div class="es-chg">{html.escape(chg_str)}</div>' if chg_str else ''}
          </div>
          <div class="es-sub">{html.escape(sub_1) if sub_1 else ''}</div>
          <div class="es-meta">{html.escape(sub_2)}</div>
        </div>
        <div class="es-right">
          <div class="es-badge">{html.escape(tier_badge)}</div>
          <div class="es-kelly">{html.escape(kelly_badge)}</div>
        </div>
      </div>
      <div class="es-body">
        {panel1}
        {panel2}
        {panel3}
      </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def render_character_decision(character_pack: Dict[str, Any]) -> None:
    """
    Render only the 'Decision' part of Character Card (for Appendix E / anti-anchoring).
    Includes: Conviction + Weaknesses + Playstyle Tags.
    """
    cp = character_pack or {}
    conv = cp.get("Conviction") or {}
    flags = cp.get("Flags") or []
    tags = cp.get("ActionTags") or []

    tier = conv.get("Tier", "N/A")
    pts = conv.get("Points", np.nan)
    guide = conv.get("SizeGuidance", "")

    st.markdown(
        f"""
        <div class="gc-sec">
          <div class="gc-sec-t">CONVICTION</div>
          <div class="gc-conv">
            <div class="gc-conv-tier">Tier: <b>{tier}</b> / 7</div>
            <div class="gc-conv-pts">Points: <b>{_val_or_na(pts)}</b></div>
            <div class="gc-conv-guide">{html.escape(str(guide))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if flags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">WEAKNESSES</div>', unsafe_allow_html=True)
        for f in flags[:8]:
            sev = int(f.get("severity", 1))
            note = f.get("note", "")
            code_ = f.get("code", "")
            st.markdown(
                f"""<div class="gc-flag"><span class="gc-sev">S{sev}</span><span class="gc-code">{html.escape(str(code_))}</span><span class="gc-note">{html.escape(str(note))}</span></div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if tags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">PLAYSTYLE TAGS</div>', unsafe_allow_html=True)
        rendered_tags: List[str] = []
        for t in tags[:8]:
            raw = str(t)
            human = PLAYSTYLE_TAG_TRANSLATIONS.get(raw, raw)
            rendered_tags.append(f"<span class='gc-tag'>{html.escape(str(human))}</span>")
        st.markdown(
            "<div class='gc-tags'>" + "".join(rendered_tags) + "</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


def _trade_plan_gate(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Anti-anchoring gate for Trade Plan rendering.
    Returns (status, meta) where status ∈ {"ACTIVE","WATCH","LOCK"}.
    Meta includes score/tier for UI copy.
    """
    ap = analysis_pack or {}
    cp = character_pack or {}
    score = _safe_float(ap.get("Conviction"), default=np.nan)
    tier = (cp.get("Conviction") or {}).get("Tier", None)

    # Default thresholds (v6.4 Appendix E)
    active = (pd.notna(score) and score >= 6.5) or (isinstance(tier, (int, float)) and tier >= 4)
    watch = (pd.notna(score) and 5.5 <= score < 6.5) or (isinstance(tier, (int, float)) and int(tier) == 3)

    if active:
        status = "ACTIVE"
    elif watch:
        status = "WATCH"
    else:
        status = "LOCK"

    return status, {"ConvictionScore": score, "Tier": tier}



def render_market_state(analysis_pack: Dict[str, Any]) -> None:
    """
    Appendix E section (2): Market State / Current Regime.
    Must never crash if market context is missing.
    """
    ap = analysis_pack or {}
    m = ap.get("Market") or {}
    vn = m.get("VNINDEX") or {}
    vn30 = m.get("VN30") or {}

    def _fmt_change(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return "N/A"
        return f"{v:+.2f}%"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">MARKET STATE (CURRENT REGIME)</div>', unsafe_allow_html=True)

    # If no market pack, show a clean fallback (no error, no stacktrace)
    if not m:
        st.info("Market State: N/A (market context not available).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <div class="ms-row"><b>VNINDEX</b>: {_val_or_na(vn.get("Regime"))} | Change: {_fmt_change(vn.get("ChangePct"))}</div>
        <div class="ms-row"><b>VN30</b>: {_val_or_na(vn30.get("Regime"))} | Change: {_fmt_change(vn30.get("ChangePct"))}</div>
        <div class="ms-row"><b>Relative Strength vs VNINDEX</b>: {_val_or_na(m.get("RelativeStrengthVsVNINDEX"))}</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


    def _fmt_change(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return "N/A"
        return f"{v:+.2f}%"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">MARKET STATE (CURRENT REGIME)</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ms-row"><b>VNINDEX</b>: {_val_or_na(vn.get("Regime"))} | Change: {_fmt_change(vn.get("ChangePct"))}</div>
        <div class="ms-row"><b>VN30</b>: {_val_or_na(vn30.get("Regime"))} | Change: {_fmt_change(vn30.get("ChangePct"))}</div>
        <div class="ms-row"><b>Relative Strength vs VNINDEX</b>: {_val_or_na(m.get("RelativeStrengthVsVNINDEX"))}</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)



def render_trade_plan_conditional(analysis_pack: Dict[str, Any], gate_status: str, trade_text: str = "") -> None:
    """
    Appendix E section (3): Trade Plan & R:R (Conditional).
    Uses AnalysisPack.TradePlans (Python-computed). No GPT math.
    Layout-only: reorders how the engine output is displayed.
    """
    ap = analysis_pack or {}
    plans = list(ap.get("TradePlans") or [])

    # Outer visual container (title already printed by render_appendix_e)
    st.markdown('<div class="gc-sec">', unsafe_allow_html=True)

    # Helper: render Risk/Reward snapshot for the primary plan
    def _render_rr_snapshot() -> None:
        primary = (ap.get("PrimarySetup") or {}) if isinstance(ap, dict) else {}
        if not primary:
            return

        name = _val_or_na(primary.get("Name"))
        risk = primary.get("RiskPct")
        reward = primary.get("RewardPct")
        rr = primary.get("RR")
        conf_tech = primary.get("Confidence (Tech)", primary.get("Probability"))

        def _fmt_pct_local(x: Any) -> str:
            try:
                if x is None or pd.isna(x) or not math.isfinite(float(x)):
                    return "N/A"
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        def _fmt_rr_local(x: Any) -> str:
            try:
                if x is None or pd.isna(x) or not math.isfinite(float(x)):
                    return "N/A"
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        st.markdown(f"#### Risk / Reward Snapshot (Primary Plan: {name})")
        st.markdown(
            f"""
            <div class="incept-metrics">
              <div class="incept-metric"><div class="k">Risk%:</div><div class="v">{_fmt_pct_local(risk)}</div></div>
              <div class="incept-metric"><div class="k">Reward%:</div><div class="v">{_fmt_pct_local(reward)}</div></div>
              <div class="incept-metric"><div class="k">RR:</div><div class="v">{_fmt_rr_local(rr)}</div></div>
              <div class="incept-metric"><div class="k">Confidence (Tech):</div><div class="v">{_val_or_na(conf_tech)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If gate is locked, show defensive posture + still expose snapshot
    if gate_status == "LOCK":
        st.info("Trade Plan đang bị khóa (chống FOMO). Ưu tiên bảo toàn vốn và chờ tín hiệu xác nhận.")
        st.markdown(
            """
            <div class="tp-lock">
              <div><b>Tư thế:</b> CHỜ / PHÒNG THỦ</div>
              <div style="margin-top:6px;"><b>Checklist kích hoạt:</b></div>
              <ul>
                <li>Conviction tăng lên ngưỡng kích hoạt</li>
                <li>Giá lấy lại vùng MA quan trọng / cấu trúc ổn định trở lại</li>
                <li>Khối lượng xác nhận (không có dấu hiệu kiệt sức), động lượng cải thiện</li>
                <li>Cấu trúc tuần còn nguyên vẹn (không breakdown cấu trúc)</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        _render_rr_snapshot()
        return

    if not plans:
        st.warning("No Trade Plan available from engine.")
        st.markdown("</div>", unsafe_allow_html=True)
        _render_rr_snapshot()
        return

    # Sort: prefer higher probability label, then higher RR
    def _prob_rank(p: Any) -> int:
        s = str(p or "").lower()
        if "high" in s:
            return 3
        if "med" in s:
            return 2
        if "low" in s:
            return 1
        return 0

    def _conf_val(p: Dict[str, Any]) -> Any:
        return p.get("Confidence (Tech)", p.get("Probability"))

    plans_sorted = sorted(
        plans,
        key=lambda x: (
            -_prob_rank(_conf_val(x)),
            -_safe_float(x.get("RR"), default=-1e9),
        ),
    )

    def _to_html_lines(txt: str) -> str:
        """Escape + preserve line breaks for HTML blocks."""
        txt = (txt or "").strip()
        if not txt:
            return ""
        return "<br>".join(html.escape(txt).splitlines())

    def _split_trade_text_by_plan(txt: str, plan_names: List[str]) -> Dict[str, str]:
        """Best-effort split legacy C-section narrative into per-plan notes.

        Expected Vietnamese/EN anchors:
          - "Kế hoạch giao dịch <PlanName> ..."
          - "Trade plan <PlanName> ..."
        If no anchors are found, return empty dict and caller will attach the whole text to the first plan.
        """
        txt = (txt or "").strip()
        names = [n for n in (plan_names or []) if n and str(n).strip() and str(n).strip().upper() != "N/A"]
        if (not txt) or (not names):
            return {}

        # Build a single regex that matches any plan name following the anchor phrase.
        alts = "|".join(sorted((re.escape(str(n)) for n in set(names)), key=len, reverse=True))
        pat = re.compile(r"(?i)(kế\s*hoạch\s*giao\s*dịch|trade\s*plan)\s+(" + alts + r")\b")

        matches = list(pat.finditer(txt))
        if not matches:
            return {}

        out: Dict[str, str] = {}
        for i, m in enumerate(matches):
            plan = m.group(2)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
            seg = txt[start:end].strip()
            if not seg:
                continue
            # Keep the segment as-is (it often reads well in Vietnamese), but preserve breaks safely.
            out[str(plan)] = _to_html_lines(seg)
        return out

    # 3.1 Setup Overview
    # NOTE: Layout-only changes (no logic changes).
    status_vi = "ĐANG KÍCH HOẠT" if gate_status == "ACTIVE" else "CHỈ THEO DÕI"
    desc_vi = (
        "Được phép thực thi (có điều kiện). Áp dụng size động và tuân thủ tuyệt đối vùng dừng lỗ."
        if gate_status == "ACTIVE"
        else "Kế hoạch chỉ mang tính tham khảo. Không FOMO, không ép lệnh."
    )

    st.markdown(
        f"""
        <div class="tp-sec-h">
          <div class="tp-sec-title">
            <span>Các yếu tố kỹ thuật kích hoạt TRADE PLAN</span>
            <span class="tp-badge {'active' if gate_status == 'ACTIVE' else 'watch'}">{status_vi}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='tp-note'><b>{status_vi}:</b> {html.escape(desc_vi)}</div>", unsafe_allow_html=True)

    # WATCH: still show 2 plans, but the 2nd plan is a dimmed "tham khảo" reference.
    show_n = 2 if gate_status in ("ACTIVE", "WATCH") else 1
    plans_to_show = plans_sorted[: min(show_n, len(plans_sorted))]
    plan_names = [str(_val_or_na(p.get("Name"))) for p in plans_to_show]
    expl_map = _split_trade_text_by_plan(trade_text, plan_names)

    # If we couldn't split, attach the entire narrative to the first plan only.
    fallback_expl = ""
    if trade_text and not expl_map:
        fallback_expl = _to_html_lines(trade_text)

    for idx, p in enumerate(plans_to_show):
        name = _val_or_na(p.get("Name"))
        entry = _safe_float(p.get("Entry"), default=np.nan)
        stop = _safe_float(p.get("Stop"), default=np.nan)
        tp = _safe_float(p.get("TP"), default=np.nan)
        rr = _safe_float(p.get("RR"), default=np.nan)
        conf_tech = _val_or_na(p.get("Confidence (Tech)", p.get("Probability")))
        status = _val_or_na(p.get("Status"))

        is_ref = (gate_status == "WATCH" and idx == 1)
        card_cls = "tp-card dim" if is_ref else "tp-card"
        ref_badge = '<span class="tp-ref">tham khảo</span>' if is_ref else ""

        rr_label = (
            "Attractive"
            if (pd.notna(rr) and rr >= 2.0)
            else ("Acceptable" if (pd.notna(rr) and rr >= 1.3) else "Thin")
        )

        if pd.notna(rr):
            rr_disp = f"{float(rr):.1f}"
        else:
            rr_disp = "N/A"

        def _fmt_px(x: Any) -> str:
            v = _safe_float(x, default=np.nan)
            return f"{v:.2f}" if pd.notna(v) else "N/A"

        def _fmt_px_with_delta(x: Any, entry_x: Any) -> str:
            v = _safe_float(x, default=np.nan)
            e = _safe_float(entry_x, default=np.nan)
            if pd.notna(v) and pd.notna(e) and e != 0:
                d = (v - e) / e * 100.0
                return f"{v:.2f} ({d:+.1f}%)"
            return _fmt_px(v)

        st.markdown(
            f"""
            <div class="{card_cls}">
              <div class="tp-title"><b>{html.escape(str(name))}</b> <span class="tp-status">[{html.escape(str(status))}]</span> {ref_badge}</div>
              <div class="tp-meta">Confidence (Tech): <b>{html.escape(str(conf_tech))}</b> | R:R: <b>{html.escape(str(rr_disp))}</b> ({rr_label})</div>
              <div class="tp-levels">
                <span>Entry: <b>{_fmt_px(entry)}</b></span>
                <span>Stop: <b>{_fmt_px_with_delta(stop, entry)}</b></span>
                <span>TP: <b>{_fmt_px_with_delta(tp, entry)}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional per-plan narrative (placed immediately under its plan card)
        key = str(name)
        expl_html = expl_map.get(key, "")
        if (not expl_html) and (fallback_expl) and idx == 0:
            expl_html = fallback_expl
        if expl_html:
            st.markdown(
                f"""
                <div class="tp-expl">{expl_html}</div>
                """,
                unsafe_allow_html=True,
            )

    # Close visual container and then show the numeric snapshot
    st.markdown("</div>", unsafe_allow_html=True)
    _render_rr_snapshot()

def render_decision_layer_switch(character_pack: Dict[str, Any], analysis_pack: Dict[str, Any], gate_status: str, exec_mode_text: str, preferred_plan: str) -> None:
    """Renderer-only: central Decision Layer switch (no scoring / rule changes).

    v7.1 renderer improvements:
      - Humanize PROTECH labels (replace underscores, add spacing)
      - Convert long BiasCode into short pills (avoid unreadable mega-string)
      - No red accents (orange/neutral only)
    """
    cp = character_pack or {}
    ap = analysis_pack or {}

    def _humanize(s: Any) -> str:
        s0 = _safe_text(s).strip()
        if not s0:
            return ""
        # Replace common separators
        s0 = s0.replace("__", " | ")
        s0 = s0.replace("_", " ")
        s0 = re.sub(r"\s+", " ", s0).strip()
        return s0

    def _bias_code_to_pills(code: str) -> List[str]:
        code = _safe_text(code).strip()
        if not code or code.upper() == "N/A":
            return []
        parts = re.split(r"__+", code)
        out: List[str] = []
        for p in parts:
            p = _humanize(p)
            if not p:
                continue
            # Compact common verbose fragments (display-only)
            p = p.replace("Hist=", "Hist: ").replace("Zero=", "Zero: ").replace("Cross=", "Cross: ").replace("Align=", "Align: ")
            p = p.replace("MACD Above Signal", "MACD > Signal")
            p = p.replace("Expanding Positive", "Expanding +")
            p = p.replace("CrossUp", "Cross Up")
            p = re.sub(r"\s+", " ", p).strip()
            out.append(p)
        # Keep it readable: show top N pills only
        return out[:8]

    conv = cp.get("Conviction") or {}
    tier = conv.get("Tier", "N/A")
    pts = _safe_float(conv.get("Points"), default=np.nan)
    guide = _safe_text(conv.get("SizeGuidance") or "").strip()

    # --- Effective execution gating vs Primary Setup status (renderer-only) ---
    primary_status = ""
    try:
        ps = ap.get("PrimarySetup") or {}
        if isinstance(ps, dict):
            primary_status = _safe_text(ps.get("Status")).strip()
        if not primary_status:
            tps = ap.get("TradePlans") or []
            if isinstance(tps, list):
                for p in tps:
                    if _safe_text(p.get("Name")).strip() == preferred_plan:
                        primary_status = _safe_text(p.get("Status")).strip()
                        break
    except Exception:
        primary_status = ""

    pstat = primary_status.upper().strip()

    effective_exec_mode_text = exec_mode_text
    effective_preferred_plan = preferred_plan
    effective_guide = guide

    # If overall mode is ACTIVE but the primary plan is still WATCH, downgrade wording to avoid "full size now" confusion.
    if _safe_text(gate_status).upper().strip() == "ACTIVE":
        if pstat in ("WATCH", "PENDING", "WAIT", "WAITING", "TRACK"):
            effective_exec_mode_text = "ACTIVE (WAIT ENTRY) – được phép triển khai khi plan kích hoạt/đạt điều kiện vào lệnh."
            if effective_guide:
                # Soften aggressive sizing language: full size only when plan becomes Active/triggered
                effective_guide = effective_guide.replace("FULL SIZE + CÓ THỂ ADD", "FULL SIZE (KHI PLAN ACTIVE) + CÓ THỂ ADD (SAU FOLLOW-THROUGH)")
                if "FULL SIZE" in effective_guide and "(KHI PLAN ACTIVE)" not in effective_guide:
                    effective_guide = effective_guide.replace("FULL SIZE", "FULL SIZE (KHI PLAN ACTIVE)")
            else:
                effective_guide = "EDGE MẠNH — FULL SIZE (KHI PLAN ACTIVE) + CÓ THỂ ADD (SAU FOLLOW-THROUGH)"
        elif pstat in ("INVALID", "DISABLED", "N/A", "NA"):
            effective_exec_mode_text = "WATCH ONLY – primary plan hiện không hợp lệ; ưu tiên quan sát."
            effective_guide = "NO TRADE / WAIT RESET"


    # --- Sanity-check layer (renderer only) ---
    # Prevent contradictory messaging between plan status and sizing guidance.
    try:
        gs2 = _safe_text(gate_status).upper().strip()
        if gs2 != "ACTIVE":
            # In non-ACTIVE modes, never suggest aggressive sizing.
            if effective_guide:
                if "FULL SIZE" in effective_guide:
                    effective_guide = "WATCH MODE — chưa vào lệnh; chỉ chuẩn bị kế hoạch và chờ điều kiện kích hoạt."
                elif "ADD" in effective_guide or "Pyramid" in effective_guide:
                    effective_guide = "WATCH MODE — không gia tăng; chỉ theo dõi theo Trade Plan."
        else:
            # ACTIVE but entry not triggered -> gate sizing to 'when plan active'
            if pstat in ("WATCH", "PENDING", "WAIT", "WAITING", "TRACK"):
                if effective_guide and "FULL SIZE" in effective_guide and "(KHI PLAN ACTIVE)" not in effective_guide:
                    effective_guide = effective_guide.replace("FULL SIZE", "FULL SIZE (KHI PLAN ACTIVE)")
                # Ensure add/pyramid is conditional
                if effective_guide and "CÓ THỂ ADD" in effective_guide and "FOLLOW-THROUGH" not in effective_guide:
                    effective_guide = effective_guide.replace("CÓ THỂ ADD", "CÓ THỂ ADD (SAU FOLLOW-THROUGH)")
            # If primary is invalid, force conservative messaging
            if pstat in ("INVALID", "DISABLED", "N/A", "NA"):
                effective_exec_mode_text = "WATCH ONLY – primary plan hiện không hợp lệ; ưu tiên quan sát."
                effective_guide = "NO TRADE / WAIT RESET"
    except Exception:
        pass

    def _conv_cls_from_tier(t: object) -> str:
        try:
            ti = int(t)
        except Exception:
            return "conv-unknown"
        if ti <= 1:
            return "conv-noedge"
        if ti == 2:
            return "conv-weak"
        if ti == 3:
            return "conv-tradeable"
        if ti in (4, 5):
            return "conv-strong"
        if ti == 6:
            return "conv-high"
        return "conv-god"

    guide_upper = effective_guide.upper() if effective_guide else ""
    guide_cls = _conv_cls_from_tier(tier)
    guide_html = f"<span class='conv-tag {guide_cls}'>{html.escape(guide_upper)}</span>" if guide_upper else " "

    # Final Bias comes from ProTech.Bias (fact-only layer)
    bias = ((ap.get("ProTech") or {}).get("Bias") or {}) if isinstance(ap, dict) else {}
    alignment_raw = _safe_text(bias.get("Alignment") or "N/A").strip()
    bias_code_raw = _safe_text(bias.get("BiasCode") or "").strip()

    alignment = _humanize(alignment_raw) or "N/A"
    bias_pills = _bias_code_to_pills(bias_code_raw)

    # Level mapping for background (layout only)
    lvl = "low"
    if pd.notna(pts):
        if pts >= 5.0:
            lvl = "high"
        elif pts >= 3.0:
            lvl = "med"

    flags = cp.get("Flags") or []
    tags = cp.get("ActionTags") or []

    # Translate tags (EN → EN/VI) where possible
    tags_vi: List[str] = []
    for t in tags:
        t0 = str(t)
        tags_vi.append(PLAYSTYLE_TAG_TRANSLATIONS.get(t0, t0))

    st.markdown('<div class="dl-wrap">', unsafe_allow_html=True)

    
    # Hero card: Conviction + Size Guidance (FINAL BIAS hidden by UI policy)
    pts_disp = f"{pts:.1f}" if pd.notna(pts) else "N/A"
    st.markdown(
        f"""
        <div class="dl-grid">
          <div class="dl-card {lvl}">
            <div class="dl-k">CONVICTION SCORE</div>
            <div class="dl-v">Tier {html.escape(str(tier))}/7  •  {html.escape(pts_disp)} pts</div>
            <div class="dl-sub"><b>Execution Mode:</b> {html.escape(effective_exec_mode_text)}<br><b>Preferred Plan:</b> {html.escape(effective_preferred_plan)}</div>
            <div class="dl-sub">{guide_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weaknesses / Flags (short list)
    if isinstance(flags, list) and flags:
        st.markdown('<div class="dl-sec"><div class="dl-sec-t">YẾU ĐIỂM VÀ RỦI RO CHÍNH CỦA TRADE PLAN</div>', unsafe_allow_html=True)
        for f in flags[:6]:
            if not isinstance(f, dict):
                continue
            try:
                sev = int(f.get("severity", 1))
            except Exception:
                sev = 1
            note = _safe_text(f.get("note", ""))
            code = _safe_text(f.get("code", ""))
            st.markdown(
                f"""<div class="dl-flag"><span class="dl-sev">S{sev}</span><span class="dl-code">{html.escape(code)}</span><span class="dl-note">{html.escape(note)}</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Playstyle tags (bilingual pills)
    if tags_vi:
        st.markdown('<div class="dl-sec"><div class="dl-sec-t">ĐỀ XUẤT XU HƯỚNG HÀNH ĐỘNG</div>', unsafe_allow_html=True)
        def _play_hint(tag: str) -> str:
            tl = (tag or '').lower()
            if 'breakout' in tl:
                return 'Ưu tiên chờ phiên xác nhận và follow-through; tránh vào sớm khi chưa có lực phá vỡ rõ ràng.'
            if 'pullback' in tl:
                return 'Ưu tiên canh hồi về vùng hỗ trợ/MA/Fib để tối ưu điểm vào; tránh đuổi giá khi chưa hồi.'
            if ('longstructure' in tl and 'shorttactical' in tl) or ('long structure' in tl and 'short tactical' in tl):
                return 'Khung dài hạn quyết định bias; tác chiến ngắn hạn chỉ để tối ưu entry/exit theo đúng cấu trúc.'
            if 'trend' in tl:
                return 'Ưu tiên đi theo xu hướng; chỉ vào khi MA/cấu trúc ủng hộ và không vi phạm stop.'
            return 'Tag này gợi ý cách hành động phù hợp bối cảnh hiện tại; làm theo để đồng bộ Trade Plan và giảm sai nhịp.'

        items = []
        for t in tags_vi[:10]:
            t1 = str(t)
            hint = _play_hint(t1)
            items.append(
                f"<div class='dl-tagitem'><span class='dl-pill'>{html.escape(t1)}</span><div class='dl-taghint'>{html.escape(hint)}</div></div>"
            )
        st.markdown("<div class='dl-tags'>" + "".join(items) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return

def render_appendix_e(result: Dict[str, Any], report_text: str, analysis_pack: Dict[str, Any]) -> None:
    """
    Decision Layer Report — Anti-Anchoring Output Order (layout only):
      1) Stock DNA (Traits)
      2) CURRENT STATUS (Scenario + Technical + Fundamental)
      3) Trade Plan & R:R (Conditional)
      4) Decision Layer (Conviction/Weakness/Tags)
    This renderer must not change any underlying calculations.
    """
    modules = (result or {}).get("Modules") or {}
    cp = modules.get("character") or {}
    ap = analysis_pack or {}

    # Trade-plan gate (for execution / anti-FOMO posture)
    gate_status, _meta = _trade_plan_gate(analysis_pack, cp)

    # ---------- HEADER: <Ticker> — <Last Close> <+/-%> ----------
    ticker = _safe_text(ap.get("Ticker") or (result or {}).get("Ticker") or "").strip().upper()
    last_pack = ap.get("Last") or {}
    close_val = _safe_float(last_pack.get("Close"), default=np.nan)

    mkt = ap.get("Market") or {}
    stock_chg = _safe_float(mkt.get("StockChangePct"), default=np.nan)

    def _fmt_close(x: Any) -> str:
        try:
            v = float(x)
            if not math.isfinite(v):
                return "N/A"
            return f"{v:.2f}"
        except Exception:
            return "N/A"

    def _fmt_change_pct(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return ""
        return f"{v:+.2f}%"

    price_str = _fmt_close(close_val)
    chg_str = _fmt_change_pct(stock_chg)

    if ticker or price_str != "N/A":
        header = ticker or ""
        if price_str != "N/A":
            header = f"{header} — {price_str}" if header else price_str
        if chg_str:
            header = f"{header} ({chg_str})"
        render_executive_snapshot(ap, cp, gate_status)

    # Pre-split legacy report once for reuse
    exp_label = "BẤM ĐỂ XEM CHI TIẾT PHÂN TÍCH & BIỂU ĐỒ"
    exp_default = True if (gate_status or "").strip().upper() == "ACTIVE" else False
    with st.expander(exp_label, expanded=exp_default):
        left_col, right_col = st.columns([0.68, 0.32], gap="large")
        with left_col:
            sections = _split_sections(report_text or "")
            a_section = sections.get("A", "") or ""
            b_section = sections.get("B", "") or ""
            c_section = sections.get("C", "") or ""
            d_section = sections.get("D", "") or ""
    
            # ============================================================
            # 1) STOCK DNA (CORE STATS – TRAITS)
            # ============================================================
            st.markdown('<div class="major-sec">STOCK DNA</div>', unsafe_allow_html=True)
            render_character_traits(cp)
            render_stock_dna_insight(cp)
    
            # ============================================================
            # 2) CURRENT STATUS
            # ============================================================
            st.markdown('<div class="major-sec">CURRENT STATUS</div>', unsafe_allow_html=True)

            # 2.1 Relative Strength vs VNINDEX
            rel = (ap.get("Market") or {}).get("RelativeStrengthVsVNINDEX")
            st.markdown(f"**Relative Strength vs VNINDEX:** {_val_or_na(rel)}")

            # 2.2 Scenario & Scores
            scenario_pack = ap.get("Scenario12") or {}
            master_pack = ap.get("MasterScore") or {}
            conviction_score = ap.get("Conviction")

            st.markdown("**State Capsule (Scenario & Scores)**")
            st.markdown(f"- Scenario: {_val_or_na(scenario_pack.get('Name'))}")

            def _bar_row_cs(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "N/A"
                else:
                    v = float(v)
                    pct = _clip(v / maxv * 100.0, 0.0, 100.0)
                    v_disp = f"{v:.1f}/{maxv:.0f}"
                st.markdown(
                    f"""
                    <div class="gc-row">
                      <div class="gc-k">{html.escape(str(label))}</div>
                      <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
                      <div class="gc-v">{html.escape(str(v_disp))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            _bar_row_cs("Điểm tổng hợp", master_pack.get("Total"), 10.0)
            _bar_row_cs("Điểm tin cậy", conviction_score, 10.0)

            # Score interpretation (single block) — place directly under the two bars
            render_current_status_insight(master_pack.get("Total"), conviction_score, gate_status)

            # Class Policy Hint (display-only)
            _final_class = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "").strip()
            _policy_hint = get_class_policy_hint_line(_final_class)
            if _policy_hint:
                st.markdown(f"**Policy:** {_policy_hint}")

            # 2.3 State Capsule (Facts-only, compact)
            st.markdown("**Structure Summary (MA/Fibo/RSI/MACD/Volume)**")
            protech = ap.get("ProTech") or {}
            protech = protech if isinstance(protech, dict) else {}
            ma = protech.get("MA") or {}
            ma = ma if isinstance(ma, dict) else {}
            rsi = protech.get("RSI") or {}
            rsi = rsi if isinstance(rsi, dict) else {}
            macd = protech.get("MACD") or {}
            macd = macd if isinstance(macd, dict) else {}
            vol = protech.get("Volume") or {}
            vol = vol if isinstance(vol, dict) else {}
            bias = protech.get("Bias") or {}
            bias = bias if isinstance(bias, dict) else {}

            fib_ctx = ((ap.get("Fibonacci") or {}).get("Context") or {})
            fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}

            ma_reg = _safe_text(ma.get("Regime"))
            rsi_zone = _safe_text(rsi.get("State"))
            rsi_dir = _safe_text(rsi.get("Direction"))
            macd_rel = _safe_text(macd.get("State"))
            macd_zero = _safe_text(macd.get("ZeroLine"))
            align = _safe_text(bias.get("Alignment"))

            short_band = _safe_text(fib_ctx.get("ShortBand"))
            long_band = _safe_text(fib_ctx.get("LongBand"))
            fib_conflict = bool(fib_ctx.get("FiboConflictFlag"))

            vol_ratio = _safe_float(vol.get("Ratio"), default=np.nan)

            st.markdown(f"- MA Structure: {_val_or_na(ma_reg)}")
            st.markdown(f"- RSI: {_val_or_na(rsi_zone)} | {_val_or_na(rsi_dir)}")
            st.markdown(f"- MACD: {_val_or_na(macd_rel)} | ZeroLine: {_val_or_na(macd_zero)}")
            st.markdown(f"- RSI+MACD Alignment: {_val_or_na(align)}")
            st.markdown(f"- Fibonacci Bands (Short/Long): {_val_or_na(short_band)} / {_val_or_na(long_band)}" + (" | Conflict" if fib_conflict else ""))
            st.markdown(f"- Volume Ratio (vs 20d): {_val_or_na(vol_ratio)}")

            # 2.4 TECHNICAL SNAPSHOT (details)
            # 2.4 TECHNICAL SNAPSHOT (detail) (reuse A-section body: MA/Fibo/RSI/MACD/Volume/PA)
            st.markdown('<div class="sec-title">TECHNICAL SNAPSHOT</div>', unsafe_allow_html=True)
            a_items = _extract_a_items(a_section)
            a_raw = (a_section or "").replace("\r\n", "\n")
            a_body = re.sub(r"(?mi)^A\..*\n?", "", a_raw).strip()
            if a_items:
                for i, body in enumerate(a_items, start=1):
                    if not body.strip():
                        continue
                    st.markdown(
                        f"""
                        <div class="incept-card">
                          <div style="font-weight:800; margin-bottom:6px;">{i}.</div>
                          <div>{body}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                if a_body:
                    st.markdown(a_body, unsafe_allow_html=False)
                else:
                    st.info("N/A")



            # 2.5 Combat Readiness (Now) — merged from legacy Combat Stats
            st.markdown("**Combat Readiness (Now)**")
            combat = cp.get("CombatStats") or {}
            combat = combat if isinstance(combat, dict) else {}

            def _bar_row_now(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "N/A"
                else:
                    v = float(v)
                    pct = _clip(v / maxv * 100.0, 0.0, 100.0)
                    v_disp = f"{v:.1f}/{maxv:.0f}"
                st.markdown(
                    f"""
                    <div class="gc-row">
                      <div class="gc-k">{html.escape(str(label))}</div>
                      <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
                      <div class="gc-v">{html.escape(str(v_disp))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            _bar_row_now("Upside Room", combat.get("UpsideRoom"), 10.0)
            _bar_row_now("Upside Quality", combat.get("UpsideQuality"), 10.0)
            _bar_row_now("Downside Safety", combat.get("DownsideRisk"), 10.0)
            _bar_row_now("R:R Efficiency", combat.get("RREfficiency"), 10.0)
            _bar_row_now("Breakout Force", combat.get("BreakoutForce"), 10.0)
            _bar_row_now("Support Resilience", combat.get("SupportResilience"), 10.0)

            # 2.6 Trigger Status (Plan-Gated)
            st.markdown("**Trigger Status (Plan-Gated)**")
            primary = ap.get("PrimarySetup") or {}
            primary = primary if isinstance(primary, dict) else {}
            setup_name = _safe_text(primary.get("Name")).strip()
            rr_val = _safe_float(primary.get("RR"), default=np.nan)

            plan_status = "N/A"
            plan_tags: List[str] = []
            for p in (ap.get("TradePlans") or []):
                if _safe_text(p.get("Name")).strip() == setup_name and setup_name and setup_name != "N/A":
                    plan_status = _safe_text(p.get("Status") or "N/A")
                    plan_tags = list(p.get("ReasonTags") or [])
                    rr_val = _safe_float(p.get("RR"), default=rr_val)
                    break

            def _status_from_val(v: Any, good: float, warn: float) -> Tuple[str, str]:
                x = _safe_float(v, default=np.nan)
                if pd.isna(x):
                    return ("N/A", "#9CA3AF")
                if x >= good:
                    return ("PASS", "#22C55E")
                if x >= warn:
                    return ("WAIT", "#F59E0B")
                return ("FAIL", "#EF4444")

            def _dot(color: str) -> str:
                return f'<span class="es-dot" style="background:{color};"></span>'

            s_break, c_break = _status_from_val(combat.get("BreakoutForce"), good=6.8, warn=5.5)
            s_vol, c_vol = _status_from_val(vol_ratio, good=1.20, warn=0.95)
            s_rr, c_rr = _status_from_val(rr_val, good=1.80, warn=1.30)

            # Structure (Ceiling) Gate
            sq = ap.get("StructureQuality", {}) if isinstance(ap, dict) else {}
            cg = ((sq or {}).get("Gates", {}) or {}).get("CeilingGate", {}) if isinstance((sq or {}).get("Gates", {}), dict) else {}
            s_struct = _safe_text(cg.get("Status") or "N/A").strip().upper()
            if s_struct not in ("PASS", "WAIT", "FAIL"):
                s_struct = "N/A"
            c_struct = "#9CA3AF"
            if s_struct == "PASS": c_struct = "#22C55E"
            elif s_struct == "WAIT": c_struct = "#F59E0B"
            elif s_struct == "FAIL": c_struct = "#EF4444"

            st.markdown(
                f"""<ul style="margin:0 0 0 16px; padding:0;">
                      <li>{_dot(c_break)} Breakout: {s_break}</li>
                      <li>{_dot(c_vol)} Volume: {s_vol}</li>
                      <li>{_dot(c_rr)} R:R: {s_rr}</li>
                      <li>{_dot(c_struct)} Structure: {s_struct}</li>
                      <li>{_dot("#60A5FA")} Gate: {html.escape(str(gate_status or "N/A"))} | Plan: {html.escape(str(setup_name or "N/A"))} ({html.escape(str(plan_status or "N/A"))})</li>
                    </ul>""",
                unsafe_allow_html=True
            )

            if plan_tags:
                tags_show = ", ".join([t for t in plan_tags if isinstance(t, str) and t.strip()][:6])
                if tags_show:
                    st.caption(f"Plan tags: {tags_show}")

            # 2.7 Risk Flags (from weakness flags + DNA modifiers)
            st.markdown("**Risk Flags**")
            flags = list(cp.get("Flags") or [])
            risk_lines = []
            for f in flags:
                try:
                    sev = int(f.get("severity", 1))
                except Exception:
                    sev = 1
                if sev >= 2:
                    note = _safe_text(f.get("note") or "").strip()
                    code = _safe_text(f.get("code") or "Flag").strip()
                    risk_lines.append(f"- [{code}] {note}" if note else f"- [{code}]")

            dna_t1 = (((cp.get("StockTraits") or {}).get("DNA") or {}).get("Tier1") or {})
            mods = dna_t1.get("Modifiers") if isinstance(dna_t1, dict) else []
            if isinstance(mods, list) and mods:
                mods_txt = ", ".join([str(x) for x in mods[:6]])
                risk_lines.append(f"- [DNA Modifiers] {mods_txt}")

            if risk_lines:
                st.markdown("\n".join(risk_lines))
            else:
                st.markdown("- None")

            # ============================================================
            # 3) TRADE PLAN & R:R (CONDITIONAL)
            # ============================================================
            st.markdown('<div class="major-sec">TRADE PLAN &amp; R:R</div>', unsafe_allow_html=True)
            # Pass legacy C-section body to the trade-plan renderer so explanation
            # lives next to the numeric setup cards.
            c_body_clean = ""
            if c_section:
                c_raw = c_section.replace("\r\n", "\n")
                c_body_clean = re.sub(r"(?m)^C\..*\n?", "", c_raw).strip()
            render_trade_plan_conditional(analysis_pack, gate_status, c_body_clean)
            # ============================================================
            # 4) DECISION LAYER (CONVICTION, WEAKNESSES, PLAYSTYLE TAGS)
            # ============================================================
            # Central switch — layout only (no scoring / rule changes)
            primary_setup = (ap.get("PrimarySetup") or {}) if isinstance(ap, dict) else {}
            primary_name = _val_or_na(primary_setup.get("Name"))
    
            if gate_status == "LOCK":
                exec_mode_text = "WATCH ONLY – chưa kích hoạt lệnh mới (ưu tiên quan sát / bảo toàn vốn)."
            elif gate_status == "ACTIVE":
                exec_mode_text = "ACTIVE – được phép triển khai kế hoạch giao dịch theo điều kiện đã nêu."
            else:
                exec_mode_text = "WATCH ONLY – setup mang tính tham khảo, chờ thêm tín hiệu xác nhận."
    
            st.markdown('<div class="major-sec">DECISION LAYER</div>', unsafe_allow_html=True)
    
            render_decision_layer_switch(cp, ap, gate_status, exec_mode_text, primary_name)
    
    
    
        with right_col:
            st.markdown("""<div style='border:1px dashed #E5E7EB;border-radius:14px;padding:14px;color:#64748B;font-weight:800;'>BIỂU ĐỒ (SẼ BỔ SUNG)</div>""", unsafe_allow_html=True)
    # ============================================================
    # 11. GPT-4o STRATEGIC INSIGHT GENERATION
    # ============================================================


# ------------------------------------------------------------
# Deterministic Report A–D (Facts-only fallback)
# - Used when OPENAI_API_KEY is missing OR GPT call fails.
# - Ensures A–D sections always exist so UI split/render stays stable.
# ------------------------------------------------------------

def _deterministic_report_ad(data: Dict[str, Any], note: str = "") -> str:
    ap = data.get("AnalysisPack", {}) or {}
    ap = ap if isinstance(ap, dict) else {}

    protech = ap.get("ProTech") or {}
    protech = protech if isinstance(protech, dict) else {}

    ma = protech.get("MA") or {}
    ma = ma if isinstance(ma, dict) else {}
    rsi = protech.get("RSI") or {}
    rsi = rsi if isinstance(rsi, dict) else {}
    macd = protech.get("MACD") or {}
    macd = macd if isinstance(macd, dict) else {}
    vol = protech.get("Volume") or {}
    vol = vol if isinstance(vol, dict) else {}
    pa = protech.get("PriceAction") or {}
    pa = pa if isinstance(pa, dict) else {}
    bias = protech.get("Bias") or {}
    bias = bias if isinstance(bias, dict) else {}

    fib_ctx = ((ap.get("Fibonacci") or {}).get("Context") or {})
    fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}

    master = ap.get("MasterScore") or {}
    master = master if isinstance(master, dict) else {}
    conv_pack = ap.get("ConvictionPack") or {}
    conv_pack = conv_pack if isinstance(conv_pack, dict) else {}

    primary = ap.get("PrimarySetup") or {}
    primary = primary if isinstance(primary, dict) else {}

    # --- Facts / labels ---
    ma_reg = _val_or_na(ma.get("Regime"))
    rsi_state = _val_or_na(rsi.get("State"))
    rsi_dir = _val_or_na(rsi.get("Direction"))
    rsi_div = _safe_text(rsi.get("Divergence")).strip()

    macd_state = _val_or_na(macd.get("State"))
    macd_zero = _val_or_na(macd.get("ZeroLine"))
    macd_hist = _safe_text(macd.get("HistState")).strip()

    align = _val_or_na(bias.get("Alignment"))

    short_band = _val_or_na(fib_ctx.get("ShortBand"))
    long_band = _val_or_na(fib_ctx.get("LongBand"))
    fib_conflict = bool(fib_ctx.get("FiboConflictFlag"))

    vol_ratio = _safe_float(vol.get("Ratio"), default=np.nan)
    vol_ratio_txt = _val_or_na(vol_ratio)
    vol_regime = _safe_text(vol.get("Regime")).strip()

    pa_patterns = pa.get("Patterns")
    if isinstance(pa_patterns, list):
        pa_pat_txt = ", ".join([_safe_text(x).strip() for x in pa_patterns if _safe_text(x).strip()][:2])
    else:
        pa_pat_txt = _safe_text(pa.get("Pattern")).strip()

    scenario12 = ap.get("Scenario12") or {}
    scenario12 = scenario12 if isinstance(scenario12, dict) else {}
    sc_name = _val_or_na(scenario12.get("Name"))

    ms_total = _safe_float(master.get("Total"), default=np.nan)
    ms_txt = _val_or_na(ms_total)
    conv_score = _safe_float(conv_pack.get("Score"), default=np.nan)
    conv_txt = _val_or_na(conv_score)

    # --- Trade plan (facts-only) ---
    setup_name = _safe_text(primary.get("Name")).strip() or "N/A"
    rr = _safe_float(primary.get("RR"), default=np.nan)
    rr_txt = _val_or_na(rr)

    plan_status = "N/A"
    plan_tags = []
    for p in (ap.get("TradePlans") or []):
        if not isinstance(p, dict):
            continue
        if _safe_text(p.get("Name")).strip() == setup_name and setup_name != "N/A":
            plan_status = _val_or_na(p.get("Status"))
            plan_tags = list(p.get("ReasonTags") or [])
            rr2 = _safe_float(p.get("RR"), default=np.nan)
            if pd.notna(rr2):
                rr_txt = _val_or_na(rr2)
            break

    tags_preview = ", ".join([str(x) for x in plan_tags[:6]]) if plan_tags else "N/A"

    # D (strict copy) - do not compute
    risk = _val_or_na(primary.get("RiskPct"))
    reward = _val_or_na(primary.get("RewardPct"))
    must_rr = _val_or_na(primary.get("RR"))
    must_conf = _val_or_na(primary.get("Confidence (Tech)", primary.get("Probability")))

    # Build sections (use 1–2 numbers per sentence max)
    lines = []
    if note:
        lines.append(note.strip())
        lines.append("")

    lines += [
        "A. Kỹ thuật",
        f"1. MA: {ma_reg}.",
        f"2. RSI: {rsi_state} | {rsi_dir}." + (f" Divergence: {_safe_text(rsi_div)}." if rsi_div else ""),
        f"3. MACD: {macd_state} | ZeroLine: {macd_zero}." + (f" Hist: {_safe_text(macd_hist)}." if macd_hist else ""),
        f"4. RSI+MACD alignment: {align}.",
        f"5. Fibonacci bands (Short/Long): {short_band} / {long_band}." + (" Conflict flagged." if fib_conflict else ""),
        f"6. Volume: Ratio {vol_ratio_txt}." + (f" Regime: {_safe_text(vol_regime)}." if vol_regime else ""),
        f"7. Scenario12: {sc_name}.",
        f"8. Master/Conviction: {ms_txt} | {conv_txt}.",
        "",
        "B. Cơ bản",
        "(Chỉ hiển thị khi có dữ liệu cơ bản trong pack.)",
        "",
        "C. TRADE PLAN",
        f"Primary setup: {setup_name} | Status: {plan_status} | RR: {rr_txt}.",
        f"Plan tags: {tags_preview}.",
        "",
        "D. Rủi ro vs lợi nhuận",
        f"Risk%: {risk}",
        f"Reward%: {reward}",
        f"RR: {must_rr}",
        f"Confidence (Tech): {must_conf}",
    ]

    return "\n".join(lines).strip() + "\n"

def generate_insight_report(data: Dict[str, Any]) -> str:
    if "Error" in data: return f"❌ {data['Error']}"
    tick = data["Ticker"]
    scenario = data["Scenario"]
    conviction = data["Conviction"]
    analysis_pack = data.get("AnalysisPack", {})
    last = data["Last"]
    close = _fmt_price(last.get("Close"))
    header_html = f"<h2 style='margin:0; padding:0; font-size:26px; line-height:1.2;'>{tick} — {close} | Điểm tin cậy: {conviction:.1f}/10 | {_scenario_vi(scenario)}</h2>"
    fund = data.get("Fundamental", {})
    fund_text = (
        f"Khuyến nghị: {fund.get('Recommendation', 'N/A')} | "
        f"Giá mục tiêu: {_fmt_thousand(fund.get('Target'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if fund else "Không có dữ liệu cơ bản"
    )
    pack_json = safe_json_dumps_strict(analysis_pack)
    primary = (analysis_pack.get("PrimarySetup") or {})
    must_risk = primary.get("RiskPct")
    must_reward = primary.get("RewardPct")
    must_rr = primary.get("RR")
    must_conf = primary.get("Confidence (Tech)", primary.get("Probability"))
    # ============================================================
    # STEP 10 — PROMPT v10 (Narrative Refinement)
    # - Context → Impact → Action
    # - Each sentence: max 1–2 numbers
    # - Use ContextPacks to avoid dry tags
    # - FUNDAMENTAL LOCK: only allowed in section B
    # - Keep C + D contiguous; D must copy PrimarySetup
    # ============================================================
    prompt = f"""
    Bạn là "INCEPTION Narrative Editor" cho báo cáo phân tích cổ phiếu.
    Vai trò của bạn: DIỄN GIẢI + BIÊN TẬP văn phong từ JSON "AnalysisPack".
    TUYỆT ĐỐI:
    - Không bịa số, không ước lượng, không tự tính.
    - Chỉ được dùng đúng con số có sẵn trong JSON.
    - Không nói "theo tôi tính", không suy ra số mới từ số cũ.
    
    RÀNG BUỘC QUAN TRỌNG (FUNDAMENTAL LOCK):
    - Fundamental (Recommendation/Target/Upside/broker...) CHỈ ĐƯỢC NHẮC Ở MỤC B.
    - Ở A/C/D: CẤM nhắc Target/Upside/Recommendation hoặc bất kỳ ý "định giá/cơ bản" nào.
    - Nếu bạn lỡ viết Fundamental ở A/C/D => sai nhiệm vụ.
    
    YÊU CẦU FORMAT OUTPUT:
    - Không emoji.
    - Không bullet kiểu 1️⃣ 2️⃣.
    - Bắt buộc đúng 4 mục A–D với cấu trúc:
    
    A. Kỹ thuật
    1) ...
    2) ...
    3) ...
    4) ...
    5) ...
    6) ...
    7) ...
    8) ...
    
    B. Cơ bản
    (chỉ 1–3 câu, dùng đúng dòng dữ liệu đã cung cấp)
    
    C. TRADE PLAN
    (viết ngắn gọn 5–9 câu)
    
    D. Rủi ro vs lợi nhuận
    Risk%: ...
    Reward%: ...
    RR: ...
    Confidence (Tech): ...
    
    QUY TẮC VĂN PHONG (chống "khô"):
    - Mỗi mục (A1→A8) viết 2–4 câu theo mẫu: (Bối cảnh) → (Tác động) → (Hành động).
    - Mỗi câu tối đa 1–2 con số (vd: RSI, khoảng cách %, RR). Không nhồi nhiều số trong 1 câu.
    - Không liệt kê tags thô. Nếu cần nhắc tag, hãy chuyển thành ý nghĩa hành động.
    - Tránh kết luận cứng “mua/bán/tốt/xấu”. Dùng ngôn ngữ điều kiện: “thiên về”, “ưu tiên”, “nếu/ khi”.
    - Ưu tiên 2–3 điểm quan trọng nhất thay vì kể hết.
    
    HƯỚNG DẪN KHAI THÁC CONTEXT PACKS (bắt buộc tận dụng để viết mượt):
    - RSIContext: Streak70, Cross70BarsAgo, Delta3/Delta5, Turning
      * Nếu RSI>=70: không gọi là “quá mua” mặc định. Diễn giải theo 2 khả năng:
        (i) “trend strength” nếu Turning không suy yếu + MACD/hist không xấu + volume không rơi;
        (ii) “exhaustion risk” nếu Turning giảm + nến/PA có doji/shooting star + volume lệch.
    - VolumeContext: VolStreakUp, VolTrend; ProTech.Volume: Ratio/Regime
      * Dùng để nói “có/không có xác nhận dòng tiền”, tránh phán đoán cảm tính.
    - LevelContext + Fibonacci.Context:
      * Nêu nearest support/resistance và DistPct (chỉ 1–2 mức quan trọng).
      * Nếu FiboConflictFlag=True: áp dụng luật "LongStructure_ShortTactical" để giải thích:
        long = khung cấu trúc/ceiling-floor; short = tác chiến entry.
    - Market: VNINDEX/VN30 + RelativeStrengthVsVNINDEX
      * Gắn ngắn 1 câu: cổ phiếu mạnh/yếu hơn thị trường theo tag rel (không tự tính).
    
    NỘI DUNG A (8 mục):
    1) MA Trend: dùng ProTech.MA (Regime, SlopeMA20/50/200, DistToMA50/200, Cross.*)
    2) RSI: dùng ProTech.RSI + RSIContext (State/Direction/Divergence/Streak/Turning)
    3) MACD: dùng ProTech.MACD (State/Cross/ZeroLine/HistState/Divergence)
    4) RSI + MACD Bias: dùng ProTech.Bias (Alignment/Facts); diễn giải như “đang đồng pha/lệch pha”
    5) Fibonacci: ShortWindow/LongWindow + SelectionReason + Fibonacci.Context (Nearest/Dist/FiboConflict)
    6) Volume & Price Action: ProTech.Volume + ProTech.PriceAction (Patterns/VolumeRegime/NearMA/NearFib)
    7) Scenario 12: Scenario12 (Name/Flags/RulesHit) → diễn giải theo bối cảnh (không kết luận cứng)
    8) Master Integration: MasterScore.Total + ConvictionPack.Score + Components (chỉ nêu 2–3 điểm đóng góp lớn)
    
    MỤC B (FUNDAMENTAL — chỉ được dùng đúng 1 dòng này, không thêm suy luận):
    {fund_text}
    
    MỤC C (TRADE PLAN):
    - Dùng TradePlans trong JSON.
    - Ưu tiên plan trùng PrimarySetup.Name (Breakout/Pullback) để viết trước.
    - Trình bày theo: điều kiện kích hoạt (Status/Trigger tags) → vùng Entry → Stop (neo level + buffer) → TP → khi nào hủy plan.
    - Không tự tính RR; nếu cần RR/Risk/Reward thì chỉ nhắc đúng số đã có trong PrimarySetup hoặc TradePlans.
    
    RÀNG BUỘC LIỀN MẠCH:
    - Mục C kết thúc xong phải in NGAY mục D (4 dòng), không chèn thêm đoạn giải thích.
    
    KHÓA CỨNG MỤC D (COPY ĐÚNG, không tự tính/ước lượng):
    - Risk% = {must_risk}
    - Reward% = {must_reward}
    - RR = {must_rr}
    - Confidence (Tech) = {must_conf}
    
    Trong mục D, bắt buộc đúng 4 dòng theo format:
    Risk%: <...>
    Reward%: <...>
    RR: <...>
    Confidence (Tech): <...>
    
    Dữ liệu (AnalysisPack JSON):
    {pack_json}
    """

    # GPT narrative is optional.
    # If OPENAI_API_KEY is missing or GPT call fails, fall back to deterministic A–D
    # to keep UI rendering stable (TECHNICAL SNAPSHOT expects A-section items).
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            api_key = None

    if not api_key:
        content = _deterministic_report_ad(
            data,
            note="⚠️ GPT narrative disabled: OPENAI_API_KEY not set. Using deterministic A–D (facts-only).",
        )
    else:
        try:
            content = call_gpt_with_guard(prompt, analysis_pack, max_retry=2)
        except Exception as e:
            content = _deterministic_report_ad(
                data,
                note=f"⚠️ Lỗi khi gọi GPT: {e}. Using deterministic A–D (facts-only).",
            )
    return f"{header_html}\n\n{content}"

# ============================================================
# 11B. UI HELPERS (PRESENTATION ONLY)
# ============================================================
def _split_sections(report_text: str) -> dict:
    parts = {"A": "", "B": "", "C": "", "D": ""}
    if not report_text:
        return parts
    text = report_text.replace("\r\n", "\n")
    pattern = re.compile(r"(?m)^(A|B|C|D)\.\s")
    matches = list(pattern.finditer(text))
    if not matches:
        parts["A"] = text
        return parts
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        parts[key] = text[start:end].strip()
    return parts

def _extract_a_items(a_text: str) -> list:
    if not a_text:
        return []
    text = a_text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^A\..*\n?", "", text).strip()
    item_pat = re.compile(r"(?ms)^\s*(\d)\.\s*(.*?)(?=^\s*\d\.|\Z)")
    found = item_pat.findall(text)
    items = [""] * 8
    for num, body in found:
        idx = int(num) - 1
        if 0 <= idx < 8:
            items[idx] = body.strip()
    non_empty = sum(1 for x in items if x.strip())
    return items if non_empty >= 4 else []

def render_report_pretty(report_text: str, analysis_pack: dict):
    sections = _split_sections(report_text)
    a_items = _extract_a_items(sections.get("A", ""))

    st.markdown('<div class="incept-wrap">', unsafe_allow_html=True)

    ap = analysis_pack or {}
    scenario_pack = ap.get("Scenario12") or {}
    master_pack = ap.get("MasterScore") or {}
    conviction_score = ap.get("Conviction", "N/A")

    def _val_or_na(v):
        if v is None: return "N/A"
        if isinstance(v, float) and pd.isna(v): return "N/A"
        text = str(v).strip()
        return text if text else "N/A"

    st.markdown(
        f"""
        <div class="report-header">
          <h2 style="margin:0; padding:0;">{_val_or_na(ap.get("Ticker"))} - {_val_or_na(scenario_pack.get("Name"))}</h2>
          <div style="font-size:16px; font-weight:700; margin-top:4px;">
            Điểm tổng hợp: {_val_or_na(master_pack.get("Total"))} | Điểm tin cậy: {_val_or_na(conviction_score)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sec-title">TECHNICAL ANALYSIS</div>', unsafe_allow_html=True)
    a_raw = sections.get("A", "").strip()
    a_body = re.sub(r"(?mi)^A\..*\n?", "", a_raw).strip()
    if a_items:
        for i, body in enumerate(a_items, start=1):
            if not body.strip():
                continue
            st.markdown(
                f"""
                <div class="incept-card">
                  <div style="font-weight:800; margin-bottom:6px;">{i}.</div>
                  <div>{body}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(a_body, unsafe_allow_html=False)

    st.markdown('<div class="sec-title">TRADE PLAN</div>', unsafe_allow_html=True)
    c = sections.get("C", "").strip()
    if c:
        c_body = re.sub(r"(?m)^C\..*\n?", "", c).strip()
        st.markdown(
            f"""
            <div class="incept-card">
              <div>{c_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("N/A")

    st.markdown('<div class="sec-title">RỦI RO &amp; LỢI NHUẬN</div>', unsafe_allow_html=True)
    ps = (analysis_pack or {}).get("PrimarySetup") or {}
    risk = ps.get("RiskPct", None)
    reward = ps.get("RewardPct", None)
    rr = ps.get("RR", None)
    prob = ps.get("Confidence (Tech)", ps.get("Probability", "N/A"))

    def _fmt_pct_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}%"
        except Exception:
            return "N/A"

    def _fmt_rr_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"

    st.markdown(
        f"""
        <div class="incept-metrics">
          <div class="incept-metric"><div class="k">Risk%:</div><div class="v">{_fmt_pct_local(risk)}</div></div>
          <div class="incept-metric"><div class="k">Reward%:</div><div class="v">{_fmt_pct_local(reward)}</div></div>
          <div class="incept-metric"><div class="k">RR:</div><div class="v">{_fmt_rr_local(rr)}</div></div>
          <div class="incept-metric"><div class="k">Confidence (Tech):</div><div class="v">{prob}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# Override with clean typography (later definition takes precedence)
def render_report_pretty(report_text: str, analysis_pack: dict):
    sections = _split_sections(report_text)
    a_items = _extract_a_items(sections.get("A", ""))

    st.markdown('<div class="incept-wrap">', unsafe_allow_html=True)

    ap = analysis_pack or {}
    scenario_pack = ap.get("Scenario12") or {}
    master_pack = ap.get("MasterScore") or {}
    conviction_score = ap.get("Conviction", "N/A")

    def _val_or_na(v):
        if v is None: return "N/A"
        if isinstance(v, float) and pd.isna(v): return "N/A"
        text = str(v).strip()
        return text if text else "N/A"

    st.markdown(
        f"""
        <div class="report-header">
          <h2 style="margin:0; padding:0;">{_val_or_na(ap.get("Ticker"))} - {_val_or_na(scenario_pack.get("Name"))}</h2>
          <div style="font-size:16px; font-weight:700; margin-top:4px;">
            Điểm tổng hợp: {_val_or_na(master_pack.get("Total"))} | Điểm tin cậy: {_val_or_na(conviction_score)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sec-title">TECHNICAL ANALYSIS</div>', unsafe_allow_html=True)
    a_raw = sections.get("A", "").strip()
    a_body = re.sub(r"(?mi)^A\..*\n?", "", a_raw).strip()
    if a_items:
        for i, body in enumerate(a_items, start=1):
            if not body.strip():
                continue
            st.markdown(
                f"""
                <div class="incept-card">
                  <div style="font-weight:800; margin-bottom:6px;">{i}.</div>
                  <div>{body}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(a_body, unsafe_allow_html=False)

    st.markdown('<div class="sec-title">TRADE PLAN</div>', unsafe_allow_html=True)
    c = sections.get("C", "").strip()
    if c:
        c_body = re.sub(r"(?m)^C\..*\n?", "", c).strip()
        st.markdown(
            f"""
            <div class="incept-card">
              <div>{c_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("N/A")

    st.markdown('<div class="sec-title">RỦI RO &amp; LỢI NHUẬN</div>', unsafe_allow_html=True)
    ps = (analysis_pack or {}).get("PrimarySetup") or {}
    risk = ps.get("RiskPct", None)
    reward = ps.get("RewardPct", None)
    rr = ps.get("RR", None)
    prob = ps.get("Confidence (Tech)", ps.get("Probability", "N/A"))

    def _fmt_pct_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}%"
        except Exception:
            return "N/A"

    def _fmt_rr_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"

    st.markdown(
        f"""
        <div class="incept-metrics">
          <div class="incept-metric"><div class="k">Risk%:</div><div class="v">{_fmt_pct_local(risk)}</div></div>
          <div class="incept-metric"><div class="k">Reward%:</div><div class="v">{_fmt_pct_local(reward)}</div></div>
          <div class="incept-metric"><div class="k">RR:</div><div class="v">{_fmt_rr_local(rr)}</div></div>
          <div class="incept-metric"><div class="k">Confidence (Tech):</div><div class="v">{prob}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

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
        ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB").upper()
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
                    result = analyze_ticker(ticker_input)

                    # ------------------------------
                    # MODULE EXECUTION (REGISTRY)
                    # ------------------------------
                    ap_base = result.get("AnalysisPack", {}) if isinstance(result, dict) else {}
                    df_used = result.get("_DF", None) if isinstance(result, dict) else None

                    ctx = {
                        "ticker": ticker_input,
                        "df": df_used,
                        "result": result,
                        "compute_character_pack": compute_character_pack,
                        "generate_insight_report": generate_insight_report,
                    }

                    modules_out, mod_errors = run_modules(
                        analysis_pack=ap_base if isinstance(ap_base, dict) else {},
                        enabled=["report_ad", "character"],
                        ctx=ctx,
                    )

                    if isinstance(result, dict):
                        result["Modules"] = modules_out
                        if mod_errors:
                            result["_ModuleErrors"] = mod_errors

                    report = ((modules_out or {}).get("report_ad") or {}).get("report") if isinstance(modules_out, dict) else ""
                    if not report:
                        report = generate_insight_report(result)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    analysis_pack = result.get("AnalysisPack", {}) if isinstance(result, dict) else {}
                    if 'output_mode' in locals() and output_mode == 'Character':
                        # Decision Layer Report (Anti-Anchoring Output Order)
                        render_appendix_e(result, report, analysis_pack)
                    else:
                        # Legacy report A–D
                        render_report_pretty(report, analysis_pack)
                except Exception as e:
                    st.error(f"⚠️ Lỗi xử lý: {e}")

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
