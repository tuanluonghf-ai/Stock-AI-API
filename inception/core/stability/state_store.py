from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Single source of truth for stability state persistence.
# All stability modules must read/write via this store.

_DEFAULT_DIRNAME = ".inception_state"
_SCHEMA_VERSION = 1


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def is_state_disabled() -> bool:
    """Global kill-switch for reading/writing stability state."""
    return _truthy(os.environ.get("INCEPTION_DISABLE_STABILITY_STATE", ""))


def is_persist_enabled() -> bool:
    """Whether writing to disk is enabled (reading still allowed unless disabled)."""
    v = os.environ.get("INCEPTION_PERSIST_STABILITY", "")
    if v == "":
        return True
    return _truthy(v)


def state_dir() -> Path:
    """Root directory for stability state files."""
    custom = os.environ.get("INCEPTION_STATE_DIR", "").strip()
    root = Path(custom) if custom else Path.cwd() / _DEFAULT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_scope(scope: str) -> str:
    s = (scope or "").strip().lower().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
    return "".join(keep) or "state"


def _safe_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    keep = []
    for ch in t:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
    return "".join(keep) or "GLOBAL"


def state_path(*, scope: str, ticker: str) -> Path:
    """Compute file path for a given scope+ticker."""
    return state_dir() / f"{_safe_scope(scope)}__{_safe_ticker(ticker)}.json"


def load_state(*, scope: str, ticker: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load state dict. Returns default if missing or invalid. Never raises."""
    if is_state_disabled():
        return dict(default or {})
    path = state_path(scope=scope, ticker=ticker)
    if not path.exists():
        return dict(default or {})
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return dict(default or {})
    return dict(default or {})


def save_state(*, scope: str, ticker: str, state: Dict[str, Any]) -> bool:
    """Persist state atomically. Returns True on success. Never raises."""
    if is_state_disabled() or not is_persist_enabled():
        return False
    if not isinstance(state, dict):
        return False
    path = state_path(scope=scope, ticker=ticker)
    # attach schema version
    st = dict(state)
    st.setdefault("schema_version", _SCHEMA_VERSION)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False
