from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List, Tuple
import pandas as pd

from inception.core.helpers import json_sanitize


@dataclass
class ModuleResult:
    ok: bool
    payload: Dict[str, Any]
    error: Optional[str] = None


# Runner receives AnalysisPack plus an optional context dict.
# ctx is intentionally loose to avoid import-time coupling.
Runner = Callable[[Dict[str, Any], Dict[str, Any]], ModuleResult]

_REGISTRY: Dict[str, Runner] = {}


def register(name: str, runner: Runner) -> None:
    _REGISTRY[name] = runner


def _validate_json_safe(payload: Any, module_name: str) -> List[str]:
    """Recursively validate payload is JSON-safe (no pandas objects)."""
    errs: List[str] = []

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, (pd.Series, pd.DataFrame, pd.Index)):
            errs.append(f"{module_name}{path} is pandas object")
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else f".{k}")
            return
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{path}[{i}]")
            return

    _walk(payload, "")
    return errs


def run_modules(
    analysis_pack: Dict[str, Any],
    enabled: List[str],
    ctx: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Run enabled modules in isolation.

    Contract:
    - Modules must NOT mutate analysis_pack.
    - Module payload MUST be JSON-safe (no pandas objects).
    - Module failures are isolated to their own output slot.
    """
    ctx = ctx or {}
    outputs: Dict[str, Any] = {}
    errors: List[str] = []

    for name in enabled:
        runner = _REGISTRY.get(name)
        if not runner:
            errors.append(f"Module '{name}' not registered")
            continue
        try:
            res = runner(analysis_pack, ctx)
        except Exception as e:
            outputs[name] = {"Error": str(e)}
            errors.append(f"{name} crash: {e}")
            continue

        payload = res.payload if isinstance(res.payload, dict) else {"payload": res.payload}
        raw_out = payload if res.ok else {"Error": res.error or "Unknown error", **payload}
        outputs[name] = json_sanitize(raw_out)
        errors.extend(_validate_json_safe(outputs[name], name))

    return outputs, errors
