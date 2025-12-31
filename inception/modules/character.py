from __future__ import annotations

from typing import Any, Dict

from .base import ModuleResult, register


def run(analysis_pack: Dict[str, Any], ctx: Dict[str, Any]) -> ModuleResult:
    """Run Character module using hooks passed via ctx."""
    fn = ctx.get("compute_character_pack")
    df = ctx.get("df")

    if fn is None or not callable(fn):
        return ModuleResult(ok=False, payload={}, error="Missing ctx['compute_character_pack']")
    if df is None:
        return ModuleResult(ok=False, payload={}, error="Missing ctx['df']")

    try:
        payload = fn(df, analysis_pack)
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        return ModuleResult(ok=True, payload=payload)
    except Exception as e:
        return ModuleResult(ok=False, payload={}, error=str(e))


register("character", run)
