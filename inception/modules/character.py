from __future__ import annotations

from typing import Any, Dict

from inception.core.character_pack import compute_character_pack_v1

from .base import ModuleResult, register


def run(analysis_pack: Dict[str, Any], ctx: Dict[str, Any]) -> ModuleResult:
    """Run Character module (Stock DNA).

    This module is intentionally self-contained: it does not rely on app-level hooks.
    It only needs:
      - ctx['df'] (preferred) or ctx['result']['_DF'] (fallback)
      - analysis_pack (dict)
    """
    df = ctx.get("df")
    if df is None and isinstance(ctx.get("result"), dict):
        df = ctx["result"].get("_DF")

    if df is None:
        return ModuleResult(ok=False, payload={}, error="Missing ctx['df'] (or ctx['result']['_DF'])")

    try:
        payload = compute_character_pack_v1(df, analysis_pack)
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        return ModuleResult(ok=True, payload=payload)
    except Exception as e:
        return ModuleResult(ok=False, payload={}, error=str(e))


register("character", run)
