from __future__ import annotations

from typing import Any, Dict

from .base import ModuleResult, register


def run(analysis_pack: Dict[str, Any], ctx: Dict[str, Any]) -> ModuleResult:
    """Build Report A–D text. Uses ctx['generate_insight_report']."""
    fn = ctx.get("generate_insight_report")
    result = ctx.get("result")

    if fn is None or not callable(fn):
        return ModuleResult(ok=False, payload={}, error="Missing ctx['generate_insight_report']")

    if result is None:
        result = {"AnalysisPack": analysis_pack}

    try:
        text = (fn(result) or "").strip()
        return ModuleResult(ok=True, payload={"report": text})
    except Exception as e:
        return ModuleResult(ok=False, payload={}, error=str(e))


register("report_ad", run)
