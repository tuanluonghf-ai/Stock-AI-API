from __future__ import annotations

from typing import Any, Dict

from inception.core.report_ad_builder import generate_insight_report

from .base import ModuleResult, register


def run(analysis_pack: Dict[str, Any], ctx: Dict[str, Any]) -> ModuleResult:
    """Build Report A–D text.

    Step 7: this module is self-contained and does NOT depend on ctx hooks.
    """
    result = ctx.get("result")
    if not isinstance(result, dict):
        result = {"AnalysisPack": analysis_pack}

    try:
        text = (generate_insight_report(result) or "").strip()
        return ModuleResult(ok=True, payload={"report": text})
    except Exception as e:
        return ModuleResult(ok=False, payload={}, error=str(e))


register("report_ad", run)
