"""Streamlit rendering helpers (UI-only).

This module intentionally avoids importing the Streamlit entrypoint module
(e.g., app_INCEPTION_*.py) to prevent circular imports and tight coupling.

If you want to reuse the existing renderers defined in the Streamlit app,
these helpers will resolve them dynamically from the currently running
__main__ module.
"""

from __future__ import annotations

from typing import Any, Dict, Callable
import sys


def _get_main_renderer(name: str) -> Callable[..., Any]:
    main_mod = sys.modules.get("__main__")
    fn = getattr(main_mod, name, None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            f"UI renderer '{name}' not found on __main__. "
            f"Ensure the Streamlit entrypoint defines '{name}' or migrate to inception/ui/renderers.py."
        )
    return fn


def render_report(report_text: str, analysis_pack: Dict[str, Any]) -> None:
    # Delegate to existing pretty renderer on the running entrypoint (if present)
    fn = _get_main_renderer("render_report_pretty")
    fn(report_text, analysis_pack)


def render_character(character_pack: Dict[str, Any]) -> None:
    fn = _get_main_renderer("render_character_card")
    fn(character_pack)
