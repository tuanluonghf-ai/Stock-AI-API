"""Public UI render entrypoints.

This module must NOT import the Streamlit entrypoint (app_INCEPTION_*.py).
All rendering functions live in :mod:`inception.ui.renderers`.

We keep this thin wrapper for backward compatibility with any code that still
imports ``inception.ui.render``.
"""

from __future__ import annotations

from typing import Any, Dict

from . import renderers


def render_report(report_text: str, analysis_pack: Dict[str, Any]) -> None:
    """Render a legacy A–D report in the pretty UI format."""
    renderers.render_report_pretty(report_text, analysis_pack)


def render_character(character_pack: Dict[str, Any]) -> None:
    """Render the Character Card UI."""
    renderers.render_character_card(character_pack)
