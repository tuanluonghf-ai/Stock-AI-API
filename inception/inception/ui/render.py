"""Streamlit rendering helpers (UI-only).

These functions are adapters around the existing Streamlit renderers in app.py,
so the module system stays UI-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict

import app


def render_report(report_text: str, analysis_pack: Dict[str, Any]) -> None:
    # Delegate to existing pretty renderer
    app.render_report_pretty(report_text, analysis_pack)


def render_character(character_pack: Dict[str, Any]) -> None:
    app.render_character_card(character_pack)
