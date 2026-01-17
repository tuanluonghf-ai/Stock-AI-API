"""Streamlit UI renderers for INCEPTION.

All Streamlit presentation logic lives here to keep the entrypoint
app (app_INCEPTION_*.py) small and wiring-only.

Note: This module is intentionally UI-only. It may import Streamlit
and generate HTML/CSS, but it must not depend on the app entrypoint.
"""

from __future__ import annotations

from .renderers_parts import (
    PLAYSTYLE_TAG_TRANSLATIONS,
    _as_scalar,
    _clip,
    _ensure_dict,
    _extract_a_items,
    _get_anchor_phrase,
    _get_dashboard_card,
    _get_plan_state,
    _get_stable_action,
    _pick_character_narrative,
    _safe_bool,
    _safe_float,
    _safe_text,
    _split_sections,
    _trade_plan_gate,
    _val_or_na,
    build_dna_line_v1,
    get_class_policy_hint_line,
    render_appendix_e,
    render_character_card,
    render_character_decision,
    render_character_traits,
    render_combat_stats_panel,
    render_current_status_insight,
    render_decision_layer_switch,
    render_executive_snapshot,
    render_game_card,
    render_investor_mapping_v1,
    render_market_state,
    render_price_map_chart_v1,
    render_report_pretty,
    render_stock_dna_insight,
    render_trade_plan_conditional,
)

__all__ = [
    "_ensure_dict",
    "_safe_text",
    "_safe_float",
    "_clip",
    "_safe_bool",
    "_as_scalar",
    "_val_or_na",
    "_pick_character_narrative",
    "build_dna_line_v1",
    "get_class_policy_hint_line",
    "PLAYSTYLE_TAG_TRANSLATIONS",
    "_split_sections",
    "_extract_a_items",
    "_get_anchor_phrase",
    "_get_plan_state",
    "_get_stable_action",
    "render_appendix_e",
    "render_report_pretty",
    "render_game_card",
    "render_price_map_chart_v1",
    "render_character_card",
    "_get_dashboard_card",
    "render_executive_snapshot",
    "render_character_traits",
    "render_combat_stats_panel",
    "render_stock_dna_insight",
    "render_investor_mapping_v1",
    "render_current_status_insight",
    "render_market_state",
    "render_character_decision",
    "render_decision_layer_switch",
    "_trade_plan_gate",
    "render_trade_plan_conditional",
]
