from .utils import (
    _ensure_dict,
    _safe_text,
    _safe_float,
    _clip,
    _safe_bool,
    _as_scalar,
    _val_or_na,
    _pick_character_narrative,
    build_dna_line_v1,
    get_class_policy_hint_line,
    PLAYSTYLE_TAG_TRANSLATIONS,
)
from .appendix_helpers import _split_sections, _extract_a_items, _get_anchor_phrase, _get_plan_state, _get_stable_action, render_appendix_e, render_report_pretty
from .charts_optional import render_game_card, render_price_map_chart_v1
from .dashboard import render_character_card, _get_dashboard_card, render_executive_snapshot
from .dna_status import render_character_traits, render_combat_stats_panel, render_stock_dna_insight, render_current_status_insight, render_market_state, render_investor_mapping_v1
from .decision_layer import render_character_decision, render_decision_layer_switch
from .trade_plan import _trade_plan_gate, render_trade_plan_conditional

__all__ = [
    '_ensure_dict',
    '_safe_text',
    '_safe_float',
    '_clip',
    '_safe_bool',
    '_as_scalar',
    '_val_or_na',
    '_pick_character_narrative',
    'build_dna_line_v1',
    'get_class_policy_hint_line',
    'PLAYSTYLE_TAG_TRANSLATIONS',
    '_split_sections',
    '_extract_a_items',
    '_get_anchor_phrase',
    '_get_plan_state',
    '_get_stable_action',
    'render_appendix_e',
    'render_report_pretty',
    'render_game_card',
    'render_price_map_chart_v1',
    'render_character_card',
    '_get_dashboard_card',
    'render_executive_snapshot',
    'render_character_traits',
    'render_combat_stats_panel',
    'render_stock_dna_insight',
    'render_investor_mapping_v1',
    'render_current_status_insight',
    'render_market_state',
    'render_character_decision',
    'render_decision_layer_switch',
    '_trade_plan_gate',
    'render_trade_plan_conditional',
]
