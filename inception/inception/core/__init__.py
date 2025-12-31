"""INCEPTION core (Streamlit-free).

This package is the long-term home for all pure computations.
Phase 4 introduces the first clusters: helpers, indicators, tradeplan, scoring.
"""

from .helpers import (
    DataError,
    assert_not_pandas_bool,
    _safe_str,
    _safe_text,
    _as_scalar,
    _coalesce,
    _safe_bool,
    _safe_float,
    _clip,
    safe_json_dumps,
)
from .indicators import sma, ema, rsi_wilder, macd
from .tradeplan import TradeSetup, build_trade_plan
from .scoring import (
    compute_conviction_pack,
    compute_conviction,
    classify_scenario,
    classify_scenario12,
    compute_master_score,
    build_rr_sim,
)

__all__ = [
    "DataError",
    "assert_not_pandas_bool",
    "_safe_str",
    "_safe_text",
    "_as_scalar",
    "_coalesce",
    "_safe_bool",
    "_safe_float",
    "_clip",
    "safe_json_dumps",
    "sma",
    "ema",
    "rsi_wilder",
    "macd",
    "TradeSetup",
    "build_trade_plan",
    "compute_conviction_pack",
    "compute_conviction",
    "classify_scenario",
    "classify_scenario12",
    "compute_master_score",
    "build_rr_sim",
]
