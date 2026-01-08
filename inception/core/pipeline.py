"""inception.core.pipeline

Step 12: Pipeline Orchestrator (single entry point).

This file provides a stable, app-agnostic entry to produce everything the UI needs:
- Load data (via DataHub)
- Build base AnalysisPack (analysis_builder)
- Inject PositionStatePack (Option 2) BEFORE Character computes TradePlan/Decision packs
- Normalize packs (contracts)
- Run enabled modules (registry)
- Build DashboardSummaryPack

The pipeline is designed to be safe-by-default:
- Missing optional inputs do not crash; they produce structured errors/warnings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import copy

import numpy as np

from inception.infra.datahub import DataHub, DataError

from inception.core.analysis_builder import build_base_result_v1
from inception.core.contracts import normalize_analysis_pack
from inception.core.dashboard_pack import compute_dashboard_summary_pack_v1
from inception.core.helpers import _safe_float

from inception.modules import load_default_modules
from inception.modules.base import run_modules


def _build_position_state_pack(
    *,
    mode: str,
    current_price: float,
    avg_cost: Optional[float] = None,
    position_size_pct_nav: Optional[float] = None,
    risk_budget_pct_nav: Optional[float] = None,
    holding_horizon: str = "SWING",
    timeframe: str = "D",
) -> Dict[str, Any]:
    """Create PositionStatePack.

    This is intentionally conservative and UI-friendly:
    - If avg_cost is missing while HOLDING, pnl fields become NaN.
    - Position sizing fields are optional.
    """

    mode_u = (mode or "FLAT").strip().upper()
    is_holding = mode_u == "HOLDING"

    c = _safe_float(current_price)
    ac = _safe_float(avg_cost, default=np.nan) if avg_cost is not None else np.nan

    pnl_pct = np.nan
    in_profit = False
    if is_holding and np.isfinite(c) and np.isfinite(ac) and ac != 0:
        pnl_pct = (c - ac) / ac * 100.0
        in_profit = bool(pnl_pct > 0)

    return {
        "Mode": "HOLDING" if is_holding else "FLAT",
        "IsHolding": bool(is_holding),
        "Timeframe": str(timeframe or "D").upper(),
        "HoldingHorizon": str(holding_horizon or "SWING").upper(),
        "AvgCost": ac,
        "CurrentPrice": c,
        "PnlPct": pnl_pct,
        "InProfit": bool(in_profit),
        "PositionSizePctNAV": _safe_float(position_size_pct_nav, default=np.nan)
        if position_size_pct_nav is not None
        else np.nan,
        "RiskBudgetPctNAV": _safe_float(risk_budget_pct_nav, default=np.nan)
        if risk_budget_pct_nav is not None
        else np.nan,
    }


def build_result(
    *,
    ticker: str,
    data_dir: Optional[str] = None,
    price_vol_path: str = "Price_Vol.xlsx",
    hsc_target_path: str = "Tickers Target Price.xlsx",
    enabled_modules: Optional[Iterable[str]] = None,
    # Position inputs (Option 2)
    position_mode: str = "FLAT",
    avg_cost: Optional[float] = None,
    position_size_pct_nav: Optional[float] = None,
    risk_budget_pct_nav: Optional[float] = None,
    holding_horizon: str = "SWING",
    timeframe: str = "D",
) -> Dict[str, Any]:
    """Build a complete ResultPack for one ticker.

    Returns a dict with at least:
      - AnalysisPack
      - _DF
      - Modules
      - DashboardSummaryPack

    On errors, returns {"Error": "..."}.
    """

    ticker_u = (ticker or "").strip().upper()
    if not ticker_u:
        return {"Error": "Missing ticker"}

    # 1) Data loading
    try:
        hub = DataHub.from_env(default_dir=data_dir)
        df_all = hub.load_price_vol(price_vol_path)
        try:
            hsc = hub.load_hsc_targets(hsc_target_path)
        except Exception:
            hsc = None
    except DataError as e:
        return {"Error": str(e)}

    # 2) Build base analysis (no TradePlan/Decision attached here)
    base = build_base_result_v1(
        ticker=ticker_u,
        df_all=df_all,
        hsc_targets=hsc,
    )
    if not isinstance(base, dict) or base.get("Error"):
        return base if isinstance(base, dict) else {"Error": "Invalid base result"}

    analysis_pack = base.get("AnalysisPack")
    df = base.get("_DF")

    if not isinstance(analysis_pack, dict) or df is None:
        return {"Error": "Base analysis missing AnalysisPack/_DF"}

    # 3) Normalize (contract gate)
    try:
        analysis_pack = normalize_analysis_pack(analysis_pack)
    except Exception:
        pass

    # 4) Inject PositionStatePack BEFORE Character computes TradePlan/Decision packs
    try:
        last_close = _safe_float(analysis_pack.get("Last", {}).get("Close"))
    except Exception:
        last_close = np.nan

    pos_pack = _build_position_state_pack(
        mode=position_mode,
        current_price=last_close,
        avg_cost=avg_cost,
        position_size_pct_nav=position_size_pct_nav,
        risk_budget_pct_nav=risk_budget_pct_nav,
        holding_horizon=holding_horizon,
        timeframe=timeframe,
    )
    analysis_pack["PositionStatePack"] = pos_pack

    # 5) Run modules (character first, then report)
    if enabled_modules is None:
        enabled_modules = ["character", "report_ad"]

    load_default_modules()

    # result container passed to modules
    result: Dict[str, Any] = {
        **{k: v for k, v in base.items() if k != "AnalysisPack"},
        "AnalysisPack": analysis_pack,
        "_DF": df,
    }

    ctx = {"df": df, "result": result}

    modules_out, module_errors = run_modules(
        analysis_pack=analysis_pack,
        ctx=ctx,
        enabled=list(enabled_modules),
    )

    result["Modules"] = modules_out
    if module_errors:
        result["_ModuleErrors"] = module_errors

    # 6) Dashboard summary
    try:
        character_pack = modules_out.get("character", {}) if isinstance(modules_out, dict) else {}
        dash = compute_dashboard_summary_pack_v1(analysis_pack, character_pack)
    except Exception as e:
        dash = {"Error": str(e)}

    result["DashboardSummaryPack"] = dash

    # 7) Convenience: pull report text
    try:
        result["Report"] = (modules_out.get("report_ad", {}) or {}).get("report", "")
    except Exception:
        result["Report"] = ""

    return result
