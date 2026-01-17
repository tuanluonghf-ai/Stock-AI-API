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
from inception.core.helpers import _safe_float, _safe_text
from inception.core.payoff_tier_pack import LookbackCfg, compute_payoff_tier_v1
from inception.core.narrative_draft_pack import build_narrative_draft_pack_v1
from inception.core.investor_mapping import build_investor_mapping_pack
import os
from inception.core.result_contract import finalize_result_pack_v1
from inception.core.stability.decision_stability import apply_decision_stability
from inception.core.stability.plan_stability import apply_plan_stability
from inception.core.stability.narrative_anchor import apply_narrative_anchor

from inception.core.stability.diagnostics import build_stability_diagnostics
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

    # Canonical snake_case (core expects these)
    p = {
        "mode": "HOLDING" if is_holding else "FLAT",
        "is_holding": bool(is_holding),
        "timeframe": str(timeframe or "D").upper(),
        "holding_horizon": str(holding_horizon or "SWING").upper(),
        "avg_cost": ac,
        "current_price": c,
        "pnl_pct": pnl_pct,
        "in_profit": bool(in_profit),
        "position_size_pct_nav": _safe_float(position_size_pct_nav, default=np.nan)
        if position_size_pct_nav is not None
        else np.nan,
        "risk_budget_pct_nav": _safe_float(risk_budget_pct_nav, default=np.nan)
        if risk_budget_pct_nav is not None
        else np.nan,
    }

    # Backward-compat: also expose TitleCase aliases (some older renderers).
    p.update(
        {
            "Mode": p["mode"],
            "IsHolding": p["is_holding"],
            "Timeframe": p["timeframe"],
            "HoldingHorizon": p["holding_horizon"],
            "AvgCost": p["avg_cost"],
            "CurrentPrice": p["current_price"],
            "PnlPct": p["pnl_pct"],
            "InProfit": p["in_profit"],
            "PositionSizePctNAV": p["position_size_pct_nav"],
            "RiskBudgetPctNAV": p["risk_budget_pct_nav"],
        }
    )
    return p


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

    # 3.1) Payoff tier (OHLC-only, deterministic)
    try:
        if isinstance(analysis_pack, dict) and "payoff" not in analysis_pack:
            ref_price = _safe_float((analysis_pack.get("Last") or {}).get("Close"), default=np.nan)
            lookback_cfg = LookbackCfg(id="PAYOFF_V1_250D", window=250)
            analysis_pack["payoff"] = compute_payoff_tier_v1(df=df, reference_price=ref_price, lookback_cfg=lookback_cfg)
    except Exception:
        pass

    # 4) Inject PositionStatePack BEFORE Character computes TradePlan/Decision packs
    try:
        last_close = _safe_float(analysis_pack.get("Last", {}).get("Close"))
    except Exception:
        last_close = np.nan

    # Robust mode inference: if the app forgets to pass mode but provides avg_cost/size, treat as HOLDING.
    pmode = (position_mode or "FLAT").strip().upper()
    try:
        psize = _safe_float(position_size_pct_nav, default=np.nan)
    except Exception:
        psize = np.nan
    try:
        pac = _safe_float(avg_cost, default=np.nan)
    except Exception:
        pac = np.nan
    coerced = False
    if pmode not in ("FLAT", "HOLDING"):
        pmode = "FLAT"
    if pmode == "FLAT":
        if (np.isfinite(psize) and float(psize) > 0) or np.isfinite(pac):
            pmode = "HOLDING"
            coerced = True
    if coerced:
        try:
            analysis_pack.setdefault("_Warnings", []).append("PositionMode auto-coerced to HOLDING because avg_cost/position_size_pct_nav is provided.")
        except Exception:
            pass
    position_mode = pmode

    pos_pack = _build_position_state_pack(
        mode=position_mode,
        current_price=last_close,
        avg_cost=avg_cost,
        position_size_pct_nav=position_size_pct_nav,
        risk_budget_pct_nav=risk_budget_pct_nav,
        holding_horizon=holding_horizon,
        timeframe=timeframe,
    )
    # Normalize to stable contract (snake_case) while keeping legacy TitleCase aliases
    try:
        from inception.core.contracts import normalize_position_state_pack
        analysis_pack["PositionStatePack"] = normalize_position_state_pack(pos_pack)
    except Exception:
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

    # 5.1) Investor Mapping Pack (UI-ready; no LLM)
    try:
        cp = modules_out.get("character", {}) if isinstance(modules_out, dict) else {}
        st = cp.get("StockTraits") if isinstance(cp.get("StockTraits"), dict) else {}
        cs = cp.get("CoreStats") if isinstance(cp.get("CoreStats"), dict) else {}
        cb = cp.get("CombatStats") if isinstance(cp.get("CombatStats"), dict) else {}
        dna = analysis_pack.get("DNAPack") if isinstance(analysis_pack.get("DNAPack"), dict) else {}
        payoff_tier = None
        try:
            payoff_tier = _safe_text((analysis_pack.get("payoff") or {}).get("payoff_tier")).strip().upper()
        except Exception:
            payoff_tier = None
        inv = build_investor_mapping_pack(st, cs, cb, dna, payoff_tier=payoff_tier)
        if isinstance(analysis_pack, dict):
            analysis_pack["InvestorMappingPack"] = inv
        result["InvestorMappingPack"] = inv
    except Exception:
        pass

    # 5.5) NarrativeDraftPack (Engine-built; no LLM). Attach to behavioural/UI layers.
    try:
        ndp = build_narrative_draft_pack_v1(analysis_pack, ctx) or {}
        # Optional (Phase 2–3): GPT rewrite-only. Disabled by default.
        # Enable via INCEPTION_ENABLE_GPT_REWRITE=1. Always guarded (never crash).
        try:
            enable = str(os.environ.get("INCEPTION_ENABLE_GPT_REWRITE", "0")).strip() in ("1", "true", "TRUE", "yes", "YES")
        except Exception:
            enable = False
        if enable:
            try:
                from inception.core.dna_rewrite_only import rewrite_dna_line_only
                ndp = rewrite_dna_line_only(draft_pack=ndp, ctx=ctx) or ndp
            except Exception:
                pass

            try:
                from inception.core.narrative_rewrite_only import rewrite_status_line_only, rewrite_plan_line_only
                ndp = rewrite_status_line_only(draft_pack=ndp, ctx=ctx) or ndp
                ndp = rewrite_plan_line_only(draft_pack=ndp, ctx=ctx) or ndp
            except Exception:
                pass

        if isinstance(analysis_pack, dict):
            analysis_pack["NarrativeDraftPack"] = ndp
        result["NarrativeDraftPack"] = ndp
    except Exception:
        pass


    # 5.6) Decision Stability (Hysteresis governor)
    # Scope-locked: does NOT modify indicators or raw DecisionPack.
    try:
        dsp = apply_decision_stability(
            ticker=ticker_u,
            analysis_pack=analysis_pack,
            base_dir=str(getattr(hub, "data_dir", None)) if getattr(hub, "data_dir", None) is not None else None,
        )
        if isinstance(analysis_pack, dict):
            analysis_pack["DecisionStabilityPack"] = dsp
        result["DecisionStabilityPack"] = dsp
    except Exception:
        pass

    # 5.7) Trade Plan Stability (Plan Persistence governor)
    # Scope-locked: does NOT modify indicators or raw TradePlanPack.
    try:
        psp = apply_plan_stability(
            ticker=ticker_u,
            analysis_pack=analysis_pack,
            base_dir=str(getattr(hub, "data_dir", None)) if getattr(hub, "data_dir", None) is not None else None,
        )
        if isinstance(analysis_pack, dict):
            analysis_pack["PlanStabilityPack"] = psp
        result["PlanStabilityPack"] = psp
    except Exception:
        pass

    # 5.8) Narrative Stability (Semantic Anchoring)
    # Scope-locked: does NOT modify indicators; may prepend anchor phrase into NarrativeDraftPack only when missing.
    try:
        nap = apply_narrative_anchor(
            ticker=ticker_u,
            analysis_pack=analysis_pack,
            base_dir=str(getattr(hub, "data_dir", None)) if getattr(hub, "data_dir", None) is not None else None,
        )
        if isinstance(analysis_pack, dict):
            analysis_pack["NarrativeAnchorPack"] = nap
        result["NarrativeAnchorPack"] = nap
    except Exception:
        pass

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

    # 8) Final contract gate (Engine → UI): ensure packs exist & are UI-safe.
    try:
        result = finalize_result_pack_v1(result)
    except Exception:
        pass

    return result
