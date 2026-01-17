from __future__ import annotations

# Auto-extracted Price Map chart renderer (Pha A)
import streamlit as st
import pandas as pd
import numpy as np

from typing import Any, Dict

from inception.ui.zone_pack import compute_zone_pack

from inception.ui.formatters import _safe_text, _safe_float

def render_price_map_chart_v1(df: Any, analysis_pack: Dict[str, Any]) -> None:
    """Render a dark professional price map chart.

    This chart is a *visual map* of facts already computed upstream.
    It intentionally avoids:
      - Buy/Sell markers
      - Signal annotations
      - Any logic that could override Decision/Narrative
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        st.info("Thiếu plotly để vẽ biểu đồ (requirements: plotly).")
        return

    ap = analysis_pack or {}
    if not isinstance(ap, dict):
        ap = {}

    # Coerce df
    if df is None:
        st.info("Chưa có dữ liệu giá (_DF).")
        return
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.info("Dữ liệu giá không hợp lệ.")
            return
    if df.empty:
        st.info("Dữ liệu giá trống.")
        return

    dfx = df.copy()
    # Normalize date/index
    if "Date" in dfx.columns:
        dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce")
        dfx = dfx.dropna(subset=["Date"]).sort_values("Date")
        x = dfx["Date"]
    else:
        # fallback: use index as datetime
        try:
            dfx = dfx.reset_index().rename(columns={"index": "Date"})
            dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce")
            dfx = dfx.dropna(subset=["Date"]).sort_values("Date")
            x = dfx["Date"]
        except Exception:
            st.info("Không tìm thấy cột Date để vẽ biểu đồ.")
            return

    # Windowing: keep chart readable
    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}
    tf = str(pos.get("timeframe") or "D").strip().upper()
    lookback = 200 if tf == "D" else 120
    if len(dfx) > lookback:
        dfx = dfx.iloc[-lookback:].copy()
        x = dfx["Date"]

    # Required OHLC
    for col in ("Open", "High", "Low", "Close"):
        if col not in dfx.columns:
            st.info("Thiếu dữ liệu OHLC để vẽ nến.")
            return

    # Volume is optional
    has_vol = "Volume" in dfx.columns

    # Colors (dark, muted)
    C_BG = "#0B0F19"
    C_GRID = "rgba(148,163,184,0.12)"
    C_TEXT = "#CBD5E1"
    C_UP = "#2A9D8F"
    C_DN = "#9B5C5C"
    C_MA20 = "#6B7FD7"
    C_MA50 = "#94A3B8"
    C_MA200 = "#E5E7EB"

    # Zone fills (opacity low)
    Z_POS = "rgba(42,157,143,0.10)"   # entry/buy zone
    Z_RISK = "rgba(155,92,92,0.10)"   # below hard stop
    Z_REC = "rgba(99,102,241,0.10)"   # reclaim zone

    # Extract packs
    tpp = ap.get("TradePlanPack") or {}
    tpp = tpp if isinstance(tpp, dict) else {}

    dp = ap.get("DecisionPack") or {}
    dp = dp if isinstance(dp, dict) else {}

    
# ZonePack (preferred source for shading + narrative alignment)
zone_pack = ap.get("_ZonePack")
if not isinstance(zone_pack, dict):
    try:
        zone_pack = compute_zone_pack(ap, dfx)
    except Exception:
        zone_pack = {}

    fib = ap.get("Fibonacci") or {}
    fib = fib if isinstance(fib, dict) else {}
    short_lv = ((fib.get("Short") or {}).get("levels") or {}) if isinstance(fib.get("Short"), dict) else {}
    long_lv = ((fib.get("Long") or {}).get("levels") or {}) if isinstance(fib.get("Long"), dict) else {}
    short_win = fib.get("ShortWindow")
    long_win = fib.get("LongWindow")

    # Trade plan zones (facts)
    plan_primary = tpp.get("plan_primary") or {}
    plan_primary = plan_primary if isinstance(plan_primary, dict) else {}
    plan_alt = tpp.get("plan_alt") or {}
    plan_alt = plan_alt if isinstance(plan_alt, dict) else {}
    plans_all = tpp.get("plans_all") or []
    plans_all = plans_all if isinstance(plans_all, list) else []

    def _sf(v: Any) -> float:
        return _safe_float(v, default=np.nan)

    def _zone_from(d: Dict[str, Any]) -> Tuple[float, float]:
        lo = _sf((d or {}).get("Low"))
        hi = _sf((d or {}).get("High"))
        if pd.isna(lo) or pd.isna(hi):
            return (np.nan, np.nan)
        lo2, hi2 = (float(min(lo, hi)), float(max(lo, hi)))
        return (lo2, hi2)

    # Primary entry zone (for FLAT / active entry)
    entry_zone = plan_primary.get("entry_zone") or {}
    entry_lo, entry_hi = _zone_from(entry_zone if isinstance(entry_zone, dict) else {})

    # Defensive reclaim zone (for HOLDING / reclaim confirmation)
    def_zone = (plan_alt.get("defensive_reclaim_zone") if isinstance(plan_alt, dict) else None) or {}
    if not isinstance(def_zone, dict):
        def_zone = {}
    reclaim_lo, reclaim_hi = _zone_from(def_zone)

    # Defensive hard stop (line)
    hard_stop = _sf(plan_alt.get("defensive_hard_stop"))

    # If missing in plan_alt, try derive from first DEFENSIVE candidate in plans_all
    if pd.isna(reclaim_lo) or pd.isna(reclaim_hi) or pd.isna(hard_stop):
        try:
            for c in plans_all:
                if not isinstance(c, dict):
                    continue
                if _safe_text(c.get("type") or "").strip().upper() != "DEFENSIVE":
                    continue
                if pd.isna(hard_stop):
                    hard_stop = _sf(c.get("defensive_hard_stop"))
                z = c.get("defensive_reclaim_zone")
                if isinstance(z, dict) and (pd.isna(reclaim_lo) or pd.isna(reclaim_hi)):
                    reclaim_lo, reclaim_hi = _zone_from(z)
                break
        except Exception:
            pass

    # Figure: 2 rows (price + volume)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.02,
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=dfx["Open"],
            high=dfx["High"],
            low=dfx["Low"],
            close=dfx["Close"],
            increasing_line_color=C_UP,
            decreasing_line_color=C_DN,
            increasing_fillcolor=C_UP,
            decreasing_fillcolor=C_DN,
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # MA lines (use upstream columns if present; else compute quickly)
    def _ma(col: str, window: int) -> pd.Series:
        if col in dfx.columns:
            return pd.to_numeric(dfx[col], errors="coerce")
        return pd.to_numeric(dfx["Close"], errors="coerce").rolling(window).mean()

    ma20 = _ma("MA20", 20)
    ma50 = _ma("MA50", 50)
    ma200 = _ma("MA200", 200)

    fig.add_trace(go.Scatter(x=x, y=ma20, mode="lines", line=dict(color=C_MA20, width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ma50, mode="lines", line=dict(color=C_MA50, width=1.2), name="MA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=ma200, mode="lines", line=dict(color=C_MA200, width=1.6), name="MA200"), row=1, col=1)

    # Fibonacci levels (no action implication)
    def _add_fib_levels(levels: Dict[str, Any], name_prefix: str, dash: str, opacity: float) -> None:
        if not isinstance(levels, dict) or not levels:
            return
        allow = {"38.2", "50", "61.8", "78.6", "127.2", "161.8"}
        for k, v in levels.items():
            kk = str(k).strip()
            if kk not in allow:
                continue
            fv = _sf(v)
            if pd.isna(fv):
                continue
            fig.add_hline(
                y=float(fv),
                line=dict(color=f"rgba(203,213,225,{opacity})", width=1, dash=dash),
                annotation_text=f"{name_prefix} {kk}",
                annotation_font=dict(size=10, color="rgba(203,213,225,0.6)"),
                annotation_position="top left",
                row=1,
                col=1,
            )

    _add_fib_levels(long_lv, f"Fibo {long_win}", dash="solid", opacity=0.22)
    _add_fib_levels(short_lv, f"Fibo {short_win}", dash="dash", opacity=0.16)

    # Zones (rectangles) — ZonePack preferred
    x0 = x.iloc[0]
    x1 = x.iloc[-1]

zp_zones = zone_pack.get("zones") if isinstance(zone_pack, dict) else None
if isinstance(zp_zones, list) and len(zp_zones) > 0:
    # Render known zones with muted opacity (dark professional theme)
    for zz in zp_zones:
        if not isinstance(zz, dict):
            continue
        nm = str(zz.get("name") or "").upper()
        lo = _sf(zz.get("low"))
        hi = _sf(zz.get("high"))
        if pd.isna(hi):
            continue
        # Fill colors (keep consistent with Spec v1)
        fill = Z_POS if nm == "POSITIVE" else (Z_REC if nm == "RECLAIM" else (Z_RISK if nm == "RISK" else "rgba(148,163,184,0.06)"))
        # RISK is -inf..hard_stop, clamp to visible min low
        if nm == "RISK":
            try:
                y_min = float(pd.to_numeric(dfx["Low"], errors="coerce").min())
            except Exception:
                y_min = float(hi) * 0.95
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=float(y_min), y1=float(hi),
                fillcolor=fill, line=dict(color="rgba(0,0,0,0)"),
                layer="below", row=1, col=1,
            )
            fig.add_hline(
                y=float(hi),
                line=dict(color="rgba(155,92,92,0.55)", width=1, dash="dot"),
                annotation_text="Hard stop",
                annotation_font=dict(size=10, color="rgba(203,213,225,0.65)"),
                annotation_position="bottom left",
                row=1, col=1,
            )
        else:
            if pd.isna(lo) or lo == hi:
                continue
            fig.add_shape(
                type="rect", xref="x", yref="y",
                x0=x0, x1=x1, y0=float(lo), y1=float(hi),
                fillcolor=fill, line=dict(color="rgba(0,0,0,0)"),
                layer="below", row=1, col=1,
            )
else:
    # Fall back to TradePlan-derived shading (legacy)

    # Entry zone (POSITIVE)
    if pd.notna(entry_lo) and pd.notna(entry_hi) and entry_lo != entry_hi:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0, x1=x1,
            y0=float(entry_lo), y1=float(entry_hi),
            fillcolor=Z_POS,
            line=dict(color="rgba(0,0,0,0)"),
            layer="below",
            row=1, col=1,
        )

    # Reclaim zone (RECLAIM)
    if pd.notna(reclaim_lo) and pd.notna(reclaim_hi) and reclaim_lo != reclaim_hi:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0, x1=x1,
            y0=float(reclaim_lo), y1=float(reclaim_hi),
            fillcolor=Z_REC,
            line=dict(color="rgba(0,0,0,0)"),
            layer="below",
            row=1, col=1,
        )

    # Risk zone (below hard stop)
    if pd.notna(hard_stop):
        try:
            y_min = float(pd.to_numeric(dfx["Low"], errors="coerce").min())
        except Exception:
            y_min = float(hard_stop) * 0.95
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=float(y_min), y1=float(hard_stop),
            fillcolor=Z_RISK,
            line=dict(color="rgba(0,0,0,0)"),
            layer="below",
            row=1, col=1,
        )
        fig.add_hline(
            y=float(hard_stop),
            line=dict(color="rgba(155,92,92,0.55)", width=1, dash="dot"),
            annotation_text="Hard stop",
            annotation_font=dict(size=10, color="rgba(203,213,225,0.65)"),
            annotation_position="bottom left",
            row=1, col=1,
        )

    # Volume
    if has_vol:
        vol = pd.to_numeric(dfx["Volume"], errors="coerce")
        fig.add_trace(
            go.Bar(
                x=x,
                y=vol,
                marker_color="rgba(148,163,184,0.55)",
                name="Volume",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Layout
    title_bits = []
    act = _safe_text((dp.get("action") or "")).strip().upper()
    if act:
        title_bits.append(f"Action: {act}")
    if short_win or long_win:
        title_bits.append("Fibo LT/ST")

    fig.update_layout(
        height=520,
        margin=dict(l=18, r=18, t=36, b=18),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font=dict(color=C_TEXT, size=12),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        xaxis2=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis2=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        title=dict(text=" | ".join(title_bits) if title_bits else "Price Map", x=0.01, y=0.98, xanchor="left"),
    )

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(side="right")

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False})
