from __future__ import annotations

# Auto-extracted Price Map chart renderer (Pha A)
import streamlit as st
import pandas as pd
import numpy as np
import math

from typing import Any, Dict, Optional, Tuple

from inception.ui.zone_pack import compute_zone_pack

from inception.ui.formatters import _safe_text, _safe_float

def render_price_map_chart_v1(df: Any, analysis_pack: Dict[str, Any] | None = None) -> None:
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


def render_battle_map_chart_v1(df: Any, battle_map_pack: Dict[str, Any], title: str = "BATTLE MAP") -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        st.info("Thiếu plotly để vẽ biểu đồ (requirements: plotly).")
        return

    if df is None:
        st.info("Chưa có dữ liệu giá (_DF).")
        return
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            st.info("Dữ liệu giá không hợp lệ.")
            return
    if df.empty or "Close" not in df.columns:
        st.info("Thiếu dữ liệu Close để vẽ biểu đồ.")
        return

    dfx = df.copy()
    if "Date" in dfx.columns:
        dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce")
        dfx = dfx.dropna(subset=["Date"]).sort_values("Date")
        x = dfx["Date"]
    else:
        try:
            dfx = dfx.reset_index().rename(columns={"index": "Date"})
            dfx["Date"] = pd.to_datetime(dfx["Date"], errors="coerce")
            dfx = dfx.dropna(subset=["Date"]).sort_values("Date")
            x = dfx["Date"]
        except Exception:
            st.info("Không tìm thấy cột Date để vẽ biểu đồ.")
            return

    close_full = pd.to_numeric(dfx["Close"], errors="coerce")
    if close_full.dropna().empty:
        st.info("Dữ liệu Close trống.")
        return

    # Windowing: 1Y default with auto-extend for low-vol series
    last_n = 252
    close_1y = close_full.tail(last_n)
    if not close_1y.dropna().empty:
        last_close = close_1y.dropna().iloc[-1]
        rng = (close_1y.max() - close_1y.min()) / last_close if last_close else 0.0
        if rng < 0.18:
            last_n = min(360, 420)

    dfx_view = dfx.tail(min(last_n, 420)).copy()
    df_view = dfx_view
    x = df_view["Date"]
    close = pd.to_numeric(df_view["Close"], errors="coerce")

    close_window = pd.to_numeric(df_view["Close"], errors="coerce").dropna()
    if close_window.empty:
        st.info("Dữ liệu Close trống.")
        return
    start_close = float(close_window.iloc[0])
    end_close = float(close_window.iloc[-1])
    is_up = end_close >= start_close

    BG_PLOT = "#0A1F3D"
    BG_PAPER = "#071A33"
    GRID_CLR = "rgba(255,255,255,0.06)"
    LINE_UP = "#79FF2A"
    FILL_UP = "rgba(121,255,42,0.14)"
    LINE_DOWN = "#FF4A3D"
    FILL_DOWN = "rgba(255,74,61,0.14)"
    line_color = LINE_UP if is_up else LINE_DOWN
    fill_color = FILL_UP if is_up else FILL_DOWN
    C_TEXT = "#CBD5E1"
    AMBER = "#FFB020"

    fig = go.Figure()
    close_plot = close.rolling(3, center=True).median()
    close_plot = close_plot.where(close_plot.notna(), close)
    x_price = x
    y_price = close_plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=close_plot,
            mode="lines",
            line=dict(color=line_color, width=1.8, shape="spline"),
            fill="tozeroy",
            fillcolor=fill_color,
            name="Close",
            showlegend=False,
        )
    )

    ref_price = float(battle_map_pack.get("reference_price") or 0.0) if isinstance(battle_map_pack, dict) else 0.0
    if not ref_price:
        try:
            ref_price = float(close.dropna().iloc[-1])
        except Exception:
            ref_price = 0.0
    atr = float(battle_map_pack.get("atr") or 0.0) if isinstance(battle_map_pack, dict) else 0.0

    if "Low" in df_view.columns and "High" in df_view.columns:
        low_series = pd.to_numeric(df_view["Low"], errors="coerce")
        high_series = pd.to_numeric(df_view["High"], errors="coerce")
        y_min_raw = float(low_series.min())
        y_max_raw = float(high_series.max())
    else:
        close_series = pd.to_numeric(df_view["Close"], errors="coerce")
        y_min_raw = float(close_series.min())
        y_max_raw = float(close_series.max())
    pad = max(atr * 0.8, ref_price * 0.04) if ref_price else atr * 0.8
    y_min = y_min_raw - pad
    y_max = y_max_raw + pad

    n = max(1, len(x))
    last_date = df_view["Date"].iloc[-1]
    left_ext = max(1, int(n * 0.10))
    right_margin = max(1, int(n * 0.10))
    future_end = last_date + pd.Timedelta(days=int(n * 0.25))
    zone_x0 = df_view["Date"].iloc[-left_ext] if len(df_view) > left_ext else df_view["Date"].iloc[0]
    zone_x1 = future_end - pd.Timedelta(days=int(n * 0.10))
    label_x = future_end - pd.Timedelta(days=int(n * 0.05))

    zones = []
    if isinstance(battle_map_pack, dict):
        zs = (battle_map_pack.get("zones_selected") or {}) if isinstance(battle_map_pack.get("zones_selected"), dict) else {}
        zones = (zs.get("resistances") or []) + (zs.get("supports") or [])

    def _resolve_pivot_pos(x_vals: Any, y_vals: Any, z: Dict[str, Any], reasons_upper: str) -> Optional[int]:
        try:
            n = len(y_vals)
        except Exception:
            return None
        if n <= 0:
            return None
        if z.get("pivot_ix") is not None:
            try:
                pos = int(z.get("pivot_ix"))
                if 0 <= pos < n:
                    return pos
            except Exception:
                pass
        if z.get("pivot_date") is not None:
            try:
                x_series = pd.to_datetime(pd.Series(x_vals), errors="coerce")
                x_dt = pd.to_datetime(z.get("pivot_date"))
                deltas = (x_series - x_dt).abs()
                pos = int(np.nanargmin(pd.to_numeric(deltas, errors="coerce").values))
                if 0 <= pos < n:
                    return pos
            except Exception:
                pass
        y_series = pd.to_numeric(pd.Series(y_vals), errors="coerce")
        if "HIGH_CLOSE" in reasons_upper:
            try:
                return int(np.nanargmax(y_series.values))
            except Exception:
                return None
        try:
            return int(np.nanargmin(y_series.values))
        except Exception:
            return None

    tier_opacity = {"A": 0.18, "B": 0.14, "C": 0.10, "D": 0.08}
    resistance_zones = []
    support_zones = []
    label_items = []
    for z in zones:
        if not isinstance(z, dict):
            continue
        lo = z.get("low")
        hi = z.get("high")
        if lo is None or hi is None:
            continue
        tier = str(z.get("tier") or "D").upper()
        zone_type = str(z.get("zone_type") or "")
        reasons = z.get("reasons") if isinstance(z.get("reasons"), list) else []
        rs = " ".join([str(x) for x in reasons]).upper()
        is_structure_extreme = zone_type == "STRUCTURE_EXTREME" and ("STRUCTURE_1Y_HIGH_CLOSE" in rs or "STRUCTURE_1Y_LOW_CLOSE" in rs)
        op = tier_opacity.get(tier, 0.05)
        line_w = {"A": 2.0, "B": 1.6, "C": 1.2, "D": 0.0}.get(tier, 0.0)
        center = z.get("center")
        try:
            center_val = float(center)
        except Exception:
            center_val = (float(lo) + float(hi)) * 0.5
        is_resistance = center_val > ref_price
        if is_resistance:
            r, g, b = 255, 0, 85
        else:
            r, g, b = 0, 240, 255
        if is_structure_extreme:
            pivot_pos = _resolve_pivot_pos(x_price, y_price, z, rs)
            if pivot_pos is not None:
                x_dot = x_price.iloc[pivot_pos] if hasattr(x_price, "iloc") else x_price[pivot_pos]
                y_dot = y_price.iloc[pivot_pos] if hasattr(y_price, "iloc") else y_price[pivot_pos]
            else:
                x_dot = zone_x0
                y_dot = float(center_val)
            y_dot = _safe_float(y_dot, default=math.nan)
            if not math.isfinite(y_dot):
                y_dot = float(center_val)
            y_level = float(y_dot)
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=x_dot,
                x1=zone_x1,
                y0=y_level,
                y1=y_level,
                line=dict(color=AMBER, width=2.2, dash="dot"),
                layer="above",
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_dot],
                    y=[float(y_dot)],
                    mode="markers",
                    marker=dict(size=10, color=AMBER, line=dict(color=AMBER, width=1)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            label_items.append(
                {
                    "y": y_level,
                    "color": AMBER,
                }
            )
            continue
        if is_resistance:
            resistance_zones.append((center_val, float(lo), float(hi)))
        else:
            support_zones.append((center_val, float(lo), float(hi)))
        line_cfg = {
            "color": f"rgba({r},{g},{b},1.0)",
            "width": line_w,
            "dash": "dash" if zone_type == "RESIST_TARGET" else "solid",
        }
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=zone_x0,
            x1=zone_x1,
            y0=float(lo),
            y1=float(hi),
            fillcolor=f"rgba({r},{g},{b},{op:.3f})",
            line=line_cfg,
            layer="below",
        )
        hover = f"{z.get('side')} | {zone_type} | Tier {tier} | " + ", ".join([str(r) for r in reasons[:4]])
        fig.add_trace(
            go.Scatter(
                x=[zone_x0, zone_x1],
                y=[float(lo), float(hi)],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        label_items.append(
            {
                "y": center_val,
                "color": f"rgba({r},{g},{b},1.0)",
            }
        )

    thickness_ref = None
    if isinstance(battle_map_pack, dict):
        try:
            thickness_ref = float(battle_map_pack.get("thickness") or 0.0)
        except Exception:
            thickness_ref = None
    if not thickness_ref:
        spans = []
        for _, lo, hi in resistance_zones + support_zones:
            if hi > lo:
                spans.append(hi - lo)
        if spans:
            thickness_ref = float(np.median(spans))
    if not thickness_ref:
        thickness_ref = 0.0

    gap_min = thickness_ref * 0.25 if thickness_ref else 0.0
    resistance_zones.sort(key=lambda z: z[0])
    support_zones.sort(key=lambda z: z[0], reverse=True)

    def _add_wash(z_list, rgb, gap_min_val):
        for idx in range(len(z_list) - 1):
            _, lo1, hi1 = z_list[idx]
            _, lo2, hi2 = z_list[idx + 1]
            if lo2 <= hi1:
                continue
            gap = lo2 - hi1
            if gap_min_val and gap <= gap_min_val:
                continue
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=zone_x0,
                x1=zone_x1,
                y0=float(hi1),
                y1=float(lo2),
                fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.025)",
                line=dict(width=0),
                layer="below",
            )

    if resistance_zones:
        _add_wash(resistance_zones, (255, 0, 85), gap_min)
    if support_zones:
        _add_wash(support_zones, (0, 240, 255), gap_min)

    label_items_sorted = sorted(label_items, key=lambda v: v["y"], reverse=True)
    prev_y = None
    min_gap = pad * 0.12 if pad else 0.0
    for item in label_items_sorted:
        y_val = float(item["y"])
        if prev_y is not None and min_gap and abs(y_val - prev_y) < min_gap:
            y_val = prev_y + min_gap
        prev_y = y_val
        fig.add_annotation(
            x=label_x,
            y=y_val,
            text=f"{y_val:.1f}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=13, color=item["color"]),
            bgcolor="rgba(0,0,0,0.25)",
        )

    fig.update_layout(
        height=420,
        margin=dict(l=18, r=18, t=30, b=18),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=C_TEXT, size=12),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
        title_text="Price Zones",
        title_x=0.0,
        title_y=0.98,
        title_xanchor="left",
        title_yanchor="top",
        showlegend=False,
    )
    fig.update_xaxes(range=[x.iloc[0], future_end], rangeslider_visible=False)
    fig.update_yaxes(
        range=[y_min, y_max],
        showgrid=True,
        gridcolor=GRID_CLR,
        zeroline=False,
        side="right",
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False})
