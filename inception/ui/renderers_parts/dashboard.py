from __future__ import annotations

from typing import Any, Dict, List, Tuple
import html
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from inception.core.dashboard_pack import compute_dashboard_summary_pack_v1
from inception.ui.render_components import sec_title, major_sec, muted, info_box, divider
from inception.ui.ui_constants import UI_PLACEHOLDER

from .utils import (
    _ensure_dict,
    _safe_text,
    _safe_float,
    _clip,
    _safe_bool,
    _as_scalar,
    _val_or_na,
    _pick_character_narrative,
)


def render_character_card(character_pack: Dict[str, Any]) -> None:
    """
    Streamlit rendering for Character Card.
    Does not affect existing report A–D.
    """
    cp = _ensure_dict(character_pack)
    core = _ensure_dict(cp.get("CoreStats"))
    combat = _ensure_dict(cp.get("CombatStats"))
    conv = _ensure_dict(cp.get("Conviction"))
    flags = cp.get("Flags") or []
    cclass = cp.get("CharacterClass") or UI_PLACEHOLDER
    err = (cp.get("Error") or "")

    ticker = _safe_text(cp.get('_Ticker') or '').strip().upper()
    headline = f"{ticker} - {cclass}" if ticker else str(cclass)
    narrative_line = _pick_character_narrative(cp)
    dash_lines: List[str] = [narrative_line] if narrative_line else [UI_PLACEHOLDER]

    def _fmt_bline(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if ":" in s:
            k, v = s.split(":", 1)
            return f'<div class="gc-bline"><b>{html.escape(k.strip())}:</b> {html.escape(v.strip())}</div>'
        return f'<div class="gc-bline">{html.escape(s)}</div>'

    blurb_html = "".join([_fmt_bline(x) for x in dash_lines if str(x).strip()])
    # Show runtime error (if CharacterPack fallback was used)
    if err:
        st.error(f"Character module error: {err}")
        tb = cp.get("Traceback")
        if tb:
            with st.expander("Character traceback (debug)"):
                st.code(str(tb))


    def _radar_svg(stats: List[Tuple[str, float]], maxv: float = 10.0, size: int = 220) -> str:
        """Return an inline SVG radar chart (0–maxv) for the Character Card."""
        n = len(stats)
        if n < 3:
            return ""
        cx = cy = size / 2.0
        r = size * 0.34
        angles = [(-math.pi / 2.0) + i * (2.0 * math.pi / n) for i in range(n)]

        def pt(angle: float, radius: float) -> Tuple[float, float]:
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        # Normalize values
        vals: List[float] = []
        for _, v in stats:
            vv = 0.0 if pd.isna(v) else float(v)
            vv = float(_clip(vv, 0.0, maxv))
            vals.append(vv)

        # Grid polygons
        grid_levels = [2, 4, 6, 8, 10]
        grid_polys = []
        for lv in grid_levels:
            rr = r * (lv / maxv)
            pts = [pt(a, rr) for a in angles]
            grid_polys.append(" ".join([f"{x:.1f},{y:.1f}" for x, y in pts]))

        # Axis endpoints
        axis_pts = [pt(a, r) for a in angles]

        # Data polygon
        data_pts = [pt(angles[i], r * (vals[i] / maxv)) for i in range(n)]
        data_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in data_pts])

        # Labels
        label_pts = [pt(a, r + 28) for a in angles]

        parts: List[str] = []
        parts.append(f'<svg class="gc-radar-svg" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">')
        # grid
        for poly in grid_polys:
            parts.append(f'<polygon points="{poly}" fill="none" stroke="#E5E7EB" stroke-width="1" />')
        # axes
        for (x, y) in axis_pts:
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" />')
        # data
        parts.append(f'<polygon points="{data_points}" fill="rgba(15,23,42,0.12)" stroke="#0F172A" stroke-width="2" />')
        # points
        for (x, y) in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#0F172A" />')
        # labels
        for i, (lx, ly) in enumerate(label_pts):
            lab = html.escape(str(stats[i][0]))
            # anchor by horizontal position
            anchor = "middle"
            if lx < cx - 10:
                anchor = "end"
            elif lx > cx + 10:
                anchor = "start"
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" font-weight="700" fill="#334155">{lab}</text>')
        parts.append('</svg>')
        return "".join(parts)


    st.markdown(
        f"""
        <div class="gc-card">
          <div class="gc-head">
            <div class="gc-h1">{html.escape(str(headline))}</div>
            <div class="gc-blurb">{blurb_html}</div>
          </div>
        """,
        unsafe_allow_html=True
    )

    
    # show CharacterPack error if present
    if cp.get("Error"):
        st.warning(f"Character module error: {cp.get('Error')}")

    # Dashboard Class Signature (Radar) — 5 long-run DNA anchors (no 'Now/Opportunity' metrics)
    dna = (_ensure_dict(cp.get("StockTraits"))).get("DNA") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"]))
    vol_risk_g = _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"]))
    dd_risk_g = _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"]))
    liq_g = _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"]))
    beh_g = _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"]))

    # Convert risk groups into positive-direction scores for the dashboard radar (higher = better control)
    vol_stab = (10.0 - float(vol_risk_g)) if pd.notna(vol_risk_g) else np.nan
    dd_res = (10.0 - float(dd_risk_g)) if pd.notna(dd_risk_g) else np.nan

    radar_stats: List[Tuple[str, float]] = [
        ("Trend", trend_g),
        ("Vol-Stability", vol_stab),
        ("DD-Resilience", dd_res),
        ("Liquidity", liq_g),
        ("Behavior", beh_g),
    ]
    svg = _radar_svg(radar_stats, maxv=10.0, size=220)

    # Side metrics list (keep numbers traceable, do not decide here)
    _metrics_html_parts: List[str] = []
    for lab, val in radar_stats:
        vv = 0.0 if pd.isna(val) else float(val)
        vv = float(_clip(vv, 0.0, 10.0))
        _metrics_html_parts.append(
            f'<div class="gc-radar-item"><span class="gc-radar-lab">{html.escape(str(lab))}</span>'
            f'<span class="gc-radar-val">{vv:.1f}/10</span></div>'
        )
    metrics_html = "".join(_metrics_html_parts)

    st.markdown(
        f'''
        <div class="gc-sec">
          <div class="gc-sec-t">CLASS SIGNATURE</div>
          <div class="gc-radar-wrap">
            {svg}
            <div class="gc-radar-metrics">{metrics_html}</div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


    tier = conv.get("Tier", "?")
    pts = conv.get("Points", np.nan)
    guide = conv.get("SizeGuidance", "")
    st.markdown(
        f"""
        <div class="gc-sec">
          <div class="gc-sec-t">CONVICTION</div>
          <div class="gc-conv">
            <div class="gc-conv-tier">Tier: <b>{tier}</b> / 7</div>
            <div class="gc-conv-pts">Points: {pts:.1f}</div>
            <div class="gc-conv-guide">{guide}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if flags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">WEAKNESSES</div>', unsafe_allow_html=True)
        for f in flags[:8]:
            sev = int(f.get("severity", 1))
            note = f.get("note", "")
            code = f.get("code", "")
            st.markdown(
                f"""<div class="gc-flag"><span class="gc-sev">S{sev}</span><span class="gc-code">{code}</span><span class="gc-note">{note}</span></div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    tags = cp.get("ActionTags") or []
    if tags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">PLAYSTYLE TAGS</div>', unsafe_allow_html=True)
        st.markdown("<div class='gc-tags'>" + "".join([f"<span class='gc-tag'>{t}</span>" for t in tags[:8]]) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def _get_dashboard_card(result: Dict[str, Any]) -> Dict[str, Any]:
    """Get DashboardSummaryPack.v1.CurrentStatusCard (if present)."""
    try:
        dsp = _ensure_dict((_ensure_dict(result)).get("DashboardSummaryPack"))
        if _safe_text(dsp.get("schema") or "").strip() != "DashboardSummaryPack.v1":
            return {}
        return _ensure_dict(dsp.get("CurrentStatusCard"))
    except Exception:
        return {}

def render_executive_snapshot(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any], gate_status: str) -> None:
    """Executive Snapshot — dashboard-style summary card.

    Renderer-only: must not change engine scoring or trade-plan math.

    Notes:
      - Uses HTML for layout; always escape dynamic strings.
      - Detail sections (Stock DNA / Current Status / Trade Plan / Decision Layer) are rendered separately under an expander.
    """
    # Backward-compat: accept either (analysis_pack, character_pack) OR (result_pack, character_pack).
    ap_in = _ensure_dict(analysis_pack)
    cp = _ensure_dict(character_pack)
    result_pack = ap_in if (isinstance(ap_in, dict) and "DashboardSummaryPack" in ap_in and "AnalysisPack" in ap_in) else {}
    card = _get_dashboard_card(result_pack) if result_pack else {}
    ap = _ensure_dict((result_pack.get("AnalysisPack") if result_pack else ap_in))

    # --------- helpers ---------
    def _sf(x: Any) -> float:
        v = _safe_float(x, default=np.nan)
        try:
            v = float(v)
        except Exception:
            return np.nan
        return v if (not pd.isna(v) and math.isfinite(v)) else np.nan

    def _fmt_num(x: Any, nd: int = 1) -> str:
        v = _sf(x)
        return UI_PLACEHOLDER if pd.isna(v) else f"{v:.{nd}f}"

    def _fmt_px(x: Any) -> str:
        v = _sf(x)
        return UI_PLACEHOLDER if pd.isna(v) else f"{v:.2f}"

    def _fmt_pct(x: Any) -> str:
        v = _sf(x)
        return "" if pd.isna(v) else f"{v:+.2f}%"

    def _bar_pct_10(x: Any) -> float:
        v = _sf(x)
        if pd.isna(v):
            return 0.0
        return float(max(0.0, min(100.0, (v / 10.0) * 100.0)))

    def _dot(val: Any, good: float, warn: float) -> str:
        v = _sf(val)
        if pd.isna(v):
            return "y"
        if v >= good:
            return "g"
        if v >= warn:
            return "y"
        return "r"

    def _tier_label(tier: Any) -> str:
        try:
            t = int(float(tier))
        except Exception:
            return "TIER ?"
        label_map = {
            7: "GOD-TIER",
            6: "VERY STRONG BUY",
            5: "STRONG BUY",
            4: "BUY",
            3: "WATCH",
            2: "CAUTIOUS",
            1: "NO EDGE",
        }
        return f"TIER {t}: {label_map.get(t, '?')}"

    def _kelly_label(tier: Any) -> str:
        try:
            t = int(float(tier))
        except Exception:
            return "KELLY BET: ?"
        if t >= 5:
            return "KELLY BET: FULL SIZE"
        if t == 4:
            return "KELLY BET: FULL SIZE"
        if t == 3:
            return "KELLY BET: HALF SIZE"
        if t == 2:
            return "KELLY BET: SMALL"
        return "KELLY BET: NO TRADE"

    # --------- data extraction ---------
    ticker = _safe_text(card.get("ticker") or ap.get("Ticker") or cp.get("_Ticker") or "").strip().upper()
    close_px = card.get("close_px") if card else (ap.get("Last") or {}).get("Close")
    chg_pct = card.get("stock_chg_pct") if card else (ap.get("Market") or {}).get("StockChangePct")
    scenario_name = _safe_text(card.get("scenario_name") or (ap.get("Scenario12") or {}).get("Name") or "?").strip()
    master_total = card.get("master_total") if card else (ap.get("MasterScore") or {}).get("Total", np.nan)
    conviction = card.get("conviction") if card else ap.get("Conviction", np.nan)

    class_name = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "?").strip()

    conv = _ensure_dict(cp.get("Conviction"))
    tier = conv.get("Tier", None)

    core = _ensure_dict(cp.get("CoreStats"))
    combat = _ensure_dict(cp.get("CombatStats"))

    # Primary setup (already computed by Python)
    primary = ap.get("PrimarySetup") or {}
    primary = primary if isinstance(primary, dict) else {}
    setup_name = _safe_text((card.get("setup_name") if card else "") or primary.get("Name") or "?").strip()

    # Prefer TradePlanPack (single source of truth for plan/state/levels). Fallback to legacy TradePlans.
    tpp = ap.get("TradePlanPack") or {}
    tpp = tpp if isinstance(tpp, dict) else {}
    pp = {}
    if _safe_text(tpp.get("schema") or "").strip() == "TradePlanPack.v1":
        pp = tpp.get("plan_primary") or {}
        pp = pp if isinstance(pp, dict) else {}

    # Levels for dashboard (keep it compact): Entry from PrimarySetup; Stop/Target/RR from TradePlanPack if available
    entry = primary.get("Entry")
    stop = (pp.get("stop") if isinstance(pp, dict) and pp else primary.get("Stop"))
    tp = (pp.get("tp1") if isinstance(pp, dict) and pp else (primary.get("TP1") if primary.get("TP1") is not None else primary.get("TP")))
    rr = (pp.get("rr_actual") if isinstance(pp, dict) and pp else primary.get("RR"))

    plan_state_es = _safe_text(pp.get("state") or "").strip().upper() if isinstance(pp, dict) and pp else ""
    plan_type_es = _safe_text(pp.get("type") or "").strip().upper() if isinstance(pp, dict) and pp else ""

    # Ensure dashboard setup label reflects the actual TradePlan primary type (avoid mismatch vs legacy PrimarySetup)
    if plan_type_es and plan_type_es != "?":
        setup_name = plan_type_es


    # Red flags (from CharacterPack weaknesses)
    flags = list(cp.get("Flags") or [])
    red_notes = []
    for f in flags:
        try:
            sev = int(f.get("severity", 0))
        except Exception:
            sev = 0
        if sev >= 2:
            note = _safe_text(f.get("note") or f.get("code") or "").strip()
            if note:
                red_notes.append(note)
        if len(red_notes) >= 2:
            break
    if not red_notes:
        red_notes = ["None"]

    # Triggers (prefer DashboardSummaryPack statuses to avoid UI drift)
    triggers = _ensure_dict(card.get("triggers")) if card else {}
    st_break = _safe_text(triggers.get("breakout") or "").strip().upper()
    st_vol = _safe_text(triggers.get("volume") or "").strip().upper()
    st_rr = _safe_text(triggers.get("rr") or "").strip().upper()

    # Raw values (fallback only; do NOT re-derive statuses here)
    struct_snap = _ensure_dict(card.get("structure_snapshot")) if card else {}
    vol_ratio = struct_snap.get("vol_ratio")
    if vol_ratio is None:
        vol_ratio = (((ap.get("ProTech") or {}).get("Volume") or {}).get("Ratio"))
    rr_val = rr

    def _dot_from_status(s: str) -> str:
        s = (s or "").strip().upper()
        if s == "PASS":
            return "g"
        if s == "FAIL":
            return "r"
        # WAIT / ? default to yellow
        return "y"

    dot_breakout = _dot_from_status(st_break) if st_break else _dot(combat.get("BreakoutForce"), good=6.8, warn=5.5)
    dot_volume = _dot_from_status(st_vol) if st_vol else _dot(vol_ratio, good=1.20, warn=0.95)
    dot_rr = _dot_from_status(st_rr) if st_rr else _dot(rr_val, good=1.80, warn=1.30)

    # --- DEBUG (auto-show only when Upside Room is ?) ---
    meta = cp.get("Meta") or {}
    if pd.isna(_sf((combat or {}).get("UpsideRoom", (combat or {}).get("UpsidePower")))):
        st.caption(f"[DEBUG] UpsideRoom=? | DenomUsed={meta.get('DenomUsed')} | ATR14={_fmt_num(meta.get('ATR14'),2)} | VolProxy={_fmt_num(meta.get('VolProxy'),2)}")
        st.caption(f"[DEBUG] Close={_fmt_num(meta.get('Close'),2)} | NR={_fmt_num(meta.get('NearestRes'),2)} | NS={_fmt_num(meta.get('NearestSup'),2)} | UpsideRaw={_fmt_num(meta.get('UpsideRaw'),2)} | DownsideRaw={_fmt_num(meta.get('DownsideRaw'),2)} | LvlSrc={meta.get('LevelCtxSource')}")
        st.caption(f"[DEBUG] UpsideNorm={_fmt_num(meta.get('UpsideNorm'),2)} | DownsideNorm={_fmt_num(meta.get('DownsideNorm'),2)} | RR={_fmt_num(meta.get('RR'),2)} | BreakoutForce={_fmt_num((combat or {}).get('BreakoutForce'),2)} | VolRatio={_fmt_num(vol_ratio,2)} | RR_plan={_fmt_num(rr_val,2)} | LvlKeys={meta.get('LevelCtxKeys')}")

    # --------- render ---------
    tier_badge = _tier_label(tier)
    kelly_badge = _kelly_label(tier)
    gate = (gate_status or UI_PLACEHOLDER).strip().upper()

    # Header strings
    title_left = f"{ticker}"
    if _fmt_px(close_px) != UI_PLACEHOLDER:
        title_left = f"{title_left} | {_fmt_px(close_px)}"
    chg_str = _fmt_pct(chg_pct)

    sub_1 = " | ".join([x for x in [class_name, scenario_name] if x and x != UI_PLACEHOLDER])
    sub_2 = f"Điểm tổng hợp: {_fmt_num(master_total,1)} | Điểm tin cậy: {_fmt_num(conviction,1)} | Gate: {gate}"

    # Pillar metrics
    def _metric_row(k: str, v: Any, nd: int = 1):
        return f"<div class='es-metric'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(_fmt_num(v, nd))}</div></div>"

        # Panel 1 (DNA) — compact class narrative + Class Signature radar (5 metrics)
    narrative_line = _pick_character_narrative(cp)
    dash_lines: List[str] = [narrative_line] if narrative_line else [UI_PLACEHOLDER]

    def _fmt_bline_es(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        if ":" in s:
            k, v = s.split(":", 1)
            return f'<div class="es-bline"><b>{html.escape(k.strip())}:</b> {html.escape(v.strip())}</div>'
        return f'<div class="es-bline">{html.escape(s)}</div>'

    narrative_html = "".join([_fmt_bline_es(x) for x in dash_lines if str(x).strip()])

    def _radar_svg_es(stats: List[Tuple[str, float]], maxv: float = 10.0, size: int = 180) -> str:
        """Inline SVG radar chart (0–maxv) for Executive Snapshot (dark background)."""
        n = len(stats)
        if n < 3:
            return ""
        cx = cy = size / 2.0
        r = size * 0.34
        angles = [(-math.pi / 2.0) + i * (2.0 * math.pi / n) for i in range(n)]

        def pt(angle: float, radius: float) -> Tuple[float, float]:
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        vals: List[float] = []
        for _, v in stats:
            vv = 0.0 if pd.isna(v) else float(v)
            vv = float(_clip(vv, 0.0, maxv))
            vals.append(vv)

        grid_levels = [2, 4, 6, 8, 10]
        grid_polys = []
        for lv in grid_levels:
            rr_ = r * (lv / maxv)
            pts = [pt(a, rr_) for a in angles]
            grid_polys.append(" ".join([f"{x:.1f},{y:.1f}" for x, y in pts]))

        axis_pts = [pt(a, r) for a in angles]
        data_pts = [pt(angles[i], r * (vals[i] / maxv)) for i in range(n)]
        data_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in data_pts])
        label_pts = [pt(a, r + 26) for a in angles]

        parts: List[str] = []
        parts.append(f'<svg class="es-radar-svg" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">')
        for poly in grid_polys:
            parts.append(f'<polygon points="{poly}" fill="none" stroke="rgba(255,255,255,0.16)" stroke-width="1" />')
        for (x, y) in axis_pts:
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,0.18)" stroke-width="1" />')
        parts.append(f'<polygon points="{data_points}" fill="rgba(124,58,237,0.20)" stroke="rgba(124,58,237,0.95)" stroke-width="2" />')
        for (x, y) in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="rgba(124,58,237,0.95)" />')
        for i, (lx, ly) in enumerate(label_pts):
            lab = html.escape(str(stats[i][0]))
            raw_v = stats[i][1]
            val_txt = "—" if pd.isna(raw_v) else f"{vals[i]:.1f}"
            anchor = "middle"
            if lx < cx - 10:
                anchor = "end"
            elif lx > cx + 10:
                anchor = "start"
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" font-weight="900" fill="rgba(255,255,255,0.85)">'
                f'<tspan x="{lx:.1f}" dy="0">{lab}</tspan>'
                f'<tspan x="{lx:.1f}" dy="13" font-size="11" font-weight="850" fill="rgba(255,255,255,0.70)">{val_txt}</tspan>'
                f'</text>'
            )
        parts.append('</svg>')
        return "".join(parts)

    # Class Signature (DNA long-run only): 5-group anchors (0–10). No 'Now/Opportunity' metrics here.
    dna_pack = (_ensure_dict(cp.get("StockTraits"))).get("DNA") or {}
    params = dna_pack.get("Params") or {}
    groups = dna_pack.get("Groups") or {}

    tier1 = dna_pack.get("Tier1") or {}
    style_axis_es = _safe_text(tier1.get("StyleAxis") or "").strip()
    primary_mod_es = _safe_text(tier1.get("PrimaryModifier") or "").strip()

    def _mod_label_es(pm: str) -> str:
        pm = (pm or "").strip().upper()
        if pm == "GAP":
            return "Event/Gap-Prone"
        if pm == "ILLIQ":
            return "Illiquid/Noisy"
        if pm == "HIVOL":
            return "High-Vol"
        if pm == "CHOPVOL":
            return "Choppy-Vol"
        if pm == "DEF":
            return "Defensive"
        if pm == "HBETA":
            return "High-Beta"
        return ""

    mod_lab_es = _mod_label_es(primary_mod_es)
    dna_conf_es = tier1.get("DNAConfidence")

    badge_bits_es: List[str] = []
    if style_axis_es:
        badge_bits_es.append(f"Style: {style_axis_es}")
    if mod_lab_es:
        badge_bits_es.append(f"Flag: {mod_lab_es}")
    conf_txt_es = _fmt_num(dna_conf_es, 0)
    if conf_txt_es != "?":
        badge_bits_es.append(f"DNA: {conf_txt_es}")

    dna_badges_es = " | ".join(badge_bits_es) if badge_bits_es else ""
    dna_badge_html_es = f'<div class="es-note" style="margin-top:4px;opacity:0.85;">{html.escape(dna_badges_es)}</div>' if dna_badges_es else ""


    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _sf(params.get(k))
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"]))
    vol_risk_g = _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"]))
    dd_risk_g = _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"]))
    liq_g = _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"]))
    beh_g = _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"]))

    # Convert risk groups into positive-direction scores (higher = better control)
    vol_stab = (10.0 - float(vol_risk_g)) if pd.notna(vol_risk_g) else float("nan")
    dd_res = (10.0 - float(dd_risk_g)) if pd.notna(dd_risk_g) else float("nan")

    sig_stats: List[Tuple[str, float]] = [
        ("Trend", trend_g),
        ("Vol-Stability", vol_stab),
        ("DD-Resilience", dd_res),
        ("Liquidity", liq_g),
        ("Behavior", beh_g),
    ]
    radar_svg = _radar_svg_es(sig_stats, maxv=10.0, size=220)

    panel1 = f"""
    <div class="es-panel">
      <div class="es-pt">1) STOCK DNA</div>
      <div class="es-note" style="font-weight:900;">{html.escape(class_name)}</div>{dna_badge_html_es}
      <div class="es-bline-wrap">{narrative_html}</div>
      <div class="es-sig-wrap">
        <div class="es-sig-radar">{radar_svg}</div>
      </div>
    </div>
    """

        # ---------------------------
    # Panel 2 — CURRENT STATUS (Dashboard)
    #   DashboardSummaryPack.v1 = single source of truth for what Dashboard displays.
    # ---------------------------
    dsp = compute_dashboard_summary_pack_v1(ap, cp, gate_status=gate_status)
    try:
        ap["DashboardSummaryPack"] = dsp
    except Exception:
        pass

    card2 = dsp.get("CurrentStatusCard") if isinstance(dsp, dict) else {}
    card2 = card2 if isinstance(card2, dict) else {}

    state_capsule_line = _safe_text(card2.get("state_capsule_line") or "?").strip()

    master_total_d = card2.get("master_total", master_total)
    conviction_d = card2.get("conviction", conviction)

    ms_pct = _bar_pct_10(master_total_d)
    cs_pct = _bar_pct_10(conviction_d)

    insight_line_es = _safe_text(card2.get("insight_line") or "").strip()
    policy_hint_es = _safe_text(card2.get("policy_hint_line") or "").strip()

    triggers2 = card2.get("triggers") if isinstance(card2.get("triggers"), dict) else {}
    st_break = _safe_text((triggers2 or {}).get("breakout") or "?").strip().upper()
    st_vol = _safe_text((triggers2 or {}).get("volume") or "?").strip().upper()
    st_rr = _safe_text((triggers2 or {}).get("rr") or "?").strip().upper()
    st_struct = _safe_text((triggers2 or {}).get("structure") or "?").strip().upper()

    def _norm_st(s: str) -> str:
        s = (s or "").strip().upper()
        if s in ("PASS", "WAIT", "FAIL"):
            return s
        return UI_PLACEHOLDER

    st_break = _norm_st(st_break)
    st_vol = _norm_st(st_vol)
    st_rr = _norm_st(st_rr)
    st_struct = _norm_st(st_struct)

    def _dot_from_status(s: str) -> str:
        s = (s or "").upper()
        if s == "PASS":
            return "g"
        if s == "WAIT":
            return "y"
        if s == "FAIL":
            return "r"
        return "y"

    dot_b2 = _dot_from_status(st_break)
    dot_v2 = _dot_from_status(st_vol)
    dot_r2 = _dot_from_status(st_rr)
    dot_s2 = _dot_from_status(st_struct)

    gate_line = _safe_text(card2.get("gate_line") or "").strip()
    next_step = _safe_text(card2.get("next_step") or "Theo dõi và chờ thêm dữ liệu.").strip()

    risk_lines = card2.get("risk_flags") if isinstance(card2.get("risk_flags"), list) else []

    # DataQualityPack (Step 9)
    dq = card2.get("data_quality") if isinstance(card2.get("data_quality"), dict) else {}
    dq = dq if isinstance(dq, dict) else {}
    try:
        dq_err = int(dq.get("error_count", 0) or 0)
    except Exception:
        dq_err = 0
    try:
        dq_warn = int(dq.get("warn_count", 0) or 0)
    except Exception:
        dq_warn = 0
    dq_dot = "g" if (dq_err == 0 and dq_warn == 0) else ("r" if dq_err > 0 else "y")
    if dq_err > 0:
        dq_label = f"ERROR ({dq_err})"
    elif dq_warn > 0:
        dq_label = f"WARN ({dq_warn})"
    else:
        dq_label = "OK"
    dq_issues = dq.get("issues") if isinstance(dq.get("issues"), list) else []
    risk_lines = [str(x) for x in risk_lines if x is not None]

    def _vn_clean_flag_line(t: str) -> str:
        tt = _safe_text(t).strip()
        tt = re.sub(r"^\[[^\]]+\]\s*", "", tt)  # drop [Code]
        tt = tt.replace("Weekly pivot (prior LOW — now resistance)", "Pivot tuần (đáy trước — nay thành kháng cự)")
        tt = tt.replace("Weekly pivot (prior LOW - now resistance)", "Pivot tuần (đáy trước — nay thành kháng cự)")
        return tt

    risk_lines = [_vn_clean_flag_line(x) for x in risk_lines if _safe_text(x).strip()]
    if not risk_lines:
        risk_lines = ["Không có"]

    # Decision & Position (dashboard capsule) — single source of truth
    decision_block_html = ""
    try:
        dsum = card2.get("decision") if isinstance(card2.get("decision"), dict) else {}
        mode_d = _safe_text(dsum.get("mode") or "?").strip().upper()
        action_d = _safe_text(dsum.get("action") or "?").strip().upper()
        urg_d = _safe_text(dsum.get("urgency") or "").strip().upper()
        cons0 = _safe_text(dsum.get("constraint0") or "").strip()
        cons0_vn = cons0
        try:
            _map = {
                "No add while StructureGate is WAIT/FAIL; prioritize reclaim/confirm.": "Không gia tăng khi StructureGate WAIT/FAIL; ưu tiên lấy lại mốc/xác nhận.",
                "Volume not confirmed; avoid aggressive buys/adds.": "Chưa có xác nhận dòng tiền; tránh mua/gia tăng mạnh.",
            }
            for k, v in _map.items():
                if cons0_vn:
                    cons0_vn = cons0_vn.replace(k, v)
        except Exception:
            cons0_vn = cons0


        pos_sz = _safe_float(dsum.get("position_size_pct_nav"), default=np.nan)
        pnl_f = _safe_float(dsum.get("pnl_pct"), default=np.nan)
        trim_pct = _safe_float(dsum.get("trim_pct_of_position"), default=np.nan)
        stop_sug = _safe_float(dsum.get("stop_suggest"), default=np.nan)

        if action_d and action_d != "?":
            lines: List[str] = []
            lines.append(
                f"<div class='es-note'><b>Mode:</b> {html.escape(mode_d)} | <b>Action:</b> {html.escape(action_d)}{(' ('+html.escape(urg_d)+')') if urg_d else ''}</div>"
            )

            if mode_d == "HOLDING":
                parts: List[str] = []
                if pd.notna(pos_sz):
                    parts.append(f"Position: {pos_sz:.0f}% NAV")
                if pd.notna(pnl_f):
                    parts.append(f"PnL: {pnl_f:+.1f}%")
                if parts:
                    lines.append(f"<div class='es-note'>{html.escape(' | '.join(parts))}</div>")

                if action_d == "TRIM" and pd.notna(trim_pct):
                    lines.append(f"<div class='es-note'>Trim guide: ~{int(round(float(trim_pct)*100))}% vị thế</div>")

                if pd.notna(stop_sug):
                    lines.append(f"<div class='es-note'>Protect stop: {html.escape(_fmt_px(stop_sug))}</div>")
            else:
                if pd.notna(stop_sug) and action_d == "BUY":
                    lines.append(f"<div class='es-note'>Protect stop: {html.escape(_fmt_px(stop_sug))}</div>")

            if cons0_vn:
                lines.append(f"<div class='es-note' style='opacity:0.88;'>Constraint: {html.escape(cons0_vn)}</div>")

            decision_block_html = (
                "<div class='es-note' style='margin-top:10px;font-weight:950;'>Decision &amp; Position</div>"
                + "".join(lines)
            )
    except Exception:
        decision_block_html = ""

    panel2 = f"""
    <div class=\"es-panel\">
      <div class=\"es-pt\">2) CURRENT STATUS</div>

      <div class=\"es-note\" style=\"font-weight:950;\">{html.escape(state_capsule_line)}</div>

      <div class=\"es-note\" style=\"margin-top:4px;\"><span class=\"es-dot {dq_dot}\"></span>Data Quality: <b>{html.escape(dq_label)}</b></div>

      <div class=\"es-metric\"><div class=\"k\">Điểm tổng hợp</div><div class=\"v\">{html.escape(_fmt_num(master_total_d,1))}</div></div>
      <div class=\"es-mini\"><div style=\"width:{ms_pct:.0f}%\"></div></div>

      <div class=\"es-metric\" style=\"margin-top:6px;\"><div class=\"k\">Điểm tin cậy</div><div class=\"v\">{html.escape(_fmt_num(conviction_d,1))}</div></div>
      <div class=\"es-mini\"><div style=\"width:{cs_pct:.0f}%\"></div></div>

      {f'<div class="es-note" style="margin-top:8px;">{html.escape(insight_line_es)}</div>' if insight_line_es else ''}

      {f'<div class="es-note" style="margin-top:6px;"><b>Policy:</b> {html.escape(policy_hint_es)}</div>' if policy_hint_es else ''}

      <div class=\"es-note\" style=\"margin-top:10px;font-weight:950;\">Trigger Status (Plan-Gated)</div>
      <div class=\"es-note\"><span class=\"es-dot {dot_b2}\"></span>Breakout: <b>{html.escape(st_break)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_v2}\"></span>Volume: <b>{html.escape(st_vol)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_r2}\"></span>R:R: <b>{html.escape(st_rr)}</b></div>
      <div class=\"es-note\"><span class=\"es-dot {dot_s2}\"></span>Structure: <b>{html.escape(st_struct)}</b></div>

      <div class=\"es-note\" style=\"margin-top:8px;\"><b>{html.escape(gate_line)}</b></div>
      <div class=\"es-note\" style=\"margin-top:6px;\">Next step: {html.escape(next_step)}</div>

      <div class=\"es-note\" style=\"margin-top:10px;font-weight:950;\">Risk Flags</div>
      <ul class=\"es-bul\">{''.join([f'<li>{html.escape(x)}</li>' for x in risk_lines])}</ul>
    </div>
    """

    def _delta_pct(entry_x: Any, level_x: Any) -> Any:
        e = _safe_float(entry_x)
        l = _safe_float(level_x)
        if pd.notna(e) and pd.notna(l) and e != 0:
            return (l - e) / e * 100.0
        return np.nan

    stop_delta = _delta_pct(entry, stop)
    tp_delta = _delta_pct(entry, tp)
    stop_str = _fmt_px(stop) if pd.isna(stop_delta) else f"{_fmt_px(stop)} ({_fmt_pct(stop_delta)})"
    tp_str = _fmt_px(tp) if pd.isna(tp_delta) else f"{_fmt_px(tp)} ({_fmt_pct(tp_delta)})"

    # --- Dashboard panel #3: Decision first, then Trade Plan (mode-aware), then Scenario ---
    blockers: List[str] = []
    if st_break in ("FAIL", "WAIT"):
        blockers.append(f"Breakout {st_break}")
    if st_vol in ("FAIL", "WAIT"):
        blockers.append(f"Volume {st_vol}")
    if st_rr in ("FAIL", "WAIT"):
        blockers.append(f"R:R {st_rr}")
    if st_struct in ("FAIL", "WAIT"):
        blockers.append(f"Structure {st_struct}")
    blockers_line = ", ".join(blockers) if blockers else "Không có"

    mode_for_plan = "FLAT"
    action_for_plan = "?"
    try:
        dsum2 = card2.get("decision") if isinstance(card2.get("decision"), dict) else {}
        mode_for_plan = _safe_text(dsum2.get("mode") or "FLAT").strip().upper()
        action_for_plan = _safe_text(dsum2.get("action") or "?").strip().upper()
    except Exception:
        pass

    # Build a compact, non-conflicting plan block (blueprint, not commands)
    # Plan intent line depends on the actual primary plan type (avoid showing "Breakout" while primary is Pullback/Reclaim)
    _pt = _safe_text(plan_type_es or "").strip().upper()
    if _pt == "PULLBACK":
        _plan_intent_flat = "Mua mới theo PULLBACK: ưu tiên hồi về hỗ trợ; chỉ mua khi Structure PASS và Volume không FAIL; R:R đủ."
        _plan_intent_hold = "Gia tăng theo PULLBACK: ưu tiên hồi về hỗ trợ; chỉ gia tăng khi Structure PASS và Volume không FAIL; R:R đủ."
    elif _pt == "RECLAIM":
        _plan_intent_flat = "Mua mới theo RECLAIM: chỉ mua khi reclaim mốc cấu trúc và giữ được; Volume ≥ WAIT; R:R đủ."
        _plan_intent_hold = "Gia tăng theo RECLAIM: chỉ gia tăng khi reclaim mốc cấu trúc và giữ được; Volume ≥ WAIT; R:R đủ."
    else:
        _plan_intent_flat = "Mua mới chỉ khi đủ điều kiện: Breakout PASS + Volume PASS + Structure PASS + R:R PASS."
        _plan_intent_hold = "Gia tăng chỉ khi đủ điều kiện: Breakout PASS + Volume PASS + Structure PASS + R:R PASS."

    plan_lines: List[str] = []
    if mode_for_plan == "HOLDING":
        plan_lines.append("<div class='es-note' style='margin-top:10px;font-weight:950;'>TRADE PLAN (Đang nắm giữ)</div>")
        parts: List[str] = []
        if pd.notna(pnl_f):
            parts.append(f"PnL: {pnl_f:+.1f}%")
        if pd.notna(stop_sug):
            parts.append(f"Stop bảo vệ: {_fmt_px(stop_sug)}")
        if action_for_plan == "TRIM" and pd.notna(trim_pct):
            parts.append(f"Gợi ý chốt: ~{int(round(float(trim_pct)*100))}% vị thế")
        if parts:
            plan_lines.append(f"<div class='es-note'>{html.escape(' | '.join(parts))}</div>")
        plan_lines.append(f"<div class='es-note' style='opacity:0.9;'>{html.escape(_plan_intent_hold)}</div>")
    else:
        plan_lines.append("<div class='es-note' style='margin-top:10px;font-weight:950;'>TRADE PLAN (Mua mới)</div>")
        plan_lines.append(f"<div class='es-note' style='opacity:0.9;'>{html.escape(_plan_intent_flat)}</div>")
        if pd.notna(entry) and pd.notna(stop) and pd.notna(tp):
            plan_lines.append(f"<div class='es-note'>Mốc tham chiếu: Entry/Stop/Target: {_fmt_px(entry)} / {_fmt_px(stop)} / {_fmt_px(tp)}</div>")

    plan_block_html = "".join(plan_lines)

    # If HOLDING is materially underwater, surface the DEFENSIVE hard stop + reclaim trigger in Scenario card.
    # This prevents the UX confusion where 'Setup: DEFENSIVE' is shown but Entry/Stop lines still reflect BUY plans.
    def_hs = np.nan
    def_rc = np.nan
    def_rz_lo = np.nan
    def_rz_hi = np.nan
    try:
        tpp2 = ap.get("TradePlanPack") or cp.get("TradePlanPack") or {}
        if isinstance(tpp2, dict) and _safe_text(tpp2.get("schema") or "").strip() == "TradePlanPack.v1":
            pool = []
            for k in ("plan_primary", "plan_alt"):
                v = tpp2.get(k)
                if isinstance(v, dict):
                    pool.append(v)
            for v in (tpp2.get("plans_all") or []):
                if isinstance(v, dict):
                    pool.append(v)
            for p in pool:
                if _safe_text(p.get("type") or "").strip().upper() == "DEFENSIVE":
                    try:
                        def_hs = float(p.get("defensive_hard_stop"))
                    except Exception:
                        def_hs = np.nan
                    try:
                        def_rc = float(p.get("defensive_reclaim"))
                    except Exception:
                        def_rc = np.nan
                    try:
                        _rz = p.get("defensive_reclaim_zone")
                        if isinstance(_rz, dict):
                            def_rz_lo = float(_rz.get("Low")) if _rz.get("Low") is not None else np.nan
                            def_rz_hi = float(_rz.get("High")) if _rz.get("High") is not None else np.nan
                    except Exception:
                        pass
                    break
    except Exception:
        pass

    show_def_overlay = False
    try:
        show_def_overlay = (mode_for_plan == "HOLDING") and (pd.notna(pnl_f)) and (float(pnl_f) <= -15.0)
    except Exception:
        show_def_overlay = False

    if show_def_overlay and (pd.isna(def_hs) and pd.notna(stop_sug)):
        def_hs = stop_sug

    if show_def_overlay:
        setup_line = "DEFENSIVE"
        hs_line = _fmt_px(def_hs)
        # Prefer reclaim zone if available; fallback to single level
        if pd.notna(def_rz_lo) and pd.notna(def_rz_hi):
            rc_line = f"{def_rz_lo:.1f}–{def_rz_hi:.1f}"
        else:
            rc_line = _fmt_px(def_rc)
        extra_items = []
        if hs_line != UI_PLACEHOLDER:
            extra_items.append(f"<li>Protect stop: {html.escape(hs_line)}</li>")
        if rc_line != UI_PLACEHOLDER:
            extra_items.append(f"<li>Reclaim trigger: {html.escape(rc_line)} (giữ lại & đồng pha volume/structure)</li>")
        if action_for_plan == "TRIM" and pd.notna(trim_pct):
            extra_items.append(f"<li>Trim guide: ~{int(round(float(trim_pct)*100))}% vị thế</li>")

        scenario_ul = """\
      <ul class=\"es-bul\">\
        <li>Setup: {setup}</li>\
        {extra}\
      </ul>\
""".format(setup=html.escape(setup_line), extra="\n        ".join(extra_items) if extra_items else "")
    else:
        scenario_ul = f"""\
      <ul class=\"es-bul\">\
        <li>Setup: {html.escape(setup_name)}</li>\
        <li>Entry/Stop: {html.escape(_fmt_px(entry))} / {html.escape(stop_str)}</li>\
        <li>Target: {html.escape(tp_str)} (RR {html.escape(_fmt_num(rr,1))})</li>\
      </ul>\
"""

    panel3 = f"""
    <div class="es-panel">
      <div class="es-pt">3) SCENARIO</div>

      {decision_block_html}

      {plan_block_html}

      <div class="es-note" style="margin-top:10px;"><b>Kịch bản chính:</b> {html.escape(scenario_name)}</div>
      {scenario_ul}
      <div class="es-note" style="margin-top:6px;opacity:0.9;"><b>Key blockers:</b> {html.escape(blockers_line)}</div>
    </div>
    """

    card_html = f"""
    <div class="es-card">
      <div class="es-head">
        <div class="es-left">
          <div class="es-tline">
            <div class="es-ticker">{html.escape(title_left)}</div>
            {f'<div class="es-chg">{html.escape(chg_str)}</div>' if chg_str else ''}
          </div>
          <div class="es-sub">{html.escape(sub_1) if sub_1 else ''}</div>
          <div class="es-meta">{html.escape(sub_2)}</div>
        </div>
        <div class="es-right">
          <div class="es-badge">{html.escape(tier_badge)}</div>
          <div class="es-kelly">{html.escape(kelly_badge)}</div>
        </div>
      </div>
      <div class="es-body">
        {panel1}
        {panel2}
        {panel3}
      </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    # ============================================================
    # Communication Layer — Baseline Spec (Co-pilot paragraph)
    # Objective: one paragraph under the dashboard; no raw system tags.
    # ============================================================
    try:
        comm_ctx = {
            "ticker": ticker,
            "gate_status": gate_status,
            "master_score": master_total_d,
            "conviction_score": conviction_d,
            "class_name": class_name,
            "dash_lines": dash_lines,
            "risk_flags": risk_lines,
            "mode": mode_d,
            "action": action_d,
            "urgency": urg_d,
            "pnl_pct": pnl_f,
            "is_holding": (mode_d == "HOLDING"),
            # Prefer defensive overlay (underwater) levels when available; otherwise use stop_suggest
            "protect_stop": (def_hs if show_def_overlay else stop_sug),
            "stop_suggest": stop_sug,
            "reclaim_low": def_rz_lo,
            "reclaim_high": def_rz_hi,
            "reclaim": def_rc,
            "blockers": {"breakout": st_break, "volume": st_vol, "rr": st_rr, "structure": st_struct},
            "next_step": next_step,
        }
    except Exception:
        # Never block the dashboard if comm layer fails.
        pass

    # ============================================================
    # Optional: Diễn giải nhanh (3 câu hỏi vàng) — secondary explainer
    # ============================================================
    try:
        # 1) Stock DNA (pack-only)
        npack = ap.get("NarrativeFinalPack") or (_ensure_dict(result)).get("NarrativeFinalPack") or ap.get("NarrativeDraftPack") or (_ensure_dict(result)).get("NarrativeDraftPack") or {}
        npack = npack if isinstance(npack, dict) else {}
        dna_cfg = _ensure_dict(npack.get("dna"))
        dna_sentence = _safe_text(dna_cfg.get("line_final") or dna_cfg.get("line_draft")).strip() or UI_PLACEHOLDER

        # 2) Current Status (pack-only)
        status_cfg = _ensure_dict(npack.get("status"))
        status_sentence = _safe_text(status_cfg.get("line_final") or status_cfg.get("line_draft")).strip() or UI_PLACEHOLDER

        # 3) Action (pack-only)
        plan_cfg = _ensure_dict(npack.get("plan"))
        action_sentence = _safe_text(plan_cfg.get("line_final") or plan_cfg.get("line_draft")).strip() or UI_PLACEHOLDER

        with st.expander("Dien giai nhanh (3 cau hoi vang)", expanded=False):
            st.markdown(f"**1) Tinh cach (Stock DNA):** {dna_sentence}")
            st.markdown(f"**2) Tinh hinh (Current Status):** {status_sentence}")
            st.markdown(f"**3) Hanh dong (Action):** {action_sentence}")

    except Exception:
        # Never block the dashboard if explainer fails.
        pass


    # Optional diagnostics block (Step 9)
    try:
        if dq_err > 0 or dq_warn > 0:
            title = f"Data Quality: {dq_err} error(s), {dq_warn} warning(s)"
            with st.expander(title, expanded=False):
                for it in dq_issues[:10]:
                    if not isinstance(it, dict):
                        continue
                    sev = _safe_text(it.get('severity') or '').strip().upper()
                    pth = _safe_text(it.get('path') or '').strip()
                    msg = _safe_text(it.get('message') or '').strip()
                    typ = _safe_text(it.get('type') or '').strip()
                    vt = _safe_text(it.get('value_type') or '').strip()
                    extra_s = f" [{typ}{(':'+vt) if vt else ''}]" if typ else ''
                    st.markdown(f"- **{sev}** `{pth}`: {msg}{extra_s}")
    except Exception:
        pass
