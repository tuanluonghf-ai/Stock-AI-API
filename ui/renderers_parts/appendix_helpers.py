from __future__ import annotations

from typing import Any, Dict, List, Tuple
import html
import math
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from inception.ui.ui_constants import UI_PLACEHOLDER
from inception.ui.renderers_parts.dashboard import _get_dashboard_card, render_executive_snapshot
from inception.ui.renderers_parts.decision_layer import render_decision_layer_switch
from inception.ui.renderers_parts.dna_status import (
    render_character_traits,
    render_current_status_insight,
    render_stock_dna_insight,
)
from inception.ui.renderers_parts.trade_plan import _trade_plan_gate, render_trade_plan_conditional
from .utils import (
    _ensure_dict,
    _safe_text,
    _safe_float,
    _clip,
    _val_or_na,
    get_class_policy_hint_line,
)


def _get_stable_action(analysis_pack: Dict[str, Any]) -> str:
    ap = _ensure_dict(analysis_pack)
    dsp = _ensure_dict(ap.get("DecisionStabilityPack"))
    stable = _safe_text(dsp.get("stable_action") or "").strip().upper()
    if not stable:
        dp = _ensure_dict(ap.get("DecisionPack"))
        stable = _safe_text(dp.get("action") or "").strip().upper()
    return stable




def _get_plan_state(analysis_pack: Dict[str, Any]) -> str:
    ap = _ensure_dict(analysis_pack)
    tpp = _ensure_dict(ap.get("TradePlanPack"))
    primary = _ensure_dict(tpp.get("plan_primary"))
    plan_state = _safe_text(primary.get("state") or primary.get("status") or "").strip().upper()
    if not plan_state:
        legacy = _ensure_dict(ap.get("PrimarySetup"))
        plan_state = _safe_text(legacy.get("Status") or "").strip().upper()
    return plan_state




def _get_anchor_phrase(analysis_pack: Dict[str, Any]) -> str:
    ap = _ensure_dict(analysis_pack)
    nap = _ensure_dict(ap.get("NarrativeAnchorPack"))
    phrase = _safe_text(nap.get("anchor_phrase") or nap.get("phrase") or "").strip()
    return phrase




def _split_sections(report_text: str) -> dict:
    parts = {"A": "", "B": "", "C": "", "D": ""}
    if not report_text:
        return parts
    text = report_text.replace("\r\n", "\n")
    pattern = re.compile(r"(?m)^(A|B|C|D)\.\s")
    matches = list(pattern.finditer(text))
    if not matches:
        parts["A"] = text
        return parts
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        parts[key] = text[start:end].strip()
    return parts

def _extract_a_items(a_text: str) -> list:
    if not a_text:
        return []
    text = a_text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^A\..*\n?", "", text).strip()
    item_pat = re.compile(r"(?ms)^\s*(\d)\.\s*(.*?)(?=^\s*\d\.|\Z)")
    found = item_pat.findall(text)
    items = [""] * 8
    for num, body in found:
        idx = int(num) - 1
        if 0 <= idx < 8:
            items[idx] = body.strip()
    non_empty = sum(1 for x in items if x.strip())
    return items if non_empty >= 4 else []

def render_appendix_e(result: Dict[str, Any], report_text: str, analysis_pack: Dict[str, Any]) -> None:
    """
    Decision Layer Report — Anti-Anchoring Output Order (layout only):
      1) Stock DNA (Traits)
      2) CURRENT STATUS (Scenario + Technical + Fundamental)
      3) Trade Plan & R:R (Conditional)
      4) Decision Layer (Conviction/Weakness/Tags)
    This renderer must not change any underlying calculations.
    """
    if os.getenv("INCEPTION_DEBUG_IMPORTS") == "1":
        assert callable(_get_dashboard_card)
        assert callable(render_executive_snapshot)
        assert callable(render_character_traits)
        assert callable(render_stock_dna_insight)
        assert callable(render_current_status_insight)
        assert callable(render_trade_plan_conditional)
        assert callable(_trade_plan_gate)
        assert callable(render_decision_layer_switch)

    r = _ensure_dict(result)
    modules = r.get("Modules") or {}
    cp = modules.get("character") or {}
    ap = _ensure_dict(r.get("AnalysisPack") or analysis_pack)
    card = _get_dashboard_card(r)

    # Phase F: resolve df_chart once (used by ZonePack), then attach packs to AnalysisPack
    try:
        df_chart = r.get("_DF")
    except Exception:
        df_chart = None

    zone_pack: Dict[str, Any] = {}
    action_bias_pack: Dict[str, Any] = {}
    zone_warn = ""
    action_bias_warn = ""

    try:
        from inception.ui.zone_pack import compute_zone_pack
        zone_pack = compute_zone_pack(ap, df_price=df_chart)
        ap["_ZonePack"] = zone_pack
    except Exception as exc:
        zone_warn = f"ZonePack unavailable: {exc.__class__.__name__}"
        zone_pack = {}
        ap["_ZonePack"] = zone_pack

    try:
        from inception.ui.action_bias_pack import compute_action_bias_pack
        action_bias_pack = compute_action_bias_pack(ap, zone_pack, character_pack=cp)
        ap["_ActionBiasPack"] = action_bias_pack
    except Exception as exc:
        action_bias_warn = f"ActionBiasPack unavailable: {exc.__class__.__name__}"
        action_bias_pack = {}
        ap["_ActionBiasPack"] = action_bias_pack

    def _format_zone_line(zp: Dict[str, Any]) -> str:
        zones = zp.get("zones") if isinstance(zp.get("zones"), list) else []

        def _zone_named(name: str) -> Dict[str, Any]:
            for z in zones:
                if _safe_text(z.get("name")).strip().upper() == name:
                    return z
            return {}

        def _fmt_range(z: Dict[str, Any]) -> str:
            if not isinstance(z, dict):
                return UI_PLACEHOLDER
            lo = z.get("low")
            hi = z.get("high")
            try:
                if lo == float("-inf"):
                    lo = None
            except Exception:
                pass
            lo_f = _safe_float(lo, default=np.nan)
            hi_f = _safe_float(hi, default=np.nan)
            if pd.isna(lo_f) and pd.isna(hi_f):
                return UI_PLACEHOLDER
            if pd.isna(lo_f):
                return f"<= {hi_f:.2f}"
            if pd.isna(hi_f):
                return f">= {lo_f:.2f}"
            return f"{lo_f:.2f}-{hi_f:.2f}"

        zone_now = _safe_text(zp.get("zone_now") or UI_PLACEHOLDER).strip() or UI_PLACEHOLDER
        reclaim = _fmt_range(_zone_named("RECLAIM"))
        risk = _fmt_range(_zone_named("RISK"))
        return f"ZONE MAP: {zone_now} | reclaim: {reclaim} | risk: {risk}"

    # Trade-plan gate (for execution / anti-FOMO posture)
    gate_status, _meta = _trade_plan_gate(ap, cp)

    # ---------- EXECUTIVE SNAPSHOT (single source of truth: DashboardSummaryPack) ----------
    # render_executive_snapshot is backward-compatible: it can accept the full result pack as its first argument.
    render_executive_snapshot(r, cp, gate_status)

    # Pre-split legacy report once for reuse
    exp_label = "BẤM ĐỂ XEM CHI TIẾT PHÂN TÍCH & BIỂU ĐỒ"
    exp_default = True if (gate_status or "").strip().upper() == "ACTIVE" else False
    with st.expander(exp_label, expanded=exp_default):
        left_col, right_col = st.columns([0.68, 0.32], gap="large")
        with left_col:
            sections = _split_sections(report_text or "")
            a_section = sections.get("A", "") or ""
            b_section = sections.get("B", "") or ""
            c_section = sections.get("C", "") or ""
            d_section = sections.get("D", "") or ""
    
            # ============================================================
            # 1) STOCK DNA (CORE STATS – TRAITS)
            # ============================================================
            st.markdown('<div class="major-sec">STOCK DNA</div>', unsafe_allow_html=True)
            anchor_phrase = _get_anchor_phrase(ap)
            if anchor_phrase:
                st.markdown(f"<div class='subtle-note'>{anchor_phrase}</div>", unsafe_allow_html=True)

            render_character_traits(cp)
            render_stock_dna_insight(cp)
    
            # ============================================================
            # 2) CURRENT STATUS
            # ============================================================
            st.markdown('<div class="major-sec">CURRENT STATUS</div>', unsafe_allow_html=True)

            # 2.1 Relative Strength vs VNINDEX (prefer DashboardSummaryPack)
            rel = card.get("relative_strength_vs_vnindex") if card else None
            if rel is None:
                rel = (ap.get("Market") or {}).get("RelativeStrengthVsVNINDEX")
            st.markdown(f"**Relative Strength vs VNINDEX:** {_val_or_na(rel)}")

            # 2.1.1 Zone Map (display-only)
            st.markdown('<div class="sec-title">ZONE MAP</div>', unsafe_allow_html=True)
            if zone_warn:
                st.warning(zone_warn)
            st.caption(_format_zone_line(zone_pack))

            # 2.1.2 Action Bias (display-only)
            st.markdown('<div class="sec-title">ACTION BIAS</div>', unsafe_allow_html=True)
            if action_bias_warn:
                st.warning(action_bias_warn)
            bias = _safe_text(action_bias_pack.get("bias") or UI_PLACEHOLDER)
            size = _safe_text(action_bias_pack.get("size_hint") or UI_PLACEHOLDER)
            one = _safe_text(action_bias_pack.get("one_liner") or "").strip()
            st.markdown(f"**Bias:** {bias}  \\n**Size hint:** {size}")
            if one:
                st.caption(one)

            # 2.1.3 Narrative (pack-only: Draft/Final)
            st.markdown('<div class="sec-title">NARRATIVE (KEYED)</div>', unsafe_allow_html=True)
            npack = ap.get("NarrativeFinalPack") or r.get("NarrativeFinalPack") or ap.get("NarrativeDraftPack") or r.get("NarrativeDraftPack") or {}
            npack = npack if isinstance(npack, dict) else {}

            def _line_from(pack: Dict[str, Any], key: str) -> str:
                sec = pack.get(key) if isinstance(pack.get(key), dict) else {}
                line = _safe_text(sec.get("line_final") or sec.get("line_draft")).strip()
                return line if line else UI_PLACEHOLDER

            dna_line = _line_from(npack, "dna")
            status_line = _line_from(npack, "status")
            plan_line = _line_from(npack, "plan")
            st.caption(f"DNA: {dna_line}")
            st.caption(f"Status: {status_line}")
            st.caption(f"Plan: {plan_line}")

            # 2.2 Scenario & Scores (prefer DashboardSummaryPack)
            scenario_pack = ap.get("Scenario12") or {}
            master_pack = ap.get("MasterScore") or {}
            conviction_score = (card.get("conviction") if card else None)
            if conviction_score is None:
                conviction_score = ap.get("Conviction")

            scenario_name = _safe_text(card.get("scenario_name") if card else "").strip() or _safe_text(scenario_pack.get("Name") or "?").strip()
            ms_total = (card.get("master_total") if card else None)
            if ms_total is None:
                ms_total = master_pack.get("Total")

            st.markdown("**State Capsule (Scenario & Scores)**")
            st.markdown(f"- Scenario: {html.escape(_safe_text(scenario_name))}")

            def _bar_row_cs(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "?"
                else:
                    v = float(v)
                    pct = _clip(v / maxv * 100.0, 0.0, 100.0)
                    v_disp = f"{v:.1f}/{maxv:.0f}"
                st.markdown(
                    f"""
                    <div class="gc-row">
                      <div class="gc-k">{html.escape(str(label))}</div>
                      <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
                      <div class="gc-v">{html.escape(str(v_disp))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            _bar_row_cs("Điểm tổng hợp", ms_total, 10.0)
            _bar_row_cs("Điểm tin cậy", conviction_score, 10.0)

            # Score interpretation (single block) — place directly under the two bars
            render_current_status_insight(ms_total, conviction_score, gate_status)

            # Class Policy Hint (display-only)
            _final_class = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "").strip()
            _policy_hint = _safe_text(card.get("policy_hint_line") if card else "").strip() or get_class_policy_hint_line(_final_class)
            if _policy_hint:
                st.markdown(f"**Policy:** {_policy_hint}")

            # 2.3 State Capsule (Facts-only, compact)
            st.markdown("**Structure Summary (MA/Fibo/RSI/MACD/Volume)**")
            ss = _ensure_dict(card.get("structure_snapshot")) if card else {}
            if ss:
                ma_reg = ss.get("ma_regime")
                rsi_zone = ss.get("rsi_state")
                rsi_dir = ss.get("rsi_direction")
                macd_rel = ss.get("macd_state")
                macd_zero = ss.get("macd_zero")
                align = ss.get("rsi_macd_alignment")
                short_band = ss.get("fib_short_band")
                long_band = ss.get("fib_long_band")
                fib_conflict = bool(ss.get("fib_conflict"))
                vol_ratio = ss.get("vol_ratio")
            else:
                protech = ap.get("ProTech") or {}
                protech = protech if isinstance(protech, dict) else {}
                ma = protech.get("MA") or {}
                ma = ma if isinstance(ma, dict) else {}
                rsi = protech.get("RSI") or {}
                rsi = rsi if isinstance(rsi, dict) else {}
                macd = protech.get("MACD") or {}
                macd = macd if isinstance(macd, dict) else {}
                vol = protech.get("Volume") or {}
                vol = vol if isinstance(vol, dict) else {}
                bias = protech.get("Bias") or {}
                bias = bias if isinstance(bias, dict) else {}

                fib_ctx = ((ap.get("Fibonacci") or {}).get("Context") or {})
                fib_ctx = fib_ctx if isinstance(fib_ctx, dict) else {}

                ma_reg = _safe_text(ma.get("Regime"))
                rsi_zone = _safe_text(rsi.get("State"))
                rsi_dir = _safe_text(rsi.get("Direction"))
                macd_rel = _safe_text(macd.get("State"))
                macd_zero = _safe_text(macd.get("ZeroLine"))
                align = _safe_text(bias.get("Alignment"))

                short_band = _safe_text(fib_ctx.get("ShortBand"))
                long_band = _safe_text(fib_ctx.get("LongBand"))
                fib_conflict = bool(fib_ctx.get("FiboConflictFlag"))

                vol_ratio = _safe_float(vol.get("Ratio"), default=np.nan)

            st.markdown(f"- MA Structure: {_val_or_na(ma_reg)}")
            st.markdown(f"- RSI: {_val_or_na(rsi_zone)} | {_val_or_na(rsi_dir)}")
            st.markdown(f"- MACD: {_val_or_na(macd_rel)} | ZeroLine: {_val_or_na(macd_zero)}")
            st.markdown(f"- RSI+MACD Alignment: {_val_or_na(align)}")
            st.markdown(f"- Fibonacci Bands (Short/Long): {_val_or_na(short_band)} / {_val_or_na(long_band)}" + (" | Conflict" if fib_conflict else ""))
            st.markdown(f"- Volume Ratio (vs 20d): {_val_or_na(vol_ratio)}")

            # 2.4 TECHNICAL SNAPSHOT (details)
            # 2.4 TECHNICAL SNAPSHOT (detail) (reuse A-section body: MA/Fibo/RSI/MACD/Volume/PA)
            st.markdown('<div class="sec-title">TECHNICAL SNAPSHOT</div>', unsafe_allow_html=True)
            a_items = _extract_a_items(a_section)
            a_raw = (a_section or "").replace("\r\n", "\n")
            a_body = re.sub(r"(?mi)^A\..*\n?", "", a_raw).strip()
            if a_items:
                for i, body in enumerate(a_items, start=1):
                    if not body.strip():
                        continue
                    st.markdown(
                        f"""
                        <div class="incept-card">
                          <div style="font-weight:800; margin-bottom:6px;">{i}.</div>
                          <div>{body}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                if a_body:
                    st.markdown(a_body, unsafe_allow_html=False)
                else:
                    st.info(UI_PLACEHOLDER)



            # 2.5 Combat Readiness (Now) — merged from legacy Combat Stats
            st.markdown("**Combat Readiness (Now)**")
            combat = _ensure_dict(cp.get("CombatStats"))
            combat = combat if isinstance(combat, dict) else {}

            def _bar_row_now(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "?"
                else:
                    v = float(v)
                    pct = _clip(v / maxv * 100.0, 0.0, 100.0)
                    v_disp = f"{v:.1f}/{maxv:.0f}"
                st.markdown(
                    f"""
                    <div class="gc-row">
                      <div class="gc-k">{html.escape(str(label))}</div>
                      <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
                      <div class="gc-v">{html.escape(str(v_disp))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            _bar_row_now("Upside Room", combat.get("UpsideRoom"), 10.0)
            _bar_row_now("Upside Quality", combat.get("UpsideQuality"), 10.0)
            _bar_row_now("Downside Safety", combat.get("DownsideRisk"), 10.0)
            _bar_row_now("R:R Efficiency", combat.get("RREfficiency"), 10.0)
            _bar_row_now("Breakout Force", combat.get("BreakoutForce"), 10.0)
            _bar_row_now("Support Resilience", combat.get("SupportResilience"), 10.0)

            # 2.6 Trigger Status (Plan-Gated)
            st.markdown("**Trigger Status (Plan-Gated)**")
            primary = ap.get("PrimarySetup") or {}
            primary = primary if isinstance(primary, dict) else {}
            setup_name = _safe_text(primary.get("Name")).strip()
            rr_val = _safe_float(primary.get("RR"), default=np.nan)

            plan_status = "?"
            plan_tags: List[str] = []
            for p in (ap.get("TradePlans") or []):
                if _safe_text(p.get("Name")).strip() == setup_name and setup_name and setup_name != "?":
                    plan_status = _safe_text(p.get("Status") or "?")
                    plan_tags = list(p.get("ReasonTags") or [])
                    rr_val = _safe_float(p.get("RR"), default=rr_val)
                    break

            def _status_from_val(v: Any, good: float, warn: float) -> Tuple[str, str]:
                x = _safe_float(v, default=np.nan)
                if pd.isna(x):
                    return ("?", "#9CA3AF")
                if x >= good:
                    return ("PASS", "#22C55E")
                if x >= warn:
                    return ("WAIT", "#F59E0B")
                return ("FAIL", "#EF4444")

            def _dot(color: str) -> str:
                return f'<span class="es-dot" style="background:{color};"></span>'

            s_break, c_break = _status_from_val(combat.get("BreakoutForce"), good=6.8, warn=5.5)
            s_vol, c_vol = _status_from_val(vol_ratio, good=1.20, warn=0.95)
            s_rr, c_rr = _status_from_val(rr_val, good=1.80, warn=1.30)

            # Structure (Ceiling) Gate
            sq = ap.get("StructureQuality", {}) if isinstance(ap, dict) else {}
            cg = ((sq or {}).get("Gates", {}) or {}).get("CeilingGate", {}) if isinstance((sq or {}).get("Gates", {}), dict) else {}
            s_struct = _safe_text(cg.get("Status") or "?").strip().upper()
            if s_struct not in ("PASS", "WAIT", "FAIL"):
                s_struct = "?"
            c_struct = "#9CA3AF"
            if s_struct == "PASS": c_struct = "#22C55E"
            elif s_struct == "WAIT": c_struct = "#F59E0B"
            elif s_struct == "FAIL": c_struct = "#EF4444"

            st.markdown(
                f"""<ul style="margin:0 0 0 16px; padding:0;">
                      <li>{_dot(c_break)} Breakout: {s_break}</li>
                      <li>{_dot(c_vol)} Volume: {s_vol}</li>
                      <li>{_dot(c_rr)} R:R: {s_rr}</li>
                      <li>{_dot(c_struct)} Structure: {s_struct}</li>
                      <li>{_dot("#60A5FA")} Gate: {html.escape(str(gate_status or "?"))} | Plan: {html.escape(str(setup_name or "?"))} ({html.escape(str(plan_status or "?"))})</li>
                    </ul>""",
                unsafe_allow_html=True
            )

            if plan_tags:
                tags_show = ", ".join([t for t in plan_tags if isinstance(t, str) and t.strip()][:6])
                if tags_show:
                    st.caption(f"Plan tags: {tags_show}")

            # 2.7 Risk Flags (from weakness flags + DNA modifiers)
            st.markdown("**Risk Flags**")
            flags = list(cp.get("Flags") or [])
            risk_lines = []
            for f in flags:
                try:
                    sev = int(f.get("severity", 1))
                except Exception:
                    sev = 1
                if sev >= 2:
                    note = _safe_text(f.get("note") or "").strip()
                    code = _safe_text(f.get("code") or "Flag").strip()
                    risk_lines.append(f"- [{code}] {note}" if note else f"- [{code}]")

            dna_t1 = (((_ensure_dict(cp.get("StockTraits"))).get("DNA") or {}).get("Tier1") or {})
            mods = dna_t1.get("Modifiers") if isinstance(dna_t1, dict) else []
            if isinstance(mods, list) and mods:
                mods_txt = ", ".join([str(x) for x in mods[:6]])
                risk_lines.append(f"- [DNA Modifiers] {mods_txt}")

            if risk_lines:
                st.markdown("\n".join(risk_lines))
            else:
                st.markdown("- None")

            # ============================================================
            # 3) TRADE PLAN & R:R (CONDITIONAL)
            # ============================================================
            st.markdown('<div class="major-sec">TRADE PLAN &amp; R:R</div>', unsafe_allow_html=True)
            # Pass legacy C-section body to the trade-plan renderer so explanation
            # lives next to the numeric setup cards.
            c_body_clean = ""
            if c_section:
                c_raw = c_section.replace("\r\n", "\n")
                c_body_clean = re.sub(r"(?m)^C\..*\n?", "", c_raw).strip()
            render_trade_plan_conditional(analysis_pack, cp, gate_status, c_body_clean)
            # ============================================================
            # 4) DECISION LAYER (CONVICTION, WEAKNESSES, PLAYSTYLE TAGS)
            # ============================================================
            # Central switch — layout only (no scoring / rule changes)
            primary_setup = (ap.get("PrimarySetup") or {}) if isinstance(ap, dict) else {}
            primary_name = _val_or_na(primary_setup.get("Name"))
    
            if gate_status == "LOCK":
                exec_mode_text = "WATCH ONLY – chưa kích hoạt lệnh mới (ưu tiên quan sát / bảo toàn vốn)."
            elif gate_status == "ACTIVE":
                exec_mode_text = "ACTIVE – được phép triển khai kế hoạch giao dịch theo điều kiện đã nêu."
            else:
                exec_mode_text = "WATCH ONLY – setup mang tính tham khảo, chờ thêm tín hiệu xác nhận."
    
            st.markdown('<div class="major-sec">DECISION LAYER</div>', unsafe_allow_html=True)
    
            render_decision_layer_switch(cp, ap, gate_status, exec_mode_text, primary_name)
    
    
    
        with right_col:
            st.markdown("""<div style='border:1px dashed #E5E7EB;border-radius:14px;padding:14px;color:#64748B;font-weight:800;'>BIỂU ĐỒ (SẼ BỔ SUNG)</div>""", unsafe_allow_html=True)

def render_report_pretty(report_text: str, analysis_pack: dict):
    sections = _split_sections(report_text)
    a_items = _extract_a_items(sections.get("A", ""))

    st.markdown('<div class="incept-wrap">', unsafe_allow_html=True)

    ap = _ensure_dict(analysis_pack)
    scenario_pack = ap.get("Scenario12") or {}
    master_pack = ap.get("MasterScore") or {}
    conviction_score = ap.get("Conviction", "?")

    st.markdown(
        f"""
        <div class="report-header">
          <h2 style="margin:0; padding:0;">{_val_or_na(ap.get("Ticker"))} - {_val_or_na(scenario_pack.get("Name"))}</h2>
          <div style="font-size:16px; font-weight:700; margin-top:4px;">
            Điểm tổng hợp: {_val_or_na(master_pack.get("Total"))} | Điểm tin cậy: {_val_or_na(conviction_score)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sec-title">TECHNICAL ANALYSIS</div>', unsafe_allow_html=True)
    a_raw = sections.get("A", "").strip()
    a_body = re.sub(r"(?mi)^A\..*\n?", "", a_raw).strip()
    if a_items:
        for i, body in enumerate(a_items, start=1):
            if not body.strip():
                continue
            st.markdown(
                f"""
                <div class="incept-card">
                  <div style="font-weight:800; margin-bottom:6px;">{i}.</div>
                  <div>{body}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(a_body, unsafe_allow_html=False)

    st.markdown('<div class="sec-title">TRADE PLAN</div>', unsafe_allow_html=True)
    c = sections.get("C", "").strip()
    if c:
        c_body = re.sub(r"(?m)^C\..*\n?", "", c).strip()
        st.markdown(
            f"""
            <div class="incept-card">
              <div>{c_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info(UI_PLACEHOLDER)

    st.markdown('<div class="sec-title">RỦI RO &amp; LỢI NHUẬN</div>', unsafe_allow_html=True)
    ps = (_ensure_dict(analysis_pack)).get("PrimarySetup") or {}
    risk = ps.get("RiskPct", None)
    reward = ps.get("RewardPct", None)
    rr = ps.get("RR", None)
    prob = ps.get("Confidence (Tech)", ps.get("Probability", UI_PLACEHOLDER))

    def _fmt_pct_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return UI_PLACEHOLDER
            return f"{float(x):.2f}%"
        except Exception:
            return UI_PLACEHOLDER

    def _fmt_rr_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return UI_PLACEHOLDER
            return f"{float(x):.2f}"
        except Exception:
            return UI_PLACEHOLDER
        try:
            return f"{float(x):.2f}"
        except Exception:
            return UI_PLACEHOLDER

    st.markdown(
        f"""
        <div class="incept-metrics">
          <div class="incept-metric"><div class="k">Risk%:</div><div class="v">{_fmt_pct_local(risk)}</div></div>
          <div class="incept-metric"><div class="k">Reward%:</div><div class="v">{_fmt_pct_local(reward)}</div></div>
          <div class="incept-metric"><div class="k">RR:</div><div class="v">{_fmt_rr_local(rr)}</div></div>
          <div class="incept-metric"><div class="k">Confidence (Tech):</div><div class="v">{prob}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
