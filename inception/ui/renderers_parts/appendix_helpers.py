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
from inception.core.battle_map_pack import compute_battle_map_pack_v1
from inception.ui.render_charts import render_battle_map_chart_v1, render_price_map_chart_v1
from .utils import (
    _ensure_dict,
    _safe_text,
    _safe_float,
    _clip,
    _val_or_na,
    get_class_policy_hint_line,
)


def render_decision_orientation_dashboard_v11(result: Dict[str, Any], analysis_pack: Dict[str, Any]) -> None:
    """Dashboard #2 (Decision Orientation Layer) - persona-first, pack-only."""
    r = _ensure_dict(result)
    ap = _ensure_dict(r.get("AnalysisPack") or analysis_pack)
    modules = r.get("Modules") if isinstance(r.get("Modules"), dict) else {}
    cp = _ensure_dict((modules or {}).get("character") or {})

    class_name = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "").strip()
    if not class_name:
        class_name = "Data unavailable."
    if class_name == "Illiquid / Noisy":
        class_name = "Thanh khoản mỏng – Biến động nhiễu"

    st.markdown("## Decision Orientation")
    row1 = st.columns(2, gap="large")
    row2 = st.columns(2, gap="large")

    with row1[0]:
        tile = st.container()
        with tile:
            st.markdown(f"**{html.escape(class_name)}**")
            st.caption("Defines the stock's dominant long-term behavior.")

            pack = None
            if isinstance(ap.get("InvestorMappingPack"), dict):
                pack = ap.get("InvestorMappingPack")
            elif isinstance(r.get("InvestorMappingPack"), dict):
                pack = r.get("InvestorMappingPack")
            pentagon = pack.get("Pentagon") if isinstance(pack, dict) and isinstance(pack.get("Pentagon"), dict) else None
            has_pack = isinstance(pack, dict) and isinstance(pentagon, dict)
            if not has_pack:
                st.markdown("Data unavailable.")
                pack = {}
                pentagon = {}

            if has_pack:
                st.caption("Investor fit and risk appetite (0-10).")

            axis_map = [
                ("Trend Power", ["TrendPower", "Trend Power"]),
                ("Explosive", ["Explosive"]),
                ("Safety Shield", ["SafetyShield", "Safety Shield"]),
                ("Trading Flow", ["TradingFlow", "Trading Flow"]),
                ("Adrenaline", ["Adrenaline"]),
            ]
            axis_display = {
                "Trend Power": ("Sức mạnh", "Tính bền vững của xu hướng dài hạn"),
                "Explosive": ("Sức bật", "Đánh giá khả năng bứt tốc ngắn hạn"),
                "Safety Shield": ("An toàn", "Khả năng phòng thủ trước các cú sập"),
                "Trading Flow": ("Lướt sóng", "Mức độ vào/thoát hàng dễ dàng và hiệu quả"),
                "Adrenaline": ("Cảm giác mạnh", "Khả năng biến động cực mạnh"),
            }

            def _axis_vi(axis_key: str) -> str:
                return axis_display.get(axis_key, (axis_key, ""))[0]

            def _pick_val(d: Dict[str, Any], keys: List[str]) -> float:
                for k in keys:
                    if k in d:
                        return _safe_float(d.get(k), default=np.nan)
                return np.nan

            def _bar(label: str, value: Any) -> None:
                v = _safe_float(value, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "?"
                else:
                    v10 = float(max(0.0, min(10.0, float(v))))
                    pct = _clip(v10 / 10.0 * 100.0, 0.0, 100.0)
                    v_disp = f"{v10:.1f}/10"
                label_vi, caption = axis_display.get(label, (label, ""))
                caption_html = (
                    f"<div style=\"font-size:11px;font-style:italic;color:#6B7280;margin-top:2px;\">"
                    f"{html.escape(caption)}</div>"
                    if caption
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="gc-row">
                      <div class="gc-k">
                        <div style="font-weight:700;">{html.escape(str(label_vi))}</div>
                        {caption_html}
                      </div>
                      <div class="gc-bar"><div class="gc-fill" style="width:{pct:.0f}%"></div></div>
                      <div class="gc-v">{html.escape(str(v_disp))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if has_pack:
                for label, keys in axis_map:
                    _bar(label, _pick_val(pentagon, keys))

            def _persona_items(p: Dict[str, Any]) -> List[Dict[str, Any]]:
                src = None
                for key in ("Personas", "PersonaMatch", "Compatibility", "personas", "persona_match", "compatibility"):
                    if key in p:
                        src = p.get(key)
                        break
                items: List[Dict[str, Any]] = []
                if isinstance(src, list):
                    for it in src:
                        if isinstance(it, dict):
                            items.append(it)
                elif isinstance(src, dict):
                    for k, it in src.items():
                        if isinstance(it, dict):
                            it = dict(it)
                            it.setdefault("name", k)
                            items.append(it)
                return items

            personas_raw = _persona_items(pack) if isinstance(pack, dict) else []
            personas = [p for p in personas_raw if isinstance(p, dict)] if isinstance(personas_raw, list) else []

            def _score(item: Dict[str, Any]) -> float:
                return _safe_float(item.get("score_10") or item.get("Score10") or item.get("score"), default=np.nan)

            def _is_vetoed(item: Dict[str, Any]) -> bool:
                return bool(item.get("vetoed") or item.get("Vetoed"))

            persona_display = {
                "Compounder": "Tích sản/Lãi kép",
                "AlphaHunter": "Săn sóng lớn",
                "CashFlowTrader": "Lướt sóng ngắn",
                "Speculator": "Đầu cơ",
            }
            persona_pref = {
                "Compounder": "xu hướng bền và an toàn, ít thích cảm giác mạnh",
                "AlphaHunter": "bùng nổ và xu hướng mạnh, chấp nhận biến động vừa phải",
                "CashFlowTrader": "lướt sóng nhanh và biến động khá, thường không đặt nặng an toàn",
                "Speculator": "biến động cực mạnh, các yếu tố khác chỉ mang tính phụ",
            }

            def _persona_name(p: Dict[str, Any]) -> str:
                return _safe_text(p.get("name") or p.get("Name") or p.get("Persona") or "").strip()

            def _persona_display_name(name: str) -> str:
                return persona_display.get(name, name)

            def _persona_pref(name: str) -> str:
                return persona_pref.get(name, "khẩu vị riêng, không nhất thiết hợp với mọi hồ sơ")

            def _persona_brief(p: Dict[str, Any]) -> str:
                b = _safe_text(p.get("brief") or "").strip()
                return b if b else "Profile details unavailable."

            ranked = [p for p in personas if isinstance(p, dict)]
            ranked.sort(key=lambda x: (_safe_float(_score(x), default=-1e9)), reverse=True)
            non_veto = [p for p in ranked if not _is_vetoed(p)]
            top_persona = non_veto[0] if non_veto else (ranked[0] if ranked else {})
            bottom_persona = ranked[-1] if ranked else {}

            persona_rows = [p for p in ranked if _persona_name(p)]
            persona_rows = persona_rows[:4]

            if persona_rows:
                for item in persona_rows:
                    name = _persona_display_name(_persona_name(item))
                    score = _score(item)
                    label = _safe_text(item.get("Label") or item.get("label") or "").strip()
                    vetoed = _is_vetoed(item)
                    pct = 0.0 if pd.isna(score) else float(max(0.0, min(100.0, (float(score) / 10.0) * 100.0)))
                    val_txt = "?" if pd.isna(score) else f"{float(score):.1f}/10"
                    tag_label = label
                    if not tag_label and not pd.isna(score) and isinstance(pack, dict):
                        meta = pack.get("Meta") if isinstance(pack.get("Meta"), dict) else {}
                        labels = meta.get("labels") if isinstance(meta.get("labels"), dict) else {}
                        match_min = _safe_float(labels.get("match_min"), default=7.5)
                        partial_min = _safe_float(labels.get("partial_min"), default=4.5)
                        if float(score) >= match_min:
                            tag_label = "Match"
                        elif float(score) >= partial_min:
                            tag_label = "Partial"
                        else:
                            tag_label = "Mismatch"
                    color = "#9CA3AF"
                    if tag_label == "Match":
                        color = "#22C55E"
                    elif tag_label == "Partial":
                        color = "#F59E0B"
                    if vetoed:
                        color = "#64748B"
                    badge_labels = {"Match": "Phù hợp", "Partial": "Cân nhắc", "Mismatch": "Không phù hợp"}
                    tag_label_vi = badge_labels.get(tag_label, tag_label)
                    if vetoed:
                        tag_label_vi = f"{tag_label_vi} (Chặn)" if tag_label_vi else "Không phù hợp (Chặn)"
                        tag_label_vi = f"⚠ {tag_label_vi}"
                    tag = tag_label_vi or ("⚠ Không phù hợp (Chặn)" if vetoed else "")
                    badge = f"<span style='padding:2px 8px;border-radius:999px;border:1px solid {color};color:{color};font-size:12px;font-weight:800;'>{html.escape(tag)}</span>" if tag else ""
                    st.markdown(
                        f"""
                        <div style="display:flex;gap:10px;align-items:center;margin:6px 0;">
                          <div style="width:150px;font-size:14px;color:#374151;font-weight:700;">{html.escape(name)}</div>
                          <div style="flex:1;height:12px;background:#F3F4F6;border-radius:999px;overflow:hidden;">
                            <div style="height:12px;width:{pct:.0f}%;background:{color};border-radius:999px;opacity:{'0.45' if vetoed else '1'};"></div>
                          </div>
                          <div style="width:70px;text-align:right;font-size:13px;color:#111827;font-weight:800;">{html.escape(val_txt)}</div>
                        </div>
                        {badge}
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("Persona mapping: Data unavailable.")

            if top_persona and bottom_persona:
                st.markdown(f"**Phù hợp nhất:** {html.escape(_persona_display_name(_persona_name(top_persona)))}")
                st.markdown(f"**Ít phù hợp:** {html.escape(_persona_display_name(_persona_name(bottom_persona)))}")
            else:
                st.markdown("**Phù hợp nhất:** Data unavailable.")
                st.markdown("**Ít phù hợp:** Data unavailable.")

            tp = _pick_val(pentagon, ["TrendPower", "Trend Power"])
            ss = _pick_val(pentagon, ["SafetyShield", "Safety Shield"])
            tf = _pick_val(pentagon, ["TradingFlow", "Trading Flow"])
            ad = _pick_val(pentagon, ["Adrenaline"])

            if has_pack and top_persona and bottom_persona:
                axes = [
                    ("Trend Power", tp),
                    ("Explosive", _pick_val(pentagon, ["Explosive"])),
                    ("Safety Shield", ss),
                    ("Trading Flow", tf),
                    ("Adrenaline", ad),
                ]
                top_axes = sorted(
                    axes,
                    key=lambda x: (float(x[1]) if not pd.isna(x[1]) else -1e9),
                    reverse=True,
                )
                low_axes = sorted(
                    axes,
                    key=lambda x: (float(x[1]) if not pd.isna(x[1]) else 1e9),
                )
                best_axis_1 = top_axes[0][0] if top_axes else "Compatibility"
                best_axis_2 = top_axes[1][0] if len(top_axes) > 1 else ""
                low_axis = low_axes[0][0] if low_axes else "Compatibility"
                best_axis_1_vi = _axis_vi(best_axis_1)
                best_axis_2_vi = _axis_vi(best_axis_2) if best_axis_2 else ""
                low_axis_vi = _axis_vi(low_axis)
                top_name = _persona_name(top_persona)
                bottom_name = _persona_name(bottom_persona)
                top_display = _persona_display_name(top_name)
                bottom_display = _persona_display_name(bottom_name)
                top_pref = _persona_pref(top_name)
                bottom_pref = _persona_pref(bottom_name)
                if best_axis_2_vi and best_axis_2_vi != best_axis_1_vi:
                    st.markdown(
                        f"{best_axis_1_vi} và {best_axis_2_vi} là hai nét nổi bật, nên hợp với {html.escape(top_display)} — nhóm thường thích {html.escape(top_pref)}."
                    )
                else:
                    st.markdown(
                        f"{best_axis_1_vi} là nét nổi bật, nên hợp với {html.escape(top_display)} — nhóm thường thích {html.escape(top_pref)}."
                    )
                st.markdown(
                    f"{low_axis_vi} ở mức khiêm tốn hơn, nên {html.escape(bottom_display)} — nhóm thường thích {html.escape(bottom_pref)} — có thể thấy không \"đúng gu\"."
                )
            else:
                st.markdown("Data unavailable.")

            st.markdown(
                "If this profile aligns with how you usually invest,\n"
                "you may continue to the detailed analysis to explore structure, risk considerations, and timing context.\n\n"
                "If this profile does not reflect your typical approach,\n"
                "it’s reasonable to stop here and focus on opportunities better aligned with your style.\n\n"
                "Market conditions remain uncertain, and compatibility does not imply timing."
            )

    with row1[1]:
        st.caption("(Reserved)")
    with row2[0]:
        st.caption("(Reserved)")
    with row2[1]:
        st.caption("(Reserved)")
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

    def _resolve_df_for_battle_map() -> Any:
        df = None
        if isinstance(r, dict):
            df = r.get("_DF")
        if df is None and isinstance(ap, dict):
            df = ap.get("_DF") or ap.get("df") or ap.get("DF")
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            return df
        try:
            return pd.DataFrame(df)
        except Exception:
            return None

    def _infer_trend_label(last_close_val: float) -> str:
        last = ap.get("Last") if isinstance(ap.get("Last"), dict) else {}
        ma50 = _safe_float(last.get("MA50"), default=math.nan)
        ma200 = _safe_float(last.get("MA200"), default=math.nan)
        if math.isfinite(last_close_val) and math.isfinite(ma50) and math.isfinite(ma200):
            if last_close_val > ma50 > ma200:
                return "Tăng"
            if last_close_val < ma50 < ma200:
                return "Giảm"
            return "Đi ngang"
        df_tmp = _resolve_df_for_battle_map()
        if df_tmp is None or df_tmp.empty or "Close" not in df_tmp.columns:
            return "Đi ngang"
        close_s = pd.to_numeric(df_tmp["Close"], errors="coerce").dropna()
        if close_s.empty:
            return "Đi ngang"
        tail = close_s.tail(60)
        if tail.empty:
            return "Đi ngang"
        first = float(tail.iloc[0])
        last_val = float(tail.iloc[-1])
        if first <= 0:
            return "Đi ngang"
        if last_val > first * 1.03:
            return "Tăng"
        if last_val < first * 0.97:
            return "Giảm"
        return "Đi ngang"

    def _zone_label(z: Dict[str, Any], side: str) -> str:
        zt = str(z.get("zone_type") or "")
        side_norm = str(side or "").upper()
        if zt == "EXPECTATION_TARGET":
            return "mốc kỳ vọng (target)"
        if zt == "EXPECTATION_POTENTIAL":
            return "mốc tiềm năng"
        return "vùng hỗ trợ (wall)" if side_norm == "SUPPORT" else "vùng cản (wall)"

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

    # ---------- Dashboard #1 (Battle Map) ----------
    df_battle = _resolve_df_for_battle_map()
    battle_map_pack = None
    if df_battle is not None:
        try:
            battle_map_pack = compute_battle_map_pack_v1(df_battle, analysis_pack=analysis_pack)
        except Exception as exc:
            st.info(f"Không thể tạo Battle Map: {exc.__class__.__name__}")
            battle_map_pack = None

    def _bm_safe_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _bm_fmt(x, nd=2):
        v = _bm_safe_float(x)
        return None if v is None else f"{v:.{nd}f}"

    def render_battle_map_header_v1(ticker_input, df_1y):
        import streamlit as st

        # ---------- HARD NORMALIZE TICKER ----------
        ticker = (ticker_input or "").strip().upper()
        if not ticker:
            ticker = "--"

        # ---------- PRICE ----------
        last_close = None
        prev_close = None

        if df_1y is not None and "Close" in df_1y.columns:
            if len(df_1y) >= 1:
                last_close = float(df_1y["Close"].iloc[-1])
            if len(df_1y) >= 2:
                prev_close = float(df_1y["Close"].iloc[-2])

        if last_close is None:
            st.markdown(
                f"<div style='font-size:32px;font-weight:800;color:#fff'>{ticker} - --</div>",
                unsafe_allow_html=True,
            )
            return

        # ---------- CHANGE ----------
        chg = None
        chg_pct = None
        if prev_close and prev_close != 0:
            chg = last_close - prev_close
            chg_pct = chg / prev_close * 100

        if chg is None:
            color = "#F5C542"  # yellow
            chg_txt = "[-]"
        else:
            if chg > 0:
                color = "#7CFF00"  # green
                sign = "+"
            elif chg < 0:
                color = "#FF2D2D"  # red
                sign = ""
            else:
                color = "#F5C542"
                sign = ""
            chg_txt = f"{sign}{chg:.2f} [{sign}{chg_pct:.2f}%]"

        # ---------- RENDER ----------
        st.markdown(
            f"""
            <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:6px;">
              <span style="font-size:34px;font-weight:800;color:#ffffff;">{ticker}</span>
              <span style="font-size:34px;font-weight:800;color:#ffffff;">-</span>
              <span style="font-size:34px;font-weight:800;color:#ffffff;">{last_close:.2f}</span>
              <span style="font-size:34px;font-weight:800;color:{color};
                           text-shadow:0 0 10px rgba(255,255,255,0.2);">
                {chg_txt}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        ticker_input = _safe_text(st.session_state.get("ticker_input") or "").strip().upper()
        render_battle_map_header_v1(ticker_input, df_battle)
        if isinstance(battle_map_pack, dict):
            render_battle_map_chart_v1(df_battle, battle_map_pack)
        elif df_battle is not None:
            render_price_map_chart_v1(df_battle, analysis_pack=analysis_pack)
        else:
            st.info("Chưa có dữ liệu giá để hiển thị Battle Map.")

    with col2:
        st.markdown("### Advisor Lens")
        if isinstance(battle_map_pack, dict):
            zs = (battle_map_pack.get("zones_selected") or {}) if isinstance(battle_map_pack.get("zones_selected"), dict) else {}
            supports = zs.get("supports") or []
            resists = zs.get("resistances") or []
            ref_price = _safe_float(battle_map_pack.get("reference_price"), default=math.nan)
            last_close = _safe_float(ref_price, default=math.nan)
            trend_label = _infer_trend_label(last_close)
            ticker = _safe_text(ap.get("Symbol") or ap.get("symbol") or ap.get("ticker") or "").upper()

            def _zone_center(z: Dict[str, Any]) -> float:
                return _safe_float(z.get("center"), default=math.nan)

            support_0 = supports[0] if supports else {}
            resist_0 = resists[0] if resists else {}
            support_px = _zone_center(support_0)
            resist_px = _zone_center(resist_0)
            support_label = _zone_label(support_0, "SUPPORT") if support_0 else "vùng hỗ trợ (wall)"
            resist_label = _zone_label(resist_0, "RESISTANCE") if resist_0 else "vùng cản (wall)"
            support_tier = _safe_text(support_0.get("tier") or "").strip().upper()
            resist_tier = _safe_text(resist_0.get("tier") or "").strip().upper()
            support_reason = _safe_text((support_0.get("reasons") or [""])[0]).strip()
            resist_reason = _safe_text((resist_0.get("reasons") or [""])[0]).strip()

            lines = []
            if math.isfinite(last_close) and math.isfinite(resist_px):
                near_high = abs(last_close - resist_px) / max(resist_px, 1e-9) < 0.08
            else:
                near_high = False
            if math.isfinite(last_close) and math.isfinite(support_px):
                near_low = abs(last_close - support_px) / max(support_px, 1e-9) < 0.08
            else:
                near_low = False
            if near_high:
                context_line = "giá đang vận động gần vùng cao của biên 1 năm."
            elif near_low:
                context_line = "giá đang vận động gần vùng thấp của biên 1 năm."
            else:
                context_line = "giá đang ở vùng giữa biên 1 năm."
            lines.append(f"Bối cảnh: Xu hướng {trend_label.lower()}, {context_line}")
            lines.append("")
            lines.append("Mốc chính:")
            if math.isfinite(support_px):
                tier_note = f"tier {support_tier}" if support_tier else "tier ?"
                reason_note = f", {support_reason}" if support_reason else ""
                lines.append(f"  – Hỗ trợ gần: {support_px:.1f} ({tier_note}{reason_note}).")
            if math.isfinite(resist_px):
                tier_note = f"tier {resist_tier}" if resist_tier else "tier ?"
                reason_note = f", {resist_reason}" if resist_reason else ""
                lines.append(f"  – Phía trên: {resist_label} {resist_px:.1f} ({tier_note}{reason_note}).")
            else:
                lines.append("  – Phía trên chưa có 'wall' gần; thị trường có thể chạy theo kỳ vọng, nhưng dễ rung lắc.")

            lines.append("")
            lines.append("Kịch bản giảm:")
            if math.isfinite(support_px):
                lines.append(f"  – Giữ trên {support_px:.1f}: có thể hình thành nhịp hồi kỹ thuật.")
                lines.append(f"  – Đóng cửa dưới {support_px:.1f}: rủi ro trượt sâu hơn, ưu tiên phòng thủ.")
            else:
                lines.append("  – Chờ phản ứng rõ tại vùng hỗ trợ.")
                lines.append("  – Mất hỗ trợ: ưu tiên phòng thủ.")
            lines.append("")
            lines.append("Kịch bản tăng:")
            if math.isfinite(resist_px):
                lines.append(f"  – Tiệm cận {resist_px:.1f}: dễ rung lắc hoặc chững lại.")
                lines.append(f"  – Vượt và giữ trên {resist_px:.1f}: cấu trúc cho phép mở rộng đà tăng.")
            else:
                lines.append("  – Quan sát nhịp mở rộng khi không có wall gần.")
                lines.append("  – Rung lắc mạnh trước khi hình thành kháng cự rõ.")
            lines.append("")
            lines.append("Lưu ý: Thị trường không chắc chắn; vùng giá cao thường đi kèm biến động lớn.")
            lines.append("")
            lines.append("Gợi ý theo khẩu vị:")
            lines.append("  – Ưu tiên an toàn: chờ phản ứng rõ tại hỗ trợ.")
            lines.append("  – Chấp nhận biến động: quan sát hành vi giá tại vùng cản phía trên.")

            for line in lines[:14]:
                if line:
                    st.markdown(f"- {line}")
                else:
                    st.markdown("")
        else:
            st.caption("Chưa đủ dữ liệu để tạo narrative.")

    # ---------- Dashboard #2 (Decision Orientation Layer) ----------
    render_decision_orientation_dashboard_v11(r, ap)

    # Optional legacy dashboard view
    with st.expander("Legacy dashboard (Executive Snapshot)", expanded=False):
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

            st.markdown("<div style='margin:6px 0 0 0;'></div>", unsafe_allow_html=True)

            # Payoff Tier (economic viability only; no gating)
            sp = ap.get("StatusPack") if isinstance(ap.get("StatusPack"), dict) else {}
            payoff = sp.get("payoff") if isinstance(sp.get("payoff"), dict) else {}
            payoff_tier = _safe_text(payoff.get("tier") or "").strip().upper()
            payoff_note = _safe_text(payoff.get("note") or "").strip()
            if payoff_tier:
                st.markdown(
                    f"<div style='font-weight:600;'>Payoff (Tier): {html.escape(payoff_tier)}</div>",
                    unsafe_allow_html=True,
                )
                if payoff_note:
                    st.markdown(
                        f"<div style='margin-top:4px;color:#6B7280;font-size:12px;line-height:1.3;'>"
                        f"{html.escape(payoff_note)}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='margin:8px 0 0 0;'></div>", unsafe_allow_html=True)

            st.markdown("### Đồ thị giá")
            show_legacy_chart = st.checkbox("Render chart here (legacy)", value=False, key="legacy_battle_map_chart")
            if not show_legacy_chart:
                st.caption("Legacy view (details).")
            else:

                if df_battle is None:
                    st.info("Chưa có dữ liệu giá để hiển thị đồ thị.")
                elif isinstance(battle_map_pack, dict):
                    render_battle_map_chart_v1(df_battle, battle_map_pack)
                else:
                    render_price_map_chart_v1(df_battle, analysis_pack=analysis_pack)

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
