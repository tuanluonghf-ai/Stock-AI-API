from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import html
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from inception.ui.ui_constants import UI_PLACEHOLDER
from .utils import (
    _ensure_dict,
    _safe_text,
    _safe_float,
    _clip,
    _val_or_na,
    _pick_character_narrative,
)


def render_character_traits(character_pack: Dict[str, Any]) -> None:
    """
    STOCK DNA (Long-run 3–5Y) — STRICT layer.
    Displays ONLY:
      - Class + stable class narrative
      - Tier-1 (StyleAxis, RiskRegime) + DNAConfidence
      - DNA group scores (5 groups)
      - The 15-parameter pack (inside an expander)

    Deliberately excludes legacy "CORE STATS" and any 'Now/Opportunity' metrics.
    """
    cp = _ensure_dict(character_pack)
    cclass = _safe_text(cp.get("CharacterClass") or UI_PLACEHOLDER).strip()
    ticker = _safe_text(cp.get("_Ticker") or "").strip().upper()

    # ---- Class narrative (stable templates) ----
    class_label = f"CLASS: {cclass}"
    st.markdown(f"**{html.escape(class_label)}**")
    st.markdown(_pick_character_narrative(cp))

    # ---- DNA pack (15 params / 5 groups) ----
    dna = (_ensure_dict(cp.get("StockTraits"))).get("DNA") or {}
    tier1 = dna.get("Tier1") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    def _bar(label: str, value: Any) -> None:
        v = _safe_float(value, default=np.nan)
        if pd.isna(v):
            pct = 0.0
            v_disp = "?"
        else:
            v10 = float(max(0.0, min(10.0, float(v))))
            pct = _clip(v10 / 10.0 * 100.0, 0.0, 100.0)
            v_disp = f"{v10:.1f}/10"
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

    style_axis = tier1.get("StyleAxis", "?")
    risk_regime = tier1.get("RiskRegime", "?")
    dna_conf = tier1.get("DNAConfidence", np.nan)
    lock_flag = tier1.get("ClassLockFlag", False)
    modifiers = tier1.get("Modifiers", []) or []

    conf_txt = "?" if pd.isna(_safe_float(dna_conf, default=np.nan)) else f"{float(dna_conf):.0f}/100"
    mod_txt = ", ".join([str(x) for x in modifiers]) if isinstance(modifiers, list) and modifiers else "None"
    lock_txt = "LOCKED (low confidence)" if bool(lock_flag) else "OK"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">STOCK DNA (LONG-RUN 3–5Y)</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='gc-muted'>Tier-1: Style {html.escape(str(style_axis))} | Risk {html.escape(str(risk_regime))} | "
        f"DNAConfidence: {html.escape(conf_txt)} | {html.escape(lock_txt)}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='gc-muted'>Modifiers: {html.escape(mod_txt)}</div>",
        unsafe_allow_html=True
    )

    # 5 groups (stable anchors)
    _bar("Trend Structure", _avg(groups.get("TrendStructure", ["TrendIntegrity", "TrendPersistence", "TrendChurnControl"])))
    _bar("Volatility & Tail Risk (higher = worse)", _avg(groups.get("VolatilityTail", ["VolRisk", "TailGapRisk", "VolOfVolRisk"])))
    _bar("Drawdown & Recovery Risk (higher = worse)", _avg(groups.get("DrawdownRecovery", ["MaxDrawdownRisk", "RecoverySlownessRisk", "DrawdownFrequencyRisk"])))
    _bar("Liquidity & Tradability", _avg(groups.get("LiquidityTradability", ["LiquidityTradability", "LiquidityLevel", "LiquidityConsistency"])))
    _bar("Behavior / Setup Bias", _avg(groups.get("BehaviorSetup", ["BreakoutQuality", "MeanReversionWhipsaw", "AutoCorrMomentum"])))
    st.markdown("</div>", unsafe_allow_html=True)

    # 15 parameters (optional detail)
    label_map = {
        "TrendIntegrity": "Trend Integrity",
        "TrendPersistence": "Trend Persistence",
        "TrendChurnControl": "Trend Churn Control",
        "VolRisk": "Volatility Level (risk)",
        "TailGapRisk": "Tail/Gap Risk",
        "VolOfVolRisk": "Vol Regime Instability",
        "MaxDrawdownRisk": "Max Drawdown Risk",
        "RecoverySlownessRisk": "Recovery Slowness",
        "DrawdownFrequencyRisk": "Drawdown Frequency",
        "LiquidityTradability": "Tradability",
        "LiquidityLevel": "Liquidity Level",
        "LiquidityConsistency": "Liquidity Consistency",
        "BreakoutQuality": "Breakout Quality",
        "MeanReversionWhipsaw": "Mean-Reversion / Whipsaw",
        "AutoCorrMomentum": "Momentum Autocorr",
    }
    group_labels = {
        "TrendStructure": "Group 1 — Trend Structure",
        "VolatilityTail": "Group 2 — Volatility & Tail",
        "DrawdownRecovery": "Group 3 — Drawdown & Recovery",
        "LiquidityTradability": "Group 4 — Liquidity & Tradability",
        "BehaviorSetup": "Group 5 — Behavior / Setup Bias",
    }

    with st.expander("DNA Parameters (15)", expanded=False):
        for gk in ["TrendStructure", "VolatilityTail", "DrawdownRecovery", "LiquidityTradability", "BehaviorSetup"]:
            keys = groups.get(gk) or []
            st.markdown(f"**{group_labels.get(gk, gk)}**")
            for k in keys:
                _bar(label_map.get(k, k), params.get(k))

def render_combat_stats_panel(character_pack: Dict[str, Any]) -> None:
    """Render Combat Stats as 'Now / Opportunity' metrics (0–10), intended to live under CURRENT STATUS."""
    cp = _ensure_dict(character_pack)
    combat = _ensure_dict(cp.get("CombatStats"))

    combat_order = [
        ("Upside Power", combat.get("UpsidePower")),
        ("Downside Risk", combat.get("DownsideRisk")),
        ("RR Efficiency", combat.get("RREfficiency")),
        ("Breakout Force", combat.get("BreakoutForce")),
        ("Support Resilience", combat.get("SupportResilience")),
    ]

    def bar_0_10(label: str, value: Any) -> None:
        v = _safe_float(value, default=np.nan)
        if pd.isna(v):
            pct = 0.0
            v_disp = "?"
        else:
            v10 = float(max(0.0, min(10.0, float(v))))
            pct = _clip(v10 / 10.0 * 100.0, 0.0, 100.0)
            v_disp = f"{v10:.1f}/10"

        st.markdown(
            f"""
            <div class="gc-row">
              <div class="gc-k">{html.escape(str(label))}</div>
              <div class="gc-bar"><div class="gc-fill" style="width:{pct:.1f}%"></div></div>
              <div class="gc-v">{html.escape(v_disp)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">COMBAT STATS (NOW)</div>', unsafe_allow_html=True)
    for label, value in combat_order:
        bar_0_10(label, value)
    st.markdown("</div>", unsafe_allow_html=True)

def render_stock_dna_insight(character_pack: Dict[str, Any]) -> None:
    """
    DNA Insight — MUST stay in the long-run layer.
    No 'Now/Opportunity' metrics allowed here.
    """
    cp = _ensure_dict(character_pack)
    cclass = _safe_text(cp.get("CharacterClass") or "?").strip()

    dna = (_ensure_dict(cp.get("StockTraits"))).get("DNA") or {}
    tier1 = dna.get("Tier1") or {}
    params = dna.get("Params") or {}
    groups = dna.get("Groups") or {}

    def _avg(keys: List[str]) -> float:
        vals: List[float] = []
        for k in keys:
            v = _safe_float(params.get(k), default=np.nan)
            if not pd.isna(v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    trend_g = _avg(groups.get("TrendStructure", []))
    vol_g = _avg(groups.get("VolatilityTail", []))
    dd_g = _avg(groups.get("DrawdownRecovery", []))
    liq_g = _avg(groups.get("LiquidityTradability", []))
    beh_g = _avg(groups.get("BehaviorSetup", []))

    style_axis = _safe_text(tier1.get("StyleAxis", "?"))
    risk_regime = _safe_text(tier1.get("RiskRegime", "?"))
    dna_conf = _safe_float(tier1.get("DNAConfidence", np.nan), default=np.nan)
    modifiers = tier1.get("Modifiers", []) or []

    # Interpretive tags (no extra computation beyond existing scores)
    strengths: List[str] = []
    cautions: List[str] = []

    if pd.notna(trend_g) and trend_g >= 6.8:
        strengths.append("Trend structure tương đối bền")
    if pd.notna(liq_g) and liq_g >= 6.2:
        strengths.append("Tính tradable/khớp lệnh khá ổn")

    if pd.notna(vol_g) and vol_g >= 6.8:
        cautions.append("Rủi ro biến động/tail cao hơn mức trung bình")
    if pd.notna(dd_g) and dd_g >= 6.8:
        cautions.append("Drawdown có thể sâu hoặc hồi phục chậm")
    if "ILLIQ" in modifiers:
        cautions.append("Execution risk cao do thanh khoản/ổn định thanh khoản yếu")
    if "GAP" in modifiers:
        cautions.append("Gap/event risk nổi bật; tránh nắm giữ quá tự tin qua thời điểm nhạy cảm")

    s_txt = "; ".join(strengths) if strengths else "Hành vi dài hạn ở mức trung tính"
    c_txt = "; ".join(cautions) if cautions else "Không có cảnh báo DNA nổi bật"

    conf_txt = "?" if pd.isna(dna_conf) else f"{dna_conf:.0f}/100"

    st.markdown("**DNA Insight (Long-run)**")
    st.markdown(f"- Class: **{html.escape(cclass)}** | Tier-1: Style **{html.escape(style_axis)}**, Risk **{html.escape(risk_regime)}** | DNAConfidence: **{html.escape(conf_txt)}**")
    st.markdown(f"- Strengths: {html.escape(s_txt)}")
    st.markdown(f"- Cautions: {html.escape(c_txt)}")

    # Strategy fit (high-level)
    if style_axis == "Trend":
        fit = "Ưu tiên chiến lược theo xu hướng: pullback/continuation; tránh bắt đáy ngược trend."
    elif style_axis == "Range":
        fit = "Ưu tiên chiến lược đánh biên: mua gần hỗ trợ – bán gần kháng cự; tránh mua đuổi giữa biên."
    elif style_axis == "Momentum":
        fit = "Ưu tiên chiến lược theo động lượng: breakout có xác nhận; quản trị vị thế chủ động."
    else:
        fit = "Ưu tiên chiến thuật chọn lọc: chỉ trade khi setup thật rõ và có xác nhận."

    st.markdown(f"- Fit: {html.escape(fit)}")


def render_investor_mapping_v1(analysis_pack: Dict[str, Any], mode: str = "teaser") -> None:
    """Render Investor Fit & Risk Appetite (pack-only)."""
    ap_in = analysis_pack if isinstance(analysis_pack, dict) else {}
    ap = _ensure_dict(ap_in.get("AnalysisPack") if isinstance(ap_in.get("AnalysisPack"), dict) else ap_in)
    pack = ap.get("InvestorMappingPack") if isinstance(ap.get("InvestorMappingPack"), dict) else None

    st.markdown("### Investor Fit & Risk Appetite")
    st.caption("Compatibility across investment styles, normalized on a 0-10 scale.")

    if not isinstance(pack, dict):
        st.markdown("Data unavailable.")
        return

    meta = pack.get("Meta") if isinstance(pack.get("Meta"), dict) else None
    pentagon = pack.get("Pentagon") if isinstance(pack.get("Pentagon"), dict) else None
    if not isinstance(meta, dict) or not isinstance(pentagon, dict):
        st.markdown("Data unavailable.")
        return

    labels = meta.get("labels") if isinstance(meta.get("labels"), dict) else {}
    match_min = _safe_float(labels.get("match_min"), default=7.5)
    partial_min = _safe_float(labels.get("partial_min"), default=4.5)

    axis_map = [
        ("Trend Power", ["TrendPower", "Trend Power"]),
        ("Explosive", ["Explosive"]),
        ("Safety Shield", ["SafetyShield", "Safety Shield"]),
        ("Trading Flow", ["TradingFlow", "Trading Flow"]),
        ("Adrenaline", ["Adrenaline"]),
    ]

    def _pick_val(d: Dict[str, Any], keys: List[str]) -> float:
        for k in keys:
            if k in d:
                return _safe_float(d.get(k), default=np.nan)
        return np.nan

    stats: List[Tuple[str, float]] = [(label, _pick_val(pentagon, keys)) for label, keys in axis_map]

    def _radar_svg(stats_in: List[Tuple[str, float]], maxv: float = 10.0, size: int = 220) -> str:
        n = len(stats_in)
        if n < 3:
            return ""
        cx = cy = size / 2.0
        r = size * 0.34
        angles = [(-math.pi / 2.0) + i * (2.0 * math.pi / n) for i in range(n)]

        def pt(angle: float, radius: float) -> Tuple[float, float]:
            return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        vals: List[float] = []
        for _, v in stats_in:
            vv = 0.0 if pd.isna(v) else float(v)
            vv = float(_clip(vv, 0.0, maxv))
            vals.append(vv)

        grid_levels = [2, 4, 6, 8, 10]
        grid_polys = []
        for lv in grid_levels:
            rr = r * (lv / maxv)
            pts = [pt(a, rr) for a in angles]
            grid_polys.append(" ".join([f"{x:.1f},{y:.1f}" for x, y in pts]))

        axis_pts = [pt(a, r) for a in angles]
        data_pts = [pt(angles[i], r * (vals[i] / maxv)) for i in range(n)]
        data_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in data_pts])
        label_pts = [pt(a, r + 26) for a in angles]

        parts: List[str] = []
        parts.append(f'<svg class="gc-radar-svg" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">')
        for poly in grid_polys:
            parts.append(f'<polygon points="{poly}" fill="none" stroke="#E5E7EB" stroke-width="1" />')
        for (x, y) in axis_pts:
            parts.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#CBD5E1" stroke-width="1" />')
        parts.append(f'<polygon points="{data_points}" fill="rgba(15,23,42,0.10)" stroke="#0F172A" stroke-width="2" />')
        for (x, y) in data_pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#0F172A" />')
        for i, (lx, ly) in enumerate(label_pts):
            lab = html.escape(str(stats_in[i][0]))
            anchor = "middle"
            if lx < cx - 10:
                anchor = "end"
            elif lx > cx + 10:
                anchor = "start"
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="12" font-weight="700" fill="#334155">{lab}</text>')
        parts.append('</svg>')
        return "".join(parts)

    def _label_from_score(v: float) -> str:
        if pd.isna(v):
            return "Mismatch"
        if v >= match_min:
            return "Match"
        if v >= partial_min:
            return "Partial"
        return "Mismatch"

    def _persona_list(p: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    personas = _persona_list(pack)

    def _persona_sort_key(item: Dict[str, Any]) -> Tuple[float, str]:
        score = _safe_float(item.get("score_10") or item.get("Score10") or item.get("score"), default=np.nan)
        s_key = -float(score) if not pd.isna(score) else 1e9
        name = _safe_text(item.get("name") or item.get("Name") or item.get("Persona") or "").strip()
        return (s_key, name)

    if mode == "teaser" and personas:
        personas = sorted(personas, key=_persona_sort_key)[:4]

    def _persona_label(item: Dict[str, Any]) -> str:
        raw = _safe_text(item.get("label") or item.get("Label") or "").strip()
        if raw == "Partial Match":
            raw = "Partial"
        if raw:
            return raw
        score = _safe_float(item.get("score_10") or item.get("Score10") or item.get("score"), default=np.nan)
        return _label_from_score(score)

    def _persona_score(item: Dict[str, Any]) -> float:
        return _safe_float(item.get("score_10") or item.get("Score10") or item.get("score"), default=np.nan)

    def _persona_name(item: Dict[str, Any]) -> str:
        return _safe_text(item.get("name") or item.get("Name") or item.get("Persona") or "").strip()

    def _color_for_label(label: str, vetoed: bool) -> str:
        if vetoed:
            return "#64748B"
        if label == "Match":
            return "#22C55E"
        if label == "Partial":
            return "#F59E0B"
        return "#9CA3AF"

    def _bar_row(label: str, score: float, color: str, dim: bool) -> str:
        pct = 0.0 if pd.isna(score) else float(max(0.0, min(100.0, (score / 10.0) * 100.0)))
        val_txt = "?" if pd.isna(score) else f"{score:.1f}/10"
        opacity = "0.45" if dim else "1"
        return (
            "<div style='display:flex;gap:10px;align-items:center;margin:6px 0;'>"
            f"<div style='width:150px;font-size:14px;color:#374151;font-weight:700;'>{html.escape(label)}</div>"
            f"<div style='flex:1;height:12px;background:#F3F4F6;border-radius:999px;overflow:hidden;'>"
            f"<div style='height:12px;width:{pct:.0f}%;background:{color};opacity:{opacity};border-radius:999px;'></div>"
            "</div>"
            f"<div style='width:70px;text-align:right;font-size:13px;color:#111827;font-weight:800;'>{html.escape(val_txt)}</div>"
            "</div>"
        )

    def _badge(label: str, vetoed: bool) -> str:
        txt = "Mismatch (Veto)" if vetoed else label
        color = _color_for_label(label, vetoed)
        return f"<span style='padding:2px 8px;border-radius:999px;border:1px solid {color};color:{color};font-size:12px;font-weight:800;'>{html.escape(txt)}</span>"

    def _collect_tags() -> List[str]:
        tags: List[str] = []
        for key in ("reasons", "tags"):
            val = pack.get(key)
            if isinstance(val, list):
                for x in val:
                    if isinstance(x, str) and x.strip():
                        tags.append(x.strip())
        for item in personas:
            for key in ("reasons", "tags", "Reasons", "Tags"):
                val = item.get(key)
                if isinstance(val, list):
                    for x in val:
                        if isinstance(x, str) and x.strip():
                            tags.append(x.strip())
        seen = set()
        out: List[str] = []
        for t in tags:
            if t in seen:
                continue
            out.append(t)
            seen.add(t)
        return out

    # Pentagon
    if mode == "full":
        svg = _radar_svg(stats, maxv=10.0, size=240)
        st.markdown(
            f"""
            <div class="gc-sec">
              <div class="gc-sec-t">PENTAGON (0-10)</div>
              <div class="gc-radar-wrap">
                {svg}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for label, value in stats:
            val = _safe_float(value, default=np.nan)
            bar_html = _bar_row(label, val, "#2563EB", False)
            st.markdown(bar_html, unsafe_allow_html=True)

    # Personas
    if personas:
        st.markdown("**Compatibility**")
        for item in personas:
            name = _persona_name(item)
            if not name:
                continue
            score = _persona_score(item)
            label = _persona_label(item)
            vetoed = bool(item.get("vetoed") or item.get("Vetoed"))
            color = _color_for_label(label, vetoed)
            row = _bar_row(name, score, color, vetoed)
            st.markdown(row, unsafe_allow_html=True)
            st.markdown(_badge(label, vetoed), unsafe_allow_html=True)

    tags = _collect_tags()
    tag_limit = 8 if mode == "full" else 4
    if tags:
        chips = "".join(
            [
                f"<span style='display:inline-block;padding:3px 8px;margin:2px 6px 2px 0;border-radius:999px;background:#111827;color:#FFFFFF;font-size:12px;font-weight:700;'>"
                f"{html.escape(t)}</span>"
                for t in tags[:tag_limit]
            ]
        )
        st.markdown(chips, unsafe_allow_html=True)

    if mode == "full":
        st.caption("Higher scores indicate stronger compatibility with the corresponding style.")
        st.caption("Market conditions remain uncertain; compatibility does not imply timing.")
    else:
        st.caption("Market conditions remain uncertain; compatibility does not imply timing.")

def render_current_status_insight(master_score_total: Any, conviction_score: Any, gate_status: Optional[str] = None) -> None:
    """Current Status Insight — concise interpretation of MasterScore & Conviction (single block).
    Note: This is intentionally short and belongs directly under the two score bars.
    """
    ms = _safe_float(master_score_total, default=np.nan)
    cs = _safe_float(conviction_score, default=np.nan)
    if pd.isna(ms) or pd.isna(cs) or (not math.isfinite(float(ms))) or (not math.isfinite(float(cs))):
        return
    ms = float(ms)
    cs = float(cs)

    def _bucket(v: float) -> str:
        if v < 4.0:
            return "low"
        if v < 6.0:
            return "mid"
        if v < 8.0:
            return "good"
        return "high"

    ms_b = _bucket(ms)
    cs_b = _bucket(cs)

    ms_meaning = {
        "low": ("Chất lượng cơ hội hiện tại kém hấp dẫn.",
                "Thường phản ánh: xu hướng/structure xấu hoặc R:R không đáng để mạo hiểm."),
        "mid": ("Cơ hội ở mức trung tính – có chất liệu nhưng chưa ‘ngon’ để commit mạnh.",
                "Thường phản ánh: trend chưa đủ sạch hoặc điểm vào chưa tối ưu; cần thêm xác nhận."),
        "good": ("Cơ hội khá hấp dẫn và có thể triển khai nếu trade plan rõ.",
                 "Thường phản ánh: cấu trúc/trend ổn và R:R tương đối tốt khi chọn đúng nhịp."),
        "high": ("Cơ hội rất hấp dẫn, thuộc nhóm ‘đáng ưu tiên’ trong watchlist.",
                 "Thường phản ánh: cấu trúc/trend đẹp và R:R/điểm vào đang ở vùng thuận lợi."),
    }[ms_b]

    cs_meaning = {
        "low": ("‘Độ chắc chắn của nhận định’ thấp (tín hiệu còn nhiễu / dễ bị đảo).",
                "Chỉ quan sát; tránh hành động lớn vì xác suất sai cao."),
        "mid": ("‘Độ chắc chắn của nhận định’ trung tính (đủ để theo dõi nghiêm túc).",
                "Có tín hiệu hợp lý nhưng chưa đủ đồng thuận để trade mạnh."),
        "good": ("‘Độ chắc chắn của nhận định’ khá tốt (đồng thuận tăng).",
                 "Có thể triển khai có kỷ luật; ưu tiên plan rõ ràng, tránh FOMO."),
        "high": ("‘Độ chắc chắn của nhận định’ cao (đồng thuận mạnh, ít nhiễu).",
                 "Phù hợp triển khai theo kế hoạch; tập trung quản trị rủi ro thay vì do dự."),
    }[cs_b]

    block = f"""Điểm tổng hợp {ms:.1f}/10
{ms_meaning[0]}
{ms_meaning[1]}

Điểm tin cậy {cs:.1f}/10
{cs_meaning[0]}
{cs_meaning[1]}"""
    st.markdown(block)

def render_market_state(analysis_pack: Dict[str, Any]) -> None:
    """
    Appendix E section (2): Market State / Current Regime.
    Must never crash if market context is missing.
    """
    ap = _ensure_dict(analysis_pack)
    m = ap.get("Market") or {}
    vn = m.get("VNINDEX") or {}
    vn30 = m.get("VN30") or {}

    def _fmt_change(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return UI_PLACEHOLDER
        return f"{v:+.2f}%"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">MARKET STATE (CURRENT REGIME)</div>', unsafe_allow_html=True)

    # If no market pack, show a clean fallback (no error, no stacktrace)
    if not m:
        st.info("Market State: ? (market context not available).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        f"""
        <div class="ms-row"><b>VNINDEX</b>: {_val_or_na(vn.get("Regime"))} | Change: {_fmt_change(vn.get("ChangePct"))}</div>
        <div class="ms-row"><b>VN30</b>: {_val_or_na(vn30.get("Regime"))} | Change: {_fmt_change(vn30.get("ChangePct"))}</div>
        <div class="ms-row"><b>Relative Strength vs VNINDEX</b>: {_val_or_na(m.get("RelativeStrengthVsVNINDEX"))}</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


    def _fmt_change(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return UI_PLACEHOLDER
        return f"{v:+.2f}%"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">MARKET STATE (CURRENT REGIME)</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ms-row"><b>VNINDEX</b>: {_val_or_na(vn.get("Regime"))} | Change: {_fmt_change(vn.get("ChangePct"))}</div>
        <div class="ms-row"><b>VN30</b>: {_val_or_na(vn30.get("Regime"))} | Change: {_fmt_change(vn30.get("ChangePct"))}</div>
        <div class="ms-row"><b>Relative Strength vs VNINDEX</b>: {_val_or_na(m.get("RelativeStrengthVsVNINDEX"))}</div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
