from __future__ import annotations

from typing import Any, Dict, List, Tuple
import html
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from inception.ui.ui_constants import UI_PLACEHOLDER
from .utils import _ensure_dict, _safe_text, _safe_float, _clip, _val_or_na, PLAYSTYLE_TAG_TRANSLATIONS


def render_character_decision(character_pack: Dict[str, Any]) -> None:
    """
    Render only the 'Decision' part of Character Card (for Appendix E / anti-anchoring).
    Includes: Conviction + Weaknesses + Playstyle Tags.
    """
    cp = _ensure_dict(character_pack)
    conv = _ensure_dict(cp.get("Conviction"))
    flags = cp.get("Flags") or []
    tags = cp.get("ActionTags") or []

    tier = conv.get("Tier", "?")
    pts = conv.get("Points", np.nan)
    guide = conv.get("SizeGuidance", "")

    st.markdown(
        f"""
        <div class="gc-sec">
          <div class="gc-sec-t">CONVICTION</div>
          <div class="gc-conv">
            <div class="gc-conv-tier">Tier: <b>{tier}</b> / 7</div>
            <div class="gc-conv-pts">Points: <b>{_val_or_na(pts)}</b></div>
            <div class="gc-conv-guide">{html.escape(str(guide))}</div>
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
            code_ = f.get("code", "")
            st.markdown(
                f"""<div class="gc-flag"><span class="gc-sev">S{sev}</span><span class="gc-code">{html.escape(str(code_))}</span><span class="gc-note">{html.escape(str(note))}</span></div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if tags:
        st.markdown('<div class="gc-sec"><div class="gc-sec-t">PLAYSTYLE TAGS</div>', unsafe_allow_html=True)
        rendered_tags: List[str] = []
        for t in tags[:8]:
            raw = str(t)
            human = PLAYSTYLE_TAG_TRANSLATIONS.get(raw, raw)
            rendered_tags.append(f"<span class='gc-tag'>{html.escape(str(human))}</span>")
        st.markdown(
            "<div class='gc-tags'>" + "".join(rendered_tags) + "</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

def render_decision_layer_switch(character_pack: Dict[str, Any], analysis_pack: Dict[str, Any], gate_status: str, exec_mode_text: str, preferred_plan: str) -> None:
    """Renderer-only: central Decision Layer switch (no scoring / rule changes).

    v7.1 renderer improvements:
      - Humanize PROTECH labels (replace underscores, add spacing)
      - Convert long BiasCode into short pills (avoid unreadable mega-string)
      - No red accents (orange/neutral only)
    """
    cp = _ensure_dict(character_pack)
    ap = _ensure_dict(analysis_pack)

    # ------------------------------------------------------------
    # DecisionPack.v1 (Portfolio-ready; Long-only) — render if available
    # ------------------------------------------------------------
    dp = ap.get("DecisionPack") or {}
    dp = dp if isinstance(dp, dict) else {}
    if _safe_text(dp.get("schema") or "").strip() == "DecisionPack.v1":
        act = _safe_text(dp.get("action") or "?").strip().upper()
        urg = _safe_text(dp.get("urgency") or "MED").strip().upper()
        rat = _safe_text(dp.get("rationale") or "").strip()
        cons = dp.get("constraints") or []
        cons = cons if isinstance(cons, list) else []

        sev_cls = "low"
        if urg == "HIGH":
            sev_cls = "high"
        elif urg == "MED":
            sev_cls = "med"

        pnl_txt = ""
        try:
            pnl = dp.get("pnl_pct", None)
            if isinstance(pnl, (int, float)) and pnl == pnl:
                pnl_txt = f"{pnl:+.1f}%"
        except Exception:
            pnl_txt = ""

        st.markdown(
            f"""
            <div class="dl-card {sev_cls}">
              <div class="dl-k">Action</div>
              <div class="dl-v" style="font-size:20px;font-weight:800;">{html.escape(act)} <span class="dl-sub" style="font-size:14px;">({html.escape(urg)})</span></div>
              {f'<div class="dl-sub">PnL: {html.escape(pnl_txt)}</div>' if pnl_txt else ''}
              {f'<div class="dl-sub" style="margin-top:6px;">{html.escape(rat)}</div>' if rat else ''}
              {(''.join([f'<div class="dl-flag" style="margin-top:6px;"><span class="dl-code">{html.escape(str(x))}</span></div>' for x in cons[:2]])) if cons else ''}
            </div>
            """,
            unsafe_allow_html=True
        )


    # ------------------------------------------------------------
    # PositionManagerPack.v1 (Portfolio-ready; Long-only) — render if available
    # ------------------------------------------------------------
    pmp = ap.get("PositionManagerPack") or {}
    pmp = pmp if isinstance(pmp, dict) else {}
    if _safe_text(pmp.get("schema") or "").strip() == "PositionManagerPack.v1":
        pm_act = _safe_text(pmp.get("action") or "?").strip().upper()
        pm_guid = _safe_text(pmp.get("guidance") or "").strip()
        pm_trim = pmp.get("trim_pct_of_position", None)
        pm_stop = _safe_float(pmp.get("stop_suggest"), default=np.nan)
        pm_cap = pmp.get("size_cap_pct_nav", None)
        pm_pos = pmp.get("position_size_pct_nav", None)

        extra_lines = []
        if isinstance(pm_cap, (int, float)) and pm_cap == pm_cap:
            extra_lines.append(f"Policy cap: {pm_cap:.0f}% NAV")
        if isinstance(pm_pos, (int, float)) and pm_pos == pm_pos:
            extra_lines.append(f"Current size: {pm_pos:.0f}% NAV")
        if isinstance(pm_trim, (int, float)) and pm_trim == pm_trim:
            extra_lines.append(f"Trim: ~{pm_trim*100:.0f}% position")
        if pd.notna(pm_stop):
            extra_lines.append(f"Stop (suggest): {pm_stop:.2f}")

        st.markdown(
            f"""
            <div class="dl-card med">
              <div class="dl-k">Position Manager</div>
              <div class="dl-v" style="font-size:18px;font-weight:800;">{html.escape(pm_act)}</div>
              {f'<div class="dl-sub" style="margin-top:6px;">{html.escape(pm_guid)}</div>' if pm_guid else ''}
              {f'<div class="dl-sub" style="margin-top:6px;">{html.escape(" | ".join(extra_lines))}</div>' if extra_lines else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

    def _humanize(s: Any) -> str:
        s0 = _safe_text(s).strip()
        if not s0:
            return ""
        # Replace common separators
        s0 = s0.replace("__", " | ")
        s0 = s0.replace("_", " ")
        s0 = re.sub(r"\s+", " ", s0).strip()
        return s0

    def _bias_code_to_pills(code: str) -> List[str]:
        code = _safe_text(code).strip()
        if not code or code.upper() == "?":
            return []
        parts = re.split(r"__+", code)
        out: List[str] = []
        for p in parts:
            p = _humanize(p)
            if not p:
                continue
            # Compact common verbose fragments (display-only)
            p = p.replace("Hist=", "Hist: ").replace("Zero=", "Zero: ").replace("Cross=", "Cross: ").replace("Align=", "Align: ")
            p = p.replace("MACD Above Signal", "MACD > Signal")
            p = p.replace("Expanding Positive", "Expanding +")
            p = p.replace("CrossUp", "Cross Up")
            p = re.sub(r"\s+", " ", p).strip()
            out.append(p)
        # Keep it readable: show top N pills only
        return out[:8]

    conv = _ensure_dict(cp.get("Conviction"))
    tier = conv.get("Tier", "?")
    pts = _safe_float(conv.get("Points"), default=np.nan)
    guide = _safe_text(conv.get("SizeGuidance") or "").strip()

    # --- Effective execution gating vs Primary Setup status (renderer-only) ---
    primary_status = ""
    try:
        ps = ap.get("PrimarySetup") or {}
        if isinstance(ps, dict):
            primary_status = _safe_text(ps.get("Status")).strip()
        if not primary_status:
            tps = ap.get("TradePlans") or []
            if isinstance(tps, list):
                for p in tps:
                    if _safe_text(p.get("Name")).strip() == preferred_plan:
                        primary_status = _safe_text(p.get("Status")).strip()
                        break
    except Exception:
        primary_status = ""

    pstat = primary_status.upper().strip()

    effective_exec_mode_text = exec_mode_text
    effective_preferred_plan = preferred_plan
    effective_guide = guide

    # If overall mode is ACTIVE but the primary plan is still WATCH, downgrade wording to avoid "full size now" confusion.
    if _safe_text(gate_status).upper().strip() == "ACTIVE":
        if pstat in ("WATCH", "PENDING", "WAIT", "WAITING", "TRACK"):
            effective_exec_mode_text = "ACTIVE (WAIT ENTRY) – được phép triển khai khi plan kích hoạt/đạt điều kiện vào lệnh."
            if effective_guide:
                # Soften aggressive sizing language: full size only when plan becomes Active/triggered
                effective_guide = effective_guide.replace("FULL SIZE + CÓ THỂ ADD", "FULL SIZE (KHI PLAN ACTIVE) + CÓ THỂ ADD (SAU FOLLOW-THROUGH)")
                if "FULL SIZE" in effective_guide and "(KHI PLAN ACTIVE)" not in effective_guide:
                    effective_guide = effective_guide.replace("FULL SIZE", "FULL SIZE (KHI PLAN ACTIVE)")
            else:
                effective_guide = "EDGE MẠNH — FULL SIZE (KHI PLAN ACTIVE) + CÓ THỂ ADD (SAU FOLLOW-THROUGH)"
        elif pstat in ("INVALID", "DISABLED", "?", "NA"):
            effective_exec_mode_text = "WATCH ONLY – primary plan hiện không hợp lệ; ưu tiên quan sát."
            effective_guide = "NO TRADE / WAIT RESET"


    # --- Sanity-check layer (renderer only) ---
    # Prevent contradictory messaging between plan status and sizing guidance.
    try:
        gs2 = _safe_text(gate_status).upper().strip()
        if gs2 != "ACTIVE":
            # In non-ACTIVE modes, never suggest aggressive sizing.
            if effective_guide:
                if "FULL SIZE" in effective_guide:
                    effective_guide = "WATCH MODE — chưa vào lệnh; chỉ chuẩn bị kế hoạch và chờ điều kiện kích hoạt."
                elif "ADD" in effective_guide or "Pyramid" in effective_guide:
                    effective_guide = "WATCH MODE — không gia tăng; chỉ theo dõi theo Trade Plan."
        else:
            # ACTIVE but entry not triggered -> gate sizing to 'when plan active'
            if pstat in ("WATCH", "PENDING", "WAIT", "WAITING", "TRACK"):
                if effective_guide and "FULL SIZE" in effective_guide and "(KHI PLAN ACTIVE)" not in effective_guide:
                    effective_guide = effective_guide.replace("FULL SIZE", "FULL SIZE (KHI PLAN ACTIVE)")
                # Ensure add/pyramid is conditional
                if effective_guide and "CÓ THỂ ADD" in effective_guide and "FOLLOW-THROUGH" not in effective_guide:
                    effective_guide = effective_guide.replace("CÓ THỂ ADD", "CÓ THỂ ADD (SAU FOLLOW-THROUGH)")
            # If primary is invalid, force conservative messaging
            if pstat in ("INVALID", "DISABLED", "?", "NA"):
                effective_exec_mode_text = "WATCH ONLY – primary plan hiện không hợp lệ; ưu tiên quan sát."
                effective_guide = "NO TRADE / WAIT RESET"
    except Exception:
        pass

    def _conv_cls_from_tier(t: object) -> str:
        try:
            ti = int(t)
        except Exception:
            return "conv-unknown"
        if ti <= 1:
            return "conv-noedge"
        if ti == 2:
            return "conv-weak"
        if ti == 3:
            return "conv-tradeable"
        if ti in (4, 5):
            return "conv-strong"
        if ti == 6:
            return "conv-high"
        return "conv-god"

    guide_upper = effective_guide.upper() if effective_guide else ""
    guide_cls = _conv_cls_from_tier(tier)
    guide_html = f"<span class='conv-tag {guide_cls}'>{html.escape(guide_upper)}</span>" if guide_upper else " "

    # Final Bias comes from ProTech.Bias (fact-only layer)
    bias = ((ap.get("ProTech") or {}).get("Bias") or {}) if isinstance(ap, dict) else {}
    alignment_raw = _safe_text(bias.get("Alignment") or "?").strip()
    bias_code_raw = _safe_text(bias.get("BiasCode") or "").strip()

    alignment = _humanize(alignment_raw) or "?"
    bias_pills = _bias_code_to_pills(bias_code_raw)

    # Level mapping for background (layout only)
    lvl = "low"
    if pd.notna(pts):
        if pts >= 5.0:
            lvl = "high"
        elif pts >= 3.0:
            lvl = "med"

    flags = cp.get("Flags") or []
    tags = cp.get("ActionTags") or []

    # Translate tags (EN → EN/VI) where possible
    tags_vi: List[str] = []
    for t in tags:
        t0 = str(t)
        tags_vi.append(PLAYSTYLE_TAG_TRANSLATIONS.get(t0, t0))

    st.markdown('<div class="dl-wrap">', unsafe_allow_html=True)

    
    # Hero card: Conviction + Size Guidance (FINAL BIAS hidden by UI policy)
    pts_disp = f"{pts:.1f}" if pd.notna(pts) else "?"
    st.markdown(
        f"""
        <div class="dl-grid">
          <div class="dl-card {lvl}">
            <div class="dl-k">CONVICTION SCORE</div>
            <div class="dl-v">Tier {html.escape(str(tier))}/7  •  {html.escape(pts_disp)} pts</div>
            <div class="dl-sub"><b>Execution Mode:</b> {html.escape(effective_exec_mode_text)}<br><b>Preferred Plan:</b> {html.escape(effective_preferred_plan)}</div>
            <div class="dl-sub">{guide_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Weaknesses / Flags (short list)
    if isinstance(flags, list) and flags:
        st.markdown('<div class="dl-sec"><div class="dl-sec-t">YẾU ĐIỂM VÀ RỦI RO CHÍNH CỦA TRADE PLAN</div>', unsafe_allow_html=True)
        for f in flags[:6]:
            if not isinstance(f, dict):
                continue
            try:
                sev = int(f.get("severity", 1))
            except Exception:
                sev = 1
            note = _safe_text(f.get("note", ""))
            code = _safe_text(f.get("code", ""))
            st.markdown(
                f"""<div class="dl-flag"><span class="dl-sev">S{sev}</span><span class="dl-code">{html.escape(code)}</span><span class="dl-note">{html.escape(note)}</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Playstyle tags (bilingual pills)
    if tags_vi:
        st.markdown('<div class="dl-sec"><div class="dl-sec-t">ĐỀ XUẤT XU HƯỚNG HÀNH ĐỘNG</div>', unsafe_allow_html=True)
        def _play_hint(tag: str) -> str:
            tl = (tag or '').lower()
            if 'breakout' in tl:
                return 'Ưu tiên chờ phiên xác nhận và follow-through; tránh vào sớm khi chưa có lực phá vỡ rõ ràng.'
            if 'pullback' in tl:
                return 'Ưu tiên canh hồi về vùng hỗ trợ/MA/Fib để tối ưu điểm vào; tránh đuổi giá khi chưa hồi.'
            if ('longstructure' in tl and 'shorttactical' in tl) or ('long structure' in tl and 'short tactical' in tl):
                return 'Khung dài hạn quyết định bias; tác chiến ngắn hạn chỉ để tối ưu entry/exit theo đúng cấu trúc.'
            if 'trend' in tl:
                return 'Ưu tiên đi theo xu hướng; chỉ vào khi MA/cấu trúc ủng hộ và không vi phạm stop.'
            return 'Tag này gợi ý cách hành động phù hợp bối cảnh hiện tại; làm theo để đồng bộ Trade Plan và giảm sai nhịp.'

        items = []
        for t in tags_vi[:10]:
            t1 = str(t)
            hint = _play_hint(t1)
            items.append(
                f"<div class='dl-tagitem'><span class='dl-pill'>{html.escape(t1)}</span><div class='dl-taghint'>{html.escape(hint)}</div></div>"
            )
        st.markdown("<div class='dl-tags'>" + "".join(items) + "</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return
