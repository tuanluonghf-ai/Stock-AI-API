from __future__ import annotations

from typing import Any, Dict, List, Tuple
import html
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from inception.ui.ui_constants import UI_PLACEHOLDER
from .utils import _ensure_dict, _safe_text, _safe_float, _clip, _safe_bool, _as_scalar, _val_or_na


def _trade_plan_gate(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Anti-anchoring gate for Trade Plan rendering.
    Returns (status, meta) where status ∈ {"ACTIVE","WATCH","LOCK"}.
    Meta includes score/tier for UI copy.
    """
    ap = _ensure_dict(analysis_pack)
    cp = _ensure_dict(character_pack)
    score = _safe_float(ap.get("Conviction"), default=np.nan)
    tier = (_ensure_dict(cp.get("Conviction"))).get("Tier", None)

    # Default thresholds (v6.4 Appendix E)
    active = (pd.notna(score) and score >= 6.5) or (isinstance(tier, (int, float)) and tier >= 4)
    watch = (pd.notna(score) and 5.5 <= score < 6.5) or (isinstance(tier, (int, float)) and int(tier) == 3)

    if active:
        status = "ACTIVE"
    elif watch:
        status = "WATCH"
    else:
        status = "LOCK"

    return status, {"ConvictionScore": score, "Tier": tier}

def render_trade_plan_conditional(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any], gate_status: str, trade_text: str = "") -> None:
    """
    Appendix E section (3): Trade Plan & R:R (Conditional).
    Uses AnalysisPack.TradePlans (Python-computed). No external text generation.
    Layout-only: reorders how the engine output is displayed.
    """
    ap = _ensure_dict(analysis_pack)

    cp = _ensure_dict(character_pack)

    # Prefer standardized TradePlanPack.v1 when available (Python pre-digest single source of truth)
    tpp = ap.get("TradePlanPack") or cp.get("TradePlanPack") or {}
    if isinstance(tpp, dict) and _safe_text(tpp.get("schema") or "").strip() == "TradePlanPack.v1":
        st.markdown("### C. Trade Plan")

        _pol = _safe_text(tpp.get("policy_hint_line") or "").strip()
        if _pol:
            st.caption(_pol)

        _explain = _safe_text(tpp.get("explain") or "").strip()
        if _explain:
            st.caption(_explain)

        def _sf(x: Any) -> float:
            v = _safe_float(x, default=np.nan)
            try:
                v = float(v)
            except Exception:
                return np.nan
            return v if (not pd.isna(v) and math.isfinite(v)) else np.nan

        def _fmt_px(v: Any) -> str:
            x = _sf(v)
            return UI_PLACEHOLDER if pd.isna(x) else f"{x:.2f}"

        def _fmt_rr(v: Any) -> str:
            x = _sf(v)
            return UI_PLACEHOLDER if pd.isna(x) else f"{x:.2f}"

        def _fmt_rrm(v: Any) -> str:
            x = _sf(v)
            return UI_PLACEHOLDER if pd.isna(x) else f"{x:.1f}"

        def _fmt_zone(z: Any) -> str:
            if not isinstance(z, dict):
                return UI_PLACEHOLDER
            lo = _sf(z.get("Low"))
            hi = _sf(z.get("High"))
            if pd.isna(lo) or pd.isna(hi):
                return UI_PLACEHOLDER
            return f"{lo:.2f} – {hi:.2f}"

        def _fmt_px_1dp(v: Any) -> str:
            x = _sf(v)
            return UI_PLACEHOLDER if pd.isna(x) else f"{x:.1f}"

        def _fmt_zone_1dp(z: Any) -> str:
            if not isinstance(z, dict):
                return UI_PLACEHOLDER
            lo = _sf(z.get("Low"))
            hi = _sf(z.get("High"))
            if pd.isna(lo) or pd.isna(hi):
                return UI_PLACEHOLDER
            return f"{lo:.1f}–{hi:.1f}"

        def _fmt_gates(g: Any) -> str:
            if not isinstance(g, dict):
                return UI_PLACEHOLDER
            # Step 11: include plan completeness gate to prevent "PASS all triggers" masking missing Stop/EntryZone
            keys = ["plan", "trigger", "volume", "rr", "exec", "structure"]
            parts = []
            for k in keys:
                parts.append(f"{k.upper()}={_safe_text(g.get(k) or '?').strip().upper()}")
            return " | ".join(parts)

        def _render_plan(title: str, p: Any) -> None:
            if not isinstance(p, dict) or not p:
                return
            ptype = _safe_text(p.get("type") or "?").strip()
            pstate = _safe_text(p.get("state") or "?").strip().upper()
            st.markdown(f"#### {title} — {ptype} | {pstate}")

            st.caption(_fmt_gates(p.get("gates")))

            # Explain blockers (top 3) — standardized Fail Reasons taxonomy (v15.5)
            _fr = p.get("fail_reasons") if isinstance(p, dict) else None
            if isinstance(_fr, list) and _fr and pstate not in ("ACTIVE",):
                # compact bullets; each item already contains UI-ready labels
                st.markdown("**Blockers / Why not ACTIVE:**")
                for r in _fr[:3]:
                    if not isinstance(r, dict):
                        continue
                    short = _safe_text(r.get("ui_short") or "").strip() or UI_PLACEHOLDER
                    label = _safe_text(r.get("label") or "").strip() or UI_PLACEHOLDER
                    st.markdown(f"- **{short}**: {label}")

            # Step 11: surface PlanCompleteness explanation in the Trade Plan section itself
            _pc = p.get("plan_completeness") or {}
            if isinstance(_pc, dict):
                _pcs = _safe_text(_pc.get("status") or "").strip().upper()
                _pcm = _safe_text(_pc.get("message") or "").strip()
                if _pcs in ("FAIL", "WARN") and _pcm:
                    (st.warning if _pcs == "FAIL" else st.info)(_pcm)

            zone = _fmt_zone(p.get("entry_zone"))
            stop = _fmt_px(p.get("stop"))
            tp1 = _fmt_px(p.get("tp1"))
            tp2 = _fmt_px(p.get("tp2"))
            rr = _fmt_rr(p.get("rr_actual"))
            rrm = _fmt_rrm(p.get("rr_min"))

            st.markdown(f"- Entry zone: **{zone}**")
            st.markdown(f"- Stop: **{stop}** | TP1: **{tp1}** | TP2: **{tp2}** | RR: **{rr}** (min {rrm})")

            # Defensive overlay: surface hard stop / reclaim trigger for underwater holding scenarios
            if ptype.upper() == "DEFENSIVE":
                hs = _fmt_px(p.get("defensive_hard_stop"))
                rz = p.get("defensive_reclaim_zone")
                rc = _fmt_zone_1dp(rz) if isinstance(rz, dict) else _fmt_px_1dp(p.get("defensive_reclaim"))
                if hs != UI_PLACEHOLDER or rc != UI_PLACEHOLDER:
                    st.markdown(f"- Defensive overlay: Hard stop **{hs}** | Reclaim trigger **{rc}**")

            note = _safe_text(p.get("notes_short") or "").strip()
            if note:
                st.caption(note)

            inv = _safe_text(p.get("invalidation") or "").strip()
            if inv:
                st.markdown(f"- {inv}")

            rules = p.get("management_rules") or []
            if isinstance(rules, list) and rules:
                st.markdown("**Management rules:**")
                for r in rules[:5]:
                    rr_ = _safe_text(r).strip()
                    if rr_:
                        st.markdown(f"- {rr_}")

        # Gate lock still blocks action; plan is displayed as blueprint only
        if (gate_status or "").strip().upper() == "LOCK":
            st.warning("Gate đang LOCK → Trade Plan chỉ mang tính blueprint. Decision Layer mới là nơi ra action/size.")

        _render_plan("Primary", tpp.get("plan_primary"))
        _render_plan("Alternative", tpp.get("plan_alt"))

        # HOLDING safety: if materially underwater, always surface a DEFENSIVE de-risk plan.
        # Prevents the UX blind spot where only BUY-oriented plans are shown while PnL is deeply negative.
        try:
            pos = ap.get("PositionStatePack") if isinstance(ap, dict) else {}
            pos = pos if isinstance(pos, dict) else {}
            mode_u = _safe_text(pos.get("mode") or "").strip().upper()
            try:
                pnl_v = float(pos.get("pnl_pct"))
            except Exception:
                pnl_v = None
            need_def = (mode_u == "HOLDING") and (pnl_v is not None) and (pnl_v <= -15.0)
            if need_def:
                pt = _safe_text(((tpp.get("plan_primary") or {}) if isinstance(tpp.get("plan_primary"), dict) else {}).get("type") or "").strip().upper()
                at = _safe_text(((tpp.get("plan_alt") or {}) if isinstance(tpp.get("plan_alt"), dict) else {}).get("type") or "").strip().upper()
                if pt != "DEFENSIVE" and at != "DEFENSIVE":
                    for c in (tpp.get("plans_all") or []):
                        if isinstance(c, dict) and _safe_text(c.get("type") or "").strip().upper() == "DEFENSIVE":
                            _render_plan("Defensive", c)
                            break
        except Exception:
            pass

        return
    plans = list(ap.get("TradePlans") or [])

    # Outer visual container (title already printed by render_appendix_e)
    st.markdown('<div class="gc-sec">', unsafe_allow_html=True)

    # Helper: render Risk/Reward snapshot for the primary plan
    def _render_rr_snapshot() -> None:
        primary = (ap.get("PrimarySetup") or {}) if isinstance(ap, dict) else {}
        if not primary:
            return

        name = _val_or_na(primary.get("Name"))
        risk = primary.get("RiskPct")
        reward = primary.get("RewardPct")
        rr = primary.get("RR")
        conf_tech = primary.get("Confidence (Tech)", primary.get("Probability"))

        def _fmt_pct_local(x: Any) -> str:
            try:
                if x is None or pd.isna(x) or not math.isfinite(float(x)):
                    return "?"
                return f"{float(x):.2f}"
            except Exception:
                return "?"

        def _fmt_rr_local(x: Any) -> str:
            try:
                if x is None or pd.isna(x) or not math.isfinite(float(x)):
                    return "?"
                return f"{float(x):.2f}"
            except Exception:
                return "?"

        st.markdown(f"#### Risk / Reward Snapshot (Primary Plan: {name})")
        st.markdown(
            f"""
            <div class="incept-metrics">
              <div class="incept-metric"><div class="k">Risk%:</div><div class="v">{_fmt_pct_local(risk)}</div></div>
              <div class="incept-metric"><div class="k">Reward%:</div><div class="v">{_fmt_pct_local(reward)}</div></div>
              <div class="incept-metric"><div class="k">RR:</div><div class="v">{_fmt_rr_local(rr)}</div></div>
              <div class="incept-metric"><div class="k">Confidence (Tech):</div><div class="v">{_val_or_na(conf_tech)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If gate is locked, show defensive posture + still expose snapshot
    if gate_status == "LOCK":
        st.info("Trade Plan đang bị khóa (chống FOMO). Ưu tiên bảo toàn vốn và chờ tín hiệu xác nhận.")
        st.markdown(
            """
            <div class="tp-lock">
              <div><b>Tư thế:</b> CHỜ / PHÒNG THỦ</div>
              <div style="margin-top:6px;"><b>Checklist kích hoạt:</b></div>
              <ul>
                <li>Conviction tăng lên ngưỡng kích hoạt</li>
                <li>Giá lấy lại vùng MA quan trọng / cấu trúc ổn định trở lại</li>
                <li>Khối lượng xác nhận (không có dấu hiệu kiệt sức), động lượng cải thiện</li>
                <li>Cấu trúc tuần còn nguyên vẹn (không breakdown cấu trúc)</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        _render_rr_snapshot()
        return

    if not plans:
        st.warning("No Trade Plan available from engine.")
        st.markdown("</div>", unsafe_allow_html=True)
        _render_rr_snapshot()
        return

    # Sort: prefer higher probability label, then higher RR
    def _prob_rank(p: Any) -> int:
        s = str(p or "").lower()
        if "high" in s:
            return 3
        if "med" in s:
            return 2
        if "low" in s:
            return 1
        return 0

    def _conf_val(p: Dict[str, Any]) -> Any:
        return p.get("Confidence (Tech)", p.get("Probability"))

    plans_sorted = sorted(
        plans,
        key=lambda x: (
            -_prob_rank(_conf_val(x)),
            -_safe_float(x.get("RR"), default=-1e9),
        ),
    )

    def _to_html_lines(txt: str) -> str:
        """Escape + preserve line breaks for HTML blocks."""
        txt = (txt or "").strip()
        if not txt:
            return ""
        return "<br>".join(html.escape(txt).splitlines())

    def _split_trade_text_by_plan(txt: str, plan_names: List[str]) -> Dict[str, str]:
        """Best-effort split legacy C-section narrative into per-plan notes.

        Expected Vietnamese/EN anchors:
          - "Kế hoạch giao dịch <PlanName> ..."
          - "Trade plan <PlanName> ..."
        If no anchors are found, return empty dict and caller will attach the whole text to the first plan.
        """
        txt = (txt or "").strip()
        names = [n for n in (plan_names or []) if n and str(n).strip() and str(n).strip().upper() != "?"]
        if (not txt) or (not names):
            return {}

        # Build a single regex that matches any plan name following the anchor phrase.
        alts = "|".join(sorted((re.escape(str(n)) for n in set(names)), key=len, reverse=True))
        pat = re.compile(r"(?i)(kế\s*hoạch\s*giao\s*dịch|trade\s*plan)\s+(" + alts + r")\b")

        matches = list(pat.finditer(txt))
        if not matches:
            return {}

        out: Dict[str, str] = {}
        for i, m in enumerate(matches):
            plan = m.group(2)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
            seg = txt[start:end].strip()
            if not seg:
                continue
            # Keep the segment as-is (it often reads well in Vietnamese), but preserve breaks safely.
            out[str(plan)] = _to_html_lines(seg)
        return out

    # 3.1 Setup Overview
    # NOTE: Layout-only changes (no logic changes).
    status_vi = "ĐANG KÍCH HOẠT" if gate_status == "ACTIVE" else "CHỈ THEO DÕI"
    desc_vi = (
        "Được phép thực thi (có điều kiện). Áp dụng size động và tuân thủ tuyệt đối vùng dừng lỗ."
        if gate_status == "ACTIVE"
        else "Kế hoạch chỉ mang tính tham khảo. Không FOMO, không ép lệnh."
    )

    st.markdown(
        f"""
        <div class="tp-sec-h">
          <div class="tp-sec-title">
            <span>Các yếu tố kỹ thuật kích hoạt TRADE PLAN</span>
            <span class="tp-badge {'active' if gate_status == 'ACTIVE' else 'watch'}">{status_vi}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='tp-note'><b>{status_vi}:</b> {html.escape(desc_vi)}</div>", unsafe_allow_html=True)

    # WATCH: still show 2 plans, but the 2nd plan is a dimmed "tham khảo" reference.
    show_n = 2 if gate_status in ("ACTIVE", "WATCH") else 1
    plans_to_show = plans_sorted[: min(show_n, len(plans_sorted))]
    plan_names = [str(_val_or_na(p.get("Name"))) for p in plans_to_show]
    expl_map = _split_trade_text_by_plan(trade_text, plan_names)

    # If we couldn't split, attach the entire narrative to the first plan only.
    fallback_expl = ""
    if trade_text and not expl_map:
        fallback_expl = _to_html_lines(trade_text)

    for idx, p in enumerate(plans_to_show):
        name = _val_or_na(p.get("Name"))
        entry = _safe_float(p.get("Entry"), default=np.nan)
        stop = _safe_float(p.get("Stop"), default=np.nan)
        tp = _safe_float(p.get("TP"), default=np.nan)
        rr = _safe_float(p.get("RR"), default=np.nan)
        conf_tech = _val_or_na(p.get("Confidence (Tech)", p.get("Probability")))
        status = _val_or_na(p.get("Status"))

        is_ref = (gate_status == "WATCH" and idx == 1)
        card_cls = "tp-card dim" if is_ref else "tp-card"
        ref_badge = '<span class="tp-ref">tham khảo</span>' if is_ref else ""

        rr_label = (
            "Attractive"
            if (pd.notna(rr) and rr >= 2.0)
            else ("Acceptable" if (pd.notna(rr) and rr >= 1.3) else "Thin")
        )

        if pd.notna(rr):
            rr_disp = f"{float(rr):.1f}"
        else:
            rr_disp = "?"

        def _fmt_px(x: Any) -> str:
            v = _safe_float(x, default=np.nan)
            return f"{v:.2f}" if pd.notna(v) else "?"

        def _fmt_px_with_delta(x: Any, entry_x: Any) -> str:
            v = _safe_float(x, default=np.nan)
            e = _safe_float(entry_x, default=np.nan)
            if pd.notna(v) and pd.notna(e) and e != 0:
                d = (v - e) / e * 100.0
                return f"{v:.2f} ({d:+.1f}%)"
            return _fmt_px(v)

        st.markdown(
            f"""
            <div class="{card_cls}">
              <div class="tp-title"><b>{html.escape(str(name))}</b> <span class="tp-status">[{html.escape(str(status))}]</span> {ref_badge}</div>
              <div class="tp-meta">Confidence (Tech): <b>{html.escape(str(conf_tech))}</b> | R:R: <b>{html.escape(str(rr_disp))}</b> ({rr_label})</div>
              <div class="tp-levels">
                <span>Entry: <b>{_fmt_px(entry)}</b></span>
                <span>Stop: <b>{_fmt_px_with_delta(stop, entry)}</b></span>
                <span>TP: <b>{_fmt_px_with_delta(tp, entry)}</b></span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional per-plan narrative (placed immediately under its plan card)
        key = str(name)
        expl_html = expl_map.get(key, "")
        if (not expl_html) and (fallback_expl) and idx == 0:
            expl_html = fallback_expl
        if expl_html:
            st.markdown(
                f"""
                <div class="tp-expl">{expl_html}</div>
                """,
                unsafe_allow_html=True,
            )

    # Close visual container and then show the numeric snapshot
    st.markdown("</div>", unsafe_allow_html=True)
    _render_rr_snapshot()
