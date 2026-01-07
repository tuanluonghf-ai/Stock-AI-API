"""Streamlit UI renderers for INCEPTION.

All Streamlit presentation logic lives here to keep the entrypoint
app (app_INCEPTION_*.py) small and wiring-only.

Note: This module is intentionally UI-only. It may import Streamlit
and generate HTML/CSS, but it must not depend on the app entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import re
import html

import numpy as np
import pandas as pd
import streamlit as st

from inception.core.helpers import _safe_text, _safe_float, _clip, _safe_bool, _as_scalar
from inception.core.dashboard_pack import compute_dashboard_summary_pack_v1

def _val_or_na(v: Any) -> str:
    """UI-friendly stringify with N/A fallback."""
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and pd.isna(v):
            return "N/A"
        s = str(v).strip()
        return s if s else "N/A"
    except Exception:
        return "N/A"

def _call_openai(prompt: str, temperature: float = 0.5) -> str:
    """Best-effort OpenAI call used only for small UI blurbs.

    If the SDK/key is unavailable, this returns an empty string and the
    UI falls back to deterministic templates.
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là INCEPTION AI, chuyên gia phân tích đầu tư."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

PLAYSTYLE_TAG_TRANSLATIONS = {
    "Pullback-buy zone (confluence)": "Vùng pullback mua (hội tụ)",
    "Breakout attempt (needs follow-through)": "Nỗ lực breakout (cần follow-through)",
    "Wait for volume confirmation": "Chờ xác nhận dòng tiền",
    "Tight risk control near resistance": "Siết rủi ro gần kháng cự",
    "Use LongStructure_ShortTactical rule": "Ưu tiên cấu trúc dài hạn; tactical dùng để vào lệnh",
}

CLASS_TEMPLATES: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Nhóm này có xu hướng dài hạn tương đối rõ và bền. Giá thường bám cấu trúc trend (đỉnh–đáy nâng dần) và tôn trọng các vùng hỗ trợ động, nên hành vi ít bị nhiễu so với nhóm biến động cao.",
        "Nhà đầu tư phù hợp là người đánh theo xu hướng, ưu tiên mua ở nhịp điều chỉnh/pullback thay vì mua đuổi. Có thể nắm giữ trung hạn khi cấu trúc còn nguyên vẹn và chỉ gia tăng khi có xác nhận tiếp diễn.",
        "Điểm lưu ý là tránh phá kỷ luật khi thị trường nhiễu: chỉ giữ vị thế khi trend còn hợp lệ và luôn có mức dừng lỗ theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Nhóm này thiên về động lượng: khi vào pha tăng/giảm, giá thường chạy nhanh theo hướng chính và khó vào lại nếu chậm nhịp. Breakout/continuation có xác suất tốt hơn so với việc bắt đáy.",
        "Phù hợp với trader chủ động, theo dõi sát và chấp nhận ra/vào theo nhịp. Ưu tiên vào khi có follow-through và quản trị vị thế bằng trailing theo cấu trúc.",
        "Cần cảnh giác khi động lượng suy yếu (thiếu follow-through, đà thu hẹp): ưu tiên chốt từng phần và không nới kỷ luật stop.",
    ],
    "Aggressive Trend": [
        "Nhóm này có xu hướng nhưng tải rủi ro cao: biến động lớn, tail risk/gap có thể xuất hiện khi dòng tiền đổi trạng thái. Lợi nhuận tiềm năng cao nhưng sai nhịp sẽ trả giá nhanh.",
        "Phù hợp với trader chịu rung lắc, kỷ luật stop và quản trị size nghiêm ngặt. Chỉ nên tham gia khi plan rõ ràng và điểm vào tối ưu (không FOMO).",
        "Ưu tiên chiến thuật hit-and-run/pyramid có điều kiện sau khi đã giảm rủi ro (free-ride). Tránh giữ vị thế quá lớn qua thời điểm nhạy cảm.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Nhóm này vận động trong biên tương đối ổn định và hay quay về vùng cân bằng. Xác suất mua gần hỗ trợ–bán gần kháng cự tốt hơn kỳ vọng chạy trend dài liên tục.",
        "Phù hợp với trader kỷ luật, kiên nhẫn chờ vùng biên; ưu tiên scale-in ở vùng hỗ trợ và chốt dần ở vùng kháng cự.",
        "Rủi ro chính là phá biên: khi đóng cửa ra khỏi hộp giá, cần cắt/giảm nhanh để tránh bị kéo sang một regime mới.",
    ],
    "Volatile Range": [
        "Nhóm này vẫn có tính chất range/mean-reversion nhưng nhiễu và rung lắc mạnh hơn; dễ false-break và quét stop nếu vào giữa biên.",
        "Phù hợp với trader chọn lọc: chỉ vào khi có xác nhận đảo chiều tại mép biên và chấp nhận tỷ trọng nhỏ hơn bình thường.",
        "Ưu tiên chốt nhanh, không bình quân giá khi cấu trúc yếu và luôn quản trị rủi ro theo volatility.",
    ],
    "Mixed / Choppy Trader": [
        "Nhóm này thiếu xu hướng rõ ràng hoặc hay đổi tính theo thời gian; tín hiệu dễ nhiễu và whipsaw cao. Edge thường chỉ xuất hiện theo từng nhịp ngắn.",
        "Phù hợp với giao dịch chiến thuật: chỉ trade khi setup đạt chất lượng cao và có xác nhận; phần lớn thời gian nên đứng ngoài.",
        "Kỷ luật vào/ra và size nhỏ là bắt buộc; tránh giữ vị thế dài khi cấu trúc không rõ.",
    ],
    "Event / Gap-Prone": [
        "Nhóm này có rủi ro sự kiện/gap cao: giá có thể nhảy mạnh ngoài dự kiến, khiến stop dễ bị trượt. Dù có thể mang lại lợi nhuận lớn, execution risk cũng cao.",
        "Phù hợp với trader chấp nhận rủi ro tail, ưu tiên giao dịch ngắn hạn và giảm nắm giữ qua thời điểm nhạy cảm.",
        "Bắt buộc dùng size nhỏ, stop theo cấu trúc + buffer và chỉ tham gia khi reward đủ lớn để bù rủi ro.",
    ],
    "Illiquid / Noisy": [
        "Nhóm này có rủi ro thực thi: thanh khoản thiếu ổn định, spread/độ trượt có thể làm sai lệch R:R thực tế. Tín hiệu kỹ thuật thường kém tin cậy hơn do nhiễu.",
        "Phù hợp với nhà đầu tư rất kỷ luật và chấp nhận giải ngân nhỏ; ưu tiên lệnh giới hạn và tránh đuổi giá.",
        "Nếu không đạt điều kiện thanh khoản tối thiểu, nên coi đây là nhóm 'NO TRADE' dù kịch bản nhìn đẹp trên giấy.",
    ],
}

CLASS_TEMPLATES_DASHBOARD: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Đặc tính: Trend bền, hành vi giá tương đối sạch, phù hợp nắm giữ theo cấu trúc.",
        "Phù hợp: Trend-follow / tích lũy theo nhịp điều chỉnh, ưu tiên kỷ luật hơn tốc độ.",
        "Chiến thuật: Pullback & trend continuation, dời stop theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Đặc tính: Động lượng mạnh, chạy nhanh khi có lực, khó vào lại nếu chậm nhịp.",
        "Phù hợp: Trader chủ động, theo dõi sát, chốt lời từng phần theo đà.",
        "Chiến thuật: Breakout/continuation có xác nhận, trailing theo nhịp.",
    ],
    "Aggressive Trend": [
        "Đặc tính: Trend có nhưng rủi ro cao (biến động/tail/gap), sai nhịp trả giá nhanh.",
        "Phù hợp: Trader chịu rung lắc, size nhỏ hơn chuẩn, kỷ luật stop tuyệt đối.",
        "Chiến thuật: Hit & Run / pyramid có điều kiện sau khi giảm rủi ro.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Đặc tính: Sideway ổn định, hay quay về vùng cân bằng, biên hỗ trợ/kháng cự rõ.",
        "Phù hợp: Trader kiên nhẫn, đánh theo biên, ưu tiên xác suất hơn kỳ vọng lớn.",
        "Chiến thuật: Buy near support – sell near resistance, scale-in/out theo vùng.",
    ],
    "Volatile Range": [
        "Đặc tính: Range nhưng nhiễu, dễ false-break và quét stop; rung lắc mạnh.",
        "Phù hợp: Trader chọn lọc, chỉ vào ở mép biên và giảm tỷ trọng.",
        "Chiến thuật: Vào khi có xác nhận đảo chiều, chốt nhanh, quản trị theo vol.",
    ],
    "Mixed / Choppy Trader": [
        "Đặc tính: Không rõ trend/range, whipsaw cao; edge chỉ xuất hiện theo nhịp ngắn.",
        "Phù hợp: Tactical trader, chấp nhận đứng ngoài phần lớn thời gian.",
        "Chiến thuật: Trade khi setup thật rõ + có xác nhận; size nhỏ.",
    ],
    "Event / Gap-Prone": [
        "Đặc tính: Rủi ro sự kiện/gap cao, stop dễ trượt; cần reward lớn để bù rủi ro.",
        "Phù hợp: Trader mạo hiểm nhưng kỷ luật, tránh giữ qua thời điểm nhạy cảm.",
        "Chiến thuật: Size nhỏ, stop + buffer, chỉ tham gia khi RR đủ dày.",
    ],
    "Illiquid / Noisy": [
        "Đặc tính: Rủi ro thực thi (thanh khoản kém/không ổn định), tín hiệu dễ nhiễu.",
        "Phù hợp: Chỉ dành cho người rất kỷ luật, chấp nhận giải ngân nhỏ.",
        "Chiến thuật: Lệnh giới hạn, tránh đuổi giá; không đạt liquidity gate thì NO TRADE.",
    ],
}

def _character_blurb_fallback(ticker: str, cclass: str) -> str:
    """
    Deterministic fallback when GPT narrative is disabled.
    Must contain no numbers. Keep short, stable, and aligned with 2-tier DNA taxonomy.
    """
    name = (ticker or "").upper().strip() or "Cổ phiếu"
    cc = (cclass or "N/A").strip()

    # Use dashboard template if available
    lines = CLASS_TEMPLATES_DASHBOARD.get(cc) or []
    if lines:
        # Convert 3-line dashboard template into a compact paragraph
        return f"{name}: {lines[0]} {lines[1]} {lines[2]}"

    # Generic fallback (class unknown)
    return (f"{name} đang được gắn nhãn DNA '{cc}'. Hãy ưu tiên bám cấu trúc giá, "
            f"chỉ triển khai khi trade plan có điều kiện rõ ràng và tuân thủ kỷ luật quản trị rủi ro.")

def get_character_blurb(ticker: str, cclass: str) -> str:
    # GPT paragraph: 100–200 words, no numbers
    cache_key = f"_gc_blurb::{(ticker or '').upper().strip()}::{(cclass or '').strip()}"
    if cache_key in st.session_state:
        return st.session_state.get(cache_key) or ""
    base = _character_blurb_fallback(ticker, cclass)
    try:
        prompt = f"""Bạn là chuyên gia tài chính. Hãy viết một đoạn ngắn tiếng Việt (khoảng 100–200 từ),
văn phong chuyên nghiệp, dễ hiểu. Tuyệt đối KHÔNG nhắc bất kỳ con số nào (không số điểm, không phần trăm, không mốc giá,
không số phiên, không ký hiệu số). Không liệt kê chỉ báo/thuật ngữ theo dạng báo cáo. Hãy mô tả:
- Cổ phiếu {ticker.upper().strip()} thuộc nhóm (class) {cclass}.
- Bản chất hành vi giá thường gặp của nhóm này.
- Phù hợp với kiểu trader/trường phái nào và không phù hợp với kiểu nào.
- Nêu một ví dụ ngắn về hành vi thường gặp (ví dụ: dao động trong biên, bật ở hỗ trợ, thất bại khi vượt cản…).
Kết thúc bằng một câu định hướng hành động theo phong cách quản trị rủi ro.
"""
        txt = _call_openai(prompt, temperature=0.5)
        txt = (txt or "").strip()
        # Safety: remove digits if model violates rule
        txt = re.sub(r"\d", "", txt)
        if len(txt) < 40:
            txt = base
    except Exception:
        txt = base
    st.session_state[cache_key] = txt
    return txt

def get_class_policy_hint_line(final_class: str) -> str:
    cn = _safe_text(final_class).strip()
    p = CLASS_POLICY_HINTS.get(cn)
    if not p:
        return ""
    rr = p.get("rr_min")
    size_cap = _safe_text(p.get("size_cap")).strip()
    overnight = _safe_text(p.get("overnight")).strip()
    rr_txt = f"RR≥{float(rr):.1f}" if isinstance(rr, (int, float)) else ""
    size_txt = f"Size≤{size_cap}" if size_cap else ""
    on_txt = f"Overnight: {overnight}" if overnight else ""
    parts = [x for x in (rr_txt, size_txt, on_txt) if x]
    return " | ".join(parts)


# ============================================================
# LEVEL LABEL PRETTY PRINTER (UI-ONLY)
# ============================================================

def _trade_plan_gate(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Anti-anchoring gate for Trade Plan rendering.
    Returns (status, meta) where status ∈ {"ACTIVE","WATCH","LOCK"}.
    Meta includes score/tier for UI copy.
    """
    ap = analysis_pack or {}
    cp = character_pack or {}
    score = _safe_float(ap.get("Conviction"), default=np.nan)
    tier = (cp.get("Conviction") or {}).get("Tier", None)

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

def render_game_card(data: Dict[str, Any]):
    """Vẽ Radar Chart cho Game Character"""
    import plotly.graph_objects as go

    if not data:
        st.warning("Chưa có dữ liệu Game Character")
        return

    stats = data.get("Stats", {})
    archetype = data.get("Class", "N/A")
    avg = data.get("AvgScore", 0)

    categories = ["ATK", "SPD", "DEF", "HP", "CRT"]
    values = [
        stats.get("ATK", 0),
        stats.get("SPD", 0),
        stats.get("DEF", 0),
        stats.get("HP", 0),
        stats.get("CRT", 0),
    ]
    # Khép kín vòng tròn
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=archetype,
            line_color="#00FF00" if avg >= 7 else ("#FFFF00" if avg >= 5 else "#FF0000"),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=30, b=30),
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Class", archetype)
        st.metric("Power Level", f"{avg}/10")
    with c2:
        st.plotly_chart(fig, use_container_width=True)

# ============================================================

def render_character_card(character_pack: Dict[str, Any]) -> None:
    """
    Streamlit rendering for Character Card.
    Does not affect existing report A–D.
    """
    cp = character_pack or {}
    core = cp.get("CoreStats") or {}
    combat = cp.get("CombatStats") or {}
    conv = cp.get("Conviction") or {}
    flags = cp.get("Flags") or []
    cclass = cp.get("CharacterClass") or "N/A"
    err = (cp.get("Error") or "")

    ticker = _safe_text(cp.get('_Ticker') or '').strip().upper()
    headline = f"{ticker} - {cclass}" if ticker else str(cclass)
    class_key = str(cclass).strip()
    dash_lines: List[str] = (CLASS_TEMPLATES_DASHBOARD.get(class_key) or []).copy()
    if not dash_lines:
        # Fallback: keep Dashboard readable even if class is unknown
        fallback = get_character_blurb(ticker, class_key)
        if fallback:
            dash_lines = [f"Đặc tính: {fallback}"]
        else:
            dash_lines = [f"Đặc tính: {class_key}"]

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
    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
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


    tier = conv.get("Tier", "N/A")
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


# ============================================================
# 10.9 DECISION LAYER REPORT — OUTPUT ORDER ANTI-ANCHORING BIAS (v6.4)
# ============================================================



# ============================================================
# CLASS TEXT TEMPLATES (STOCK DNA)
# ============================================================

# Fixed 3-paragraph templates per class to keep text stable.
CLASS_TEMPLATES: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Nhóm này có xu hướng dài hạn tương đối rõ và bền. Giá thường bám cấu trúc trend (đỉnh–đáy nâng dần) và tôn trọng các vùng hỗ trợ động, nên hành vi ít bị nhiễu so với nhóm biến động cao.",
        "Nhà đầu tư phù hợp là người đánh theo xu hướng, ưu tiên mua ở nhịp điều chỉnh/pullback thay vì mua đuổi. Có thể nắm giữ trung hạn khi cấu trúc còn nguyên vẹn và chỉ gia tăng khi có xác nhận tiếp diễn.",
        "Điểm lưu ý là tránh phá kỷ luật khi thị trường nhiễu: chỉ giữ vị thế khi trend còn hợp lệ và luôn có mức dừng lỗ theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Nhóm này thiên về động lượng: khi vào pha tăng/giảm, giá thường chạy nhanh theo hướng chính và khó vào lại nếu chậm nhịp. Breakout/continuation có xác suất tốt hơn so với việc bắt đáy.",
        "Phù hợp với trader chủ động, theo dõi sát và chấp nhận ra/vào theo nhịp. Ưu tiên vào khi có follow-through và quản trị vị thế bằng trailing theo cấu trúc.",
        "Cần cảnh giác khi động lượng suy yếu (thiếu follow-through, đà thu hẹp): ưu tiên chốt từng phần và không nới kỷ luật stop.",
    ],
    "Aggressive Trend": [
        "Nhóm này có xu hướng nhưng tải rủi ro cao: biến động lớn, tail risk/gap có thể xuất hiện khi dòng tiền đổi trạng thái. Lợi nhuận tiềm năng cao nhưng sai nhịp sẽ trả giá nhanh.",
        "Phù hợp với trader chịu rung lắc, kỷ luật stop và quản trị size nghiêm ngặt. Chỉ nên tham gia khi plan rõ ràng và điểm vào tối ưu (không FOMO).",
        "Ưu tiên chiến thuật hit-and-run/pyramid có điều kiện sau khi đã giảm rủi ro (free-ride). Tránh giữ vị thế quá lớn qua thời điểm nhạy cảm.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Nhóm này vận động trong biên tương đối ổn định và hay quay về vùng cân bằng. Xác suất mua gần hỗ trợ–bán gần kháng cự tốt hơn kỳ vọng chạy trend dài liên tục.",
        "Phù hợp với trader kỷ luật, kiên nhẫn chờ vùng biên; ưu tiên scale-in ở vùng hỗ trợ và chốt dần ở vùng kháng cự.",
        "Rủi ro chính là phá biên: khi đóng cửa ra khỏi hộp giá, cần cắt/giảm nhanh để tránh bị kéo sang một regime mới.",
    ],
    "Volatile Range": [
        "Nhóm này vẫn có tính chất range/mean-reversion nhưng nhiễu và rung lắc mạnh hơn; dễ false-break và quét stop nếu vào giữa biên.",
        "Phù hợp với trader chọn lọc: chỉ vào khi có xác nhận đảo chiều tại mép biên và chấp nhận tỷ trọng nhỏ hơn bình thường.",
        "Ưu tiên chốt nhanh, không bình quân giá khi cấu trúc yếu và luôn quản trị rủi ro theo volatility.",
    ],
    "Mixed / Choppy Trader": [
        "Nhóm này thiếu xu hướng rõ ràng hoặc hay đổi tính theo thời gian; tín hiệu dễ nhiễu và whipsaw cao. Edge thường chỉ xuất hiện theo từng nhịp ngắn.",
        "Phù hợp với giao dịch chiến thuật: chỉ trade khi setup đạt chất lượng cao và có xác nhận; phần lớn thời gian nên đứng ngoài.",
        "Kỷ luật vào/ra và size nhỏ là bắt buộc; tránh giữ vị thế dài khi cấu trúc không rõ.",
    ],
    "Event / Gap-Prone": [
        "Nhóm này có rủi ro sự kiện/gap cao: giá có thể nhảy mạnh ngoài dự kiến, khiến stop dễ bị trượt. Dù có thể mang lại lợi nhuận lớn, execution risk cũng cao.",
        "Phù hợp với trader chấp nhận rủi ro tail, ưu tiên giao dịch ngắn hạn và giảm nắm giữ qua thời điểm nhạy cảm.",
        "Bắt buộc dùng size nhỏ, stop theo cấu trúc + buffer và chỉ tham gia khi reward đủ lớn để bù rủi ro.",
    ],
    "Illiquid / Noisy": [
        "Nhóm này có rủi ro thực thi: thanh khoản thiếu ổn định, spread/độ trượt có thể làm sai lệch R:R thực tế. Tín hiệu kỹ thuật thường kém tin cậy hơn do nhiễu.",
        "Phù hợp với nhà đầu tư rất kỷ luật và chấp nhận giải ngân nhỏ; ưu tiên lệnh giới hạn và tránh đuổi giá.",
        "Nếu không đạt điều kiện thanh khoản tối thiểu, nên coi đây là nhóm 'NO TRADE' dù kịch bản nhìn đẹp trên giấy.",
    ],
}



# Dashboard (Character Card) short narrative per class — single source of truth for Dashboard narrative
CLASS_TEMPLATES_DASHBOARD: Dict[str, List[str]] = {
    "Smooth Trend": [
        "Đặc tính: Trend bền, hành vi giá tương đối sạch, phù hợp nắm giữ theo cấu trúc.",
        "Phù hợp: Trend-follow / tích lũy theo nhịp điều chỉnh, ưu tiên kỷ luật hơn tốc độ.",
        "Chiến thuật: Pullback & trend continuation, dời stop theo cấu trúc.",
    ],
    "Momentum Trend": [
        "Đặc tính: Động lượng mạnh, chạy nhanh khi có lực, khó vào lại nếu chậm nhịp.",
        "Phù hợp: Trader chủ động, theo dõi sát, chốt lời từng phần theo đà.",
        "Chiến thuật: Breakout/continuation có xác nhận, trailing theo nhịp.",
    ],
    "Aggressive Trend": [
        "Đặc tính: Trend có nhưng rủi ro cao (biến động/tail/gap), sai nhịp trả giá nhanh.",
        "Phù hợp: Trader chịu rung lắc, size nhỏ hơn chuẩn, kỷ luật stop tuyệt đối.",
        "Chiến thuật: Hit & Run / pyramid có điều kiện sau khi giảm rủi ro.",
    ],
    "Range / Mean-Reversion (Stable)": [
        "Đặc tính: Sideway ổn định, hay quay về vùng cân bằng, biên hỗ trợ/kháng cự rõ.",
        "Phù hợp: Trader kiên nhẫn, đánh theo biên, ưu tiên xác suất hơn kỳ vọng lớn.",
        "Chiến thuật: Buy near support – sell near resistance, scale-in/out theo vùng.",
    ],
    "Volatile Range": [
        "Đặc tính: Range nhưng nhiễu, dễ false-break và quét stop; rung lắc mạnh.",
        "Phù hợp: Trader chọn lọc, chỉ vào ở mép biên và giảm tỷ trọng.",
        "Chiến thuật: Vào khi có xác nhận đảo chiều, chốt nhanh, quản trị theo vol.",
    ],
    "Mixed / Choppy Trader": [
        "Đặc tính: Không rõ trend/range, whipsaw cao; edge chỉ xuất hiện theo nhịp ngắn.",
        "Phù hợp: Tactical trader, chấp nhận đứng ngoài phần lớn thời gian.",
        "Chiến thuật: Trade khi setup thật rõ + có xác nhận; size nhỏ.",
    ],
    "Event / Gap-Prone": [
        "Đặc tính: Rủi ro sự kiện/gap cao, stop dễ trượt; cần reward lớn để bù rủi ro.",
        "Phù hợp: Trader mạo hiểm nhưng kỷ luật, tránh giữ qua thời điểm nhạy cảm.",
        "Chiến thuật: Size nhỏ, stop + buffer, chỉ tham gia khi RR đủ dày.",
    ],
    "Illiquid / Noisy": [
        "Đặc tính: Rủi ro thực thi (thanh khoản kém/không ổn định), tín hiệu dễ nhiễu.",
        "Phù hợp: Chỉ dành cho người rất kỷ luật, chấp nhận giải ngân nhỏ.",
        "Chiến thuật: Lệnh giới hạn, tránh đuổi giá; không đạt liquidity gate thì NO TRADE.",
    ],
}


# ============================================================
# CLASS POLICY HINTS (CURRENT STATUS) — DISPLAY-ONLY
# ============================================================
# Purpose: provide a one-line execution policy hint based on FINAL CLASS.
# Option C1 (hint only): does NOT modify scores, triggers, or trade plan logic.
CLASS_POLICY_HINTS: Dict[str, Dict[str, Any]] = {
    "Smooth Trend": {"rr_min": 1.8, "size_cap": "100%", "overnight": "Normal"},
    "Momentum Trend": {"rr_min": 2.0, "size_cap": "85%", "overnight": "Caution"},
    "Aggressive Trend": {"rr_min": 2.2, "size_cap": "70%", "overnight": "Limit"},
    "Range / Mean-Reversion (Stable)": {"rr_min": 1.6, "size_cap": "100%", "overnight": "Normal"},
    "Volatile Range": {"rr_min": 2.0, "size_cap": "70%", "overnight": "Caution"},
    "Mixed / Choppy Trader": {"rr_min": 2.2, "size_cap": "60%", "overnight": "Limit"},
    "Event / Gap-Prone": {"rr_min": 2.5, "size_cap": "50%", "overnight": "Avoid"},
    "Illiquid / Noisy": {"rr_min": 2.5, "size_cap": "40%", "overnight": "Caution"},
}

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
    cp = character_pack or {}
    cclass = _safe_text(cp.get("CharacterClass") or "N/A").strip()
    ticker = _safe_text(cp.get("_Ticker") or "").strip().upper()

    # ---- Class narrative (stable templates) ----
    class_label = f"CLASS: {cclass}"
    st.markdown(f"**{html.escape(class_label)}**")
    paras = CLASS_TEMPLATES.get(cclass) or []
    if paras:
        for para in paras:
            st.markdown(str(para))
    else:
        # fallback if template is missing
        st.markdown(get_character_blurb(ticker, cclass) or "")

    # ---- DNA pack (15 params / 5 groups) ----
    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
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
            v_disp = "N/A"
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

    style_axis = tier1.get("StyleAxis", "N/A")
    risk_regime = tier1.get("RiskRegime", "N/A")
    dna_conf = tier1.get("DNAConfidence", np.nan)
    lock_flag = tier1.get("ClassLockFlag", False)
    modifiers = tier1.get("Modifiers", []) or []

    conf_txt = "N/A" if pd.isna(_safe_float(dna_conf, default=np.nan)) else f"{float(dna_conf):.0f}/100"
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
    cp = character_pack or {}
    combat = cp.get("CombatStats") or {}

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
            v_disp = "N/A"
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
    cp = character_pack or {}
    cclass = _safe_text(cp.get("CharacterClass") or "N/A").strip()

    dna = (cp.get("StockTraits") or {}).get("DNA") or {}
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

    style_axis = _safe_text(tier1.get("StyleAxis", "N/A"))
    risk_regime = _safe_text(tier1.get("RiskRegime", "N/A"))
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

    conf_txt = "N/A" if pd.isna(dna_conf) else f"{dna_conf:.0f}/100"

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

# ============================================================
# 10. DASHBOARD SUMMARY PACK (v1) — Single source of truth for Executive Snapshot
# ============================================================

def render_executive_snapshot(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any], gate_status: str) -> None:
    """Executive Snapshot — dashboard-style summary card.

    Renderer-only: must not change engine scoring or trade-plan math.

    Notes:
      - Uses HTML for layout; always escape dynamic strings.
      - Detail sections (Stock DNA / Current Status / Trade Plan / Decision Layer) are rendered separately under an expander.
    """
    ap = analysis_pack or {}
    cp = character_pack or {}

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
        return "N/A" if pd.isna(v) else f"{v:.{nd}f}"

    def _fmt_px(x: Any) -> str:
        v = _sf(x)
        return "N/A" if pd.isna(v) else f"{v:.2f}"

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
            return "TIER N/A"
        label_map = {
            7: "GOD-TIER",
            6: "VERY STRONG BUY",
            5: "STRONG BUY",
            4: "BUY",
            3: "WATCH",
            2: "CAUTIOUS",
            1: "NO EDGE",
        }
        return f"TIER {t}: {label_map.get(t, 'N/A')}"

    def _kelly_label(tier: Any) -> str:
        try:
            t = int(float(tier))
        except Exception:
            return "KELLY BET: N/A"
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
    ticker = _safe_text(ap.get("Ticker") or cp.get("_Ticker") or "").strip().upper()
    last_pack = ap.get("Last") or {}
    close_px = last_pack.get("Close")

    mkt = ap.get("Market") or {}
    chg_pct = mkt.get("StockChangePct")

    scenario_name = _safe_text((ap.get("Scenario12") or {}).get("Name") or "N/A").strip()

    master_total = (ap.get("MasterScore") or {}).get("Total", np.nan)
    conviction = ap.get("Conviction", np.nan)

    class_name = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "N/A").strip()

    conv = cp.get("Conviction") or {}
    tier = conv.get("Tier", None)

    core = cp.get("CoreStats") or {}
    combat = cp.get("CombatStats") or {}

    # Primary setup (already computed by Python)
    primary = ap.get("PrimarySetup") or {}
    primary = primary if isinstance(primary, dict) else {}
    setup_name = _safe_text(primary.get("Name") or "N/A").strip()

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
    if plan_type_es and plan_type_es != "N/A":
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

    # Triggers
    vol_ratio = (ap.get("ProTech") or {}).get("Volume", {}).get("Ratio")
    rr_val = rr

    dot_breakout = _dot(combat.get("BreakoutForce"), good=6.8, warn=5.5)
    dot_volume = _dot(vol_ratio, good=1.20, warn=0.95)
    dot_rr = _dot(rr_val, good=1.80, warn=1.30)

    # --- DEBUG (auto-show only when Upside Room is N/A) ---
    meta = cp.get("Meta") or {}
    if pd.isna(_sf((combat or {}).get("UpsideRoom", (combat or {}).get("UpsidePower")))):
        st.caption(f"[DEBUG] UpsideRoom=N/A | DenomUsed={meta.get('DenomUsed')} | ATR14={_fmt_num(meta.get('ATR14'),2)} | VolProxy={_fmt_num(meta.get('VolProxy'),2)}")
        st.caption(f"[DEBUG] Close={_fmt_num(meta.get('Close'),2)} | NR={_fmt_num(meta.get('NearestRes'),2)} | NS={_fmt_num(meta.get('NearestSup'),2)} | UpsideRaw={_fmt_num(meta.get('UpsideRaw'),2)} | DownsideRaw={_fmt_num(meta.get('DownsideRaw'),2)} | LvlSrc={meta.get('LevelCtxSource')}")
        st.caption(f"[DEBUG] UpsideNorm={_fmt_num(meta.get('UpsideNorm'),2)} | DownsideNorm={_fmt_num(meta.get('DownsideNorm'),2)} | RR={_fmt_num(meta.get('RR'),2)} | BreakoutForce={_fmt_num((combat or {}).get('BreakoutForce'),2)} | VolRatio={_fmt_num(vol_ratio,2)} | RR_plan={_fmt_num(rr_val,2)} | LvlKeys={meta.get('LevelCtxKeys')}")

    # --------- render ---------
    tier_badge = _tier_label(tier)
    kelly_badge = _kelly_label(tier)
    gate = (gate_status or "N/A").strip().upper()

    # Header strings
    title_left = f"{ticker}"
    if _fmt_px(close_px) != "N/A":
        title_left = f"{title_left} | {_fmt_px(close_px)}"
    chg_str = _fmt_pct(chg_pct)

    sub_1 = " | ".join([x for x in [class_name, scenario_name] if x and x != "N/A"])
    sub_2 = f"Điểm tổng hợp: {_fmt_num(master_total,1)} | Điểm tin cậy: {_fmt_num(conviction,1)} | Gate: {gate}"

    # Pillar metrics
    def _metric_row(k: str, v: Any, nd: int = 1):
        return f"<div class='es-metric'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(_fmt_num(v, nd))}</div></div>"

        # Panel 1 (DNA) — compact class narrative + Class Signature radar (5 metrics)
    dash_lines: List[str] = (CLASS_TEMPLATES_DASHBOARD.get(class_name) or []).copy()
    if not dash_lines:
        # Fallback: keep Dashboard readable even if class is unknown
        fallback = get_character_blurb(ticker, class_name)
        if fallback:
            dash_lines = [f"Đặc tính: {fallback}"]
        else:
            dash_lines = [f"Đặc tính: {class_name}"]

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
    dna_pack = (cp.get("StockTraits") or {}).get("DNA") or {}
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
    if conf_txt_es != "N/A":
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

    state_capsule_line = _safe_text(card2.get("state_capsule_line") or "N/A").strip()

    master_total_d = card2.get("master_total", master_total)
    conviction_d = card2.get("conviction", conviction)

    ms_pct = _bar_pct_10(master_total_d)
    cs_pct = _bar_pct_10(conviction_d)

    insight_line_es = _safe_text(card2.get("insight_line") or "").strip()
    policy_hint_es = _safe_text(card2.get("policy_hint_line") or "").strip()

    triggers2 = card2.get("triggers") if isinstance(card2.get("triggers"), dict) else {}
    st_break = _safe_text((triggers2 or {}).get("breakout") or "N/A").strip().upper()
    st_vol = _safe_text((triggers2 or {}).get("volume") or "N/A").strip().upper()
    st_rr = _safe_text((triggers2 or {}).get("rr") or "N/A").strip().upper()
    st_struct = _safe_text((triggers2 or {}).get("structure") or "N/A").strip().upper()

    def _norm_st(s: str) -> str:
        s = (s or "").strip().upper()
        if s in ("PASS", "WAIT", "FAIL"):
            return s
        return "N/A"

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
        mode_d = _safe_text(dsum.get("mode") or "N/A").strip().upper()
        action_d = _safe_text(dsum.get("action") or "N/A").strip().upper()
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

        if action_d and action_d != "N/A":
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
    action_for_plan = "N/A"
    try:
        dsum2 = card2.get("decision") if isinstance(card2.get("decision"), dict) else {}
        mode_for_plan = _safe_text(dsum2.get("mode") or "FLAT").strip().upper()
        action_for_plan = _safe_text(dsum2.get("action") or "N/A").strip().upper()
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

    panel3 = f"""
    <div class="es-panel">
      <div class="es-pt">3) SCENARIO</div>

      {decision_block_html}

      {plan_block_html}

      <div class="es-note" style="margin-top:10px;"><b>Kịch bản chính:</b> {html.escape(scenario_name)}</div>
      <ul class="es-bul">
        <li>Setup: {html.escape(setup_name)}</li>
        <li>Entry/Stop: {html.escape(_fmt_px(entry))} / {html.escape(stop_str)}</li>
        <li>Target: {html.escape(tp_str)} (RR {html.escape(_fmt_num(rr,1))})</li>
      </ul>
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

def render_character_decision(character_pack: Dict[str, Any]) -> None:
    """
    Render only the 'Decision' part of Character Card (for Appendix E / anti-anchoring).
    Includes: Conviction + Weaknesses + Playstyle Tags.
    """
    cp = character_pack or {}
    conv = cp.get("Conviction") or {}
    flags = cp.get("Flags") or []
    tags = cp.get("ActionTags") or []

    tier = conv.get("Tier", "N/A")
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

def render_market_state(analysis_pack: Dict[str, Any]) -> None:
    """
    Appendix E section (2): Market State / Current Regime.
    Must never crash if market context is missing.
    """
    ap = analysis_pack or {}
    m = ap.get("Market") or {}
    vn = m.get("VNINDEX") or {}
    vn30 = m.get("VN30") or {}

    def _fmt_change(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return "N/A"
        return f"{v:+.2f}%"

    st.markdown('<div class="gc-sec"><div class="gc-sec-t">MARKET STATE (CURRENT REGIME)</div>', unsafe_allow_html=True)

    # If no market pack, show a clean fallback (no error, no stacktrace)
    if not m:
        st.info("Market State: N/A (market context not available).")
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
            return "N/A"
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

def render_trade_plan_conditional(analysis_pack: Dict[str, Any], character_pack: Dict[str, Any], gate_status: str, trade_text: str = "") -> None:
    """
    Appendix E section (3): Trade Plan & R:R (Conditional).
    Uses AnalysisPack.TradePlans (Python-computed). No GPT math.
    Layout-only: reorders how the engine output is displayed.
    """
    ap = analysis_pack or {}

    cp = character_pack or {}

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
            return "N/A" if pd.isna(x) else f"{x:.2f}"

        def _fmt_rr(v: Any) -> str:
            x = _sf(v)
            return "N/A" if pd.isna(x) else f"{x:.2f}"

        def _fmt_rrm(v: Any) -> str:
            x = _sf(v)
            return "N/A" if pd.isna(x) else f"{x:.1f}"

        def _fmt_zone(z: Any) -> str:
            if not isinstance(z, dict):
                return "N/A"
            lo = _sf(z.get("Low"))
            hi = _sf(z.get("High"))
            if pd.isna(lo) or pd.isna(hi):
                return "N/A"
            return f"{lo:.2f} – {hi:.2f}"

        def _fmt_gates(g: Any) -> str:
            if not isinstance(g, dict):
                return "N/A"
            keys = ["trigger", "volume", "rr", "exec", "structure"]
            parts = []
            for k in keys:
                parts.append(f"{k.upper()}={_safe_text(g.get(k) or 'N/A').strip().upper()}")
            return " | ".join(parts)

        def _render_plan(title: str, p: Any) -> None:
            if not isinstance(p, dict) or not p:
                return
            ptype = _safe_text(p.get("type") or "N/A").strip()
            pstate = _safe_text(p.get("state") or "N/A").strip().upper()
            st.markdown(f"#### {title} — {ptype} | {pstate}")

            st.caption(_fmt_gates(p.get("gates")))

            zone = _fmt_zone(p.get("entry_zone"))
            stop = _fmt_px(p.get("stop"))
            tp1 = _fmt_px(p.get("tp1"))
            tp2 = _fmt_px(p.get("tp2"))
            rr = _fmt_rr(p.get("rr_actual"))
            rrm = _fmt_rrm(p.get("rr_min"))

            st.markdown(f"- Entry zone: **{zone}**")
            st.markdown(f"- Stop: **{stop}** | TP1: **{tp1}** | TP2: **{tp2}** | RR: **{rr}** (min {rrm})")

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

        if trade_text:
            with st.expander("Narrative (optional)"):
                st.markdown(trade_text)

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
                    return "N/A"
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

        def _fmt_rr_local(x: Any) -> str:
            try:
                if x is None or pd.isna(x) or not math.isfinite(float(x)):
                    return "N/A"
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"

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
        names = [n for n in (plan_names or []) if n and str(n).strip() and str(n).strip().upper() != "N/A"]
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
            rr_disp = "N/A"

        def _fmt_px(x: Any) -> str:
            v = _safe_float(x, default=np.nan)
            return f"{v:.2f}" if pd.notna(v) else "N/A"

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

def render_decision_layer_switch(character_pack: Dict[str, Any], analysis_pack: Dict[str, Any], gate_status: str, exec_mode_text: str, preferred_plan: str) -> None:
    """Renderer-only: central Decision Layer switch (no scoring / rule changes).

    v7.1 renderer improvements:
      - Humanize PROTECH labels (replace underscores, add spacing)
      - Convert long BiasCode into short pills (avoid unreadable mega-string)
      - No red accents (orange/neutral only)
    """
    cp = character_pack or {}
    ap = analysis_pack or {}

    # ------------------------------------------------------------
    # DecisionPack.v1 (Portfolio-ready; Long-only) — render if available
    # ------------------------------------------------------------
    dp = ap.get("DecisionPack") or {}
    dp = dp if isinstance(dp, dict) else {}
    if _safe_text(dp.get("schema") or "").strip() == "DecisionPack.v1":
        act = _safe_text(dp.get("action") or "N/A").strip().upper()
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
        pm_act = _safe_text(pmp.get("action") or "N/A").strip().upper()
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
        if not code or code.upper() == "N/A":
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

    conv = cp.get("Conviction") or {}
    tier = conv.get("Tier", "N/A")
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
        elif pstat in ("INVALID", "DISABLED", "N/A", "NA"):
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
            if pstat in ("INVALID", "DISABLED", "N/A", "NA"):
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
    alignment_raw = _safe_text(bias.get("Alignment") or "N/A").strip()
    bias_code_raw = _safe_text(bias.get("BiasCode") or "").strip()

    alignment = _humanize(alignment_raw) or "N/A"
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
    pts_disp = f"{pts:.1f}" if pd.notna(pts) else "N/A"
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

def render_appendix_e(result: Dict[str, Any], report_text: str, analysis_pack: Dict[str, Any]) -> None:
    """
    Decision Layer Report — Anti-Anchoring Output Order (layout only):
      1) Stock DNA (Traits)
      2) CURRENT STATUS (Scenario + Technical + Fundamental)
      3) Trade Plan & R:R (Conditional)
      4) Decision Layer (Conviction/Weakness/Tags)
    This renderer must not change any underlying calculations.
    """
    modules = (result or {}).get("Modules") or {}
    cp = modules.get("character") or {}
    ap = analysis_pack or {}

    # Trade-plan gate (for execution / anti-FOMO posture)
    gate_status, _meta = _trade_plan_gate(analysis_pack, cp)

    # ---------- HEADER: <Ticker> — <Last Close> <+/-%> ----------
    ticker = _safe_text(ap.get("Ticker") or (result or {}).get("Ticker") or "").strip().upper()
    last_pack = ap.get("Last") or {}
    close_val = _safe_float(last_pack.get("Close"), default=np.nan)

    mkt = ap.get("Market") or {}
    stock_chg = _safe_float(mkt.get("StockChangePct"), default=np.nan)

    def _fmt_close(x: Any) -> str:
        try:
            v = float(x)
            if not math.isfinite(v):
                return "N/A"
            return f"{v:.2f}"
        except Exception:
            return "N/A"

    def _fmt_change_pct(x: Any) -> str:
        v = _safe_float(x, default=np.nan)
        if pd.isna(v):
            return ""
        return f"{v:+.2f}%"

    price_str = _fmt_close(close_val)
    chg_str = _fmt_change_pct(stock_chg)

    if ticker or price_str != "N/A":
        header = ticker or ""
        if price_str != "N/A":
            header = f"{header} — {price_str}" if header else price_str
        if chg_str:
            header = f"{header} ({chg_str})"
        render_executive_snapshot(ap, cp, gate_status)

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
            render_character_traits(cp)
            render_stock_dna_insight(cp)
    
            # ============================================================
            # 2) CURRENT STATUS
            # ============================================================
            st.markdown('<div class="major-sec">CURRENT STATUS</div>', unsafe_allow_html=True)

            # 2.1 Relative Strength vs VNINDEX
            rel = (ap.get("Market") or {}).get("RelativeStrengthVsVNINDEX")
            st.markdown(f"**Relative Strength vs VNINDEX:** {_val_or_na(rel)}")

            # 2.2 Scenario & Scores
            scenario_pack = ap.get("Scenario12") or {}
            master_pack = ap.get("MasterScore") or {}
            conviction_score = ap.get("Conviction")

            st.markdown("**State Capsule (Scenario & Scores)**")
            st.markdown(f"- Scenario: {_val_or_na(scenario_pack.get('Name'))}")

            def _bar_row_cs(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "N/A"
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

            _bar_row_cs("Điểm tổng hợp", master_pack.get("Total"), 10.0)
            _bar_row_cs("Điểm tin cậy", conviction_score, 10.0)

            # Score interpretation (single block) — place directly under the two bars
            render_current_status_insight(master_pack.get("Total"), conviction_score, gate_status)

            # Class Policy Hint (display-only)
            _final_class = _safe_text(cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "").strip()
            _policy_hint = get_class_policy_hint_line(_final_class)
            if _policy_hint:
                st.markdown(f"**Policy:** {_policy_hint}")

            # 2.3 State Capsule (Facts-only, compact)
            st.markdown("**Structure Summary (MA/Fibo/RSI/MACD/Volume)**")
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
                    st.info("N/A")



            # 2.5 Combat Readiness (Now) — merged from legacy Combat Stats
            st.markdown("**Combat Readiness (Now)**")
            combat = cp.get("CombatStats") or {}
            combat = combat if isinstance(combat, dict) else {}

            def _bar_row_now(label: str, val: Any, maxv: float = 10.0) -> None:
                v = _safe_float(val, default=np.nan)
                if pd.isna(v):
                    pct = 0.0
                    v_disp = "N/A"
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

            plan_status = "N/A"
            plan_tags: List[str] = []
            for p in (ap.get("TradePlans") or []):
                if _safe_text(p.get("Name")).strip() == setup_name and setup_name and setup_name != "N/A":
                    plan_status = _safe_text(p.get("Status") or "N/A")
                    plan_tags = list(p.get("ReasonTags") or [])
                    rr_val = _safe_float(p.get("RR"), default=rr_val)
                    break

            def _status_from_val(v: Any, good: float, warn: float) -> Tuple[str, str]:
                x = _safe_float(v, default=np.nan)
                if pd.isna(x):
                    return ("N/A", "#9CA3AF")
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
            s_struct = _safe_text(cg.get("Status") or "N/A").strip().upper()
            if s_struct not in ("PASS", "WAIT", "FAIL"):
                s_struct = "N/A"
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
                      <li>{_dot("#60A5FA")} Gate: {html.escape(str(gate_status or "N/A"))} | Plan: {html.escape(str(setup_name or "N/A"))} ({html.escape(str(plan_status or "N/A"))})</li>
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

            dna_t1 = (((cp.get("StockTraits") or {}).get("DNA") or {}).get("Tier1") or {})
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
    # ============================================================
    # 11. GPT-4o STRATEGIC INSIGHT GENERATION
    # ============================================================


# ------------------------------------------------------------
# Deterministic Report A–D (Facts-only fallback)
# - Used when OPENAI_API_KEY is missing OR GPT call fails.
# - Ensures A–D sections always exist so UI split/render stays stable.
# ------------------------------------------------------------

def render_report_pretty(report_text: str, analysis_pack: dict):
    sections = _split_sections(report_text)
    a_items = _extract_a_items(sections.get("A", ""))

    st.markdown('<div class="incept-wrap">', unsafe_allow_html=True)

    ap = analysis_pack or {}
    scenario_pack = ap.get("Scenario12") or {}
    master_pack = ap.get("MasterScore") or {}
    conviction_score = ap.get("Conviction", "N/A")

    def _val_or_na(v):
        if v is None: return "N/A"
        if isinstance(v, float) and pd.isna(v): return "N/A"
        text = str(v).strip()
        return text if text else "N/A"

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
        st.info("N/A")

    st.markdown('<div class="sec-title">RỦI RO &amp; LỢI NHUẬN</div>', unsafe_allow_html=True)
    ps = (analysis_pack or {}).get("PrimarySetup") or {}
    risk = ps.get("RiskPct", None)
    reward = ps.get("RewardPct", None)
    rr = ps.get("RR", None)
    prob = ps.get("Confidence (Tech)", ps.get("Probability", "N/A"))

    def _fmt_pct_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}%"
        except Exception:
            return "N/A"

    def _fmt_rr_local(x):
        try:
            if x is None or pd.isna(x) or not math.isfinite(float(x)):
                return "N/A"
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "N/A"

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

# ============================================================

