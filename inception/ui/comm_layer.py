"""
Communication Layer (Co-pilot) — Behavioral Linguistics Policy
INCEPTION v15.x

Purpose:
- Convert internal dashboard/status/plan/decision signals into ONE client-friendly paragraph.
- 100% Vietnamese, professional but easy for mainstream users.
- Reduce blame: third-party anchor + A/B agency shift + pre-mortem (1 invalidation level)
- No raw system tags (Gate/Plan/PASS/ACTIVE/WATCH/TRIM/structure/flow...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import math
import re


# ---------------------------------------------------------------------
# A) Terms dictionary (EN/UI label -> Vietnamese)
# ---------------------------------------------------------------------
TERMS_MAP: Dict[str, Dict[str, str]] = {
    "actions": {
        "BUY": "mở mua",
        "ADD": "gia tăng vị thế",
        "HOLD": "tiếp tục nắm giữ",
        "WAIT": "quan sát thêm",
        "TRIM": "giảm tỷ trọng",
        "EXIT": "thoát vị thế",
        "SELL": "bán",
    },
    "action_intensity": {
        "HIGH": "mạnh",
        "MED": "vừa",
        "LOW": "nhẹ",
    },
    "mode": {
        "FLAT": "mua mới",
        "HOLDING": "đang nắm giữ",
    },
    "gate": {
        "ACTIVE": "trạng thái có thể hành động",
        "WATCH": "trạng thái cần quan sát thêm",
        "AVOID": "trạng thái nên tránh (rủi ro cao)",
    },
    "technical_phrases": {
        "Below MA200": "dưới MA200 (xu hướng dài hạn chưa thuận)",
        "Above MA200": "trên MA200 (xu hướng dài hạn thuận)",
        "Breakout Attempt": "đang thử vượt cản",
        "Volume Spike": "khối lượng tăng đột biến",
        "Structure PASS": "cấu trúc giá ủng hộ",
        "Volume PASS": "dòng tiền ủng hộ",
        "R:R PASS": "tỷ lệ lợi nhuận/rủi ro đạt yêu cầu",
    },
    "plan_words": {
        "protect_stop": "mốc bảo vệ vốn",
        "reclaim_zone": "vùng xác nhận hồi phục",
        "trim_guide": "gợi ý giảm tỷ trọng",
        "key_blockers": "điểm nghẽn chính",
    },
}

# ---------------------------------------------------------------------
# B) DNA/Flag dictionary — stable Vietnamese phrasing
# ---------------------------------------------------------------------
DNA_FLAG_MAP: Dict[str, Dict[str, str]] = {
    "Event/Gap-Prone": {
        "label": "nhóm biến động theo tin (dễ nhảy giá)",
        "risk": "giá có thể biến động bất ngờ, stop dễ bị trượt",
        "rule": "ưu tiên kỷ luật chặn lỗ và quản trị tỷ trọng; không phù hợp gồng dài khi vị thế xấu",
    },
    "Illiquid": {
        "label": "nhóm thanh khoản thấp",
        "risk": "khó vào/ra đúng giá, dễ bị trượt giá khi thị trường xấu",
        "rule": "chỉ nên vào nhỏ, ưu tiên lệnh giới hạn; tránh đánh lớn",
    },
    "High Volatility": {
        "label": "nhóm biến động mạnh",
        "risk": "dao động rộng, dễ bị rung lắc khỏi vị thế",
        "rule": "giảm quy mô, stop có đệm; chỉ tham gia khi điểm vào rõ",
    },
    "Trend-Strong": {
        "label": "nhóm xu hướng rõ",
        "risk": "rủi ro chủ yếu là mua đuổi sai điểm",
        "rule": "ưu tiên chiến lược mua khi điều chỉnh; tránh mua đuổi",
    },
    "Range/Mean-Revert": {
        "label": "nhóm đi ngang (hồi quy)",
        "risk": "break giả nhiều, mua đuổi dễ kẹt",
        "rule": "ưu tiên mua gần hỗ trợ/bán gần kháng cự; kỷ luật biên độ",
    },
    "Momentum-Aligned": {
        "label": "động lượng đang đồng thuận",
        "risk": "hưng phấn ngắn hạn có thể đảo chiều nhanh",
        "rule": "chỉ gia tăng khi có xác nhận bằng dòng tiền và cấu trúc giá",
    },
    "Breakout-Prone": {
        "label": "nhóm hay có nhịp bứt phá",
        "risk": "break giả nếu thiếu dòng tiền",
        "rule": "chỉ mua khi vượt cản kèm khối lượng; không mua trước xác nhận",
    },
    "Defensive": {
        "label": "chế độ phòng thủ",
        "risk": "ưu tiên bảo toàn vốn hơn tìm lợi nhuận",
        "rule": "giảm rủi ro, giữ kỷ luật mốc bảo vệ; chỉ chủ động lại khi có vùng xác nhận",
    },
}

# ---------------------------------------------------------------------
# C) Action sentences — to avoid “random” wording
# ---------------------------------------------------------------------
ACTION_SENTENCES: Dict[str, Dict[str, Dict[str, str]]] = {
    "HOLDING": {
        "TRIM": {
            "HIGH": "Nên giảm khoảng 1/4 vị thế để hạ rủi ro và giảm áp lực tâm lý.",
            "MED": "Nên giảm một phần vị thế để hạ rủi ro.",
            "LOW": "Có thể giảm nhẹ để chủ động quản trị rủi ro.",
        },
        "HOLD": {
            "ANY": "Có thể giữ vị thế, nhưng cần bám sát mốc bảo vệ vốn và tránh gia tăng khi chưa có xác nhận.",
        },
        "EXIT": {
            "ANY": "Ưu tiên thoát vị thế để bảo toàn phần vốn còn lại và giảm rủi ro kéo dài.",
        },
        "WAIT": {
            "ANY": "Có thể giữ ở mức kiểm soát và quan sát thêm; tránh hành động mạnh khi điểm vào chưa chín.",
        },
        "BUY": {
            "ANY": "Nếu đang nắm giữ, ưu tiên quản trị rủi ro trước; chưa nên gia tăng khi chưa có xác nhận.",
        },
        "ADD": {
            "ANY": "Chỉ gia tăng khi có xác nhận rõ ràng; tránh gia tăng trong nhịp rung lắc.",
        },
    },
    "FLAT": {
        "BUY": {
            "ANY": "Có thể mở mua thăm dò, ưu tiên kỷ luật điểm vào và quản trị rủi ro.",
        },
        "WAIT": {
            "ANY": "Nên quan sát thêm, chưa vội mở mua.",
        },
        "HOLD": {
            "ANY": "Nên quan sát thêm, chưa vội mở mua.",
        },
        "TRIM": {
            "ANY": "Ưu tiên đứng ngoài; chưa có vị thế để giảm tỷ trọng.",
        },
        "EXIT": {
            "ANY": "Ưu tiên đứng ngoài để tránh rủi ro.",
        },
        "ADD": {
            "ANY": "Chưa có vị thế; nếu muốn tham gia, hãy chờ xác nhận rồi mở mua thăm dò.",
        },
    },
}

# Forbidden raw system tokens in client paragraph (case-insensitive)
FORBIDDEN_SUBSTRINGS = [
    "gate:", "plan:", "pass", "fail", "wait", "active", "watch", "avoid",
    "trim", "structuregate", "volumegate", "rrgate",
]


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _fmt_price_1dp(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "-"
    return f"{v:.1f}"


def _fmt_price_2dp_or_1dp(x: Any) -> str:
    """Keep existing system stop formatting flexible:
    - if close to 2dp precision (has cents), show 2dp
    - else show 1dp
    """
    v = _safe_float(x)
    if v is None:
        return "-"
    # Heuristic: if 2dp carries information, keep 2dp; else 1dp
    if abs(v - round(v, 1)) > 1e-9:
        return f"{v:.2f}"
    return f"{v:.1f}"


def _fmt_zone_1dp(lo: Any, hi: Any, single: Any = None) -> str:
    vlo = _safe_float(lo)
    vhi = _safe_float(hi)
    if vlo is not None and vhi is not None:
        return f"{vlo:.1f}–{vhi:.1f}"
    vs = _safe_float(single)
    return "-" if vs is None else f"{vs:.1f}"


def _vn_action(action: str) -> str:
    a = (action or "").strip().upper()
    return TERMS_MAP["actions"].get(a, "quan sát thêm")


def _vn_intensity(urgency: str) -> str:
    u = (urgency or "").strip().upper()
    return TERMS_MAP["action_intensity"].get(u, "")


def _vn_gate(gate_status: str) -> str:
    g = (gate_status or "").strip().upper()
    return TERMS_MAP["gate"].get(g, "trạng thái cần quan sát thêm")


def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    # Remove accidental raw tokens
    for tok in FORBIDDEN_SUBSTRINGS:
        s = re.sub(re.escape(tok), "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_dna_key(class_name: str, dash_lines: List[str], risk_flags: List[str]) -> Optional[str]:
    # Prefer explicit known flags if present in risk flags
    joined = " | ".join([_safe_text(x) for x in (risk_flags or [])]).lower()
    for k in DNA_FLAG_MAP.keys():
        if k.lower() in joined:
            return k
    # Try class name
    cn = (class_name or "").strip()
    if cn in DNA_FLAG_MAP:
        return cn
    # Try dash_lines[0] keywords
    if dash_lines:
        s0 = _safe_text(dash_lines[0]).lower()
        for k in DNA_FLAG_MAP.keys():
            if k.lower() in s0:
                return k
    return None


def _build_dna_block(ticker: str, class_name: str, dash_lines: List[str], risk_flags: List[str]) -> str:
    key = _pick_dna_key(class_name, dash_lines, risk_flags)
    if key and key in DNA_FLAG_MAP:
        d = DNA_FLAG_MAP[key]
        return f"{ticker} thuộc {d['label']}. Đặc điểm là {d['risk']}. Vì vậy, nguyên tắc phù hợp là {d['rule']}."
    # Fallback: use first dash line if it looks Vietnamese and not too raw
    if dash_lines:
        s0 = _safe_text(dash_lines[0]).strip()
        s0 = re.sub(r"^Đặc\s*tính\s*:\s*", "", s0, flags=re.IGNORECASE).strip()
        if s0 and len(s0) <= 180:
            return f"{ticker} thuộc nhóm {s0}."
    return f"{ticker} thuộc nhóm {class_name or 'DNA hiện tại'}; ưu tiên kỷ luật quản trị rủi ro."




def _soft_translate_next_step(s: str) -> str:
    """Translate common residual English/UI shorthand into Vietnamese, without leaking raw tags."""
    if not s:
        return ""
    out = str(s)

    # Common jargon replacements (case-insensitive)
    repl = {
        "fomo": "mua đuổi",
        "plan": "kế hoạch",
        "stop": "mốc bảo vệ vốn",
        "breakout": "vượt cản",
        "structure": "cấu trúc giá",
        "flow": "dòng tiền",
        "watch": "cần quan sát thêm",
        "active": "có thể hành động",
        "pass": "đạt yêu cầu",
        "fail": "chưa đạt",
    }
    for k, v in repl.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)

    # Remove any leftover raw tag hints like "Gate:" "Plan:" etc (defensive)
    out = re.sub(r"\bGate\s*:\s*\w+\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bPlan\s*:\s*[^,;]+", "", out, flags=re.IGNORECASE)

    # Cleanup punctuation/spacing
    out = out.replace("|", " ")
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = out.strip(" -;|")
    return out

def _build_status_one_liner(blockers: Dict[str, str], gate_status: str, next_step: str) -> str:
    # Map blocker WAIT/FAIL into Vietnamese reason snippets (1–2 items only)
    reasons: List[str] = []
    def _st(x: Any) -> str:
        return (x or "").strip().upper()
    b = blockers or {}
    if _st(b.get("structure")) in ("WAIT", "FAIL"):
        reasons.append("cấu trúc giá chưa xác nhận")
    if _st(b.get("volume")) in ("WAIT", "FAIL"):
        reasons.append("dòng tiền chưa ủng hộ")
    if _st(b.get("breakout")) in ("WAIT", "FAIL"):
        reasons.append("chưa vượt cản rõ ràng")
    if _st(b.get("rr")) in ("WAIT", "FAIL"):
        reasons.append("tỷ lệ lợi nhuận/rủi ro chưa đủ hấp dẫn")

    g = (gate_status or "").strip().upper()
    ns = _safe_text(next_step).strip()
    ns = _clean_sentence(ns)
    ns = _soft_translate_next_step(ns)

    reason = ""
    if reasons:
        reason = ", ".join(reasons[:2])
    elif ns:
        # use next_step as a soft one-liner if it is already Vietnamese
        reason = ns
    else:
        reason = "điểm vào theo kế hoạch chưa chín"

    if g == "ACTIVE":
        if reasons:
            return f"Tổng trạng thái đang thuận, nhưng điểm vào theo kế hoạch vẫn ở trạng thái cần quan sát thêm vì {reason}."
        return "Tổng trạng thái đang thuận và điều kiện xác nhận đã rõ ràng hơn; có thể ưu tiên hành động theo kỷ luật."
    if g == "AVOID":
        return "Hiện tại rủi ro đang lấn át; ưu tiên phòng thủ và bảo vệ vốn."
    # WATCH / default
    return f"Điểm đáng chú ý lúc này: {reason}."


def _select_invalidation_level(protect_stop: Any, stop_suggest: Any, reclaim_low: Any) -> Tuple[str, Optional[float]]:
    # Prefer protect stop, else stop_suggest, else reclaim_low
    for x in (protect_stop, stop_suggest, reclaim_low):
        v = _safe_float(x)
        if v is not None:
            return ("đóng cửa dưới", v)
    return ("đóng cửa dưới", None)


def _should_empathize(is_holding: bool, pnl_pct: Optional[float], action: str) -> bool:
    if (action or "").strip().upper() in ("TRIM", "EXIT"):
        return True
    if is_holding and pnl_pct is not None and pnl_pct < 0:
        return True
    return False


def build_comm_paragraph(ctx: Dict[str, Any]) -> str:
    """Build ONE Vietnamese paragraph for client-facing Communication Layer."""
    ticker = _safe_text(ctx.get("ticker") or "").strip().upper() or "MÃ"
    gate_status = _safe_text(ctx.get("gate_status") or "").strip().upper()

    master = _safe_float(ctx.get("master_score"))
    conv = _safe_float(ctx.get("conviction_score"))

    class_name = _safe_text(ctx.get("class_name") or "").strip()
    dash_lines = ctx.get("dash_lines") if isinstance(ctx.get("dash_lines"), list) else []
    risk_flags = ctx.get("risk_flags") if isinstance(ctx.get("risk_flags"), list) else []

    mode = _safe_text(ctx.get("mode") or "FLAT").strip().upper()
    action = _safe_text(ctx.get("action") or "WAIT").strip().upper()
    urgency = _safe_text(ctx.get("urgency") or "").strip().upper()

    pnl_pct = _safe_float(ctx.get("pnl_pct"))
    is_holding = (mode == "HOLDING") or bool(ctx.get("is_holding"))

    blockers = ctx.get("blockers") if isinstance(ctx.get("blockers"), dict) else {}
    next_step = _safe_text(ctx.get("next_step") or "").strip()

    # Key levels (prefer defensive overlay levels when provided upstream)
    protect_stop = ctx.get("protect_stop")
    stop_suggest = ctx.get("stop_suggest")
    reclaim_low = ctx.get("reclaim_low")
    reclaim_high = ctx.get("reclaim_high")
    reclaim_single = ctx.get("reclaim")

    # --- 1) DNA sentence ---
    dna_block = _build_dna_block(ticker, class_name, dash_lines, risk_flags)
    dna_block = _clean_sentence(dna_block)

    # --- 2) Status sentence(s) ---
    gate_vn = _vn_gate(gate_status)
    parts: List[str] = []
    parts.append(f"INCEPTION đang đánh giá: {dna_block}")

    if master is not None and conv is not None:
        # sentence with 2 numbers (allowed)
        parts.append(f"Hiện tại điểm tổng hợp khoảng {master:.1f}/10 và độ tin cậy {conv:.1f}/10, nên đây là {gate_vn}.")
    else:
        parts.append(f"Hiện tại hệ thống đang ở {gate_vn}.")

    status_one_liner = _build_status_one_liner(blockers, gate_status, next_step)
    parts.append(_clean_sentence(status_one_liner))

    # --- 3) Empathy (contextual) ---
    if _should_empathize(is_holding, pnl_pct, action):
        parts.append("Nếu anh/chị đang chịu áp lực vì vị thế chưa thuận lợi, điều này là bình thường; ưu tiên lúc này là kỷ luật và bảo vệ vốn.")

    # --- 4) Action A/B ---
    act_vn = _vn_action(action)
    inten_vn = _vn_intensity(urgency)

    # Defensive A-branch: align with system action but keep co-pilot tone
    if action == "TRIM":
        a_phrase = f"{act_vn} ({inten_vn})" if inten_vn else act_vn
    elif action == "EXIT":
        a_phrase = "ưu tiên thoát vị thế"
    elif action in ("BUY", "ADD"):
        # Still keep agency shift; for safety branch we soften
        a_phrase = "quan sát thêm, chưa vội mở mua"
    else:
        a_phrase = "tiếp tục nắm giữ nhưng không gia tăng"

    # Guidance sentence from library
    guidance = ""
    lib_mode = "HOLDING" if is_holding else "FLAT"
    lib = ACTION_SENTENCES.get(lib_mode, {})
    lib_act = lib.get(action, {}) or lib.get("WAIT", {})
    if lib_act:
        guidance = lib_act.get(urgency, "") or lib_act.get("HIGH", "") or lib_act.get("ANY", "") or ""

    guidance = _clean_sentence(guidance)

    # Protect stop
    ps = _fmt_price_2dp_or_1dp(protect_stop if _safe_float(protect_stop) is not None else stop_suggest)
    protect_word = TERMS_MAP["plan_words"]["protect_stop"]
    if ps != "-":
        parts.append(
            f"Nếu ưu tiên an toàn, {a_phrase}. {guidance} "
            f"Mốc quan trọng: {protect_word} tại {ps}; kịch bản này sai nếu giá đóng cửa dưới mốc này."
        )
    else:
        parts.append(f"Nếu ưu tiên an toàn, {a_phrase}. {guidance}")

    # Reclaim zone (conditional B-branch)
    rz = _fmt_zone_1dp(reclaim_low, reclaim_high, single=reclaim_single)
    reclaim_word = TERMS_MAP["plan_words"]["reclaim_zone"]
    if rz != "-":
        parts.append(f"Nếu chấp nhận rủi ro để chờ xác nhận, chỉ nâng mức chủ động khi giá quay lại {reclaim_word} {rz} kèm tín hiệu đồng thuận (dòng tiền và cấu trúc giá).")

    # --- 5) Pre-mortem (invalidation) ---
    verb, inval = _select_invalidation_level(protect_stop, stop_suggest, reclaim_low)
    if inval is not None:
        inval_s = _fmt_price_2dp_or_1dp(inval)
        parts.append(f"Kịch bản này sai nếu giá {verb} {inval_s}.")
    else:
        parts.append("Kịch bản này sai nếu giá phá vỡ mốc quản trị rủi ro quan trọng.")

    # Final clean & ensure single paragraph
    paragraph = " ".join([_clean_sentence(p) for p in parts if _clean_sentence(p)])
    paragraph = re.sub(r"\s+", " ", paragraph).strip()

    # Final hard ban (avoid leakage)
    low = paragraph.lower()
    for tok in ["gate:", "plan:", "pass", "fail", "active", "watch", "avoid", "trim", "structure", "flow"]:
        # Note: allow Vietnamese 'cấu trúc'/'dòng tiền' — only ban English tokens.
        if tok in ("structure", "flow"):
            continue
        if tok in low:
            paragraph = re.sub(re.escape(tok), "", paragraph, flags=re.IGNORECASE).strip()
            low = paragraph.lower()

    return paragraph
