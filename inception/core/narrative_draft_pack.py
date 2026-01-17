"""Deterministic narrative draft pack builder (v1).

Engine facts-first contract
--------------------------
This pack provides short, deterministic Vietnamese draft lines for:
  - DNA (long-run trait + edge + rule-of-engagement)
  - Current Status (context labels; may reference score elsewhere but not required)
  - Trade Plan (primary plan type/state + top blockers)

No LLM calls, no trading-logic changes.

Design rules
------------
- DNA must not use time-sensitive verbs (đang/hiện tại/tín hiệu/hôm nay).
- DNA must not mention MasterScore/Conviction (those belong to Status/UI).
- DNA always includes a discipline clause (rule-of-engagement).
- Output must remain safe if upstream packs are missing ("-" placeholders).

Note: The UI may optionally pass ctx['dna_variant_pref'] in {"B","C","BLEND"}
for deterministic style preference. If missing, we pick stably by ticker+class.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import hashlib
import re

import numpy as np

from inception.core.helpers import _safe_float, _safe_text


# ---- Hard constraints (DNA) ----
_BANNED_DNA_WORDS = ["đang", "hiện tại", "tín hiệu", "hôm nay"]
_FORBID_MS_CV_TOKENS = [
    "masterscore",
    "conviction",
    "điểm tổng hợp",
    "điểm tin cậy",
    "ms",
    "cv",
]
_TRADEOFF_KEYWORDS = [
    "biên",
    "biên độ",
    "payoff",
    "opportunity cost",
    "đáng để",
    "không đủ biên",
    "biên hạn chế",
]


def _as_text(x: Any, fallback: str = "-") -> str:
    s = _safe_text(x).strip()
    return s if s else fallback


def _has_tradeoff_cue(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _TRADEOFF_KEYWORDS)


def _primary_persona_from_pack(ap: Dict[str, Any]) -> str:
    pack = ap.get("InvestorMappingPack") if isinstance(ap.get("InvestorMappingPack"), dict) else {}
    src = None
    for key in ("Personas", "PersonaMatch", "Compatibility", "personas", "persona_match", "compatibility"):
        if key in pack:
            src = pack.get(key)
            break
    items: list[Dict[str, Any]] = []
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

    def _score(item: Dict[str, Any]) -> float:
        return _safe_float(item.get("score_10"), default=np.nan)

    def _name(item: Dict[str, Any]) -> str:
        return _as_text(item.get("name") or item.get("Name") or item.get("Persona"), fallback="").strip()

    items.sort(key=lambda it: (-( _score(it) if _score(it) == _score(it) else -1e9), _name(it)))
    return _name(items[0]) if items else ""


def _stable_pick_variant(ticker: str, class_primary: str, style_primary: str, risk_regime: str) -> str:
    key = f"{(ticker or '').strip().upper()}|{(class_primary or '').strip()}|{(style_primary or '').strip()}|{(risk_regime or '').strip()}|DNA_VARIANT"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    pick = int(h, 16) % 3
    return "B" if pick == 0 else ("C" if pick == 1 else "BLEND")


def _dna_mapping_edge_discipline(
    *,
    class_primary: str,
    style_primary: str,
    risk_regime: str,
    modifiers: list[str],
    variant: str,
) -> Tuple[str, str, str]:
    """Return (dna_mapping, edge, discipline_rule) for DNALineDraft.

    The content is *long-run* and deterministic.
    """

    cls = (class_primary or "").strip()
    sty = (style_primary or "").strip()
    rsk = (risk_regime or "").strip()
    mods = set([str(m).strip().upper() for m in (modifiers or []) if str(m).strip()])

    v = (variant or "").strip().upper()
    if v not in ("B", "C", "BLEND"):
        v = "BLEND"

    # ---- Base rules by class (8 buckets used in CharacterPack) ----
    if "ILLIQ" in mods or "Illiquid" in cls:
        dna_mapping = "cổ phiếu thanh khoản mỏng, dễ nhiễu"
        edge = "có thể cho nhịp lợi nhuận nhanh khi dòng tiền vào đúng lúc"
        discipline = "chỉ vào nhỏ bằng lệnh giới hạn; hễ mất thanh khoản hoặc gãy hỗ trợ thì rút ngay, không bình quân giá"

    elif "GAP" in mods or "Gap" in cls or "Event" in cls:
        dna_mapping = "cổ phiếu nhạy tin, dễ nhảy giá"
        edge = "thường mở biên lợi nhuận mạnh khi catalyst rõ ràng"
        discipline = "chỉ tham gia khi có xác nhận; luôn đặt mốc bảo vệ và quản trị tỷ trọng chặt, đặc biệt với rủi ro qua đêm"

    elif cls == "Smooth Trend" or ("Trend" in sty and "Aggressive" not in cls):
        dna_mapping = "cổ phiếu xu hướng mượt"
        edge = "dễ tối ưu bằng chiến lược đi theo xu hướng"
        discipline = "mua theo nhịp điều chỉnh, vào từng phần; nếu mất cấu trúc xu hướng thì giảm/thoát, tránh mua đuổi"

    elif cls == "Aggressive Trend":
        dna_mapping = "cổ phiếu xu hướng mạnh nhưng rung lắc lớn"
        edge = "có thể kéo lợi nhuận nhanh khi thuận nhịp"
        discipline = "quản trị vị thế theo cấu trúc, không all-in; hễ gãy vùng bảo vệ thì giảm/cắt dứt khoát, tránh FOMO"

    elif cls == "Momentum Trend" or "Momentum" in sty:
        dna_mapping = "cổ phiếu thiên bứt phá theo động lượng"
        edge = "thưởng lớn khi bứt phá thật và có dòng tiền xác nhận"
        discipline = "chỉ mua sau xác nhận vượt cản; bứt phá thất bại thì hạ nhanh, không giữ trong vùng nhiễu"

    elif cls == "Range / Mean-Reversion (Stable)" or ("Range" in sty and "Volatile" not in cls):
        dna_mapping = "cổ phiếu đi ngang ổn định, hồi quy tốt"
        edge = "tối ưu bằng chiến lược mua thấp – bán cao theo biên"
        discipline = "mua gần hỗ trợ, bán gần kháng cự; không đuổi breakout khi chưa xác nhận, kỷ luật theo biên"

    elif cls == "Volatile Range":
        dna_mapping = "cổ phiếu đi ngang nhưng biến động mạnh"
        edge = "biên dao động tạo cơ hội trading ngắn hạn"
        discipline = "giảm quy mô và ưu tiên phản ứng nhanh; chỉ tham gia khi vùng giá rõ ràng, mất biên là đứng ngoài"

    else:  # Mixed / Choppy Trader (default)
        dna_mapping = "cổ phiếu tính khí pha trộn, dễ quét hai đầu"
        edge = "có thể xuất hiện nhịp ngắn nhưng khó gồng"
        discipline = "chỉ hành động khi có xác nhận đa lớp; cấu trúc không rõ thì ưu tiên quan sát, tránh kéo dài vị thế"

    # ---- Variant shaping (B/C/Blend) without changing facts ----
    if v == "B":
        # VIP tone: slightly more confident phrasing
        if "dòng tiền" not in edge:
            edge = edge.replace("thường", "thường")  # no-op; keep deterministic
        discipline = discipline.replace("chỉ", "ưu tiên") if discipline.startswith("chỉ") else discipline
    elif v == "C":
        # Rule-of-engagement: sharper reward/punish framing (still long-run)
        discipline = discipline.replace("tránh", "tuyệt đối tránh") if "tránh" in discipline else discipline
    else:
        # Blend keeps base text as-is
        pass

    # ---- Risk regime adjustment (keep short, no numbers) ----
    if rsk.strip().upper() == "HIGH" or ("HIVOL" in mods) or ("CHOPVOL" in mods):
        if "giảm quy mô" not in discipline:
            discipline = discipline + "; ưu tiên giảm quy mô và hạn chế đòn bẩy"

    if "DEF" in mods:
        if "bảo toàn vốn" not in discipline:
            discipline = discipline + "; ưu tiên bảo toàn vốn, chỉ chủ động hơn khi có xác nhận lại cấu trúc"

    # Cleanups: avoid multiple spaces, avoid accidental banned words
    dna_mapping = re.sub(r"\s+", " ", dna_mapping).strip()
    edge = re.sub(r"\s+", " ", edge).strip()
    discipline = re.sub(r"\s+", " ", discipline).strip()

    # Replace any banned tokens if they accidentally appear
    for w in _BANNED_DNA_WORDS:
        dna_mapping = dna_mapping.replace(w, "")
        edge = edge.replace(w, "")
        discipline = discipline.replace(w, "")

    dna_mapping = re.sub(r"\s+", " ", dna_mapping).strip()
    edge = re.sub(r"\s+", " ", edge).strip()
    discipline = re.sub(r"\s+", " ", discipline).strip()

    # Ensure discipline exists
    if not discipline:
        discipline = "vào từng phần theo vùng; luôn có mốc bảo vệ; và tránh mua đuổi"

    return dna_mapping, edge, discipline


def _dna_line(ap: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    dna = ap.get("DNAPack") or {}
    dna = dna if isinstance(dna, dict) else {}

    ticker = _as_text(ap.get("Ticker"), fallback="—").upper()
    class_primary = _as_text(dna.get("class_primary") or ap.get("CharacterClass"))
    style_primary = _as_text(dna.get("style_primary"))
    risk_regime = _as_text(dna.get("risk_regime"))
    modifiers = dna.get("modifiers") if isinstance(dna.get("modifiers"), list) else []

    pref = _as_text((ctx or {}).get("dna_variant_pref"), fallback="").upper()
    variant = pref if pref in ("B", "C", "BLEND") else _stable_pick_variant(ticker, class_primary, style_primary, risk_regime)

    if class_primary == "-" and style_primary == "-":
        # Keep deterministic placeholder; still respects DNA boundary.
        line = f"{ticker} là kiểu cổ phiếu chưa đủ dữ liệu DNA; lợi thế nằm ở tính ổn định dữ liệu. Đổi lại, điều kiện bắt buộc là chỉ hành động khi dữ liệu đầy đủ."
        facts = {
            "ticker": ticker,
            "dna_mapping": "-",
            "edge": "-",
            "discipline_rule": "chỉ hành động khi dữ liệu đầy đủ",
            "variant": variant,
            "source": "fallback",
        }
        return line, facts

    dna_mapping, edge, discipline = _dna_mapping_edge_discipline(
        class_primary=class_primary,
        style_primary=style_primary,
        risk_regime=risk_regime,
        modifiers=[str(x) for x in (modifiers or [])],
        variant=variant,
    )

    # Strict skeleton (single line; avoid explicit multi-sentence when possible)
    line = f"{ticker} là kiểu {dna_mapping}; lợi thế nằm ở {edge}; đổi lại, điều kiện bắt buộc là {discipline}."

    facts = {
        "ticker": ticker,
        "class_primary": class_primary,
        "style_primary": style_primary,
        "risk_regime": risk_regime,
        "modifiers": list(modifiers) if isinstance(modifiers, list) else [],
        "dna_mapping": dna_mapping,
        "edge": edge,
        "discipline_rule": discipline,
        "variant": variant,
        "source": "DNAPack",
    }

    return line, facts


_STATUS_MAP = {
    "ABOVE_MA200": "trên MA200 (xu hướng dài hạn thuận)",
    "HEALTHY_PULLBACK": "trên MA200 nhưng đang điều chỉnh lành mạnh",
    "POSITIVE_PULLBACK_ZONE": "gần MA200, vùng dễ có nhịp hồi nếu cấu trúc giữ được",
    "NEGATIVE_TREND_BROKEN": "dưới MA200 (xu hướng dài hạn chưa thuận)",
    "INTACT": "cấu trúc còn giữ",
    "CEILING": "đang vướng vùng cản",
    "BROKEN": "cấu trúc bị suy yếu",
    "BREAKOUT_READY": "động lượng tích cực, có thể tiến tới bứt phá",
    "MOMO_BUILDING": "động lượng đang xây dựng",
    "MOMO_WEAK": "động lượng yếu",
    "CONFIRM": "dòng tiền ủng hộ",
    "FADE": "dòng tiền suy yếu",
    "NEUTRAL": "dòng tiền trung tính",
    "CLIMAX_RISK": "dòng tiền cao nhưng có dấu hiệu quá đà",
    "LOW": "rủi ro thấp",
    "MID": "rủi ro trung bình",
    "HIGH": "rủi ro cao",
}


def _status_line(ap: Dict[str, Any]) -> str:
    sp = ap.get("StatusPack") or {}
    sp = sp if isinstance(sp, dict) else {}
    tech = sp.get("technicals") or {}
    tech = tech if isinstance(tech, dict) else {}

    ma200 = _as_text(tech.get("ma200_context"))
    struct = _as_text(tech.get("structure_context"))
    momo = _as_text(tech.get("momentum_context"))
    vol = _as_text(tech.get("volume_context"))
    risk = _as_text(tech.get("risk_context"))

    bits = []
    if ma200 != "-":
        bits.append(_STATUS_MAP.get(ma200, ma200))
    if struct != "-":
        bits.append(_STATUS_MAP.get(struct, struct))
    if momo != "-":
        bits.append(_STATUS_MAP.get(momo, momo))
    if vol != "-":
        bits.append(_STATUS_MAP.get(vol, vol))

    s1 = "; ".join([b for b in bits if b])
    if not s1:
        s1 = "Bối cảnh kỹ thuật chưa đủ nhãn để diễn giải ngắn."

    r_txt = _STATUS_MAP.get(risk, risk)
    if r_txt and r_txt != "-":
        return f"{s1}. Mức {r_txt}."
    return f"{s1}."


_PLAN_TYPE_MAP = {
    "PULLBACK": "Pullback",
    "BREAKOUT": "Breakout",
    "MEAN_REV": "Mean-reversion",
    "RECLAIM": "Reclaim",
    "DEFENSIVE": "Defensive",
}


def _plan_line(ap: Dict[str, Any]) -> str:
    tpp = ap.get("TradePlanPack") or {}
    tpp = tpp if isinstance(tpp, dict) else {}
    primary = tpp.get("plan_primary") or {}
    primary = primary if isinstance(primary, dict) else {}

    ptype = _as_text(primary.get("type"))
    state = _as_text(primary.get("state")).upper()
    ptype_label = _PLAN_TYPE_MAP.get(ptype.upper(), ptype if ptype != "-" else "-")

    # Sentence 1: plan focus (+ optional blocker) in a single sentence.
    if ptype_label == "-":
        s1 = "Trade Plan chưa đủ dữ liệu để chốt trọng tâm"
    else:
        s1 = f"Trade Plan ưu tiên {ptype_label} (trạng thái {state})"

    reasons = primary.get("fail_reasons") or []
    top_code = ""
    if isinstance(reasons, list) and reasons:
        top = reasons[0] if isinstance(reasons[0], dict) else {}
        top_code = _as_text(top.get("code"), fallback="").strip()

    if top_code:
        s1 = f"{s1}, điểm nghẽn chính: {top_code}"
    s1 = s1.rstrip(".") + "."

    # Sentence 2 (optional): RR/Risk discipline, keep numbers minimal.
    ps = ap.get("PrimarySetup") or {}
    ps = ps if isinstance(ps, dict) else {}
    rr = _safe_float(ps.get("RR"))
    risk_pct = _safe_float(ps.get("RiskPct"))

    rr_txt = f"{rr:.1f}" if np.isfinite(rr) else "-"
    risk_txt = f"{risk_pct:.1f}%" if np.isfinite(risk_pct) else "-"

    s2 = ""
    if rr_txt != "-" and risk_txt != "-":
        s2 = f"Trọng tâm kỷ luật: RR {rr_txt}; rủi ro {risk_txt}."
    elif rr_txt != "-":
        s2 = f"Trọng tâm kỷ luật: RR {rr_txt}."
    elif risk_txt != "-":
        s2 = f"Trọng tâm kỷ luật: rủi ro {risk_txt}."

    return f"{s1} {s2}".strip() if s2 else s1


def _empathy_allowed(ap: Dict[str, Any]) -> bool:
    pos = ap.get("PositionStatePack") or {}
    pos = pos if isinstance(pos, dict) else {}
    pnl_pct = _safe_float(pos.get("pnl_pct"), default=np.nan)
    underwater = bool(np.isfinite(pnl_pct) and float(pnl_pct) <= -15.0)

    dp = ap.get("DecisionPack") or {}
    dp = dp if isinstance(dp, dict) else {}
    action = _as_text(dp.get("action")).upper()
    return bool(underwater or action in ("TRIM", "EXIT"))


def build_narrative_draft_pack_v1(analysis_pack: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    ctxd = ctx if isinstance(ctx, dict) else {}

    dna_line, dna_facts = _dna_line(ap, ctxd)
    status_line = _status_line(ap)
    plan_line = _plan_line(ap)

    payoff = ap.get("payoff") if isinstance(ap.get("payoff"), dict) else {}
    payoff_tier = _as_text(payoff.get("payoff_tier"), fallback="").upper()
    persona = _primary_persona_from_pack(ap)
    persona_u = persona.strip().upper()
    if payoff_tier in {"LOW", "MEDIUM"} and persona_u in {"SPECULATOR", "ALPHAHUNTER", "COMPOUNDER"}:
        combined = f"{dna_line} {status_line} {plan_line}"
        if not _has_tradeoff_cue(combined):
            anchor = (
                f"Biên độ khai thác thuộc nhóm {payoff_tier}, vì vậy ưu tiên kỷ luật chọn điểm và cân nhắc chi phí cơ hội."
            )
            if dna_line and dna_line[-1] not in ".!?":
                dna_line = dna_line.rstrip() + "."
            dna_line = f"{dna_line} {anchor}".strip()

    # Minimal facts for rewrite-only payloads (Phase 3)
    sp = ap.get("StatusPack") or {}
    sp = sp if isinstance(sp, dict) else {}
    tech = sp.get("technicals") or {}
    tech = tech if isinstance(tech, dict) else {}

    status_facts = {
        "ma200_context": _as_text(tech.get("ma200_context")),
        "structure_context": _as_text(tech.get("structure_context")),
        "momentum_context": _as_text(tech.get("momentum_context")),
        "volume_context": _as_text(tech.get("volume_context")),
        "risk_context": _as_text(tech.get("risk_context")),
    }

    tpp = ap.get("TradePlanPack") or {}
    tpp = tpp if isinstance(tpp, dict) else {}
    primary = tpp.get("plan_primary") or {}
    primary = primary if isinstance(primary, dict) else {}
    reasons = primary.get("fail_reasons") or []
    top_code = ""
    if isinstance(reasons, list) and reasons:
        top = reasons[0] if isinstance(reasons[0], dict) else {}
        top_code = _as_text(top.get("code"), fallback="").strip()

    ps = ap.get("PrimarySetup") or {}
    ps = ps if isinstance(ps, dict) else {}
    rr = _safe_float(ps.get("RR"))
    risk_pct = _safe_float(ps.get("RiskPct"))
    rr_txt = f"{rr:.1f}" if np.isfinite(rr) else "-"
    risk_txt = f"{risk_pct:.1f}%" if np.isfinite(risk_pct) else "-"
    req_phrases = []
    if rr_txt != "-":
        req_phrases.append(f"RR {rr_txt}")
    if risk_txt != "-":
        req_phrases.append(f"rủi ro {risk_txt}")

    plan_facts = {
        "type": _as_text(primary.get("type")),
        "state": _as_text(primary.get("state")).upper(),
        "blocker_code": top_code,
        "rr_text": rr_txt,
        "risk_text": risk_txt,
        "required_phrases": req_phrases,
    }

    return {
        "schema": "NarrativeDraftPack.v1",
        "dna": {
            "line_draft": dna_line,
            "facts": dna_facts,
            "variant_pref": _as_text(ctxd.get("dna_variant_pref"), fallback="AUTO"),
            "banned_words": list(_BANNED_DNA_WORDS),
            "forbid_tokens": list(_FORBID_MS_CV_TOKENS),
            "max_numbers_per_sentence": 2,
        },
        "status": {
            "line_draft": status_line,
            "facts": status_facts,
            "max_sentences": 2,
            "max_numbers_per_sentence": 2,
        },
        "plan": {
            "line_draft": plan_line,
            "facts": plan_facts,
            "max_sentences": 2,
            "max_numbers_per_sentence": 2,
            "forbid_price_levels": True,
        },
        "comms_policy": {
            "max_numbers_per_sentence": 2,
            "no_slogans": True,
            "empathy_allowed": _empathy_allowed(ap),
        },
    }
