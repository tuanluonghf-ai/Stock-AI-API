"""Rewrite-only layer for DNALine (Phase 2).

Contract
--------
- Engine remains the single source of truth for facts/labels.
- GPT is only allowed to polish wording for DNALine using a minimal payload:
    draft sentence + (ticker, dna_mapping, edge, discipline_rule) + hard constraints.
- Always validate; on failure fallback to draft (never crash).

This module must NOT parse the full AnalysisPack JSON.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import os
import re

from inception.core.validate import validate_dna_line


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _truthy_env(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def should_rewrite_dna(ctx: Optional[Dict[str, Any]] = None) -> bool:
    """Decide whether DNALine rewrite-only is enabled.

    Priority:
      1) ctx['enable_gpt_rewrite_dna'] if explicitly set
      2) env INCEPTION_GPT_REWRITE_DNA (default ON when key exists)
    """
    c = ctx if isinstance(ctx, dict) else {}
    if "enable_gpt_rewrite_dna" in c:
        return bool(c.get("enable_gpt_rewrite_dna"))
    return _truthy_env("INCEPTION_GPT_REWRITE_DNA", default="1")


def _normalize_one_sentence(text: str) -> str:
    t = re.sub(r"[\r\n\t]+", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    # Remove wrapping quotes if any
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
        t = t[1:-1].strip()
    return t


def _build_prompt(
    *,
    ticker: str,
    dna_mapping: str,
    edge: str,
    discipline_rule: str,
    banned_words: Optional[list[str]] = None,
    forbid_tokens: Optional[list[str]] = None,
    variant_pref: str = "BLEND",
) -> str:
    bw = [w for w in (banned_words or []) if str(w).strip()]
    ft = [w for w in (forbid_tokens or []) if str(w).strip()]

    # Keep anchors verbatim to prevent fact drift.
    # GPT is allowed to adjust connectors only.
    constraints = [
        "Output đúng 1 câu tiếng Việt, không xuống dòng, không gạch đầu dòng.",
        "Giữ nguyên 3 cụm FACTS dưới đây (copy đúng từng chữ, không được sửa nội dung facts).",
        "Không thêm dữ kiện mới, không suy diễn, không thêm mốc giá hoặc số liệu.",
        "Không nhắc MasterScore/Conviction hay các điểm số.",
        "Không dùng các từ thời điểm như: " + ", ".join([f"'{x}'" for x in bw]) if bw else "Không dùng ngôn ngữ thời điểm (đang/hiện tại/tín hiệu/hôm nay).",
        "Tránh mọi token sau: " + ", ".join([f"'{x}'" for x in ft]) if ft else "",
        "Giữ cấu trúc 3 mệnh đề phân tách bằng dấu ';' như skeleton: '{TICKER} là kiểu {DNA_MAPPING}; lợi thế nằm ở {EDGE}; đổi lại, điều kiện bắt buộc là {DISCIPLINE_RULE}.'",
        f"Biến thể giọng điệu: {variant_pref} (B: gọn tự tin; C: kỷ luật; BLEND: pha trộn).",
        "Chỉ trả về câu cuối cùng, không giải thích.",
    ]
    constraints = [c for c in constraints if c.strip()]

    prompt = f"""Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP. Nhiệm vụ: viết lại 1 câu 'Stock DNA' theo tone Private Banker nhưng kỷ luật.

FACTS (phải giữ nguyên y nguyên, copy đúng):
- TICKER: {ticker}
- DNA_MAPPING: {dna_mapping}
- EDGE: {edge}
- DISCIPLINE_RULE: {discipline_rule}

RÀNG BUỘC:
- {' '.join(constraints)}

Hãy viết câu theo đúng skeleton (chỉ được chỉnh từ nối, nhịp câu), và đảm bảo tự nhiên, không khẩu hiệu.
"""
    return prompt


def rewrite_dna_line_only(
    *,
    draft_pack: Dict[str, Any],
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rewrite-only DNALine using OpenAI. Always validate & fallback.

    Expects draft_pack schema 'NarrativeDraftPack.v1' (or compatible).
    Returns the updated draft_pack (in-place safe copy style).
    """
    ndp = draft_pack if isinstance(draft_pack, dict) else {}
    dna = (ndp.get("dna") or {}) if isinstance(ndp.get("dna"), dict) else {}
    line_draft = _as_text(dna.get("line_draft")).strip()
    facts = (dna.get("facts") or {}) if isinstance(dna.get("facts"), dict) else {}

    # Default: final == draft
    dna.setdefault("line_final", line_draft)
    dna.setdefault("rewrite_used", False)
    dna.setdefault("rewrite_fail_issues", [])
    dna.setdefault("rewrite_model", "")
    dna.setdefault("rewrite_temperature", None)

    if not line_draft:
        ndp["dna"] = dna
        return ndp

    if not should_rewrite_dna(ctx):
        ndp["dna"] = dna
        return ndp

    # Lazy import so the app can run without OpenAI SDK
    try:
        from inception.infra.llm_client import call_openai_chat, get_openai_api_key
    except Exception:
        ndp["dna"] = dna
        return ndp

    if not get_openai_api_key():
        ndp["dna"] = dna
        return ndp

    ticker = _as_text(facts.get("ticker") or facts.get("TICKER") or "").strip().upper() or _as_text(facts.get("ticker")).upper()
    dna_mapping = _as_text(facts.get("dna_mapping") or "").strip()
    edge = _as_text(facts.get("edge") or "").strip()
    discipline_rule = _as_text(facts.get("discipline_rule") or "").strip()

    # Minimal payload only (facts + constraints)
    prompt = _build_prompt(
        ticker=ticker or "—",
        dna_mapping=dna_mapping or _as_text(facts.get("DNA_MAPPING")),
        edge=edge or _as_text(facts.get("EDGE")),
        discipline_rule=discipline_rule or _as_text(facts.get("DISCIPLINE_RULE")),
        banned_words=dna.get("banned_words") if isinstance(dna.get("banned_words"), list) else None,
        forbid_tokens=dna.get("forbid_tokens") if isinstance(dna.get("forbid_tokens"), list) else None,
        variant_pref=_as_text(dna.get("variant_pref") or "BLEND") or "BLEND",
    )

    model = os.getenv("INCEPTION_GPT_MODEL", "gpt-4o")
    temperature = float(os.getenv("INCEPTION_GPT_TEMP_DNA", "0.4") or 0.4)

    try:
        out = call_openai_chat(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=120,
            system="Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP. Viết gọn, kỷ luật, không suy diễn.",
        )
    except Exception:
        out = ""

    out = _normalize_one_sentence(out)

    v = validate_dna_line(
        out,
        banned_words=dna.get("banned_words"),
        forbid_tokens=dna.get("forbid_tokens"),
        max_numbers_per_sentence=int(dna.get("max_numbers_per_sentence") or 2),
    )

    if isinstance(v, dict) and v.get("ok"):
        dna["line_final"] = out
        dna["rewrite_used"] = True
        dna["rewrite_fail_issues"] = []
        dna["rewrite_model"] = model
        dna["rewrite_temperature"] = temperature
    else:
        # fallback to draft
        dna["line_final"] = line_draft
        dna["rewrite_used"] = False
        dna["rewrite_fail_issues"] = (v or {}).get("issues") if isinstance(v, dict) else ["VALIDATION_FAIL"]

    ndp["dna"] = dna
    return ndp
