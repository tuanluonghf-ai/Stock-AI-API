"""Rewrite-only layer for StatusLine and PlanLine (Phase 3).

Contract
--------
- Engine remains the single source of truth for facts/labels.
- GPT is only allowed to polish wording for short draft lines using a minimal payload.
- Always validate; on failure fallback to draft (never crash).

This module must NOT parse the full AnalysisPack JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import os
import re

from inception.core.validate import validate_contains_phrases, validate_short_text


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


def should_rewrite_status(ctx: Optional[Dict[str, Any]] = None) -> bool:
    """Enable flag for StatusLine rewrite-only.

    Priority:
      1) ctx['enable_gpt_rewrite_status'] if set
      2) env INCEPTION_GPT_REWRITE_STATUS (default ON)
    """
    c = ctx if isinstance(ctx, dict) else {}
    if "enable_gpt_rewrite_status" in c:
        return bool(c.get("enable_gpt_rewrite_status"))
    return _truthy_env("INCEPTION_GPT_REWRITE_STATUS", default="1")


def should_rewrite_plan(ctx: Optional[Dict[str, Any]] = None) -> bool:
    """Enable flag for PlanLine rewrite-only.

    Priority:
      1) ctx['enable_gpt_rewrite_plan'] if set
      2) env INCEPTION_GPT_REWRITE_PLAN (default ON)
    """
    c = ctx if isinstance(ctx, dict) else {}
    if "enable_gpt_rewrite_plan" in c:
        return bool(c.get("enable_gpt_rewrite_plan"))
    return _truthy_env("INCEPTION_GPT_REWRITE_PLAN", default="1")


def _normalize_text(text: str) -> str:
    t = re.sub(r"[\r\n\t]+", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
        t = t[1:-1].strip()
    return t


_NUM_RE = re.compile(r"\d+(?:[\.,]\d+)?")


def _has_suspect_price_level(text: str) -> bool:
    """Heuristic: reject added price levels.

    Acceptable numbers:
      - RR numbers (typically <10) when preceded by 'RR'
      - Percent numbers when followed by '%'

    Any other number >= 10 is treated as a likely price level and triggers rejection.
    """
    t = text or ""
    for m in _NUM_RE.finditer(t):
        raw = m.group(0)
        try:
            val = float(raw.replace(",", "."))
        except Exception:
            continue

        if val < 10:
            continue

        # allow percentages
        after = t[m.end() : m.end() + 1]
        if after == "%":
            continue

        # allow RR numbers if preceded by 'RR'
        before = t[max(0, m.start() - 4) : m.start()].upper()
        if "RR" in before:
            continue

        return True

    return False


def _build_status_prompt(*, draft: str, facts: Dict[str, Any], max_sentences: int, max_numbers: int) -> str:
    f = facts or {}
    prompt = f"""Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP.

Nhiệm vụ: viết lại 1–2 câu 'Current Status' cho mượt, gọn, dễ hiểu.

DRAFT (đã đúng facts, chỉ được chỉnh văn phong):
{draft}

FACTS/LABELS (chỉ để bạn giữ đúng ý, không được thêm dữ kiện mới):
- MA200: {f.get('ma200_context','-')}
- Structure: {f.get('structure_context','-')}
- Momentum: {f.get('momentum_context','-')}
- Volume: {f.get('volume_context','-')}
- Risk: {f.get('risk_context','-')}

RÀNG BUỘC:
- Không thêm mốc giá, không thêm số liệu mới.
- Không suy diễn ngoài draft.
- Không xuống dòng, không gạch đầu dòng.
- Tối đa {max_sentences} câu; mỗi câu tối đa {max_numbers} con số.
- Chỉ trả về đoạn cuối cùng, không giải thích.
"""
    return prompt


def _build_plan_prompt(
    *,
    draft: str,
    facts: Dict[str, Any],
    max_sentences: int,
    max_numbers: int,
) -> str:
    f = facts or {}
    req = f.get("required_phrases") or []
    if not isinstance(req, list):
        req = []
    req_txt = "; ".join([str(x) for x in req if str(x).strip()]) or "(none)"

    prompt = f"""Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP.

Nhiệm vụ: viết lại 1–2 câu 'Trade Plan (tóm tắt)' cho gọn, kỷ luật.

DRAFT (đã đúng facts, chỉ được chỉnh văn phong):
{draft}

FACTS (không được thay đổi nội dung, không được thêm dữ kiện mới):
- Plan type: {f.get('type','-')}
- Plan state: {f.get('state','-')}
- Blocker code: {f.get('blocker_code','-')}
- RR text: {f.get('rr_text','-')}
- Risk text: {f.get('risk_text','-')}

RÀNG BUỘC:
- Không thêm mốc giá (Entry/Stop/TP...), không thêm vùng giá.
- Không suy diễn ngoài draft.
- Không xuống dòng, không gạch đầu dòng.
- Tối đa {max_sentences} câu; mỗi câu tối đa {max_numbers} con số.
- Bắt buộc giữ nguyên các cụm sau nếu có trong draft: {req_txt}
- Chỉ trả về đoạn cuối cùng, không giải thích.
"""
    return prompt


def rewrite_status_line_only(*, draft_pack: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ndp = draft_pack if isinstance(draft_pack, dict) else {}
    status = (ndp.get("status") or {}) if isinstance(ndp.get("status"), dict) else {}
    line_draft = _as_text(status.get("line_draft")).strip()
    facts = (status.get("facts") or {}) if isinstance(status.get("facts"), dict) else {}

    status.setdefault("line_final", line_draft)
    status.setdefault("rewrite_used", False)
    status.setdefault("rewrite_fail_issues", [])
    status.setdefault("rewrite_model", "")
    status.setdefault("rewrite_temperature", None)

    if not line_draft:
        ndp["status"] = status
        return ndp

    if not should_rewrite_status(ctx):
        ndp["status"] = status
        return ndp

    try:
        from inception.infra.llm_client import call_openai_chat, get_openai_api_key
    except Exception:
        ndp["status"] = status
        return ndp

    if not get_openai_api_key():
        ndp["status"] = status
        return ndp

    model = os.getenv("INCEPTION_GPT_MODEL", "gpt-4o")
    temperature = float(os.getenv("INCEPTION_GPT_TEMP_STATUS", "0.35") or 0.35)
    max_sentences = int(status.get("max_sentences") or 2)
    max_numbers = int(status.get("max_numbers_per_sentence") or 2)

    prompt = _build_status_prompt(draft=line_draft, facts=facts, max_sentences=max_sentences, max_numbers=max_numbers)

    try:
        out = call_openai_chat(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=160,
            system="Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP. Viết gọn, rủi ro rõ, không suy diễn.",
        )
    except Exception:
        out = ""

    out = _normalize_text(out)

    v1 = validate_short_text(out, max_sentences=max_sentences, max_numbers_per_sentence=max_numbers, forbid_bullets=True)
    if isinstance(v1, dict) and v1.get("ok"):
        status["line_final"] = out
        status["rewrite_used"] = True
        status["rewrite_fail_issues"] = []
        status["rewrite_model"] = model
        status["rewrite_temperature"] = temperature
    else:
        status["line_final"] = line_draft
        status["rewrite_used"] = False
        status["rewrite_fail_issues"] = (v1 or {}).get("issues") if isinstance(v1, dict) else ["VALIDATION_FAIL"]

    ndp["status"] = status
    return ndp


def rewrite_plan_line_only(*, draft_pack: Dict[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ndp = draft_pack if isinstance(draft_pack, dict) else {}
    plan = (ndp.get("plan") or {}) if isinstance(ndp.get("plan"), dict) else {}
    line_draft = _as_text(plan.get("line_draft")).strip()
    facts = (plan.get("facts") or {}) if isinstance(plan.get("facts"), dict) else {}

    plan.setdefault("line_final", line_draft)
    plan.setdefault("rewrite_used", False)
    plan.setdefault("rewrite_fail_issues", [])
    plan.setdefault("rewrite_model", "")
    plan.setdefault("rewrite_temperature", None)

    if not line_draft:
        ndp["plan"] = plan
        return ndp

    if not should_rewrite_plan(ctx):
        ndp["plan"] = plan
        return ndp

    try:
        from inception.infra.llm_client import call_openai_chat, get_openai_api_key
    except Exception:
        ndp["plan"] = plan
        return ndp

    if not get_openai_api_key():
        ndp["plan"] = plan
        return ndp

    model = os.getenv("INCEPTION_GPT_MODEL", "gpt-4o")
    temperature = float(os.getenv("INCEPTION_GPT_TEMP_PLAN", "0.30") or 0.30)
    max_sentences = int(plan.get("max_sentences") or 2)
    max_numbers = int(plan.get("max_numbers_per_sentence") or 2)
    forbid_prices = bool(plan.get("forbid_price_levels", True))

    prompt = _build_plan_prompt(draft=line_draft, facts=facts, max_sentences=max_sentences, max_numbers=max_numbers)

    try:
        out = call_openai_chat(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=200,
            system="Bạn là chuyên gia quản lý tài sản tư vấn khách hàng VIP. Viết kỷ luật, không thêm mốc giá, không suy diễn.",
        )
    except Exception:
        out = ""

    out = _normalize_text(out)

    issues: List[str] = []
    v1 = validate_short_text(out, max_sentences=max_sentences, max_numbers_per_sentence=max_numbers, forbid_bullets=True)
    if not (isinstance(v1, dict) and v1.get("ok")):
        issues.extend((v1 or {}).get("issues") if isinstance(v1, dict) else ["VALIDATION_FAIL"])

    req = facts.get("required_phrases") if isinstance(facts.get("required_phrases"), list) else []
    if req:
        v2 = validate_contains_phrases(out, required_phrases=req)
        if not (isinstance(v2, dict) and v2.get("ok")):
            issues.extend((v2 or {}).get("issues") if isinstance(v2, dict) else ["MISSING_REQUIRED_PHRASE"])

    if forbid_prices and _has_suspect_price_level(out):
        issues.append("SUSPECT_PRICE_LEVEL")

    if not issues:
        plan["line_final"] = out
        plan["rewrite_used"] = True
        plan["rewrite_fail_issues"] = []
        plan["rewrite_model"] = model
        plan["rewrite_temperature"] = temperature
    else:
        plan["line_final"] = line_draft
        plan["rewrite_used"] = False
        plan["rewrite_fail_issues"] = issues

    ndp["plan"] = plan
    return ndp
