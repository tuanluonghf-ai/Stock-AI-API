"""Report A–D builder (core).

Step 7 goals:
- Move deterministic report + GPT narrative + guard logic out of app.
- Keep modules self-contained (no ctx hook required).
- Ensure Section D is a strict copy from PrimarySetup.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import math
import os
import re

import numpy as np
import pandas as pd

from inception.core.helpers import _safe_float, _safe_text, json_sanitize
from inception.core.contracts import normalize_analysis_pack
from inception.core.phrase_bank import pick_phrase
from inception.infra.llm_client import call_openai_chat, get_openai_api_key


def _val_or_na(v: Any) -> str:
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and pd.isna(v):
            return "N/A"
        s = str(v).strip()
        return s if s else "N/A"
    except Exception:
        return "N/A"


def _disp(v: Any) -> str:
    """User-facing display for missing values."""
    s = _val_or_na(v)
    return "–" if s == "N/A" else s



def _as_dict(x: Any) -> Dict[str, Any]:
    """Defensive cast to dict.

    Upstream packs can occasionally carry scalars (e.g., int) due to
    incomplete normalization. Report generation must never crash on `.get()`.
    """
    return x if isinstance(x, dict) else {}


def _fmt_price(v: Any) -> str:
    x = _safe_float(v, default=np.nan)
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "N/A"


def _fmt_thousand(v: Any) -> str:
    x = _safe_float(v, default=np.nan)
    if pd.isna(x):
        return "N/A"
    # Display in thousands if it looks like a VND price
    try:
        if x >= 1000:
            return f"{x/1000.0:.1f}"
        return f"{x:.1f}"
    except Exception:
        return "N/A"


def _fmt_pct(v: Any) -> str:
    x = _safe_float(v, default=np.nan)
    if pd.isna(x):
        return "N/A"
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "N/A"


def _json_strict(obj: Any) -> str:
    """JSON dump that forbids NaN/Inf by replacing them with None."""
    sanitized = json_sanitize(obj)

    def fix(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x

    fixed = fix(sanitized)
    return json.dumps(fixed, ensure_ascii=False, allow_nan=False)


# ============================================================
# Guard Section D
# ============================================================

def _extract_d_block(text: str) -> str:
    m = re.search(
        r"(^|\n)\s*D\.\s*R[ủu]i\s*ro.*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return m.group(0) if m else text


def _grab_number(block: str, label_patterns: List[str]) -> float:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            s = (m.group(1) or "").strip()
            s = s.replace(",", "").replace("%", "")
            return _safe_float(s)
    return np.nan


def _grab_text(block: str, label_patterns: List[str]) -> Optional[str]:
    for pat in label_patterns:
        m = re.search(pat, block, flags=re.IGNORECASE)
        if m:
            return (m.group(1) or "").strip()
    return None


def _close_enough(a: float, b: float, tol: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol


def validate_section_d(text: str, primary: Dict[str, Any]) -> bool:
    block = _extract_d_block(text)

    got_risk = _grab_number(
        block,
        [
            r"Risk%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"R[ủu]i\s*ro\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"RiskPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )
    got_reward = _grab_number(
        block,
        [
            r"Reward%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"L[ợo]i\s*nhu[ậa]n\s*%\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"RewardPct\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )
    got_rr = _grab_number(
        block,
        [
            r"\bRR\b\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"R\s*[:/]\s*R\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        ],
    )
    got_prob = (
        _grab_text(
            block,
            [
                r"Confidence\s*\(\s*Tech\s*\)\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
                r"Probability\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
                r"X[aá]c\s*su[ấa]t\s*:\s*([A-Za-zÀ-ỹ0-9\-\s]+)",
            ],
        )
        or ""
    )

    exp_risk = _safe_float(primary.get("RiskPct"))
    exp_reward = _safe_float(primary.get("RewardPct"))
    exp_rr = _safe_float(primary.get("RR"))
    exp_prob = _safe_text(primary.get("Confidence (Tech)", primary.get("Probability"))).strip().lower()

    ok = True
    if pd.notna(exp_risk):
        ok &= _close_enough(got_risk, exp_risk, tol=0.05)
    if pd.notna(exp_reward):
        ok &= _close_enough(got_reward, exp_reward, tol=0.05)
    if pd.notna(exp_rr):
        ok &= _close_enough(got_rr, exp_rr, tol=0.05)

    gp = str(got_prob).strip().lower() if got_prob is not None else ""
    if exp_prob:
        ok &= (exp_prob in gp) if gp else False

    return bool(ok)


def call_gpt_with_guard(prompt: str, analysis_pack: Dict[str, Any], *, max_retry: int = 2, model: str = "gpt-4o") -> str:
    primary = (analysis_pack.get("PrimarySetup") or {})
    temps = [0.7, 0.2, 0.0]
    last_text = ""

    for i in range(min(max_retry + 1, len(temps))):
        temp = temps[i]
        extra = ""
        if i > 0:
            extra = f"""
SUA LOI BAT BUOC (chi sua muc D, giu nguyen cac muc khac):
Muc D dang sai so. Hay sua lai muc D bang cach COPY DUNG cac so sau (khong duoc tu tinh/uoc luong):
Risk%={primary.get('RiskPct')}, Reward%={primary.get('RewardPct')}, RR={primary.get('RR')}, Confidence (Tech)={primary.get('Confidence (Tech)', primary.get('Probability'))}.

Muc D bat buoc dung format 4 dong:
Risk%: <...>
Reward%: <...>
RR: <...>
Confidence (Tech): <...>
"""

        text = call_openai_chat(prompt=prompt + extra, model=model, temperature=temp)
        last_text = text

        if validate_section_d(text, primary):
            return text

    return last_text


# ============================================================
# Deterministic fallback report
# ============================================================

def _deterministic_report_ad(data: Dict[str, Any], note: str = "") -> str:
    """Build a deterministic A–D report (VIP-ready, VN) using 3-step advisory structure:
    Context → Assessment → Action.

    This function MUST remain deterministic and must not require any LLM/API key.
    """
    ap = normalize_analysis_pack((data or {}).get("AnalysisPack"))

    tick = str((data or {}).get("Ticker") or ap.get("Ticker") or "").strip().upper() or "N/A"

    pro = _as_dict(ap.get("ProTech"))
    ma = _as_dict(pro.get("MA"))
    rsi = _as_dict(pro.get("RSI"))
    macd = _as_dict(pro.get("MACD"))
    bias = _as_dict(pro.get("Bias"))
    vol = _as_dict(pro.get("Volume"))
    fib = _as_dict(ap.get("Fibonacci"))
    scen = _as_dict(ap.get("Scenario12"))
    ms = _as_dict(ap.get("MasterScore"))
    conv = ap.get("Conviction")

    primary = _as_dict(ap.get("PrimarySetup"))

    # Prefer TradePlanPack/DecisionPack for execution clarity
    tpp = _as_dict(ap.get("TradePlanPack"))
    pp = _as_dict(tpp.get("plan_primary"))
    comp = _as_dict(pp.get("plan_completeness") or tpp.get("plan_completeness"))
    comp_status = _safe_text(comp.get("status") or "").strip().upper() or "N/A"
    comp_missing = comp.get("missing") or []
    comp_missing = comp_missing if isinstance(comp_missing, list) else []
    comp_msg = _safe_text(comp.get("message") or "").strip()

    dp = _as_dict(ap.get("DecisionPack"))
    dp_act = _safe_text(dp.get("action") or "").strip().upper() or "N/A"
    dp_urg = _safe_text(dp.get("urgency") or "").strip().upper() or "N/A"

    setup_name = _val_or_na(primary.get("Name"))
    plan_state = _val_or_na(pp.get("state")) if pp else _val_or_na(primary.get("Status"))
    rr_txt = _val_or_na(primary.get("RR"))

    tags = primary.get("ReasonTags") or []
    tags_preview = "N/A"
    if isinstance(tags, list):
        tags_preview = ", ".join([str(x) for x in tags[:6]]) if tags else "N/A"

    risk = _val_or_na(primary.get("RiskPct"))
    reward = _val_or_na(primary.get("RewardPct"))
    must_rr = _val_or_na(primary.get("RR"))
    must_conf = _val_or_na(primary.get("Confidence (Tech)", primary.get("Probability")))

    # Fundamental (optional)
    fund = _as_dict(ap.get("Fundamental"))
    fund_rec = _disp(fund.get("Recommendation"))
    fund_target_k = _disp(_fmt_thousand(fund.get("TargetVND")))
    fund_upside = _disp(_fmt_pct(fund.get("UpsidePct")))

    # --- helper: deterministic action phrasing (no price levels)
    def _tech_action(ma_regime: str, align: str) -> str:
        ma_l = (ma_regime or "").lower()
        al_l = (align or "").lower()
        seed = f"{tick}|A|{ma_regime}|{align}"
        if ("bull" in ma_l or "up" in ma_l or "tăng" in ma_l) and ("bull" in al_l or "tăng" in al_l or "đồng thuận" in al_l):
            return pick_phrase("TECH_ACT_BULL", seed) or "Theo xu hướng là ưu tiên; vào từng phần và chỉ gia tăng khi có xác nhận, tránh mua đuổi."
        if ("bear" in ma_l or "down" in ma_l or "giảm" in ma_l) or ("bear" in al_l or "giảm" in al_l):
            return pick_phrase("TECH_ACT_BEAR", seed) or "Thiên về phòng thủ; giảm quy mô và chờ cấu trúc ổn định trở lại trước khi hành động."
        return pick_phrase("TECH_ACT_NEUTRAL", seed) or "Kiên nhẫn quan sát; chỉ hành động khi cấu trúc và xung lực đồng thuận."

    def _plan_action(act: str) -> str:
        a = (act or "").upper()
        seed = f"{tick}|C|{act}|{dp_urg}|{comp_status}"
        if a in {"BUY", "ADD", "ACCUMULATE", "ENTER"}:
            return pick_phrase("PLAN_ACT_BUY", seed) or "Giải ngân từng phần; chỉ gia tăng khi điều kiện vào lệnh được kích hoạt rõ ràng."
        if a in {"TRIM", "REDUCE"}:
            return pick_phrase("PLAN_ACT_TRIM", seed) or "Hạ tỷ trọng để quản trị rủi ro; giữ phần lõi nếu cấu trúc còn giữ được."
        if a in {"EXIT", "CUT"}:
            return pick_phrase("PLAN_ACT_EXIT", seed) or "Ưu tiên thoát/giảm mạnh để bảo toàn vốn; chỉ cân nhắc lại khi cấu trúc phục hồi."
        if a in {"HOLD"}:
            return pick_phrase("PLAN_ACT_HOLD", seed) or "Giữ vị thế và theo dõi kỷ luật; chỉ hành động khi điểm xác nhận xuất hiện."
        return pick_phrase("PLAN_ACT_WAIT", seed) or "Kiên nhẫn quan sát; tránh hành động khi điều kiện chưa rõ."

    # --- Fib bands (labels only; may include % but kept minimal)
    fib_short_band = _disp(_as_dict(_as_dict(fib.get("ShortWindow")).get("Summary")).get("Band") or _as_dict(fib.get("ShortWindow")).get("Band"))
    fib_long_band = _disp(_as_dict(_as_dict(fib.get("LongWindow")).get("Summary")).get("Band") or _as_dict(fib.get("LongWindow")).get("Band"))

    lines: List[str] = []
    if note:
        lines.append(note.strip())
        lines.append("")
    # ----------------------------
    # A. TECHNICAL ANALYSIS
    # ----------------------------
    ma_reg = _disp(ma.get("Regime"))
    align = _disp(bias.get("Alignment"))
    seed_a = f"{tick}|A|{ma_reg}|{align}|{_disp(scen.get('Name'))}"

    a_ctx = (pick_phrase("A_CTX", seed_a) or "Kịch bản {scen}; nền xu hướng (MA) {ma_reg}.").format(
        scen=_disp(scen.get("Name")),
        ma_reg=ma_reg,
    )
    a_ass1 = (pick_phrase("A_ASSESS_1", seed_a) or "RSI {rsi_state}; MACD {macd_state}; mức đồng thuận RSI+MACD {align}.").format(
        rsi_state=_disp(rsi.get("State")),
        macd_state=_disp(macd.get("State")),
        align=align,
    )
    a_ass2 = (pick_phrase("A_ASSESS_2", seed_a) or "Fibonacci ngắn {fib_short} | dài {fib_long}; dòng tiền {vol_reg}.").format(
        fib_short=fib_short_band,
        fib_long=fib_long_band,
        vol_reg=_disp(vol.get("Regime")),
    )
    a_quant = (pick_phrase("A_QUANT_NOTE", seed_a) or "Ghi chú định lượng: Điểm tổng hợp {ms_total} | Tin cậy {conv}.").format(
        ms_total=_disp(ms.get("Total")),
        conv=_disp(conv),
    )

    lines += [
        "A. TECHNICAL ANALYSIS",
        f"• Context: {a_ctx}",
        f"• Assessment: {a_ass1}",
        f"• Assessment: {a_ass2}",
        f"• Action: {_tech_action(ma_reg, align)}",
        f"• {a_quant}",
        "",
    ]
    # ----------------------------
    # B. FUNDAMENTAL ANALYSIS
    # ----------------------------
    lines.append("B. FUNDAMENTAL ANALYSIS")
    seed_b = f"{tick}|B|{fund_rec}|{fund_target_k}|{fund_upside}"
    has_fund = (fund_target_k != "–") or (fund_rec != "–") or (fund_upside != "–")

    if has_fund:
        b_ctx = pick_phrase("B_CTX_HAS", seed_b) or "Dữ liệu cơ bản có sẵn trong gói hiện tại."
        b_ass = (pick_phrase("B_ASSESS", seed_b) or "Khuyến nghị (tham khảo) {rec}; Giá mục tiêu (k) {target_k} | Upside {upside}.").format(
            rec=fund_rec,
            target_k=fund_target_k,
            upside=fund_upside,
        )
        b_act = pick_phrase("B_ACT_HAS", seed_b) or "Dùng làm neo kỳ vọng trung hạn; vẫn ưu tiên kỷ luật kỹ thuật và quản trị rủi ro khi triển khai."
        lines += [
            f"• Context: {b_ctx}",
            f"• Assessment: {b_ass}",
            f"• Action: {b_act}",
        ]
    else:
        b_ctx = pick_phrase("B_CTX_NONE", seed_b) or "Chưa có dữ liệu cơ bản trong gói hiện tại."
        b_act = pick_phrase("B_ACT_NONE", seed_b) or "Tạm thời dùng khung kỹ thuật và kỷ luật giao dịch; bổ sung dữ liệu cơ bản khi cần quyết định trung hạn."
        lines += [
            f"• Context: {b_ctx}",
            f"• Action: {b_act}",
        ]
    # ----------------------------
    # C. TRADE PLAN
    # ----------------------------
    seed_c = f"{tick}|C|{setup_name}|{plan_state}|{_disp(rr_txt)}|{dp_act}|{dp_urg}|{comp_status}"

    c_ctx = (pick_phrase("C_CTX", seed_c) or "Setup chính {setup}; trạng thái {plan_state}; RR {rr}.").format(
        setup=setup_name,
        plan_state=plan_state,
        rr=_disp(rr_txt),
    )
    c_exec = (pick_phrase("C_ASSESS_EXEC", seed_c) or "Kỷ luật triển khai {act} ({urg}).").format(
        act=dp_act,
        urg=dp_urg,
    )

    if comp_missing:
        miss = ", ".join([str(x) for x in comp_missing[:6]])
        c_comp = (pick_phrase("C_ASSESS_COMP_MISS", seed_c) or "Hoàn thiện kế hoạch {comp} (thiếu: {miss}).").format(
            comp=comp_status,
            miss=miss,
        )
    else:
        c_comp = (pick_phrase("C_ASSESS_COMP_OK", seed_c) or "Hoàn thiện kế hoạch {comp}.").format(
            comp=comp_status,
        )

    lines += [
        "",
        "C. TRADE PLAN",
        f"• Context: {c_ctx}",
        f"• Assessment: {c_exec}",
        f"• Assessment: {c_comp}",
    ]

    if comp_msg:
        c_note = (pick_phrase("C_NOTE", seed_c) or "Ghi chú: {msg}.").format(msg=comp_msg)
        lines.append(f"• {c_note}")

    if tags_preview != "N/A":
        c_tags = (pick_phrase("C_TAGS", seed_c) or "Tags: {tags}.").format(tags=tags_preview)
        lines.append(f"• {c_tags}")

    lines += [
        f"• Action: {_plan_action(dp_act)}",
        "",
        "D. Rủi ro / Lợi nhuận",
        f"Risk%: {risk}",
        f"Reward%: {reward}",
        f"RR: {must_rr}",
        f"Confidence (Tech): {must_conf}",
    ]

    return "\n".join([x for x in lines if x is not None and str(x).strip() != ""]).strip() + "\n"


# ============================================================
# Phase 4: Rewrite-only (draft -> GPT polish) with validators
# ============================================================

_FUND_LOCK_TOKENS = [
    "target", "upside", "recommendation",
    "khuyến nghị", "gia muc tieu", "giá mục tiêu", "mục tiêu", "định giá",
]

def _section_slice(text: str, start_pat: str, end_pat: str) -> str:
    if not isinstance(text, str):
        return ""
    m1 = re.search(start_pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m1:
        return ""
    s = text[m1.end():]
    m2 = re.search(end_pat, s, flags=re.IGNORECASE | re.MULTILINE)
    return (s[:m2.start()] if m2 else s).strip()

def validate_fund_lock(text: str) -> bool:
    """Ensure fundamental tokens only appear in section B."""
    a = _section_slice(text, r"^\s*A\.", r"^\s*B\.")
    c = _section_slice(text, r"^\s*C\.", r"^\s*D\.")
    d = _section_slice(text, r"^\s*D\.", r"\Z")
    blob = f"\n{a}\n{c}\n{d}\n".lower()
    for tok in _FUND_LOCK_TOKENS:
        if tok.lower() in blob:
            return False
    return True

def _number_tokens(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    return re.findall(r"\b\d+(?:\.\d+)?\b", s)

def validate_no_new_numbers(draft: str, out: str) -> bool:
    """Heuristic: forbid introducing new numeric tokens beyond the draft.

    - Ignores small section numbering (<=9).
    - Allows exact reuse of any numeric token in draft.
    """
    dset = set(_number_tokens(draft))
    oset = set(_number_tokens(out))
    allow_small = {str(i) for i in range(0, 10)}
    new = {x for x in oset - dset if x not in allow_small}
    return len(new) == 0

def _rewrite_report_ad_only(draft: str, primary: Dict[str, Any], *, model: str = "gpt-4o") -> str:
    """Rewrite-only: GPT receives the deterministic draft only (no full JSON)."""
    prompt = f"""
Bạn là biên tập viên ngôn ngữ cho báo cáo A–D. Dưới đây là BẢN NHÁP do engine đã chuẩn hoá.
NHIỆM VỤ: chỉ chỉnh câu chữ cho mượt, rõ ràng hơn; KHÔNG thêm dữ kiện; KHÔNG suy luận; KHÔNG đổi số.

RÀNG BUỘC BẮT BUỘC:
- Giữ nguyên cấu trúc và tiêu đề: A. ... / B. ... / C. ... / D. ...
- Mọi con số phải GIỮ NGUYÊN y hệt (không đổi, không thêm số mới).
- Mục B là nơi DUY NHẤT được nhắc "khuyến nghị/giá mục tiêu/upside/target". Tuyệt đối không nhắc các nội dung này ở A/C/D.
- Mục D bắt buộc đúng 4 dòng và COPY ĐÚNG số (không được tự tính):
  Risk%: ...
  Reward%: ...
  RR: ...
  Confidence (Tech): ...

BẢN NHÁP (không được làm thay đổi dữ kiện, chỉ làm mượt câu chữ):
{draft}
""".strip()

    guarded = call_gpt_with_guard(prompt, {"PrimarySetup": primary}, max_retry=2, model=model)

    if not guarded:
        return draft

    if not validate_fund_lock(guarded):
        return draft

    if not validate_no_new_numbers(draft, guarded):
        return draft

    return guarded


# ============================================================
# Public entrypoint
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    """Generate A–D report.

    Phase 4:
    - Engine builds a deterministic draft.
    - GPT (optional) only polishes the draft (rewrite-only), without access to full JSON.
    - Validators + fallback prevent drift (Section D, fund lock, new numbers).
    """

    if not isinstance(data, dict):
        return ""

    if "Error" in data:
        return f"ERROR: {data.get('Error')}"

    tick = str(data.get("Ticker") or (data.get("AnalysisPack") or {}).get("Ticker") or "").strip().upper()
    scenario = str(data.get("Scenario") or "N/A")
    conviction = _safe_float(data.get("Conviction"), default=np.nan)

    ap = normalize_analysis_pack(data.get("AnalysisPack"))
    data["AnalysisPack"] = ap

    last = data.get("Last") or (ap.get("Last") or {})
    close = _fmt_price((last or {}).get("Close"))

    conv_txt = f"{conviction:.1f}/10" if pd.notna(conviction) else "N/A"
    header_html = (
        f"<h2 style='margin:0; padding:0; font-size:26px; line-height:1.2;'>"
        f"{tick} — {close} | Diem tin cay: {conv_txt} | {scenario}</h2>"
    )

    draft = _deterministic_report_ad(data)

    key = get_openai_api_key()
    enabled = str(os.getenv("INCEPTION_GPT_REWRITE_REPORT_AD", "1")).strip() not in {"0", "false", "False", "no", "NO"}
    model = str(os.getenv("INCEPTION_GPT_MODEL_REPORT_AD", os.getenv("INCEPTION_GPT_MODEL", "gpt-4o"))).strip() or "gpt-4o"

    if (not key) or (not enabled):
        note = "NOTE: GPT rewrite-only disabled (no API key or feature flag off). Using deterministic A–D."
        return f"{header_html}\n\n{note}\n\n{draft}".strip()

    primary = _as_dict(ap.get("PrimarySetup"))

    try:
        content = _rewrite_report_ad_only(draft, primary, model=model)
        if not content:
            raise RuntimeError("Empty rewrite result")
        return f"{header_html}\n\n{content}".strip()
    except Exception as e:
        note = f"NOTE: GPT rewrite-only failed ({e}). Using deterministic A–D."
        return f"{header_html}\n\n{note}\n\n{draft}".strip()
