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
    ap = normalize_analysis_pack((data or {}).get("AnalysisPack"))

    pro = _as_dict(ap.get("ProTech"))
    ma = _as_dict(pro.get("MA"))
    rsi = _as_dict(pro.get("RSI"))
    macd = _as_dict(pro.get("MACD"))
    bias = _as_dict(pro.get("Bias"))
    fib = _as_dict(ap.get("Fibonacci"))
    vol = _as_dict(pro.get("Volume"))
    scen = _as_dict(ap.get("Scenario12"))
    ms = _as_dict(ap.get("MasterScore"))
    conv = ap.get("Conviction")

    primary = _as_dict(ap.get("PrimarySetup"))

    # Step 11: prefer TradePlanPack/DecisionPack for execution clarity
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
    # If TradePlanPack exists, show its computed state (it already accounts for PlanCompleteness)
    plan_status = _val_or_na(pp.get("state")) if pp else _val_or_na(primary.get("Status"))
    rr_txt = _val_or_na(primary.get("RR"))

    tags = primary.get("ReasonTags") or []
    if isinstance(tags, list):
        tags_preview = ", ".join([str(x) for x in tags[:6]]) if tags else "N/A"
    else:
        tags_preview = "N/A"

    risk = _val_or_na(primary.get("RiskPct"))
    reward = _val_or_na(primary.get("RewardPct"))
    must_rr = _val_or_na(primary.get("RR"))
    must_conf = _val_or_na(primary.get("Confidence (Tech)", primary.get("Probability")))

    lines: List[str] = []
    if note:
        lines.append(note.strip())
        lines.append("")

    lines += [
        "A. Ky thuat",
        f"1. MA: {_val_or_na(ma.get('Regime'))}.",
        f"2. RSI: {_val_or_na(rsi.get('State'))} | {_val_or_na(rsi.get('Direction'))}.",
        f"3. MACD: {_val_or_na(macd.get('State'))} | ZeroLine: {_val_or_na(macd.get('ZeroLine'))}.",
        f"4. RSI+MACD alignment: {_val_or_na(bias.get('Alignment'))}.",
        f"5. Fibonacci: Short/Long = {_val_or_na(_as_dict(fib.get('ShortWindow')).get('Band'))} / {_val_or_na(_as_dict(fib.get('LongWindow')).get('Band'))}.",
        f"6. Volume: Ratio {_val_or_na(vol.get('Ratio'))} | Regime: {_val_or_na(vol.get('Regime'))}.",
        f"7. Scenario12: {_val_or_na(scen.get('Name'))}.",
        f"8. Master/Conviction: {_val_or_na(ms.get('Total'))} | {_val_or_na(conv)}.",
        "",
        "B. Co ban",
        "(Chi hien thi khi co du lieu co ban trong pack.)",
        "",
        "C. TRADE PLAN",
        f"Primary setup: {setup_name} | Status: {plan_status} | RR: {rr_txt}.",
        f"Plan tags: {tags_preview}.",
        (f"PlanCompleteness: {comp_status} (missing: {', '.join([str(x) for x in comp_missing])})" if comp_status != "N/A" and comp_missing else f"PlanCompleteness: {comp_status}."),
        (f"Reason: {comp_msg}." if comp_msg else ""),
        (f"Decision Layer: {dp_act} ({dp_urg})." if dp_act != "N/A" else ""),
        "",
        "D. Rui ro vs loi nhuan",
        f"Risk%: {risk}",
        f"Reward%: {reward}",
        f"RR: {must_rr}",
        f"Confidence (Tech): {must_conf}",
    ]

    return "\n".join(lines).strip() + "\n"


# ============================================================
# Public entrypoint
# ============================================================

def generate_insight_report(data: Dict[str, Any]) -> str:
    """Generate A–D report.

    - Uses GPT narrative if OPENAI_API_KEY exists.
    - Falls back to deterministic A–D if key missing or call fails.
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

    fund = ap.get("Fundamental") or {}
    fund_text = (
        f"Khuyen nghi: {fund.get('Recommendation', 'N/A')} | "
        f"Gia muc tieu: {_fmt_thousand(fund.get('Target'))} | "
        f"Upside: {_fmt_pct(fund.get('UpsidePct'))}"
        if isinstance(fund, dict) and fund
        else "Khong co du lieu co ban"
    )

    pack_json = _json_strict(ap)

    primary = (ap.get("PrimarySetup") or {})
    must_risk = primary.get("RiskPct")
    must_reward = primary.get("RewardPct")
    must_rr = primary.get("RR")
    must_conf = primary.get("Confidence (Tech)", primary.get("Probability"))

    prompt = f"""
Ban la "INCEPTION Narrative Editor" cho bao cao phan tich co phieu.
Vai tro: dien giai + bien tap van phong tu JSON "AnalysisPack".
Tuyet doi:
- Khong bia so, khong uoc luong, khong tu tinh.
- Chi duoc dung dung con so co san trong JSON.

RANG BUOC QUAN TRONG (FUNDAMENTAL LOCK):
- Fundamental (Recommendation/Target/Upside/...) CHI DUOC NHAC O MUC B.
- O A/C/D: CAM nhac Target/Upside/Recommendation.

FORMAT OUTPUT:
A. Ky thuat
1) ...
...
8) ...

B. Co ban
(1-3 cau, dung dung dong du lieu cung cap)

C. TRADE PLAN
(5-9 cau)

QUY TAC BAT BUOC CHO MUC C (PLAN COMPLETENESS):
- Neu AnalysisPack.TradePlanPack.plan_primary.plan_completeness.status = FAIL (hoac gates.plan=FAIL):
  + PHAI noi ro plan chua hoan chinh (thieu Stop/Entry zone) va KHONG duoc khuyen nghi vao lenh.
  + Neu can, giai thich ngan gon vi sao he thong tu ha trang thai (WATCH/INVALID).

D. Rui ro vs loi nhuan
Risk%: ...
Reward%: ...
RR: ...
Confidence (Tech): ...

MUC B (FUNDAMENTAL - chi dung dong nay, khong suy luan them):
{fund_text}

RANG BUOC MUC D (COPY DUNG, khong tu tinh/uoc luong):
- Risk% = {must_risk}
- Reward% = {must_reward}
- RR = {must_rr}
- Confidence (Tech) = {must_conf}

Du lieu (AnalysisPack JSON):
{pack_json}
""".strip()

    key = get_openai_api_key()
    if not key:
        content = _deterministic_report_ad(
            data,
            note="NOTE: GPT narrative disabled (OPENAI_API_KEY not set). Using deterministic A–D.",
        )
        return f"{header_html}\n\n{content}"

    try:
        content = call_gpt_with_guard(prompt, ap, max_retry=2)
        if not content:
            raise RuntimeError("Empty GPT response")
        return f"{header_html}\n\n{content}"
    except Exception as e:
        content = _deterministic_report_ad(
            data,
            note=f"NOTE: GPT call failed ({e}). Using deterministic A–D.",
        )
        return f"{header_html}\n\n{content}"
