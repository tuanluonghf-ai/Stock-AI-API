from __future__ import annotations

"""ActionBiasPack builder (UI-side only).

Purpose
- Bridge DNA bucket × ZonePack.zone_now → Action Bias + Position Size Guidance.
- Provide deterministic, client-friendly expectation layer.
- MUST NOT compute new signals; only transforms existing packs/labels.
- MUST NOT override Decision Layer; bias/size are guidance only.

Notes
- Designed for INCEPTION UI/Narrative refactor (Phase F).
"""

from typing import Any, Dict
import hashlib


def _safe_text(x: Any) -> str:
    try:
        return str(x).strip()
    except Exception:
        return ""


def _bucket_dna(class_name: str) -> str:
    s = _safe_text(class_name).upper()
    if "ILL" in s or "THANH KHOAN" in s:
        return "Illiquid/Noisy"
    if "EVENT" in s or "GAP" in s:
        return "Event/Gap-Prone"
    if "GLASS" in s or "CANNON" in s:
        return "Glass Cannon"
    if "DEF" in s or "SAFE" in s:
        return "Defensive"
    return "Balanced"


def _stable_pick(key: str, options: list[str], salt: str = "") -> str:
    if not options:
        return ""
    h = hashlib.md5(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return options[int(h, 16) % len(options)]


def compute_action_bias_pack(
    analysis_pack: Dict[str, Any],
    zone_pack: Dict[str, Any],
    character_pack: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ap = analysis_pack if isinstance(analysis_pack, dict) else {}
    zp = zone_pack if isinstance(zone_pack, dict) else {}
    cp = character_pack if isinstance(character_pack, dict) else {}

    # Mode (prefer DashboardSummaryPack if present)
    dsum = ap.get("DashboardSummaryPack") if isinstance(ap.get("DashboardSummaryPack"), dict) else {}
    mode = _safe_text(dsum.get("mode") or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        pos = ap.get("Position") if isinstance(ap.get("Position"), dict) else {}
        mode = _safe_text(pos.get("mode") or "").upper()
    if mode not in {"HOLDING", "FLAT"}:
        mode = "UNKNOWN"

    # DNA bucket
    class_name = _safe_text(
        (cp.get("ClassName") or cp.get("CharacterClass") or cp.get("Class") or "")
        if isinstance(cp, dict) else ""
    )
    dna_bucket = _bucket_dna(class_name)

    # Zone now
    zone_now = _safe_text(zp.get("zone_now") or "NEUTRAL").upper()
    if zone_now not in {"POSITIVE", "RECLAIM", "RISK", "NEUTRAL"}:
        zone_now = "NEUTRAL"

    # Bias (stable rules)
    if zone_now == "RISK":
        bias = "DEFENSIVE"
        rationale_keys = ["ZONE_RISK"]
    elif zone_now == "RECLAIM":
        bias = "CAUTIOUS"
        rationale_keys = ["ZONE_RECLAIM"]
    elif zone_now == "POSITIVE":
        bias = "AGGRESSIVE"
        rationale_keys = ["ZONE_POSITIVE"]
    else:
        bias = "CAUTIOUS"
        rationale_keys = ["ZONE_NEUTRAL"]

    # Size hint (mode-aware, no override)
    riskier_bucket = dna_bucket in {"Glass Cannon", "Event/Gap-Prone", "Illiquid/Noisy"}

    if mode != "HOLDING":
        # FLAT / entering
        if bias == "AGGRESSIVE":
            size_hint = "PARTIAL" if riskier_bucket else "FULL"
            rationale_keys += ["SIZE_ENTER"]
        elif bias == "CAUTIOUS":
            size_hint = "PROBE"
            rationale_keys += ["SIZE_PROBE"]
        else:
            size_hint = "FLAT"
            rationale_keys += ["SIZE_FLAT"]
    else:
        # HOLDING
        if bias == "AGGRESSIVE":
            size_hint = "PARTIAL"   # hold core; add only on confirmation
            rationale_keys += ["SIZE_HOLD_ADD_COND"]
        elif bias == "CAUTIOUS":
            size_hint = "PROBE"     # hold/observe; avoid increasing commitment
            rationale_keys += ["SIZE_HOLD_OBSERVE"]
        else:
            size_hint = "FLAT"      # reduce commitment posture
            rationale_keys += ["SIZE_REDUCE_RISK"]

    # One-liner (client-facing, 1 sentence, deterministic rotation)
    salt = _safe_text(ap.get("Ticker") or dsum.get("ticker") or "")
    if bias == "DEFENSIVE":
        one_liner = _stable_pick(
            "one_liner_def",
            [
                "Khuynh hướng phù hợp là phòng thủ: ưu tiên giảm cam kết và chờ tái chiếm vùng an toàn trước khi chủ động hơn.",
                "Bias nghiêng phòng thủ: giữ rủi ro trong tầm kiểm soát, chỉ nâng cam kết khi có xác nhận phục hồi rõ ràng.",
            ],
            salt=salt + "|" + dna_bucket + "|" + zone_now,
        )
    elif bias == "CAUTIOUS":
        one_liner = _stable_pick(
            "one_liner_caut",
            [
                "Khuynh hướng phù hợp là thận trọng: ưu tiên thăm dò/giữ nhỏ và chờ xác nhận theo đúng kỷ luật vùng giá.",
                "Bias nghiêng thận trọng: đi chậm để tối ưu giá vốn và tránh vào sớm khi xác suất chưa đủ dày.",
            ],
            salt=salt + "|" + dna_bucket + "|" + zone_now,
        )
    else:
        if riskier_bucket:
            one_liner = _stable_pick(
                "one_liner_aggr_risky",
                [
                    "Khuynh hướng tích cực nhưng nên kiểm soát quy mô: ưu tiên tham gia một phần và chỉ gia tăng khi tín hiệu xác nhận bền.",
                    "Bias nghiêng chủ động, song nhóm này biến động cao: tham gia có kỷ luật và giữ dư địa xử lý rủi ro.",
                ],
                salt=salt + "|" + dna_bucket + "|" + zone_now,
            )
        else:
            one_liner = _stable_pick(
                "one_liner_aggr",
                [
                    "Khuynh hướng tích cực: có thể ưu tiên kịch bản thuận xu hướng, miễn là mốc rủi ro vẫn được giữ chặt.",
                    "Bias nghiêng chủ động: phù hợp triển khai theo kế hoạch nếu điều kiện vùng giá được giữ vững.",
                ],
                salt=salt + "|" + dna_bucket + "|" + zone_now,
            )

    return {
        "version": "1.0",
        "mode": mode,
        "dna_bucket": dna_bucket,
        "zone_now": zone_now,
        "bias": bias,
        "size_hint": size_hint,
        "one_liner": one_liner,
        "rationale_keys": rationale_keys,
    }
