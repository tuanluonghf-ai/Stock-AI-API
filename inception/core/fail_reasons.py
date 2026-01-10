"""Fail Reasons taxonomy (v1).

Goal
----
Provide a single deterministic mapping from common failure/"not-ready" conditions
to stable reason codes and renderer-friendly UI labels.

Design constraints
------------------
- Core-only (no Streamlit).
- Deterministic and backward compatible.
- Prefer concise UI labels; use optional detail text when available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Stable catalog. Keep codes immutable once released.
FAIL_REASON_CATALOG: Dict[str, Dict[str, Any]] = {
    # Plan completeness
    "PLAN_MISSING_LEVELS": {
        "severity": 3,
        "ui_short": "Thiếu levels",
        "ui_long": "Trade plan thiếu levels bắt buộc (Entry/Stop/TP/Zone).",
    },

    # Structure / regime
    "STRUCTURE_BROKEN": {
        "severity": 3,
        "ui_short": "Structure gãy",
        "ui_long": "Cấu trúc bị phá vỡ; ưu tiên phòng thủ và chờ tái lập cấu trúc.",
    },
    "STRUCTURE_CEILING": {
        "severity": 2,
        "ui_short": "Chạm trần",
        "ui_long": "Đang chạm vùng kháng cự/cấu trúc; cần reclaim hoặc xác nhận trước khi tăng xác suất.",
    },
    "REGIME_CONFLICT": {
        "severity": 2,
        "ui_short": "Regime lệch",
        "ui_long": "Regime/Trend hiện tại không đồng pha với plan type; xác suất thấp.",
    },

    # Risk / execution gates
    "EXEC_LOCKED": {
        "severity": 3,
        "ui_short": "Gate LOCK",
        "ui_long": "Bị khóa bởi gate/điều kiện hệ thống; không triển khai plan chủ động.",
    },
    "DATA_QUALITY_LOW": {
        "severity": 2,
        "ui_short": "Data thiếu",
        "ui_long": "Dữ liệu/pack thiếu hoặc không ổn định; cần bổ sung để tạo plan đáng tin.",
    },

    # RR
    "RR_BELOW_MIN": {
        "severity": 2,
        "ui_short": "RR thấp",
        "ui_long": "R:R dưới mức tối thiểu theo policy; cần điểm vào tốt hơn hoặc stop hợp lý hơn.",
    },
    "RR_BORDERLINE": {
        "severity": 1,
        "ui_short": "RR sát ngưỡng",
        "ui_long": "R:R chưa đủ tốt; chỉ cân nhắc khi có trigger/volume/structure đồng pha.",
    },

    # Volume
    "VOLUME_NOT_CONFIRM": {
        "severity": 1,
        "ui_short": "Volume fail",
        "ui_long": "Volume không xác nhận; rủi ro false-break cao.",
    },
    "VOLUME_NEED_CONFIRM": {
        "severity": 1,
        "ui_short": "Chờ volume",
        "ui_long": "Chưa có volume xác nhận; ưu tiên chờ tăng xác suất.",
    },

    # Trigger
    "TRIGGER_INVALID": {
        "severity": 2,
        "ui_short": "Trigger invalid",
        "ui_long": "Trigger bị invalid theo setup; không triển khai plan này.",
    },
    "TRIGGER_NOT_READY": {
        "severity": 1,
        "ui_short": "Chờ trigger",
        "ui_long": "Trigger chưa kích hoạt; ưu tiên chờ xác nhận.",
    },

    # Breakout quality proxy
    "BREAKOUT_QUALITY_LOW": {
        "severity": 1,
        "ui_short": "Breakout yếu",
        "ui_long": "Chất lượng breakout thấp (BreakoutForce yếu); dễ fail/reject.",
    },

    # HOLDING distress overlays (position management)
    "HOLD_UNDERWATER_SEVERE": {
        "severity": 3,
        "ui_short": "Lỗ sâu",
        "ui_long": "Vị thế đang lỗ sâu; ưu tiên giảm rủi ro, không add cho tới khi reclaim cấu trúc.",
    },
    "HOLD_UNDERWATER_MEDIUM": {
        "severity": 2,
        "ui_short": "Lỗ đáng kể",
        "ui_long": "Vị thế đang lỗ đáng kể; chỉ cân nhắc giữ nếu còn cấu trúc, ưu tiên plan reclaim/defensive.",
    },
    "HOLD_ADD_DISABLED": {
        "severity": 2,
        "ui_short": "Tạm ngừng add",
        "ui_long": "Underwater/Regime chưa phù hợp; tạm ngừng mua thêm cho tới khi có reclaim xác nhận.",
    },


}


def _get_meta(code: str) -> Dict[str, Any]:
    base = FAIL_REASON_CATALOG.get(code, {})
    return {
        "code": code,
        "severity": int(base.get("severity", 1)),
        "ui_short": str(base.get("ui_short", code)),
        "ui_long": str(base.get("ui_long", code)),
    }


def make_reason(code: str, detail: str = "") -> Dict[str, Any]:
    """Create a standardized reason payload.

    Output is intentionally UI-friendly:
      - label: long label (can append detail)
      - ui_short: short label for dashboard chips
      - severity: int (3=hard block, 2=block/major issue, 1=soft wait)
    """
    m = _get_meta(code)
    label = m["ui_long"]
    d = (detail or "").strip()
    if d and d != "-":
        # keep label readable; detail appended after dash
        label = f"{label} — {d}"
    return {
        "code": m["code"],
        "severity": m["severity"],
        "ui_short": m["ui_short"],
        "label": label,
    }


def rank_reasons(reasons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable ranking: severity desc, then catalog order (implicit), then code."""

    # catalog order for deterministic tie-breaks
    order = {c: i for i, c in enumerate(FAIL_REASON_CATALOG.keys())}

    def _k(r: Dict[str, Any]) -> Tuple[int, int, str]:
        sev = int(r.get("severity", 0))
        code = str(r.get("code") or "")
        return (-sev, order.get(code, 10_000), code)

    return sorted([x for x in reasons if isinstance(x, dict)], key=_k)


def summarize_top_reason(reasons: Any) -> Dict[str, str]:
    """Return top reason for dashboard.

    Returns: {"code": "-", "ui_short": "-", "label": "-"}
    """
    if not isinstance(reasons, list) or not reasons:
        return {"code": "-", "ui_short": "-", "label": "-"}
    ranked = rank_reasons(reasons)
    top = ranked[0] if ranked else {}
    return {
        "code": str(top.get("code") or "-"),
        "ui_short": str(top.get("ui_short") or "-"),
        "label": str(top.get("label") or "-"),
    }
