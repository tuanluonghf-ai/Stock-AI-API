"""Narrative Stability: Semantic Anchoring.

Scope
-----
- Do NOT modify core indicator logic.
- Provide a stable semantic anchor (DNA + dominant regime) so narrative tone/meaning
  does not drift between runs unless regime truly changes.

Outputs
-------
NarrativeAnchorPack (attached to AnalysisPack + ResultPack).
Optionally, this module can *lightly* inject the anchor phrase into NarrativeDraftPack
in a non-destructive way (prepend only when missing).

Persistence
-----------
Stores last confirmed regime/anchor per ticker under .inception_state by default.
Disable read/write via:
  - INCEPTION_DISABLE_STABILITY_STATE=1
Disable write via:
  - INCEPTION_PERSIST_STABILITY=0
"""

from __future__ import annotations

def _load_prev_regime(*, ticker: str) -> str:
    st = load_state(scope="narrative", ticker=ticker, default={})
    v = st.get("prev_regime") if isinstance(st, dict) else ""
    return str(v or "").strip()


def _save_prev_regime(*, ticker: str, prev_regime: str) -> None:
    save_state(scope="narrative", ticker=ticker, state={"prev_regime": str(prev_regime or "").strip()})



from inception.core.stability.state_store import load_state, save_state
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from inception.core.helpers import _safe_bool, _safe_text


# ----------------------------
# Persistence
# ----------------------------

def _state_dir(base_dir: Optional[str] = None) -> Path:
    env = _safe_text(os.environ.get("INCEPTION_STATE_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if base_dir:
        return (Path(base_dir).expanduser().resolve() / ".inception_state")
    return (Path.cwd().resolve() / ".inception_state")


def _disabled_state() -> bool:
    return _safe_text(os.environ.get("INCEPTION_DISABLE_STABILITY_STATE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )


def _allow_persist() -> bool:
    v = _safe_text(os.environ.get("INCEPTION_PERSIST_STABILITY") or "1").strip().lower()
    return v in ("1", "true", "yes", "y")


def _load_prev_anchor(*, ticker: str, base_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if _disabled_state():
        return None
    try:
        p = _state_dir(base_dir) / "narrative_anchor.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        v = data.get(ticker)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _save_prev_anchor(*, ticker: str, anchor: Dict[str, Any], base_dir: Optional[str]) -> None:
    if _disabled_state() or not _allow_persist():
        return
    try:
        d = _state_dir(base_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "narrative_anchor.json"

        data: Dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

        data[ticker] = anchor
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


# ----------------------------
# Regime detection (conservative)
# ----------------------------

def _pick_pack(analysis_pack: Dict[str, Any], name: str) -> Dict[str, Any]:
    p = analysis_pack.get(name) if isinstance(analysis_pack, dict) else None
    return p if isinstance(p, dict) else {}


def _detect_regime(analysis_pack: Dict[str, Any]) -> Tuple[str, str]:
    """Return (regime, source).

    Regime is one of: Trend | Range | Defensive.

    This function is intentionally conservative and prioritizes existing, explicit
    labels/packs if available.
    """
    # 1) Explicit Narrative/Regime packs (if your engine already produces them)
    for key in ("RegimePack", "MarketRegimePack", "DominantRegimePack"):
        rp = _pick_pack(analysis_pack, key)
        r = _safe_text(rp.get("regime") or rp.get("Regime") or "").strip().title()
        if r in ("Trend", "Range", "Defensive"):
            return r, key

    # 2) DashboardSummaryPack is computed later; avoid relying on it here.

    # 3) Use Style tilt as proxy (Trend/Momentum/Range)
    dash_like = _pick_pack(analysis_pack, "DashboardSummaryPack")
    st = _safe_text(dash_like.get("style_tilt") or dash_like.get("StyleTilt") or "").strip().upper()
    if st:
        if "RANGE" in st:
            return "Range", "DashboardSummaryPack.style_tilt"
        if "TREND" in st or "MOM" in st:
            return "Trend", "DashboardSummaryPack.style_tilt"

    # 4) Fallback heuristics from TradePlanPack gates / holding distress
    tpp = _pick_pack(analysis_pack, "TradePlanPack")
    ho = tpp.get("holding_overlay") if isinstance(tpp.get("holding_overlay"), dict) else {}
    distress = _safe_text(ho.get("distress_level") or "").strip().upper()
    if distress in ("MEDIUM", "SEVERE"):
        return "Defensive", "TradePlanPack.holding_overlay.distress_level"

    # If structure gate FAIL often implies defensive posture
    plan_primary = tpp.get("plan_primary") if isinstance(tpp.get("plan_primary"), dict) else {}
    gates = plan_primary.get("gates") if isinstance(plan_primary.get("gates"), dict) else {}
    structure = _safe_text(gates.get("structure") or "").strip().upper()
    if structure == "FAIL":
        return "Defensive", "TradePlanPack.plan_primary.gates.structure"

    # Default
    return "Trend", "default"


def _anchor_phrase_for(regime: str) -> str:
    r = _safe_text(regime).strip().title()
    if r == "Range":
        return "Cổ phiếu vận động theo vùng giá; ưu tiên biên an toàn và kỷ luật mua thấp–bán cao hơn là phản ứng ngắn hạn."
    if r == "Defensive":
        return "Cổ phiếu đang ở trạng thái phòng thủ; ưu tiên bảo toàn vốn và giảm nhịp giao dịch cho đến khi tín hiệu hồi phục rõ ràng hơn."
    # Trend default
    return "Cổ phiếu vận động theo xu hướng; ưu tiên kỷ luật hơn là phản ứng ngắn hạn."


def apply_narrative_anchor(
    *,
    ticker: str,
    analysis_pack: Dict[str, Any],
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute NarrativeAnchorPack and optionally inject into NarrativeDraftPack."""

    t = _safe_text(ticker).strip().upper() or ""
    prev = _load_prev_anchor(ticker=t, base_dir=base_dir) or {}

    detected_regime, source = _detect_regime(analysis_pack)
    detected_regime = _safe_text(detected_regime).strip().title() or "Trend"

    prev_regime = _safe_text(prev.get("regime") or "").strip().title()
    prev_streak = prev.get("_streak") if isinstance(prev.get("_streak"), int) else 0

    # Regime hysteresis: require 2 consecutive detections to switch.
    # If no prior regime, accept detected.
    if not prev_regime:
        regime = detected_regime
        streak = 1
        anchor_changed = True
        reason = "cold_start"
    else:
        if detected_regime == prev_regime:
            regime = prev_regime
            streak = min(prev_streak + 1, 9)
            anchor_changed = False
            reason = "regime_unchanged"
        else:
            # candidate shift
            streak = prev_streak + 1 if prev.get("_candidate") == detected_regime else 1
            if streak >= 2:
                regime = detected_regime
                anchor_changed = True
                reason = "regime_changed_confirmed"
            else:
                regime = prev_regime
                anchor_changed = False
                reason = "regime_change_pending"

    phrase = _anchor_phrase_for(regime)

    pack: Dict[str, Any] = {
        "schema": "NarrativeAnchorPack.v1",
        "prev_regime": prev_regime or "",
        "detected_regime": detected_regime,
        "regime": regime,
        "anchor_phrase": phrase,
        "anchor_changed": bool(anchor_changed),
        "reason": reason,
        "source": source,
        "hysteresis": {
            "confirm_n": 2,
            "streak": int(streak),
            "candidate": detected_regime if detected_regime != regime else "",
        },
    }

    # Light injection into NarrativeDraftPack (non-breaking): add fields; optionally prepend to status line.
    try:
        ndp = analysis_pack.get("NarrativeDraftPack") if isinstance(analysis_pack, dict) else None
        if isinstance(ndp, dict):
            ndp.setdefault("anchor", {})
            if isinstance(ndp.get("anchor"), dict):
                ndp["anchor"].update({"regime": regime, "phrase": phrase})

            # Prepend to status line only if not already present (kept short).
            status = ndp.get("status") if isinstance(ndp.get("status"), dict) else None
            if isinstance(status, dict):
                line = _safe_text(status.get("line_draft") or "")
                if phrase and phrase not in line:
                    # Keep to one anchor sentence; then a space + old content.
                    status["line_draft"] = (phrase + " " + line).strip() if line else phrase
    except Exception:
        pass

    # Persist
    if t:
        persist_obj = {
            "regime": regime,
            "anchor_phrase": phrase,
            "_streak": int(streak),
            "_candidate": detected_regime if detected_regime != regime else "",
        }
        _save_prev_anchor(ticker=t, anchor=persist_obj, base_dir=base_dir)

    return pack
