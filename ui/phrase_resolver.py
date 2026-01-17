from __future__ import annotations

"""Deterministic phrase resolver for keyed narrative."""

from typing import Any, Dict, List
import hashlib

from inception.ui.phrase_bank_vi import PHRASE_BANK_VI


def _safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _stable_index(key: str, salt: str, n: int) -> int:
    if n <= 1:
        return 0
    h = hashlib.md5(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % n


def resolve_phrases_vi(keys: List[str], ctx: Dict[str, Any]) -> List[str]:
    """Resolve keys to Vietnamese sentences deterministically.

    - Dedupe keys while preserving order.
    - If a key is missing from the bank, it is skipped.
    - Uses stable hash to pick a variant when multiple templates exist.
    """
    seen = set()
    out: List[str] = []

    ticker = _safe_text(ctx.get("ticker") or "").upper()
    salt = "|".join([
        ticker,
        _safe_text(ctx.get("mode")),
        _safe_text(ctx.get("zone_now")),
        _safe_text(ctx.get("primary_action")),
        _safe_text(ctx.get("bias")),
        _safe_text(ctx.get("size_hint")),
    ])

    for k in keys:
        k = _safe_text(k)
        if not k or k in seen:
            continue
        seen.add(k)
        templates = PHRASE_BANK_VI.get(k) or []
        if not templates:
            continue
        idx = _stable_index(k, salt=salt, n=len(templates))
        tmpl = templates[idx]
        try:
            sent = tmpl.format(**ctx)
        except Exception:
            sent = tmpl
        sent = _safe_text(sent)
        if sent:
            out.append(sent)
    return out
