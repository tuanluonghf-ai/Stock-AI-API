from __future__ import annotations

"""Narrative governance utilities (UI-side only).

Phase H:
- Enforce deterministic key order (stable, policy-compliant).
- Apply sentence budget.
- Validate common guardrails (uncertainty line, numeric density).
"""

from typing import Any, Dict, List, Tuple
import re

from inception.ui.phrase_resolver import resolve_phrases_vi


_CANON_PREFIX_ORDER: List[str] = ["DNA_", "ZONE_", "BIAS_", "SIZE_", "DECISION_", "SAFETY_"]
_NUM_RE = re.compile(r"(?<![A-Za-z])\d+(?:[\.,]\d+)?%?")

def _safe_text(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _dedupe_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        x = _safe_text(x)
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _bucket_key(k: str) -> int:
    ku = _safe_text(k).upper()
    for i, p in enumerate(_CANON_PREFIX_ORDER):
        if ku.startswith(p):
            return i
    # unknown keys go after known ones but before SAFETY by default
    return len(_CANON_PREFIX_ORDER) - 1

def normalize_keys(keys: List[str], must_include_uncertainty: bool = True) -> List[str]:
    """Return keys in canonical order while preserving within-bucket order."""
    ks = _dedupe_preserve(keys or [])
    if must_include_uncertainty and "SAFETY_LINE" not in ks:
        ks.append("SAFETY_LINE")

    # stable bucket sort: collect by bucket preserving original order
    buckets: Dict[int, List[str]] = {}
    for k in ks:
        b = _bucket_key(k)
        buckets.setdefault(b, []).append(k)

    out: List[str] = []
    for b in range(len(_CANON_PREFIX_ORDER)):
        out.extend(buckets.get(b, []))
    # include any truly unknown buckets (should be rare)
    for b in sorted([x for x in buckets.keys() if x >= len(_CANON_PREFIX_ORDER)]):
        out.extend(buckets[b])
    return _dedupe_preserve(out)

def enforce_sentence_budget(lines: List[str], max_sentences: int = 5) -> List[str]:
    if not isinstance(lines, list):
        return []
    max_s = max(1, int(max_sentences or 5))
    return [ _safe_text(x) for x in lines if _safe_text(x) ][:max_s]

def count_numbers(s: str) -> int:
    return len(_NUM_RE.findall(_safe_text(s)))

def validate_narrative(
    keys: List[str],
    ctx: Dict[str, Any],
    max_sentences: int = 5,
    must_include_uncertainty: bool = True,
    max_numbers_per_sentence: int = 2,
) -> Tuple[List[str], List[str], List[str]]:
    """Return (final_keys, final_lines, warnings)."""
    final_keys = normalize_keys(keys or [], must_include_uncertainty=must_include_uncertainty)
    lines = resolve_phrases_vi(final_keys, ctx or {})
    final_lines = enforce_sentence_budget(lines, max_sentences=max_sentences)

    warnings: List[str] = []

    if must_include_uncertainty and "SAFETY_LINE" not in final_keys:
        warnings.append("MISSING_SAFETY_LINE")

    # numeric density
    maxn = int(max_numbers_per_sentence or 2)
    for i, ln in enumerate(final_lines):
        if count_numbers(ln) > maxn:
            warnings.append(f"TOO_MANY_NUMBERS:L{i+1}:{count_numbers(ln)}")

    # key order sanity: ensure canonical progression
    # (DNA before ZONE before BIAS before SIZE before DECISION before SAFETY)
    seen_bucket = -1
    for k in final_keys:
        b = _bucket_key(k)
        if b < seen_bucket:
            warnings.append("KEY_ORDER_VIOLATION")
            break
        seen_bucket = b

    return final_keys, final_lines, warnings
