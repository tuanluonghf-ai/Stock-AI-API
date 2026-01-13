from __future__ import annotations

import hashlib
from typing import List


def _stable_index(key: str, n: int) -> int:
    """Return a stable index in [0, n) derived from key."""
    if n <= 0:
        return 0
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    # take 8 hex chars -> int
    v = int(h[:8], 16)
    return v % n


def pick_phrase(*, key: str, variants: List[str], default: str = "") -> str:
    """Pick a phrase deterministically from variants using a stable hash of key."""
    if not variants:
        return default
    i = _stable_index(key, len(variants))
    s = variants[i].strip()
    return s or default
