"""INCEPTION module package.

This package provides pluggable report/section generators.

Important:
- Each module file should register itself via inception.modules.base.register(...)
- Registration happens at import time, therefore callers should import modules
  (or call load_default_modules()) before invoking run_modules().
"""

from __future__ import annotations

from typing import Iterable, List, Optional


_DEFAULT_MODULES: List[str] = [
    "character",
    "fundamental",
    "news",
    "portfolio",
    "report_ad",
    "slots",
]


def load_default_modules(extra: Optional[Iterable[str]] = None) -> None:
    """Import and register all built-in modules.

    This is intentionally side-effectful: importing a module triggers its
    register(...) call.
    """
    names: List[str] = list(_DEFAULT_MODULES)
    if extra:
        for x in extra:
            if x and str(x).strip() and str(x) not in names:
                names.append(str(x))

    for name in names:
        __import__(f"inception.modules.{name}", fromlist=["*"])
