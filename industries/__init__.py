"""Industries package — auto-discovers all industry modules in this directory.

To add a new industry:
  1. Create  industries/<name>.py
  2. Define  SPEC = CatalogSpec(...)   in that file
  3. Optionally define  BM25_QUERY = "..."
  That's it — no changes needed anywhere else.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .base import CatalogSpec, _WORD_RE

# -----------------------------------------------------------------------
# Auto-discover all industry modules (every .py except __init__ and base)
# -----------------------------------------------------------------------

_SKIP = {"__init__", "base"}

CATALOG_SPECS: tuple[CatalogSpec, ...] = ()
_C4A_INDUSTRY_QUERIES: dict[str, str] = {
    "default": "services products prices hours location contact about policies team",
}

_pkg_path = str(Path(__file__).parent)
for _mod_info in pkgutil.iter_modules([_pkg_path]):
    if _mod_info.name in _SKIP:
        continue
    _mod = importlib.import_module(f".{_mod_info.name}", package=__name__)
    if hasattr(_mod, "SPEC") and isinstance(_mod.SPEC, CatalogSpec):
        CATALOG_SPECS = (*CATALOG_SPECS, _mod.SPEC)
    if hasattr(_mod, "BM25_QUERY") and isinstance(_mod.BM25_QUERY, str):
        _C4A_INDUSTRY_QUERIES[_mod.SPEC.industry] = _mod.BM25_QUERY

SUPPORTED_INDUSTRY_LABELS: tuple[str, ...] = tuple(s.label for s in CATALOG_SPECS)


# -----------------------------------------------------------------------
# Spec resolver
# -----------------------------------------------------------------------

def resolve_catalog_spec(industry: str | None) -> CatalogSpec | None:
    """Best-effort map of a free-text industry name onto a CatalogSpec."""
    if not industry:
        return None
    words = set(_WORD_RE.findall(industry.lower()))
    normalized = " ".join(_WORD_RE.findall(industry.lower()))
    if not normalized:
        return None

    best: tuple[int, CatalogSpec] | None = None
    for spec in CATALOG_SPECS:
        for alias in spec.aliases:
            matched = alias in normalized if " " in alias else alias in words
            if matched and (best is None or len(alias) > best[0]):
                best = (len(alias), spec)
    return best[1] if best else None


# -----------------------------------------------------------------------
# BM25 query helper — used by crawl4ai content filter
# -----------------------------------------------------------------------

def _c4a_query(industry: str | None) -> str:
    if not industry:
        return _C4A_INDUSTRY_QUERIES["default"]
    key = industry.lower().strip()
    for k, q in _C4A_INDUSTRY_QUERIES.items():
        if k in key:
            return q
    return _C4A_INDUSTRY_QUERIES["default"]
