"""Base types and shared scalar helpers for industry catalog normalizers.

All imports are stdlib-only — no imports from website_crawl or Django.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urljoin, urlparse, urlunparse

# -----------------------------------------------------------------------
# Sampling-limit constants shared across industries
# -----------------------------------------------------------------------

MAX_IMAGES_PER_ENTITY = 6
MAX_TAGS_PER_ENTITY = 20
MAX_VARIANTS_PER_PRODUCT = 25
MAX_AMENITIES = 30

# -----------------------------------------------------------------------
# Scalar-coercion helpers
# -----------------------------------------------------------------------

_PRICE_RE = re.compile(r"-?\d[\d.,\s\u00a0]*")
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_CURRENCY_SYMBOLS = {
    "$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR",
    "₺": "TRY", "₽": "RUB", "₩": "KRW", "₪": "ILS", "﷼": "SAR", "₨": "PKR",
}
_CURRENCY_CODES = frozenset(
    (
        "USD", "EUR", "GBP", "JPY", "INR", "TRY", "RUB", "AED", "SAR", "IRR", "QAR",
        "KWD", "BHD", "OMR", "EGP", "JOD", "ILS", "CAD", "AUD", "NZD", "CHF", "SEK",
        "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "CNY", "KRW", "HKD", "SGD", "MYR",
        "THB", "IDR", "PHP", "VND", "BRL", "MXN", "ARS", "CLP", "COP", "ZAR", "NGN",
        "KES", "MAD", "TND", "PKR", "BDT", "LKR", "UAH",
    )
)


def _text(value: Any, limit: int | None = None) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        parts = [_text(item) for item in value]
        value = " ".join(part for part in parts if part)
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "n/a", "-", "unknown"):
        return None
    return text[:limit] if limit else text


def _price(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("price", "amount", "value", "lowPrice", "minPrice", "highPrice"):
            if key in value:
                found = _price(value[key])
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _price(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, str):
        return None

    match = _PRICE_RE.search(value.replace("\u00a0", " "))
    if not match:
        return None
    raw = match.group(0).strip().replace(" ", "")
    if "," in raw and "." in raw:
        raw = raw.replace(",", "") if raw.rindex(".") > raw.rindex(",") else raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        tail = raw.rsplit(",", 1)[1]
        raw = raw.replace(",", ".") if len(tail) in (1, 2) else raw.replace(",", "")
    raw = raw.rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def _currency(value: Any, *, fallback: str | None = None) -> str | None:
    text = _text(value)
    if not text:
        return fallback
    upper = text.upper()
    for code in _CURRENCY_CODES:
        if re.search(rf"\b{code}\b", upper):
            return code
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return fallback


def _int(value: Any, *, minimum: int = 0, maximum: int = 10_000) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = int(value)
    else:
        match = re.search(r"\d+", str(value))
        if not match:
            return None
        number = int(match.group(0))
    return number if minimum <= number <= maximum else None


def _bool(value: Any) -> bool | None:
    """Only an explicit signal becomes a boolean; anything vague stays None."""
    if isinstance(value, bool):
        return value
    text = _text(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in ("true", "yes", "y", "1", "in stock", "instock", "available", "bookable"):
        return True
    if lowered in ("false", "no", "n", "0", "out of stock", "outofstock", "sold out", "unavailable"):
        return False
    return None


def _slugify(value: str | None) -> str | None:
    if not value:
        return None
    slug = _SLUG_STRIP_RE.sub("-", value.lower()).strip("-")
    return slug[:120] or None


def _str_list(value: Any, limit: int) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, 160)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _images(value: Any, limit: int = MAX_IMAGES_PER_ENTITY) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            url = _text(item.get("url") or item.get("image_url") or item.get("src"), 1000)
            alt = _text(item.get("alt") or item.get("name"), 200)
        else:
            url, alt = _text(item, 1000), None
        if not url or not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append({"url": url, "alt": alt})
        if len(out) >= limit:
            break
    return out


def _options_map(value: Any) -> dict[str, str]:
    """Variant options as ``{"Color": "Black"}``, accepting Shopify-style lists too."""
    if isinstance(value, dict):
        pairs = value.items()
    elif isinstance(value, (list, tuple)):
        pairs = [
            (item.get("name") or item.get("option"), item.get("value"))
            for item in value
            if isinstance(item, dict)
        ]
    else:
        return {}
    out: dict[str, str] = {}
    for name, option_value in pairs:
        key, val = _text(name, 60), _text(option_value, 120)
        if key and val:
            out[key] = val
    return out


# -----------------------------------------------------------------------
# Private URL utilities (stdlib-only) used by _identity / _dedupe_key
# -----------------------------------------------------------------------

def _url_slug(url: str | None) -> str | None:
    """Last meaningful path segment — natural source_id for a crawled entity."""
    if not url:
        return None
    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    return segments[-1] if segments else None


def _resolve_link(base_url: str, href: str) -> str | None:
    """Absolute URL for href relative to base_url, or None if not an HTTP page link."""
    href = (href or "").strip()
    if not href:
        return None
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    return urlunparse(parsed._replace(fragment=""))


def _canonical_url(url: str) -> str:
    """Stable string key for URL de-duplication (strips fragment, normalises case)."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


# -----------------------------------------------------------------------
# Entity identity helpers (used by both ecommerce and hospitality)
# -----------------------------------------------------------------------

def _identity(raw: dict[str, Any], name: str, *, base_url: str) -> tuple[str | None, str]:
    """Return (source_url, source_id) for a crawled entity."""
    raw_url = _text(raw.get("source_url") or raw.get("url"), 1000)
    source_url = _resolve_link(base_url, raw_url) if raw_url else None
    source_id = _text(raw.get("source_id"), 200) or _url_slug(source_url) or _slugify(name)
    return source_url, source_id or _slugify(name) or name[:200]


def _entity_currency(raw: dict[str, Any], *, fallback: str | None) -> str | None:
    """Currency from the currency field, else inferred from the price string."""
    return _currency(raw.get("currency")) or _currency(raw.get("price"), fallback=fallback)


def _dedupe_key(source_url: str | None, source_id: str) -> str:
    if source_url:
        try:
            return _canonical_url(source_url)
        except Exception:
            return source_url
    return f"id:{source_id.lower()}"


# -----------------------------------------------------------------------
# Shared LLM prompt rules injected into every extraction prompt
# -----------------------------------------------------------------------

_SHARED_PROMPT_RULES = """
Rules:
- Respond with a single JSON object only, no markdown fences.
- Use only information visibly present on the page (text, links, images, JSON-LD). Never invent entities, prices, or descriptions.
- Prefer the schema.org JSON-LD block when it disagrees with the rendered text.
- Return absolute https URLs exactly as they appear in the LINKS/IMAGES sections.
- Use null for unknown scalars and [] for unknown lists. Omitting an entity is always better than guessing one.
- Do not emit navigation chrome (login, cart, blog, careers, newsletter) as catalog entities.
"""


# -----------------------------------------------------------------------
# CatalogSpec dataclass
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogSpec:
    industry: str
    label: str
    aliases: tuple[str, ...]
    business_description: str
    link_guidance: str
    item_key: str
    group_key: str | None
    extra_keys: tuple[str, ...]
    listing_hints: tuple[str, ...]
    detail_hints: tuple[str, ...]
    jsonld_item_types: frozenset[str]
    listing_prompt: str
    detail_prompt: str
    max_listing_pages: int
    max_detail_pages: int
    max_items_per_listing: int
    normalizer: Callable[[dict[str, Any], str], dict[str, Any]]

    def normalize(self, draft: dict[str, Any], *, base_url: str) -> dict[str, Any]:
        return self.normalizer(draft, base_url)

    def empty_catalog(self, base_url: str) -> dict[str, Any]:
        return self.normalize({}, base_url=base_url)

    def entity_keys(self) -> tuple[str, ...]:
        keys = [self.item_key, *self.extra_keys]
        if self.group_key:
            keys.append(self.group_key)
        return tuple(keys)
