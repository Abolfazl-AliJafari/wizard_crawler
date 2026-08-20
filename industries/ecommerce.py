"""Ecommerce industry: normalizer, prompts, and ECOMMERCE_SPEC."""
from __future__ import annotations

from typing import Any

from .base import (
    CatalogSpec,
    MAX_TAGS_PER_ENTITY,
    MAX_VARIANTS_PER_PRODUCT,
    _SHARED_PROMPT_RULES,
    _text,
    _price,
    _currency,
    _bool,
    _str_list,
    _images,
    _options_map,
    _slugify,
    _url_slug,
    _dedupe_key,
    _identity,
    _entity_currency,
)

# -----------------------------------------------------------------------
# Sampling limits
# -----------------------------------------------------------------------

ECOMMERCE_MAX_CATEGORIES = 5
ECOMMERCE_MAX_PRODUCTS_PER_CATEGORY = 5
ECOMMERCE_MAX_PRODUCTS_TOTAL = 50
ECOMMERCE_MAX_LISTING_PAGES = 4
ECOMMERCE_MAX_DETAIL_PAGES = 6


# -----------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------

def _normalize_variants(
    value: Any,
    *,
    product_source_id: str,
    product_sku: str | None,
    product_price: float | None,
    product_in_stock: bool | None,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value if isinstance(value, (list, tuple)) else []):
        if not isinstance(raw, dict):
            continue
        options = _options_map(raw.get("options") or raw.get("selected_options"))
        name = _text(raw.get("name") or raw.get("title"), 255) or (
            " / ".join(options.values()) if options else None
        )
        if not name and not options:
            continue
        source_id = _text(raw.get("source_id"), 200) or _slugify(f"{product_source_id}-{name or index}")
        if source_id in seen:
            continue
        seen.add(source_id)
        variants.append(
            {
                "variant_id": None,
                "source_id": source_id,
                "sku": _text(raw.get("sku"), 100),
                "name": name,
                "options": options,
                "price": _price(raw.get("price")),
                "in_stock": _bool(raw.get("in_stock") or raw.get("availability")),
            }
        )
        if len(variants) >= MAX_VARIANTS_PER_PRODUCT:
            break

    if variants:
        return variants

    # Merge rule 3: a simple product still exposes exactly one buyable variant.
    return [
        {
            "variant_id": None,
            "source_id": product_source_id,
            "sku": product_sku,
            "name": None,
            "options": {},
            "price": product_price,
            "in_stock": product_in_stock,
        }
    ]


def _normalize_filter_schema(value: Any, *, fallback_currency: str | None) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    variant_options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw.get("variant_options") or []:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 60)
        values = _str_list(item.get("values"), 40)
        if not name or not values or name.lower() in seen:
            continue
        seen.add(name.lower())
        variant_options.append(
            {
                "name": name,
                "filter_key": _text(item.get("filter_key"), 60) or name,
                "values": values,
            }
        )

    price_range = raw.get("price_range") if isinstance(raw.get("price_range"), dict) else {}
    return {
        "variant_options": variant_options,
        "tags": _str_list(raw.get("tags"), MAX_TAGS_PER_ENTITY),
        "vendors": _str_list(raw.get("vendors"), MAX_TAGS_PER_ENTITY),
        "price_range": {
            "min": _price(price_range.get("min")),
            "max": _price(price_range.get("max")),
            "currency": _currency(price_range.get("currency"), fallback=fallback_currency),
        },
    }


def _observe_filter_schema(category: dict[str, Any], products: list[dict[str, Any]]) -> None:
    """Fill gaps in a category filter schema from the products actually crawled."""
    members = [
        product
        for product in products
        if category["source_id"] in (product.get("source_category_ids") or [])
    ]
    if not members:
        return

    schema = category["filter_schema"]

    if not schema["variant_options"]:
        collected: dict[str, list[str]] = {}
        for product in members:
            for variant in product["variants"]:
                for name, value in variant["options"].items():
                    values = collected.setdefault(name, [])
                    if value not in values:
                        values.append(value)
        schema["variant_options"] = [
            {"name": name, "filter_key": name, "values": values[:40]}
            for name, values in list(collected.items())[:10]
        ]

    if not schema["tags"]:
        schema["tags"] = _str_list([tag for product in members for tag in product["tags"]], MAX_TAGS_PER_ENTITY)
    if not schema["vendors"]:
        schema["vendors"] = _str_list([product["vendor"] for product in members], MAX_TAGS_PER_ENTITY)

    prices = [product["price"] for product in members if product["price"] is not None]
    if prices:
        price_range = schema["price_range"]
        if price_range["min"] is None:
            price_range["min"] = min(prices)
        if price_range["max"] is None:
            price_range["max"] = max(prices)
        if price_range["currency"] is None:
            price_range["currency"] = next(
                (product["currency"] for product in members if product["currency"]), None
            )


def _normalize_ecommerce(draft: dict[str, Any], base_url: str) -> dict[str, Any]:
    store_raw = draft.get("store") if isinstance(draft.get("store"), dict) else {}
    store_currency = _currency(store_raw.get("currency"))

    categories: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for raw in draft.get("categories") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 255)
        if not name:
            continue
        source_url, source_id = _identity(raw, name, base_url=base_url)
        key = _dedupe_key(source_url, source_id)
        if key in seen_categories:
            continue
        seen_categories.add(key)
        kind = (_text(raw.get("kind")) or "").lower()
        categories.append(
            {
                "id": None,
                "parent_id": None,
                "source_parent_id": _text(raw.get("source_parent_id"), 200),
                "name": name,
                "slug": _text(raw.get("slug"), 200) or _url_slug(source_url) or _slugify(name),
                "kind": kind if kind in ("category", "collection") else "category",
                "source_id": source_id,
                "source_url": source_url,
                "description": _text(raw.get("description"), 2000),
                "filter_schema": _normalize_filter_schema(
                    raw.get("filter_schema"), fallback_currency=store_currency
                ),
            }
        )
        if len(categories) >= ECOMMERCE_MAX_CATEGORIES:
            break

    known_category_ids = {category["source_id"] for category in categories}

    products: list[dict[str, Any]] = []
    seen_products: set[str] = set()
    for raw in draft.get("products") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 255)
        if not name:
            continue
        source_url, source_id = _identity(raw, name, base_url=base_url)
        key = _dedupe_key(source_url, source_id)
        if key in seen_products:
            continue
        seen_products.add(key)

        price = _price(raw.get("price"))
        sku = _text(raw.get("sku"), 100)
        in_stock = _bool(raw.get("in_stock") or raw.get("availability"))
        products.append(
            {
                "product_id": None,
                "source_id": source_id,
                "source_url": source_url,
                "sku": sku,
                "name": name,
                "handle": _text(raw.get("handle"), 200) or _url_slug(source_url) or _slugify(name),
                "vendor": _text(raw.get("vendor"), 160),
                "product_type": _text(raw.get("product_type"), 160),
                "tags": _str_list(raw.get("tags"), MAX_TAGS_PER_ENTITY),
                "category_ids": [],
                "source_category_ids": [
                    category_id
                    for category_id in _str_list(raw.get("source_category_ids"), 10)
                    if category_id in known_category_ids
                ],
                "description": _text(raw.get("description"), 6000),
                "price": price,
                "currency": _entity_currency(raw, fallback=store_currency),
                "in_stock": in_stock,
                "images": _images(raw.get("images")),
                "variants": _normalize_variants(
                    raw.get("variants"),
                    product_source_id=source_id,
                    product_sku=sku,
                    product_price=price,
                    product_in_stock=in_stock,
                ),
            }
        )
        if len(products) >= ECOMMERCE_MAX_PRODUCTS_TOTAL:
            break

    for category in categories:
        _observe_filter_schema(category, products)

    if store_currency is None:
        store_currency = next((product["currency"] for product in products if product["currency"]), None)

    return {
        "industry": "ecommerce",
        "provider": None,
        "store": {
            "url": _text(store_raw.get("url"), 1000) or base_url,
            "name": _text(store_raw.get("name"), 255),
            "currency": store_currency,
        },
        "categories": categories,
        "products": products,
    }


# -----------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------

ECOMMERCE_LISTING_PROMPT = f"""
You read one page of an online store and report the product-catalog entities on it.
The page may be a category/collection listing, a "shop all" index, or a search result page.

Return exactly this shape:

{{
  "category": {{
    "name": "string or null — the category/collection this page represents",
    "slug": "string or null",
    "kind": "category | collection",
    "description": "string or null",
    "filter_schema": {{
      "variant_options": [{{"name": "Color", "filter_key": "Color", "values": ["Black", "White"]}}],
      "tags": ["string"],
      "vendors": ["string"],
      "price_range": {{"min": 0, "max": 0, "currency": "USD"}}
    }}
  }},
  "products": [
    {{
      "name": "string",
      "url": "string — the product detail page URL",
      "sku": "string or null",
      "price": 0,
      "currency": "string or null",
      "vendor": "string or null",
      "product_type": "string or null",
      "tags": ["string"],
      "description": "string or null",
      "images": [{{"url": "string", "alt": "string or null"}}]
    }}
  ],
  "child_listing_urls": ["string — sub-category or sub-collection listing pages linked from here"]
}}

Extra rules:
- "products" comes from the product cards on this page. Include every distinct card, up to 40.
- filter_schema describes the visible filter/facet controls (Color, Size, Brand, price slider). Leave lists empty when the page has no filters.
- If this page is only an index of other categories and shows no product cards, return "products": [] and fill "child_listing_urls".
- Never list a category or collection URL inside "products".
{_SHARED_PROMPT_RULES}
"""

ECOMMERCE_DETAIL_PROMPT = f"""
You read one product detail page of an online store and extract that single product.

Return exactly this shape:

{{
  "product": {{
    "name": "string",
    "description": "string or null — the full product description",
    "sku": "string or null",
    "handle": "string or null — URL slug",
    "vendor": "string or null — brand",
    "product_type": "string or null",
    "tags": ["string"],
    "price": 0,
    "currency": "string or null",
    "in_stock": true,
    "images": [{{"url": "string", "alt": "string or null"}}],
    "variants": [
      {{
        "name": "string — e.g. Black / 42",
        "sku": "string or null",
        "options": {{"Color": "Black", "Size": "42"}},
        "price": 0,
        "in_stock": true
      }}
    ]
  }}
}}

Extra rules:
- variants come from the selectable option combinations (colour swatches, size buttons, dropdowns). Emit one entry per concrete combination the page exposes, up to 25.
- Put option values in variants[].options only. Never add "color" or "size" as top-level product fields.
- Set in_stock only when the page states availability; otherwise null.
{_SHARED_PROMPT_RULES}
"""


# -----------------------------------------------------------------------
# Spec instance
# -----------------------------------------------------------------------

ECOMMERCE_SPEC = CatalogSpec(
    industry="ecommerce",
    label="Ecommerce",
    aliases=(
        "ecommerce", "e commerce", "ecomm", "online store", "online shop", "online retail",
        "webshop", "web shop", "web store", "retail", "shop", "store", "marketplace",
        "dropshipping", "d2c", "b2c retail",
    ),
    business_description="an online store that sells physical or digital products",
    link_guidance=(
        "Catalog pages for an online store are category pages, collection pages, "
        '"shop all" / "all products" indexes, brand pages, and product detail pages. '
        "Typical paths look like /collections/..., /category/..., /shop/..., /product/..., "
        "/products/..., /p/..., but many stores use custom paths — judge by the link label too."
    ),
    item_key="products",
    group_key="categories",
    extra_keys=(),
    listing_hints=(
        "collection", "collections", "category", "categories", "catalog", "catalogue",
        "shop", "store", "products", "all-products", "brand", "brands", "department",
        "range", "shop-all", "new-in", "bestseller", "best-sellers",
    ),
    detail_hints=("product", "products", "/p/", "item", "sku", "dp"),
    jsonld_item_types=frozenset({"product", "productgroup", "itemlist", "offercatalog"}),
    listing_prompt=ECOMMERCE_LISTING_PROMPT,
    detail_prompt=ECOMMERCE_DETAIL_PROMPT,
    max_listing_pages=ECOMMERCE_MAX_LISTING_PAGES,
    max_detail_pages=ECOMMERCE_MAX_DETAIL_PAGES,
    max_items_per_listing=ECOMMERCE_MAX_PRODUCTS_PER_CATEGORY,
    normalizer=_normalize_ecommerce,
)

# Canonical export name used by auto-discovery in industries/__init__.py
SPEC = ECOMMERCE_SPEC

# BM25 query string used by crawl4ai content filter for this industry
BM25_QUERY = (
    "product price buy cart checkout shipping return refund warranty brand description features"
)
