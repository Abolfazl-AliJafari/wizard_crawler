"""Hospitality industry: normalizer, prompts, and HOSPITALITY_SPEC."""
from __future__ import annotations

from typing import Any

from .base import (
    CatalogSpec,
    MAX_IMAGES_PER_ENTITY,
    MAX_TAGS_PER_ENTITY,
    MAX_AMENITIES,
    _SHARED_PROMPT_RULES,
    _text,
    _price,
    _currency,
    _bool,
    _int,
    _str_list,
    _images,
    _url_slug,
    _slugify,
    _dedupe_key,
    _identity,
    _entity_currency,
)

# -----------------------------------------------------------------------
# Sampling limits
# -----------------------------------------------------------------------

HOSPITALITY_MAX_ROOM_TYPES = 5
HOSPITALITY_MAX_ROOMS_PER_TYPE = 3
HOSPITALITY_MAX_RATES = 5
HOSPITALITY_MAX_FACILITIES = 5
HOSPITALITY_MAX_LISTING_PAGES = 4
HOSPITALITY_MAX_DETAIL_PAGES = 10


# -----------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------

def _normalize_rooms(value: Any, *, room_type_source_id: str) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value if isinstance(value, (list, tuple)) else []:
        if not isinstance(raw, dict):
            continue
        room_number = _text(raw.get("room_number") or raw.get("number"), 40)
        source_id = _text(raw.get("source_id"), 200) or room_number
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        rooms.append(
            {
                "room_id": None,
                "source_id": source_id,
                "room_number": room_number,
                "floor": _text(raw.get("floor"), 40),
                "description": _text(raw.get("description"), 2000),
                "room_type_id": None,
                "source_room_type_id": room_type_source_id,
            }
        )
        if len(rooms) >= HOSPITALITY_MAX_ROOMS_PER_TYPE:
            break
    return rooms


def _normalize_resource_categories(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value if isinstance(value, (list, tuple)) else []:
        if isinstance(raw, dict):
            name = _text(raw.get("name"), 255)
            capacity = _int(raw.get("capacity"))
        else:
            name, capacity = _text(raw, 255), None
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"room_type_id": None, "name": name, "capacity": capacity})
        if len(out) >= 12:
            break
    return out


def _normalize_hospitality(draft: dict[str, Any], base_url: str) -> dict[str, Any]:
    property_raw = draft.get("property") if isinstance(draft.get("property"), dict) else {}
    property_currency = _currency(property_raw.get("currency"))

    room_types: list[dict[str, Any]] = []
    seen_room_types: set[str] = set()
    for raw in draft.get("room_types") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 255)
        if not name:
            continue
        source_url, source_id = _identity(raw, name, base_url=base_url)
        key = _dedupe_key(source_url, source_id)
        name_key = name.lower()
        if key in seen_room_types or name_key in seen_room_types:
            continue
        seen_room_types.add(key)
        seen_room_types.add(name_key)
        capacity = _int(raw.get("capacity"), maximum=99)
        occupancy = _int(raw.get("default_max_occupancy"), maximum=99)
        room_types.append(
            {
                "room_type_id": None,
                "source_id": source_id,
                "source_url": source_url,
                "name": name,
                "default_max_occupancy": occupancy or capacity,
                "capacity": capacity or occupancy,
                "extra_capacity": _int(raw.get("extra_capacity"), maximum=99),
                "service_id": None,
                "type": _text(raw.get("type") or raw.get("bed_type"), 80),
                "is_active": _bool(raw.get("is_active")),
                "description": _text(raw.get("description"), 6000),
                "amenities": _str_list(raw.get("amenities"), MAX_AMENITIES),
                "images": _images(raw.get("images")),
                "display_price": _price(raw.get("display_price") or raw.get("price")),
                "display_currency": _entity_currency(raw, fallback=property_currency),
                "rooms": _normalize_rooms(raw.get("rooms"), room_type_source_id=source_id),
            }
        )
        if len(room_types) >= HOSPITALITY_MAX_ROOM_TYPES:
            break

    rates: list[dict[str, Any]] = []
    seen_rates: set[str] = set()
    for raw in draft.get("rates") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 255)
        if not name or name.lower() in seen_rates:
            continue
        seen_rates.add(name.lower())
        source_url, source_id = _identity(raw, name, base_url=base_url)
        rates.append(
            {
                "rate_id": None,
                "source_id": source_id,
                "source_url": source_url,
                "name": name,
                "code": _text(raw.get("code"), 60),
                "type": _text(raw.get("type"), 60),
                "is_active": _bool(raw.get("is_active")),
                "is_enabled": _bool(raw.get("is_enabled")),
                "is_public": _bool(raw.get("is_public")),
                "service_id": None,
                "cancellation_summary": _text(raw.get("cancellation_summary"), 2000),
                "description": _text(raw.get("description"), 2000),
                "display_price": _price(raw.get("display_price") or raw.get("price")),
                "display_currency": _entity_currency(raw, fallback=property_currency),
            }
        )
        if len(rates) >= HOSPITALITY_MAX_RATES:
            break

    facilities: list[dict[str, Any]] = []
    seen_facilities: set[str] = set()
    for raw in draft.get("facilities") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 255)
        if not name:
            continue
        source_url, source_id = _identity(raw, name, base_url=base_url)
        key = _dedupe_key(source_url, source_id)
        if key in seen_facilities or name.lower() in seen_facilities:
            continue
        seen_facilities.add(key)
        seen_facilities.add(name.lower())
        facilities.append(
            {
                "facility_id": None,
                "service_id": None,
                "source_id": source_id,
                "source_url": source_url,
                "name": name,
                "category": _text(raw.get("category"), 80) or "bookable_service",
                "bookable": _bool(raw.get("bookable")),
                "resource_categories": _normalize_resource_categories(raw.get("resource_categories")),
                "description": _text(raw.get("description"), 4000),
                "duration_minutes": _int(raw.get("duration_minutes"), maximum=100_000),
                "images": _images(raw.get("images")),
            }
        )
        if len(facilities) >= HOSPITALITY_MAX_FACILITIES:
            break

    age_categories: list[dict[str, Any]] = []
    seen_ages: set[str] = set()
    for raw in draft.get("age_categories") or []:
        if not isinstance(raw, dict):
            continue
        name = _text(raw.get("name"), 80)
        if not name or name.lower() in seen_ages:
            continue
        seen_ages.add(name.lower())
        role = (_text(raw.get("role")) or "").lower()
        age_categories.append(
            {
                "age_category_id": None,
                "name": name,
                "classification": _text(raw.get("classification"), 80),
                "role": role if role in ("adult", "child") else None,
                "min_age": _int(raw.get("min_age"), maximum=130),
                "max_age": _int(raw.get("max_age"), maximum=130),
                "is_active": _bool(raw.get("is_active")),
            }
        )
        if len(age_categories) >= 6:
            break

    if property_currency is None:
        property_currency = next(
            (room["display_currency"] for room in room_types if room["display_currency"]), None
        )

    return {
        "industry": "hospitality",
        "provider": None,
        "property": {
            "hotel_id": None,
            "name": _text(property_raw.get("name"), 255),
            "url": _text(property_raw.get("url"), 1000) or base_url,
            "timezone": _text(property_raw.get("timezone"), 80),
            "currency": property_currency,
            "language": _text(property_raw.get("language"), 20),
            "accommodation_service_id": None,
            "pricing_mode": None,
        },
        "room_types": room_types,
        "rates": rates,
        "age_categories": age_categories,
        "facilities": facilities,
    }


# -----------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------

HOSPITALITY_PROMPT = f"""
You read one page of a hotel / lodging website and report the bookable entities on it.
The page may be a rooms index, a single room-type page, a rates page, or a facility/spa page.

Return exactly this shape:

{{
  "property": {{
    "name": "string or null",
    "timezone": "string or null — IANA name only if stated",
    "currency": "string or null",
    "language": "string or null"
  }},
  "room_types": [
    {{
      "name": "string — e.g. Deluxe King",
      "url": "string or null — the room-type detail page",
      "description": "string or null",
      "default_max_occupancy": 0,
      "capacity": 0,
      "extra_capacity": 0,
      "type": "string or null — bed type, e.g. Bed, Twin, King",
      "amenities": ["string"],
      "price": 0,
      "currency": "string or null",
      "images": [{{"url": "string", "alt": "string or null"}}],
      "rooms": [{{"room_number": "string", "floor": "string or null", "description": "string or null"}}]
    }}
  ],
  "rates": [
    {{
      "name": "string — e.g. Fully Flexible, Bed & Breakfast",
      "code": "string or null",
      "type": "string or null",
      "cancellation_summary": "string or null",
      "description": "string or null",
      "price": 0,
      "currency": "string or null"
    }}
  ],
  "facilities": [
    {{
      "name": "string — e.g. Spa, Meeting Room, Restaurant",
      "url": "string or null",
      "category": "string or null",
      "bookable": true,
      "description": "string or null",
      "duration_minutes": 0,
      "resource_categories": [{{"name": "string — a space inside the facility", "capacity": 0}}],
      "images": [{{"url": "string", "alt": "string or null"}}]
    }}
  ],
  "age_categories": [
    {{"name": "string — Adult or Child", "role": "adult | child", "min_age": 0, "max_age": 0}}
  ],
  "child_listing_urls": ["string — further room, rate, or facility listing pages linked from here"]
}}

Extra rules:
- A room type is a sellable stay product (Deluxe, Suite, Twin). Physical numbered rooms go in room_types[].rooms only when the page actually lists them.
- Rate plans are board/cancellation packages (Room Only, B&B, Fully Flexible, Non-Refundable), not room types.
- Facilities are extras bookable alongside a stay (spa, meeting room, restaurant, tour). Do not put room types here.
- Nightly prices are display-only; report them as seen and never compute or estimate them.
- Only emit age_categories when the page states an explicit adult/child age policy.
- Return [] for every array the page says nothing about.
{_SHARED_PROMPT_RULES}
"""


# -----------------------------------------------------------------------
# Spec instance
# -----------------------------------------------------------------------

HOSPITALITY_SPEC = CatalogSpec(
    industry="hospitality",
    label="Hospitality",
    aliases=(
        "hospitality", "hotel", "hotels", "motel", "resort", "hostel", "guest house",
        "guesthouse", "bed and breakfast", "b and b", "lodging", "lodge", "accommodation",
        "boutique hotel", "apart hotel", "aparthotel", "serviced apartment", "villa rental",
        "vacation rental", "holiday rental", "inn",
    ),
    business_description="a hotel or lodging property that sells overnight stays and related services",
    link_guidance=(
        "Catalog pages for a lodging property are room / room-type / suite / accommodation "
        "pages, rate or package pages, and bookable extras such as spa, wellness, meeting "
        "rooms, dining, and activities. Typical paths look like /rooms/..., /accommodation/..., "
        "/suites/..., /rates/..., /offers/..., /packages/..., /spa, /facilities, /meetings, "
        "but judge by the link label too."
    ),
    item_key="room_types",
    group_key=None,
    extra_keys=("rates", "facilities", "age_categories"),
    listing_hints=(
        "room", "rooms", "roomtype", "room-types", "accommodation", "accommodations",
        "suite", "suites", "apartment", "apartments", "stay", "rate", "rates", "package",
        "packages", "offer", "offers", "facility", "facilities", "amenities", "spa",
        "wellness", "meeting", "meetings", "conference", "dining", "restaurant", "services",
        "experiences", "activities",
    ),
    detail_hints=("room", "suite", "accommodation", "apartment", "spa", "facility", "rate", "package"),
    jsonld_item_types=frozenset(
        {
            "hotel", "lodgingbusiness", "resort", "bedandbreakfast", "hostel", "motel",
            "hotelroom", "room", "suite", "accommodation", "apartment", "campground",
            "touristattraction", "service", "offer", "itemlist",
        }
    ),
    listing_prompt=HOSPITALITY_PROMPT,
    detail_prompt=HOSPITALITY_PROMPT,
    max_listing_pages=HOSPITALITY_MAX_LISTING_PAGES,
    max_detail_pages=HOSPITALITY_MAX_DETAIL_PAGES,
    max_items_per_listing=HOSPITALITY_MAX_ROOM_TYPES,
    normalizer=_normalize_hospitality,
)

# Canonical export name used by auto-discovery in industries/__init__.py
SPEC = HOSPITALITY_SPEC

# BM25 query string used by crawl4ai content filter for this industry
BM25_QUERY = (
    "hotel rooms suites amenities services dining spa pool gym check-in check-out "
    "policies prices rates contact hours location booking reservation cancellation"
)
