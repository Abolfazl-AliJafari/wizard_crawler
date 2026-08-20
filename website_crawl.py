"""Job 7 — Website Crawl.

Runs automatically (in a background thread) after Job 6 when the intake has a
website. It crawls the public site and produces two separate outputs:

``knowledge``
    The existing fixed five-block business knowledge contract — what an AI agent
    should *know* about the business (about, contact, hours, policies, FAQs,
    services and product overviews).

``catalog``
    An industry-shaped sample of what the business actually *sells* — categories
    and products for ecommerce, room types, rates and facilities for hospitality.
    Shaped by ``Ecommerse_CATALOG_STRUCTURE.md`` /
    ``Hospitality_CATALOG_STRUCTURE.md`` so it later merges onto a live store or
    PMS connection field-for-field.

The two never overwrite each other: ``services_overview`` / ``product_overview``
stay descriptive knowledge prose, and the catalog stays structured entities.

Flow::

    website + industry + user_request
      -> fetch landing page
      -> validate the site matches the industry   (1 page, 1 LLM call)
      -> classify links into knowledge + catalog  (1 LLM call)
      -> knowledge pages  -> five fixed blocks
      -> catalog pages    -> industry catalog
      -> persist on WebsiteCrawlJob, trigger Job 8

An industry mismatch stops the workflow before the full crawl.

It must never block agent generation — callers should use
``schedule_website_crawl`` (fire-and-forget) rather than ``execute_website_crawl``.

File structure
--------------
SECTION 1  LLM helpers
SECTION 2  HTTP fetching and HTML parsing
SECTION 3  Industry catalog contracts (normalizers, prompts, registry)
SECTION 4  Industry-aware catalog crawl (validation, link classification, crawl)
SECTION 5  Knowledge crawl and job orchestration (entry points, persistence)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

try:
    from crawl4ai.content_filter_strategy import BM25ContentFilter as _BM25ContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator as _C4AMarkdownGenerator
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    _CRAWL4AI_AVAILABLE = False

def _extract_text_crawl4ai(html: str, url: str, industry: str | None = None) -> str | None:
    """Use crawl4ai BM25ContentFilter to extract the most relevant content as clean markdown.

    Pre-cleans HTML with BeautifulSoup (removes script/style/svg) before passing to
    crawl4ai so the BM25 filter only scores real content blocks.
    Returns None if crawl4ai is unavailable or fails (caller falls back to BeautifulSoup).
    """
    if not _CRAWL4AI_AVAILABLE:
        return None
    try:
        # Pre-clean: strip noise tags so crawl4ai scores actual content
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "svg", "iframe", "link", "meta"]):
            tag.decompose()
        clean_html = str(soup)

        query = _c4a_query(industry)
        gen = _C4AMarkdownGenerator(
            content_filter=_BM25ContentFilter(
                user_query=query,
                bm25_threshold=1.2,
            )
        )
        result = gen.generate_markdown(input_html=clean_html, base_url=url)
        fit = (result.fit_markdown or "").strip()
        # Fall back if crawl4ai filtered out too much
        if len(fit) < 200:
            return None
        return fit
    except Exception as exc:
        logger.debug("[C4A] Text extraction failed for %s: %s", url, exc)
        return None
from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from apps.wizard.constants.choices import CrawlStatus, KnowledgeStatus
from apps.wizard.integrations.knowledge import (
    KNOWLEDGE_SUBSECTION_KEYS,
    normalize_knowledge_blocks,
)
from apps.wizard.jobs.base import JobResult, JobTrigger, WizardJob
from apps.wizard.jobs.exceptions import WizardJobValidationError
from apps.wizard.jobs.helpers import get_intake_or_raise, lines_to_text
from apps.wizard.models import (
    AgentIntake,
    KnowledgeJob,
    WebsiteCrawlFaq,
    WebsiteCrawlJob,
    WebsiteCrawlService,
)

from industries import (
    CATALOG_SPECS,
    SUPPORTED_INDUSTRY_LABELS,
    resolve_catalog_spec,
    _c4a_query,
)
from industries.base import CatalogSpec

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 1 — LLM helpers
# ======================================================================
# Single entry point for every OpenAI call: JSON-mode handling, parse
# failures and model selection all live here so nothing is duplicated.
# ======================================================================

DEFAULT_MODEL = "gpt-4o-mini"

LINK_MODEL_SETTING = "WIZARD_CRAWL_LINK_MODEL"
EXTRACT_MODEL_SETTING = "WIZARD_CRAWL_EXTRACT_MODEL"
VALIDATE_MODEL_SETTING = "WIZARD_CRAWL_VALIDATE_MODEL"
CATALOG_MODEL_SETTING = "WIZARD_CRAWL_CATALOG_MODEL"


def openai_client():
    api_key = getattr(settings, "OPENAI_API_KEY", None) or getattr(settings, "OPENAI_KET", None)
    if not api_key:
        raise WizardJobValidationError("OPENAI_API_KEY is not configured; cannot crawl website.")
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def model_for(setting_name: str) -> str:
    return getattr(settings, setting_name, None) or DEFAULT_MODEL


def chat_json(*, model: str, system: str, user: str) -> dict[str, Any]:
    """Run a JSON-mode completion and return the parsed object ({} on bad JSON)."""
    client = openai_client()
    _t = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    logger.debug("[TIMING] LLM call model=%s: %.2fs", model, time.perf_counter() - _t)
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON content for model=%s", model)
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ======================================================================
# SECTION 2 — HTTP fetching and HTML parsing
# ======================================================================
# URL helpers → page model → HTML parsing → low-level HTTP → PageFetcher
# (cached, budgeted) → sitemap discovery.
#
# The knowledge pass only needs page text, but catalog extraction also
# needs links, images, and schema.org blocks. PageDocument carries every
# signal; PageDocument.body reproduces the legacy text-only view.
#
# All network access goes through PageFetcher, which caches by canonical
# URL so validation, knowledge, and catalog passes share one download per
# page and stay inside a single request budget.
# ======================================================================

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

MAX_PAGE_CHARS = 6000
FETCH_TIMEOUT = 15
FETCH_MAX_RETRIES = 3
FETCH_RETRY_DELAYS = (1.0, 2.0, 4.0)

MAX_LINKS_PER_PAGE = 400
MAX_IMAGES_PER_PAGE = 60
MAX_JSONLD_NODES = 60
MAX_JSONLD_CHARS = 6000

DEFAULT_PAGE_BUDGET = 70

MAX_SITEMAP_URLS = 400
MAX_NESTED_SITEMAPS = 3

_SKIP_PREFIXES = ("#", "mailto:", "tel:", "sms:", "javascript:", "whatsapp:", "data:")
_SKIP_EXTENSIONS = (
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".rar", ".gz",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".avif", ".ico", ".bmp",
    ".mp3", ".mp4", ".avi", ".mov", ".webm", ".css", ".js", ".woff", ".woff2", ".ttf",
)
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "_ga",
}

_RETRIABLE_STATUS = (408, 429, 500, 502, 503, 504)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


# --- 2.1  URL helpers ---------------------------------------------------

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise WizardJobValidationError("Website URL is required.")
    if not url.lower().startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def canonical_url(url: str) -> str:
    """Stable key for caching and de-duplication."""
    parsed = urlparse(normalize_url(url))
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parsed.query) if key.lower() not in _TRACKING_PARAMS]
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def resolve_link(base_url: str, href: str) -> str | None:
    """Absolute, crawlable URL for ``href``, or None when it is not a page link."""
    href = (href or "").strip()
    if not href or href.lower().startswith(_SKIP_PREFIXES):
        return None
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.path.lower().endswith(_SKIP_EXTENSIONS):
        return None
    return urlunparse(parsed._replace(fragment=""))


def same_host(a: str, b: str) -> bool:
    host_a = urlparse(a).netloc.lower().removeprefix("www.")
    host_b = urlparse(b).netloc.lower().removeprefix("www.")
    return not host_a or not host_b or host_a == host_b


def url_slug(url: str | None) -> str | None:
    """Last meaningful path segment — the natural ``source_id`` for a crawled entity."""
    if not url:
        return None
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    return segments[-1] if segments else None


def site_root(url: str) -> str:
    parsed = urlparse(normalize_url(url))
    return f"{parsed.scheme}://{parsed.netloc}"


# --- 2.2  Page model ----------------------------------------------------

@dataclass(frozen=True)
class PageLink:
    url: str
    text: str = ""
    in_nav: bool = False


@dataclass(frozen=True)
class PageImage:
    url: str
    alt: str | None = None


@dataclass
class PageDocument:
    url: str
    title: str = ""
    meta_description: str = ""
    text: str = ""
    links: list[PageLink] = field(default_factory=list)
    images: list[PageImage] = field(default_factory=list)
    jsonld: list[dict] = field(default_factory=list)

    @property
    def body(self) -> str:
        """Title + text, matching the legacy text-only page representation."""
        return f"{self.title}\n\n{self.text}".strip()

    def link_urls(self) -> list[str]:
        return [link.url for link in self.links]

    def nav_labels(self, limit: int = 40) -> list[str]:
        labels = [link.text for link in self.links if link.in_nav and link.text]
        return _dedupe_preserving_order(labels)[:limit]

    def jsonld_json(self, max_chars: int = MAX_JSONLD_CHARS) -> str:
        if not self.jsonld:
            return ""
        return json.dumps(self.jsonld, ensure_ascii=False)[:max_chars]

    def condensed(
        self,
        *,
        max_text_chars: int = 3500,
        max_links: int = 90,
        max_images: int = 20,
        max_jsonld_chars: int = MAX_JSONLD_CHARS,
    ) -> str:
        """Compact multi-signal rendering of the page for an LLM prompt."""
        sections = [f"URL: {self.url}"]
        if self.title:
            sections.append(f"TITLE: {self.title}")
        if self.meta_description:
            sections.append(f"META DESCRIPTION: {self.meta_description}")

        structured = self.jsonld_json(max_jsonld_chars)
        if structured:
            sections.append(f"STRUCTURED DATA (schema.org JSON-LD):\n{structured}")

        if self.links:
            rendered_links = [
                f"- {link.text or '(no label)'} -> {link.url}" for link in self.links[:max_links]
            ]
            sections.append("LINKS ON PAGE:\n" + "\n".join(rendered_links))

        if self.images:
            rendered_images = [
                f"- {image.alt or '(no alt)'} -> {image.url}" for image in self.images[:max_images]
            ]
            sections.append("IMAGES ON PAGE:\n" + "\n".join(rendered_images))

        if self.text:
            sections.append(f"PAGE TEXT:\n{self.text[:max_text_chars]}")

        return "\n\n".join(sections)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(value.strip())
    return out


# --- 2.3  HTML parsing --------------------------------------------------

def _clean_text(value: str) -> str:
    value = _WHITESPACE_RE.sub(" ", value)
    return _BLANKLINES_RE.sub("\n\n", value).strip()


def _image_src(tag) -> str | None:
    for attribute in ("src", "data-src", "data-original", "data-lazy-src", "data-image"):
        value = (tag.get(attribute) or "").strip()
        if value and not value.startswith("data:"):
            return value
    srcset = (tag.get("srcset") or tag.get("data-srcset") or "").strip()
    if srcset:
        first = srcset.split(",")[0].strip().split(" ")[0]
        if first and not first.startswith("data:"):
            return first
    return None


def _flatten_jsonld(data) -> list[dict]:
    nodes: list[dict] = []
    stack = [data]
    while stack and len(nodes) < MAX_JSONLD_NODES:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, dict):
            graph = current.get("@graph")
            if isinstance(graph, (list, dict)):
                stack.append(graph)
            else:
                nodes.append(current)
    return nodes


def _parse_jsonld(soup: BeautifulSoup) -> list[dict]:
    nodes: list[dict] = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (tag.string or tag.get_text() or "").strip()
        if not raw:
            continue
        try:
            nodes.extend(_flatten_jsonld(json.loads(raw)))
        except json.JSONDecodeError:
            continue
        if len(nodes) >= MAX_JSONLD_NODES:
            break
    return nodes[:MAX_JSONLD_NODES]


def _extract_spa_text(soup: BeautifulSoup) -> str:
    """Extract readable text from JS-framework data blobs (Next.js, Nuxt, Shopify…).

    Many modern SPAs ship all page data inside <script> tags as JSON.
    We pull out string values and join them so the rest of the pipeline has
    something to work with even when the visible HTML body is nearly empty.
    """
    chunks: list[str] = []

    def _walk_json(obj: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, str):
            cleaned = _clean_text(obj)
            if len(cleaned) > 20:
                chunks.append(cleaned)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk_json(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj[:30]:
                _walk_json(item, depth + 1)

    for script in soup.find_all("script"):
        script_id = script.get("id", "")
        script_type = script.get("type", "")
        src = script.get("src", "")
        if src:
            continue
        raw = (script.string or "").strip()
        if not raw:
            continue
        if script_type == "application/ld+json":
            continue

        def _try_parse_json(text: str) -> Any:
            """Parse JSON, stripping Unicode replacement characters if needed."""
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                cleaned = text.replace("\ufffd", " ").replace("\x00", "")
                try:
                    return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    return None

        # Next.js / Nuxt data blobs
        if script_id in ("__NEXT_DATA__", "__NUXT_DATA__", "__NUXT__") or raw.startswith("{"):
            obj = _try_parse_json(raw)
            if obj is not None:
                _walk_json(obj)
            continue

        # Shopify / generic window.* assignments: window.__data__ = {...}
        json_match = re.search(r'=\s*(\{.*\}|\[.*\])\s*;?\s*$', raw, re.S)
        if json_match:
            obj = _try_parse_json(json_match.group(1))
            if obj is not None:
                _walk_json(obj)

    seen: set[str] = set()
    unique: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return "\n".join(unique[:200])


def parse_document(url: str, content: bytes, industry: str | None = None) -> PageDocument:
    soup = BeautifulSoup(content, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_description = ""
    meta_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if meta_tag:
        meta_description = (meta_tag.get("content") or "").strip()

    jsonld = _parse_jsonld(soup)

    nav_hrefs: set[str] = set()
    for container in soup.find_all(["nav", "header"]):
        for anchor in container.find_all("a", href=True):
            nav_hrefs.add(anchor["href"].strip())

    links: list[PageLink] = []
    seen_links: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        absolute = resolve_link(url, href)
        if not absolute:
            continue
        key = canonical_url(absolute)
        if key in seen_links:
            continue
        seen_links.add(key)
        links.append(
            PageLink(
                url=absolute,
                text=_clean_text(anchor.get_text(" ", strip=True))[:120],
                in_nav=href in nav_hrefs,
            )
        )
        if len(links) >= MAX_LINKS_PER_PAGE:
            break

    images: list[PageImage] = []
    seen_images: set[str] = set()
    for tag in soup.find_all("img"):
        src = _image_src(tag)
        if not src:
            continue
        absolute = urljoin(url, src)
        if absolute in seen_images:
            continue
        seen_images.add(absolute)
        alt = _clean_text(tag.get("alt") or "")[:160] or None
        images.append(PageImage(url=absolute, alt=alt))
        if len(images) >= MAX_IMAGES_PER_PAGE:
            break

    spa_text = _extract_spa_text(soup)

    # Try crawl4ai BM25 first — cleaner, ~50% fewer tokens, no navigation noise
    raw_html = content.decode("utf-8", errors="replace")
    text = _extract_text_crawl4ai(raw_html, url, industry=industry)

    if text is None:
        # Fallback: BeautifulSoup plain text extraction
        if soup.body:
            for tag in soup.body(["script", "style", "noscript", "template", "svg"]):
                tag.decompose()
            body_text = _clean_text(soup.body.get_text(separator="\n", strip=True))
        else:
            body_text = ""

        if len(body_text) >= 200:
            text = body_text
        else:
            text = (body_text + "\n" + spa_text).strip()
            logger.debug(
                "[SPA] %s — body=%d chars, spa=%d chars, combined=%d chars",
                url, len(body_text), len(spa_text), len(text),
            )
    else:
        logger.debug("[C4A] %s — fit_markdown=%d chars (BM25 industry=%s)", url, len(text), industry)

    return PageDocument(
        url=url,
        title=title,
        meta_description=meta_description,
        text=text,
        links=links,
        images=images,
        jsonld=jsonld,
    )


# --- 2.4  Low-level HTTP ------------------------------------------------

def _request_is_retriable(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRIABLE_STATUS
    return False


def _http_get(url: str) -> requests.Response:
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=FETCH_TIMEOUT, verify=True)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=FETCH_TIMEOUT, verify=False)
        response.raise_for_status()
        return response


def fetch_response(url: str) -> requests.Response | None:
    """GET with bounded retries."""
    url = normalize_url(url)
    last_exc: Exception | None = None

    for attempt in range(FETCH_MAX_RETRIES):
        try:
            return _http_get(url)
        except Exception as exc:
            last_exc = exc
            if attempt < FETCH_MAX_RETRIES - 1 and _request_is_retriable(exc):
                time.sleep(FETCH_RETRY_DELAYS[attempt])
                continue
            break

    logger.warning("Failed to fetch %s after %s attempts: %s", url, FETCH_MAX_RETRIES, last_exc)
    return None


# --- 2.5  Budgeted, cached fetcher --------------------------------------

class PageFetcher:
    """Caches parsed pages per crawl run and caps total downloads."""

    def __init__(self, *, page_budget: int = DEFAULT_PAGE_BUDGET, industry: str | None = None):
        self.page_budget = page_budget
        self.industry = industry
        self.errors: list[str] = []
        self._cache: dict[str, PageDocument | None] = {}
        self._budget_exhausted_logged = False

    @property
    def pages_attempted(self) -> int:
        return len(self._cache)

    @property
    def pages_fetched(self) -> int:
        return sum(1 for document in self._cache.values() if document is not None)

    @property
    def pages_failed(self) -> int:
        return sum(1 for document in self._cache.values() if document is None)

    @property
    def budget_left(self) -> int:
        return max(self.page_budget - len(self._cache), 0)

    def cached_urls(self) -> list[str]:
        return [document.url for document in self._cache.values() if document is not None]

    def get(self, url: str) -> PageDocument | None:
        try:
            key = canonical_url(url)
        except WizardJobValidationError:
            return None

        if key in self._cache:
            return self._cache[key]

        if self.budget_left <= 0:
            if not self._budget_exhausted_logged:
                logger.info("Crawl page budget of %s reached; skipping %s", self.page_budget, url)
                self._budget_exhausted_logged = True
            return None

        _t = time.perf_counter()
        response = fetch_response(url)
        document = parse_document(url, response.content, industry=self.industry) if response is not None else None
        elapsed = time.perf_counter() - _t
        if document is None:
            logger.debug("[TIMING] Fetch %s: %.2fs (failed)", url, elapsed)
            self.errors.append(f"Could not load page: {url}")
        else:
            logger.debug(
                "[TIMING] Fetch %s: %.2fs | text=%d chars links=%d",
                url, elapsed, len(document.text), len(document.links),
            )
        self._cache[key] = document
        return document


# --- 2.6  Sitemap discovery ---------------------------------------------

def _sitemap_locs(xml: str) -> list[str]:
    return [match.strip() for match in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml, re.I)]


def _fetch_text(url: str) -> str | None:
    response = fetch_response(url)
    if response is None:
        return None
    try:
        return response.text
    except Exception:
        return None


def discover_sitemap_urls(base_url: str, *, limit: int = MAX_SITEMAP_URLS) -> list[str]:
    """Collect on-host URLs advertised by robots.txt / sitemap.xml.

    Gives the catalog pass structural candidates on sites whose homepage does not
    link to listing pages directly. Sitemap reads bypass the page budget because
    they are cheap and not part of the crawled sample.
    """
    root = site_root(base_url)
    queue: list[str] = []

    robots = _fetch_text(f"{root}/robots.txt")
    if robots:
        queue.extend(
            line.split(":", 1)[1].strip()
            for line in robots.splitlines()
            if line.lower().startswith("sitemap:")
        )
    queue.extend([f"{root}/sitemap.xml", f"{root}/sitemap_index.xml"])

    found: list[str] = []
    seen_sitemaps: set[str] = set()
    nested_budget = MAX_NESTED_SITEMAPS

    while queue and len(found) < limit:
        sitemap_url = queue.pop(0)
        key = canonical_url(sitemap_url) if sitemap_url else ""
        if not key or key in seen_sitemaps:
            continue
        seen_sitemaps.add(key)

        xml = _fetch_text(sitemap_url)
        if not xml:
            continue

        locations = _sitemap_locs(xml)
        is_index = "<sitemapindex" in xml.lower()
        for location in locations:
            if is_index or location.lower().endswith((".xml", ".xml.gz")):
                if nested_budget > 0:
                    nested_budget -= 1
                    queue.append(location)
                continue
            if same_host(root, location):
                found.append(location)
            if len(found) >= limit:
                break

    deduped: list[str] = []
    seen: set[str] = set()
    for url in found:
        try:
            key = canonical_url(url)
        except WizardJobValidationError:
            continue
        if key not in seen:
            seen.add(key)
            deduped.append(url)
    return deduped[:limit]



# ======================================================================
# SECTION 4 — Industry-aware catalog crawl
# ======================================================================
# Industry validation → link classification → sampling → schema.org
# readers → draft accumulation → crawl_catalog (the main entry point).
#
# The LLM only understands and selects; fetching, following, budgeting,
# and de-duplication stay in normal crawl code.
# ======================================================================

INDUSTRY_REJECT_CONFIDENCE = 0.5

MAX_LINKS_FOR_CLASSIFICATION = 220
MAX_SITEMAP_CANDIDATES = 60
MAX_CHILD_LISTINGS_PER_PAGE = 6
LLM_CALL_HEADROOM = 4

_LISTING_CONDENSE = {"max_text_chars": 4500, "max_links": 150, "max_images": 50}
_DETAIL_CONDENSE = {"max_text_chars": 5000, "max_links": 40, "max_images": 25}

_PREFER_LONGER_TEXT_KEYS = frozenset(
    {"description", "cancellation_summary", "services_overview", "about"}
)
_UNION_LIST_KEYS = frozenset({"tags", "amenities", "source_category_ids", "vendors"})


# --- 4.1  Industry validation -------------------------------------------

VALIDATE_INDUSTRY_PROMPT = """
You verify that a website really belongs to a claimed industry, using only signals
from its homepage. Be strict about the industry, tolerant about wording.

Respond with a single JSON object only:

{
  "matches": true,
  "detected_industry": "string — the industry this website actually serves",
  "confidence": 0.0,
  "reason": "one short sentence citing the decisive signal"
}

Rules:
- "matches" is true when the site's actual business is the claimed industry, or a normal sub-type of it.
  An online clothing store matches "ecommerce". A boutique hotel matches "hospitality". A hotel does NOT match "ecommerce".
- Judge the business itself, not whether the site happens to sell something. A hotel with a gift shop is still hospitality.
- "confidence" is 0.0-1.0 for how sure you are about "matches".
- Use a low confidence when the homepage signals are too thin to tell, rather than guessing.
- A schema.org @type such as Hotel, LodgingBusiness, Restaurant, or Product is strong evidence.
"""


@dataclass
class IndustryMatch:
    matches: bool
    confidence: float
    claimed_industry: str | None
    detected_industry: str | None = None
    reason: str = ""
    checked: bool = True
    signals: dict[str, Any] = field(default_factory=dict)

    @property
    def should_stop(self) -> bool:
        """Only a confident negative from a completed check stops the workflow."""
        return self.checked and not self.matches and self.confidence >= INDUSTRY_REJECT_CONFIDENCE

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "matches": self.matches,
            "confidence": round(self.confidence, 3),
            "claimed_industry": self.claimed_industry,
            "detected_industry": self.detected_industry,
            "reason": self.reason,
            "signals": self.signals,
        }


def _node_types(node: dict[str, Any]) -> set[str]:
    raw = node.get("@type")
    values = raw if isinstance(raw, list) else [raw]
    return {str(value).lower() for value in values if value}


def _jsonld_types(document: PageDocument, limit: int = 12) -> list[str]:
    types: list[str] = []
    for node in document.jsonld:
        for value in sorted(_node_types(node)):
            if value not in types:
                types.append(value)
    return types[:limit]


def validate_industry_match(
    document: PageDocument,
    *,
    industry: str | None,
    user_request: str | None = None,
) -> IndustryMatch:
    """Lightweight homepage-only relevance check against the selected industry."""
    if not industry:
        return IndustryMatch(
            matches=True,
            confidence=0.0,
            claimed_industry=None,
            reason="No industry selected; skipping relevance check.",
            checked=False,
        )

    signals = {
        "url": document.url,
        "title": document.title,
        "meta_description": document.meta_description,
        "nav_labels": document.nav_labels(),
        "jsonld_types": _jsonld_types(document),
    }

    user_prompt = "\n".join(
        [
            f"Claimed industry: {industry}",
            f"User request context: {user_request}" if user_request else "",
            "",
            f"Website URL: {document.url}",
            f"Page title: {document.title or '(none)'}",
            f"Meta description: {document.meta_description or '(none)'}",
            f"schema.org @type values: {', '.join(signals['jsonld_types']) or '(none)'}",
            f"Navigation labels: {', '.join(signals['nav_labels']) or '(none)'}",
            "",
            "Homepage text:",
            document.text[:2500],
        ]
    ).strip()

    try:
        parsed = chat_json(
            model=model_for(VALIDATE_MODEL_SETTING),
            system=VALIDATE_INDUSTRY_PROMPT,
            user=user_prompt,
        )
    except Exception as exc:
        logger.warning("Industry validation failed for %s: %s", document.url, exc)
        return IndustryMatch(
            matches=True,
            confidence=0.0,
            claimed_industry=industry,
            reason=f"Industry check unavailable ({exc}); continuing with the crawl.",
            checked=False,
            signals=signals,
        )

    try:
        confidence = min(max(float(parsed.get("confidence") or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return IndustryMatch(
        matches=bool(parsed.get("matches")),
        confidence=confidence,
        claimed_industry=industry,
        detected_industry=(str(parsed.get("detected_industry")).strip() or None)
        if parsed.get("detected_industry")
        else None,
        reason=str(parsed.get("reason") or "").strip(),
        signals=signals,
    )


# --- 4.2  Link classification -------------------------------------------

@dataclass(frozen=True)
class CatalogCandidate:
    url: str
    role: str
    entity: str | None = None
    label: str = ""
    score: float = 0.0
    origin: str = "llm"


@dataclass
class LinkPlan:
    knowledge_links: list[dict[str, Any]] = field(default_factory=list)
    catalog_candidates: list[CatalogCandidate] = field(default_factory=list)


_KNOWLEDGE_PROMPT_BLOCK = """
"knowledge_links": [
    {"type": "about page", "url": "https://example.com/about"}
]
"""

_CATALOG_PROMPT_BLOCK = """
"catalog_links": [
    {"role": "listing | index | detail", "entity": "string", "url": "https://example.com/collections/shoes", "label": "Shoes"}
]
"""


def _link_classification_prompt(spec: CatalogSpec | None, industry: str | None) -> str:
    industry_line = f"The business operates in this industry: {industry}." if industry else ""

    if spec is None:
        catalog_rules = '- "catalog_links": always return an empty array for this industry.'
    else:
        catalog_rules = f"""- "catalog_links": pages that lead to what the business actually sells or provides.
  {spec.link_guidance}
  Set "role" to:
    "listing" — a page that lists many items (a category, collection, rooms overview, rates page)
    "index"   — a page that mostly links to other listing pages (a "shop by category" or "our rooms" hub)
    "detail"  — a page about one single item (one product, one room type, one facility)
  Set "entity" to what the page is about, e.g. {", ".join(spec.entity_keys())}.
  Prefer breadth: pick links covering different parts of the catalog rather than several near-duplicates.
  Return at most 25 catalog links, best first.
  Never put About, Contact, FAQ, blog, careers, login, or policy pages in catalog_links."""

    return f"""
You are given the links found on a business website, each with its anchor label.
{industry_line}

Split them into two groups and respond with a single JSON object only:

{{
  {_KNOWLEDGE_PROMPT_BLOCK.strip()},
  {_CATALOG_PROMPT_BLOCK.strip()}
}}

Rules:
- "knowledge_links": pages describing the business itself. INCLUDE about, company, team, services
  overview, contact, FAQ, pricing, business hours, locations, privacy policy, terms, refund and
  cancellation policies. Prefer at most 12, and always include privacy/terms when present.
{catalog_rules}
- Use the absolute URLs exactly as given. Never invent a URL that is not in the list.
- Skip login, sign-in, cart, checkout, account, search, and social-media links entirely.
- A page may legitimately appear in only one group. Do not duplicate a URL across both groups.
"""


def _segments(url: str) -> list[str]:
    return [segment.lower() for segment in urlparse(url).path.split("/") if segment]


def _hint_position(segments: list[str], hints: tuple[str, ...]) -> int | None:
    for index, segment in enumerate(segments):
        tokens = segment.replace("_", "-").split("-")
        for hint in hints:
            if segment == hint or hint in tokens or segment.startswith(hint):
                return index
    return None


def _heuristic_role(url: str, spec: CatalogSpec) -> tuple[str, float] | None:
    """Guess whether a URL is a listing or an item page from its path shape."""
    segments = _segments(url)
    if not segments:
        return None

    detail_at = _hint_position(segments, spec.detail_hints)
    if detail_at is not None and detail_at < len(segments) - 1:
        return "detail", 2.0 + min(len(segments) - detail_at - 1, 2) * 0.25

    listing_at = _hint_position(segments, spec.listing_hints)
    if listing_at is not None:
        return "listing", 2.0 - listing_at * 0.25

    if detail_at is not None:
        return "listing", 1.5
    return None


def _heuristic_candidates(
    links: list[tuple[str, str, bool]],
    *,
    spec: CatalogSpec,
    origin: str,
) -> list[CatalogCandidate]:
    candidates: list[CatalogCandidate] = []
    for url, label, in_nav in links:
        guess = _heuristic_role(url, spec)
        if guess is None:
            continue
        role, score = guess
        lowered = label.lower()
        if any(hint in lowered for hint in spec.listing_hints):
            score += 0.75
        if in_nav:
            score += 0.5
        candidates.append(
            CatalogCandidate(url=url, role=role, label=label, score=score, origin=origin)
        )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def _dedupe_candidates(candidates: list[CatalogCandidate]) -> list[CatalogCandidate]:
    seen: set[str] = set()
    out: list[CatalogCandidate] = []
    for candidate in candidates:
        try:
            key = canonical_url(candidate.url)
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def classify_site_links(
    document: PageDocument,
    *,
    base_url: str,
    spec: CatalogSpec | None,
    industry: str | None,
    user_request: str | None = None,
    include_sitemap: bool = True,
) -> LinkPlan:
    """Split the site's links into knowledge pages and catalog entry points."""
    links = [
        (link.url, link.text, link.in_nav)
        for link in document.links
        if same_host(base_url, link.url)
    ]
    if not links:
        return LinkPlan()

    prompt_lines = [
        f"Root website: {base_url}",
        f"Industry: {industry}" if industry else "",
        f"User request context: {user_request}" if user_request else "",
        "",
        "Links:",
        *[f"- {label or '(no label)'} -> {url}" for url, label, _ in links[:MAX_LINKS_FOR_CLASSIFICATION]],
    ]

    parsed: dict[str, Any] = {}
    try:
        parsed = chat_json(
            model=model_for(LINK_MODEL_SETTING),
            system=_link_classification_prompt(spec, industry),
            user="\n".join(line for line in prompt_lines if line != ""),
        )
    except Exception as exc:
        logger.warning("Link classification failed for %s: %s", base_url, exc)

    knowledge_links: list[dict[str, Any]] = []
    for item in parsed.get("knowledge_links") or []:
        if not isinstance(item, dict):
            continue
        resolved = resolve_link(base_url, str(item.get("url") or ""))
        if resolved:
            knowledge_links.append({"type": str(item.get("type") or "page"), "url": resolved})

    if spec is None:
        return LinkPlan(knowledge_links=knowledge_links)

    llm_candidates: list[CatalogCandidate] = []
    for item in parsed.get("catalog_links") or []:
        if not isinstance(item, dict):
            continue
        resolved = resolve_link(base_url, str(item.get("url") or ""))
        if not resolved or not same_host(base_url, resolved):
            continue
        role = str(item.get("role") or "").lower()
        if role not in ("listing", "index", "detail"):
            role = (_heuristic_role(resolved, spec) or ("listing", 0.0))[0]
        llm_candidates.append(
            CatalogCandidate(
                url=resolved,
                role=role,
                entity=str(item.get("entity")).strip() or None if item.get("entity") else None,
                label=str(item.get("label") or "")[:120],
                score=10.0,
                origin="llm",
            )
        )

    candidates = llm_candidates + _heuristic_candidates(links, spec=spec, origin="homepage")

    if include_sitemap:
        sitemap_links = [(url, "", False) for url in discover_sitemap_urls(base_url)]
        if sitemap_links:
            candidates += _heuristic_candidates(sitemap_links, spec=spec, origin="sitemap")[
                :MAX_SITEMAP_CANDIDATES
            ]

    return LinkPlan(knowledge_links=knowledge_links, catalog_candidates=_dedupe_candidates(candidates))


# --- 4.3  Sampling ------------------------------------------------------

def _stride(items: list[str], groups: int = 4) -> list[str]:
    """Reorder so early picks are spread across the list instead of the first N."""
    total = len(items)
    if total <= groups:
        return list(items)
    step = max(total // groups, 1)
    out: list[str] = []
    taken: set[int] = set()
    for offset in range(step):
        for index in range(offset, total, step):
            if index not in taken:
                taken.add(index)
                out.append(items[index])
    return out


def diverse_sample(buckets: list[list[str]], *, budget: int) -> list[str]:
    """Round-robin across buckets, striding within each, de-duplicated."""
    ordered = [_stride(bucket) for bucket in buckets if bucket]
    picked: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(picked) < budget and any(depth < len(bucket) for bucket in ordered):
        for bucket in ordered:
            if depth >= len(bucket):
                continue
            url = bucket[depth]
            try:
                key = canonical_url(url)
            except Exception:
                continue
            if key not in seen:
                seen.add(key)
                picked.append(url)
            if len(picked) >= budget:
                break
        depth += 1
    return picked


# --- 4.4  schema.org readers --------------------------------------------

def _first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _plain(value: Any) -> str | None:
    value = _first(value)
    if isinstance(value, dict):
        value = value.get("name") or value.get("@id")
    if value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    return text or None


def _jsonld_images(node: dict[str, Any]) -> list[dict[str, Any]]:
    raw = node.get("image")
    items = raw if isinstance(raw, list) else [raw]
    images: list[dict[str, Any]] = []
    for item in items:
        url = (item.get("url") or item.get("contentUrl")) if isinstance(item, dict) else item
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            images.append({"url": url, "alt": None})
    return images[:6]


def _jsonld_offer(node: dict[str, Any]) -> dict[str, Any]:
    offer = _first(node.get("offers")) or {}
    if not isinstance(offer, dict):
        return {}
    availability = _plain(offer.get("availability")) or ""
    in_stock: bool | None = None
    if availability:
        lowered = availability.lower()
        if "instock" in lowered or "limitedavailability" in lowered:
            in_stock = True
        elif "outofstock" in lowered or "soldout" in lowered or "discontinued" in lowered:
            in_stock = False
    return {
        "price": offer.get("price") or offer.get("lowPrice") or offer.get("priceSpecification"),
        "currency": _plain(offer.get("priceCurrency")),
        "in_stock": in_stock,
    }


def _jsonld_products(document: PageDocument) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []

    def _from_node(node: dict[str, Any]) -> dict[str, Any] | None:
        name = _plain(node.get("name"))
        if not name:
            return None
        offer = _jsonld_offer(node)
        url = _plain(node.get("url")) or _plain(node.get("@id"))
        return {
            "name": name,
            "source_url": resolve_link(document.url, url) if url else document.url,
            "sku": _plain(node.get("sku")) or _plain(node.get("mpn")),
            "description": _plain(node.get("description")),
            "vendor": _plain(node.get("brand")) or _plain(node.get("manufacturer")),
            "product_type": _plain(node.get("category")),
            "price": offer.get("price"),
            "currency": offer.get("currency"),
            "in_stock": offer.get("in_stock"),
            "images": _jsonld_images(node),
        }

    for node in document.jsonld:
        types = _node_types(node)
        if {"product", "productgroup"} & types:
            parsed = _from_node(node)
            if parsed:
                variants = [
                    {
                        "name": _plain(variant.get("name")),
                        "sku": _plain(variant.get("sku")),
                        "price": _jsonld_offer(variant).get("price"),
                        "in_stock": _jsonld_offer(variant).get("in_stock"),
                        "options": {},
                    }
                    for variant in (node.get("hasVariant") or [])
                    if isinstance(variant, dict)
                ]
                parsed["variants"] = [variant for variant in variants if variant["name"]]
                products.append(parsed)
        elif "itemlist" in types:
            for element in node.get("itemListElement") or []:
                item = element.get("item") if isinstance(element, dict) else None
                if isinstance(item, dict) and {"product", "productgroup"} & _node_types(item):
                    parsed = _from_node(item)
                    if parsed:
                        products.append(parsed)
    return products


_ROOM_JSONLD_TYPES = frozenset({"hotelroom", "room", "suite", "accommodation", "apartment"})


def _jsonld_rooms(document: PageDocument) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    for node in document.jsonld:
        candidates = [node] if _ROOM_JSONLD_TYPES & _node_types(node) else []
        for nested in node.get("containsPlace") or []:
            if isinstance(nested, dict) and _ROOM_JSONLD_TYPES & _node_types(nested):
                candidates.append(nested)
        for candidate in candidates:
            name = _plain(candidate.get("name"))
            if not name:
                continue
            occupancy = candidate.get("occupancy")
            url = _plain(candidate.get("url")) or _plain(candidate.get("@id"))
            rooms.append(
                {
                    "name": name,
                    "source_url": resolve_link(document.url, url) if url else document.url,
                    "description": _plain(candidate.get("description")),
                    "default_max_occupancy": (
                        occupancy.get("maxValue") or occupancy.get("value")
                        if isinstance(occupancy, dict)
                        else occupancy
                    ),
                    "type": _plain(candidate.get("bed")) or _plain(candidate.get("bedType")),
                    "amenities": [
                        _plain(feature.get("name") if isinstance(feature, dict) else feature)
                        for feature in (candidate.get("amenityFeature") or [])
                    ],
                    "images": _jsonld_images(candidate),
                    "price": _jsonld_offer(candidate).get("price"),
                    "currency": _jsonld_offer(candidate).get("currency"),
                }
            )
    return rooms


_PROPERTY_JSONLD_TYPES = frozenset(
    {"hotel", "lodgingbusiness", "resort", "bedandbreakfast", "hostel", "motel"}
)


def _jsonld_property(document: PageDocument) -> dict[str, Any]:
    for node in document.jsonld:
        if _PROPERTY_JSONLD_TYPES & _node_types(node):
            return {
                "name": _plain(node.get("name")),
                "currency": _plain(node.get("priceRange")),
                "language": _plain(node.get("inLanguage")),
            }
    return {}


# --- 4.5  Draft accumulation --------------------------------------------

def _url_key(url: str) -> str:
    try:
        return canonical_url(url)
    except Exception:
        return url


def _name_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or name.strip().lower()


def _merge_into(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if value in (None, "", [], {}):
            continue
        current = target.get(key)
        if current in (None, "", [], {}):
            target[key] = value
        elif isinstance(current, list) and isinstance(value, list):
            if key in _UNION_LIST_KEYS:
                target[key] = current + [item for item in value if item not in current]
            elif len(value) > len(current):
                target[key] = value
        elif (
            isinstance(current, str)
            and isinstance(value, str)
            and key in _PREFER_LONGER_TEXT_KEYS
            and len(value) > len(current)
        ):
            target[key] = value


class _CatalogDraft:
    """Collects raw entities across pages, merging the ones that are the same thing.

    Identity is the entity's own page when it has one — the URL from a listing
    card, or the page itself when a page is dedicated to a single entity. Entities
    that only ever share a page (several rate plans on one /rates page) fall back
    to name-scoped keys so they stay distinct.
    """

    def __init__(self, spec: CatalogSpec, base_url: str):
        self.spec = spec
        self.base_url = base_url
        self.root: dict[str, dict[str, Any]] = {"store": {}, "property": {}}
        self._buckets: dict[str, dict[str, dict[str, Any]]] = {
            key: {} for key in spec.entity_keys()
        }

    def add(
        self,
        entity_key: str,
        raw: dict[str, Any],
        *,
        page_url: str,
        page_is_dedicated: bool = False,
        category_source_id: str | None = None,
    ) -> str | None:
        if entity_key not in self._buckets or not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        if not name:
            return None

        raw = dict(raw)
        candidate_url = raw.pop("url", None) or raw.get("source_url")
        resolved = resolve_link(page_url, str(candidate_url)) if candidate_url else None
        own_url = (
            resolved
            if resolved and same_host(self.base_url, resolved) and _url_key(resolved) != _url_key(page_url)
            else None
        )
        raw["source_url"] = own_url or page_url
        raw.setdefault("source_id", url_slug(raw["source_url"]) or _name_slug(name))

        if category_source_id:
            raw["source_category_ids"] = [
                *(raw.get("source_category_ids") or []),
                category_source_id,
            ]

        identity_url = own_url or (page_url if page_is_dedicated else None)
        key = _url_key(identity_url) if identity_url else f"{_url_key(page_url)}#{_name_slug(name)}"

        bucket = self._buckets[entity_key]
        if key in bucket:
            _merge_into(bucket[key], raw)
        else:
            bucket[key] = raw
        return bucket[key].get("source_id")

    def add_root(self, root_key: str, raw: dict[str, Any]) -> None:
        if isinstance(raw, dict) and root_key in self.root:
            _merge_into(self.root[root_key], raw)

    def counts(self) -> dict[str, int]:
        return {key: len(bucket) for key, bucket in self._buckets.items()}

    def build(self) -> dict[str, Any]:
        draft: dict[str, Any] = {
            key: list(bucket.values()) for key, bucket in self._buckets.items()
        }
        draft["store"] = self.root["store"]
        draft["property"] = self.root["property"]
        return draft


# --- 4.6  Catalog crawl orchestration -----------------------------------

@dataclass
class CatalogCrawlResult:
    catalog: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class _LlmBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def take(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _extract_page(
    document: PageDocument,
    *,
    spec: CatalogSpec,
    kind: str,
    industry: str | None,
    user_request: str | None,
    budget: _LlmBudget,
) -> dict[str, Any]:
    if not budget.take():
        return {}

    system = spec.listing_prompt if kind == "listing" else spec.detail_prompt
    condense = _LISTING_CONDENSE if kind == "listing" else _DETAIL_CONDENSE
    header = [
        f"Industry: {industry}" if industry else "",
        f"Focus from the user: {user_request}" if user_request else "",
    ]
    user_prompt = "\n".join([line for line in header if line] + ["", document.condensed(**condense)])

    try:
        return chat_json(
            model=model_for(CATALOG_MODEL_SETTING),
            system=system,
            user=user_prompt,
        )
    except Exception as exc:
        logger.warning("Catalog extraction failed for %s: %s", document.url, exc)
        return {}


def _ingest_ecommerce(
    payload: dict[str, Any],
    document: PageDocument,
    draft: _CatalogDraft,
    *,
    kind: str,
    max_items: int,
) -> list[str]:
    """Merge one page's payload and return the item URLs worth visiting next."""
    category_source_id: str | None = None
    if kind == "listing":
        category = payload.get("category")
        if isinstance(category, dict) and category.get("name"):
            category_source_id = draft.add(
                "categories", category, page_url=document.url, page_is_dedicated=True
            )

    structured = _jsonld_products(document)
    for product in structured:
        draft.add(
            "products",
            product,
            page_url=document.url,
            page_is_dedicated=kind == "detail" and len(structured) == 1,
            category_source_id=category_source_id,
        )

    raw_products = [
        raw
        for raw in (payload.get("products") if kind == "listing" else [payload.get("product")]) or []
        if isinstance(raw, dict)
    ][:max_items]

    child_urls: list[str] = []
    for raw in raw_products:
        item_url = raw.get("url") or raw.get("source_url")
        draft.add(
            "products",
            raw,
            page_url=document.url,
            page_is_dedicated=kind == "detail" and len(raw_products) == 1,
            category_source_id=category_source_id,
        )
        resolved = resolve_link(document.url, str(item_url)) if item_url else None
        if resolved and same_host(draft.base_url, resolved) and resolved != document.url:
            child_urls.append(resolved)

    return child_urls


def _ingest_hospitality(
    payload: dict[str, Any],
    document: PageDocument,
    draft: _CatalogDraft,
    *,
    kind: str,
    max_items: int,
) -> list[str]:
    draft.add_root("property", _jsonld_property(document))
    if isinstance(payload.get("property"), dict):
        draft.add_root("property", payload["property"])

    structured = _jsonld_rooms(document)
    for room in structured:
        draft.add(
            "room_types",
            room,
            page_url=document.url,
            page_is_dedicated=kind == "detail" and len(structured) == 1,
        )

    child_urls: list[str] = []
    for entity_key in ("room_types", "rates", "facilities", "age_categories"):
        raws = [raw for raw in (payload.get(entity_key) or []) if isinstance(raw, dict)][:max_items]
        for raw in raws:
            item_url = raw.get("url") or raw.get("source_url")
            draft.add(
                entity_key,
                raw,
                page_url=document.url,
                page_is_dedicated=kind == "detail" and len(raws) == 1,
            )
            if entity_key in ("room_types", "facilities"):
                resolved = resolve_link(document.url, str(item_url)) if item_url else None
                if resolved and same_host(draft.base_url, resolved) and resolved != document.url:
                    child_urls.append(resolved)

    return child_urls


_INGESTORS = {
    "ecommerce": _ingest_ecommerce,
    "hospitality": _ingest_hospitality,
}


def _child_listing_urls(payload: dict[str, Any], document: PageDocument, base_url: str) -> list[str]:
    urls: list[str] = []
    for raw in (payload.get("child_listing_urls") or [])[:MAX_CHILD_LISTINGS_PER_PAGE]:
        resolved = resolve_link(document.url, str(raw)) if raw else None
        if resolved and same_host(base_url, resolved) and resolved != document.url:
            urls.append(resolved)
    return urls


def crawl_catalog(
    fetcher: PageFetcher,
    base_url: str,
    *,
    spec: CatalogSpec,
    candidates: list[CatalogCandidate],
    industry: str | None = None,
    user_request: str | None = None,
) -> CatalogCrawlResult:
    """Walk industry-relevant pages and build the catalog for ``spec``."""
    ingest = _INGESTORS[spec.industry]
    draft = _CatalogDraft(spec, base_url)
    budget = _LlmBudget(spec.max_listing_pages + spec.max_detail_pages + LLM_CALL_HEADROOM)

    if spec.industry == "ecommerce":
        draft.add_root("store", {"url": base_url})
    else:
        draft.add_root("property", {"url": base_url})

    pending = deque(
        candidate for candidate in candidates if candidate.role in ("listing", "index")
    )
    direct_details = [candidate.url for candidate in candidates if candidate.role == "detail"]

    listing_pages = 0
    item_buckets: list[list[str]] = []
    visited_listings: set[str] = set()

    logger.info(
        "[TIMING] Catalog crawl started — industry=%s, candidates=%d (listing/index), direct_details=%d",
        spec.industry, sum(1 for c in candidates if c.role in ("listing", "index")), len(direct_details),
    )
    _t_listing = time.perf_counter()

    while pending and listing_pages < spec.max_listing_pages:
        candidate = pending.popleft()
        try:
            key = canonical_url(candidate.url)
        except Exception:
            continue
        if key in visited_listings:
            continue
        visited_listings.add(key)

        document = fetcher.get(candidate.url)
        if document is None:
            continue
        listing_pages += 1

        payload = _extract_page(
            document,
            spec=spec,
            kind="listing",
            industry=industry,
            user_request=user_request,
            budget=budget,
        )
        found = ingest(payload, document, draft, kind="listing", max_items=spec.max_items_per_listing)
        if found:
            item_buckets.append(found)

        remaining = spec.max_listing_pages - listing_pages - len(pending)
        if remaining > 0:
            for child in _child_listing_urls(payload, document, base_url)[:remaining]:
                pending.append(CatalogCandidate(url=child, role="listing", origin="drilldown"))

    logger.info(
        "[TIMING] Catalog listing phase — %d pages, %d LLM calls so far: %.2fs",
        listing_pages, budget.used, time.perf_counter() - _t_listing,
    )

    detail_urls = diverse_sample(
        [direct_details, *item_buckets], budget=spec.max_detail_pages
    )

    detail_pages = 0
    _t_detail = time.perf_counter()
    for url in detail_urls:
        document = fetcher.get(url)
        if document is None:
            continue
        detail_pages += 1
        payload = _extract_page(
            document,
            spec=spec,
            kind="detail",
            industry=industry,
            user_request=user_request,
            budget=budget,
        )
        ingest(payload, document, draft, kind="detail", max_items=spec.max_items_per_listing)

    logger.info(
        "[TIMING] Catalog detail phase — %d pages, total LLM calls %d: %.2fs",
        detail_pages, budget.used, time.perf_counter() - _t_detail,
    )

    counts = draft.counts()
    catalog = spec.normalize(draft.build(), base_url=base_url)

    return CatalogCrawlResult(
        catalog=catalog,
        stats={
            "industry": spec.industry,
            "candidates_considered": len(candidates),
            "listing_pages": listing_pages,
            "detail_pages": detail_pages,
            "llm_calls": budget.used,
            "entities_found": counts,
            "entities_kept": {
                key: len(catalog.get(key) or []) for key in spec.entity_keys()
            },
        },
    )


# ======================================================================
# SECTION 5 — Knowledge crawl and job orchestration
# ======================================================================
# Job metadata → knowledge extraction prompt → exceptions + dataclasses
# → fetch helpers → knowledge pages → normalization → persistence →
# orchestration (execute_website_crawl, schedule_website_crawl, …).
# ======================================================================

RUNS_WHEN: tuple[str, ...] = (
    "Website exists in business information",
)

TASKS: tuple[str, ...] = (
    "Validate the website matches the selected industry",
    "Discover knowledge and industry-relevant catalog URLs",
    "Crawl public website pages",
    "Extract business summary",
    "Extract services/products",
    "Extract FAQs",
    "Extract hours",
    "Extract locations",
    "Extract contact information",
    "Extract policies/pricing if public",
    "Extract an industry catalog sample",
    "Store crawl result",
    "Trigger knowledge creation job",
)

TRIGGERS: tuple[str, ...] = (JobTrigger.INTERNAL,)


# --- 5.1  Crawler configuration -----------------------------------------

MAX_TOTAL_CHARS = 48000
MAX_RELEVANT_LINKS = 14
MIN_USABLE_SITE_CHARS = 150

_FETCH_ERROR_MARKERS = (
    "Could not fetch content from",
    "HTTPSConnectionPool",
    "HTTPConnectionPool",
    "Max retries exceeded",
    "Failed to resolve",
    "Name or service not known",
    "Connection refused",
    "Connection timed out",
    "SSLError",
)

_BUSINESS_JSONLD_TYPES = frozenset(
    {
        "organization", "corporation", "localbusiness", "store", "shop", "restaurant",
        "hotel", "lodgingbusiness", "professionalservice", "travelagency", "openinghoursspecification",
        "postaladdress", "contactpoint", "faqpage",
    }
)
MAX_STRUCTURED_DATA_CHARS = 4000

EXTRACT_SYSTEM_PROMPT = """
You extract structured business information from website text for an AI voice agent knowledge base.
Respond with a single JSON object only (no markdown fences). Use null for missing text fields and [] for empty FAQ arrays.
Use EXACTLY this top-level shape — five fixed blocks with fixed subsection keys:

{
  "company": {
    "about_us": "string or null — mission, story, who we are",
    "company_profile": "string or null — company overview, industry, size, founding, differentiators",
    "brand_voice": "string or null — tone, personality, communication style if stated",
    "locations": "string or null — physical locations and addresses",
    "contact_channels": "string or null — phone, email, chat, social, contact form details",
    "business_hours": "string or null — hours of operation"
  },
  "services": {
    "services_overview": "string or null — all services/offerings described in full detail"
  },
  "products": {
    "product_overview": "string or null — products, packages, pricing tiers, SKUs if publicly listed"
  },
  "faq": {
    "general_faq": [{"question": "string", "answer": "string"}],
    "sales_faq": [{"question": "string", "answer": "string"}],
    "support_faq": [{"question": "string", "answer": "string"}],
    "billing_faq": [{"question": "string", "answer": "string"}],
    "technical_faq": [{"question": "string", "answer": "string"}]
  },
  "policies": {
    "refund_policy": "string or null — refund, return, or money-back policy; for hotels also include no-show/deposit rules",
    "cancellation_policy": "string or null — cancellation windows, fees, free cancellation conditions; for hotels check FAQs and booking terms",
    "privacy_policy": "string or null — summary of how personal data is collected and used",
    "terms_conditions": "string or null — key terms of service, age restrictions, liability",
    "warranty_policy": "string or null",
    "escalation_policy": "string or null — complaint handling, dispute resolution"
  },
  "services_list": [{"name": "string", "description": "string or null"}]
}

Rules:
- Every top-level key above must be present. Subsection keys must not be renamed or omitted.
- Be exhaustive: capture ALL relevant website content in the most appropriate subsection.
- FAQ arrays: categorize each Q&A into the best matching faq subsection; use [] only when none exist for that category.
- services_list: every distinct service/offering (name required); used for indexing — mirror what you put in services_overview.
- Policy subsections: split policy content by topic when possible; do not merge unrelated policies into one field.
  Search FAQ sections, terms pages, and booking conditions carefully — policies are often buried in Q&A or footers.
- Text fields: use readable prose or bullet lists; preserve phone numbers, emails, URLs, and prices exactly as shown.
- These blocks are descriptive business knowledge, not a product catalog. Summarise the offering ranges in
  services_overview / product_overview; individual SKUs, room numbers, and per-item prices are captured elsewhere.
- When a "STRUCTURED BUSINESS DATA" section is present, trust it over the prose for addresses, phone numbers, and hours.
- Be factual; only use information present in the provided website text. Do not invent prices or policies.
"""


# --- 5.2  Exceptions and dataclasses ------------------------------------

class CrawlFetchError(Exception):
    """Raised when the website could not be crawled into usable content."""

    def __init__(self, message: str, *, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


class IndustryMismatchError(Exception):
    """Raised when the website does not belong to the selected industry."""

    def __init__(self, message: str, *, match: IndustryMatch):
        super().__init__(message)
        self.match = match


@dataclass
class CrawlFetchResult:
    site_text: str
    source_urls: list[str] = field(default_factory=list)
    pages_fetched: int = 0
    pages_failed: int = 0
    errors: list[str] = field(default_factory=list)


# --- 5.3  Fetch helpers -------------------------------------------------

def fetch_website_contents(url: str, max_chars: int = MAX_PAGE_CHARS) -> str | None:
    """Fetch a single page as text. Returns None if the page cannot be loaded."""
    document = PageFetcher(page_budget=1).get(url)
    if document is None:
        return None
    body = document.body
    return body[:max_chars] if body else None


def fetch_website_links(url: str) -> list[str]:
    document = PageFetcher(page_budget=1).get(url)
    return document.link_urls() if document else []


def _is_fetch_error_text(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in _FETCH_ERROR_MARKERS)


def _strip_fetch_error_content(text: str) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if not _is_fetch_error_text(line)]
    return "\n".join(lines).strip()


def _site_text_is_usable(site_text: str) -> bool:
    return len(_strip_fetch_error_content(site_text)) >= MIN_USABLE_SITE_CHARS


# --- 5.4  Knowledge pages -----------------------------------------------

def select_relevant_links(base_url: str) -> list[dict[str, Any]]:
    """LLM-selected general business pages for the knowledge pass."""
    base = normalize_url(base_url)
    fetcher = PageFetcher(page_budget=1)
    document = fetcher.get(base)
    if document is None:
        return []
    plan = classify_site_links(
        document, base_url=base, spec=None, industry=None, include_sitemap=False
    )
    return plan.knowledge_links


def _structured_business_data(documents: list[PageDocument]) -> str | None:
    """schema.org blocks describing the business itself.

    Addresses, phone numbers, and opening hours are far more reliable here than
    in scraped prose, so they are appended to the text handed to the extractor.
    """
    nodes: list[dict[str, Any]] = []
    for document in documents:
        for node in document.jsonld:
            raw_types = node.get("@type")
            types = raw_types if isinstance(raw_types, list) else [raw_types]
            if any(str(value).lower() in _BUSINESS_JSONLD_TYPES for value in types if value):
                nodes.append(node)
    if not nodes:
        return None
    return json.dumps(nodes, ensure_ascii=False)[:MAX_STRUCTURED_DATA_CHARS]


def collect_knowledge_pages(
    fetcher: PageFetcher,
    base_url: str,
    *,
    landing: PageDocument,
    knowledge_links: list[dict[str, Any]],
) -> CrawlFetchResult:
    """Assemble the knowledge-pass site text from the landing page + selected pages."""
    parts = [f"## Landing Page\n\n{landing.body[:MAX_PAGE_CHARS]}"]
    source_urls = [landing.url]
    documents = [landing]
    errors: list[str] = []
    pages_fetched = 1
    pages_failed = 0

    total_len = sum(len(part) for part in parts)
    for item in knowledge_links[:MAX_RELEVANT_LINKS]:
        link_url = item.get("url") if isinstance(item, dict) else None
        if not link_url:
            continue
        link_url = resolve_link(base_url, link_url)
        if not link_url or not same_host(base_url, link_url):
            continue

        document = fetcher.get(link_url)
        if document is None:
            pages_failed += 1
            errors.append(f"Could not load page: {link_url}")
            continue

        page_body = document.body[:MAX_PAGE_CHARS]
        if not page_body:
            continue

        page_type = item.get("type", "page") if isinstance(item, dict) else "page"
        chunk = f"\n\n### {page_type}\n{page_body}"
        if total_len + len(chunk) > MAX_TOTAL_CHARS:
            break
        parts.append(chunk)
        source_urls.append(link_url)
        documents.append(document)
        total_len += len(chunk)
        pages_fetched += 1

    structured = _structured_business_data(documents)
    if structured:
        parts.append(f"\n\n### STRUCTURED BUSINESS DATA (schema.org)\n{structured}")

    return CrawlFetchResult(
        site_text="".join(parts),
        source_urls=source_urls,
        pages_fetched=pages_fetched,
        pages_failed=pages_failed,
        errors=errors,
    )


def fetch_page_and_all_relevant_links(url: str) -> CrawlFetchResult:
    """Crawl landing page plus LLM-selected knowledge pages. Failed pages are omitted."""
    base = normalize_url(url)
    fetcher = PageFetcher()
    landing = fetcher.get(base)
    if landing is None:
        return CrawlFetchResult(
            site_text="",
            pages_failed=1,
            errors=[f"Could not load landing page: {base}"],
        )
    plan = classify_site_links(
        landing, base_url=base, spec=None, industry=None, include_sitemap=False
    )
    return collect_knowledge_pages(
        fetcher, base, landing=landing, knowledge_links=plan.knowledge_links
    )


def extract_business_data(
    url: str,
    site_text: str,
    *,
    industry_name: str | None = None,
    user_request: str | None = None,
) -> dict[str, Any]:
    header = [
        f"Industry context: {industry_name}" if industry_name else "",
        f"What the operator wants the agent to handle: {user_request}" if user_request else "",
        f"Website URL: {url}",
    ]
    user_prompt = "\n".join(
        [line for line in header if line]
        + ["", "Extract all business fields from the following website content:", "", site_text[:MAX_TOTAL_CHARS]]
    )
    return chat_json(
        model=model_for(EXTRACT_MODEL_SETTING),
        system=EXTRACT_SYSTEM_PROMPT,
        user=user_prompt,
    )


# --- 5.5  Normalization helpers -----------------------------------------

def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, indent=2)
        return text if text.strip() not in ("", "[]", "{}") else None
    text = str(value).strip()
    if not text or _is_fetch_error_text(text):
        return None
    return text


def _normalize_list_raw(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def parse_faq_items(raw: Any) -> list[dict[str, Any]]:
    items = []
    for index, entry in enumerate(_normalize_list_raw(raw)):
        if isinstance(entry, dict):
            question = _as_text(entry.get("question") or entry.get("q"))
            answer = _as_text(entry.get("answer") or entry.get("a"))
        else:
            question, answer = _as_text(entry), None
        if question or answer:
            items.append({"question": question, "answer": answer, "sort_order": index})
    return items


def parse_service_items(raw: Any) -> list[dict[str, Any]]:
    items = []
    for index, entry in enumerate(_normalize_list_raw(raw)):
        if isinstance(entry, dict):
            name = _as_text(entry.get("name") or entry.get("title") or entry.get("service"))
            description = _as_text(entry.get("description") or entry.get("desc"))
        else:
            name, description = _as_text(entry), None
        if name:
            items.append({"name": name[:255], "description": description, "sort_order": index})
    return items


def _services_to_products_text(services: list[dict[str, Any]]) -> str | None:
    """Readable text form of the services list for WebsiteCrawlJob.extracted_products."""
    lines = []
    for service in services:
        name = service.get("name")
        description = service.get("description")
        lines.append(f"{name}: {description}" if description else name)
    return lines_to_text(lines)


def collect_faq_items_from_extraction(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    faq_data = extracted.get("faq")
    if isinstance(faq_data, dict):
        for subsection in KNOWLEDGE_SUBSECTION_KEYS.get("faq", ()):
            items.extend(parse_faq_items(faq_data.get(subsection)))
    if not items:
        items = parse_faq_items(extracted.get("faqs"))
    # Re-number sort_order sequentially.
    for index, item in enumerate(items):
        item["sort_order"] = index
    return items


def _policies_to_legacy_text(blocks: dict[str, dict[str, str]]) -> str | None:
    policy_block = blocks.get("policies", {})
    parts = [text.strip() for text in policy_block.values() if text and str(text).strip()]
    return "\n\n".join(parts) or None


# --- 5.6  Persistence — wizard models -----------------------------------

def create_website_crawl_job(
    intake: AgentIntake,
    *,
    website: str,
    industry: str | None = None,
    user_request: str | None = None,
) -> WebsiteCrawlJob:
    return WebsiteCrawlJob.objects.create(
        intake=intake,
        website=website,
        industry=industry or "",
        user_request=user_request or "",
        status=CrawlStatus.QUEUED,
    )


def store_crawl_result(
    crawl_job: WebsiteCrawlJob,
    *,
    extracted: dict[str, Any],
    fetch_result: CrawlFetchResult,
    industry_match: IndustryMatch | None = None,
    catalog_result: CatalogCrawlResult | None = None,
    fetcher: PageFetcher | None = None,
) -> WebsiteCrawlJob:
    blocks = normalize_knowledge_blocks(extracted)
    services = parse_service_items(extracted.get("services_list") or extracted.get("services"))
    faqs = collect_faq_items_from_extraction(extracted)

    crawl_job.extracted_knowledge_blocks = blocks
    crawl_job.extracted_summary = (
        blocks["company"]["about_us"]
        or blocks["company"]["company_profile"]
        or _as_text(extracted.get("summary"))
    )
    crawl_job.extracted_products = _services_to_products_text(services) or blocks["services"]["services_overview"]
    crawl_job.extracted_hours = blocks["company"]["business_hours"] or _as_text(extracted.get("hours"))
    crawl_job.extracted_locations = blocks["company"]["locations"] or _as_text(extracted.get("locations"))
    crawl_job.extracted_contact_info = (
        blocks["company"]["contact_channels"] or _as_text(extracted.get("contact_info"))
    )
    crawl_job.extracted_policies = _policies_to_legacy_text(blocks) or _as_text(extracted.get("policies"))
    crawl_job.extracted_pricing = blocks["products"]["product_overview"] or _as_text(extracted.get("pricing"))

    crawl_job.extracted_catalog = catalog_result.catalog if catalog_result else None
    crawl_job.catalog_stats = catalog_result.stats if catalog_result else None
    crawl_job.catalog_pages_processed = (
        (catalog_result.stats.get("listing_pages", 0) + catalog_result.stats.get("detail_pages", 0))
        if catalog_result
        else 0
    )
    if industry_match is not None:
        crawl_job.industry_match = industry_match.as_dict()

    crawl_job.source_references = lines_to_text(fetch_result.source_urls)
    crawl_job.pages_found = (
        fetcher.pages_attempted
        if fetcher
        else fetch_result.pages_fetched + fetch_result.pages_failed
    )
    crawl_job.pages_processed = fetcher.pages_fetched if fetcher else fetch_result.pages_fetched
    crawl_job.status = CrawlStatus.PARTIALLY_COMPLETED if fetch_result.pages_failed else CrawlStatus.COMPLETED
    crawl_job.completed_at = timezone.now()

    errors = list(fetch_result.errors)
    if catalog_result:
        errors += catalog_result.errors
    crawl_job.error_message = "\n".join(errors[:20]) or None
    crawl_job.save()

    WebsiteCrawlFaq.objects.filter(crawl_job=crawl_job).delete()
    WebsiteCrawlFaq.objects.bulk_create(
        [
            WebsiteCrawlFaq(
                crawl_job=crawl_job,
                question=faq["question"],
                answer=faq["answer"],
                sort_order=faq["sort_order"],
            )
            for faq in faqs
        ]
    )

    WebsiteCrawlService.objects.filter(crawl_job=crawl_job).delete()
    WebsiteCrawlService.objects.bulk_create(
        [
            WebsiteCrawlService(
                crawl_job=crawl_job,
                name=service["name"],
                description=service["description"],
                sort_order=service["sort_order"],
            )
            for service in services
        ]
    )
    return crawl_job


def update_intake_crawl_status(intake: AgentIntake, crawl_job: WebsiteCrawlJob) -> AgentIntake:
    intake.crawl_status = crawl_job.status
    if crawl_job.status in {CrawlStatus.COMPLETED, CrawlStatus.PARTIALLY_COMPLETED}:
        intake.knowledge_status = KnowledgeStatus.NOT_STARTED
    intake.save(update_fields=["crawl_status", "knowledge_status"])
    return intake


def trigger_knowledge_creation_job(*, intake_id: int, crawl_job_id: int) -> dict[str, Any]:
    from apps.wizard.jobs.knowledge_creation import run_knowledge_creation

    try:
        result = run_knowledge_creation(intake_id=intake_id, crawl_job_id=crawl_job_id)
        return result.data
    except Exception as exc:  # knowledge creation must never break the crawl job
        logger.warning("Knowledge creation trigger failed for intake %s: %s", intake_id, exc)
        return {"triggered": False, "error": str(exc)}


# --- 5.7  Orchestration -------------------------------------------------

def _mark_crawl_failed(
    crawl_job: WebsiteCrawlJob,
    error: Exception,
    *,
    industry_match: IndustryMatch | None = None,
) -> None:
    crawl_job.status = CrawlStatus.FAILED
    crawl_job.error_message = str(error)
    crawl_job.completed_at = timezone.now()
    fields = ["status", "error_message", "completed_at"]
    if industry_match is not None:
        crawl_job.industry_match = industry_match.as_dict()
        fields.append("industry_match")
    crawl_job.save(update_fields=fields)
    AgentIntake.objects.filter(pk=crawl_job.intake_id).update(crawl_status=CrawlStatus.FAILED)


def resolve_crawl_industry(crawl_job: WebsiteCrawlJob, intake: AgentIntake) -> str | None:
    return (
        (crawl_job.industry or "").strip()
        or (intake.selected_industry or "").strip()
        or (intake.extracted_industry or "").strip()
        or None
    )


def resolve_crawl_user_request(crawl_job: WebsiteCrawlJob, intake: AgentIntake) -> str | None:
    return (
        (crawl_job.user_request or "").strip()
        or (getattr(intake, "user_request", "") or "").strip()
        or None
    )


def execute_website_crawl(crawl_job_id: int, *, trigger_knowledge_creation: bool = True) -> JobResult:
    """Run the crawl synchronously for an existing crawl job row."""
    crawl_job = WebsiteCrawlJob.objects.select_related("intake").get(pk=crawl_job_id)
    intake = crawl_job.intake

    crawl_job.status = CrawlStatus.RUNNING
    crawl_job.save(update_fields=["status"])
    AgentIntake.objects.filter(pk=intake.id).update(crawl_status=CrawlStatus.RUNNING)

    industry_match: IndustryMatch | None = None
    _t_total = time.perf_counter()
    try:
        url = normalize_url(crawl_job.website)
        industry_name = resolve_crawl_industry(crawl_job, intake)
        user_request = resolve_crawl_user_request(crawl_job, intake)
        spec = resolve_catalog_spec(industry_name)
        logger.info("[TIMING] Crawl start — url=%s industry=%s", url, industry_name or "none")

        fetcher = PageFetcher(page_budget=DEFAULT_PAGE_BUDGET, industry=industry_name)
        _t = time.perf_counter()
        landing = fetcher.get(url)
        logger.info("[TIMING] Landing page fetch: %.2fs", time.perf_counter() - _t)
        if landing is None:
            raise CrawlFetchError(
                "Could not load this website. Check the URL is correct and publicly reachable.",
                errors=fetcher.errors,
            )

        _t = time.perf_counter()
        industry_match = validate_industry_match(
            landing, industry=industry_name, user_request=user_request
        )
        logger.info("[TIMING] Industry validation (matches=%s, confidence=%.2f): %.2fs",
                    industry_match.matches, industry_match.confidence, time.perf_counter() - _t)
        if industry_match.should_stop:
            raise IndustryMismatchError(
                f"This website does not look like {industry_name}. "
                f"Detected instead: {industry_match.detected_industry or 'something else'}."
                + (f" {industry_match.reason}" if industry_match.reason else ""),
                match=industry_match,
            )

        _t = time.perf_counter()
        link_plan = classify_site_links(
            landing,
            base_url=url,
            spec=spec,
            industry=industry_name,
            user_request=user_request,
        )
        logger.info(
            "[TIMING] Link classification (knowledge=%d, catalog=%d): %.2fs",
            len(link_plan.knowledge_links), len(link_plan.catalog_candidates), time.perf_counter() - _t,
        )

        _t = time.perf_counter()
        fetch_result = collect_knowledge_pages(
            fetcher, url, landing=landing, knowledge_links=link_plan.knowledge_links
        )
        logger.info(
            "[TIMING] Knowledge pages collection (%d fetched, %d failed): %.2fs",
            fetch_result.pages_fetched, fetch_result.pages_failed, time.perf_counter() - _t,
        )
        if fetch_result.pages_fetched == 0 or not _site_text_is_usable(fetch_result.site_text):
            raise CrawlFetchError(
                "Could not load this website. Check the URL is correct and publicly reachable.",
                errors=fetch_result.errors,
            )

        site_text = _strip_fetch_error_content(fetch_result.site_text)
        _t = time.perf_counter()
        extracted = extract_business_data(
            url, site_text, industry_name=industry_name, user_request=user_request
        )
        logger.info("[TIMING] Business data extraction (LLM): %.2fs", time.perf_counter() - _t)

        catalog_result: CatalogCrawlResult | None = None
        if spec is not None:
            _t = time.perf_counter()
            try:
                catalog_result = crawl_catalog(
                    fetcher,
                    url,
                    spec=spec,
                    candidates=link_plan.catalog_candidates,
                    industry=industry_name,
                    user_request=user_request,
                )
                logger.info("[TIMING] Catalog crawl total: %.2fs", time.perf_counter() - _t)
            except Exception as exc:  # catalog is additive; never lose the knowledge result
                logger.exception("Catalog crawl failed for crawl_job=%s", crawl_job_id)
                catalog_result = CatalogCrawlResult(
                    catalog=spec.empty_catalog(url),
                    stats={"industry": spec.industry, "failed": True},
                    errors=[f"Catalog extraction failed: {exc}"],
                )

        crawl_job = store_crawl_result(
            crawl_job,
            extracted=extracted,
            fetch_result=fetch_result,
            industry_match=industry_match,
            catalog_result=catalog_result,
            fetcher=fetcher,
        )
        logger.info(
            "[TIMING] ✓ Crawl complete — total: %.2fs | pages fetched: %d | url: %s",
            time.perf_counter() - _t_total, fetcher.pages_fetched, url,
        )
    except IndustryMismatchError as exc:
        logger.info("Website crawl rejected for crawl_job=%s: %s", crawl_job_id, exc)
        _mark_crawl_failed(crawl_job, exc, industry_match=exc.match)
        return JobResult(
            success=False,
            job_name="Website Crawl",
            data={
                "crawl_job_id": crawl_job.id,
                "intake_id": intake.id,
                "crawl_status": CrawlStatus.FAILED,
                "industry_match": exc.match.as_dict(),
                "stopped_before_crawl": True,
            },
            errors=[str(exc)],
        )
    except Exception as exc:
        logger.exception("Website crawl failed for crawl_job=%s", crawl_job_id)
        _mark_crawl_failed(crawl_job, exc, industry_match=industry_match)
        return JobResult(
            success=False,
            job_name="Website Crawl",
            data={"crawl_job_id": crawl_job.id, "intake_id": intake.id},
            errors=[str(exc)],
        )

    intake = update_intake_crawl_status(intake, crawl_job)
    knowledge_job_data: dict[str, Any] = {}
    if trigger_knowledge_creation:
        knowledge_job_data = trigger_knowledge_creation_job(
            intake_id=intake.id, crawl_job_id=crawl_job.id
        )
    return JobResult(
        success=True,
        job_name="Website Crawl",
        data={
            "crawl_job_id": crawl_job.id,
            "intake_id": intake.id,
            "crawl_status": crawl_job.status,
            "pages_found": crawl_job.pages_found,
            "pages_processed": crawl_job.pages_processed,
            "industry": industry_name,
            "industry_match": industry_match.as_dict() if industry_match else None,
            "knowledge": crawl_job.extracted_knowledge_blocks,
            "catalog": crawl_job.extracted_catalog,
            "catalog_stats": crawl_job.catalog_stats,
            "knowledge_job": knowledge_job_data,
        },
    )


def _run_crawl_in_thread(crawl_job_id: int) -> None:
    close_old_connections()
    try:
        execute_website_crawl(crawl_job_id)
    except Exception:
        logger.exception("Background website crawl crashed for crawl_job=%s", crawl_job_id)
    finally:
        close_old_connections()


def schedule_website_crawl(
    intake: AgentIntake,
    *,
    website: str | None = None,
    industry: str | None = None,
    user_request: str | None = None,
) -> WebsiteCrawlJob | None:
    """Queue a background crawl for the intake's website. Non-blocking; returns the job row.

    Safe to call from inside an atomic block — the thread starts only after commit.
    Returns ``None`` when no website is available (nothing to crawl).
    """
    target_website = (website or intake.website or "").strip()
    if not target_website:
        return None

    crawl_job = create_website_crawl_job(
        intake,
        website=target_website,
        industry=industry or intake.selected_industry or intake.extracted_industry,
        user_request=user_request or getattr(intake, "user_request", None),
    )
    AgentIntake.objects.filter(pk=intake.id).update(crawl_status=CrawlStatus.QUEUED)

    def _start_thread() -> None:
        threading.Thread(
            target=_run_crawl_in_thread,
            args=(crawl_job.id,),
            daemon=True,
            name=f"wizard-website-crawl-{crawl_job.id}",
        ).start()

    transaction.on_commit(_start_thread)
    return crawl_job


_ACTIVE_CRAWL_STATUSES = {CrawlStatus.QUEUED, CrawlStatus.RUNNING}
_DONE_CRAWL_STATUSES = {CrawlStatus.COMPLETED, CrawlStatus.PARTIALLY_COMPLETED}


def ensure_website_knowledge_pipeline(
    intake: AgentIntake,
    *,
    industry: str | None = None,
    user_request: str | None = None,
) -> dict[str, Any]:
    """After agent generation: full site crawl + knowledge blocks + catalog + MD upload.

    Step-1 mini-crawl is only for GPT recommendations. This is Jobs 7/8.
    Idempotent — will not start a second crawl while one is queued, running, or done.
    """
    website = (intake.website or "").strip()
    if not website:
        return {"started": False, "reason": "no_website"}

    latest = WebsiteCrawlJob.objects.filter(intake_id=intake.id).order_by("-created_at").first()
    if latest and latest.status in _ACTIVE_CRAWL_STATUSES:
        return {"started": False, "reason": "crawl_in_progress", "crawl_job_id": latest.id}

    if latest and latest.status in _DONE_CRAWL_STATUSES:
        from apps.wizard.jobs.knowledge_creation import (
            attach_knowledge_to_agent_if_exists,
            run_knowledge_creation,
        )

        knowledge = (
            KnowledgeJob.objects.filter(intake_id=intake.id, crawl_job_id=latest.id)
            .order_by("-created_at")
            .first()
        )
        if knowledge is None or knowledge.status == KnowledgeStatus.FAILED:
            try:
                result = run_knowledge_creation(intake_id=intake.id, crawl_job_id=latest.id)
                return {
                    "started": True,
                    "reason": "knowledge_from_existing_crawl",
                    "crawl_job_id": latest.id,
                    "knowledge": result.data,
                    "errors": result.errors,
                }
            except Exception as exc:
                logger.warning(
                    "Knowledge creation from existing crawl failed intake=%s: %s",
                    intake.id,
                    exc,
                )
                return {"started": False, "reason": "knowledge_failed", "error": str(exc)}
        if knowledge.md_studio_knowledge_id:
            attach_knowledge_to_agent_if_exists(knowledge, latest)
            knowledge.refresh_from_db()
            return {
                "started": False,
                "reason": "knowledge_exists",
                "crawl_job_id": latest.id,
                "knowledge_job_id": knowledge.id,
                "knowledge_status": knowledge.status,
            }
        return {
            "started": False,
            "reason": "knowledge_in_progress",
            "knowledge_job_id": knowledge.id,
        }

    crawl_job = schedule_website_crawl(
        intake, website=website, industry=industry, user_request=user_request
    )
    return {
        "started": True,
        "reason": "crawl_queued",
        "crawl_job_id": crawl_job.id if crawl_job else None,
    }


@dataclass
class WebsiteCrawlJobRunner(WizardJob):
    job_number = 7
    name = "Website Crawl"
    runs_when = RUNS_WHEN
    tasks = TASKS
    triggers = TRIGGERS

    def run(
        self,
        *,
        intake_id: int,
        website: str | None = None,
        industry: str | None = None,
        user_request: str | None = None,
        run_in_background: bool = True,
        **_: Any,
    ) -> JobResult:
        intake = get_intake_or_raise(intake_id)
        target_website = (website or intake.website or "").strip()
        if not target_website:
            raise WizardJobValidationError("Website is required to start a crawl job.")

        if run_in_background:
            crawl_job = schedule_website_crawl(
                intake,
                website=target_website,
                industry=industry,
                user_request=user_request,
            )
            return JobResult(
                success=True,
                job_name=self.name,
                data={
                    "queued": True,
                    "crawl_job_id": crawl_job.id if crawl_job else None,
                    "intake_id": intake.id,
                    "message": "Crawl queued for background worker.",
                },
            )

        crawl_job = create_website_crawl_job(
            intake,
            website=target_website,
            industry=industry or intake.selected_industry or intake.extracted_industry,
            user_request=user_request or getattr(intake, "user_request", None),
        )
        return execute_website_crawl(crawl_job.id)


def run_website_crawl(**kwargs: Any) -> JobResult:
    return WebsiteCrawlJobRunner().run(**kwargs)
