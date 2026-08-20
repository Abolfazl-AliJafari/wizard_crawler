"""
Console test runner for website_crawl.py
=========================================
Usage:
    python test_console.py

Prompts for URL, industry (numbered), and optional user request.
Streams logs to terminal and saves the result to results/.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import time
import types
from pathlib import Path
from types import ModuleType


# ─── Load .env ────────────────────────────────────────────────────────────────
def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv(Path(__file__).parent / ".env")


# ─── Django / project stubs ───────────────────────────────────────────────────
def _stub(dotted_name: str, **attrs):
    parts = dotted_name.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            m = ModuleType(parent)
            m.__path__ = []
            sys.modules[parent] = m
    m = sys.modules.get(dotted_name) or ModuleType(dotted_name)
    if not hasattr(m, "__path__"):
        m.__path__ = []
    for key, value in attrs.items():
        setattr(m, key, value)
    sys.modules[dotted_name] = m
    return m


class WizardJobValidationError(Exception):
    pass

_stub("apps")
_stub("apps.wizard")
_stub("apps.wizard.jobs")
_stub("apps.wizard.jobs.exceptions", WizardJobValidationError=WizardJobValidationError)


class _FakeSettings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    def __getattr__(self, name: str):
        return None

_fake_settings = _FakeSettings()

django_mod = ModuleType("django");         django_mod.__path__ = [];      sys.modules["django"] = django_mod
django_conf = ModuleType("django.conf");   django_conf.settings = _fake_settings; sys.modules["django.conf"] = django_conf
django_db   = ModuleType("django.db");     django_db.close_old_connections = lambda: None
django_db.transaction = types.SimpleNamespace(on_commit=lambda f: None);  sys.modules["django.db"] = django_db
django_utils = ModuleType("django.utils"); django_utils.__path__ = [];    sys.modules["django.utils"] = django_utils
import datetime as _dt
django_tz = ModuleType("django.utils.timezone"); django_tz.now = lambda: _dt.datetime.utcnow(); sys.modules["django.utils.timezone"] = django_tz


class JobResult:
    def __init__(self, success, job_name, data, errors=None):
        self.success = success; self.job_name = job_name; self.data = data or {}; self.errors = errors or []

class JobTrigger:
    INTERNAL = "internal"

class WizardJob:
    pass

_stub("apps.wizard.jobs.base", JobResult=JobResult, JobTrigger=JobTrigger, WizardJob=WizardJob)

def _lines_to_text(lines):
    return "\n".join(str(l) for l in lines if l) or None

_stub("apps.wizard.jobs.helpers", get_intake_or_raise=lambda _: None, lines_to_text=_lines_to_text)

_KNOWLEDGE_SUBSECTION_KEYS = {
    "company":  ("about_us", "company_profile", "brand_voice", "locations", "contact_channels", "business_hours"),
    "services": ("services_overview",),
    "products": ("product_overview",),
    "faq":      ("general_faq", "sales_faq", "support_faq", "billing_faq", "technical_faq"),
    "policies": ("refund_policy", "cancellation_policy", "privacy_policy", "terms_conditions", "warranty_policy", "escalation_policy"),
}

def _normalize_knowledge_blocks(extracted: dict) -> dict:
    blocks: dict = {}
    for section, subkeys in _KNOWLEDGE_SUBSECTION_KEYS.items():
        raw = extracted.get(section) if isinstance(extracted.get(section), dict) else {}
        blocks[section] = {k: raw.get(k) for k in subkeys}
    return blocks

_stub("apps.wizard.integrations")
_stub("apps.wizard.integrations.knowledge",
      KNOWLEDGE_SUBSECTION_KEYS=_KNOWLEDGE_SUBSECTION_KEYS,
      normalize_knowledge_blocks=_normalize_knowledge_blocks)

class CrawlStatus:
    QUEUED = "queued"; RUNNING = "running"; COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"; FAILED = "failed"

class KnowledgeStatus:
    NOT_STARTED = "not_started"; FAILED = "failed"

_stub("apps.wizard.constants")
_stub("apps.wizard.constants.choices", CrawlStatus=CrawlStatus, KnowledgeStatus=KnowledgeStatus)
_stub("apps.wizard.models", AgentIntake=None, KnowledgeJob=None,
      WebsiteCrawlFaq=None, WebsiteCrawlJob=None, WebsiteCrawlService=None)


# ─── Import crawler ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import website_crawl as wc  # noqa: E402


# ─── Logging ──────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"

def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not any(getattr(h, "_crawl_terminal", False) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_LOG_FORMAT))
        sh.setLevel(logging.INFO)
        sh._crawl_terminal = True  # type: ignore[attr-defined]
        root.addHandler(sh)

_setup_logging()


# ─── Results ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(result: dict, *, url: str, industry: str | None) -> Path:
    from urllib.parse import urlparse
    ts = time.strftime("%Y%m%d_%H%M%S")
    domain = urlparse(url).netloc.replace("www.", "").replace(".", "_")
    ind = f"_{industry}" if industry else ""
    filename = f"crawl_{ts}{ind}_{domain}.json"
    path = RESULTS_DIR / filename
    payload = {
        "meta": {"url": url, "industry": industry, "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "knowledge": result.get("knowledge"),
        "catalog": result.get("catalog"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


# ─── Core crawl runner ────────────────────────────────────────────────────────
def run_crawl_pure(url: str, *, industry: str | None, user_request: str | None) -> dict:
    timings: dict[str, float] = {}
    errors: list[str] = []
    _t_total = time.perf_counter()

    url = wc.normalize_url(url)

    fetcher = wc.PageFetcher(page_budget=wc.DEFAULT_PAGE_BUDGET, industry=industry)
    _t = time.perf_counter()
    landing = fetcher.get(url)
    timings["landing_page_fetch"] = round(time.perf_counter() - _t, 3)

    if landing is None:
        raise wc.CrawlFetchError(
            "Could not load this website. Check the URL is correct and publicly reachable.",
            errors=fetcher.errors,
        )

    _t = time.perf_counter()
    industry_match = wc.validate_industry_match(landing, industry=industry, user_request=user_request)
    timings["industry_validation"] = round(time.perf_counter() - _t, 3)

    if industry_match.should_stop:
        raise wc.IndustryMismatchError(
            f"Website does not look like '{industry}'. "
            f"Detected: {industry_match.detected_industry or 'unknown'}. "
            f"{industry_match.reason}",
            match=industry_match,
        )

    spec = wc.resolve_catalog_spec(industry)
    _t = time.perf_counter()
    link_plan = wc.classify_site_links(
        landing, base_url=url, spec=spec, industry=industry, user_request=user_request,
    )
    timings["link_classification"] = round(time.perf_counter() - _t, 3)

    _t = time.perf_counter()
    fetch_result = wc.collect_knowledge_pages(
        fetcher, url, landing=landing, knowledge_links=link_plan.knowledge_links
    )
    timings["knowledge_pages_collection"] = round(time.perf_counter() - _t, 3)
    errors += fetch_result.errors

    if not wc._site_text_is_usable(fetch_result.site_text):
        raise wc.CrawlFetchError(
            "Website content is too thin to extract useful information.",
            errors=fetch_result.errors,
        )

    site_text = wc._strip_fetch_error_content(fetch_result.site_text)
    _t = time.perf_counter()
    extracted = wc.extract_business_data(
        url, site_text, industry_name=industry, user_request=user_request
    )
    timings["knowledge_extraction_llm"] = round(time.perf_counter() - _t, 3)

    catalog_result = None
    if spec is not None:
        _t = time.perf_counter()
        try:
            catalog_result = wc.crawl_catalog(
                fetcher, url, spec=spec,
                candidates=link_plan.catalog_candidates,
                industry=industry, user_request=user_request,
            )
        except Exception as exc:
            errors.append(f"Catalog crawl failed: {exc}")
            catalog_result = wc.CatalogCrawlResult(
                catalog=spec.empty_catalog(url),
                stats={"industry": spec.industry, "failed": True},
                errors=[str(exc)],
            )
        timings["catalog_crawl_total"] = round(time.perf_counter() - _t, 3)
        if catalog_result:
            errors += catalog_result.errors

    timings["total"] = round(time.perf_counter() - _t_total, 3)

    return {
        "knowledge": extracted,
        "catalog": catalog_result.catalog if catalog_result else None,
        "timings": timings,
        "errors": errors,
    }


# ─── Console UI ───────────────────────────────────────────────────────────────
def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return value or (default or "")


def _pick_industry() -> str | None:
    from industries import CATALOG_SPECS
    if not CATALOG_SPECS:
        return None

    print("\nIndustry:")
    for i, spec in enumerate(CATALOG_SPECS, start=1):
        print(f"  {i}) {spec.label}")
    print("  0) Generic (no industry)")

    while True:
        raw = _prompt("Choice", "0")
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(CATALOG_SPECS):
            return CATALOG_SPECS[int(raw) - 1].industry
        print(f"  Please enter a number between 0 and {len(CATALOG_SPECS)}.")


def main() -> None:
    print("=" * 60)
    print("  Wizard Crawler — Console Test Runner")
    print("=" * 60)

    url = ""
    while not url:
        url = _prompt("\nURL")
        if not url:
            print("  URL is required.")

    industry = _pick_industry()

    user_request = _prompt("\nUser request (optional, press Enter to skip)") or None

    print()
    print("=" * 60)
    print(f"  URL       : {url}")
    print(f"  Industry  : {industry or 'generic'}")
    if user_request:
        print(f"  Request   : {user_request}")
    print("=" * 60)
    print()

    try:
        result = run_crawl_pure(url, industry=industry, user_request=user_request)
    except wc.CrawlFetchError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except wc.IndustryMismatchError as exc:
        print(f"\n[MISMATCH] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        raise

    saved = _save_result(result, url=url, industry=industry)

    errors = result.get("errors") or []
    timings = result.get("timings") or {}

    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Saved     → {saved.name}")
    print(f"  Total time: {timings.get('total', '?')}s")
    if errors:
        print(f"  Warnings  : {len(errors)}")
        for e in errors[:3]:
            print(f"    - {e}")
    print()


if __name__ == "__main__":
    main()
