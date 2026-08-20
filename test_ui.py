"""
Standalone test UI for website_crawl.py
=========================================
Lets you run the complete crawl flow through a browser form, without
needing the full Django project running.

Usage
-----
1. Install dependencies:
       pip install flask requests beautifulsoup4 openai

2. Set your OpenAI API key (or paste it in the UI):
       export OPENAI_API_KEY=sk-...

3. Run:
       python test_ui.py

4. Open http://localhost:5001 in your browser.

How it works
------------
- Stubs Django and project-specific imports so website_crawl.py can be
  imported and all pure-Python crawl functions (Sections 1-4) work normally.
- Calls the pure crawl functions directly (no DB writes).
- Exposes a /crawl SSE endpoint that streams log lines + final JSON result.
"""

from __future__ import annotations

import os
import sys
import json
import logging
import queue
import threading
import time
import types
from pathlib import Path
from types import ModuleType


# ─── Load .env file (before anything reads os.environ) ───────────────────────
def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no extra dependencies needed."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:   # env var wins over .env
            os.environ[key] = value

_load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Stub Django and project dependencies BEFORE importing website_crawl
# ─────────────────────────────────────────────────────────────────────────────

def _stub(dotted_name: str, **attrs):
    """Register a lightweight stub module, creating parent stubs as needed."""
    parts = dotted_name.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        if parent not in sys.modules:
            m = ModuleType(parent)
            m.__path__ = []  # make it a package
            sys.modules[parent] = m
    m = sys.modules.get(dotted_name) or ModuleType(dotted_name)
    if not hasattr(m, "__path__"):
        m.__path__ = []
    for key, value in attrs.items():
        setattr(m, key, value)
    sys.modules[dotted_name] = m
    return m


# --- WizardJobValidationError (used throughout website_crawl.py) ---
class WizardJobValidationError(Exception):
    pass

_stub("apps")
_stub("apps.wizard")
_stub("apps.wizard.jobs")
_stub("apps.wizard.jobs.exceptions", WizardJobValidationError=WizardJobValidationError)


# --- Django settings (only OPENAI_API_KEY is needed) ---
class _FakeSettings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

    def __getattr__(self, name: str):
        return None


_fake_settings = _FakeSettings()

django_mod = ModuleType("django")
django_mod.__path__ = []
sys.modules["django"] = django_mod

django_conf = ModuleType("django.conf")
django_conf.settings = _fake_settings
sys.modules["django.conf"] = django_conf

django_db = ModuleType("django.db")
django_db.close_old_connections = lambda: None
django_db.transaction = types.SimpleNamespace(on_commit=lambda f: None)
sys.modules["django.db"] = django_db

django_utils = ModuleType("django.utils")
django_utils.__path__ = []
sys.modules["django.utils"] = django_utils

import datetime as _dt
django_tz = ModuleType("django.utils.timezone")
django_tz.now = lambda: _dt.datetime.utcnow()
sys.modules["django.utils.timezone"] = django_tz


# --- JobResult / JobTrigger / WizardJob ---
class JobResult:
    def __init__(self, success, job_name, data, errors=None):
        self.success = success
        self.job_name = job_name
        self.data = data or {}
        self.errors = errors or []

class JobTrigger:
    INTERNAL = "internal"

class WizardJob:
    pass

_stub("apps.wizard.jobs.base",
      JobResult=JobResult, JobTrigger=JobTrigger, WizardJob=WizardJob)


# --- lines_to_text helper ---
def _lines_to_text(lines):
    return "\n".join(str(line) for line in lines if line) or None

_stub("apps.wizard.jobs.helpers",
      get_intake_or_raise=lambda _: None,
      lines_to_text=_lines_to_text)


# --- Knowledge helpers (minimal) ---
_KNOWLEDGE_SUBSECTION_KEYS = {
    "company":  ("about_us", "company_profile", "brand_voice", "locations",
                 "contact_channels", "business_hours"),
    "services": ("services_overview",),
    "products": ("product_overview",),
    "faq":      ("general_faq", "sales_faq", "support_faq",
                 "billing_faq", "technical_faq"),
    "policies": ("refund_policy", "cancellation_policy", "privacy_policy",
                 "terms_conditions", "warranty_policy", "escalation_policy"),
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


# --- CrawlStatus / KnowledgeStatus ---
class CrawlStatus:
    QUEUED = "queued"; RUNNING = "running"; COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"; FAILED = "failed"

class KnowledgeStatus:
    NOT_STARTED = "not_started"; FAILED = "failed"

_stub("apps.wizard.constants")
_stub("apps.wizard.constants.choices",
      CrawlStatus=CrawlStatus, KnowledgeStatus=KnowledgeStatus)

# --- Model stubs (never called in test mode) ---
_stub("apps.wizard.models",
      AgentIntake=None, KnowledgeJob=None,
      WebsiteCrawlFaq=None, WebsiteCrawlJob=None, WebsiteCrawlService=None)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Import website_crawl (pure crawl logic, Sections 1-4 + helpers)
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import website_crawl as wc  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Log handler that broadcasts to a per-request queue (for SSE)
# ─────────────────────────────────────────────────────────────────────────────

class _QueueHandler(logging.Handler):
    """Push every log record into a queue so the SSE stream can forward it."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        self._q.put(("log", self.format(record)))


_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s"

# Results folder — created once at startup
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Always attach our terminal handler (identified by name so we don't duplicate)
    if not any(getattr(h, "_crawl_terminal", False) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(_LOG_FORMAT))
        sh.setLevel(logging.INFO)
        sh._crawl_terminal = True  # type: ignore[attr-defined]
        root.addHandler(sh)


_setup_logging()


def _save_result(result: dict, *, url: str, industry: str | None) -> Path:
    """Write one crawl result to results/<timestamp>_<domain>.json and return the path."""
    from urllib.parse import urlparse
    ts = time.strftime("%Y%m%d_%H%M%S")
    domain = urlparse(url).netloc.replace("www.", "").replace(".", "_")
    ind = f"_{industry}" if industry else ""
    filename = f"crawl_{ts}{ind}_{domain}.json"
    path = RESULTS_DIR / filename

    payload = {
        "meta": {
            "url": url,
            "industry": industry,
            "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "knowledge": result.get("knowledge"),
        "catalog": result.get("catalog"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Core crawl runner (pure, no Django persistence)
# ─────────────────────────────────────────────────────────────────────────────

def run_crawl_pure(
    url: str,
    *,
    industry: str | None,
    user_request: str | None,
    api_key: str | None,
) -> dict:
    """
    Execute the complete crawl flow without any Django DB writes.

    Returns a dict with keys:
        knowledge, catalog, catalog_stats, industry_match,
        timing, pages_fetched, pages_failed, source_urls, errors
    """
    # Override API key if provided via UI
    if api_key:
        _fake_settings.OPENAI_API_KEY = api_key.strip()

    timings: dict[str, float] = {}
    errors: list[str] = []
    _t_total = time.perf_counter()

    # 1. Normalize URL
    url = wc.normalize_url(url)

    # 2. Fetch landing page
    fetcher = wc.PageFetcher(page_budget=wc.DEFAULT_PAGE_BUDGET, industry=industry)
    _t = time.perf_counter()
    landing = fetcher.get(url)
    timings["landing_page_fetch"] = round(time.perf_counter() - _t, 3)

    if landing is None:
        raise wc.CrawlFetchError(
            "Could not load this website. Check the URL is correct and publicly reachable.",
            errors=fetcher.errors,
        )

    # 3. Industry validation
    _t = time.perf_counter()
    industry_match = wc.validate_industry_match(
        landing, industry=industry, user_request=user_request
    )
    timings["industry_validation"] = round(time.perf_counter() - _t, 3)

    if industry_match.should_stop:
        raise wc.IndustryMismatchError(
            f"Website does not look like '{industry}'. "
            f"Detected: {industry_match.detected_industry or 'unknown'}. "
            f"{industry_match.reason}",
            match=industry_match,
        )

    # 4. Resolve catalog spec + classify links
    spec = wc.resolve_catalog_spec(industry)
    _t = time.perf_counter()
    link_plan = wc.classify_site_links(
        landing,
        base_url=url,
        spec=spec,
        industry=industry,
        user_request=user_request,
    )
    timings["link_classification"] = round(time.perf_counter() - _t, 3)

    # 5. Collect knowledge pages
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

    # 6. Extract business knowledge
    site_text = wc._strip_fetch_error_content(fetch_result.site_text)
    _t = time.perf_counter()
    extracted = wc.extract_business_data(
        url, site_text, industry_name=industry, user_request=user_request
    )
    timings["knowledge_extraction_llm"] = round(time.perf_counter() - _t, 3)

    # 7. Catalog crawl (optional — only when industry is recognized)
    catalog_result = None
    if spec is not None:
        _t = time.perf_counter()
        try:
            catalog_result = wc.crawl_catalog(
                fetcher,
                url,
                spec=spec,
                candidates=link_plan.catalog_candidates,
                industry=industry,
                user_request=user_request,
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
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Flask web app
# ─────────────────────────────────────────────────────────────────────────────

try:
    from flask import Flask, request, Response, stream_with_context
except ImportError:
    print("\n[ERROR] Flask is not installed. Run:  pip install flask\n")
    sys.exit(1)

app = Flask(__name__)

# ─── HTML template ────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Crawler Test UI</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:      #0f1117;
      --surface: #1a1d27;
      --border:  #2e3244;
      --accent:  #6c63ff;
      --accent2: #00d4aa;
      --text:    #e2e8f0;
      --muted:   #8892a4;
      --danger:  #ff6b6b;
      --warn:    #ffd166;
      --success: #06d6a0;
      --info:    #118ab2;
      --timing:  #f77f00;
    }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    header {
      padding: 16px 28px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header h1 { font-size: 1.15rem; font-weight: 600; letter-spacing: .4px; }
    header span { font-size: .8rem; color: var(--muted); margin-left: auto; }

    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      height: calc(100vh - 57px);
      overflow: hidden;
    }

    /* ── Left panel ── */
    .left {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
    }
    .left section { padding: 22px 20px; border-bottom: 1px solid var(--border); }
    .left section:last-child { border-bottom: none; }
    .left h2 { font-size: .75rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 1px; color: var(--muted); margin-bottom: 14px; }

    label { display: block; font-size: .82rem; color: var(--muted);
            margin-bottom: 5px; margin-top: 12px; }
    label:first-of-type { margin-top: 0; }
    input, textarea, select {
      width: 100%; background: var(--bg); border: 1px solid var(--border);
      color: var(--text); border-radius: 7px; padding: 9px 12px;
      font-size: .88rem; font-family: inherit; outline: none;
      transition: border-color .2s;
    }
    input:focus, textarea:focus, select:focus { border-color: var(--accent); }
    textarea { resize: vertical; min-height: 70px; }

    select option { background: var(--surface); }

    .btn {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      width: 100%; padding: 11px; border: none; border-radius: 8px;
      font-size: .92rem; font-weight: 600; cursor: pointer;
      transition: opacity .2s, transform .1s;
    }
    .btn:active { transform: scale(.98); }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover { opacity: .9; }
    .btn-primary:disabled { opacity: .45; cursor: not-allowed; }

    .timing-box {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px; font-size: .8rem;
    }
    .timing-row { display: flex; justify-content: space-between; padding: 3px 0; }
    .timing-row .label { color: var(--muted); }
    .timing-row .val { color: var(--timing); font-weight: 600; font-family: monospace; }
    .timing-row.total .label { color: var(--text); font-weight: 600; }
    .timing-row.total .val { color: var(--success); font-size: .9rem; }

    .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .stat-card {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px 12px;
    }
    .stat-card .num { font-size: 1.4rem; font-weight: 700; color: var(--accent2); }
    .stat-card .lbl { font-size: .72rem; color: var(--muted); margin-top: 2px; }

    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 20px;
      font-size: .73rem; font-weight: 600;
    }
    .badge-ok  { background: #06d6a022; color: var(--success); border: 1px solid #06d6a055; }
    .badge-err { background: #ff6b6b22; color: var(--danger);  border: 1px solid #ff6b6b55; }
    .badge-warn{ background: #ffd16622; color: var(--warn);    border: 1px solid #ffd16655; }

    /* ── Right panel ── */
    .right {
      display: flex; flex-direction: column; overflow: hidden;
    }

    .tabs {
      display: flex; border-bottom: 1px solid var(--border);
      background: var(--surface); padding: 0 20px;
    }
    .tab {
      padding: 12px 18px; font-size: .85rem; font-weight: 500; cursor: pointer;
      border-bottom: 2px solid transparent; color: var(--muted);
      transition: color .2s, border-color .2s;
    }
    .tab.active { color: var(--accent); border-color: var(--accent); }
    .tab:hover:not(.active) { color: var(--text); }

    .panels { flex: 1; overflow: hidden; position: relative; }
    .panel {
      position: absolute; inset: 0; overflow-y: auto; padding: 20px;
      display: none;
    }
    .panel.active { display: block; }

    /* Log panel */
    #log-panel {
      background: #0a0c13; font-family: 'Cascadia Code', 'Fira Code', monospace;
      font-size: .78rem; line-height: 1.7;
    }
    .log-line { padding: 1px 0; white-space: pre-wrap; word-break: break-all; }
    .log-line.INFO    { color: #7ecfff; }
    .log-line.DEBUG   { color: #5a6a7a; }
    .log-line.WARNING { color: var(--warn); }
    .log-line.ERROR, .log-line.CRITICAL { color: var(--danger); }
    .log-line.TIMING  { color: var(--timing); font-weight: 600; }

    /* JSON panels */
    .json-view {
      background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
      padding: 16px; font-family: 'Cascadia Code', 'Fira Code', monospace;
      font-size: .8rem; line-height: 1.6; overflow-x: auto; white-space: pre-wrap;
      word-break: break-all;
    }
    .json-view .k { color: #79b8ff; }
    .json-view .s { color: #9ecbff; }
    .json-view .n { color: var(--accent2); }
    .json-view .b { color: var(--warn); }
    .json-view .nil { color: var(--muted); }

    .placeholder {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; height: 100%; color: var(--muted);
      font-size: .9rem; gap: 12px;
    }
    .placeholder svg { opacity: .3; }

    /* Spinner */
    .spinner {
      width: 18px; height: 18px; border: 2px solid #ffffff44;
      border-top-color: #fff; border-radius: 50%;
      animation: spin .7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Error banner */
    .error-banner {
      background: #ff6b6b18; border: 1px solid #ff6b6b55;
      border-radius: 8px; padding: 12px 16px; color: var(--danger);
      font-size: .88rem; margin-bottom: 16px;
    }

    /* Match indicator */
    .match-block { display: flex; flex-direction: column; gap: 10px; }
    .match-row { display: flex; align-items: center; gap: 10px; font-size: .88rem; }
    .match-row .label { color: var(--muted); width: 140px; flex-shrink: 0; }
    .confidence-bar { flex: 1; background: var(--bg); border-radius: 20px; height: 6px; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 20px; background: var(--accent); }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  </style>
</head>
<body>
<header>
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
       stroke="var(--accent)" stroke-width="2" stroke-linecap="round">
    <circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/>
    <path d="M2 12h20"/>
  </svg>
  <h1>Website Crawler — Test UI</h1>
  <span>website_crawl.py &nbsp;·&nbsp; Pure crawl mode (no DB)</span>
</header>

<div class="layout">
  <!-- ── Left panel ── -->
  <div class="left">
    <section>
      <h2>Input</h2>

      <label for="url">Website URL *</label>
      <input id="url" type="url" placeholder="https://example.com" value=""/>

      <label for="industry">Industry</label>
      <select id="industry">
        <option value="">— None / Auto-detect —</option>
        <option value="ecommerce">Ecommerce</option>
        <option value="hospitality">Hospitality</option>
      </select>

      <label for="user_req">User request (optional)</label>
      <textarea id="user_req" placeholder="e.g. Focus on room types and pricing…"></textarea>

      <label for="api_key">OpenAI API key (overrides env var)</label>
      <input id="api_key" type="password" placeholder="sk-… (leave blank to use env)"/>

      <div style="margin-top: 18px;">
        <button id="run-btn" class="btn btn-primary" onclick="startCrawl()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.2">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          Run Crawl
        </button>
      </div>
    </section>

    <!-- Stats (visible after crawl) -->
    <section id="stats-section" style="display:none">
      <h2>Stats</h2>
      <div id="industry-match-display"></div>
      <div class="stat-grid" style="margin-top:12px" id="stat-grid"></div>
    </section>

    <!-- Timing (visible after crawl) -->
    <section id="timing-section" style="display:none">
      <h2>Timing</h2>
      <div class="timing-box" id="timing-box"></div>
    </section>
  </div>

  <!-- ── Right panel ── -->
  <div class="right">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('log')">Logs</div>
      <div class="tab" onclick="switchTab('knowledge')">Knowledge</div>
      <div class="tab" onclick="switchTab('catalog')">Catalog</div>
    </div>

    <div class="panels">
      <!-- Log panel -->
      <div class="panel active" id="log-panel">
        <div class="placeholder" id="log-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.5">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <path d="M7 8h10M7 12h10M7 16h6"/>
          </svg>
          <span>Logs will appear here when you start a crawl.</span>
        </div>
      </div>

      <!-- Knowledge panel -->
      <div class="panel" id="knowledge-panel">
        <div class="placeholder" id="knowledge-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.5">
            <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
          <span>Knowledge output will appear after the crawl.</span>
        </div>
        <div id="knowledge-content" style="display:none"></div>
      </div>

      <!-- Catalog panel -->
      <div class="panel" id="catalog-panel">
        <div class="placeholder" id="catalog-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="1.5">
            <path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/>
            <line x1="3" y1="6" x2="21" y2="6"/>
            <path d="M16 10a4 4 0 0 1-8 0"/>
          </svg>
          <span>Catalog output will appear after the crawl (only for supported industries).</span>
        </div>
        <div id="catalog-content" style="display:none"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const names = ['log', 'knowledge', 'catalog'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(name + '-panel').classList.add('active');
}

// ── JSON syntax highlighter ────────────────────────────────────────────────
function syntaxHighlight(json) {
  const s = JSON.stringify(json, null, 2);
  return s.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, m => {
    if (/^"/.test(m)) {
      return /:$/.test(m)
        ? `<span class="k">${m}</span>`
        : `<span class="s">${m}</span>`;
    }
    if (/true|false/.test(m)) return `<span class="b">${m}</span>`;
    if (/null/.test(m)) return `<span class="nil">${m}</span>`;
    return `<span class="n">${m}</span>`;
  });
}

// ── Log renderer ──────────────────────────────────────────────────────────
function appendLog(text) {
  const el = document.getElementById('log-panel');
  const placeholder = document.getElementById('log-placeholder');
  if (placeholder) placeholder.style.display = 'none';

  const div = document.createElement('div');
  div.className = 'log-line';

  const isTiming = text.includes('[TIMING]');
  if (isTiming) {
    div.classList.add('TIMING');
  } else if (/\sINFO\s/.test(text) || text.includes('  INFO  ')) {
    div.classList.add('INFO');
  } else if (/\sDEBUG\s/.test(text) || text.includes('  DEBUG  ')) {
    div.classList.add('DEBUG');
  } else if (/\sWARNING\s/.test(text) || text.includes('  WARNING')) {
    div.classList.add('WARNING');
  } else if (/\sERROR\s|CRITICAL/.test(text)) {
    div.classList.add('ERROR');
  }
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

// ── Timing renderer ───────────────────────────────────────────────────────
function renderTiming(timing) {
  const box = document.getElementById('timing-box');
  const labels = {
    landing_page_fetch:       'Landing page fetch',
    industry_validation:      'Industry validation',
    link_classification:      'Link classification',
    knowledge_pages_collection:'Knowledge pages',
    knowledge_extraction_llm: 'Knowledge LLM',
    catalog_crawl_total:      'Catalog crawl',
    total:                    'TOTAL',
  };
  box.innerHTML = Object.entries(timing).map(([k, v]) => {
    const isTotal = k === 'total';
    const label = labels[k] || k;
    return `<div class="timing-row${isTotal ? ' total' : ''}">
      <span class="label">${label}</span>
      <span class="val">${v}s</span>
    </div>`;
  }).join('');
  document.getElementById('timing-section').style.display = 'block';
}

// ── Stats renderer ────────────────────────────────────────────────────────
function renderStats(result) {
  const { industry_match, pages_fetched, pages_failed, catalog_stats, errors } = result;
  const im = industry_match || {};

  // Industry match block
  const matchHtml = `
    <div class="match-block">
      <div class="match-row">
        <span class="label">Match result</span>
        <span class="badge ${im.matches ? 'badge-ok' : 'badge-err'}">
          ${im.matches ? '✓ Matches' : '✗ Mismatch'}
        </span>
      </div>
      ${im.claimed_industry ? `<div class="match-row">
        <span class="label">Claimed industry</span>
        <span>${im.claimed_industry}</span>
      </div>` : ''}
      ${im.detected_industry ? `<div class="match-row">
        <span class="label">Detected</span>
        <span>${im.detected_industry}</span>
      </div>` : ''}
      <div class="match-row">
        <span class="label">Confidence</span>
        <div class="confidence-bar" style="flex:1">
          <div class="confidence-fill" style="width:${Math.round((im.confidence||0)*100)}%"></div>
        </div>
        <span style="font-size:.8rem;color:var(--muted);margin-left:6px">${Math.round((im.confidence||0)*100)}%</span>
      </div>
      ${im.reason ? `<div style="font-size:.78rem;color:var(--muted);font-style:italic;margin-top:4px">${im.reason}</div>` : ''}
    </div>`;
  document.getElementById('industry-match-display').innerHTML = matchHtml;

  // Stat cards
  const catalog = catalog_stats || {};
  const cards = [
    { num: pages_fetched || 0, lbl: 'Pages fetched' },
    { num: pages_failed  || 0, lbl: 'Pages failed'  },
    { num: catalog.listing_pages || 0, lbl: 'Catalog listing pages' },
    { num: catalog.detail_pages  || 0, lbl: 'Catalog detail pages'  },
    { num: catalog.llm_calls     || 0, lbl: 'LLM calls'            },
    { num: (errors || []).length, lbl: 'Errors'                    },
  ];
  document.getElementById('stat-grid').innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="num">${c.num}</div>
      <div class="lbl">${c.lbl}</div>
    </div>`).join('');

  document.getElementById('stats-section').style.display = 'block';
}

// ── Main crawl function ───────────────────────────────────────────────────
let _active = false;
function startCrawl() {
  if (_active) return;

  const url    = document.getElementById('url').value.trim();
  const indust = document.getElementById('industry').value;
  const req    = document.getElementById('user_req').value.trim();
  const apiKey = document.getElementById('api_key').value.trim();

  if (!url) { alert('Please enter a URL.'); return; }

  // Reset UI
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Crawling…';
  _active = true;

  // Clear panels
  const logPanel = document.getElementById('log-panel');
  logPanel.innerHTML = '';
  document.getElementById('knowledge-content').style.display = 'none';
  document.getElementById('knowledge-placeholder').style.display = 'flex';
  document.getElementById('catalog-content').style.display = 'none';
  document.getElementById('catalog-placeholder').style.display = 'flex';
  document.getElementById('stats-section').style.display = 'none';
  document.getElementById('timing-section').style.display = 'none';

  switchTab('log');

  // Open SSE stream
  const params = new URLSearchParams({ url, industry: indust, user_request: req, api_key: apiKey });
  const es = new EventSource('/crawl?' + params.toString());

  es.addEventListener('log', e => {
    appendLog(e.data);
  });

  es.addEventListener('result', e => {
    es.close();
    _active = false;
    btn.disabled = false;
    btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Crawl';

    let result;
    try { result = JSON.parse(e.data); } catch { result = { error: e.data }; }

    if (result.error) {
      appendLog('[ERROR] ' + result.error);
      return;
    }

    // Knowledge tab
    const kc = document.getElementById('knowledge-content');
    kc.innerHTML = `<div class="json-view">${syntaxHighlight(result.knowledge || {})}</div>`;
    kc.style.display = 'block';
    document.getElementById('knowledge-placeholder').style.display = 'none';

    // Catalog tab
    const cc = document.getElementById('catalog-content');
    if (result.catalog) {
      cc.innerHTML = `<div class="json-view">${syntaxHighlight(result.catalog)}</div>`;
      cc.style.display = 'block';
      document.getElementById('catalog-placeholder').style.display = 'none';
    }

    renderTiming(result.timing || {});
    renderStats(result);

    appendLog('');
    appendLog('✓ Crawl complete. See Knowledge and Catalog tabs for results.');
    if (result._saved_file) {
      appendLog('💾 Saved → ' + result._saved_file);
    }
  });

  es.addEventListener('error', e => {
    if (_active) {
      appendLog('[ERROR] Connection lost or crawl failed.');
      es.close();
      _active = false;
      btn.disabled = false;
      btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Crawl';
    }
  });
}
</script>
</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return _HTML


@app.get("/crawl")
def crawl_sse():
    """SSE endpoint: streams log lines then emits the final result as JSON."""
    url          = request.args.get("url", "").strip()
    industry     = request.args.get("industry", "").strip() or None
    user_request = request.args.get("user_request", "").strip() or None
    api_key      = request.args.get("api_key", "").strip() or None

    log_q: queue.Queue[tuple[str, str]] = queue.Queue()
    result_holder: list[dict] = []

    # QueueHandler feeds into the SSE stream
    handler = _QueueHandler(log_q)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    _crawl_logger = logging.getLogger("crawl.run")

    def _worker():
        _crawl_logger.info("=" * 60)
        _crawl_logger.info("CRAWL START  url=%s  industry=%s", url, industry or "none")
        _crawl_logger.info("=" * 60)
        try:
            result = run_crawl_pure(
                url,
                industry=industry,
                user_request=user_request,
                api_key=api_key,
            )
            # Save to results/ folder
            saved_path = _save_result(result, url=url, industry=industry)
            result["_saved_file"] = str(saved_path)
            _crawl_logger.info("=" * 60)
            _crawl_logger.info("RESULT SAVED → %s", saved_path)
            _crawl_logger.info(
                "SUMMARY  pages_fetched=%d  pages_failed=%d  total_time=%ss",
                result.get("pages_fetched", 0),
                result.get("pages_failed", 0),
                result.get("timing", {}).get("total", "?"),
            )
            _crawl_logger.info("=" * 60)
            result_holder.append(result)
        except wc.IndustryMismatchError as exc:
            _crawl_logger.warning("INDUSTRY MISMATCH: %s", exc)
            result_holder.append({
                "error": str(exc),
                "industry_match": exc.match.as_dict(),
                "timing": {}, "pages_fetched": 0, "pages_failed": 0,
                "source_urls": [], "errors": [str(exc)],
            })
        except Exception as exc:
            _crawl_logger.error("CRAWL FAILED: %s", exc, exc_info=True)
            result_holder.append({"error": str(exc)})
        finally:
            log_q.put(("done", ""))
            root_logger.removeHandler(handler)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    def _generate():
        while True:
            try:
                kind, data = log_q.get(timeout=120)
            except queue.Empty:
                yield "event: error\ndata: timeout\n\n"
                break

            if kind == "log":
                # SSE: escape newlines inside the data field
                safe = data.replace("\n", " ")
                yield f"event: log\ndata: {safe}\n\n"
            elif kind == "done":
                payload = result_holder[0] if result_holder else {"error": "No result"}
                yield f"event: result\ndata: {json.dumps(payload)}\n\n"
                break

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n  Crawler Test UI running at  http://localhost:{port}\n")
    if not _fake_settings.OPENAI_API_KEY:
        print("  ⚠  OPENAI_API_KEY not set in environment.")
        print("     You can paste it in the API key field in the UI.\n")
    app.run(debug=False, port=port, threaded=True)
