"""
SEC company registry — resolves ANY publicly traded company (ticker or name)
to a {ticker, title, cik} record, not just the 12 bundled in config.COMPANIES.

Backed by SEC's public company_tickers.json (one entry per registered filer),
cached to disk so normal queries never touch the network.
"""

import json
import time
from typing import Dict, List, Optional

import requests
from loguru import logger

from config import settings, TICKER_TO_COMPANY

_REGISTRY_URL  = "https://www.sec.gov/files/company_tickers.json"
_CACHE_PATH    = settings.data_dir / "company_tickers.json"
_CACHE_MAX_AGE = 7 * 24 * 3600   # 1 week — filer list changes rarely

_by_ticker: Dict[str, dict] = {}
_titles:    List[tuple]     = []   # (lowercase title, record)
_loaded = False


def _headers() -> dict:
    # SEC fair-use policy requires an identifying User-Agent on every request.
    return {"User-Agent": f"FinancialRAG {settings.edgar_email}"}


def _load_registry() -> None:
    global _loaded
    if _loaded:
        return

    data = None
    if _CACHE_PATH.exists() and (time.time() - _CACHE_PATH.stat().st_mtime) < _CACHE_MAX_AGE:
        try:
            data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            data = None

    if data is None:
        try:
            resp = requests.get(_REGISTRY_URL, headers=_headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
            logger.info(f"Fetched SEC company registry ({len(data)} filers)")
        except Exception as exc:
            logger.warning(f"Could not fetch SEC company registry ({exc})")
            if _CACHE_PATH.exists():
                data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            else:
                data = {}

    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper()
        title  = str(entry.get("title", ""))
        cik    = str(entry.get("cik_str", "")).zfill(10)
        if not ticker:
            continue
        rec = {"ticker": ticker, "title": title, "cik": cik}
        _by_ticker[ticker] = rec
        if title:
            _titles.append((title.lower(), rec))

    _loaded = True
    logger.debug(f"SEC company registry ready: {len(_by_ticker)} tickers indexed")


def resolve_company(mention: str) -> Optional[dict]:
    """
    Resolve a free-text ticker or company-name mention to {ticker, title, cik}.
    Returns None if nothing plausible is found.
    """
    m = mention.strip()
    if not m:
        return None

    # Already one of the 12 bundled companies — no network needed.
    if m.upper() in TICKER_TO_COMPANY:
        info = TICKER_TO_COMPANY[m.upper()]
        return {"ticker": m.upper(), "title": info["name"], "cik": None}

    _load_registry()

    if m.upper() in _by_ticker:
        return _by_ticker[m.upper()]

    # Company-name match — shortest containing title wins (most specific).
    ml = m.lower()
    candidates = [rec for title, rec in _titles if ml in title or title in ml]
    if candidates:
        return min(candidates, key=lambda r: len(r["title"]))

    return None
