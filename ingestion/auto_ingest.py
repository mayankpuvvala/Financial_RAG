"""
On-demand single-company ingestion.

If a user asks about a company outside the 12 bundled tickers, we fetch just
that one company's latest 10-K, parse/chunk/index it, and answer — instead
of being limited to the pre-ingested set. Kept fast by:
  - downloading only the single latest filing (limit=1), not all 3 years
  - reusing the dense/sparse/reranker models the API already warmed up
  - a per-ticker lock so concurrent requests for the same new company don't
    duplicate the download/parse/embed work
  - an in-memory "already tried and failed" cache so a bad ticker mention
    doesn't re-hit SEC EDGAR on every message in a chat session

One-time cost is a few minutes on CPU-only hardware (dominated by embedding
the new filing's chunks — a typical 10-K is 100-300 chunks), but every
subsequent question about that company is instant, same as the bundled 12,
because the result is persisted to disk/Qdrant exactly like they are.
"""

import threading
import time
from typing import Dict, Optional, Tuple

from loguru import logger

from config import settings, TICKER_TO_COMPANY
from ingestion.downloader import download_all_filings
from ingestion.parser import parse_all_filings
from ingestion.chunker import chunk_all_documents
from ingestion.embedder import index_chunks
from retrieval.vector_store import list_collections
from retrieval.parent_store import parent_store

_locks:       Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_failed:      Dict[str, float]          = {}   # ticker -> time.time() of last failure
_FAIL_TTL     = 300  # don't retry a failed ticker for 5 minutes


def _lock_for(ticker: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(ticker, threading.Lock())


def _existing_year(ticker: str) -> Optional[int]:
    """Latest fiscal year already indexed for this ticker, if any."""
    years = []
    for c in list_collections():
        t, _, y = c.rpartition("_")
        if t == ticker and y.isdigit():
            years.append(int(y))
    return max(years) if years else None


def ensure_ticker_indexed(ticker: str, company_name: str) -> Optional[Tuple[str, int]]:
    """
    Make sure at least one fiscal year of `ticker`'s 10-K is indexed and
    searchable. Returns (company_name, fiscal_year) on success, None on
    failure (unknown ticker, no 10-K on file, network error, etc.).
    """
    ticker = ticker.upper()

    # Fast path — already indexed, no I/O at all.
    existing = _existing_year(ticker)
    if existing is not None:
        return company_name, existing

    last_fail = _failed.get(ticker)
    if last_fail and (time.time() - last_fail) < _FAIL_TTL:
        return None

    with _lock_for(ticker):
        # Re-check inside the lock: another thread may have just finished.
        existing = _existing_year(ticker)
        if existing is not None:
            return company_name, existing

        logger.info(f"Auto-ingest: '{ticker}' not indexed yet — fetching latest 10-K …")
        TICKER_TO_COMPANY.setdefault(ticker, {"name": company_name, "sector": "Unknown"})

        try:
            records = download_all_filings(
                companies=[{"ticker": ticker, "name": company_name, "sector": "Unknown"}],
                limit=1,
            )
        except Exception as exc:
            logger.error(f"Auto-ingest download failed for {ticker}: {exc}")
            _failed[ticker] = time.time()
            return None

        if not records:
            logger.warning(f"Auto-ingest: no 10-K found on EDGAR for {ticker}")
            _failed[ticker] = time.time()
            return None

        try:
            documents = parse_all_filings(records, settings.parsed_dir)
            if not documents:
                _failed[ticker] = time.time()
                return None
            chunks = chunk_all_documents(documents, settings.chunks_dir)
            index_chunks(chunks)
            parent_store.reload()
        except Exception as exc:
            logger.error(f"Auto-ingest parse/chunk/index failed for {ticker}: {exc}")
            _failed[ticker] = time.time()
            return None

        doc = documents[0]
        logger.success(
            f"Auto-ingest complete: {ticker} FY{doc.fiscal_year} ready ({len(chunks)} chunks)"
        )
        _failed.pop(ticker, None)
        return doc.company, doc.fiscal_year
