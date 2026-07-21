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
  - PRIORITIZED indexing: embedding is the slow part on CPU-only hardware
    (roughly linear in total tokens embedded — smaller chunks don't help,
    a smaller model would but at a quality cost). Most first questions about
    a new company are about revenue/margins/risk/segments, so we embed the
    chunks from the sections that answer those (financial statements, MD&A,
    risk factors, business overview) FIRST and return as soon as THAT
    subset is searchable. Every other section keeps embedding in a
    background thread so later, more specific questions eventually have
    full coverage too — without the first question waiting for all of it.

One-time cost for the prioritized subset is well under a minute for a
typical 10-K; the remaining sections finish over the following minutes in
the background. Every subsequent question about that company is instant,
same as the bundled 12, because the result is persisted to disk/Qdrant
exactly like they are.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config import settings, TICKER_TO_COMPANY
from ingestion.downloader import download_all_filings
from ingestion.parser import parse_all_filings
from ingestion.chunker import chunk_all_documents
from ingestion.embedder import index_chunks
from models import Chunk
from retrieval.vector_store import list_collections
from retrieval.parent_store import parent_store

_locks:       Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_failed:      Dict[str, float]          = {}   # ticker -> time.time() of last failure
_FAIL_TTL     = 300  # don't retry a failed ticker for 5 minutes

# Section-title keywords covering the questions people actually ask first:
# revenue/margins/net income (financial statements), outlook (MD&A), risk,
# and general business/segment info. Matched against chunk.section_name,
# which is always parser.py's clean display title (never the disambiguated
# internal section_id), so this works regardless of how many segments a
# filing was split across.
_PRIORITY_SECTION_KEYWORDS = (
    "risk factors", "md&a", "management's discussion", "business",
    "income", "balance sheet", "cash flow", "equity", "financial statements",
)
# Below this many chunks, splitting isn't worth the complexity — just index
# everything in one synchronous pass.
_MIN_CHUNKS_TO_SPLIT = 40


def _split_by_priority(chunks: List[Chunk]) -> Tuple[List[Chunk], List[Chunk]]:
    priority, remaining = [], []
    for c in chunks:
        name = c.section_name.lower()
        if any(k in name for k in _PRIORITY_SECTION_KEYWORDS):
            priority.append(c)
        else:
            remaining.append(c)
    return priority, remaining


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
            parent_store.reload()   # parsed doc text is ready regardless of embed progress
        except Exception as exc:
            logger.error(f"Auto-ingest parse/chunk failed for {ticker}: {exc}")
            _failed[ticker] = time.time()
            return None

        doc = documents[0]

        priority, remaining = _split_by_priority(chunks)
        if not priority or len(chunks) < _MIN_CHUNKS_TO_SPLIT:
            priority, remaining = chunks, []

        try:
            index_chunks(priority)
        except Exception as exc:
            logger.error(f"Auto-ingest embedding failed for {ticker}: {exc}")
            _failed[ticker] = time.time()
            return None

        logger.success(
            f"Auto-ingest ready: {ticker} FY{doc.fiscal_year} — "
            f"{len(priority)} priority chunks searchable now"
            + (f", {len(remaining)} more indexing in the background" if remaining else "")
        )

        if remaining:
            def _finish_background() -> None:
                try:
                    index_chunks(remaining, force_reindex=True)
                    logger.success(
                        f"Auto-ingest background completion done for {ticker}: "
                        f"{len(remaining)} additional chunks now searchable"
                    )
                except Exception as exc:
                    logger.error(f"Auto-ingest background completion failed for {ticker}: {exc}")

            threading.Thread(
                target=_finish_background,
                name=f"auto-ingest-finish-{ticker}",
                daemon=False,
            ).start()

        _failed.pop(ticker, None)
        return doc.company, doc.fiscal_year
