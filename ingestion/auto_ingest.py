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

import json
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger

from config import settings, COMPANIES, TICKER_TO_COMPANY
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

_BUNDLED_TICKERS = {c["ticker"] for c in COMPANIES}

# Every auto-ingested company (i.e. everything NOT in the bundled 12 — see
# _BUNDLED_TICKERS) sticks around in Qdrant/data/parsed forever once
# indexed, same as the bundled ones, with no cap. That's fine for a
# handful of companies someone actually asks about, but unbounded on a
# storage-capped host if people keep asking about new ones. This log is
# the "last asked about" signal evict_stale_companies() uses to decide
# what's safe to remove — every ensure_ticker_indexed() call touches it,
# whether it's a fresh ingest or an instant "already indexed" hit.
_ACCESS_LOG_PATH = settings.data_dir / "auto_ingest_access.json"


def _load_access_log() -> Dict[str, str]:
    if not _ACCESS_LOG_PATH.exists():
        return {}
    try:
        return json.loads(_ACCESS_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _touch_access(ticker: str) -> None:
    log = _load_access_log()
    log[ticker] = datetime.now(timezone.utc).isoformat()
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _ACCESS_LOG_PATH.write_text(json.dumps(log), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not persist auto-ingest access log: {exc}")


def evict_stale_companies(max_age_days: int = 30, dry_run: bool = True) -> Dict[str, dict]:
    """
    Remove auto-ingested companies (never the bundled 12 — guarded by
    _BUNDLED_TICKERS regardless of what the access log contains) not asked
    about in over max_age_days: their Qdrant collections and data/parsed
    JSON, which otherwise sit there indefinitely. dry_run (default) reports
    what WOULD be removed without deleting anything.
    """
    from retrieval.vector_store import delete_collection

    log = _load_access_log()
    cutoff = time.time() - max_age_days * 86400
    stale = [
        ticker for ticker, iso in log.items()
        if ticker not in _BUNDLED_TICKERS
        and datetime.fromisoformat(iso).timestamp() < cutoff
    ]

    all_collections = list_collections()
    report: Dict[str, dict] = {}
    for ticker in stale:
        cols = [c for c in all_collections if c.rsplit("_", 1)[0] == ticker]
        parsed_files = sorted(settings.parsed_dir.glob(f"{ticker}_*.json"))
        report[ticker] = {
            "collections":  cols,
            "parsed_files": [p.name for p in parsed_files],
        }
        if not dry_run:
            for c in cols:
                delete_collection(c)
            for p in parsed_files:
                p.unlink(missing_ok=True)
            log.pop(ticker, None)

    if not dry_run and stale:
        try:
            _ACCESS_LOG_PATH.write_text(json.dumps(log), encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Could not persist auto-ingest access log after eviction: {exc}")
        parent_store.reload()
        logger.warning(f"Evicted {len(stale)} stale auto-ingested compan(ies): {stale}")

    return report

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


class YearNotAvailable(Exception):
    """
    Raised when `ticker` resolves and ingests fine, but the SPECIFIC
    fiscal year asked for isn't among the filings SEC EDGAR actually has
    for it (either the company's history doesn't go back that far, or the
    request is for a year not yet filed). Distinct from returning None:
    that still means "something about this ticker failed"; this means
    "the ticker is fine, just not for this particular year" — a caller
    needs to tell those apart to give an accurate message.
    """
    def __init__(self, ticker: str, requested: int, available: List[int]):
        self.ticker = ticker
        self.requested = requested
        self.available = available
        super().__init__(f"{ticker}: FY{requested} not available (have {available})")


def _indexed_years(ticker: str) -> List[int]:
    years = []
    for c in list_collections():
        t, _, y = c.rpartition("_")
        if t == ticker and y.isdigit():
            years.append(int(y))
    return sorted(years)


def ensure_ticker_indexed(
    ticker: str,
    company_name: str,
    target_year: Optional[int] = None,
) -> Optional[Tuple[str, int]]:
    """
    Make sure `ticker`'s 10-K is indexed and searchable, for `target_year`
    specifically if given, otherwise whatever's most recent. Returns
    (company_name, fiscal_year) on success, None on failure (unknown
    ticker, no 10-K on file at all, network error, etc.), or raises
    YearNotAvailable if the ticker is fine but not for that year.
    """
    ticker = ticker.upper()
    if ticker not in _BUNDLED_TICKERS:
        _touch_access(ticker)

    # Fast path — already indexed, no I/O at all. When a specific year is
    # requested, only THAT year satisfies the fast path — other years
    # already being indexed for this ticker doesn't mean this one is, so
    # don't take the shortcut on their account.
    indexed = _indexed_years(ticker)
    if target_year is not None:
        if target_year in indexed:
            return company_name, target_year
    elif indexed:
        return company_name, max(indexed)

    last_fail = _failed.get(ticker)
    if last_fail and (time.time() - last_fail) < _FAIL_TTL:
        return None

    with _lock_for(ticker):
        # Re-check inside the lock: another thread may have just finished.
        indexed = _indexed_years(ticker)
        if target_year is not None:
            if target_year in indexed:
                return company_name, target_year
        elif indexed:
            return company_name, max(indexed)

        logger.info(
            f"Auto-ingest: '{ticker}' "
            + (f"FY{target_year} not indexed yet — fetching …" if target_year is not None
               else "not indexed yet — fetching latest 10-K …")
        )
        TICKER_TO_COMPANY.setdefault(ticker, {"name": company_name, "sector": "Unknown"})

        # A specific year might not be the LATEST filing, so fetch the
        # recent window (matches settings.filings_per_company — the same
        # window the bundled-12 pipeline indexes, and not coincidentally
        # the same span as classifier.py's VALID_YEARS) instead of just
        # the single latest one. Whatever else comes back in that window
        # gets indexed too, at no extra network cost — a query for FY2023
        # leaves FY2024/2025 already searchable as a side effect.
        limit = settings.filings_per_company if target_year is not None else 1
        try:
            records = download_all_filings(
                companies=[{"ticker": ticker, "name": company_name, "sector": "Unknown"}],
                limit=limit,
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

        if target_year is not None:
            available = sorted({d.fiscal_year for d in documents})
            if target_year not in available:
                # Not a failure of the ticker itself — everything fetched
                # gets indexed below regardless, same as any other run.
                logger.warning(f"Auto-ingest: {ticker} has no FY{target_year} filing (have {available})")
                doc = documents[0]
            else:
                doc = next(d for d in documents if d.fiscal_year == target_year)
        else:
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

        # Same pruning api/app.py's bundled-12 ingestion does, and for the
        # same reason: raw HTML and the chunks JSON are scratch space for
        # getting to embeddings, never read again once index_chunks() has
        # run (it takes `chunks`/`remaining` as in-memory Chunk lists, not
        # by re-reading these files). The bundled pipeline already prunes
        # itself; this path didn't, so every auto-ingested company outside
        # the original 12 — which is now most of them, see /health's
        # collection count — left its raw filing (tens of MB) and chunk
        # JSON sitting on the Railway volume forever. Safe to do before the
        # background `remaining` thread finishes, since it doesn't touch
        # either file either. One chunks.json per document now that a
        # single call can fetch several years' worth.
        raw_ticker_dir = settings.raw_dir / "sec-edgar-filings" / ticker
        shutil.rmtree(raw_ticker_dir, ignore_errors=True)
        for d in documents:
            (settings.chunks_dir / f"{ticker}_{d.fiscal_year}_chunks.json").unlink(missing_ok=True)

        years_indexed = sorted({d.fiscal_year for d in documents})
        logger.success(
            f"Auto-ingest ready: {ticker} FY{'/'.join(map(str, years_indexed))} — "
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

        if target_year is not None and target_year not in years_indexed:
            raise YearNotAvailable(ticker, target_year, years_indexed)

        return doc.company, doc.fiscal_year
