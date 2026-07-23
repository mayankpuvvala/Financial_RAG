"""
Main retrieval pipeline.

Flow per query:
  1. encode_query  → dense vector + sparse (indices, values)
  2. hybrid_search → top-20 results per collection (dense + sparse, RRF fused)
  3. merge results across collections, deduplicate, sort by score
  4. rerank        → top-5 via cross-encoder
  5. fetch parent section text for each surviving chunk
  6. return List[RetrievedChunk]

The caller (generation layer) receives the parent section text as LLM context
and the child chunk's metadata for citations.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from loguru import logger

from config import settings
from models import Chunk, RetrievedChunk
from ingestion.embedder import encode_query
from retrieval.vector_store import (
    collection_exists,
    get_collection_name,
    hybrid_search,
    list_collections,
    scroll_by_section,
)
from retrieval.reranker import rerank
from retrieval.parent_store import parent_store

# Which section a given `focus` (see routing/classifier.py's VALID_FOCUS)
# should be guaranteed a scroll-based candidate pass for, the same way
# "Consolidated Statements of Income" already gets one below. Casual phrasing
# ("nvidia sells what?") has poor lexical/semantic overlap with a section's
# formal wording, so hybrid search alone can miss it entirely — scrolling by
# section name sidesteps that instead of trying to out-guess every possible
# phrasing with keywords.
_FOCUS_SECTION_PASS = {
    "business_overview": "Item 1: Business",
    "risk_factors":       "Item 1A: Risk Factors",
    "legal_proceedings":  "Item 3: Legal Proceedings",
    "cybersecurity":      "Item 1C: Cybersecurity",
}


def _target_collections(
    tickers: List[str],
    years:   List[int],
) -> List[str]:
    """
    Resolve which Qdrant collections to search based on query context.

    Priority:
      tickers + years  → exact collections (AAPL_2024, AAPL_2023, …)
      tickers only     → all years for those tickers
      years only       → all tickers for those years
      neither          → every available collection
    """
    available = set(list_collections())

    if tickers and years:
        names = [get_collection_name(t, y) for t in tickers for y in years]
    elif tickers:
        names = [c for c in available if c.rsplit("_", 1)[0] in tickers]
    elif years:
        names = [
            c for c in available
            if c.rsplit("_", 1)[-1].isdigit() and int(c.rsplit("_", 1)[-1]) in years
        ]
    else:
        names = list(available)

    existing = [n for n in names if n in available]
    if not existing:
        logger.warning(f"No collections found for tickers={tickers} years={years}")
    return existing


def retrieve(
    query:   str,
    tickers: List[str]   = (),
    years:   List[int]   = (),
    top_k:   int         = settings.rerank_top_k,
    focus:   str         = "other",
) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline — returns RetrievedChunk objects ready for the LLM.

    `focus` comes from routing/classifier.py's per-query classification (the
    same Groq call that already extracts query_type/tickers/years, so this
    costs nothing extra) and tells retrieval which metric/section the query
    is actually about — see VALID_FOCUS there for the fixed category list.
    """
    collections = _target_collections(list(tickers), list(years))
    if not collections:
        return []

    # --- 1. Encode query ---
    dense, sparse_idx, sparse_val = encode_query(query)

    # --- 2. Hybrid search across all target collections, in parallel ---
    # Run four passes per collection:
    #   a) Main search across all chunk types (text + table + footnote)
    #   b) Table-only search — XBRL-formatted tables embed poorly and get
    #      pushed out of the top-20 by text chunks.  Searching tables
    #      separately guarantees financial statement tables (income statement,
    #      R&D table, balance sheet) are in the candidate pool.
    #   c) Consolidated Statements of Income pass — bank filings (JPM, BAC,
    #      GS, etc.) put the income statement in a separate named section that
    #      hybrid search misses because query_points filters are ignored in
    #      local Qdrant. Use scroll-based lookup instead.
    #   d) focus-driven section pass — see _FOCUS_SECTION_PASS above.
    #
    # Collections are independent, read-only lookups (each is its own
    # LocalCollection object, populated once at client startup and never
    # mutated afterward — see qdrant_client/local/qdrant_local.py), so
    # running them concurrently is safe: verified with 30 concurrent calls
    # across 3 collections (10 trials), comparing against sequential results
    # — zero mismatches, zero errors. For a multi-year query (3+ collections)
    # this cuts retrieval latency roughly in proportion to collection count
    # instead of paying for each one back-to-back.
    def _search_one_collection(col: str) -> List[dict]:
        results = hybrid_search(
            collection_name      = col,
            query_dense          = dense,
            query_sparse_indices = sparse_idx,
            query_sparse_values  = sparse_val,
            top_k                = settings.retrieval_top_k,
        )
        results += hybrid_search(
            collection_name      = col,
            query_dense          = dense,
            query_sparse_indices = sparse_idx,
            query_sparse_values  = sparse_val,
            top_k                = 10,
            chunk_type_filter    = "table",
        )
        results += scroll_by_section(col, "Consolidated Statements of Income", limit=5)

        focus_section = _FOCUS_SECTION_PASS.get(focus)
        if focus_section:
            results += scroll_by_section(col, focus_section, limit=5)

        # balance_sheet/segment_info live inside table-heavy or note sections
        # rather than a clean dedicated Item section, so they get their own
        # scroll target instead of an entry in _FOCUS_SECTION_PASS (whose
        # boost branch below only rescues TEXT chunks, not table rows).
        if focus == "balance_sheet":
            results += scroll_by_section(col, "Consolidated Balance Sheets", limit=5)
        elif focus == "segment_info":
            results += scroll_by_section(col, "Notes to Financial Statements", limit=5)

        return results

    raw_results: List[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(collections))) as pool:
        for hits in pool.map(_search_one_collection, collections):
            raw_results.extend(hits)

    if not raw_results:
        logger.warning("Hybrid search returned no results")
        return []

    # --- 3. Deduplicate + sort ---
    seen: set = set()
    unique: List[dict] = []
    for r in sorted(raw_results, key=lambda x: x["score"], reverse=True):
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)

    # Pass unique candidates to the cross-encoder so income-statement and
    # MD&A table chunks (which score ~#11-20 in BM25/dense hybrid) can beat
    # Notes segment-level chunks via the more precise reranker signal. Capped
    # rather than unbounded — for content-heavy filers (WFC's XBRL tables
    # alone produce 2000+ chunks/year) the raw candidate count can run into
    # the 60-90s, and reranking is CPU-bound (a full cross-encoder pass per
    # candidate), so that's real, avoidable per-query latency. #40 is a wide
    # enough margin above top_k=3 that nothing legitimately in contention
    # gets cut — anything ranked below #40 by hybrid dense+BM25+RRF was never
    # going to win the rerank anyway.
    _RERANK_CANDIDATE_CAP = 40
    candidates = unique[:_RERANK_CANDIDATE_CAP]

    # --- 4. Rerank ---
    # Score every candidate (within the cap above) so the focus-aware boost
    # below can still rescue income-statement chunks that rank outside the
    # top few by raw cross-encoder score.
    reranked = rerank(query, candidates, top_k=len(candidates))

    # --- 4b. Focus-aware aggregate boost.
    #
    # The ms-marco cross-encoder underranks income-statement rows because they
    # appear late in long XBRL table chunks. Apply targeted boosts using both
    # the chunk's SECTION and precise ROW LABEL patterns so we prefer the
    # consolidated P&L row over footnote mentions of the same metric. Which
    # branch applies is decided by `focus` (from the classifier — see its
    # docstring), not by re-deriving intent from query keywords here; the row-
    # label regexes below are a different, legitimate use of pattern matching
    # — extracting a specific row from a known-shape markdown table, not
    # guessing what the query means.
    import re as _re

    def _score(r: dict) -> float:
        return r.get("rerank_score", r["score"])

    def _section(r: dict) -> str:
        return r["payload"].get("section_name", "").lower()

    def _text(r: dict) -> str:
        return r["payload"].get("text", "")

    if focus == "revenue":
        _rev_row = _re.compile(r"\|\s*(?:total\s+)?(?:net\s+)?(?:revenue|net\s+sales|net\s+revenues)\s*\|", _re.IGNORECASE)
        for r in reranked:
            t  = _text(r)
            tl = t.lower()
            if _rev_row.search(t) or "total net sales" in tl or "total revenue" in tl or "total net revenues" in tl:
                r["rerank_score"] = _score(r) + 8.0   # consolidated total line / income-stmt row label
            elif "net sales" in tl or "revenue" in tl:
                r["rerank_score"] = _score(r) + 2.0   # segment or detail mention

    elif focus == "rd_expense":
        # Only boost the standalone "Research and development" income-statement
        # row, NOT "Capitalized research and development" or other footnotes.
        _rd_row = _re.compile(r"\|\s*research\s+and\s+development\s*\|", _re.IGNORECASE)
        for r in reranked:
            t = _text(r)
            if _rd_row.search(t):
                r["rerank_score"] = _score(r) + 10.0  # exact P&L row label
            elif "research and development" in t.lower():
                r["rerank_score"] = _score(r) + 2.0   # prose/other mentions

    elif focus == "net_income":
        # Prefer "Consolidated Statements of Income" section, then exact row
        # label. +10 for the income-statement section (which has higher Notes
        # CE scores of ~6 to beat).
        _ni_row = _re.compile(r"\|\s*net\s+income\s*\|", _re.IGNORECASE)
        for r in reranked:
            t = _text(r)
            sec = _section(r)
            if "consolidated statements" in sec:
                if _ni_row.search(t):
                    r["rerank_score"] = _score(r) + 10.0  # income stmt + exact NI row
                else:
                    r["rerank_score"] = _score(r) + 6.0   # income stmt, other rows
            elif _ni_row.search(t):
                r["rerank_score"] = _score(r) + 2.0       # exact NI row in other section
            elif "net income" in t.lower():
                r["rerank_score"] = _score(r) + 0.5       # prose mention

    elif focus == "operating_income":
        _oi_row = _re.compile(r"\|\s*(?:total\s+)?(?:operating\s+income|income\s+from\s+operations)\s*\|", _re.IGNORECASE)
        for r in reranked:
            t   = _text(r)
            tl  = t.lower()
            sec = _section(r)
            if "consolidated statements" in sec:
                if _oi_row.search(t):
                    r["rerank_score"] = _score(r) + 10.0  # income stmt + exact OI row
                else:
                    r["rerank_score"] = _score(r) + 6.0   # income stmt, other rows
            elif _oi_row.search(t):
                r["rerank_score"] = _score(r) + 2.0
            elif "operating income" in tl or "income from operations" in tl:
                r["rerank_score"] = _score(r) + 0.5

    elif focus == "balance_sheet":
        _bs_row = _re.compile(r"\|\s*(?:total\s+)?cash\s+and\s+cash\s+equivalents\s*\|", _re.IGNORECASE)
        for r in reranked:
            t   = _text(r)
            tl  = t.lower()
            sec = _section(r)
            if "balance sheet" in sec or "financial condition" in sec:
                if _bs_row.search(t):
                    r["rerank_score"] = _score(r) + 10.0  # balance sheet + exact row
                else:
                    r["rerank_score"] = _score(r) + 6.0   # balance sheet, other rows
            elif _bs_row.search(t) or "cash and cash equivalents" in tl:
                r["rerank_score"] = _score(r) + 2.0       # mention elsewhere

    elif focus == "segment_info":
        # No dedicated section exists for this — it's a footnote inside
        # "Notes to Financial Statements" — so boost purely on keyword
        # presence rather than section name.
        for r in reranked:
            tl = _text(r).lower()
            if "segment" in tl:
                r["rerank_score"] = _score(r) + (8.0 if "notes" in _section(r) else 4.0)

    elif focus in _FOCUS_SECTION_PASS:
        # business_overview / risk_factors — casual phrasing ("nvidia sells
        # what?") has poor lexical overlap with a section's own formal
        # wording, so the cross-encoder alone tends to rank financial-
        # statement tables above it. Boost the target section's text chunks
        # so the actual descriptive content wins.
        target_section = _FOCUS_SECTION_PASS[focus].lower()
        for r in reranked:
            if _section(r) == target_section and r["payload"].get("chunk_type") == "text":
                r["rerank_score"] = _score(r) + 8.0

    reranked = sorted(reranked, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    # --- 4c. Section + type diversity.
    #
    # Allow up to 2 TABLE chunks and 2 TEXT chunks per section.  Allowing 2
    # (rather than 1) per type+section pair is critical for XBRL tables that
    # are split into sub-chunks: the year-header sub-chunk and the financial-
    # data sub-chunk are both needed so the LLM can match numbers to years.
    from collections import defaultdict
    type_section_counts: dict = defaultdict(int)
    diverse: List[dict] = []
    for r in reranked:
        section    = r["payload"].get("section_name", "")
        chunk_type = r["payload"].get("chunk_type", "text")
        key = (section, chunk_type)
        if type_section_counts[key] < 2:
            diverse.append(r)
            type_section_counts[key] += 1
        if len(diverse) >= top_k:
            break
    reranked = diverse

    # --- 5. Build RetrievedChunk objects ---
    retrieved: List[RetrievedChunk] = []
    for result in reranked:
        payload = result["payload"]
        try:
            chunk = Chunk.model_validate({"chunk_id": result["id"], **payload})
        except Exception as exc:
            logger.warning(f"Could not reconstruct Chunk from payload: {exc}")
            continue

        parent_text = parent_store.get_section_text(
            doc_id     = chunk.doc_id,
            section_id = chunk.parent_id,
        )

        retrieved.append(RetrievedChunk(
            chunk       = chunk,
            score       = float(result.get("rerank_score", result["score"])),
            parent_text = parent_text if parent_text else chunk.text,
        ))

    logger.debug(
        f"Retrieved {len(retrieved)} chunks from {len(collections)} collection(s) "
        f"for: '{query[:60]}'"
    )
    return retrieved
