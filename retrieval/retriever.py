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
) -> List[RetrievedChunk]:
    """
    Full retrieval pipeline — returns RetrievedChunk objects ready for the LLM.
    """
    collections = _target_collections(list(tickers), list(years))
    if not collections:
        return []

    # --- 1. Encode query ---
    dense, sparse_idx, sparse_val = encode_query(query)

    # --- 2. Hybrid search across all target collections ---
    # Run two passes per collection:
    #   a) Main search across all chunk types (text + table + footnote)
    #   b) Table-only search — XBRL-formatted tables embed poorly and get
    #      pushed out of the top-20 by text chunks.  Searching tables
    #      separately guarantees financial statement tables (income statement,
    #      R&D table, balance sheet) are in the candidate pool.
    raw_results = []
    for col in collections:
        hits = hybrid_search(
            collection_name      = col,
            query_dense          = dense,
            query_sparse_indices = sparse_idx,
            query_sparse_values  = sparse_val,
            top_k                = settings.retrieval_top_k,
        )
        raw_results.extend(hits)

        table_hits = hybrid_search(
            collection_name      = col,
            query_dense          = dense,
            query_sparse_indices = sparse_idx,
            query_sparse_values  = sparse_val,
            top_k                = 10,
            chunk_type_filter    = "table",
        )
        raw_results.extend(table_hits)

        # c) Consolidated Statements of Income pass — bank filings (JPM, BAC,
        #    GS, etc.) put the income statement in a separate named section that
        #    hybrid search misses because query_points filters are ignored in
        #    local Qdrant. Use scroll-based lookup instead.
        stmt_hits = scroll_by_section(col, "Consolidated Statements of Income", limit=5)
        raw_results.extend(stmt_hits)

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

    # Pass ALL unique candidates to the cross-encoder so income-statement and
    # MD&A table chunks (which score ~#11-20 in BM25/dense hybrid) can beat
    # Notes segment-level chunks via the more precise reranker signal.
    candidates = unique

    # --- 4. Rerank ---
    # Score ALL unique candidates so the query-aware boost below can rescue
    # income-statement chunks that rank outside the top-9 cross-encoder window.
    reranked = rerank(query, candidates, top_k=len(candidates))

    # --- 4b. Query-aware aggregate boost.
    #
    # The ms-marco cross-encoder underranks income-statement rows because they
    # appear late in long XBRL table chunks.  Apply targeted boosts using both
    # the chunk's SECTION and precise ROW LABEL patterns so we prefer the
    # consolidated P&L row over footnote mentions of the same metric.
    import re as _re
    _q = query.lower()

    def _score(r: dict) -> float:
        return r.get("rerank_score", r["score"])

    def _section(r: dict) -> str:
        return r["payload"].get("section_name", "").lower()

    def _text(r: dict) -> str:
        return r["payload"].get("text", "")

    # Revenue / net sales
    # Additive bonuses so the boost works for both positive and negative CE scores.
    if any(k in _q for k in ["revenue", "net sales", "total sales"]):
        for r in reranked:
            t = _text(r).lower()
            if "total net sales" in t or "total revenue" in t or "total net revenues" in t:
                r["rerank_score"] = _score(r) + 8.0   # consolidated total line
            elif "net sales" in t or "revenue" in t:
                r["rerank_score"] = _score(r) + 2.0   # segment or detail mention

    # R&D: only boost the standalone "Research and development" income-statement row,
    # NOT "Capitalized research and development" or other footnote variants.
    elif any(k in _q for k in ["research", "r&d", "development expense"]):
        _rd_row = _re.compile(r"\|\s*research\s+and\s+development\s*\|", _re.IGNORECASE)
        for r in reranked:
            t = _text(r)
            if _rd_row.search(t):
                r["rerank_score"] = _score(r) + 10.0  # exact P&L row label
            elif "research and development" in t.lower():
                r["rerank_score"] = _score(r) + 2.0   # prose/other mentions

    # Net income: prefer "Consolidated Statements of Income" section, then exact row label.
    # +10 for the income-statement section (which has higher Notes CE scores of ~6 to beat).
    elif any(k in _q for k in ["net income", "earnings", "profit", "net loss"]):
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
