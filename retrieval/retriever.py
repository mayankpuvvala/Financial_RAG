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

    candidates = unique[: settings.retrieval_top_k]

    # --- 4. Rerank ---
    reranked = rerank(query, candidates, top_k=top_k)

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
