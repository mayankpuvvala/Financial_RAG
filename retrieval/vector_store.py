"""
Qdrant wrapper — collection management, upsert, and hybrid search.

Schema per collection (one collection = one ticker + fiscal year):
  dense vector  : 1024-dim COSINE  (BAAI/bge-large-en-v1.5)
  sparse vector : BM25              (Qdrant/bm25 via fastembed)
  payload       : all Chunk fields  (filterable)

Hybrid search uses Qdrant's built-in RRF fusion over prefetch results.
"""

import concurrent.futures
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config import settings


# ---------------------------------------------------------------------------
# Client — single shared instance
# ---------------------------------------------------------------------------

# A plain @lru_cache doesn't protect against a cold-cache race: retrieval
# now searches collections concurrently (see retriever.py), and if
# get_client() is first called from multiple threads at once — e.g. the
# very first multi-collection query after startup, since the API's warm-up
# path never touches Qdrant — two threads can both start constructing
# QdrantClient before either finishes and populates the cache. In local
# mode the second one then fails outright with "already accessed by
# another instance" (the exclusive file lock), even though both are in the
# same process. Double-checked locking makes only the first caller actually
# construct it; everyone else just reads the cache.
_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()

# Kept short — shorter than Railway's own proxy timeout for an unresponsive
# app (observed to give up around 15s with its own 502 "Application failed
# to respond"), so /health gets a chance to return ITS clear error first.
_CLIENT_CONSTRUCT_TIMEOUT = 8
_client_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qdrant-client-init")


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if settings.qdrant_url:
                    # Remote server (Qdrant Cloud or self-hosted) — just an
                    # HTTP client handle, no local storage to scan/lock, so
                    # no thread+timeout wrapper needed here. The `timeout`
                    # kwarg bounds each REST call instead.
                    _client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                        timeout=_CLIENT_CONSTRUCT_TIMEOUT,
                    )
                else:
                    # Local mode takes an EXCLUSIVE file lock on the storage
                    # folder at construction time and scans every collection
                    # in it. A corrupted/partially-written collection
                    # directory (e.g. an interrupted extraction — see
                    # api/app.py's restore-data endpoint) can make that scan
                    # hang rather than raise, wedging every caller waiting on
                    # _client_lock — including /health — indistinguishably
                    # from the process being down. Bounding it means at
                    # least one caller gets a clear, fast error instead of
                    # the platform's own opaque request timeout.
                    os.makedirs(settings.qdrant_path, exist_ok=True)
                    future = _client_executor.submit(QdrantClient, path=settings.qdrant_path)
                    try:
                        _client = future.result(timeout=_CLIENT_CONSTRUCT_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        raise RuntimeError(
                            f"Qdrant client construction did not complete within "
                            f"{_CLIENT_CONSTRUCT_TIMEOUT}s — data/qdrant may contain a "
                            f"corrupted collection (e.g. from an interrupted write)."
                        )
    return _client


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def get_collection_name(ticker: str, fiscal_year: int) -> str:
    return f"{ticker}_{fiscal_year}"


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

def collection_exists(name: str) -> bool:
    existing = {c.name for c in get_client().get_collections().collections}
    return name in existing


def _create_collection_on(client: QdrantClient, name: str) -> None:
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=settings.embedding_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    # Index payload fields so metadata filters are fast
    for field, schema in [
        ("ticker",       "keyword"),
        ("section_name", "keyword"),
        ("chunk_type",   "keyword"),
        ("fiscal_year",  "integer"),
    ]:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )
    logger.debug(f"Created Qdrant collection: {name}")


def create_collection(name: str) -> None:
    _create_collection_on(get_client(), name)


def delete_collection(name: str) -> None:
    get_client().delete_collection(name)
    logger.warning(f"Deleted collection: {name}")


def list_collections() -> List[str]:
    return sorted(c.name for c in get_client().get_collections().collections)


def get_collection_stats(name: str) -> Dict:
    info = get_client().get_collection(name)
    return {
        "name":         name,
        "points_count": info.points_count,
    }


# ---------------------------------------------------------------------------
# Local -> remote migration
# ---------------------------------------------------------------------------

def migrate_local_to_remote(
    remote_url: str,
    remote_api_key: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """
    Copy every collection from the process's current local-mode client into
    a fresh remote Qdrant instance, point-for-point (vectors + payload) —
    an alternative to re-running ingestion against the new store, which
    re-fetches and re-embeds every filing from scratch (see
    api/app.py's _run_ingestion_background — observed to take hours per
    company).

    Must be called before settings.qdrant_url is set (i.e. get_client()
    still resolves to local mode): local mode holds an exclusive file lock
    on qdrant_path, so a second local client can't be opened alongside the
    process's existing one to serve as the read side of the copy — this
    reuses get_client() itself as the source instead.

    Safe to call again after a partial/interrupted run: a collection whose
    remote point count already matches the local one is skipped outright;
    one that exists but is short is resumed (points are upserted by id, so
    re-sending already-copied ones is a harmless no-op) instead of trying
    to recreate it, which would error since it's already there.
    """
    source = get_client()
    dest = QdrantClient(url=remote_url, api_key=remote_api_key, timeout=30)
    log = on_progress or (lambda msg: None)
    existing_remote = {c.name for c in dest.get_collections().collections}

    counts: Dict[str, int] = {}
    for name in list_collections():
        if name in existing_remote:
            local_count = source.count(name).count
            remote_count = dest.count(name).count
            if remote_count == local_count:
                counts[name] = remote_count
                log(f"[{name}] already migrated ({remote_count} point(s)) — skipping")
                continue
            # Partially copied (e.g. a prior run got interrupted mid-collection)
            # — points are upserted by id below, so resuming is safe; no need
            # to recreate the collection.
        else:
            _create_collection_on(dest, name)
        moved = 0
        offset = None
        while True:
            batch, offset = source.scroll(
                collection_name=name,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not batch:
                break
            dest.upsert(
                collection_name=name,
                points=[
                    PointStruct(id=pt.id, vector=pt.vector, payload=pt.payload)
                    for pt in batch
                ],
                wait=True,
            )
            moved += len(batch)
            if offset is None:
                break
        counts[name] = moved
        logger.success(f"Migrated {moved} point(s): {name}")
        log(f"[{name}] migrated {moved} point(s)")

    return counts


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    collection_name: str,
    chunks: List[Any],                               # List[Chunk]
    dense_vectors: List[List[float]],
    sparse_vectors: List[Tuple[List[int], List[float]]],
    batch_size: int = 64,
) -> None:
    """Write chunks as Qdrant points in batches."""
    client = get_client()

    for i in range(0, len(chunks), batch_size):
        b_chunks  = chunks[i : i + batch_size]
        b_dense   = dense_vectors[i : i + batch_size]
        b_sparse  = sparse_vectors[i : i + batch_size]

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": (
                        dense.tolist() if hasattr(dense, "tolist") else list(dense)
                    ),
                    "sparse": SparseVector(
                        indices=sp_idx,
                        values=sp_val,
                    ),
                },
                payload=chunk.model_dump(exclude={"chunk_id"}),
            )
            for chunk, dense, (sp_idx, sp_val)
            in zip(b_chunks, b_dense, b_sparse)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    collection_name: str,
    query_dense: List[float],
    query_sparse_indices: List[int],
    query_sparse_values: List[float],
    top_k: int = settings.retrieval_top_k,
    chunk_type_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Dense + sparse prefetch with RRF fusion.
    Returns list of {id, score, payload} dicts — no Qdrant types leak out.
    """
    client = get_client()

    conditions = []
    if chunk_type_filter:
        conditions.append(
            FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type_filter))
        )
    qdrant_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=query_dense,               using="dense",  limit=top_k),
            Prefetch(
                query=SparseVector(
                    indices=query_sparse_indices,
                    values=query_sparse_values,
                ),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    return [
        {"id": str(p.id), "score": p.score, "payload": p.payload}
        for p in response.points
    ]


def scroll_by_section(
    collection_name: str,
    section_name: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Return all chunks whose section_name exactly matches *section_name*.

    Uses scroll (no vector scoring) so the caller must rank externally.
    Assigns a fixed score of 0.4 so these entries are included as candidates
    but don't dominate before the cross-encoder reranks them.

    Note: query_points() with a payload filter is silently ignored in local
    Qdrant (no payload indexes), so we fall back to scroll + Python filter.
    """
    client = get_client()
    results: List[Dict] = []
    offset = None

    while len(results) < limit:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=min(200, limit * 10),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in batch:
            if pt.payload.get("section_name", "") == section_name:
                results.append({"id": str(pt.id), "score": 0.4, "payload": pt.payload})
                if len(results) >= limit:
                    break
        if offset is None:
            break

    return results


def scroll_by_section_id(
    collection_name: str,
    section_id: str,
    limit: int = 10,
) -> List[Dict]:
    """
    Same as scroll_by_section, but matches the stable internal section id
    (Chunk.parent_id, e.g. "fs_income_stmt") instead of the display title.
    Display titles vary by filer wording for the same logical section —
    "Consolidated Statements of Operations" (Apple) vs "...of Income" (most
    others), "Consolidated Balance Sheets" vs "...Statements of Financial
    Condition" (banks) — so an exact-string match against one variant
    silently misses filers that use the other, starving them of the
    guaranteed-candidate pass entirely.
    """
    client = get_client()
    results: List[Dict] = []
    offset = None

    while len(results) < limit:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=min(200, limit * 10),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in batch:
            if pt.payload.get("parent_id", "") == section_id:
                results.append({"id": str(pt.id), "score": 0.4, "payload": pt.payload})
                if len(results) >= limit:
                    break
        if offset is None:
            break

    return results
