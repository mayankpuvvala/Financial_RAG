"""
Qdrant wrapper — collection management, upsert, and hybrid search.

Schema per collection (one collection = one ticker + fiscal year):
  dense vector  : 1024-dim COSINE  (BAAI/bge-large-en-v1.5)
  sparse vector : BM25              (Qdrant/bm25 via fastembed)
  payload       : all Chunk fields  (filterable)

Hybrid search uses Qdrant's built-in RRF fusion over prefetch results.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

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

# Local Qdrant takes an EXCLUSIVE file lock on the storage folder at
# construction time. A plain @lru_cache doesn't protect against a cold-cache
# race: retrieval now searches collections concurrently (see retriever.py),
# and if get_client() is first called from multiple threads at once — e.g.
# the very first multi-collection query after startup, since the API's
# warm-up path never touches Qdrant — two threads can both start
# constructing QdrantClient before either finishes and populates the cache,
# and the second one fails with "already accessed by another instance" even
# though they're in the same process. Double-checked locking makes only the
# first caller actually construct it; everyone else just reads the cache.
_client: Optional[QdrantClient] = None
_client_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                os.makedirs(settings.qdrant_path, exist_ok=True)
                _client = QdrantClient(path=settings.qdrant_path)
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


def create_collection(name: str) -> None:
    client = get_client()
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
