"""
Embedder + indexer — fastembed (ONNX Runtime) only, no PyTorch.

Dense  : BAAI/bge-large-en-v1.5  via fastembed.TextEmbedding
Sparse : Qdrant/bm25              via fastembed.SparseTextEmbedding
"""

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding
from loguru import logger

from config import settings
from models import Chunk
from retrieval.vector_store import (
    collection_exists,
    create_collection,
    get_collection_name,
    list_collections,
    upsert_chunks,
)

_dense_model:  Optional[TextEmbedding]       = None
_sparse_model: Optional[SparseTextEmbedding] = None


def _get_dense() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense model: {settings.embedding_model}")
        kwargs = {"model_name": settings.embedding_model}
        if settings.model_cache_dir is not None:
            kwargs["cache_dir"] = str(settings.model_cache_dir)
        _dense_model = TextEmbedding(**kwargs)
    return _dense_model


def _get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info(f"Loading sparse model: {settings.sparse_model}")
        kwargs = {"model_name": settings.sparse_model}
        if settings.model_cache_dir is not None:
            kwargs["cache_dir"] = str(settings.model_cache_dir)
        _sparse_model = SparseTextEmbedding(**kwargs)
    return _sparse_model


def encode_dense(
    texts:      List[str],
    is_query:   bool = False,
    batch_size: int  = settings.embedding_batch_size,
) -> np.ndarray:
    model = _get_dense()
    if is_query:
        vecs = list(model.query_embed(texts))
    else:
        vecs = list(model.embed(texts, batch_size=batch_size))
    return np.array(vecs, dtype=np.float32)


def encode_sparse(
    texts:      List[str],
    batch_size: int = settings.embedding_batch_size,
) -> List[Tuple[List[int], List[float]]]:
    model = _get_sparse()
    return [(e.indices.tolist(), e.values.tolist()) for e in model.embed(texts, batch_size=batch_size)]


def encode_query(text: str) -> Tuple[List[float], List[int], List[float]]:
    dense  = encode_dense([text], is_query=True)[0]
    sparse = encode_sparse([text])[0]
    return dense.tolist(), sparse[0], sparse[1]


def index_chunks(
    chunks:        List[Chunk],
    batch_size:    int  = settings.embedding_batch_size,
    force_reindex: bool = False,
) -> None:
    grouped: dict = defaultdict(list)
    for chunk in chunks:
        grouped[get_collection_name(chunk.ticker, chunk.fiscal_year)].append(chunk)

    needs_indexing = []
    for name, col_chunks in sorted(grouped.items()):
        if collection_exists(name) and not force_reindex:
            logger.info(f"Skipping {name} — already indexed")
        else:
            needs_indexing.append((name, col_chunks))

    if not needs_indexing:
        logger.success(f"Done. Collections: {list_collections()}")
        return

    # Pre-warm both models before the loop so the download (if any) happens
    # upfront rather than mid-way through the first batch.
    _get_dense()
    _get_sparse()

    for col_name, col_chunks in needs_indexing:
        if not collection_exists(col_name):
            create_collection(col_name)

        logger.info(f"Embedding {len(col_chunks)} chunks → {col_name} …")
        texts = [c.text for c in col_chunks]

        dense_vecs  = encode_dense(texts,  batch_size=batch_size)
        sparse_vecs = encode_sparse(texts, batch_size=batch_size)

        upsert_chunks(
            collection_name=col_name,
            chunks=col_chunks,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            batch_size=64,
        )
        logger.success(f"Indexed → {col_name}  ({len(col_chunks)} points)")

    logger.success(f"Done. Collections: {list_collections()}")
