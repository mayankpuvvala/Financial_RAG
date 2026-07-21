"""
Embedder + indexer — fastembed (ONNX Runtime) only, no PyTorch.

Dense  : BAAI/bge-large-en-v1.5  via fastembed.TextEmbedding
Sparse : Qdrant/bm25              via fastembed.SparseTextEmbedding
"""

import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set, Tuple

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


def _cuda_providers() -> list:
    """Return CUDA+CPU providers if a GPU is present, else CPU only."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            logger.info("GPU detected — using CUDAExecutionProvider for embeddings")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _get_dense() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info(f"Loading dense model: {settings.embedding_model}")
        kwargs: dict = {
            "model_name": settings.embedding_model,
            "providers": _cuda_providers(),
            # ONNX Runtime's default CPU memory arena pre-allocates and holds
            # onto memory well beyond the model's own weights (measured ~800MB
            # resident for a 215MB model) to avoid malloc/free overhead across
            # repeated inference calls. Disabling it cuts that to ~450-650MB —
            # the difference between fitting and OOM-killing on a 512MB-1GB
            # memory-capped host (Railway's smaller tiers, etc.) — at the cost
            # of allocating fresh buffers per call instead of reusing a pool,
            # which barely matters here since compute time already dominates.
            "enable_cpu_mem_arena": False,
        }
        if settings.model_cache_dir is not None:
            kwargs["cache_dir"] = str(settings.model_cache_dir)
        _dense_model = TextEmbedding(**kwargs)
    return _dense_model


def _get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info(f"Loading sparse model: {settings.sparse_model}")
        kwargs: dict = {
            "model_name": settings.sparse_model,
            "providers": _cuda_providers(),
            "enable_cpu_mem_arena": False,   # see _get_dense()
        }
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

    # Pre-warm both models before embedding so any model download happens once.
    _get_dense()
    _get_sparse()

    # Embed ALL chunks across every collection in a single pass so the ONNX
    # model processes one large batch instead of many small per-collection ones.
    # This is significantly faster on both CPU (better SIMD utilisation) and
    # GPU (hides kernel launch latency).
    all_texts = [c.text for _, col_chunks in needs_indexing for c in col_chunks]
    logger.info(f"Embedding {len(all_texts)} chunks in one batch (batch_size={batch_size}) …")
    all_dense  = encode_dense(all_texts,  batch_size=batch_size)
    all_sparse = encode_sparse(all_texts, batch_size=batch_size)

    offset = 0
    for col_name, col_chunks in needs_indexing:
        if not collection_exists(col_name):
            create_collection(col_name)

        n = len(col_chunks)
        upsert_chunks(
            collection_name=col_name,
            chunks=col_chunks,
            dense_vectors=all_dense[offset : offset + n],
            sparse_vectors=all_sparse[offset : offset + n],
            batch_size=64,
        )
        logger.success(f"Indexed → {col_name}  ({n} points)")
        offset += n

    logger.success(f"Done. Collections: {list_collections()}")


# ---------------------------------------------------------------------------
# Isolated-process variant
# ---------------------------------------------------------------------------
#
# Local (embedded, file-based) Qdrant takes an EXCLUSIVE lock on its storage
# folder per OS process — only one process can ever hold it. When ingestion
# runs as a separate subprocess (to protect the API server from an OOM crash
# during the heavy parse/embed work), that subprocess can never open Qdrant
# itself: the API server already holds the lock for the lifetime of its
# process. So the split here is: the subprocess computes embeddings (the part
# that actually needs isolating) and writes them to disk as plain pickles;
# the API server process — which already holds the lock — picks them up and
# does the actual (cheap, fast) Qdrant write via flush_pending_artifacts().

def embed_chunks_to_artifacts(
    chunks:            List[Chunk],
    artifacts_dir:     Path,
    skip_collections:  Set[str],
    batch_size:        int = settings.embedding_batch_size,
) -> int:
    """
    Compute embeddings and write one pickle per collection to *artifacts_dir*,
    without touching Qdrant. Collections already present in *skip_collections*
    are skipped (the caller supplies this since collection_exists() would
    require the Qdrant lock this process doesn't have). Returns the number of
    artifact files written.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict = defaultdict(list)
    for chunk in chunks:
        grouped[get_collection_name(chunk.ticker, chunk.fiscal_year)].append(chunk)

    needs_indexing = [
        (name, col_chunks) for name, col_chunks in sorted(grouped.items())
        if name not in skip_collections
    ]

    if not needs_indexing:
        logger.success("No new collections to embed.")
        return 0

    _get_dense()
    _get_sparse()

    all_texts = [c.text for _, col_chunks in needs_indexing for c in col_chunks]
    logger.info(f"Embedding {len(all_texts)} chunks in one batch (batch_size={batch_size}) …")
    all_dense  = encode_dense(all_texts,  batch_size=batch_size)
    all_sparse = encode_sparse(all_texts, batch_size=batch_size)

    offset = 0
    written = 0
    for col_name, col_chunks in needs_indexing:
        n = len(col_chunks)
        artifact_path = artifacts_dir / f"{col_name}.pkl"
        with artifact_path.open("wb") as fh:
            pickle.dump(
                {
                    "collection_name": col_name,
                    "chunks":          col_chunks,
                    "dense_vectors":   all_dense[offset : offset + n],
                    "sparse_vectors":  all_sparse[offset : offset + n],
                },
                fh,
            )
        logger.success(f"Embedded → {artifact_path.name}  ({n} points)")
        offset += n
        written += 1

    return written


def flush_pending_artifacts(artifacts_dir: Path) -> int:
    """
    Read every *.pkl in *artifacts_dir* (written by embed_chunks_to_artifacts)
    and upsert it into Qdrant. Must run in a process that already holds the
    Qdrant lock (the API server). Deletes each artifact after a successful
    upsert. Returns the number of collections flushed.
    """
    if not artifacts_dir.exists():
        return 0

    flushed = 0
    for artifact_path in sorted(artifacts_dir.glob("*.pkl")):
        with artifact_path.open("rb") as fh:
            data = pickle.load(fh)

        col_name = data["collection_name"]
        if not collection_exists(col_name):
            create_collection(col_name)

        upsert_chunks(
            collection_name=col_name,
            chunks=data["chunks"],
            dense_vectors=data["dense_vectors"],
            sparse_vectors=data["sparse_vectors"],
            batch_size=64,
        )
        logger.success(f"Flushed → {col_name}  ({len(data['chunks'])} points)")
        artifact_path.unlink()
        flushed += 1

    return flushed
