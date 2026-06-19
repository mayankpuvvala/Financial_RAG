"""
Cross-encoder reranker — BAAI/bge-reranker-base.

Takes the top-K hybrid search results and re-scores each (query, chunk_text)
pair with a dedicated relevance model. Much more accurate than embedding
cosine similarity because it sees both query and passage together.
"""

from typing import List, Dict
from functools import lru_cache

from loguru import logger
from sentence_transformers import CrossEncoder

from config import settings


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    logger.info(f"Loading reranker: {settings.reranker_model}  (first load)")
    return CrossEncoder(settings.reranker_model)


def rerank(
    query:      str,
    candidates: List[Dict],
    top_k:      int = settings.rerank_top_k,
) -> List[Dict]:
    """
    Score each candidate with the cross-encoder and return the top_k highest.

    Each candidate dict must have a 'payload' key with a 'text' field.
    A 'rerank_score' key is added to each returned dict.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    texts    = [c["payload"]["text"] for c in candidates]
    pairs    = [(query, t) for t in texts]
    scores   = reranker.predict(pairs, show_progress_bar=False)

    scored = sorted(
        zip(scores.tolist(), candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, candidate in scored[:top_k]:
        candidate = dict(candidate)          # shallow copy — don't mutate caller's list
        candidate["rerank_score"] = score
        results.append(candidate)

    return results
