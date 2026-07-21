"""
Cross-encoder reranker — ms-marco-MiniLM-L-12-v2, via fastembed's ONNX
TextCrossEncoder (same model weights as the sentence-transformers/PyTorch
cross-encoder this used to run, just ONNX-converted) so the whole retrieval
stack — dense embedding, sparse embedding, and reranking — shares one ONNX
runtime instead of also pulling in PyTorch. That matters most on memory-
constrained hosts (e.g. Railway's smaller tiers): PyTorch's own runtime
footprint is large regardless of model size, so dropping it here is the
single biggest lever for staying under a low memory cap.

Takes the top-K hybrid search results and re-scores each (query, chunk_text)
pair with a dedicated relevance model. Much more accurate than embedding
cosine similarity because it sees both query and passage together.
"""

import re
from typing import List, Dict
from functools import lru_cache

from loguru import logger
from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import settings

# Matches XBRL table pipe rows where a cell repeats ≥2 times consecutively.
_XBRL_REPEAT_RE = re.compile(r"(\|[^|]+)\1+", re.MULTILINE)


_TOTAL_ROW_RE = re.compile(r"^\|\s*total\b", re.IGNORECASE)


def _compress_xbrl(text: str) -> str:
    """
    Collapse XBRL markdown table noise so the cross-encoder can see key metrics.

    Two transformations:
    1. De-duplicate repeated XBRL cells:
       "| Net sales | Net sales | $ | 391035 | 391035 |" → "| Net sales | $ | 391035 |"
    2. Float "Total" rows to the top of each table chunk so the cross-encoder's
       early attention tokens see the consolidated figure first, not segment rows.
       This fixes the case where "Total net sales | 391035" is buried under
       many geographic or product segment rows.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if "|" not in line or "---" in line:
            cleaned.append(line)
            continue
        # De-duplicate adjacent identical cells
        compressed = _XBRL_REPEAT_RE.sub(r"\1", line)
        # Remove trailing empty cells
        compressed = re.sub(r"(\|\s*)+$", "|", compressed.rstrip())
        cleaned.append(compressed)

    # Float "Total …" rows to just below the header (first two lines)
    if len(cleaned) > 3:
        header     = cleaned[:2]          # chunk title + markdown separator row
        data_rows  = cleaned[2:]
        total_rows = [r for r in data_rows if _TOTAL_ROW_RE.match(r.lstrip())]
        other_rows = [r for r in data_rows if not _TOTAL_ROW_RE.match(r.lstrip())]
        if total_rows:
            cleaned = header + total_rows + other_rows

    return "\n".join(cleaned)


@lru_cache(maxsize=1)
def _get_reranker() -> TextCrossEncoder:
    logger.info(f"Loading reranker: {settings.reranker_model}  (first load)")
    return TextCrossEncoder(
        model_name=settings.reranker_model,
        providers=["CPUExecutionProvider"],
        enable_cpu_mem_arena=False,   # see ingestion/embedder.py::_get_dense()
    )


def unload_reranker() -> None:
    """Drop the cached reranker — see ingestion/embedder.py::unload_models()."""
    _get_reranker.cache_clear()


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
    # Compress XBRL noise before scoring so the cross-encoder sees clean
    # "label | value" pairs rather than "label | label | label | value | value".
    texts  = [_compress_xbrl(c["payload"]["text"]) for c in candidates]
    scores = list(reranker.rerank(query, texts))

    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, candidate in scored[:top_k]:
        candidate = dict(candidate)          # shallow copy — don't mutate caller's list
        candidate["rerank_score"] = score
        results.append(candidate)

    return results
