"""
Answer generator — Groq (llama-3.3-70b-versatile) + citation builder.

Flow:
  1. Build a numbered context block from retrieved chunks (parent section text)
  2. Call Groq with a strict grounding prompt
  3. Return answer + structured citation list
"""

from functools import lru_cache
from typing import List, Optional

import tiktoken
from groq import Groq
from loguru import logger

from config import settings
from models import QueryResult, RetrievedChunk

_ENCODER     = tiktoken.get_encoding("cl100k_base")
MAX_CTX_TOKS = 2500   # max tokens per source in the context block

SYSTEM_PROMPT = """\
You are a financial analyst with access to official SEC 10-K filings.

RULES — follow them exactly:
1. Answer ONLY from the CONTEXT provided below. Do not use any external knowledge.
2. Cite every factual claim using [N] where N is the source number.
3. Use precise numbers and dates exactly as they appear in the context.
4. If the answer is not in the context, respond with:
   "The information requested is not available in the provided documents."
5. Do not speculate or extrapolate beyond what the documents state."""


def _truncate(text: str, max_tokens: int = MAX_CTX_TOKS) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[... truncated ...]"


def _build_context(retrieved: List[RetrievedChunk]) -> tuple[str, List[dict]]:
    """
    Build a numbered context string and a parallel citations list.
    Deduplicates by (ticker, fiscal_year, section_name) so the same
    section isn't shown twice.
    """
    seen      : set      = set()
    ctx_parts : List[str] = []
    citations : List[dict] = []
    idx = 1

    for rc in retrieved:
        key = (rc.chunk.ticker, rc.chunk.fiscal_year, rc.chunk.section_name)
        if key in seen:
            continue
        seen.add(key)

        header = (
            f"[{idx}] {rc.chunk.company} ({rc.chunk.ticker}) | "
            f"FY{rc.chunk.fiscal_year} | {rc.chunk.section_name}"
        )
        body = _truncate(rc.parent_text or rc.chunk.text)

        ctx_parts.append(f"{header}\n{'─'*60}\n{body}")
        citations.append({
            "index":       idx,
            "company":     rc.chunk.company,
            "ticker":      rc.chunk.ticker,
            "fiscal_year": rc.chunk.fiscal_year,
            "section":     rc.chunk.section_name,
            "chunk_type":  rc.chunk.chunk_type,
            "score":       round(rc.score, 4),
        })
        idx += 1

    context_block = "\n\n".join(ctx_parts)
    return context_block, citations


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.grok_api)


def generate_answer(
    query:      str,
    retrieved:  List[RetrievedChunk],
    query_type: str = "single_doc",
) -> QueryResult:
    """
    Generate a grounded, cited answer from retrieved chunks.
    Returns a QueryResult with answer text, citations, and chunks used.
    """
    if not retrieved:
        return QueryResult(
            query=query,
            answer="No relevant information was found in the available documents for this query.",
            citations=[],
            chunks_used=[],
            query_type=query_type,
        )

    # Hallucination guard — if best score is very low, refuse
    best_score = max(rc.score for rc in retrieved)
    if best_score < 0.05:
        return QueryResult(
            query=query,
            answer="The retrieved context has very low relevance to your query. "
                   "The information may not be available in the ingested documents.",
            citations=[],
            chunks_used=retrieved,
            query_type=query_type,
        )

    context_block, citations = _build_context(retrieved)

    user_message = (
        f"CONTEXT:\n{'═'*60}\n{context_block}\n{'═'*60}\n\n"
        f"QUESTION: {query}"
    )

    logger.debug(f"Calling Groq ({settings.generation_model}) for: '{query[:60]}'")

    response = _get_client().chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()

    logger.debug(f"Answer ({len(answer)} chars), {len(citations)} citations")

    return QueryResult(
        query=query,
        answer=answer,
        citations=citations,
        chunks_used=retrieved,
        query_type=query_type,
    )
