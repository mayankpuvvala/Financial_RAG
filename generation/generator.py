"""
Answer generator — Groq (llama-3.3-70b-versatile) + citation builder.

Flow:
  1. Build a numbered context block from retrieved chunks (parent section text)
  2. Call Groq with a strict grounding prompt
  3. Return answer + structured citation list
"""

import time
from functools import lru_cache
from typing import List, Optional

import tiktoken
from groq import Groq, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
from loguru import logger

from config import settings
from models import QueryResult, RetrievedChunk
from retrieval.reranker import _compress_xbrl

_ENCODER     = tiktoken.get_encoding("cl100k_base")
MAX_CTX_TOKS = 1500   # per source — Groq free-tier 100K TPD budget; raised from 600
                      # since large sections (Notes, MD&A) were getting chopped
                      # before reaching the relevant table/paragraph. Re-lower if
                      # daily token usage becomes a problem under real query volume.

SYSTEM_PROMPT = """\
You are a financial analyst with access to official SEC 10-K filings.

RULES:
1. Answer ONLY from the CONTEXT provided below. Do not use any external knowledge.
2. Cite every factual claim using [N] where N is the source number.
3. Use precise numbers from the context. Convert raw numbers to readable form
   (e.g. 29510 → $29,510 million = $29.5 billion).
4. If the answer genuinely cannot be found, say so explicitly.
5. Do not speculate beyond what the documents state.

READING XBRL FINANCIAL TABLES:
SEC filings use XBRL formatting that creates noisy markdown tables:
- Column headers (years like 2024, 2023) appear as DATA ROWS, not column names
- Currency symbols "$" appear in their own cell immediately LEFT of the numeric value
- Values are often duplicated across adjacent columns — read each unique number once
- All numbers are in MILLIONS of dollars unless the table header says otherwise
- If year header rows are not visible in the chunk, treat the FIRST numeric column
  as the fiscal year stated in the source header (e.g. "FY2024"), with subsequent
  columns being prior years in descending order (FY2023, FY2022, …)

To extract a value: find the row with the metric name, locate the column under
the year heading in the header row, then read the numeric value in that column.

Example from a filing:
  Header row:  | | 2024 | 2024 | | | 2023 | 2023 |
  Data row:    | Research and development | | $ | 29510 | | | $ | 27195 | |
  → FY2024 R&D = $29,510 million.  FY2023 R&D = $27,195 million."""


def _truncate(text: str, max_tokens: int = MAX_CTX_TOKS) -> str:
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[... truncated ...]"


def _build_context(retrieved: List[RetrievedChunk]) -> tuple[str, List[dict]]:
    """
    Build a numbered context string and a parallel citations list.

    Groups multiple chunks from the same section so all retrieved passages
    for that section are visible to the model, then appends parent context
    with remaining token budget.
    """
    from collections import defaultdict

    # Group by section, preserving first-seen order
    section_chunks: dict = defaultdict(list)
    order: List[tuple] = []
    seen_keys: set = set()

    for rc in retrieved:
        key = (rc.chunk.ticker, rc.chunk.fiscal_year, rc.chunk.section_name)
        section_chunks[key].append(rc)
        if key not in seen_keys:
            seen_keys.add(key)
            order.append(key)

    ctx_parts : List[str] = []
    citations : List[dict] = []
    idx = 1

    for key in order:
        group   = section_chunks[key]
        best_rc = group[0]

        header = (
            f"[{idx}] {best_rc.chunk.company} ({best_rc.chunk.ticker}) | "
            f"FY{best_rc.chunk.fiscal_year} | {best_rc.chunk.section_name}"
        )

        # Concatenate all chunk texts from this section (each chunk was
        # deemed relevant — keeping all avoids the problem of the "right"
        # passage being in a deduped-out later chunk).
        # Apply XBRL compression to remove repeated cells / empty columns so
        # key metric rows (like "Research and development | $ | 29510") are
        # not pushed past the truncation point by noisy XBRL preamble.
        all_chunks_text = "\n\n".join(_compress_xbrl(rc.chunk.text) for rc in group)
        all_toks        = len(_ENCODER.encode(all_chunks_text))
        remaining       = MAX_CTX_TOKS - all_toks - 30

        if best_rc.parent_text and remaining > 300:
            parent_excerpt = _truncate(best_rc.parent_text, remaining)
            body = f"{all_chunks_text}\n\n--- section context ---\n{parent_excerpt}"
        else:
            body = _truncate(all_chunks_text)

        ctx_parts.append(f"{header}\n{'─'*60}\n{body}")
        citations.append({
            "index":       idx,
            "company":     best_rc.chunk.company,
            "ticker":      best_rc.chunk.ticker,
            "fiscal_year": best_rc.chunk.fiscal_year,
            "section":     best_rc.chunk.section_name,
            "chunk_type":  best_rc.chunk.chunk_type,
            "score":       round(best_rc.score, 4),
        })
        idx += 1

    return "\n\n".join(ctx_parts), citations


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api)


def _call_generation(user_message: str, retries: int = 2, backoff: float = 1.5) -> str:
    """
    Call Groq for the final answer, retrying transient errors (connection
    blips, 5xx) a couple of times with short backoff. Rate-limit errors are
    NOT retried — the daily/per-minute quota won't clear in seconds, so this
    fails fast and lets the caller degrade gracefully instead of stalling.
    """
    last_exc: Exception = None
    for attempt in range(retries + 1):
        try:
            response = _get_client().chat.completions.create(
                model=settings.generation_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            raise
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(f"Groq call failed (attempt {attempt + 1}/{retries + 1}): {exc}")
                time.sleep(backoff * (attempt + 1))
                continue
            raise
    raise last_exc


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

    context_block, citations = _build_context(retrieved)

    user_message = (
        f"CONTEXT:\n{'═'*60}\n{context_block}\n{'═'*60}\n\n"
        f"QUESTION: {query}"
    )

    logger.debug(f"Calling Groq ({settings.generation_model}) for: '{query[:60]}'")

    try:
        answer = _call_generation(user_message)
    except RateLimitError as exc:
        logger.warning(f"Groq rate limit hit during generation: {exc}")
        return QueryResult(
            query=query,
            answer=(
                "I found relevant source material below, but the answer-generation "
                "service has hit its usage limit and can't write a summary right now. "
                "Please try again in a few minutes."
            ),
            citations=citations,
            chunks_used=retrieved,
            query_type=query_type,
        )
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        logger.error(f"Groq generation call failed after retries: {exc}")
        return QueryResult(
            query=query,
            answer=(
                "I found relevant source material below, but the answer-generation "
                "service is temporarily unavailable. Please try again shortly."
            ),
            citations=citations,
            chunks_used=retrieved,
            query_type=query_type,
        )

    logger.debug(f"Answer ({len(answer)} chars), {len(citations)} citations")

    return QueryResult(
        query=query,
        answer=answer,
        citations=citations,
        chunks_used=retrieved,
        query_type=query_type,
    )
