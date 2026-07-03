"""
Synthesizer — handles multi_doc and temporal queries.

Flow:
  1. Decompose query into atomic sub-questions
  2. Retrieve for every sub-question sequentially (Qdrant SQLite lock
     prevents concurrent access)
  3. Generate sub-answers in parallel via a thread pool — Groq HTTP calls
     release the GIL so threads genuinely overlap, saving 6-15 s per query
  4. Combine all sub-answers with a final synthesis call
  5. Return a single QueryResult with merged citations
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from groq import Groq
from loguru import logger

from config import settings
from models import QueryResult, RetrievedChunk
from routing.decomposer import decompose_query
from retrieval.retriever import retrieve
from generation.generator import generate_answer, _get_client

SYNTHESIS_SYSTEM = """\
You are a financial analyst synthesizing multiple research findings.

You will receive:
- The original question
- A set of sub-answers, each with their own citations

Your task:
1. Combine the sub-answers into ONE cohesive, well-structured response.
2. Preserve all citation references [N] exactly as they appear.
3. Add a brief summary or conclusion where relevant.
4. Do not add information not present in the sub-answers.
5. Use markdown tables or bullet points for comparisons."""


def synthesize(
    query:      str,
    tickers:    List[str],
    years:      List[int],
    query_type: str = "multi_doc",
    top_k:      int = settings.rerank_top_k,
) -> QueryResult:
    """
    Multi-document synthesis pipeline.

    Retrieval is sequential (Qdrant SQLite lock).
    Generation is parallel (thread pool — Groq I/O releases the GIL).
    """
    sub_questions = decompose_query(query, tickers, years)
    logger.info(f"Synthesizing {len(sub_questions)} sub-questions for: '{query[:60]}'")

    # ── Phase 1: sequential retrieval ────────────────────────────────────────
    retrieval_data: List[tuple[Dict, list]] = []
    for sub in sub_questions:
        try:
            retrieved = retrieve(
                query=sub["question"],
                tickers=[sub["ticker"]],
                years=[sub["year"]],
                top_k=top_k,
            )
            retrieval_data.append((sub, retrieved))
        except Exception as exc:
            logger.error(f"  ✗ Retrieval failed ({sub['ticker']} FY{sub['year']}): {exc}")

    if not retrieval_data:
        return QueryResult(
            query=query,
            answer="Could not retrieve information for this query.",
            citations=[], chunks_used=[], query_type=query_type,
        )

    # ── Phase 2: parallel generation ─────────────────────────────────────────
    def _generate(item: tuple[Dict, list]) -> tuple[Dict, QueryResult]:
        sub, retrieved = item
        result = generate_answer(
            query=sub["question"],
            retrieved=retrieved,
            query_type="sub_question",
        )
        return sub, result

    sub_results: List[tuple[Dict, QueryResult]] = []
    with ThreadPoolExecutor(max_workers=len(retrieval_data)) as pool:
        futures = {pool.submit(_generate, item): item[0] for item in retrieval_data}
        for future in as_completed(futures):
            sub = futures[future]
            try:
                result = future.result()
                sub_results.append(result)
                logger.debug(f"  ✓ {sub['ticker']} FY{sub['year']}: {sub['question'][:50]}")
            except Exception as exc:
                logger.error(f"  ✗ Generation failed ({sub['ticker']} FY{sub['year']}): {exc}")

    # Build the synthesis input
    combined_parts = []
    all_citations:  List[dict] = []
    all_chunks:     List[RetrievedChunk] = []
    citation_offset = 0

    for sub, result in sub_results:
        if result.answer.startswith("No relevant information was found"):
            continue

        remapped_answer = result.answer
        renumbered_cits = []
        for cit in result.citations:
            new_idx = cit["index"] + citation_offset
            remapped_answer = remapped_answer.replace(
                f"[{cit['index']}]", f"[{new_idx}]"
            )
            renumbered_cits.append({**cit, "index": new_idx})

        citation_offset += len(result.citations)

        combined_parts.append(
            f"### {sub['ticker']} FY{sub['year']}\n"
            f"Sub-question: {sub['question']}\n\n"
            f"{remapped_answer}"
        )
        all_citations.extend(renumbered_cits)
        all_chunks.extend(result.chunks_used)

    if not combined_parts:
        return QueryResult(
            query=query,
            answer="Could not find relevant information for this query across the available documents.",
            citations=[],
            chunks_used=[],
            query_type=query_type,
        )

    synthesis_input = (
        f"ORIGINAL QUESTION: {query}\n\n"
        + "\n\n---\n\n".join(combined_parts)
    )

    logger.debug(f"Running synthesis call for {len(sub_results)} sub-answers")

    try:
        synthesis_response = _get_client().chat.completions.create(
            model=settings.generation_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user",   "content": synthesis_input},
            ],
            temperature=0.1,
            max_tokens=768,
        )
        answer = synthesis_response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"Synthesis call failed: {exc}")
        answer = "\n\n".join(combined_parts)   # fall back to concatenated sub-answers

    return QueryResult(
        query=query,
        answer=answer,
        citations=all_citations,
        chunks_used=all_chunks,
        query_type=query_type,
    )
