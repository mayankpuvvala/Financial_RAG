"""
Synthesizer — handles multi_doc and temporal queries.

Flow:
  1. Decompose query into atomic sub-questions
  2. Retrieve + generate a sub-answer for each sub-question (sequentially —
     local Qdrant embedded mode does not support concurrent file access)
  3. Combine all sub-answers with a synthesis prompt
  4. Return a single QueryResult with merged citations
"""

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


def _answer_sub_question(sub: Dict, top_k: int) -> tuple[Dict, QueryResult]:
    retrieved = retrieve(
        query=sub["question"],
        tickers=[sub["ticker"]],
        years=[sub["year"]],
        top_k=top_k,
    )
    result = generate_answer(
        query=sub["question"],
        retrieved=retrieved,
        query_type="sub_question",
    )
    return sub, result


def synthesize(
    query:      str,
    tickers:    List[str],
    years:      List[int],
    query_type: str = "multi_doc",
    top_k:      int = settings.rerank_top_k,
) -> QueryResult:
    """
    Full multi-document synthesis pipeline — sequential execution.
    Local Qdrant embedded mode uses SQLite which rejects concurrent openers,
    so sub-questions run one at a time.
    """
    sub_questions = decompose_query(query, tickers, years)
    logger.info(f"Synthesizing {len(sub_questions)} sub-questions for: '{query[:60]}'")

    sub_results: List[tuple[Dict, QueryResult]] = []

    for sub in sub_questions:
        try:
            result = _answer_sub_question(sub, top_k)
            sub_results.append(result)
            logger.debug(f"  ✓ {sub['ticker']} FY{sub['year']}: {sub['question'][:50]}")
        except Exception as exc:
            logger.error(f"  ✗ Sub-question failed ({sub['ticker']} FY{sub['year']}): {exc}")

    # Build the synthesis input
    combined_parts = []
    all_citations:  List[dict] = []
    all_chunks:     List[RetrievedChunk] = []
    citation_offset = 0

    for sub, result in sub_results:
        if result.answer.startswith("No relevant"):
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

    synthesis_response = _get_client().chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": synthesis_input},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    return QueryResult(
        query=query,
        answer=synthesis_response.choices[0].message.content.strip(),
        citations=all_citations,
        chunks_used=all_chunks,
        query_type=query_type,
    )
