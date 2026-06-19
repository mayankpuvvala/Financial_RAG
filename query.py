"""
Query entry point — routes a question through the full pipeline.

    python query.py "What was Apple's revenue in FY2024?"
    python query.py "Compare Microsoft and Google R&D spend in 2024"
    python query.py "How did Amazon's operating margin trend from 2023 to 2025?"
"""

import sys
import json
from loguru import logger

from routing.classifier import classify_query
from retrieval.retriever import retrieve
from generation.generator import generate_answer
from generation.synthesizer import synthesize
from models import QueryResult


def ask(query: str, verbose: bool = False) -> QueryResult:
    """Run a query through the full RAG pipeline and return a QueryResult."""

    # 1 — Classify
    classification = classify_query(query)
    logger.info(
        f"Query type : {classification.query_type}\n"
        f"Tickers    : {classification.tickers}\n"
        f"Years      : {classification.years}\n"
        f"Reasoning  : {classification.reasoning}"
    )

    # 2 — Route
    if classification.query_type == "out_of_scope":
        from models import QueryResult
        return QueryResult(
            query=query,
            answer="This query is outside the scope of the available SEC 10-K filings.",
            citations=[],
            chunks_used=[],
            query_type="out_of_scope",
        )

    if classification.query_type in ("multi_doc", "temporal"):
        result = synthesize(
            query=query,
            tickers=classification.tickers,
            years=classification.years,
            query_type=classification.query_type,
        )
    else:
        # single_doc or summarization — standard retrieval
        retrieved = retrieve(
            query=query,
            tickers=classification.tickers,
            years=classification.years,
        )
        result = generate_answer(
            query=query,
            retrieved=retrieved,
            query_type=classification.query_type,
        )

    return result


def _print_result(result: QueryResult) -> None:
    print("\n" + "═" * 60)
    print(f"QUERY: {result.query}")
    print("═" * 60)
    print(f"\n{result.answer}\n")

    if result.citations:
        print("─" * 60)
        print("SOURCES:")
        for c in result.citations:
            print(
                f"  [{c['index']}] {c['company']} ({c['ticker']}) "
                f"FY{c['fiscal_year']} — {c['section']}"
            )

    print("─" * 60)
    print(f"Query type : {result.query_type}")
    print(f"Chunks used: {len(result.chunks_used)}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if len(sys.argv) < 2:
        print('Usage: python query.py "Your financial question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result   = ask(question)
    _print_result(result)
