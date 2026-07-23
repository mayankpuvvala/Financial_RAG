"""
Query entry point — routes a question through the full pipeline.

    python query.py "What was Apple's revenue in FY2024?"
    python query.py "Compare Microsoft and Google R&D spend in 2024"
    python query.py "How did Amazon's operating margin trend from 2023 to 2025?"
"""

import re
import sys
from loguru import logger

from config import settings
from routing.resolver import classify_and_ensure
from retrieval.retriever import retrieve
from generation.generator import generate_answer
from generation.synthesizer import synthesize
from models import QueryResult

# Catches the model's own "not found" phrasing so a refused single_doc
# answer can trigger one broadened retry instead of being accepted as final.
# Deliberately over-inclusive (false positives just cost one extra, usually-
# redundant retry; false negatives silently ship a wrong "not found").
_REFUSAL_PATTERN = re.compile(
    r"cannot be found|cannot be determined|can.t be found|"
    r"does not (?:explicitly )?(?:mention|state|provide|discuss|contain)|"
    r"not explicitly (?:mentioned|stated)|no relevant information|"
    r"is not (?:mentioned|provided|available|explicitly stated) in the",
    re.IGNORECASE,
)


def _is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(answer))


def ask(query: str) -> QueryResult:
    """Run a query through the full RAG pipeline and return a QueryResult."""

    # 1 — Classify (auto-ingesting any company outside the bundled 12)
    classification = classify_and_ensure(query)
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
            focus=classification.focus,
        )
    else:
        # single_doc or summarization — standard retrieval.
        # "single_doc" queries can still span multiple years (e.g. "most
        # recent" expands to all 3 years, or the user lists several years
        # explicitly without phrasing it as a trend). top_k must scale with
        # that, or years/tickers beyond the first few get starved out by the
        # fixed default — capped at 5x to bound context size/cost.
        n_targets = max(1, len(classification.tickers)) * max(1, len(classification.years))
        top_k = settings.rerank_top_k * min(n_targets, 5)
        retrieved = retrieve(
            query=query,
            tickers=classification.tickers,
            years=classification.years,
            top_k=top_k,
            focus=classification.focus,
        )
        result = generate_answer(
            query=query,
            retrieved=retrieved,
            query_type=classification.query_type,
        )

        # One bounded retry: a narrow focus boost can mis-target the wrong
        # section, or top_k can just be too tight — before accepting "not
        # found" as final, retry once with the focus restriction dropped and
        # top_k widened. Capped at a single retry so a genuinely
        # out-of-scope query still fails fast rather than doubling Groq
        # cost on every miss.
        if _is_refusal(result.answer):
            logger.info(f"Refusal detected, retrying with broadened search: '{query[:60]}'")
            retried = retrieve(
                query=query,
                tickers=classification.tickers,
                years=classification.years,
                top_k=min(top_k * 2, 15),
                focus="other",
            )
            retried_result = generate_answer(
                query=query,
                retrieved=retried,
                query_type=classification.query_type,
            )
            if not _is_refusal(retried_result.answer):
                result = retried_result

    return result


def _print_result(result: QueryResult) -> None:
    print("\n" + "=" * 60)
    print(f"QUERY: {result.query}")
    print("=" * 60)
    print(f"\n{result.answer}\n")

    if result.citations:
        print("-" * 60)
        print("SOURCES:")
        for c in result.citations:
            print(
                f"  [{c['index']}] {c['company']} ({c['ticker']}) "
                f"FY{c['fiscal_year']} - {c['section']}"
            )

    print("-" * 60)
    print(f"Query type : {result.query_type}")
    print(f"Chunks used: {len(result.chunks_used)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows (cp1252 terminals crash on LLM Unicode)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if len(sys.argv) < 2:
        print('Usage: python query.py "Your financial question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result   = ask(question)
    _print_result(result)
