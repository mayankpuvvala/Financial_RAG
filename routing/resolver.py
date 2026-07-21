"""
classify_and_ensure() = classify_query() + on-demand auto-ingest of any
company the classifier flagged as outside the bundled 12.

This is the entry point CLI/API callers should use instead of
classify_query() directly, so "ask about any SEC-listed company" works
everywhere a query first gets classified.
"""

from loguru import logger

from routing.classifier import classify_query, ClassifiedQuery
from ingestion.registry import resolve_company
from ingestion.auto_ingest import ensure_ticker_indexed


def classify_and_ensure(query: str) -> ClassifiedQuery:
    classification = classify_query(query)

    for mention in classification.unresolved:
        info = resolve_company(mention)
        if not info:
            logger.info(f"Auto-ingest: could not resolve company mention {mention!r}")
            continue

        result = ensure_ticker_indexed(info["ticker"], info["title"] or info["ticker"])
        if not result:
            continue

        _, year = result
        if info["ticker"] not in classification.tickers:
            classification.tickers.append(info["ticker"])
        if year not in classification.years:
            classification.years.append(year)

    if classification.tickers and classification.query_type == "out_of_scope":
        classification.query_type = (
            "multi_doc" if len(classification.tickers) > 1 else "single_doc"
        )

    return classification
