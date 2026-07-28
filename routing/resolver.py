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
from ingestion.auto_ingest import ensure_ticker_indexed, YearNotAvailable

# classify_query() always fills in SOME year per its output schema, even
# for questions that never named one ("What are Nike's segments?" still
# came back with a year — observed defaulting to the earliest valid one).
# For these qualitative focuses the year is incidental, not requested, so
# treating it as a hard requirement rejects an otherwise-fully-answerable
# query just because the classifier's filler year isn't the one that
# happened to get indexed. Only focuses where a specific year is actually
# load-bearing (a financial figure that differs release to release) should
# ever turn into a hard target_year.
_YEAR_SENSITIVE_FOCUSES = {"revenue", "rd_expense", "net_income", "operating_income", "balance_sheet"}


def classify_and_ensure(query: str) -> ClassifiedQuery:
    classification = classify_query(query)

    for mention in classification.unresolved:
        info = resolve_company(mention)
        if not info:
            logger.info(f"Auto-ingest: could not resolve company mention {mention!r}")
            classification.failed_lookups.append(mention)
            continue

        # If the query named a year AND it actually matters for this kind
        # of question, try to fetch THAT year specifically rather than
        # whatever's most recent — a query for FY2024 revenue on a
        # brand-new company shouldn't silently get FY2026 instead.
        # Excludes `temporal`: it inherently wants a trend across whichever
        # recent years are available, not exactly years[0] or nothing —
        # hard-requiring that one would reject an otherwise-fully-
        # answerable trend query just because the auto-ingest fetch
        # window (recent, not tied to any specific year) didn't happen to
        # include it.
        target_year = (
            classification.years[0]
            if (
                classification.query_type != "temporal"
                and classification.years
                and classification.focus in _YEAR_SENSITIVE_FOCUSES
            )
            else None
        )
        try:
            result = ensure_ticker_indexed(
                info["ticker"], info["title"] or info["ticker"], target_year=target_year
            )
        except YearNotAvailable as exc:
            classification.year_not_available.append(
                f"{exc.ticker} (asked for FY{exc.requested}, have FY"
                + "/FY".join(map(str, exc.available)) + ")"
            )
            continue

        if not result:
            # Resolved to a real SEC filer (info is not None) but the
            # download/parse/embed pipeline itself failed — a technical
            # hiccup, not "no such company." Tracked separately from
            # failed_lookups so the caller doesn't tell the user a real,
            # obviously-public company (e.g. a transient EDGAR download
            # blip on ExxonMobil) "isn't a public company."
            classification.ingest_failed.append(info["ticker"])
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
