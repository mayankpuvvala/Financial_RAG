"""
Query classifier — routes each query to a pipeline type using Groq llama-3.1-8b.

Output types:
  single_doc   — one company, one year
  multi_doc    — multiple companies or sector-level comparison
  temporal     — trend over multiple years, one company
  out_of_scope — cannot be answered from available documents
"""

import json
import re
from functools import lru_cache
from typing import List

from groq import Groq
from loguru import logger
from pydantic import BaseModel

from config import settings, COMPANIES

VALID_TICKERS = {c["ticker"] for c in COMPANIES}
VALID_YEARS   = {2023, 2024, 2025}

VALID_QUERY_TYPES = {"single_doc", "multi_doc", "temporal", "out_of_scope"}

SYSTEM_PROMPT = """\
You are a query classifier for a financial document RAG system.

Documents already indexed and ready: 10-K annual filings for these companies:
  Technology    : AAPL, MSFT, GOOGL, AMZN
  Banking       : JPM, WFC, BAC, GS
  Asset Mgmt    : BLK, STT, TROW, IVZ
Fiscal years covered: 2023, 2024, 2025

The system is NOT limited to that list — it can fetch and index the latest
10-K for ANY publicly traded US company on demand. So never treat a company
outside the list above as out_of_scope just because it's unfamiliar; extract
it into "other_companies" instead (see below) so it can be looked up.

Classify the user query into exactly one type:
  single_doc   — asks about ONE company in ONE specific year — this includes
                 ANY question a 10-K would answer: financial metrics, but also
                 qualitative ones like business description, products sold,
                 segments, risk factors, or strategy. "What does Apple sell?"
                 is single_doc, NOT out_of_scope — 10-Ks include a full
                 business description (Item 1) precisely for questions like this.
  multi_doc    — compares companies, asks about a sector, or involves multiple firms
  temporal     — asks about a trend or change across MULTIPLE years for one company
  out_of_scope — ONLY for topics no 10-K could ever answer: macro/market
                 trivia (stock prices, crypto), general knowledge unrelated to
                 any company, or the query mentions no identifiable company at
                 all. If a real company is named or implied, it is NOT out_of_scope.

Also extract every company mentioned or clearly implied, as either a ticker
symbol or company name:
  - "tickers": tickers FROM THE INDEXED LIST ABOVE ONLY.
  - "other_companies": any other company/ticker mentioned that is NOT in the
    indexed list above (e.g. "Netflix", "NFLX", "Tesla") — pass through
    whatever the user wrote, ticker or name, don't normalize it.

Also extract fiscal years mentioned or clearly implied, from 2023-2025.
If the user says "last year" or "recent" without a year, include all three years.
If no specific company is mentioned, return empty lists for both company fields.

Respond with ONLY valid JSON — no markdown, no extra text:
{
  "query_type": "<type>",
  "tickers": ["TICKER", ...],
  "other_companies": ["Netflix", ...],
  "years": [2023, ...],
  "reasoning": "<one short sentence>"
}"""


class ClassifiedQuery(BaseModel):
    query_type: str
    tickers:    List[str]
    years:      List[int]
    reasoning:  str
    unresolved: List[str] = []   # company/ticker mentions outside the bundled 12


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api)


def classify_query(query: str) -> ClassifiedQuery:
    """Classify a user query and extract target tickers / years."""
    try:
        response = _get_client().chat.completions.create(
            model=settings.routing_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps the JSON
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        data = json.loads(raw)

        query_type = data.get("query_type", "single_doc")
        if query_type not in VALID_QUERY_TYPES:
            query_type = "single_doc"

        tickers = [t.upper() for t in data.get("tickers", []) if t.upper() in VALID_TICKERS]
        years   = [int(y) for y in data.get("years", [])   if int(y) in VALID_YEARS]

        # Anything the model tagged as a ticker but that isn't in our bundled
        # list is also an unresolved mention (models sometimes put it there
        # despite instructions), in addition to the dedicated field.
        raw_tickers    = [t for t in data.get("tickers", []) if t.upper() not in VALID_TICKERS]
        other_mentions = data.get("other_companies", [])
        unresolved = [m.strip() for m in (raw_tickers + other_mentions) if m and m.strip()]

        result = ClassifiedQuery(
            query_type=query_type,
            tickers=tickers,
            years=years,
            reasoning=data.get("reasoning", ""),
            unresolved=unresolved,
        )
        logger.debug(
            f"Classified '{query[:60]}…' → {result.query_type} | {result.tickers} | "
            f"{result.years} | unresolved={result.unresolved}"
        )
        return result

    except Exception as exc:
        logger.warning(f"Classification failed ({exc}), defaulting to single_doc / no filters")
        return ClassifiedQuery(
            query_type="single_doc",
            tickers=[],
            years=[],
            reasoning="classification error — using fallback",
        )
