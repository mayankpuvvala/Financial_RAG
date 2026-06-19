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
VALID_YEARS   = {2022, 2023, 2024}

VALID_QUERY_TYPES = {"single_doc", "multi_doc", "temporal", "out_of_scope"}

SYSTEM_PROMPT = """\
You are a query classifier for a financial document RAG system.

Available documents: 10-K annual filings for these companies:
  Technology    : AAPL, MSFT, GOOGL, AMZN
  Banking       : JPM, WFC, BAC, GS
  Asset Mgmt    : BLK, STT, TROW, IVZ
Fiscal years covered: 2022, 2023, 2024

Classify the user query into exactly one type:
  single_doc   — asks about ONE company in ONE specific year
  multi_doc    — compares companies, asks about a sector, or involves multiple firms
  temporal     — asks about a trend or change across MULTIPLE years for one company
  out_of_scope — topic outside these documents (macro, non-financial, etc.)

Also extract any tickers and fiscal years mentioned or clearly implied.
Only include tickers from the available list and years from 2022-2024.
If the user says "last year" or "recent" without a year, include all three years.
If no specific company is mentioned, return an empty tickers list.

Respond with ONLY valid JSON — no markdown, no extra text:
{
  "query_type": "<type>",
  "tickers": ["TICKER", ...],
  "years": [2023, ...],
  "reasoning": "<one short sentence>"
}"""


class ClassifiedQuery(BaseModel):
    query_type: str
    tickers:    List[str]
    years:      List[int]
    reasoning:  str


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.grok_api)


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

        result = ClassifiedQuery(
            query_type=query_type,
            tickers=tickers,
            years=years,
            reasoning=data.get("reasoning", ""),
        )
        logger.debug(f"Classified '{query[:60]}…' → {result.query_type} | {result.tickers} | {result.years}")
        return result

    except Exception as exc:
        logger.warning(f"Classification failed ({exc}), defaulting to single_doc / no filters")
        return ClassifiedQuery(
            query_type="single_doc",
            tickers=[],
            years=[],
            reasoning="classification error — using fallback",
        )
