"""
Sub-question decomposer for multi_doc and temporal queries.

Takes a complex query and breaks it into atomic sub-questions,
each targeting a specific (ticker, year) pair that can be answered
from a single collection.
"""

import json
import re
from functools import lru_cache
from typing import List, Dict

from groq import Groq
from loguru import logger

from config import settings


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api)

SYSTEM_PROMPT = """\
You decompose complex financial queries into atomic sub-questions for searching SEC 10-K filings.

Each sub-question must:
- Ask about ONE metric / fact
- Target ONE company (ticker)
- Target ONE fiscal year
- Use the EXACT financial terminology found in SEC filings (not abbreviations)

Available tickers : AAPL, MSFT, GOOGL, AMZN, JPM, WFC, BAC, GS, BLK, STT, TROW, IVZ
Available years   : 2023, 2024, 2025

SEC filing terminology — always use the long form, never abbreviations:
  "R&D"              → "research and development expenses"
  "CapEx"            → "capital expenditures"
  "revenue"          → "total net sales" or "total revenue"
  "earnings"         → "net income"
  "margins"          → "gross margin" or "operating margin"
  "SG&A"             → "selling general and administrative expenses"
  "operating income" → "income from operations" or "operating income"

Rules:
- One sub-question per company per year.
- Use the full company name in the question, not just the ticker.
- Write questions as if searching a document — use words the filing would contain.

SPECIAL CASE — open-ended comparison with NO specific financial metric:
  Applies ONLY when the query is a general "how is X different from Y?", "compare X and Y",
  or "what does X do vs Y?" AND the query does NOT mention any specific metric such as:
  revenue, income, profit, earnings, R&D, expenses, margin, cash flow, operating, sales, assets.
  In that case, generate ONE sub-question per company asking for a business overview:
    "Describe <Company>'s core business, primary products or services, revenue sources,
     operating segments, and key financial performance"
  Use the most recent year in the years list.
  If the query DOES mention a specific metric, ignore this SPECIAL CASE and decompose normally.

Respond with ONLY valid JSON — no markdown, no extra text:
{
  "sub_questions": [
    {"question": "What were Apple total net sales in fiscal year 2024?", "ticker": "AAPL", "year": 2024},
    {"question": "What were Microsoft total net sales in fiscal year 2024?", "ticker": "MSFT", "year": 2024}
  ]
}"""


def decompose_query(
    query:      str,
    tickers:    List[str],
    years:      List[int],
) -> List[Dict]:
    """
    Break a complex query into sub-questions.
    Returns list of {"question": str, "ticker": str, "year": int}.
    Falls back to a single entry if decomposition fails.
    """
    try:
        response = _get_client().chat.completions.create(
            model=settings.routing_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Query: {query}\nTickers: {tickers}\nYears: {years}"},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        data = json.loads(raw)
        subs = data.get("sub_questions", [])

        if not subs:
            raise ValueError("Empty sub_questions list")

        logger.debug(f"Decomposed into {len(subs)} sub-questions")
        return subs

    except Exception as exc:
        logger.warning(f"Decomposition failed ({exc}), using original query")
        # Fallback: one sub-question per ticker-year combination
        fallback = []
        for t in (tickers or ["AAPL"]):
            for y in (years or [2024]):
                fallback.append({"question": query, "ticker": t, "year": y})
        return fallback
