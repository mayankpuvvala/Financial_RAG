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

from config import settings, TICKER_TO_COMPANY


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


# Mirrors decompose_query's own SEC-terminology rewriting, keyed off the
# classifier's own `focus` field (routing/classifier.py's VALID_FOCUS)
# instead of asking an LLM to infer the metric from free text.
_FOCUS_TO_METRIC = {
    "revenue":           "total net sales or total revenue",
    "rd_expense":        "research and development expenses",
    "net_income":        "net income",
    "operating_income":  "income from operations",
    "balance_sheet":     "total assets, liabilities, and cash and cash equivalents",
    "segment_info":      "business segment breakdown",
    "business_overview": "core business, products, and segments",
    "risk_factors":      "key risk factors",
    "legal_proceedings": "legal proceedings",
    "cybersecurity":     "cybersecurity risk management",
    "other":             "financial performance",
}


def decompose_temporal(ticker: str, years: List[int], focus: str = "other") -> List[Dict]:
    """
    Deterministic decomposition for temporal queries (one company, a trend
    across years) — no Groq call, no chance of the LLM decomposer picking
    its multi-company "describe the business" special case for a
    single-company query that names a specific metric (observed:
    "how did Boeing's revenue trend recently" got decomposed into that
    boilerplate instead of a real per-year revenue question), and no risk
    of collapsing to fewer sub-questions than years requested. One
    sub-question per year, guaranteed.
    """
    ticker = ticker.upper()
    company_name = TICKER_TO_COMPANY.get(ticker, {}).get("name", ticker)
    metric = _FOCUS_TO_METRIC.get(focus, _FOCUS_TO_METRIC["other"])
    return [
        {"question": f"What was {company_name}'s {metric} in fiscal year {year}?", "ticker": ticker, "year": year}
        for year in sorted(set(years))
    ]
