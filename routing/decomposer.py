"""
Sub-question decomposer for multi_doc and temporal queries.

Takes a complex query and breaks it into atomic sub-questions,
each targeting a specific (ticker, year) pair that can be answered
from a single collection.
"""

import json
import re
from typing import List, Dict

from groq import Groq
from loguru import logger

from config import settings

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
  "R&D"       → "research and development expenses"
  "CapEx"     → "capital expenditures"
  "revenue"   → "total net sales" or "total revenue"
  "earnings"  → "net income"
  "margins"   → "gross margin" or "operating margin"
  "SG&A"      → "selling general and administrative expenses"

Rules:
- One sub-question per company per year.
- Use the full company name in the question, not just the ticker.
- Write questions as if searching a document — use words the filing would contain.

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
        client   = Groq(api_key=settings.groq_api)
        response = client.chat.completions.create(
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
