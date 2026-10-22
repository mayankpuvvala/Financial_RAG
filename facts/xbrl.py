"""
Structured XBRL facts from SEC's companyfacts API.

The 10-K text/tables are what the RAG retrieves, but reading a number out of
a noisy markdown table is where the LLM goes wrong (wrong column, wrong
year, wrong scale). The same filing's iXBRL tags are already published as
clean machine-readable facts, so for headline metrics this module answers
deterministically — and gives the verifier something authoritative to check
the LLM's answer against.
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional

import requests
from loguru import logger

from config import settings
from ingestion.registry import cik_for_ticker

_URL       = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_CACHE_DIR = settings.data_dir / "xbrl"
_MAX_AGE   = 24 * 3600

# metric -> concepts to try, "duration" | "instant", unit, and how to choose
# between concepts. For revenue several concepts exist depending on filer
# (banks use Revenues, product companies use RevenueFromContract...); when
# more than one has a value for the period the largest wins, since the
# total is >= any component.
METRICS: Dict[str, dict] = {
    "revenue": {
        "concepts": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "RevenuesNetOfInterestExpense",
                     "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
        "kind": "duration", "unit": "USD", "pick": "max",
    },
    "net_income":       {"concepts": ["NetIncomeLoss", "ProfitLoss"], "kind": "duration", "unit": "USD", "pick": "first"},
    "operating_income": {"concepts": ["OperatingIncomeLoss"], "kind": "duration", "unit": "USD", "pick": "first"},
    "rd_expense":       {"concepts": ["ResearchAndDevelopmentExpense",
                                      "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
                         "kind": "duration", "unit": "USD", "pick": "first"},
    "gross_profit":     {"concepts": ["GrossProfit"], "kind": "duration", "unit": "USD", "pick": "first"},
    "operating_cash_flow": {"concepts": ["NetCashProvidedByUsedInOperatingActivities"],
                            "kind": "duration", "unit": "USD", "pick": "first"},
    "capex":            {"concepts": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                         "kind": "duration", "unit": "USD", "pick": "first"},
    "eps_diluted":      {"concepts": ["EarningsPerShareDiluted"], "kind": "duration",
                         "unit": "USD/shares", "pick": "first"},
    "total_assets":     {"concepts": ["Assets"], "kind": "instant", "unit": "USD", "pick": "first"},
    "total_liabilities": {"concepts": ["Liabilities"], "kind": "instant", "unit": "USD", "pick": "first"},
    "cash":             {"concepts": ["CashAndCashEquivalentsAtCarryingValue",
                                      "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                         "kind": "instant", "unit": "USD", "pick": "first"},
}

# The classifier's `focus` -> the metric this module can answer for it.
FOCUS_TO_METRIC = {
    "revenue":          "revenue",
    "rd_expense":       "rd_expense",
    "net_income":       "net_income",
    "operating_income": "operating_income",
}

_ALL_CONCEPTS = sorted({c for m in METRICS.values() for c in m["concepts"]})


@dataclass
class Figure:
    ticker:       str
    metric:       str
    fiscal_year:  int
    value:        float          # raw units (dollars, or dollars/share for EPS)
    unit:         str
    concept:      str
    period_start: Optional[str]
    period_end:   str
    form:         str
    accession:    str
    filed:        str

    @property
    def value_millions(self) -> float:
        return self.value / 1e6

    @property
    def source_url(self) -> str:
        cik = (cik_for_ticker(self.ticker) or "").lstrip("0")
        return (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                f"{self.accession.replace('-', '')}/") if cik else ""

    def human(self) -> str:
        if self.unit == "USD/shares":
            return f"${self.value:,.2f} per share"
        m = self.value / 1e6
        if abs(m) >= 1000:
            return f"${m:,.0f} million (~${m / 1000:,.1f} billion)"
        return f"${m:,.0f} million"

    def to_dict(self) -> dict:
        return {**asdict(self), "human": self.human(), "source_url": self.source_url}


def fiscal_year_of(end: str) -> int:
    """52/53-week filers can end in the first days of January for the prior year."""
    d = date.fromisoformat(end)
    return d.year - 1 if (d.month == 1 and d.day <= 7) else d.year


def _trim(raw: dict) -> dict:
    """Keep only the annual 10-K facts for the concepts we use."""
    out: Dict[str, List[dict]] = {}
    gaap = raw.get("facts", {}).get("us-gaap", {})
    for concept in _ALL_CONCEPTS:
        rows = []
        for unit, facts in gaap.get(concept, {}).get("units", {}).items():
            for f in facts:
                if f.get("form") not in ("10-K", "10-K/A") or f.get("fp") != "FY":
                    continue
                rows.append({"u": unit, "s": f.get("start"), "e": f["end"], "v": f["val"],
                             "a": f.get("accn", ""), "d": f.get("filed", ""), "f": f["form"]})
        if rows:
            out[concept] = rows
    return out


def _load(ticker: str) -> Optional[Dict[str, List[dict]]]:
    ticker = ticker.upper()
    path = _CACHE_DIR / f"{ticker}.json"
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if cached is not None and time.time() - path.stat().st_mtime < _MAX_AGE:
            return cached

    cik = cik_for_ticker(ticker)
    if not cik:
        return cached
    try:
        resp = requests.get(
            _URL.format(cik=cik),
            headers={"User-Agent": f"FinancialRAG {settings.edgar_email}"},
            timeout=60,
        )
        resp.raise_for_status()
        trimmed = _trim(resp.json())
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trimmed), encoding="utf-8")
        return trimmed
    except Exception as exc:
        logger.warning(f"XBRL companyfacts fetch failed for {ticker}: {exc}")
        return cached   # stale beats nothing


def _candidates(rows: List[dict], spec: dict, fiscal_year: int) -> List[dict]:
    best: Dict[str, dict] = {}
    for r in rows:
        if r["u"] != spec["unit"] or fiscal_year_of(r["e"]) != fiscal_year:
            continue
        if spec["kind"] == "duration":
            if not r["s"]:
                continue
            days = (date.fromisoformat(r["e"]) - date.fromisoformat(r["s"])).days
            if not 350 <= days <= 380:
                continue
        elif r["s"]:
            continue
        # The same period is re-reported as the prior-year comparative in later
        # 10-Ks, sometimes recast (BAC's FY2024 net income: 27,132 as filed,
        # 26,973 after a later accounting change). The filing indexed for
        # fiscal year N is that year's own 10-K, so the earliest report is the
        # one the cited document actually states.
        key = r["e"]
        if key not in best or r["d"] < best[key]["d"]:
            best[key] = r
    return list(best.values())


def get_figure(ticker: str, metric: str, fiscal_year: int) -> Optional[Figure]:
    """The reported annual value of `metric` for `ticker`'s fiscal year, or None."""
    spec = METRICS.get(metric)
    if not spec:
        return None
    data = _load(ticker)
    if not data:
        return None

    found: List[tuple] = []   # (concept, row)
    for concept in spec["concepts"]:
        cands = _candidates(data.get(concept, []), spec, fiscal_year)
        if cands:
            found.append((concept, max(cands, key=lambda r: r["e"])))
            if spec["pick"] == "first":
                break
    if not found:
        return None
    concept, r = max(found, key=lambda x: x[1]["v"]) if spec["pick"] == "max" else found[0]
    return Figure(
        ticker=ticker.upper(), metric=metric, fiscal_year=fiscal_year, value=float(r["v"]),
        unit=spec["unit"], concept=concept, period_start=r["s"], period_end=r["e"],
        form=r["f"], accession=r["a"], filed=r["d"],
    )


def available_years(ticker: str, metric: str = "revenue") -> List[int]:
    spec = METRICS[metric]
    data = _load(ticker) or {}
    years = set()
    for concept in spec["concepts"]:
        for r in data.get(concept, []):
            fy = fiscal_year_of(r["e"])
            if _candidates([r], spec, fy):
                years.add(fy)
    return sorted(years)


METRIC_LABEL = {
    "revenue": "total revenue (net sales)", "net_income": "net income",
    "operating_income": "operating income", "rd_expense": "research and development expense",
    "gross_profit": "gross profit", "operating_cash_flow": "net cash provided by operating activities",
    "capex": "capital expenditures (purchases of property, plant and equipment)",
    "eps_diluted": "diluted earnings per share", "total_assets": "total assets",
    "total_liabilities": "total liabilities", "cash": "cash and cash equivalents",
}


def as_source(fig: Figure, company: Optional[str] = None):
    """Wrap a Figure as a RetrievedChunk so it flows through the normal
    context-building and citation machinery like any retrieved passage."""
    from config import TICKER_TO_COMPANY
    from models import Chunk, RetrievedChunk

    company = company or TICKER_TO_COMPANY.get(fig.ticker, {}).get("name", fig.ticker)
    period = f"{fig.period_start} to {fig.period_end}" if fig.period_start else f"as of {fig.period_end}"
    text = (
        f"{company} ({fig.ticker}) reported {METRIC_LABEL.get(fig.metric, fig.metric)} of "
        f"{fig.human()} for fiscal year {fig.fiscal_year} ({period}). "
        f"Source: SEC XBRL tag us-gaap:{fig.concept} in its {fig.form} (accession {fig.accession}). "
        f"Raw reported value: {fig.value:,.0f} {fig.unit}."
    )
    chunk = Chunk(
        chunk_id=f"xbrl:{fig.ticker}:{fig.fiscal_year}:{fig.metric}", parent_id="xbrl",
        doc_id=f"xbrl:{fig.ticker}:{fig.fiscal_year}", text=text, company=company,
        ticker=fig.ticker, filing_type=fig.form, fiscal_year=fig.fiscal_year,
        section_name=f"SEC XBRL structured data: {METRIC_LABEL.get(fig.metric, fig.metric)}",
        chunk_type="xbrl", token_count=len(text) // 4,
    )
    return RetrievedChunk(chunk=chunk, score=100.0, parent_text=text)
