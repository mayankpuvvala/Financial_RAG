"""
MCP server exposing the Financial RAG as tools any MCP client (Claude Desktop,
Claude Code, another agent) can call.

    python mcp_server.py                     # stdio (default)
    python mcp_server.py --http --port 8765  # streamable HTTP

Claude Desktop / Claude Code config (stdio):

    { "mcpServers": { "financial-rag": {
        "command": "<repo>/venv/Scripts/python.exe",
        "args": ["<repo>/mcp_server.py"] } } }

Tools
  search_filings        hybrid search + rerank over indexed 10-K passages
  fetch_10k             make sure a company's 10-K is downloaded and indexed
  get_xbrl_figure       the exact figure the company reported to the SEC (XBRL)
  ask_financial_question  the full agent: plan, retrieve, answer, verify
  list_indexed_filings  what's searchable right now

stdout carries the MCP protocol on stdio, so everything else logs to stderr.
"""

import argparse
import sys
from typing import List, Optional

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from mcp.server.mcpserver import MCPServer  # noqa: E402

from config import TICKER_TO_COMPANY  # noqa: E402
from facts.xbrl import METRICS, get_figure  # noqa: E402

MAX_PASSAGE_CHARS = 1500


# ── tool implementations (plain functions: easy to test without MCP) ────────

def search_filings_impl(
    query: str, ticker: Optional[str] = None, fiscal_year: Optional[int] = None, top_k: int = 5,
) -> dict:
    from retrieval.retriever import retrieve

    tickers = [ticker.upper()] if ticker else []
    years = [fiscal_year] if fiscal_year else []
    top_k = max(1, min(top_k, 15))
    chunks = retrieve(query=query, tickers=tickers, years=years, top_k=top_k, focus="other")
    return {
        "query": query,
        "results": [
            {
                "ticker": rc.chunk.ticker, "company": rc.chunk.company,
                "fiscal_year": rc.chunk.fiscal_year, "section": rc.chunk.section_name,
                "chunk_type": rc.chunk.chunk_type, "score": round(rc.score, 3),
                "text": rc.chunk.text[:MAX_PASSAGE_CHARS],
            }
            for rc in chunks
        ],
    }


def fetch_10k_impl(company: str, fiscal_year: Optional[int] = None) -> dict:
    from ingestion.auto_ingest import YearNotAvailable, ensure_ticker_indexed
    from ingestion.registry import resolve_company

    info = resolve_company(company)
    if not info:
        return {"status": "not_found", "message": f"No SEC filer matches {company!r}."}
    name = info["title"] or info["ticker"]
    try:
        result = ensure_ticker_indexed(info["ticker"], name, target_year=fiscal_year)
    except YearNotAvailable as exc:
        return {"status": "year_not_available", "ticker": exc.ticker,
                "requested": exc.requested, "available": list(exc.available)}
    if not result:
        return {"status": "failed", "ticker": info["ticker"],
                "message": "The filing could not be downloaded or indexed; try again shortly."}
    _, year = result
    return {"status": "indexed", "ticker": info["ticker"], "company": name, "fiscal_year": year}


def get_xbrl_figure_impl(ticker: str, metric: str, fiscal_year: int) -> dict:
    if metric not in METRICS:
        return {"error": f"Unknown metric {metric!r}.", "metrics": sorted(METRICS)}
    fig = get_figure(ticker, metric, fiscal_year)
    if fig is None:
        return {"error": f"No reported {metric} for {ticker.upper()} FY{fiscal_year} in SEC XBRL data "
                         f"(some filers, e.g. banks, don't report every metric)."}
    return fig.to_dict()


def ask_financial_question_impl(question: str) -> dict:
    from query import ask

    r = ask(question)
    return {
        "answer": r.answer, "query_type": r.query_type, "citations": r.citations,
        "verification": r.verification,
    }


def list_indexed_filings_impl() -> dict:
    from retrieval.vector_store import list_collections

    filings = []
    for name in list_collections():
        ticker, _, year = name.rpartition("_")
        if year.isdigit():
            filings.append({"ticker": ticker, "fiscal_year": int(year),
                            "company": TICKER_TO_COMPANY.get(ticker, {}).get("name", ticker)})
    return {"count": len(filings), "filings": filings}


# ── MCP registration ────────────────────────────────────────────────────────

def build_server() -> MCPServer:
    server = MCPServer(
        "financial-rag",
        instructions=(
            "Answers questions from SEC 10-K filings. Prefer get_xbrl_figure for a single "
            "headline number (revenue, net income, R&D, ...); use search_filings for passages "
            "and context; use ask_financial_question for a full cited, verified answer. "
            "Call fetch_10k first for a company that isn't indexed yet."
        ),
    )

    @server.tool()
    def search_filings(query: str, ticker: Optional[str] = None,
                       fiscal_year: Optional[int] = None, top_k: int = 5) -> dict:
        """Search indexed 10-K passages (dense + BM25 hybrid, reranked). Optionally restrict
        to a ticker and/or fiscal year. Returns passages with section, year and score."""
        return search_filings_impl(query, ticker, fiscal_year, top_k)

    @server.tool()
    def fetch_10k(company: str, fiscal_year: Optional[int] = None) -> dict:
        """Ensure a company's 10-K is downloaded, parsed and indexed. `company` may be a
        ticker or a name. Can take a few minutes the first time for a new company."""
        return fetch_10k_impl(company, fiscal_year)

    @server.tool()
    def get_xbrl_figure(ticker: str, metric: str, fiscal_year: int) -> dict:
        """The exact annual value a company reported to the SEC (XBRL), with the source
        filing. metric is one of: """ + ", ".join(sorted(METRICS)) + "."
        return get_xbrl_figure_impl(ticker, metric, fiscal_year)

    @server.tool()
    def ask_financial_question(question: str) -> dict:
        """Run the full agent (plan, retrieve, generate, verify against SEC XBRL) and return
        a cited answer plus the verification result."""
        return ask_financial_question_impl(question)

    @server.tool()
    def list_indexed_filings() -> dict:
        """List the ticker / fiscal-year filings currently searchable."""
        return list_indexed_filings_impl()

    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Financial RAG MCP server")
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    server = build_server()
    if args.http:
        server.run(transport="streamable-http", port=args.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
