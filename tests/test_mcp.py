"""
Integration test: spawn the MCP server as a subprocess over stdio and call
its tools with a real MCP client. Needs network only for get_xbrl_figure
(SEC), and the local Qdrant index for the search/list tools.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def _payload(result):
    if getattr(result, "structured_content", None):
        sc = result.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(result.content[0].text)


async def _session(fn):
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], cwd=str(ROOT))
    async with Client(params) as client:
        return await fn(client)


def run(fn):
    return asyncio.run(_session(fn))


def test_lists_all_tools():
    async def go(c):
        return {t.name for t in (await c.list_tools()).tools}
    assert run(go) == {"search_filings", "fetch_10k", "get_xbrl_figure",
                       "ask_financial_question", "list_indexed_filings"}


def test_get_xbrl_figure_over_mcp():
    async def go(c):
        return _payload(await c.call_tool("get_xbrl_figure",
                        {"ticker": "AAPL", "metric": "revenue", "fiscal_year": 2024}))
    p = run(go)
    assert p["value"] == 391_035_000_000
    assert p["period_end"] == "2024-09-28"


def test_unknown_metric_is_reported_not_raised():
    async def go(c):
        return _payload(await c.call_tool("get_xbrl_figure",
                        {"ticker": "AAPL", "metric": "vibes", "fiscal_year": 2024}))
    assert "revenue" in run(go)["metrics"]


@pytest.mark.skipif(not (ROOT / "data" / "qdrant" / "collection" / "AAPL_2024").exists(),
                    reason="AAPL_2024 index not present")
def test_search_filings_over_mcp():
    async def go(c):
        return _payload(await c.call_tool("search_filings",
                        {"query": "total net sales", "ticker": "AAPL", "fiscal_year": 2024, "top_k": 3}))
    p = run(go)
    assert 1 <= len(p["results"]) <= 3
    assert all(r["ticker"] == "AAPL" and r["fiscal_year"] == 2024 for r in p["results"])
