"""
Offline tests for the LangGraph QA loop: retrieval, generation and the XBRL
lookup are stubbed, so these exercise only the routing/corrective logic.
"""
import pytest

import agent.graph as G
from facts.xbrl import Figure
from models import Chunk, QueryResult, RetrievedChunk

REV = Figure(ticker="AAPL", metric="revenue", fiscal_year=2024, value=391_035_000_000, unit="USD",
             concept="Revenues", period_start="2023-10-01", period_end="2024-09-28",
             form="10-K", accession="0000320193-24-000123", filed="2024-11-01")


def _chunk(text="Total net sales | 391035 | 383285"):
    c = Chunk(chunk_id="c1", parent_id="item_8", doc_id="d1", text=text, company="Apple Inc.",
              ticker="AAPL", filing_type="10-K", fiscal_year=2024, section_name="Item 8",
              chunk_type="table", token_count=10)
    return RetrievedChunk(chunk=c, score=1.0, parent_text=text)


@pytest.fixture
def stub(monkeypatch):
    """Returns a recorder; set .answers to the sequence the 'LLM' will produce."""
    class S:
        answers = []
        gen_calls = []
        retrieve_calls = []
    s = S()

    def fake_retrieve(query, tickers, years, top_k, focus):
        s.retrieve_calls.append(focus)
        return [_chunk()]

    def fake_generate(query, retrieved, query_type="single_doc"):
        s.gen_calls.append({"query": query, "n": len(retrieved),
                            "has_xbrl": any(r.chunk.chunk_type == "xbrl" for r in retrieved)})
        ans = s.answers.pop(0)
        return QueryResult(query=query, answer=ans, citations=[{"index": 1, "company": "Apple Inc.",
                           "ticker": "AAPL", "fiscal_year": 2024, "section": "Item 8"}],
                           chunks_used=retrieved, query_type=query_type)

    s.answers, s.gen_calls, s.retrieve_calls = [], [], []
    monkeypatch.setattr(G, "retrieve", fake_retrieve)
    monkeypatch.setattr(G, "generate_answer", fake_generate)
    monkeypatch.setattr(G, "targets_for", lambda t, y, f: [REV] if f == "revenue" else [])
    return s


def _run(focus="revenue"):
    out = G.build_qa_graph().invoke({
        "query": "What was Apple revenue in FY2024?", "tickers": ["AAPL"], "years": [2024],
        "focus": focus, "query_type": "single_doc", "top_k": 3, "require_all": True,
    })
    return out["result"]


def test_correct_answer_passes_straight_through(stub):
    stub.answers = ["Revenue was $391,035 million [1]."]
    r = _run()
    assert r.verification["status"] == "verified"
    assert len(stub.gen_calls) == 1
    assert "inject_facts" not in r.verification["steps"]


def test_wrong_number_triggers_fact_injection_then_passes(stub):
    stub.answers = ["Revenue was $383,285 million [1].", "Revenue was $391,035 million [1]."]
    r = _run()
    assert r.verification["steps"].count("inject_facts") == 1
    assert stub.gen_calls[1]["has_xbrl"] and not stub.gen_calls[0]["has_xbrl"]
    assert "391,035" in r.answer
    assert r.verification["status"] == "verified"


def test_still_wrong_falls_back_to_deterministic_answer(stub):
    stub.answers = ["Revenue was $383,285 million [1].", "Revenue was $1 million [1]."]
    r = _run()
    assert "fallback_facts" in r.verification["steps"]
    assert "$391,035 million" in r.answer and "[1]" in r.answer
    assert r.citations and r.citations[0]["chunk_type"] == "xbrl"
    assert len(stub.gen_calls) == 2        # bounded: no endless loop


def test_refusal_with_xbrl_target_recovers(stub):
    stub.answers = ["The revenue is not mentioned in the provided context.", "Revenue was $391,035 million [1]."]
    r = _run()
    assert r.verification["status"] == "verified"
    assert "inject_facts" in r.verification["steps"]


def test_refusal_without_xbrl_broadens_once(stub):
    stub.answers = ["That is not stated in the provided context.", "Still not mentioned in the provided context."]
    r = _run(focus="business_overview")
    assert stub.retrieve_calls == ["business_overview", "other"]
    assert len(stub.gen_calls) == 2
    assert r.verification["steps"].count("broaden") == 1


def test_unsupported_figure_gets_one_strict_retry(stub):
    stub.answers = ["The segment earned $77.7 billion [1].", "The segment earned $77.7 billion [1]."]
    r = _run(focus="segment_info")
    assert len(stub.gen_calls) == 2
    assert "figures above could not be matched" in r.answer
    assert "strict" in stub.gen_calls[1]["query"].lower() or "verbatim" in stub.gen_calls[1]["query"]


def test_llm_outage_is_not_reported_as_missing_company():
    from routing.classifier import ClassifiedQuery
    c = ClassifiedQuery(query_type="single_doc", tickers=[], years=[], reasoning="classification error",
                        error="RateLimitError: insufficient_quota")
    r = G._early_result("What was Amazon's revenue in 2024?", c)
    assert r.query_type == "service_unavailable"
    assert "which company" not in r.answer.lower()


def test_no_company_and_no_error_still_asks_which_company():
    from routing.classifier import ClassifiedQuery
    c = ClassifiedQuery(query_type="single_doc", tickers=[], years=[], reasoning="x")
    assert G._early_result("What was the revenue?", c).query_type == "clarification_needed"


def test_non_numeric_answer_is_not_touched(stub):
    stub.answers = ["Apple designs and sells smartphones, computers and services [1]."]
    r = _run(focus="business_overview")
    assert r.verification["status"] == "unchecked"
    assert len(stub.gen_calls) == 1
