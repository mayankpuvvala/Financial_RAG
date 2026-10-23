"""The chat UI's company/year chips must override the classifier — and still
go through the agent (no bypass of verification)."""
import agent.graph as G
from routing.classifier import ClassifiedQuery


def _plan(monkeypatch, classified, **state):
    monkeypatch.setattr(G, "classify_and_ensure", lambda q: classified)
    return G._plan_node({"query": "tell me about it", **state})["classification"]


def test_ticker_and_year_filters_override_classifier(monkeypatch):
    c = ClassifiedQuery(query_type="single_doc", tickers=["MSFT"], years=[2023], reasoning="x")
    out = _plan(monkeypatch, c, tickers=["aapl"], years=[2024])
    assert out.tickers == ["AAPL"] and out.years == [2024]


def test_chosen_company_is_never_out_of_scope(monkeypatch):
    c = ClassifiedQuery(query_type="out_of_scope", tickers=[], years=[], reasoning="x")
    assert _plan(monkeypatch, c, tickers=["AAPL"]).query_type == "single_doc"
    c2 = ClassifiedQuery(query_type="out_of_scope", tickers=[], years=[], reasoning="x")
    assert _plan(monkeypatch, c2, tickers=["AAPL", "MSFT"]).query_type == "multi_doc"


def test_no_filters_leaves_classifier_untouched(monkeypatch):
    c = ClassifiedQuery(query_type="single_doc", tickers=["MSFT"], years=[2023], reasoning="x")
    out = _plan(monkeypatch, c, tickers=None, years=None)
    assert out.tickers == ["MSFT"] and out.years == [2023]
