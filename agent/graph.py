"""
LangGraph agent for Financial RAG.

Two graphs:

  QA graph  (one question about specific ticker(s)/year(s))

      retrieve -> generate -> verify --ok--------------------------> finalize
                                |
                                |-- figure wrong/missing, XBRL known --> inject_facts -> generate -> verify
                                |        (second failure) -----------> fallback_facts -> finalize
                                |-- refusal --------------------------> broaden -> generate -> verify
                                '-- unsupported figure ---------------> strict_retry -> generate -> verify

      Every loop-back edge is guarded by a flag in state, so each
      corrective step runs at most once.

  Main graph

      plan -> (early exit | qa | decompose -> [answer_sub x N, parallel] -> synthesize) -> caveats

`answer_sub` runs the QA graph once per sub-question, so comparison and
trend questions get the same verify-and-correct loop as single questions.
"""

import operator
import threading
from typing import Annotated, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger

from config import settings
from facts.xbrl import METRIC_LABEL, as_source
from generation.generator import _build_context, generate_answer
from generation.synthesizer import merge_sub_results
from models import QueryResult, RetrievedChunk
from retrieval.retriever import retrieve
from routing.classifier import ClassifiedQuery
from routing.decomposer import decompose_query, decompose_temporal
from routing.resolver import classify_and_ensure
from verification.verify import Verdict, is_refusal, targets_for, verify_answer

# Local Qdrant is only ever read here, but the original synthesizer retrieved
# sequentially on purpose (SQLite lock); keep that, and let generation —
# network-bound — overlap across sub-questions.
_RETRIEVE_LOCK = threading.Lock()

_STRICT_SUFFIX = (
    "\n\nIMPORTANT: state only dollar figures that appear verbatim in the sources. "
    "Do not estimate, round differently, or invent any figure."
)

MAX_PARALLEL = 4


# ─────────────────────────────────────────────────────────────────────────────
# QA graph
# ─────────────────────────────────────────────────────────────────────────────

class QAState(TypedDict, total=False):
    query:        str
    gen_query:    str            # what the generator is asked (query + strict suffix)
    tickers:      List[str]
    years:        List[int]
    focus:        str
    query_type:   str
    top_k:        int
    retrieved:    List[RetrievedChunk]
    figures:      list           # XBRL Figures this answer is expected to state
    require_all:  bool
    result:       QueryResult
    verdict:      Verdict
    broadened:       bool
    facts_injected:  bool
    strict_retried:  bool
    trace:        List[str]


def _retrieve_node(state: QAState) -> dict:
    with _RETRIEVE_LOCK:
        retrieved = retrieve(
            query=state["query"], tickers=state["tickers"], years=state["years"],
            top_k=state["top_k"], focus=state["focus"],
        )
    figures = targets_for(state["tickers"], state["years"], state["focus"])
    return {
        "retrieved": retrieved,
        "figures": figures,
        "gen_query": state["query"],
        "trace": [f"retrieve:{len(retrieved)} chunks, {len(figures)} xbrl target(s)"],
    }


def _generate_node(state: QAState) -> dict:
    result = generate_answer(
        query=state.get("gen_query") or state["query"],
        retrieved=state["retrieved"],
        query_type=state["query_type"],
    )
    return {"result": result}


def _verify_node(state: QAState) -> dict:
    result, figures = state["result"], state.get("figures") or []
    retrieved = state["retrieved"]
    context = [rc.chunk.text for rc in retrieved] + [rc.parent_text for rc in retrieved]

    if not figures and is_refusal(result.answer):
        verdict = Verdict("refused", "answer says the information was not found")
    else:
        verdict = verify_answer(result.answer, context, figures, require_all=state.get("require_all", False))
    logger.info(f"Verify [{state['query'][:50]}]: {verdict.status} {verdict.details}")
    return {"verdict": verdict, "trace": state.get("trace", []) + [f"verify:{verdict.status}"]}


def _after_verify(state: QAState) -> str:
    v, figures = state["verdict"], state.get("figures")
    if v.ok:
        return "finalize"
    if figures:
        return "fallback_facts" if state.get("facts_injected") else "inject_facts"
    if v.status == "refused" and not state.get("broadened"):
        return "broaden"
    if v.status == "unsupported" and not state.get("strict_retried"):
        return "strict_retry"
    return "finalize"


def _inject_facts_node(state: QAState) -> dict:
    sources = [as_source(f) for f in state["figures"]]
    return {
        "retrieved": sources + state["retrieved"],
        "facts_injected": True,
        "trace": state["trace"] + ["inject_facts"],
    }


def _broaden_node(state: QAState) -> dict:
    # A narrow focus boost can mis-target the wrong section, or top_k can be
    # too tight — retry once with the focus restriction dropped and top_k widened.
    with _RETRIEVE_LOCK:
        retrieved = retrieve(
            query=state["query"], tickers=state["tickers"], years=state["years"],
            top_k=min(state["top_k"] * 2, 15), focus="other",
        )
    return {"retrieved": retrieved, "broadened": True, "trace": state["trace"] + ["broaden"]}


def _strict_retry_node(state: QAState) -> dict:
    return {
        "gen_query": state["query"] + _STRICT_SUFFIX,
        "strict_retried": True,
        "trace": state["trace"] + ["strict_retry"],
    }


def _fallback_facts_node(state: QAState) -> dict:
    """The LLM still got a headline figure wrong with the XBRL fact in front of
    it — answer deterministically from the SEC data instead."""
    sources = [as_source(f) for f in state["figures"]]
    _, citations = _build_context(sources)
    index_of = {(c["ticker"], c["fiscal_year"]): c["index"] for c in citations}
    lines = []
    for f, src in zip(state["figures"], sources):
        idx = index_of.get((f.ticker, f.fiscal_year), 1)
        lines.append(
            f"According to its SEC filing data, {src.chunk.company}'s "
            f"{METRIC_LABEL.get(f.metric, f.metric)} for fiscal year {f.fiscal_year} was {f.human()} [{idx}]."
        )
    prior = state["result"]
    result = QueryResult(
        query=prior.query, answer="\n".join(lines), citations=citations,
        chunks_used=sources, query_type=prior.query_type,
    )
    return {
        "result": result,
        "verdict": Verdict("verified", "answered deterministically from SEC XBRL facts",
                           targets=[f.to_dict() for f in state["figures"]]),
        "trace": state["trace"] + ["fallback_facts"],
    }


def _finalize_node(state: QAState) -> dict:
    result, verdict = state["result"], state["verdict"]
    if verdict.status == "unsupported":
        result.answer += ("\n\n*Note: some figures above could not be matched to the "
                          "retrieved filing text — please check them against the cited sources.*")
    result.verification = {**verdict.to_dict(), "steps": state.get("trace", [])}
    return {"result": result}


def build_qa_graph():
    g = StateGraph(QAState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("generate", _generate_node)
    g.add_node("verify", _verify_node)
    g.add_node("inject_facts", _inject_facts_node)
    g.add_node("broaden", _broaden_node)
    g.add_node("strict_retry", _strict_retry_node)
    g.add_node("fallback_facts", _fallback_facts_node)
    g.add_node("finalize", _finalize_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _after_verify, {
        "finalize": "finalize", "inject_facts": "inject_facts", "fallback_facts": "fallback_facts",
        "broaden": "broaden", "strict_retry": "strict_retry",
    })
    for node in ("inject_facts", "broaden", "strict_retry"):
        g.add_edge(node, "generate")
    g.add_edge("fallback_facts", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Main graph
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    query:          str
    tickers:        Optional[List[str]]    # optional caller filters (override the classifier)
    years:          Optional[List[int]]
    classification: ClassifiedQuery
    sub_questions:  List[Dict]
    sub_results:    Annotated[list, operator.add]
    result:         QueryResult


_qa_graph = None
_main_graph = None
_build_lock = threading.Lock()


def _qa():
    global _qa_graph
    if _qa_graph is None:
        with _build_lock:
            if _qa_graph is None:
                _qa_graph = build_qa_graph()
    return _qa_graph


def _early_result(query: str, c: ClassifiedQuery) -> Optional[QueryResult]:
    """Cases that must not reach retrieval (nothing sensible to search)."""
    def done(answer: str, qtype: str) -> QueryResult:
        return QueryResult(query=query, answer=answer, citations=[], chunks_used=[], query_type=qtype)

    if c.error:
        # The classification call itself failed (provider outage, exhausted quota, ...).
        # Say so — answering "which company?" would blame the user for our outage.
        logger.error(f"Classification unavailable: {c.error}")
        return done("I couldn't process your question just now because the language-model "
                    "service is unavailable. Please try again in a few minutes.", "service_unavailable")

    if c.query_type == "out_of_scope":
        return done("This query is outside the scope of the available SEC 10-K filings.", "out_of_scope")

    # A named company that couldn't be resolved must not fall through to
    # "search every company" — that returns irrelevant citations.
    if not c.tickers and (c.failed_lookups or c.ingest_failed or c.year_not_available):
        parts = []
        if c.failed_lookups:
            parts.append(
                f"I couldn't find any SEC filings for {', '.join(c.failed_lookups)} — it doesn't "
                f"appear to be a public company that files with the SEC, or its filings aren't available here."
            )
        if c.ingest_failed:
            parts.append(
                f"I found that {', '.join(c.ingest_failed)} files with the SEC, but hit a technical "
                f"problem fetching or indexing its 10-K just now — please try again in a moment."
            )
        if c.year_not_available:
            parts.append(f"I don't have that fiscal year for: {'; '.join(c.year_not_available)}.")
        return done(" ".join(parts), "unresolved_company")

    if c.query_type in ("single_doc", "summarization") and not c.tickers:
        return done("Which company are you asking about? I can only answer from indexed SEC 10-K "
                    "filings for a specific, named company.", "clarification_needed")
    return None


def _plan_node(state: AgentState) -> dict:
    classification = classify_and_ensure(state["query"])

    # Caller-supplied filters (the chat UI's company/year chips) win over what
    # the classifier inferred, exactly as the chat endpoint always intended.
    if state.get("tickers"):
        classification.tickers = [t.upper() for t in state["tickers"]]
        if classification.query_type == "out_of_scope":   # a chosen company is, by definition, in scope
            classification.query_type = "multi_doc" if len(classification.tickers) > 1 else "single_doc"
    if state.get("years"):
        classification.years = list(state["years"])
    logger.info(
        f"Plan: {classification.query_type} | {classification.tickers} | "
        f"{classification.years} | focus={classification.focus}"
    )
    return {"classification": classification}


def _after_plan(state: AgentState) -> str:
    c = state["classification"]
    if _early_result(state["query"], c) is not None:
        return "early"
    return "decompose" if c.query_type in ("multi_doc", "temporal") else "qa"


def _early_node(state: AgentState) -> dict:
    return {"result": _early_result(state["query"], state["classification"])}


def _qa_node(state: AgentState) -> dict:
    c = state["classification"]
    # A single_doc query can still span several years/companies ("most recent"
    # expands to all three years) — scale top_k, capped to bound context size.
    n_targets = max(1, len(c.tickers)) * max(1, len(c.years))
    out = _qa().invoke({
        "query": state["query"], "tickers": c.tickers, "years": c.years, "focus": c.focus,
        "query_type": c.query_type, "top_k": settings.rerank_top_k * min(n_targets, 5),
        # With exactly one ticker/year the answer must state that figure; with
        # several, a latest-year-only answer is legitimate.
        "require_all": n_targets == 1,
    })
    return {"result": out["result"]}


def _decompose_node(state: AgentState) -> dict:
    c = state["classification"]
    # temporal = one company across years: one sub-question per year,
    # deterministic (the LLM decomposer misapplies its "describe the
    # business" special case to single-company metric questions).
    if c.query_type == "temporal" and len(c.tickers) == 1 and c.years:
        # "from 2023 to 2025" is often extracted as just the endpoints; a trend
        # covers every year in between.
        ys = sorted(set(c.years))
        years = list(range(ys[0], ys[-1] + 1)) if len(ys) > 1 else ys
        subs = decompose_temporal(c.tickers[0], years, c.focus)
    else:
        subs = decompose_query(state["query"], c.tickers, c.years)
    logger.info(f"Decomposed into {len(subs)} sub-question(s)")
    return {"sub_questions": subs}


def _fan_out(state: AgentState):
    c = state["classification"]
    sends = [
        Send("answer_sub", {"sub": sub, "focus": c.focus})
        for sub in state["sub_questions"]
    ]
    return sends or "synthesize"


def _answer_sub_node(payload: dict) -> dict:
    sub = payload["sub"]
    try:
        out = _qa().invoke({
            "query": sub["question"], "tickers": [sub["ticker"]], "years": [sub["year"]],
            "focus": payload["focus"], "query_type": "sub_question",
            "top_k": settings.rerank_top_k, "require_all": True,
        })
        return {"sub_results": [(sub, out["result"])]}
    except Exception as exc:
        logger.error(f"Sub-question failed ({sub.get('ticker')} FY{sub.get('year')}): {exc}")
        return {"sub_results": []}


_STATUS_RANK = {"verified": 0, "unchecked": 0, "unsupported": 1, "refused": 2, "mismatch": 3}


def _synthesize_node(state: AgentState) -> dict:
    c, subs = state["classification"], state.get("sub_results", [])
    if not subs:
        return {"result": QueryResult(
            query=state["query"], answer="Could not retrieve information for this query.",
            citations=[], chunks_used=[], query_type=c.query_type,
        )}
    result = merge_sub_results(state["query"], c.query_type, subs)
    verdicts = [(s, r.verification or {}) for s, r in subs]
    worst = max((v.get("status", "unchecked") for _, v in verdicts), key=lambda s: _STATUS_RANK.get(s, 0))
    result.verification = {
        "status": worst,
        "subs": [{"ticker": s["ticker"], "year": s["year"], **v} for s, v in verdicts],
    }
    return {"result": result}


def _caveats_node(state: AgentState) -> dict:
    """Some named companies resolved and the query ran on those, but at least
    one didn't — say so instead of silently answering about a subset."""
    c, result = state["classification"], state["result"]
    if c.tickers and (c.failed_lookups or c.ingest_failed or c.year_not_available):
        caveats = []
        if c.failed_lookups:
            caveats.append(f"couldn't find SEC filings for {', '.join(c.failed_lookups)}")
        if c.ingest_failed:
            caveats.append(f"hit a technical problem indexing {', '.join(c.ingest_failed)}")
        if c.year_not_available:
            caveats.append(f"don't have the requested year for {', '.join(c.year_not_available)}")
        result.answer += f"\n\n*Note: I {'; '.join(caveats)}, so this only covers the companies I could find.*"
    return {"result": result}


def build_main_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", _plan_node)
    g.add_node("early", _early_node)
    g.add_node("qa", _qa_node)
    g.add_node("decompose", _decompose_node)
    g.add_node("answer_sub", _answer_sub_node)
    g.add_node("synthesize", _synthesize_node)
    g.add_node("caveats", _caveats_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _after_plan, {"early": "early", "qa": "qa", "decompose": "decompose"})
    g.add_conditional_edges("decompose", _fan_out, ["answer_sub", "synthesize"])
    g.add_edge("answer_sub", "synthesize")
    g.add_edge("early", END)
    g.add_edge("qa", "caveats")
    g.add_edge("synthesize", "caveats")
    g.add_edge("caveats", END)
    return g.compile()


def run_agent(query: str, tickers: Optional[List[str]] = None, years: Optional[List[int]] = None) -> QueryResult:
    global _main_graph
    if _main_graph is None:
        with _build_lock:
            if _main_graph is None:
                _main_graph = build_main_graph()
    out = _main_graph.invoke(
        {"query": query, "tickers": tickers, "years": years}, config={"max_concurrency": MAX_PARALLEL},
    )
    return out["result"]
