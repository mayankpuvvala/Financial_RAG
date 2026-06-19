"""
Financial RAG — FastAPI backend

Endpoints:
  POST /query        ask a question, get answer + citations
  POST /ingest       trigger the full ingestion pipeline
  GET  /documents    list all indexed documents
  GET  /collections  Qdrant collection stats
  GET  /health       system status
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from config import settings, COMPANIES
from query import ask
from retrieval.vector_store import list_collections, get_collection_stats, collection_exists

app = FastAPI(
    title="Financial RAG API",
    description="Multi-document financial analyst RAG over SEC 10-K filings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────

class QueryRequest(BaseModel):
    question: str
    tickers:  Optional[List[str]] = None
    years:    Optional[List[int]] = None

class QueryResponse(BaseModel):
    query:            str
    answer:           str
    citations:        List[dict]
    query_type:       str
    chunks_used_count: int

class DocumentInfo(BaseModel):
    ticker:      str
    company:     str
    sector:      str
    fiscal_year: int
    collection:  str
    indexed:     bool

class IngestResponse(BaseModel):
    status:     str
    documents:  int
    chunks:     int
    collections: List[str]


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    """System status — lists active Qdrant collections."""
    cols = list_collections()
    return {
        "status":      "healthy",
        "collections": cols,
        "total_indexed": len(cols),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a financial question.
    Optionally filter by tickers and/or fiscal years.

    Examples:
      {"question": "What was Apple revenue in FY2024?"}
      {"question": "Compare MSFT vs GOOGL R&D in 2024", "tickers": ["MSFT","GOOGL"], "years": [2024]}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        # If caller passes explicit filters, inject them into the query context
        # by overriding the classifier (useful for API clients that already know
        # which documents to search)
        if req.tickers or req.years:
            from routing.classifier import ClassifiedQuery
            from retrieval.retriever import retrieve
            from generation.generator import generate_answer
            from generation.synthesizer import synthesize
            from routing.classifier import classify_query

            classification = classify_query(req.question)
            # Caller-supplied filters take precedence
            tickers = req.tickers or classification.tickers
            years   = req.years   or classification.years

            if classification.query_type in ("multi_doc", "temporal"):
                result = synthesize(req.question, tickers, years, classification.query_type)
            else:
                retrieved = retrieve(req.question, tickers, years)
                result    = generate_answer(req.question, retrieved, classification.query_type)
        else:
            result = ask(req.question)

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            citations=result.citations,
            query_type=result.query_type,
            chunks_used_count=len(result.chunks_used),
        )
    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents", response_model=List[DocumentInfo])
def documents():
    """List all documents and whether they are indexed in Qdrant."""
    from ingestion.downloader import load_manifest

    try:
        manifest = load_manifest(settings.raw_dir)
    except FileNotFoundError:
        return []

    company_map = {c["ticker"]: c for c in COMPANIES}
    docs = []
    seen = set()

    for record in manifest:
        key = (record["ticker"], record["fiscal_year"])
        if key in seen:
            continue
        seen.add(key)

        col     = f"{record['ticker']}_{record['fiscal_year']}"
        company = company_map.get(record["ticker"], {})

        docs.append(DocumentInfo(
            ticker      = record["ticker"],
            company     = record.get("company", company.get("name", record["ticker"])),
            sector      = record.get("sector",  company.get("sector", "Unknown")),
            fiscal_year = record["fiscal_year"],
            collection  = col,
            indexed     = collection_exists(col),
        ))

    docs.sort(key=lambda d: (d.ticker, d.fiscal_year))
    return docs


@app.get("/collections")
def collections():
    """Qdrant collection stats — point counts per ticker-year."""
    cols = list_collections()
    stats = []
    for col in cols:
        try:
            s = get_collection_stats(col)
            stats.append(s)
        except Exception:
            stats.append({"name": col, "points_count": "unknown"})
    return {"collections": stats, "total": len(stats)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(background_tasks: BackgroundTasks):
    """
    Trigger the full ingestion pipeline in the background.
    Returns immediately — check /health and /collections to monitor progress.

    Note: first run downloads ~500 MB of model weights and may take 20+ minutes.
    """
    def _run():
        from ingestion.downloader import download_all_filings
        from ingestion.parser import parse_all_filings
        from ingestion.chunker import chunk_all_documents
        from ingestion.embedder import index_chunks

        manifest  = download_all_filings()
        documents = parse_all_filings(manifest, settings.parsed_dir)
        chunks    = chunk_all_documents(documents, settings.chunks_dir)
        index_chunks(chunks)
        logger.success(f"Ingest complete: {len(documents)} docs, {len(chunks)} chunks")

    background_tasks.add_task(_run)

    return IngestResponse(
        status="ingestion started in background",
        documents=0,
        chunks=0,
        collections=list_collections(),
    )


@app.get("/evaluate")
def evaluate():
    """
    Run RAGAS evaluation on the test set.
    Returns faithfulness, answer_relevance, context_recall, context_precision.
    (Stub — implemented in the evaluation layer.)
    """
    return {"status": "evaluation endpoint — coming in next step"}
