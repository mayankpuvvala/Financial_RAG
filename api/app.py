"""
FastAPI server for the Financial RAG system.

Run:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    GET  /health       — server status + loaded collections
    GET  /collections  — list available ticker/year collections
    POST /query        — ask a financial question
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field

from query import ask
from retrieval.vector_store import list_collections
from ingestion.embedder import encode_query
from retrieval.reranker import _get_reranker


# ---------------------------------------------------------------------------
# Lifespan — warm up models before the first request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models …")
    try:
        encode_query("warm up")
        _get_reranker()
        logger.success("Models ready.")
    except Exception as exc:
        logger.warning(f"Warm-up failed (non-fatal): {exc}")
    yield
    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial RAG API",
    description="Question-answering over SEC 10-K filings (2023–2025)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="Your financial question")

    model_config = {"json_schema_extra": {"example": {"question": "How did JPMorgan net income trend from 2023 to 2025?"}}}


class CitationOut(BaseModel):
    index:       int
    company:     str
    ticker:      str
    fiscal_year: int
    section:     str
    score:       float


class QueryResponse(BaseModel):
    query:       str
    answer:      str
    query_type:  str
    citations:   List[CitationOut]
    chunks_used: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    cols = list_collections()
    return {
        "status": "ok",
        "collections_loaded": len(cols),
        "collections": cols,
    }


@app.get("/collections", tags=["meta"])
def collections():
    return {"collections": list_collections()}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Incoming query: {req.question[:80]}")

    try:
        result = ask(req.question)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    citations = [
        CitationOut(
            index=c["index"],
            company=c["company"],
            ticker=c["ticker"],
            fiscal_year=c["fiscal_year"],
            section=c["section"],
            score=c.get("score", 0.0),
        )
        for c in result.citations
    ]

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        query_type=result.query_type,
        citations=citations,
        chunks_used=len(result.chunks_used),
    )
