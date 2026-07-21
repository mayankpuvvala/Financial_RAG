"""
FastAPI server for the Financial RAG system.

Run:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    GET  /health       — server status + loaded collections
    GET  /collections  — list available ticker/year collections
    POST /query        — ask a financial question
    POST /ingest        — trigger the bundled-12 ingestion pipeline in the background
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from query import ask
from retrieval.vector_store import list_collections
from ingestion.embedder import encode_query
from retrieval.reranker import _get_reranker
from api.chat import router as chat_router

_UI_FILE = Path(__file__).parent.parent / "ui" / "index.html"


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

app.include_router(chat_router)


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

@app.get("/", include_in_schema=False)
def ui():
    return HTMLResponse(_UI_FILE.read_text(encoding="utf-8"))


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


@app.get("/ingest/debug-download", tags=["meta"])
def ingest_debug_download():
    """
    Synchronously attempt a single-company, single-filing SEC EDGAR download
    and return the raw outcome/exception. ingestion/downloader.py catches
    per-ticker download exceptions and just returns [] — correct for not
    letting one bad ticker abort a 12-company run, but it also means a
    total download failure (e.g. network/EDGAR blocking a host's egress)
    completes "successfully" with zero records and no visible error. This
    bypasses that swallowing to surface exactly what SEC EDGAR/network says.
    """
    import traceback
    from sec_edgar_downloader import Downloader

    result: dict = {}
    try:
        dl = Downloader(
            company_name="FinancialRAG",
            email_address=settings.edgar_email,
            download_folder=str(settings.raw_dir),
        )
        dl.get("10-K", "AAPL", limit=1)
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()

    return result


# Guards against two ingestion runs overlapping (e.g. a double-click, or a
# retry while the first request is still running) — the pipeline's own
# skip-if-already-done checks make repeat calls cheap once data exists, but
# a genuinely concurrent second run would duplicate download/parse work.
# _last_ingest_error/_last_ingest_step exist purely for remote diagnosis —
# there's no shell/log access on a host like Railway without the CLI set up,
# so without this a failed run is invisible other than "it stopped running".
_ingest_lock = threading.Lock()
_ingest_running = False
_last_ingest_error: Optional[str] = None
_last_ingest_step: Optional[str] = None


def _run_ingestion_background() -> None:
    global _ingest_running, _last_ingest_error, _last_ingest_step
    _last_ingest_error = None
    try:
        import run_ingestion
        _last_ingest_step = "starting"
        logger.info("Background ingestion started (bundled 12 companies)")
        run_ingestion.main()
        _last_ingest_step = "complete"
        logger.success("Background ingestion complete")
    except Exception as exc:
        _last_ingest_step = "failed"
        _last_ingest_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Background ingestion failed")
    finally:
        with _ingest_lock:
            _ingest_running = False


@app.get("/ingest/status", tags=["meta"])
def ingest_status():
    """Remote-diagnosis endpoint — last run's outcome when there's no log access."""
    return {
        "running": _ingest_running,
        "last_step": _last_ingest_step,
        "last_error": _last_ingest_error,
        "collections_loaded": len(list_collections()),
    }


@app.post("/ingest", tags=["meta"])
def ingest(background_tasks: BackgroundTasks, token: Optional[str] = Query(None)):
    """
    Trigger the bundled-12-company ingestion pipeline (download → parse →
    chunk → embed) in the background. Returns immediately — poll /health for
    collections_loaded, or /ingest/status for step/error detail, to track
    progress (35 collections when complete). Safe to call repeatedly:
    already-downloaded/parsed/indexed companies are skipped, so a second
    call after a partial or failed run just resumes.
    """
    if settings.admin_token and token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    global _ingest_running
    with _ingest_lock:
        if _ingest_running:
            return {"status": "already running", "collections_loaded": len(list_collections())}
        _ingest_running = True

    background_tasks.add_task(_run_ingestion_background)
    return {
        "status": "started",
        "collections_loaded_now": len(list_collections()),
        "note": "poll GET /health — expect 35 collections when done",
    }


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
