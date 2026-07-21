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


# Download/parse/chunk run as a genuinely separate OS PROCESS
# (subprocess.Popen), not a background thread in this process, because that
# step is where most exceptions and heavy CPU/memory use come from (lxml
# parsing dozens of large filings) and a crash there should never take the
# API down with it. Embedding + the Qdrant write happen back in THIS
# process afterward, reusing the dense/sparse/reranker models already
# warmed at startup (see lifespan()) instead of loading a second copy —
# local Qdrant also holds an exclusive per-process file lock, so a separate
# embed+index subprocess could never open it anyway (this process already
# holds it). Two things learned the hard way getting here: (1) running the
# whole pipeline in-process took the whole API server down when downloads
# started actually succeeding and the pipeline did real work for the first
# time — a subprocess crash only ends the subprocess, so /health keeps
# responding throughout; (2) a container's memory limit is per-container,
# not per-process, so a subprocess loading its OWN copy of the embedding
# models on top of this process's already-warm copies was enough to exceed
# it before a single chunk was even processed — hence embedding is done
# here, reusing what's already loaded, rather than in the subprocess.
_ingest_lock = threading.Lock()
_ingest_running = False
_ingest_process: Optional[object] = None
_ingest_tail: List[str] = []
_ingest_exit_code: Optional[int] = None
_INGEST_TAIL_MAXLEN = 200


def _run_ingestion_subprocess() -> None:
    global _ingest_running, _ingest_process, _ingest_tail, _ingest_exit_code
    import subprocess
    import sys

    from ingestion.chunker import load_chunks
    from ingestion.embedder import index_chunks

    _ingest_tail = []
    _ingest_exit_code = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "run_ingestion.py", "--skip-index"],
            cwd=str(Path(__file__).parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _ingest_process = proc
        for line in proc.stdout:
            _ingest_tail.append(line.rstrip("\n"))
            if len(_ingest_tail) > _INGEST_TAIL_MAXLEN:
                _ingest_tail.pop(0)
        proc.wait()
        _ingest_exit_code = proc.returncode
        if proc.returncode == 0:
            logger.success("Download/parse/chunk subprocess completed — embedding + indexing here")
            chunks = load_chunks(settings.chunks_dir)
            before = len(list_collections())
            index_chunks(chunks)
            after = len(list_collections())
            _ingest_tail.append(f"[index] {after - before} new collection(s) indexed ({after} total)")
            logger.success(f"Indexed {after - before} new collection(s) ({after} total)")
        else:
            logger.error(f"Ingestion subprocess exited with code {proc.returncode}")
    except Exception as exc:
        _ingest_tail.append(f"[launcher error] {type(exc).__name__}: {exc}")
        logger.exception("Ingestion failed")
    finally:
        _ingest_process = None
        with _ingest_lock:
            _ingest_running = False


@app.get("/ingest/status", tags=["meta"])
def ingest_status(tail: int = Query(40, ge=1, le=200)):
    """Remote-diagnosis endpoint — last run's outcome when there's no log access."""
    return {
        "running": _ingest_running,
        "exit_code": _ingest_exit_code,
        "log_tail": _ingest_tail[-tail:],
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

    background_tasks.add_task(_run_ingestion_subprocess)
    return {
        "status": "started",
        "collections_loaded_now": len(list_collections()),
        "note": "poll GET /ingest/status (log tail + exit code) or GET /health (collections_loaded) — expect 35 when done",
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
        raise HTTPException(status_code=500, detail="Something went wrong while answering your question. Please try again.")

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
