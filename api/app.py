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

import os
import shutil
import tarfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from config import settings
from query import ask
from retrieval.vector_store import list_collections
from api.chat import router as chat_router

_UI_FILE = Path(__file__).parent.parent / "ui" / "index.html"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
#
# Deliberately NOT eagerly warming models here. This used to call
# encode_query("warm up") + _get_reranker() at startup so the first real
# query wouldn't pay the model-load cost — but that leaves the dense/
# sparse/reranker models (several hundred MB combined) resident from the
# moment the container starts, for the container's entire lifetime. On
# Railway's memory-capped tier that baseline was apparently already close
# enough to the limit that ingestion's download/parse/chunk/embed work —
# even with every other fix applied (no subprocess, no forking, no
# duplicate model copies, sequential parsing) — still crashed the whole
# container within seconds, every time. Lazy-loading instead (see
# ingestion/embedder.py's _get_dense()/_get_sparse(), retrieval/reranker.py's
# _get_reranker(), all cached-singleton) means a fresh deploy has just
# uvicorn/FastAPI/Python's own footprint until something actually needs a
# model — giving ingestion real headroom on an otherwise-idle container.
# The trade-off is a slower first real query (pays the load cost inline);
# ingestion's own embed step loads them anyway, so any query after a
# completed ingestion run pays nothing extra.

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    # list_collections() -> get_client() can raise (bounded timeout — see
    # vector_store.get_client()) if data/qdrant has a corrupted collection.
    # /health must still respond in that case: it's the one endpoint that
    # needs to answer regardless of Qdrant's state, both for the platform's
    # own health checks and for diagnosing exactly this failure remotely.
    try:
        cols = list_collections()
        return {
            "status": "ok",
            "collections_loaded": len(cols),
            "collections": cols,
        }
    except Exception as exc:
        logger.exception("Health check: Qdrant unavailable")
        return {
            "status": "degraded",
            "error": f"{type(exc).__name__}: {exc}",
            "collections_loaded": 0,
            "collections": [],
        }


@app.get("/collections", tags=["meta"])
def collections():
    return {"collections": list_collections()}


# Runs in-process, as a plain background task — not a subprocess.
# subprocess.Popen was tried and abandoned: our call (cwd set, close_fds
# left at its POSIX default of True) is disqualified from Python's
# posix_spawn fast path, which requires cwd=None AND close_fds=False, so it
# always falls back to fork()+exec(). fork() from a process that already
# holds the dense/sparse/reranker models warm (several hundred MB) forces
# the kernel to momentarily account for a full copy of that memory — and on
# Railway's cgroup-limited container, every single subprocess attempt (four
# different redesigns, including ones with zero extra work happening in the
# child) killed the WHOLE container within seconds, wiping this process's
# own state too. That's the signature of a fork-time kill, not anything the
# child was actually doing. Running in-process removes the fork entirely;
# the trade-off is a crash mid-pipeline can take the API down with it, same
# as before subprocess isolation was tried — but every step here already
# skips already-done work (see each ingestion/*.py step), so a retry after
# Railway's restartPolicy brings the container back just resumes.
_ingest_lock = threading.Lock()
_ingest_running = False
_ingest_tail: List[str] = []
_ingest_exit_code: Optional[int] = None
_INGEST_TAIL_MAXLEN = 200

# Persisted alongside the Qdrant data on the same volume, so /ingest/status
# still reports the last known outcome after a restart — even a hard SIGKILL
# (OOM-kill) skips every except/finally block in this process, so in-memory
# globals alone go blank on every restart with no way to tell "never ran"
# apart from "was killed mid-run". Written after each company, not just at
# the end, so a mid-run kill still leaves a useful trail.
_INGEST_STATUS_FILE = settings.data_dir / "ingest_status.json"


def _save_ingest_status() -> None:
    import json
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _INGEST_STATUS_FILE.write_text(json.dumps({
            "running":   _ingest_running,
            "exit_code": _ingest_exit_code,
            "log_tail":  _ingest_tail[-_INGEST_TAIL_MAXLEN:],
        }), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not persist ingest status: {exc}")


def _load_ingest_status() -> None:
    global _ingest_running, _ingest_tail, _ingest_exit_code
    import json
    if not _INGEST_STATUS_FILE.exists():
        return
    try:
        data = json.loads(_INGEST_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    # A restart always means nothing is actually running anymore, whatever
    # the last-written file said (it may have been mid-run when killed).
    _ingest_running = False
    _ingest_exit_code = data.get("exit_code")
    _ingest_tail = data.get("log_tail", [])
    if _ingest_exit_code is None and _ingest_tail:
        _ingest_tail.append("[restart] container restarted while ingestion was running — resume with POST /ingest")


_load_ingest_status()


def _run_ingestion_background() -> None:
    """
    One company at a time (download → parse → chunk → embed → upsert),
    rather than the whole bundled-12 in one pass. Two independent reasons:

    1. Peak memory. embedder.index_chunks() joins every chunk's text across
       every collection into one array before calling the ONNX models —
       fine for a handful of tickers, but WFC/BLK-style content-heavy
       filers alone produce thousands of table chunks each; all 12
       companies' chunks held in memory simultaneously plus the embedding
       output arrays is a very different memory profile than one company
       at a time. On Railway's memory-capped tier that's a plausible OOM
       kill — which would explain /ingest/status staying at
       exit_code=null indefinitely: a hard kill skips the except/finally
       below entirely, so nothing ever gets the chance to record failure.
    2. Resumability. Each company's Qdrant collections are committed
       before moving to the next, so a kill mid-run (whatever the cause)
       loses at most the one in-flight company's work — the next
       POST /ingest call resumes from there via each step's existing
       skip-if-already-done logic, instead of re-doing everything.

    Also prunes data/raw and data/chunks for a company right after its
    embeddings land in Qdrant. The running server never reads either
    directory again afterward — data/raw (source HTML) is only needed to
    parse, data/chunks only to embed, while data/qdrant (the index) and
    data/parsed (parent_store's full-section context, loaded at query time)
    are the only things actually needed to serve queries. On a small
    Railway volume (e.g. 500MB) the raw HTML alone is easily 1.5GB+ across
    12 companies, so keeping it around after it's served its purpose would
    fill the volume long before ingestion finishes. This pruning is
    deliberately scoped to THIS remote/deployment path only — the local
    run_ingestion.py dev workflow keeps everything on disk for iterating.
    """
    global _ingest_running, _ingest_tail, _ingest_exit_code
    from config import COMPANIES
    from ingestion.downloader import download_all_filings
    from ingestion.parser import parse_all_filings
    from ingestion.chunker import chunk_all_documents
    from ingestion.embedder import index_chunks

    _ingest_tail = []
    _ingest_exit_code = None
    _save_ingest_status()

    any_failed = False
    for company in COMPANIES:
        ticker = company["ticker"]

        if any(c.rsplit("_", 1)[0] == ticker for c in list_collections()):
            _ingest_tail.append(f"[{ticker}] already indexed — skipping")
            _save_ingest_status()
            continue

        try:
            manifest = download_all_filings(
                companies=[company], filing_type=settings.filing_type,
                limit=settings.filings_per_company, raw_dir=settings.raw_dir,
            )
            documents = parse_all_filings(manifest=manifest, parsed_dir=settings.parsed_dir)
            chunks = chunk_all_documents(documents=documents, chunks_dir=settings.chunks_dir)

            before = len(list_collections())
            index_chunks(chunks)
            after = len(list_collections())

            _ingest_tail.append(
                f"[{ticker}] {len(manifest)} filing(s), {len(chunks)} chunk(s), "
                f"{after - before} new collection(s) ({after} total)"
            )
            logger.success(f"Ingestion — {ticker} done ({after} collections total)")

            raw_ticker_dir = settings.raw_dir / "sec-edgar-filings" / ticker
            shutil.rmtree(raw_ticker_dir, ignore_errors=True)
            for doc in documents:
                chunk_file = settings.chunks_dir / f"{doc.ticker}_{doc.fiscal_year}_chunks.json"
                chunk_file.unlink(missing_ok=True)
        except Exception as exc:
            any_failed = True
            _ingest_tail.append(f"[{ticker}] {type(exc).__name__}: {exc}")
            logger.exception(f"Ingestion failed for {ticker}")
        _save_ingest_status()

    _ingest_exit_code = 1 if any_failed else 0
    _ingest_tail.append(f"[done] exit_code={_ingest_exit_code}")
    with _ingest_lock:
        _ingest_running = False
    _save_ingest_status()


@app.get("/ingest/status", tags=["meta"])
def ingest_status(tail: int = Query(40, ge=1, le=200)):
    """Remote-diagnosis endpoint — last run's outcome when there's no log access."""
    return {
        "running": _ingest_running,
        "exit_code": _ingest_exit_code,
        "log_tail": _ingest_tail[-tail:],
        "collections_loaded": len(list_collections()),
    }


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> int:
    """
    Extract a tar archive into dest, refusing any member whose resolved path
    would land outside dest (a malicious ../../ entry — "tar-slip"). Returns
    the number of members extracted.

    Validates and extracts one member at a time in a single forward pass —
    NOT tar.getmembers() followed by tar.extractall(), which reads the
    stream fully to build the member list and then needs to seek back to
    the start to actually extract; the request's upload stream (mode
    "r|gz") doesn't support that ("seeking backwards is not allowed").

    Skips any qdrant/.lock entry. That file is local-mode Qdrant's own
    exclusive-lock marker for the CURRENT process's storage handle, not
    portable data — a naively-built archive (`tar -czf x.tar.gz -C data
    qdrant parsed`) captures it, and overwriting a lock file this process
    itself already holds open fails hard on Windows (PermissionError) and
    is pointless everywhere else (the running process's actual lock isn't
    affected by replacing the file's bytes anyway).
    """
    dest = dest.resolve()
    count = 0
    for member in tar:
        if Path(member.name).name == ".lock":
            continue
        resolved = (dest / member.name).resolve()
        if resolved != dest and dest not in resolved.parents:
            raise ValueError(f"Refusing unsafe archive member: {member.name!r}")
        tar.extract(member, dest)
        count += 1
    return count


@app.post("/admin/restore-data", tags=["meta"])
def restore_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    token: Optional[str] = Query(None),
):
    """
    Seed the volume from a pre-built local index instead of re-running
    ingestion on Railway's constrained CPU (which, at observed local
    embedding speeds, can take hours per company — see _run_ingestion_background).

    Upload a .tar.gz containing two top-level entries, `qdrant/` and
    `parsed/`, matching data/qdrant and data/parsed exactly (build it with
    `tar -czf restore.tar.gz -C data qdrant parsed` locally, where data/
    already has verified, working collections). Anything else in data/
    (raw/, chunks/) is ingestion-only scratch space the running server
    never reads — no need to include it.

    The archive is extracted directly onto the volume, then the process
    exits so Railway's restart policy brings up a fresh instance. A clean
    Python-level restart is required, not optional: local-mode Qdrant's
    client caches its collection registry for the life of the process (see
    retrieval/vector_store.get_client()), so a process that already has an
    (empty) client open would never notice files dropped in underneath it.
    """
    if settings.admin_token and token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    if not (file.filename or "").endswith((".tar.gz", ".tgz")):
        raise HTTPException(status_code=400, detail="Expected a .tar.gz/.tgz archive")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=file.file, mode="r|gz") as tar:
            n = _safe_extract(tar, settings.data_dir)
    except (tarfile.TarError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Bad archive: {exc}")
    except Exception as exc:
        # Deliberately broad and detailed (unlike every other endpoint's
        # generic error message): this is admin_token-gated, remote-only
        # diagnosis for exactly this feature — a first attempt failed with
        # a bare "Internal Server Error" and no way to see why without
        # Railway CLI/log access, so the real exception needs to reach the
        # response body to be debuggable at all.
        logger.exception("restore-data extraction failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    logger.success(f"Restored {n} file(s) from uploaded archive — restarting to pick them up")

    def _restart_soon() -> None:
        time.sleep(1)   # let the HTTP response flush before the process dies
        os._exit(0)

    background_tasks.add_task(_restart_soon)
    return {
        "status": "extracted",
        "members_extracted": n,
        "note": "process is restarting now — poll GET /health in ~10-30s for the new collection count",
    }


@app.delete("/admin/data-path", tags=["meta"])
def delete_data_path(
    background_tasks: BackgroundTasks,
    path: str = Query(..., description="Path relative to data/, e.g. qdrant/collection/WFC_2023"),
    token: Optional[str] = Query(None),
    restart: bool = Query(True, description="Restart the process after deleting (recommended if Qdrant is wedged)"),
):
    """
    Delete a specific file or directory under data/ directly on disk — no
    QdrantClient involved. Exists for exactly one scenario: a corrupted or
    partially-written collection (e.g. from an interrupted restore-data
    extraction) makes get_client() hang past its timeout, which means
    EVERY endpoint that touches Qdrant — including delete_collection() —
    hangs too, since they all need a working client first. Operating on
    the filesystem directly sidesteps that entirely. Defaults to
    restarting afterward since a wedged in-memory client (if one was ever
    successfully constructed before the timeout) won't un-wedge itself.
    """
    if settings.admin_token and token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid or missing token")

    target = (settings.data_dir / path).resolve()
    data_dir_resolved = settings.data_dir.resolve()
    if target != data_dir_resolved and data_dir_resolved not in target.parents:
        raise HTTPException(status_code=400, detail="Path escapes data/ — refusing")

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"No such path: {path}")

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    logger.warning(f"Deleted data path via admin endpoint: {target}")

    if restart:
        def _restart_soon() -> None:
            time.sleep(1)
            os._exit(0)
        background_tasks.add_task(_restart_soon)

    return {"status": "deleted", "path": path, "restarting": restart}


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
