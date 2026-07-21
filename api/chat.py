"""
Chat session management — SQLite storage + conversation memory.

Endpoints:
  POST   /chat                         — send a message (creates session if needed)
  GET    /chat/sessions                — list all sessions
  GET    /chat/sessions/{sid}          — get all turns in a session
  DELETE /chat/sessions/{sid}          — delete a session + its turns
  PATCH  /chat/sessions/{sid}/turns/{tid} — mark a turn correct/incorrect
  GET    /chat/review                  — all turns for quality review (filterable)
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from config import settings
from query import ask
from routing.resolver import classify_and_ensure
from retrieval.retriever import retrieve
from generation.generator import generate_answer
from generation.synthesizer import synthesize
from models import QueryResult

# ── DB setup ──────────────────────────────────────────────────────────────────

DB_PATH = settings.data_dir / "chat_history.db"
settings.data_dir.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _init_db() -> None:
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                query_type  TEXT NOT NULL DEFAULT 'single_doc',
                citations   TEXT NOT NULL DEFAULT '[]',
                is_correct  INTEGER,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        """)


_init_db()


# ── Pipeline helper ───────────────────────────────────────────────────────────

def _run_pipeline(question: str, tickers=None, years=None) -> QueryResult:
    """Run the RAG pipeline with optional caller-supplied ticker/year filters."""
    if tickers or years:
        cls = classify_and_ensure(question)
        t = tickers or cls.tickers
        y = years   or cls.years
        if cls.query_type in ("multi_doc", "temporal"):
            return synthesize(question, t, y, cls.query_type, focus=cls.focus)
        retrieved = retrieve(question, t, y, focus=cls.focus)
        return generate_answer(question, retrieved, cls.query_type)
    return ask(question)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question:   str
    tickers:    Optional[List[str]] = None
    years:      Optional[List[int]] = None


class TurnOut(BaseModel):
    id:          str
    session_id:  str
    question:    str
    answer:      str
    query_type:  str
    citations:   list
    is_correct:  Optional[bool] = None
    created_at:  str


class SessionOut(BaseModel):
    id:          str
    title:       str
    created_at:  str
    updated_at:  str
    turn_count:  int = 0


class ReviewPatch(BaseModel):
    is_correct: Optional[bool] = None


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=TurnOut)
def chat(req: ChatRequest):
    """Send a message; creates a new session when session_id is omitted."""
    if not req.question.strip():
        raise HTTPException(400, "question cannot be empty")

    now = datetime.now(timezone.utc).isoformat()
    sid = req.session_id

    if sid:
        with _conn() as con:
            if not con.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
                raise HTTPException(404, f"Session {sid!r} not found")
    else:
        sid = str(uuid.uuid4())
        title = req.question[:60] + ("…" if len(req.question) > 60 else "")
        with _conn() as con:
            con.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, title, now, now))

    # Fetch last 3 turns as memory context
    with _conn() as con:
        prior = con.execute(
            "SELECT question, answer FROM turns WHERE session_id=? ORDER BY created_at DESC LIMIT 3",
            (sid,),
        ).fetchall()

    # Build question with conversation context so the classifier/LLM can resolve
    # follow-up references like "this", "compare to last year", etc.
    question = req.question
    if prior:
        ctx = "\n---\n".join(
            f"Q: {r['question']}\nA: {r['answer'][:400]}" for r in reversed(prior)
        )
        question = f"[Conversation history]\n{ctx}\n\n[Current question]\n{req.question}"

    try:
        result = _run_pipeline(question, req.tickers, req.years)
    except Exception as exc:
        logger.exception("Chat pipeline error")
        raise HTTPException(500, str(exc))

    tid = str(uuid.uuid4())
    with _conn() as con:
        con.execute(
            "INSERT INTO turns VALUES (?,?,?,?,?,?,?,?)",
            (tid, sid, req.question, result.answer,
             result.query_type, json.dumps(result.citations), None, now),
        )
        con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))

    return TurnOut(
        id=tid, session_id=sid,
        question=req.question, answer=result.answer,
        query_type=result.query_type,
        citations=result.citations,
        is_correct=None, created_at=now,
    )


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions():
    with _conn() as con:
        rows = con.execute("""
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(t.id) AS turn_count
            FROM   sessions s
            LEFT JOIN turns t ON t.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """).fetchall()
    return [SessionOut(**dict(r)) for r in rows]


@router.get("/sessions/{sid}", response_model=List[TurnOut])
def get_session(sid: str):
    with _conn() as con:
        if not con.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone():
            raise HTTPException(404, "Session not found")
        rows = con.execute(
            "SELECT * FROM turns WHERE session_id=? ORDER BY created_at", (sid,)
        ).fetchall()

    def _to_out(r):
        d = dict(r)
        d["citations"]  = json.loads(d["citations"])
        d["is_correct"] = bool(d["is_correct"]) if d["is_correct"] is not None else None
        return TurnOut(**d)

    return [_to_out(r) for r in rows]


@router.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: str):
    with _conn() as con:
        con.execute("DELETE FROM turns   WHERE session_id=?", (sid,))
        con.execute("DELETE FROM sessions WHERE id=?",        (sid,))


@router.patch("/sessions/{sid}/turns/{tid}", response_model=TurnOut)
def review_turn(sid: str, tid: str, body: ReviewPatch):
    """Mark a turn as correct (true), incorrect (false), or unreviewed (null)."""
    val = None if body.is_correct is None else int(body.is_correct)
    with _conn() as con:
        if not con.execute("SELECT 1 FROM turns WHERE id=? AND session_id=?", (tid, sid)).fetchone():
            raise HTTPException(404, "Turn not found")
        con.execute("UPDATE turns SET is_correct=? WHERE id=?", (val, tid))
        row = con.execute("SELECT * FROM turns WHERE id=?", (tid,)).fetchone()
    d = dict(row)
    d["citations"]  = json.loads(d["citations"])
    d["is_correct"] = bool(d["is_correct"]) if d["is_correct"] is not None else None
    return TurnOut(**d)


@router.get("/review")
def review_all(filter: Optional[str] = Query(None)):
    """
    Return all turns for review.
    ?filter=correct | incorrect | unreviewed   (omit for all)
    """
    where = {
        "correct":    "WHERE t.is_correct = 1",
        "incorrect":  "WHERE t.is_correct = 0",
        "unreviewed": "WHERE t.is_correct IS NULL",
    }.get(filter or "", "")

    with _conn() as con:
        rows = con.execute(f"""
            SELECT t.*, s.title AS session_title
            FROM   turns t
            JOIN   sessions s ON s.id = t.session_id
            {where}
            ORDER BY t.created_at DESC
        """).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["citations"]  = json.loads(d["citations"])
        d["is_correct"] = bool(d["is_correct"]) if d["is_correct"] is not None else None
        result.append(d)
    return result
