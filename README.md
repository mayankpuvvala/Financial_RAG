# Financial RAG — Agentic SEC 10-K Question Answering

An agentic Retrieval-Augmented Generation system that answers precise financial questions from real SEC 10-K annual filings. Ask about revenue, R&D spend, net income trends, or segment performance — the system plans how to answer, fetches whatever data it's missing, retrieves the exact tables and passages from official filings, and cites every number.

Unlike a fixed retrieve-then-generate pipeline, the agent makes decisions at each step:

- **Plans** — classifies each query (single-company, comparison, trend, or out-of-scope) and decides what to retrieve before running any search.
- **Uses tools autonomously** — if a question names a company that isn't indexed yet, the agent resolves it against SEC's public filer registry and fetches + indexes its 10-K on the fly, mid-query, no human in the loop.
- **Verifies and self-heals** — every ticker a query touches is checked against what's actually on disk before answering; missing or stale collections (e.g. built by a since-superseded parser) are silently re-indexed rather than serving stale data.
- **Self-corrects** — if the generated answer looks like a refusal ("not found in the provided context"), the agent detects that itself and retries once with broadened retrieval before giving up.
- **Decomposes and synthesizes** — comparison and trend questions are planned into atomic sub-questions, answered independently (in parallel), and merged into one cited answer.

---

## What it does

```
"What was Apple's revenue in FY2024?"
→ Apple's total net sales for FY2024 were $391,035 million (~$391 billion). [1]
  [1] Apple Inc. (AAPL) | FY2024 | Item 8: Financial Statements

"Compare Microsoft and Google R&D spending in 2024"
→ Microsoft FY2024 R&D: $29,510M (~$29.5B) [1]
  Alphabet FY2024 R&D: $49,326M (~$49.3B) [2]

"What was Netflix's revenue in their latest 10-K?"
→ (first time asked: fetches and indexes Netflix's latest 10-K on the fly)
  Netflix's revenue in FY2025 was $45,183,036 thousand (~$45.2 billion). [1]
```

**Pre-indexed** — 35 filings (FY2023–2025) across:

| Sector | Tickers |
|--------|---------|
| Technology | AAPL · MSFT · GOOGL · AMZN |
| Banking | JPM · WFC · BAC · GS |
| Asset Management | BLK · STT · TROW · IVZ |

**Not limited to those 12** — ask about any other publicly traded US company and it's resolved via SEC's company registry, fetched, and indexed on the spot. See [On-demand ingestion](#on-demand-ingestion-any-sec-listed-company) below.

---

## Architecture

```
 User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  1. PLAN  (Groq llama-3.1-8b-instant)                       │
│  Classify → single_doc / multi_doc / temporal / out_of_scope│
│  Extract companies, years, and metric focus                 │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. VERIFY & ACT — agent decides whether it needs more data │
│  For every company the plan touches:                        │
│  ├── already indexed & current?  → proceed                  │
│  ├── unindexed / unrecognized?   → TOOL CALL: resolve via   │
│  │      SEC registry → download → parse → chunk → embed,    │
│  │      mid-query, then proceed                             │
│  └── stale / missing collection? → re-index (self-heal)     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. RETRIEVE  — Qdrant (on-disk), one collection per        │
│     ticker+year; multi_doc/temporal decomposed into         │
│     atomic per-company/year sub-questions first             │
│  ├── Dense   BAAI/bge-base-en-v1.5 (768-dim ONNX)           │
│  ├── Sparse  Qdrant/BM25                                    │
│  └── RRF fusion → cross-encoder rerank → top-3 + parent text│
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. GENERATE  (Groq llama-3.3-70b-versatile)                │
│  Grounded sub-answer(s) with [N] citations, XBRL-aware      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. SELF-CHECK  — agent inspects its own output             │
│  Looks like a refusal? → retry once, focus dropped,         │
│  retrieval widened                                          │
│  Multiple sub-answers (step 3)? → synthesize into one       │
│  answer                                                     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 Cited Answer + Sources + Query type
```

**Parent-child chunking** — child chunks (~1000 tokens) are embedded for precise retrieval; the full parent section text is passed to the LLM for rich context.

---

## Quick Start
or (try live at): https://financialrag-production-420e.up.railway.app/
### Option A — Google Colab (recommended, no GPU needed)

Open [`colab.ipynb`](colab.ipynb) — handles drive mounting, cloning, secrets, ingestion, and queries end-to-end.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayankpuvvala/Financial_RAG/blob/main/colab.ipynb)

Two secrets needed in Colab (Settings → Secrets): `groq_api` ([console.groq.com](https://console.groq.com)) and `edgar_email` (any valid email).

### Option B — Local

```bash
git clone https://github.com/mayankpuvvala/Financial_RAG.git
cd Financial_RAG

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

echo "groq_api=YOUR_GROQ_KEY"      >> .env
echo "edgar_email=you@example.com" >> .env

python run_ingestion.py    # download, parse, chunk, embed all 35 filings (~15 min)
python query.py "What was Apple revenue in FY2024?"
```

### Option C — Docker / Railway

A `Dockerfile` (CPU-only, no GPU needed) and `railway.toml` are included. `data/` must be a mounted volume — without one, the Qdrant index and downloaded filings don't survive a restart. Secrets (`groq_api`, `edgar_email`) are passed as env vars, never baked into the image.

```bash
docker build -t financial-rag .
docker run -p 8000:8000 \
  -e groq_api=YOUR_GROQ_KEY -e edgar_email=you@example.com \
  -v financial_rag_data:/app/data \
  financial-rag
```

For Railway: deploy from the GitHub repo, add a **Volume** at `/app/data`, and set `groq_api`/`edgar_email` under Variables. `$PORT` is handled automatically. A fresh volume starts empty — either run `python run_ingestion.py` once, or let [on-demand ingestion](#on-demand-ingestion-any-sec-listed-company) populate it as questions come in.

---

## Ingestion Pipeline

```
SEC EDGAR → data/raw/<TICKER>/<YEAR>/primary-document.html
    │  ingestion/parser.py   — iXBRL-aware, section boundary detection, tables → markdown
    ▼
data/parsed/<TICKER>_<YEAR>.json   (ParsedDocument)
    │  ingestion/chunker.py  — sentence-boundary text chunks + table sub-chunks
    ▼
data/chunks/<TICKER>_<YEAR>_chunks.json
    │  ingestion/embedder.py — fastembed ONNX (BAAI/bge-base-en-v1.5) + BM25
    ▼
data/qdrant/<TICKER>_<YEAR>/   (Qdrant, local on-disk)
```

`run_ingestion.py` flags: `--skip-download` (reuse existing manifest), `--skip-index` (parse + chunk only).

---

## Query Interface

```python
from query import ask

result = ask("Compare Microsoft and Google R&D spending in 2024")
print(result.answer)       # synthesized answer
print(result.citations)    # [{"company": "Microsoft", "fiscal_year": 2024, …}]
print(result.query_type)   # multi_doc | temporal | single_doc | out_of_scope
```

| Type | Example | Routing |
|------|---------|---------|
| `single_doc` | "Apple revenue FY2024?" | Verify/ingest → retrieve → generate → self-check retry |
| `multi_doc` | "Compare MSFT vs GOOGL R&D" | Verify/ingest each → decompose → retrieve each → generate → synthesize |
| `temporal` | "JPM net income 2023–2025" | Verify/ingest → decompose by year → retrieve each → generate → synthesize |
| `out_of_scope` | "Bitcoin price?" | Early exit, no retrieval |

Every one of these — including `single_doc` — goes through the verify/self-heal step first (see below); it's just a no-op when the ticker is already indexed and current.

---

## On-demand ingestion (any SEC-listed company)

Mentioning a company outside the bundled 12 (e.g. "Tesla", "NFLX") triggers `ingestion/registry.py` to resolve it against SEC's public ticker registry, then `ingestion/auto_ingest.py` downloads just its latest 10-K and indexes it — reusing the already-warm embedding models rather than reloading them. This is the agent's one real "tool call": nothing is pre-fetched, it decides mid-query that it's missing data and goes and gets it.

Try it: `python query.py "What was Netflix's revenue in their latest 10-K?"`

It's a **one-time** cost per company (a few minutes on CPU-only hardware, dominated by embedding), persisted to disk/Qdrant exactly like the bundled 12 — every later question about that company is instant. The chunks from the sections that answer the most common questions (financial statements, MD&A, risk factors, business overview) are embedded and searchable first; everything else finishes indexing in the background so the first answer doesn't wait for the whole filing. A per-ticker lock prevents duplicate work from concurrent requests, and failed lookups are cached for 5 minutes.

The same code path also runs for every ticker already in the bundled 12 on every query that touches it (`routing/resolver.py:classify_and_ensure`) — not to fetch new data, but to verify the collection actually exists and is current, and transparently re-index it if not. A cached-parse collection built by a since-superseded version of `parser.py` self-heals the next time anyone asks about it, instead of silently serving stale data forever.

---

## Configuration

All settings live in `config.py`, overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `groq_api` | *(required)* | Groq API key |
| `edgar_email` | *(required)* | Email for SEC EDGAR |
| `generation_model` | `llama-3.3-70b-versatile` | LLM for answer generation |
| `routing_model` | `llama-3.1-8b-instant` | Fast LLM for routing/decomposition |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Dense embedding model (fastembed) |
| `max_chunk_tokens` | `1000` | Max tokens per chunk |
| `retrieval_top_k` | `10` | Hybrid search candidates per collection |
| `rerank_top_k` | `3` | Final chunks after cross-encoder reranking |
| `model_cache_dir` | `None` | Set to a Drive path in Colab to persist embeddings |

---

## Evaluation with RAGAS

```bash
pip install "ragas>=0.2.0" langchain-groq langchain-huggingface
python -m evaluation.ragas_eval                              # built-in 8-question test set
python -m evaluation.ragas_eval --test-set my_set.json --output scores.json
```

Scores faithfulness, answer relevancy, context precision, and context recall (0–1 each) using the same Groq LLM as judge. Bring your own test set as a JSON list of `{"question": ..., "ground_truth": ...}`.

---

## LangGraph agent, verification, structured facts, MCP

`query.ask()` now runs a LangGraph agent (`agent/graph.py`). The original linear pipeline is kept as `ask_baseline()` for A/B measurement.

```
plan → early-exit | qa | decompose → [answer_sub × N, parallel] → synthesize → caveats

qa:  retrieve → generate → verify ──ok──────────────────────────────→ finalize
                              ├─ wrong/missing figure → inject SEC XBRL fact → generate → verify
                              │       (still wrong) ───→ deterministic answer from XBRL
                              ├─ refusal ──────────────→ broaden retrieval → generate → verify
                              └─ unsupported figure ───→ strict retry → generate → verify
```

- **`facts/xbrl.py`** — exact annual figures from SEC's `companyfacts` API (revenue, net income, operating income, R&D, assets, cash, …), taken from the fiscal year's *own* 10-K (not later restatements). Cached under `data/xbrl/`.
- **`verification/verify.py`** — checks the answer's headline figure against XBRL and every other dollar amount against the retrieved text (or as a sum/difference of stated figures). Each corrective step runs at most once.
- **`mcp_server.py`** — MCP tools: `search_filings`, `fetch_10k`, `get_xbrl_figure`, `ask_financial_question`, `list_indexed_filings` (`python mcp_server.py`, or `--http --port 8765`).
- **`llm.py`** — one place for Groq/OpenAI calls (`llm_provider=groq|openai`): retries empty replies from reasoning models and per-minute rate limits, fails fast on daily limits and exhausted billing.

**Measured** (53 questions with SEC-reported ground truth, `python -m evaluation.numeric_eval`; OpenAI `gpt-4.1`): baseline 41/53 (77%), agent 53/53 on the first run and on 3 further runs (a 5th and 6th run, cut short when the OpenAI account ran out of credit, had 3 misses — all quota errors, none wrong figures). The ground truth and the fallback share the same XBRL source, so this shows the pipeline now reliably states the SEC figure, not that every answer is independently audited. On 12 qualitative/early-exit questions (`evaluation.qualitative_eval`) both systems score the same.

Groq's free tier is limited to 8000 tokens/minute per model, roughly one answer a minute with this context size; set `llm_provider=openai` (+ `openai_api_key`) for real traffic.

---

## Project Structure

```
Financial_RAG/
├── colab.ipynb              # End-to-end Colab notebook
├── run_ingestion.py         # Bulk ingestion entry point (bundled 12)
├── query.py                 # Query entry point (CLI + library)
├── config.py / models.py    # Settings + Pydantic data models
├── Dockerfile / railway.toml
│
├── api/            app.py (FastAPI + UI), chat.py (sessions/history)
├── ui/             index.html — self-contained chat UI
│
├── ingestion/
│   ├── downloader.py        # SEC EDGAR downloader (+ EX-13 exhibit merge)
│   ├── parser.py             # HTML → ParsedDocument (iXBRL-aware)
│   ├── chunker.py / embedder.py
│   ├── registry.py           # SEC ticker/company resolver (any filer)
│   └── auto_ingest.py        # On-demand single-company ingestion
│
├── retrieval/      vector_store.py, retriever.py, reranker.py, parent_store.py
├── routing/        classifier.py, decomposer.py, resolver.py
├── generation/     generator.py, synthesizer.py
├── evaluation/     ragas_eval.py
│
└── data/                     # Gitignored except test_sets/
    ├── raw/ parsed/ chunks/ qdrant/
    ├── company_tickers.json  # SEC ticker registry cache
    └── test_sets/
```

---

## Key Technical Decisions

- **fastembed (ONNX) end-to-end — dense embedding, sparse embedding, AND reranking** — no PyTorch anywhere in the stack. The reranker uses `Xenova/ms-marco-MiniLM-L-12-v2`, an ONNX port of the same weights the original sentence-transformers cross-encoder used. This matters most on memory-capped hosts (Railway's smaller tiers, etc.) — PyTorch's own runtime footprint is large regardless of model size, so not loading a second ML framework alongside ONNX Runtime is the single biggest lever for staying under a low memory limit.
- **Hybrid search (dense + BM25)** — dense vectors handle semantic queries ("revenue growth"); BM25 handles exact terms ("total net sales", "291035").
- **Parent-child chunking** — small chunks for precise retrieval, full parent section text for LLM context.
- **Cross-encoder reranking** — a joint (query, chunk) relevance signal, sharper than bi-encoder cosine similarity alone.
- **"Incorporated by reference" filings** — some bank holding companies (e.g. Wells Fargo) file a slim 10-K that points to a separate "Annual Report to Shareholders" exhibit (EX-13) instead of including Risk Factors/MD&A/Financials inline. The downloader merges EX-13 in when present; the parser boundary-detects each embedded document separately (concatenating raw HTML into one tree causes `lxml` to drop everything after the first `</html>`, and each document needs its own independent Item-ordering pass).
- **Verify-on-every-query, not just on ingest** — the classifier's system prompt tells the model the bundled 12 are "already indexed and ready," but that's a static assumption, not a live fact (a collection can be deleted, corrupted, or stale from an old parser version). Rather than trusting the prompt, every ticker a query touches — bundled or auto-ingested — is checked against what's actually on disk before answering, and silently re-indexed if it's missing or stale. Costs one cheap collection-existence check in the common case.
- **Bounded self-correction, not a blind retry** — a refused answer ("not explicitly stated in the provided context") triggers exactly one retry with the metric focus dropped and retrieval widened, keyed off a regex over the model's own refusal phrasing. Capped at one retry so a genuinely out-of-scope query still fails fast instead of doubling LLM cost on every miss.
- **Deterministic decomposition for temporal queries** — trend questions ("how did X change from 2023 to 2025") skip the LLM decomposer entirely and build one sub-question per year directly from the classifier's `focus` field. The LLM decomposer is tuned for multi-company comparisons and was observed misapplying its "describe the business" special case to single-company metric questions, or under-delivering sub-questions for the requested year span.

---

## Requirements

- Python 3.10+
- Free [Groq API key](https://console.groq.com)
- SEC EDGAR email (any address, required by SEC fair-use policy)
- ~2 GB disk for the bundled 35 filings + vectors
- No GPU required

---

## License

MIT
