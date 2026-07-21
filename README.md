# Financial RAG — SEC 10-K Question Answering

A Retrieval-Augmented Generation system that answers precise financial questions from real SEC 10-K annual filings. Ask about revenue, R&D spend, net income trends, or segment performance — the system retrieves the exact tables and passages from official filings and cites every number.

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
│  ROUTING LAYER  (Groq llama-3.1-8b-instant)                │
│  Classifier (single_doc / multi_doc / temporal / oos)      │
│  + Decomposer for multi-doc & temporal queries              │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                            │
│  Qdrant (on-disk) — one collection per ticker+year          │
│  ├── Dense   BAAI/bge-base-en-v1.5 (768-dim ONNX)          │
│  ├── Sparse  Qdrant/BM25                                    │
│  └── RRF fusion → cross-encoder rerank → top-3 + parent text│
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATION LAYER  (Groq llama-3.3-70b-versatile)          │
│  Grounded answer with [N] citations, XBRL-aware prompt      │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 Cited Answer + Sources + Query type
```

**Parent-child chunking** — child chunks (~1000 tokens) are embedded for precise retrieval; the full parent section text is passed to the LLM for rich context.

---

## Quick Start

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
| `single_doc` | "Apple revenue FY2024?" | Direct retrieval → generation |
| `multi_doc` | "Compare MSFT vs GOOGL R&D" | Decompose → retrieve each → synthesize |
| `temporal` | "JPM net income 2023–2025" | Decompose by year → retrieve each → synthesize |
| `out_of_scope` | "Bitcoin price?" | Early exit, no retrieval |

---

## On-demand ingestion (any SEC-listed company)

Mentioning a company outside the bundled 12 (e.g. "Tesla", "NFLX") triggers `ingestion/registry.py` to resolve it against SEC's public ticker registry, then `ingestion/auto_ingest.py` downloads just its latest 10-K and indexes it — reusing the already-warm embedding models rather than reloading them.

Try it: `python query.py "What was Netflix's revenue in their latest 10-K?"`

It's a **one-time** cost per company (a few minutes on CPU-only hardware, dominated by embedding), persisted to disk/Qdrant exactly like the bundled 12 — every later question about that company is instant. The chunks from the sections that answer the most common questions (financial statements, MD&A, risk factors, business overview) are embedded and searchable first; everything else finishes indexing in the background so the first answer doesn't wait for the whole filing. A per-ticker lock prevents duplicate work from concurrent requests, and failed lookups are cached for 5 minutes.

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
