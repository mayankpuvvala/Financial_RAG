# Financial RAG — SEC 10-K Question Answering

A production-quality Retrieval-Augmented Generation system that answers precise financial questions from real SEC 10-K annual filings. Ask anything about revenue, R&D spend, net income trends, or segment performance — the system retrieves the exact tables and passages from official filings and cites every number.

---

## What it does

```
"What was Apple's revenue in FY2024?"
→ Apple's total net sales for FY2024 were $391,035 million (~$391 billion). [1]
  [1] Apple Inc. (AAPL) | FY2024 | Item 8: Financial Statements

"Compare Microsoft and Google R&D spending in 2024"
→ Microsoft FY2024 R&D: $29,510M (~$29.5B) [1]
  Alphabet FY2024 R&D: $49,326M (~$49.3B) [2]

"How did JPMorgan net income trend from 2023 to 2025?"
→ FY2023: $49,552M | FY2024: $58,471M | FY2025: …
```

**Covered companies** — 35 filings (FY2023, FY2024, FY2025):

| Sector | Tickers |
|--------|---------|
| Technology | AAPL · MSFT · GOOGL · AMZN |
| Banking | JPM · WFC · BAC · GS |
| Asset Management | BLK · STT · TROW · IVZ |

---

## Architecture

```
 User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  ROUTING LAYER  (Groq llama-3.1-8b-instant)                │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │  Classifier  │  │   Decomposer   │  │  Out-of-scope  │  │
│  │  single_doc  │  │  multi / temp  │  │  early exit    │  │
│  └──────────────┘  └────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                            │
│                                                             │
│  Qdrant (on-disk)  ←  one collection per ticker+year       │
│  ├── Dense search   BAAI/bge-base-en-v1.5 (768-dim ONNX)  │
│  ├── Sparse search  Qdrant/BM25                            │
│  └── RRF fusion    top-20 hybrid candidates                │
│                                                             │
│  Cross-encoder reranker  ms-marco-MiniLM-L-12-v2           │
│  └── top-3 final chunks + parent section text              │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATION LAYER  (Groq llama-3.3-70b-versatile)          │
│  ├── Grounded answer with [N] citations                     │
│  └── XBRL-aware prompt (handles noisy SEC table format)    │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
 Cited Answer  +  Source list  +  Query type
```

**Parent-child chunking** — child chunks (~1 000 tokens) are embedded for precise retrieval; the full parent section text is passed to the LLM for rich context.

---

## Quick Start

### Option A — Google Colab (recommended, no GPU needed)

Open [`colab.ipynb`](colab.ipynb) — it handles drive mounting, cloning, secret injection, ingestion, and queries end-to-end.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayankpuvvala/Financial_RAG/blob/main/colab.ipynb)

You need two secrets in Colab (Settings → Secrets):
- `groq_api` — get a free key at [console.groq.com](https://console.groq.com)
- `edgar_email` — any valid email for SEC EDGAR

### Option B — Local

```bash
git clone https://github.com/mayankpuvvala/Financial_RAG.git
cd Financial_RAG

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create .env
echo "groq_api=YOUR_GROQ_KEY"   >> .env
echo "edgar_email=you@example.com" >> .env

# Download, parse, chunk, embed all 35 filings (~15 min first run)
python run_ingestion.py

# Ask a question
python query.py "What was Apple revenue in FY2024?"
```

---

## Ingestion Pipeline

```
SEC EDGAR
    │  (sec-edgar-downloader — respects rate limits)
    ▼
data/raw/<TICKER>/<YEAR>/primary-document.html
    │
    │  ingestion/parser.py
    │  • iXBRL-aware BeautifulSoup parser
    │  • section boundary detection (Item 1, Item 7 MD&A, Item 8 FS, …)
    │  • table extraction → pandas → markdown
    ▼
data/parsed/<TICKER>_<YEAR>_parsed.json   (ParsedDocument)
    │
    │  ingestion/chunker.py
    │  • text  → sentence-boundary chunks (1 000 tok, 2-sentence overlap)
    │  • table → sub-chunks with context header + repeated year rows
    ▼
data/chunks/<TICKER>_<YEAR>_chunks.json   (List[Chunk])
    │
    │  ingestion/embedder.py
    │  • fastembed ONNX (no PyTorch GPU needed) — BAAI/bge-base-en-v1.5
    │  • BM25 sparse vectors via Qdrant/bm25
    ▼
data/qdrant/<TICKER>_<YEAR>/              (Qdrant local on-disk)
```

Flags for `run_ingestion.py`:

| Flag | Effect |
|------|--------|
| *(none)* | Full pipeline: download → parse → chunk → embed |
| `--skip-download` | Reuse existing manifest (already downloaded) |
| `--skip-index` | Parse + chunk only, skip Qdrant indexing |

---

## Query Interface

```python
from query import ask

result = ask("Compare Microsoft and Google R&D spending in 2024")

print(result.answer)          # synthesized answer
print(result.citations)       # [{"company": "Microsoft", "fiscal_year": 2024, …}]
print(result.query_type)      # multi_doc | temporal | single_doc | out_of_scope
```

### Query types

| Type | Example | Routing |
|------|---------|---------|
| `single_doc` | "Apple revenue FY2024?" | Direct retrieval → generation |
| `multi_doc` | "Compare MSFT vs GOOGL R&D" | Decompose → retrieve each → synthesize |
| `temporal` | "JPM net income 2023–2025" | Decompose by year → retrieve each → synthesize |
| `out_of_scope` | "Bitcoin price?" | Early exit, no retrieval |

---

## Configuration

All settings live in `config.py` and can be overridden via `.env`:

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
| `model_cache_dir` | `None` | Set to Drive path in Colab to persist embeddings |

---

## Evaluation with RAGAS

The system ships with a RAGAS evaluation harness that measures four RAG quality dimensions using the same Groq LLM as a judge.

### Install evaluation extras

```bash
pip install "ragas>=0.2.0" langchain-groq langchain-huggingface
```

### Run evaluation

```bash
# Evaluate the built-in 8-question financial test set
python -m evaluation.ragas_eval

# Save detailed per-question scores
python -m evaluation.ragas_eval --output results/ragas_scores.json

# Quick smoke test — first 3 questions only
python -m evaluation.ragas_eval --limit 3
```

### Metrics explained

| Metric | Range | What it measures |
|--------|-------|-----------------|
| **faithfulness** | 0–1 | Every claim in the answer is grounded in retrieved context (no hallucination). High = trustworthy. |
| **answer_relevancy** | 0–1 | The answer actually addresses the question asked. High = on-topic. |
| **context_precision** | 0–1 | Retrieved chunks are signal, not noise. High = precise retrieval. |
| **context_recall** | 0–1 | Key facts from the ground truth appear in the retrieved context. High = complete retrieval. |

### Bring your own test set

Create a JSON file and pass it with `--test-set`:

```json
[
  {
    "question": "What was Apple iPhone revenue in FY2024?",
    "ground_truth": "Apple's iPhone revenue in FY2024 was $201,183 million."
  },
  {
    "question": "What were Goldman Sachs total assets in FY2024?",
    "ground_truth": "Goldman Sachs total assets as of December 31, 2024 were approximately $1.68 trillion."
  }
]
```

```bash
python -m evaluation.ragas_eval --test-set my_test_set.json --output scores.json
```

### Sample output

```
==================================================
  RAGAS Evaluation Results
==================================================
  faithfulness           0.9250  [##################..]
  answer_relevancy       0.9100  [##################..]
  context_precision      0.8750  [#################...]
  context_recall         0.8500  [#################...]
--------------------------------------------------
  Average                0.8900
==================================================
```

---

## Project Structure

```
Financial_RAG/
├── colab.ipynb              # End-to-end Colab notebook
├── run_ingestion.py         # Ingestion pipeline entry point
├── query.py                 # Query entry point (CLI + library)
├── config.py                # Settings (pydantic-settings)
├── models.py                # Pydantic data models
├── requirements.txt         # All dependencies
│
├── ingestion/
│   ├── downloader.py        # SEC EDGAR downloader
│   ├── parser.py            # HTML → ParsedDocument (iXBRL-aware)
│   ├── chunker.py           # Hierarchical chunker (text + tables)
│   └── embedder.py          # fastembed ONNX + Qdrant indexer
│
├── retrieval/
│   ├── vector_store.py      # Qdrant client + hybrid search
│   ├── retriever.py         # Full retrieval pipeline
│   ├── reranker.py          # Cross-encoder reranking
│   └── parent_store.py      # Parent section text lookup
│
├── routing/
│   ├── classifier.py        # Query type classifier (Groq)
│   └── decomposer.py        # Sub-question decomposer (Groq)
│
├── generation/
│   ├── generator.py         # Answer generation (Groq)
│   └── synthesizer.py       # Multi-doc / temporal synthesis
│
├── evaluation/
│   └── ragas_eval.py        # RAGAS evaluation harness
│
└── data/                    # Auto-generated, gitignored
    ├── raw/                 # Downloaded HTML filings
    ├── parsed/              # Structured JSON documents
    ├── chunks/              # Chunked text for embedding
    ├── qdrant/              # Vector store (on-disk)
    └── test_sets/           # Evaluation datasets (tracked)
```

---

## Key Technical Decisions

**Why fastembed (ONNX) instead of sentence-transformers?**
No PyTorch required — runs on CPU-only Colab free tier without loading GPU drivers. Embedding speed is comparable.

**Why hybrid search (dense + BM25)?**
Financial queries mix semantic intent ("revenue growth") with exact terms ("total net sales", "291035"). Dense vectors handle semantics; BM25 handles exact financial terminology. RRF fusion combines both.

**Why parent-child chunking?**
Child chunks (≤1 000 tokens) are small enough for precise retrieval. But financial tables often span multiple pages. Fetching the full parent section gives the LLM the complete table context without inflating the index.

**Why a cross-encoder reranker?**
Bi-encoder retrieval (cosine similarity) is fast but imprecise. The cross-encoder sees (query, chunk) jointly and produces a much sharper relevance signal — especially important for XBRL tables where many chunks look superficially similar.

**XBRL table handling**
SEC filings use inline XBRL which creates markdown tables with numeric column indices, duplicate cells, and year labels appearing as data rows rather than headers. The system handles this via:
- Context headers prepended to every table chunk
- Year rows repeated in every sub-chunk of split tables
- LLM system prompt that explains XBRL formatting conventions

---

## Requirements

- Python 3.10+
- Free [Groq API key](https://console.groq.com) (llama-3.3-70b-versatile)
- SEC EDGAR email (any address, required by SEC fair-use policy)
- ~2 GB disk for all 35 filings + vectors
- No GPU required (fastembed ONNX runs on CPU)

---

## License

MIT
