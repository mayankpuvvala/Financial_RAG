# Financial RAG — Verified SEC 10-K Question Answering

An agentic RAG system that answers precise financial questions from real SEC 10-K filings — and **checks its own numbers against what the company reported to the SEC** before it answers. Ask about revenue, R&D, net income, trends, segments, risks or legal proceedings; every figure is cited, and headline figures are verified against SEC XBRL data.

**Live demo:** https://financialrag-production-420e.up.railway.app/

| Measured on 53 SEC-grounded questions | Original pipeline | This agent |
|---|---|---|
| Correct headline figure | 41 / 53 (77%) | 53 / 53 (100%) on the first run and on 3 further runs |

See [Evaluation](#evaluation) for how that was measured and what it does and doesn't prove.

---

## What it does

```
"What was Apple's revenue in FY2024?"
→ Apple's total net sales for FY2024 were $391,035 million (~$391 billion). [1]
  [1] Apple Inc. (AAPL) | FY2024 | Item 8: Financial Statements

"Compare Microsoft and Google R&D spending in 2024"
→ a cited side-by-side table: Microsoft $29,510M, Alphabet $49,326M

"How did JPMorgan's net income trend from 2023 to 2025?"
→ one sub-question per year, answered in parallel, merged into one cited trend

"What was Netflix's revenue in their latest 10-K?"
→ (first time asked: fetches and indexes Netflix's latest 10-K on the fly)
```

The agent:

- **Plans** — classifies each query (single company, comparison, trend, out of scope) and extracts companies, years and the metric it is about.
- **Fetches what's missing** — a company that isn't indexed is resolved via SEC's registry and its 10-K is downloaded, parsed and indexed mid-query.
- **Verifies** — the answer's headline figure is compared with the figure SEC XBRL reports for that company and fiscal year; every other dollar amount must appear in the retrieved text (or be a sum/difference of stated figures).
- **Corrects itself, once per failure mode** — see the graph below. Worst case it answers directly from the SEC data with a citation.
- **Decomposes** comparison and trend questions into per-company/per-year sub-questions, answered in parallel.

**Coverage** — 12 bundled companies, FY2023–2025 (AAPL · MSFT · GOOGL · AMZN · JPM · WFC · BAC · GS · BLK · STT · TROW · IVZ), plus any other SEC-listed company on demand (the live demo has also indexed e.g. NVDA, TSLA, NFLX, META, V, COST).

---

## How it works

`query.ask()` runs a [LangGraph](https://langchain-ai.github.io/langgraph/) agent ([agent/graph.py](agent/graph.py)).

```
plan ──► early exit (out of scope · unknown company · no company named · LLM outage)
     ├─► qa                                   (one company / year)
     └─► decompose ─► answer_sub × N (parallel, each runs `qa`) ─► synthesize
                                         │
                                         ▼
                                      caveats  (say so if some named companies couldn't be found)

qa:  retrieve ─► generate ─► verify ──ok─────────────────────────────────► finalize
                               │
                               ├─ headline figure wrong/missing ─► inject SEC XBRL fact ─► generate ─► verify
                               │                 (still wrong) ─► answer deterministically from XBRL
                               ├─ refusal ("not found") ────────► broaden retrieval ─► generate ─► verify
                               └─ unsupported figure ───────────► strict retry ─► generate ─► verify
```

Each corrective edge is guarded by a flag in the graph state, so every step runs at most once — bounded cost, no loops. The original linear pipeline is kept as `ask_baseline()` so improvements can be measured rather than asserted.

### Retrieval

- **Hybrid search** — dense (`BAAI/bge-base-en-v1.5`, ONNX) + sparse BM25, fused with RRF, in Qdrant (one collection per ticker + fiscal year).
- **Cross-encoder rerank** (`Xenova/ms-marco-MiniLM-L-12-v2`, ONNX) with metric-aware boosts so the consolidated income-statement row beats footnote mentions.
- **Parent–child chunking** — ~1000-token child chunks are embedded; the full parent section goes to the LLM.

### Structured facts (SEC XBRL) — [facts/xbrl.py](facts/xbrl.py)

Reading a number out of a noisy markdown table is where LLMs go wrong (wrong column, year or scale). The same filing's iXBRL tags are published by SEC as clean facts, so for revenue, net income, operating income, R&D, assets, cash and a few more, the agent knows the exact reported value.

- Taken from the **fiscal year's own 10-K** — a later filing's restated comparative (e.g. Bank of America's recast FY2024 net income) is not used, so the answer never contradicts the document it cites.
- "Net income" means the headline figure *attributable to the company* (the basis for EPS), not including noncontrolling interests.
- Banks and asset managers use different revenue tags; the module tries the known ones and takes the largest total.

### Verification — [verification/verify.py](verification/verify.py)

Two checks: the **key figure** must match XBRL (rounding-aware, e.g. `$391,035 million` = `~$391.0 billion`), and every **other** dollar amount must be supported by the retrieved text at any of $/$K/$M scale, or be a difference/sum of other stated figures. Percentages and per-share values aren't checked. It also normalises gpt-oss quirks (`391 035` with a narrow no-break space).

### LLM layer — [llm.py](llm.py)

One entry point for routing, generation and synthesis, switchable between **Groq** (default) and **OpenAI** with `llm_provider`. It exists because of three real failure modes:

1. Groq's replacement models (`openai/gpt-oss-20b` / `-120b`) are *reasoning* models; some prompts burn the whole token budget thinking and return an **empty** reply. An empty reply is retried with a doubled budget.
2. Per-minute rate limits are retried after the wait the API asks for; **daily limits and exhausted billing fail fast** instead of being retried.
3. Citation markers like `【1】` are normalised to `[1]`.

If the classification call itself fails, the user is told the service is unavailable — not asked "which company?".

---

## Quick start

### Option A — Google Colab (no GPU needed)

Open [`colab.ipynb`](colab.ipynb) — handles drive mounting, cloning, secrets, ingestion and queries.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayankpuvvala/Financial_RAG/blob/main/colab.ipynb)

Secrets needed (Settings → Secrets): `groq_api` ([console.groq.com](https://console.groq.com)) and `edgar_email` (any valid email).

### Option B — Local

```bash
git clone https://github.com/mayankpuvvala/Financial_RAG.git
cd Financial_RAG

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

echo "groq_api=YOUR_GROQ_KEY"      >> .env
echo "edgar_email=you@example.com" >> .env

python run_ingestion.py    # download, parse, chunk, embed the bundled filings (~15 min)
python query.py "What was Apple revenue in FY2024?"
```

### Option C — Docker / Railway

A `Dockerfile` (CPU-only) and `railway.toml` are included. `data/` must be a mounted volume — otherwise the index, downloaded filings and the XBRL cache don't survive a restart. Secrets are env vars, never baked into the image.

```bash
docker build -t financial-rag .
docker run -p 8000:8000 \
  -e groq_api=YOUR_GROQ_KEY -e edgar_email=you@example.com \
  -v financial_rag_data:/app/data \
  financial-rag
```

For Railway: deploy from the GitHub repo, add a **Volume** at `/app/data`, set `groq_api` / `edgar_email` under Variables. `$PORT` is handled automatically. A fresh volume starts empty — run `python run_ingestion.py` once, or let [on-demand ingestion](#on-demand-ingestion) fill it as questions arrive.

### Groq free tier vs OpenAI

Groq's free tier allows **8,000 tokens/minute per model**, and one answer here sends roughly 6–8k tokens — about **one answer a minute**. That's fine for a demo and a bottleneck under real use. For real traffic set `llm_provider=openai` and `openai_api_key` (defaults to `gpt-4.1` for generation and `gpt-4.1-mini` for routing), or use a paid Groq tier.

---

## Using it

### Python

```python
from query import ask

result = ask("Compare Microsoft and Google R&D spending in 2024")
print(result.answer)         # synthesized, cited answer
print(result.citations)      # [{"index": 1, "company": ..., "fiscal_year": 2024, "section": ...}]
print(result.query_type)     # single_doc | multi_doc | temporal | out_of_scope | ...
print(result.verification)   # {"status": "verified", "steps": ["retrieve…", "verify:verified"], ...}
```

`query_type` can also be `unresolved_company`, `clarification_needed` or `service_unavailable` for the early-exit cases.

### HTTP API ([api/app.py](api/app.py), [api/chat.py](api/chat.py))

| Endpoint | Purpose |
|---|---|
| `POST /query` | One-shot question → answer + citations |
| `POST /chat` | Chat turn (keeps a session) |
| `GET /chat/sessions`, `GET/DELETE /chat/sessions/{id}` | Session history |
| `PATCH /chat/sessions/{id}/turns/{tid}` | Mark an answer correct / incorrect |
| `GET /chat/review` | Review queue |
| `GET /health`, `GET /collections` | Liveness and what's indexed |

### MCP server ([mcp_server.py](mcp_server.py))

Exposes the system as tools for Claude Desktop, Claude Code or any MCP client.

| Tool | What it does |
|---|---|
| `search_filings` | Hybrid search + rerank over indexed passages, optionally filtered by ticker / year |
| `fetch_10k` | Ensure a company's 10-K is downloaded and indexed |
| `get_xbrl_figure` | The exact value a company reported to the SEC for a metric and fiscal year |
| `ask_financial_question` | The full agent: plan → retrieve → answer → verify |
| `list_indexed_filings` | What is searchable right now |

```bash
python mcp_server.py                      # stdio
python mcp_server.py --http --port 8765   # streamable HTTP
```

```json
{ "mcpServers": { "financial-rag": {
    "command": "/path/to/Financial_RAG/venv/bin/python",
    "args": ["/path/to/Financial_RAG/mcp_server.py"] } } }
```

---

## Ingestion

```
SEC EDGAR → data/raw/<TICKER>/<YEAR>/primary-document.html
    │  ingestion/parser.py   — iXBRL-aware, section boundary detection, tables → markdown
    ▼
data/parsed/<TICKER>_<YEAR>.json
    │  ingestion/chunker.py  — sentence-boundary text chunks + table sub-chunks
    ▼
data/chunks/<TICKER>_<YEAR>_chunks.json
    │  ingestion/embedder.py — fastembed ONNX (BAAI/bge-base-en-v1.5) + BM25
    ▼
Qdrant (local on-disk, or remote via qdrant_url)
```

`run_ingestion.py` flags: `--skip-download` (reuse manifest), `--skip-index` (parse + chunk only).

### On-demand ingestion

Naming a company outside the bundled 12 triggers `ingestion/registry.py` (SEC's public ticker registry) and then `ingestion/auto_ingest.py`, which downloads just its latest 10-K and indexes it, reusing the warm embedding models. It's a **one-time** cost per company (minutes on CPU, dominated by embedding); the sections that answer the most common questions (financial statements, MD&A, risk factors, business) are searchable first and the rest finishes in the background. A per-ticker lock prevents duplicate work, and failed lookups are cached for 5 minutes.

The same path runs for the bundled companies on every query — not to fetch anything new, but to verify the collection exists and is current, and silently re-index it if it was built by a superseded parser.

---

## Configuration

Settings live in [config.py](config.py), overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `groq_api` | *(required for Groq)* | Groq API key |
| `edgar_email` | *(required)* | Email for SEC EDGAR / XBRL requests |
| `llm_provider` | `groq` | `groq` or `openai` |
| `generation_model` | `openai/gpt-oss-120b` | Answer generation / synthesis (`gpt-4.1` with `openai`) |
| `routing_model` | `openai/gpt-oss-20b` | Classification / decomposition (`gpt-4.1-mini` with `openai`) |
| `openai_api_key` | – | Needed for `llm_provider=openai`; also enables the independent RAGAS judge |
| `judge_model` | `gpt-4.1-mini` | RAGAS judge model |
| `embedding_model` | `BAAI/bge-base-en-v1.5` | Dense embeddings (fastembed) |
| `max_chunk_tokens` | `1000` | Max tokens per chunk |
| `retrieval_top_k` / `rerank_top_k` | `10` / `3` | Candidates per collection / final chunks |
| `qdrant_url`, `qdrant_api_key` | – | Use a remote Qdrant instead of embedded local mode |
| `qdrant_init_timeout` | `8` | Seconds to wait for embedded Qdrant to open (raise for many local collections) |
| `admin_token` | – | Shared secret for `POST /ingest` |
| `model_cache_dir` | – | Persist embedding models (e.g. a Drive path in Colab) |

---

## Evaluation

All scorers are deterministic — no LLM grades its own work.

```bash
# Build the numeric test set from SEC XBRL (writes data/test_sets/numeric_xbrl.json)
python -m evaluation.build_test_set

# Original pipeline vs agent; resumable; --trials N repeats
python -m evaluation.numeric_eval --systems baseline,agent --trials 1

# Non-numeric regression: business / risks / legal / cybersecurity / early exits
python -m evaluation.qualitative_eval --systems baseline,agent

# RAGAS (faithfulness, relevancy, context precision/recall); OpenAI judge if openai_api_key is set
python -m evaluation.ragas_eval --system agent
```

**Numeric set** — 53 questions (39 single, 8 comparison, 6 trend) over revenue, net income, operating income and R&D, with the SEC-reported figure as ground truth. A case passes only if *every* expected figure appears in the answer.

| | Original pipeline | Agent |
|---|---|---|
| Overall | 41/53 (77%) | **53/53** |
| Comparison | 4/8 | 8/8 |
| Trend | 4/6 | 6/6 |
| Single company | 33/39 | 39/39 |
| Net income / operating income | 11/17 · 9/14 | 17/17 · 14/14 |

The original pipeline's misses included plainly wrong figures (Microsoft FY2024 "operating loss of $1,362 million"; Wells Fargo revenue of $37.4B vs $82.3B) and refusals ("cannot be determined") on data that is in the index.

**Read this before quoting the numbers**
- The ground truth and the agent's fallback come from the same SEC XBRL data, so 53/53 shows the pipeline now reliably states the SEC-reported figure — it is not an independent audit of every sentence an answer contains.
- The original pipeline was run once; the agent several times (three further 53/53 runs; two later runs were cut short by an exhausted OpenAI balance — 3 misses, all quota errors, none wrong figures).
- Measured with `llm_provider=openai` (`gpt-4.1`). The live demo defaults to Groq's `gpt-oss` models, whose throughput is limited (see above).
- **Qualitative set** — 12 questions (business, segments, risks, legal, cybersecurity, derived margin, early exits): both systems score the same, and no answer carried a spurious "figures could not be verified" note.

## Tests

```bash
pytest tests -q                      # full suite (includes an MCP-over-stdio test that needs network + a local index)
pytest tests/test_verify.py -q       # one module
```

Covers the verifier, citation renumbering, the LLM layer (empty replies, rate limits, billing errors), XBRL selection (restatements, fiscal years), the graph's corrective paths with the LLM and retriever stubbed, config/provider logic, and the MCP server over a real stdio client.

---

## Project structure

```
Financial_RAG/
├── query.py                 # ask() → LangGraph agent;  ask_baseline() → original pipeline
├── agent/graph.py           # LangGraph: plan · qa subgraph · parallel sub-answers · synthesize
├── facts/xbrl.py            # SEC companyfacts → exact reported figures
├── verification/verify.py   # figure verification + refusal detection
├── llm.py                   # Groq / OpenAI calls, retries, empty-reply handling
├── mcp_server.py            # MCP tools
├── config.py / models.py    # Settings + Pydantic models
├── run_ingestion.py         # bulk ingestion (bundled 12)
├── Dockerfile / railway.toml
│
├── api/            app.py (FastAPI + UI), chat.py (sessions / review)
├── ui/             index.html — self-contained chat UI
├── ingestion/      downloader · parser · chunker · embedder · registry · auto_ingest
├── retrieval/      vector_store · retriever · reranker · parent_store
├── routing/        classifier · decomposer · resolver
├── generation/     generator · synthesizer
├── evaluation/     build_test_set · numeric_eval · qualitative_eval · ragas_eval
├── tests/
└── data/           # gitignored except test_sets/
    ├── raw/ parsed/ chunks/ qdrant/ xbrl/
    ├── company_tickers.json
    └── test_sets/
```

---

## Key technical decisions

- **RAG first, XBRL second.** The answer is generated from the filing text, then verified against XBRL; only a miss injects the fact. That keeps retrieval doing what it's good at (qualitative and derived questions) and makes the verifier a real check rather than a rubber stamp. Cost: one extra LLM call on a miss.
- **Deterministic fallback.** If the model still gets a headline figure wrong with the SEC fact in front of it, the answer is built from the fact directly, with a citation.
- **Every corrective step runs once.** A genuinely out-of-scope query fails fast instead of multiplying LLM cost.
- **Own-filing figures, not restatements.** For fiscal year N the indexed document is that year's 10-K, so that is the number the answer should match.
- **fastembed (ONNX) end to end** — dense, sparse and reranking, no PyTorch — the biggest lever for staying under a low memory limit on small hosts.
- **Hybrid search + cross-encoder rerank** — dense for semantics, BM25 for exact terms (`"total net sales"`, `"391035"`).
- **"Incorporated by reference" filings** — some banks (e.g. Wells Fargo) file a slim 10-K pointing to an EX-13 annual report. The downloader merges EX-13 in, and the parser boundary-detects each embedded document separately (concatenated HTML makes `lxml` drop everything after the first `</html>`).
- **Verify on every query, not just at ingest** — a collection can be deleted, corrupted or built by an old parser, so each ticker a query touches is checked against what's actually indexed and re-indexed if needed.
- **Deterministic decomposition for trends** — one sub-question per year, built from the classifier's `focus`; the LLM decomposer misapplied its "describe the business" special case to single-company metric questions.
- **Provider switch, not a hard dependency** — Groq by default (free), OpenAI opt-in for volume.
- **MCP is packaging, not accuracy** — it exposes the same functions to other agents.

---

## Known limits

- **Throughput on Groq's free tier** — ~1 answer/minute (8,000 TPM per model). Use `llm_provider=openai` or a paid tier for concurrent users.
- **Smaller context on Groq** — a single Groq request must stay under 8,000 tokens, so the answer sees ~4,500 tokens of source text there versus ~8,500 with OpenAI. Very long comparisons can be less detailed on the free tier.
- **Verification covers dollar amounts** — not percentages, ratios or qualitative claims. XBRL covers headline metrics only; other questions rely on retrieval quality.
- **Metric definitions** — "net income" is the figure attributable to the company; "revenue" for banks is net revenue. Some filers don't report every metric (e.g. banks have no operating income or R&D).
- **Chat endpoints are unauthenticated** — protect them before exposing a deployment publicly.
- 10-Ks only (no 10-Q, 8-K or proxy statements, so e.g. executive compensation isn't answerable).

---

## Requirements

- Python 3.10+
- A free [Groq API key](https://console.groq.com) (or an OpenAI key with `llm_provider=openai`)
- SEC EDGAR email (any address — required by SEC fair-use policy)
- ~2 GB disk for the bundled filings + vectors
- No GPU required

---

## License

MIT
