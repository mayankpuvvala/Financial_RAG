from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent

COMPANIES: List[Dict[str, str]] = [
    {"ticker": "AAPL",  "name": "Apple Inc.",                   "sector": "Technology"},
    {"ticker": "MSFT",  "name": "Microsoft Corporation",        "sector": "Technology"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.",                "sector": "Technology"},
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",              "sector": "Technology"},
    {"ticker": "JPM",   "name": "JPMorgan Chase & Co.",         "sector": "Banking"},
    {"ticker": "WFC",   "name": "Wells Fargo & Company",        "sector": "Banking"},
    {"ticker": "BAC",   "name": "Bank of America Corporation",  "sector": "Banking"},
    {"ticker": "GS",    "name": "The Goldman Sachs Group Inc.", "sector": "Banking"},
    {"ticker": "BLK",   "name": "BlackRock Inc.",               "sector": "Asset Management"},
    {"ticker": "STT",   "name": "State Street Corporation",     "sector": "Asset Management"},
    {"ticker": "TROW",  "name": "T. Rowe Price Group Inc.",     "sector": "Asset Management"},
    {"ticker": "IVZ",   "name": "Invesco Ltd.",                 "sector": "Asset Management"},
]

TICKER_TO_COMPANY: Dict[str, Dict[str, str]] = {
    c["ticker"]: c for c in COMPANIES
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown env vars
    )

    # API keys
    groq_api: Optional[str] = None     # required unless llm_provider=openai
    edgar_email: str

    # Optional shared secret for POST /ingest (api/app.py). If unset, the
    # endpoint is open — fine for a low-stakes deploy, but set this via an
    # env var on any deployment reachable by more than just you.
    admin_token: Optional[str] = None

    # Paths
    data_dir:       Path = BASE_DIR / "data"
    raw_dir:        Path = BASE_DIR / "data" / "raw"
    parsed_dir:     Path = BASE_DIR / "data" / "parsed"
    test_sets_dir:  Path = BASE_DIR / "data" / "test_sets"
    chunks_dir:     Path = BASE_DIR / "data" / "chunks"
    qdrant_path:    str  = str(BASE_DIR / "data" / "qdrant")
    # When set, get_client() connects to this remote Qdrant instance (Qdrant
    # Cloud or a self-hosted server) instead of embedded local-mode storage
    # under qdrant_path. See retrieval/vector_store.get_client() — local
    # mode loads/scans every collection into the API process's own memory,
    # which doesn't scale on a memory-capped host once you're past a
    # handful of collections.
    qdrant_url:     Optional[str] = None
    qdrant_api_key: Optional[str] = None
    # Seconds to wait for embedded-Qdrant to open data/qdrant (it scans every
    # collection). 8 suits a hosted health check; raise it for local dev with
    # dozens of collections.
    qdrant_init_timeout: int = 8
    # Set MODEL_CACHE_DIR=/content/drive/MyDrive/fastembed_cache in Colab to
    # persist the 219 MB embedding model across runtime restarts.
    model_cache_dir: Optional[Path] = None

    # Ingestion
    filing_type:          str = "10-K"
    filings_per_company:  int = 3
    # Parse/chunk ProcessPoolExecutor size. Each worker is a whole separate
    # Python process (lxml/BeautifulSoup imports and all) — os.cpu_count()
    # reflects the HOST's core count, not what a container is actually
    # allocated, so sizing off it on a memory-capped host (Railway, etc.)
    # spawns far more processes than the container can hold at once and
    # exhausts it within seconds. Defaults to sequential (1); raise via the
    # PARSE_WORKERS env var on a host known to have room (local dev, Colab).
    parse_workers: int = 1

    # Chunking
    max_chunk_tokens:      int = 1000
    chunk_overlap_sentences: int = 2

    # Embeddings  (fastembed / ONNX — no PyTorch required)
    embedding_model:     str = "BAAI/bge-base-en-v1.5" #change to a larger model if you have a GPU with enough VRAM
    embedding_dim:       int = 768
    embedding_batch_size: int = 64
    sparse_model:        str = "Qdrant/bm25"

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k:    int = 3
    reranker_model:  str = "Xenova/ms-marco-MiniLM-L-12-v2"  # ONNX port via fastembed — no PyTorch

    # Groq models
    generation_model: str = "openai/gpt-oss-120b"
    routing_model:    str = "openai/gpt-oss-20b"

    # Which API serves routing/generation/synthesis: "groq" (default, free tier)
    # or "openai". Groq's free tier allows 8000 tokens/minute per model and one
    # answer here sends ~6-8k, i.e. roughly one query a minute — fine for a
    # demo, a bottleneck under real use. With llm_provider=openai (needs
    # openai_api_key), generation_model/routing_model fall back to the two
    # defaults below unless set explicitly.
    llm_provider:            str = "groq"
    openai_api_key:          Optional[str] = None
    openai_generation_model: str = "gpt-4.1"
    openai_routing_model:    str = "gpt-4.1-mini"

    # Independent judge for evaluation/ragas_eval.py, so the system under test
    # isn't also grading its own answers. Needs openai_api_key; otherwise the
    # eval falls back to the generation model (self-judging).
    judge_model: str = "gpt-4.1-mini"

    @model_validator(mode="after")
    def _apply_provider(self):
        if self.llm_provider not in ("groq", "openai"):
            raise ValueError("llm_provider must be 'groq' or 'openai'")
        if self.llm_provider == "groq" and not self.groq_api:
            raise ValueError("groq_api is required when llm_provider=groq")
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("llm_provider=openai requires openai_api_key")
            if self.generation_model == "openai/gpt-oss-120b":
                self.generation_model = self.openai_generation_model
            if self.routing_model == "openai/gpt-oss-20b":
                self.routing_model = self.openai_routing_model
        return self

    # Dashboard env-var inputs (Railway, Render, ...) commonly end up with a
    # trailing newline or extra whitespace from copy-paste — invisible in the
    # UI, but any of these get embedded directly in an HTTP header (User-
    # Agent for SEC EDGAR, Authorization for Groq), and a bare "\n" in a
    # header value is rejected outright by the HTTP client with an opaque
    # InvalidHeader error. Strip defensively at the source instead of
    # depending on every caller/platform to paste cleanly.
    @field_validator("groq_api", "edgar_email", "admin_token", "qdrant_url", "qdrant_api_key", "openai_api_key", mode="after")
    @classmethod
    def _strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

settings = Settings()  # pyright: ignore[reportCallIssue]
