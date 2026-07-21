from pydantic import field_validator
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
    groq_api: str
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
    generation_model: str = "llama-3.3-70b-versatile"
    routing_model:    str = "llama-3.1-8b-instant"

    # Dashboard env-var inputs (Railway, Render, ...) commonly end up with a
    # trailing newline or extra whitespace from copy-paste — invisible in the
    # UI, but any of these get embedded directly in an HTTP header (User-
    # Agent for SEC EDGAR, Authorization for Groq), and a bare "\n" in a
    # header value is rejected outright by the HTTP client with an opaque
    # InvalidHeader error. Strip defensively at the source instead of
    # depending on every caller/platform to paste cleanly.
    @field_validator("groq_api", "edgar_email", "admin_token", mode="after")
    @classmethod
    def _strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

settings = Settings()  # pyright: ignore[reportCallIssue]
