from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List, Dict

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
    groq_api: str = Field(validation_alias="GROK_API")
    edgar_email: str = Field(validation_alias="EDGAR_EMAIL")

    # Paths
    data_dir:      Path = BASE_DIR / "data"
    raw_dir:       Path = BASE_DIR / "data" / "raw"
    parsed_dir:    Path = BASE_DIR / "data" / "parsed"
    test_sets_dir: Path = BASE_DIR / "data" / "test_sets"
    chunks_dir:    Path = BASE_DIR / "data" / "chunks"
    qdrant_path:   str  = str(BASE_DIR / "data" / "qdrant")

    # Ingestion
    filing_type:          str = "10-K"
    filings_per_company:  int = 3

    # Chunking
    max_chunk_tokens:      int = 1000
    chunk_overlap_sentences: int = 2

    # Embeddings  (fastembed / ONNX — no PyTorch required)
    embedding_model:     str = "BAAI/bge-base-en-v1.5" #change to a larger model if you have a GPU with enough VRAM
    embedding_dim:       int = 768
    embedding_batch_size: int = 32
    sparse_model:        str = "Qdrant/bm25"

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k:    int = 3
    reranker_model:  str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Groq models
    generation_model: str = "llama-3.3-70b-versatile"
    routing_model:    str = "llama-3.1-8b-instant"

settings = Settings()  # pyright: ignore[reportCallIssue]
