"""
Entry point for the ingestion pipeline.

Run:
    python run_ingestion.py

Steps:
    1. Download  — 10-K filings from SEC EDGAR  → data/raw/
    2. Parse     — HTML → structured JSON        → data/parsed/
    3. Chunk     — sentences + tables            → data/chunks/
    4. Index     — embed + upsert into Qdrant   → data/qdrant/

Flags:
    --skip-download   reuse existing manifest (already downloaded)
    --skip-index      skip embedding / Qdrant step (parse + chunk only)
"""

import sys
from loguru import logger

# config is lightweight — always safe to import at the top
from config import settings, COMPANIES


def main(skip_download: bool = False, skip_index: bool = False) -> None:

    # ------------------------------------------------------------------
    # Lazy imports — heavy ML packages are only loaded when their step
    # actually runs. This lets the downloader work even if sentence-
    # transformers / fastembed / qdrant-client aren't installed yet.
    # ------------------------------------------------------------------
    from ingestion.downloader import download_all_filings, load_manifest
    from ingestion.parser    import parse_all_filings
    from ingestion.chunker   import chunk_all_documents

    # Set up logging (must come after imports so logger is ready)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    logger.add(settings.data_dir / "ingestion.log", level="DEBUG", rotation="10 MB")

    # ------------------------------------------------------------------
    # Step 1 — Download
    # ------------------------------------------------------------------
    if skip_download:
        logger.info("Skipping download — loading existing manifest")
        manifest = load_manifest(settings.raw_dir)
    else:
        logger.info(f"Starting download for {len(COMPANIES)} companies …")
        manifest = download_all_filings(
            companies=COMPANIES,
            filing_type=settings.filing_type,
            limit=settings.filings_per_company,
            raw_dir=settings.raw_dir,
        )

    logger.info(f"Manifest: {len(manifest)} filing(s)")

    # ------------------------------------------------------------------
    # Step 2 — Parse
    # ------------------------------------------------------------------
    logger.info("Parsing HTML filings …")
    documents = parse_all_filings(
        manifest=manifest,
        parsed_dir=settings.parsed_dir,
    )

    # ------------------------------------------------------------------
    # Step 3 — Chunk
    # ------------------------------------------------------------------
    logger.info("Chunking documents …")
    chunks = chunk_all_documents(
        documents=documents,
        chunks_dir=settings.chunks_dir,
    )

    # ------------------------------------------------------------------
    # Step 4 — Embed + Index  (skipped if --skip-index)
    # ------------------------------------------------------------------
    if not skip_index:
        # Import here so missing ML packages don't break steps 1-3
        from ingestion.embedder import index_chunks
        logger.info("Embedding and indexing into Qdrant …")
        index_chunks(chunks)
    else:
        logger.info("Skipping Qdrant indexing (--skip-index)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.success("=" * 52)
    logger.success(f"Ingestion complete")
    logger.success(f"  Documents : {len(documents)}")
    logger.success(f"  Chunks    : {len(chunks)}")
    logger.success(f"  Parsed    → {settings.parsed_dir}")
    logger.success(f"  Chunks    → {settings.chunks_dir}")
    if not skip_index:
        logger.success(f"  Qdrant    → {settings.qdrant_path}")
    tickers = sorted({d.ticker for d in documents})
    years   = sorted({d.fiscal_year for d in documents})
    logger.success(f"  Tickers   : {tickers}")
    logger.success(f"  Years     : {years}")
    logger.success("=" * 52)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Financial RAG — ingestion pipeline")
    ap.add_argument("--skip-download", action="store_true",
                    help="Skip EDGAR download; use existing manifest")
    ap.add_argument("--skip-index", action="store_true",
                    help="Skip embedding / Qdrant indexing")
    args = ap.parse_args()
    main(skip_download=args.skip_download, skip_index=args.skip_index)
