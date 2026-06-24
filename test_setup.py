"""
Run this first to verify your environment before running the full pipeline.

    python test_setup.py

It checks packages, config, and downloads ONE filing (AAPL, 1 year) as a smoke test.
"""

import sys

def check(label: str, fn):
    try:
        result = fn()
        print(f"  OK   {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  FAIL {label} — {e}")
        return False


print("\n=== 1. Package imports ===")
check("loguru",              lambda: __import__("loguru"))
check("pydantic",            lambda: __import__("pydantic").__version__)
check("pydantic_settings",   lambda: __import__("pydantic_settings"))
check("sec_edgar_downloader",lambda: __import__("sec_edgar_downloader"))
check("bs4 (beautifulsoup)", lambda: __import__("bs4"))
check("lxml",                lambda: __import__("lxml"))
check("pandas",              lambda: __import__("pandas").__version__)
check("nltk",                lambda: __import__("nltk").__version__)
check("tiktoken",            lambda: __import__("tiktoken"))
check("sentence_transformers",lambda: __import__("sentence_transformers").__version__)
check("fastembed",           lambda: __import__("fastembed").__version__)
check("qdrant_client",       lambda: __import__("qdrant_client").__version__)
check("groq",                lambda: __import__("groq").__version__)

print("\n=== 2. qdrant_client hybrid-search models ===")
check("FusionQuery / Prefetch / Fusion",
      lambda: __import__("qdrant_client.models", fromlist=["FusionQuery","Prefetch","Fusion"]))

print("\n=== 3. Config / .env ===")
check("config loads",        lambda: __import__("config").settings.filing_type)
check("groq_api present",    lambda: bool(__import__("config").settings.groq_api))
check("edgar_email present", lambda: __import__("config").settings.edgar_email)

print("\n=== 4. NLTK tokenizer ===")
def _nltk_check():
    import nltk
    for r in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{r}")
            return r
        except LookupError:
            pass
    return "neither punkt nor punkt_tab found — run chunker once to auto-download"
check("punkt / punkt_tab", _nltk_check)

print("\n=== 5. Download smoke test (AAPL, 1 filing) ===")
print("  (this contacts SEC EDGAR — may take 10-30 seconds)")
def _dl_test():
    from ingestion.downloader import download_all_filings
    from config import COMPANIES
    records = download_all_filings(
        companies=[c for c in COMPANIES if c["ticker"] == "AAPL"],
        limit=1,
    )
    return f"{len(records)} filing(s) downloaded"

check("AAPL download", _dl_test)

print()
