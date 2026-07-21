import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger
from sec_edgar_downloader import Downloader

from config import settings, COMPANIES, TICKER_TO_COMPANY


# ---------------------------------------------------------------------------
# HTML extraction from full-submission.txt
# ---------------------------------------------------------------------------

def _extract_fiscal_year_from_sgml(submission_path: Path) -> Optional[int]:
    """
    Read the SGML header in full-submission.txt and return the fiscal year
    from the CONFORMED PERIOD OF REPORT field (most accurate source).
    """
    try:
        with open(submission_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("CONFORMED PERIOD OF REPORT:"):
                    date_str = line.split(":", 1)[1].strip()  # e.g. "20240928"
                    if len(date_str) >= 4:
                        return int(date_str[:4])
                # Header ends at first <DOCUMENT> tag — stop scanning
                if line == "<DOCUMENT>":
                    break
    except Exception:
        pass
    return None


# Some issuers — historically large bank holding companies like Wells Fargo —
# file a slim 10-K "wrapper" that incorporates Item 1A (Risk Factors), Item 7
# (MD&A), and Item 8 (Financial Statements) BY REFERENCE to a separately
# exhibited "Annual Report to Shareholders" (EX-13). If we only parse the
# wrapper, those items are just one-line pointers ("see Exhibit 13") with no
# actual content. Merging EX-13 into the same HTML lets the section parser
# see the real text. No other bundled filer uses this pattern, so this is a
# no-op for everyone else.
_PRIMARY_TYPES  = ("10-K", "10-K405", "10-KSB")
_EXHIBIT_TYPES  = ("EX-13",)


def _extract_html_from_full_submission(
    submission_path: Path,
    output_path: Path,
) -> bool:
    """
    Stream through full-submission.txt and write the primary 10-K HTML —
    plus any incorporated-by-reference exhibits (EX-13) — to output_path,
    concatenated in document order.

    EDGAR MIME format inside full-submission.txt:
        <DOCUMENT>
        <TYPE>10-K
        ...
        <TEXT>
        <XBRL>          ← optional wrapper, strip it
        <!DOCTYPE html>
        ...HTML content...
        </html>
        </XBRL>         ← optional, marks end
        </TEXT>
        </DOCUMENT>
    """
    wanted_prefixes = _PRIMARY_TYPES + _EXHIBIT_TYPES

    in_doc          = False
    doc_type        = None
    wanted          = False
    in_text         = False
    skip_xbrl_close = False
    current_lines: List[str] = []
    combined:      List[str] = []

    def _flush() -> None:
        if current_lines:
            if combined:
                combined.append(f"\n<!-- ===== embedded document: {doc_type} ===== -->\n")
            combined.extend(current_lines)

    with open(submission_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            stripped = raw_line.rstrip("\n").rstrip("\r")

            if stripped == "<DOCUMENT>":
                in_doc          = True
                doc_type        = None
                wanted          = False
                in_text         = False
                skip_xbrl_close = False
                current_lines   = []
                continue

            if stripped == "</DOCUMENT>":
                if wanted:
                    _flush()
                in_doc = False
                continue

            if in_doc and doc_type is None:
                if stripped.startswith("<TYPE>"):
                    doc_type = stripped[6:].strip()
                    wanted   = any(
                        doc_type == t or doc_type.startswith(t) for t in wanted_prefixes
                    )
                continue

            if wanted and not in_text:
                if stripped == "<TEXT>":
                    in_text = True
                continue

            if wanted and in_text:
                if stripped in ("</TEXT>", "</XBRL></TEXT>"):
                    in_text = False
                    continue

                if not current_lines and stripped == "<XBRL>":
                    skip_xbrl_close = True
                    continue

                if skip_xbrl_close and stripped == "</XBRL>":
                    skip_xbrl_close = False
                    continue

                current_lines.append(raw_line)

    if not combined:
        return False

    output_path.write_text("".join(combined), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# File discovery inside an accession directory
# ---------------------------------------------------------------------------

def _find_primary_document(accession_dir: Path) -> Optional[Path]:
    """
    Return the path to the primary 10-K HTML for this accession.

    Priority:
      1. Already-extracted primary-document.html (idempotent on re-runs)
      2. Any .htm/.html file that isn't an index
      3. Extract from full-submission.txt (EDGAR MIME format)
    """
    # 1. Previously extracted
    for name in ("primary-document.html", "primary-document.htm"):
        p = accession_dir / name
        if p.exists() and p.stat().st_size > 1000:
            return p

    # 2. Standalone HTML files
    html_files = [
        f for f in (
            list(accession_dir.glob("*.htm")) +
            list(accession_dir.glob("*.html"))
        )
        if "index" not in f.name.lower()
    ]
    if html_files:
        candidate = max(html_files, key=lambda f: f.stat().st_size)
        if candidate.stat().st_size > 1000:
            return candidate

    # 3. Extract from full-submission.txt
    submission = accession_dir / "full-submission.txt"
    if submission.exists():
        output = accession_dir / "primary-document.html"
        logger.debug(f"  Extracting HTML from full-submission.txt ({accession_dir.name})")
        if _extract_html_from_full_submission(submission, output):
            return output
        logger.warning(f"  Could not extract 10-K HTML from {accession_dir.name}")

    return None


def _extract_fiscal_year(accession_dir: Path) -> int:
    """
    Return the fiscal year for this filing.

    Priority:
      1. CONFORMED PERIOD OF REPORT in full-submission.txt SGML header
      2. reportDate in filing-details.json
      3. Year embedded in accession number (filed_year - 1, rough fallback)
    """
    # 1. SGML header (most accurate)
    submission = accession_dir / "full-submission.txt"
    if submission.exists():
        yr = _extract_fiscal_year_from_sgml(submission)
        if yr:
            return yr

    # 2. filing-details.json
    details_file = accession_dir / "filing-details.json"
    if details_file.exists():
        try:
            with open(details_file) as f:
                details = json.load(f)
            report_date = details.get("reportDate", "")
            if report_date:
                return int(report_date[:4])
        except Exception:
            pass

    # 3. Accession number: XXXXXXXXXX-YY-ZZZZZZ  (YY = 2-digit filing year)
    parts = accession_dir.name.split("-")
    if len(parts) >= 2:
        try:
            return 2000 + int(parts[1]) - 1
        except ValueError:
            pass

    return 2023


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def _collect_filing_records(
    ticker: str, filing_type: str, raw_dir: Path
) -> List[Dict]:
    ticker_dir   = raw_dir / "sec-edgar-filings" / ticker / filing_type
    company_info = TICKER_TO_COMPANY.get(ticker, {"name": ticker, "sector": "Unknown"})
    records: List[Dict] = []

    if not ticker_dir.exists():
        logger.warning(f"Directory not found after download: {ticker_dir}")
        return records

    for accession_dir in sorted(ticker_dir.iterdir()):
        if not accession_dir.is_dir():
            continue

        html_file   = _find_primary_document(accession_dir)
        if not html_file:
            logger.warning(f"No HTML found in {accession_dir.name} for {ticker}")
            continue

        fiscal_year = _extract_fiscal_year(accession_dir)

        records.append({
            "ticker":           ticker,
            "company":          company_info["name"],
            "sector":           company_info["sector"],
            "filing_type":      filing_type,
            "fiscal_year":      fiscal_year,
            "accession_number": accession_dir.name,
            "file_path":        str(html_file),
        })
        logger.debug(f"  {ticker} FY{fiscal_year} → {html_file.name} ({html_file.stat().st_size // 1024} KB)")

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _download_ticker_worker(
    company: Dict, filing_type: str, limit: int, raw_dir: Path
) -> List[Dict]:
    """Thread worker: download one ticker and return its filing records."""
    ticker = company["ticker"]
    dl = Downloader(
        company_name="FinancialRAG",
        email_address=settings.edgar_email,
        download_folder=str(raw_dir),
    )
    try:
        logger.info(f"Downloading {filing_type} filings for {ticker} …")
        dl.get(filing_type, ticker, limit=limit)
        time.sleep(0.5)   # be polite to EDGAR
    except Exception as exc:
        logger.error(f"Download failed for {ticker}: {exc}")
        return []
    records = _collect_filing_records(ticker, filing_type, raw_dir)
    logger.success(f"{ticker}: {len(records)} filing(s) ready")
    return records


def download_all_filings(
    companies:    List[Dict] = COMPANIES,
    filing_type:  str        = settings.filing_type,
    limit:        int        = settings.filings_per_company,
    raw_dir:      Path       = settings.raw_dir,
) -> List[Dict]:
    """
    Download 10-K filings for all companies and return a manifest list.
    HTML is extracted from EDGAR's full-submission.txt on the fly.
    Tickers are downloaded concurrently (4 threads) — each thread creates its
    own Downloader instance so they don't share mutable state.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []

    # 4 concurrent threads keeps us comfortably under EDGAR's 10 req/s limit
    # while still being ~4× faster than a sequential loop over 12 tickers.
    max_workers = min(4, len(companies))
    logger.info(f"Downloading {len(companies)} tickers in parallel (workers={max_workers}) …")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_ticker_worker, c, filing_type, limit, raw_dir): c["ticker"]
            for c in companies
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                all_records.extend(future.result())
            except Exception as exc:
                logger.error(f"Download worker failed for {ticker}: {exc}")

    # Merge into any existing manifest rather than overwriting it — this
    # function is also called for single-ticker on-demand ingestion, and a
    # blind overwrite would wipe out every other already-downloaded company.
    manifest_path = raw_dir / "manifest.json"
    existing_records: List[Dict] = []
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                existing_records = json.load(f)
        except Exception:
            existing_records = []

    updated_keys = {(r["ticker"], r["fiscal_year"]) for r in all_records}
    merged = [
        r for r in existing_records
        if (r["ticker"], r["fiscal_year"]) not in updated_keys
    ]
    merged.extend(all_records)
    merged.sort(key=lambda r: (r["ticker"], r["fiscal_year"]))

    with open(manifest_path, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(
        f"Download complete — {len(all_records)} filing(s) this run, "
        f"{len(merged)} total in manifest → {manifest_path}"
    )
    return all_records


def load_manifest(raw_dir: Path = settings.raw_dir) -> List[Dict]:
    """
    Load a previously saved manifest.
    If manifest.json is missing, rebuild it from whatever is already on disk.
    """
    manifest_path = raw_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        if data:           # non-empty — use it
            return data
        logger.warning("manifest.json exists but is empty — rebuilding from disk")

    # Rebuild: walk every ticker/form/accession directory already downloaded
    logger.info("Rebuilding manifest from existing downloaded files …")
    all_records: List[Dict] = []

    filings_root = raw_dir / "sec-edgar-filings"
    if not filings_root.exists():
        raise FileNotFoundError(
            f"No downloads found at {filings_root}. "
            "Run without --skip-download first."
        )

    for ticker_dir in sorted(filings_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for form_dir in ticker_dir.iterdir():
            if not form_dir.is_dir():
                continue
            records = _collect_filing_records(ticker, form_dir.name, raw_dir)
            all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No filings found on disk. Run without --skip-download."
        )

    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.success(f"Manifest rebuilt — {len(all_records)} filings → {manifest_path}")
    return all_records
