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


def _extract_html_from_full_submission(
    submission_path: Path,
    output_path: Path,
) -> bool:
    """
    Stream through full-submission.txt and write the primary 10-K HTML to output_path.

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
    in_doc       = False
    is_10k_doc   = False
    in_text      = False
    skip_xbrl_close = False
    html_lines: List[str] = []

    with open(submission_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            stripped = raw_line.rstrip("\n").rstrip("\r")

            if stripped == "<DOCUMENT>":
                in_doc     = True
                is_10k_doc = False
                in_text    = False
                continue

            if in_doc and not is_10k_doc:
                if stripped.startswith("<TYPE>"):
                    doc_type = stripped[6:].strip()
                    if doc_type in ("10-K", "10-K405", "10-KSB"):
                        is_10k_doc = True
                    else:
                        in_doc = False   # not the doc we want
                continue

            if is_10k_doc and not in_text:
                if stripped == "<TEXT>":
                    in_text = True
                continue

            if in_text:
                # End of text section — we're done
                if stripped in ("</TEXT>", "</XBRL></TEXT>"):
                    break

                # Skip the bare <XBRL> opening line
                if not html_lines and stripped == "<XBRL>":
                    skip_xbrl_close = True
                    continue

                # Skip the bare </XBRL> closing line
                if skip_xbrl_close and stripped == "</XBRL>":
                    break

                html_lines.append(raw_line)

    if not html_lines:
        return False

    output_path.write_text("".join(html_lines), encoding="utf-8")
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

    manifest_path = raw_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(
        f"Download complete — {len(all_records)} total filings. "
        f"Manifest → {manifest_path}"
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
