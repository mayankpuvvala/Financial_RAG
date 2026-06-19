import re
import json
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger

from models import ContentBlock, ParsedDocument, ParsedSection

# ---------------------------------------------------------------------------
# SEC 10-K section patterns — ordered most-specific first.
# Each entry: (regex, section_id, human title)
# ---------------------------------------------------------------------------
SECTION_PATTERNS: List[Tuple[str, str, str]] = [
    # Standard "Item N[A]" patterns
    (r"item\s*1a[\.\s:]*risk\s*factor",         "item_1a_risk_factors", "Item 1A: Risk Factors"),
    (r"item\s*1b[\.\s:]*unresolved",             "item_1b_staff",        "Item 1B: Unresolved Staff Comments"),
    (r"item\s*1[\.\s:]*business",                "item_1_business",      "Item 1: Business"),
    (r"item\s*2[\.\s:]*propert",                 "item_2_properties",    "Item 2: Properties"),
    (r"item\s*3[\.\s:]*legal",                   "item_3_legal",         "Item 3: Legal Proceedings"),
    (r"item\s*4[\.\s:]*mine",                    "item_4_mine",          "Item 4: Mine Safety"),
    (r"item\s*5[\.\s:]*market",                  "item_5_market",        "Item 5: Market for Equity"),
    (r"item\s*6[\.\s:]*selected",                "item_6_selected",      "Item 6: Selected Financial Data"),
    (r"item\s*7a[\.\s:]*quantitative",           "item_7a_market_risk",  "Item 7A: Quantitative Disclosures"),
    (r"item\s*7[\.\s:]*management",              "item_7_mda",           "Item 7: MD&A"),
    (r"item\s*8[\.\s:]*financial\s*statement",   "item_8_financials",    "Item 8: Financial Statements"),
    (r"item\s*9a[\.\s:]*controls",               "item_9a_controls",     "Item 9A: Controls and Procedures"),
    (r"item\s*9b[\.\s:]*other",                  "item_9b_other",        "Item 9B: Other Information"),
    (r"item\s*9[\.\s:]*change",                  "item_9_accountants",   "Item 9: Disagreements with Accountants"),
    (r"item\s*1[0-4][\.\s:]",                    "item_governance",      "Items 10-14: Corporate Governance"),
    (r"item\s*15[\.\s:]*exhibit",                "item_15_exhibits",     "Item 15: Exhibits"),
    # Fallback patterns — match standalone section titles without "Item N"
    # (used by Amazon, Google and other companies that omit Item numbers in headers)
    (r"^risk\s+factors$",                        "item_1a_risk_factors", "Item 1A: Risk Factors"),
    (r"management.s\s+discussion\s+and\s+analysis", "item_7_mda",        "Item 7: MD&A"),
    (r"^quantitative\s+and\s+qualitative",       "item_7a_market_risk",  "Item 7A: Quantitative Disclosures"),
    (r"financial\s+statements\s+and\s+supplementary", "item_8_financials","Item 8: Financial Statements"),
    (r"^controls\s+and\s+procedures$",           "item_9a_controls",     "Item 9A: Controls and Procedures"),
]


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _strip_ixbrl(html: str) -> str:
    """Remove iXBRL / XBRL namespace tags, keeping their text content."""
    # Strip XML declaration that triggers BeautifulSoup parser warnings
    html = re.sub(r"^\s*<\?xml[^?]*\?>", "", html, count=1, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)ix:[^>]*>",     "", html, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)xbrli:[^>]*>",  "", html, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)xbrldi:[^>]*>", "", html, flags=re.IGNORECASE)
    return html


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------

def _is_data_table(tag: Tag) -> bool:
    """Return True if this <table> looks like a financial data table."""
    # Only look at direct-child rows to avoid counting nested-table rows
    rows = tag.find_all("tr", recursive=True)
    if len(rows) < 2:
        return False
    first_row_cells = rows[0].find_all(["td", "th"], recursive=False)
    if len(first_row_cells) < 2:
        # Try one level deeper (some tables wrap cells in a tbody)
        first_row_cells = rows[0].find_all(["td", "th"])
    if len(first_row_cells) < 2:
        return False
    text = tag.get_text(strip=True)
    return len(text) > 60


def _table_to_markdown(df: pd.DataFrame) -> str:
    df = df.fillna("").astype(str)
    headers    = list(df.columns)
    rows       = df.values.tolist()
    header_ln  = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_ln     = "| " + " | ".join("---" for _ in headers) + " |"
    data_lns   = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_ln, sep_ln] + data_lns)


def _extract_tables(soup: BeautifulSoup) -> Dict[str, Tuple[str, List]]:
    """
    Find all data tables, replace each with a placeholder, return the mapping.

    We process only tables that are still attached to the document
    (tag.parent is not None) to avoid re-processing nested tables whose
    parent was already replaced.
    """
    tables: Dict[str, Tuple[str, List]] = {}
    idx = 0

    for tag in soup.find_all("table"):
        # Skip if this tag was removed when its parent was replaced
        if tag.parent is None:
            continue

        if not _is_data_table(tag):
            tag.decompose()
            continue

        table_html  = str(tag)
        placeholder = f"<<<TABLE_{idx}>>>"
        idx        += 1

        try:
            dfs = pd.read_html(StringIO(table_html), flavor="lxml")
            if dfs and not dfs[0].empty:
                df       = dfs[0]
                markdown = _table_to_markdown(df)
                raw      = df.fillna("").astype(str).values.tolist()
                tables[placeholder] = (markdown, raw)
                tag.replace_with(soup.new_string(f"\n{placeholder}\n"))
            else:
                logger.debug(f"Table {idx}: empty DataFrame, skipping")
                tag.decompose()
        except ValueError as exc:
            # pd.read_html raises ValueError when it finds no valid tables
            logger.debug(f"Table {idx}: pd.read_html ValueError — {exc}")
            tag.decompose()
        except Exception as exc:
            logger.debug(f"Table {idx}: unexpected error — {type(exc).__name__}: {exc}")
            tag.decompose()

    return tables


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _match_section(text: str) -> Optional[Tuple[str, str]]:
    t = text.lower().strip()
    for pattern, section_id, title in SECTION_PATTERNS:
        if re.search(pattern, t):
            return section_id, title
    return None


def _find_section_boundaries(lines: List[str]) -> List[Tuple[int, str, str]]:
    """
    Scan document lines for SEC section headers.

    Strategy: for each section_id keep the LAST occurrence.
    "Last" reliably lands on the actual content header rather than the
    Table-of-Contents entry (TOC comes first, content comes later).
    No skip-zone — relying on "last wins" is sufficient and avoids
    filtering out filings (e.g. Amazon, Google) where the content
    header appears near the beginning.
    """
    last_seen: Dict[str, Tuple[int, str, str]] = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 250:
            continue
        match = _match_section(stripped)
        if match:
            section_id, title = match
            last_seen[section_id] = (i, section_id, title)

    boundaries = list(last_seen.values())
    boundaries.sort(key=lambda x: x[0])
    return boundaries


# ---------------------------------------------------------------------------
# Section content assembly
# ---------------------------------------------------------------------------

def _build_section(
    section_id: str,
    title:      str,
    text_slice: str,
    tables:     Dict[str, Tuple[str, List]],
    order:      int,
) -> ParsedSection:
    blocks: List[ContentBlock] = []
    position = 0

    parts = re.split(r"(<<<TABLE_\d+>>>)", text_slice)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<<<TABLE_") and part in tables:
            markdown, raw = tables[part]
            blocks.append(ContentBlock(
                block_type="table",
                text=markdown,
                raw_table=raw,
                position=position,
            ))
            position += 1
            continue

        # Split text into paragraphs on blank lines
        paragraphs = re.split(r"\n{2,}", part)
        for para in paragraphs:
            para = re.sub(r"[ \t]{2,}", " ", para).replace("\n", " ").strip()
            if len(para) < 30:
                continue
            is_footnote = len(para) < 500 and bool(re.match(r"^[\(\*\d\†‡§¶]", para))
            blocks.append(ContentBlock(
                block_type="footnote" if is_footnote else "text",
                text=para,
                position=position,
            ))
            position += 1

    return ParsedSection(
        section_id=section_id,
        title=title,
        content_blocks=blocks,
        order=order,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_filing(
    file_path:        Path,
    company:          str,
    ticker:           str,
    fiscal_year:      int,
    accession_number: str,
    filing_date:      Optional[str] = None,
) -> ParsedDocument:
    logger.info(f"Parsing {ticker} FY{fiscal_year} — {file_path.name}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        raw_html = fh.read()

    cleaned_html = _strip_ixbrl(raw_html)
    soup = BeautifulSoup(cleaned_html, "lxml")

    for tag in soup(["script", "style", "meta", "link", "head", "noscript"]):
        tag.decompose()

    tables   = _extract_tables(soup)
    raw_text = soup.get_text(separator="\n")
    raw_text = re.sub(r"\n{4,}", "\n\n\n", raw_text)

    lines      = raw_text.split("\n")
    boundaries = _find_section_boundaries(lines)

    logger.info(
        f"  → {len(boundaries)} sections detected, "
        f"{len(tables)} tables extracted"
    )

    sections: List[ParsedSection] = []

    if not boundaries:
        logger.warning(f"  No section headers found for {ticker} FY{fiscal_year}; storing as full document")
        sections.append(_build_section("full_document", "Full Document", raw_text, tables, 0))
    else:
        for i, (start_line, section_id, title) in enumerate(boundaries):
            end_line   = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            slice_text = "\n".join(lines[start_line + 1 : end_line])
            section    = _build_section(section_id, title, slice_text, tables, i)
            if section.content_blocks:
                sections.append(section)

    doc = ParsedDocument(
        source_path=str(file_path),
        company=company,
        ticker=ticker,
        fiscal_year=fiscal_year,
        filing_date=filing_date,
        accession_number=accession_number,
        sections=sections,
    )

    block_count = sum(len(s.content_blocks) for s in sections)
    logger.success(f"  {ticker} FY{fiscal_year}: {len(sections)} sections, {block_count} blocks")
    return doc


def parse_all_filings(
    manifest:   List[dict],
    parsed_dir: Path,
) -> List[ParsedDocument]:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    documents: List[ParsedDocument] = []

    for record in manifest:
        out_file = parsed_dir / f"{record['ticker']}_{record['fiscal_year']}.json"

        if out_file.exists():
            logger.info(f"Skipping {record['ticker']} FY{record['fiscal_year']} (already parsed)")
            with open(out_file, encoding="utf-8") as f:
                documents.append(ParsedDocument.model_validate(json.load(f)))
            continue

        try:
            doc = parse_filing(
                file_path=Path(record["file_path"]),
                company=record["company"],
                ticker=record["ticker"],
                fiscal_year=record["fiscal_year"],
                accession_number=record["accession_number"],
                filing_date=record.get("filing_date"),
            )
            # Always write JSON as UTF-8 — Windows default (cp1252) breaks
            # on Unicode characters present in SEC filings (bullet points, etc.)
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(doc.model_dump_json(indent=2))
            documents.append(doc)
        except Exception as exc:
            logger.error(f"Failed to parse {record['ticker']} FY{record['fiscal_year']}: {exc}")

    logger.success(f"Parsed {len(documents)} / {len(manifest)} filings")
    return documents
