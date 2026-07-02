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
# Section patterns — two levels deep:
#
#  Level 1  "Item N" headers — split the 10-K into major parts
#  Level 2  Financial statement headers — break Item 8 into individual
#           statements so citations say "Consolidated Statements of
#           Operations" instead of the 243k-char "Item 8" blob
#
# Order matters: more-specific patterns must come first.
# ---------------------------------------------------------------------------
SECTION_PATTERNS: List[Tuple[str, str, str]] = [

    # ── Item N patterns (Level 1) ──────────────────────────────────────────
    (r"item\s*1a[\.\s:]*risk\s*factor",          "item_1a_risk_factors", "Item 1A: Risk Factors"),
    (r"item\s*1b[\.\s:]*unresolved",              "item_1b_staff",        "Item 1B: Unresolved Staff Comments"),
    (r"item\s*1[\.\s:]*business",                 "item_1_business",      "Item 1: Business"),
    (r"item\s*2[\.\s:]*propert",                  "item_2_properties",    "Item 2: Properties"),
    (r"item\s*3[\.\s:]*legal",                    "item_3_legal",         "Item 3: Legal Proceedings"),
    (r"item\s*4[\.\s:]*mine",                     "item_4_mine",          "Item 4: Mine Safety"),
    (r"item\s*5[\.\s:]*market",                   "item_5_market",        "Item 5: Market for Equity"),
    (r"item\s*6[\.\s:]*selected",                 "item_6_selected",      "Item 6: Selected Financial Data"),
    (r"item\s*7a[\.\s:]*quantitative",            "item_7a_market_risk",  "Item 7A: Quantitative Disclosures"),
    (r"item\s*7[\.\s:]*management",               "item_7_mda",           "Item 7: MD&A"),
    (r"item\s*8[\.\s:]*financial",                "item_8_financials",    "Item 8: Financial Statements"),
    (r"item\s*9a[\.\s:]*controls",                "item_9a_controls",     "Item 9A: Controls and Procedures"),
    (r"item\s*9b[\.\s:]*other",                   "item_9b_other",        "Item 9B: Other Information"),
    (r"item\s*9[\.\s:]*change",                   "item_9_accountants",   "Item 9: Disagreements with Accountants"),
    (r"item\s*1[0-4][\.\s:]",                     "item_governance",      "Items 10-14: Corporate Governance"),
    (r"item\s*15[\.\s:]*exhibit",                 "item_15_exhibits",     "Item 15: Exhibits"),

    # ── Financial statement sub-sections (Level 2, inside Item 8) ─────────
    # These give precise citations ("Consolidated Statements of Operations")
    # instead of the giant "Item 8" blob.  Patterns cover naming variants
    # across tech, banking, and asset-management 10-Ks.
    (r"consolidated\s+statements?\s+of\s+operations",
        "fs_income_stmt",   "Consolidated Statements of Operations"),
    (r"consolidated\s+statements?\s+of\s+(income|earnings)",
        "fs_income_stmt",   "Consolidated Statements of Income"),
    (r"consolidated\s+(income|earnings)\s+statements?",
        "fs_income_stmt",   "Consolidated Statements of Income"),
    (r"consolidated\s+balance\s+sheets?",
        "fs_balance_sheet", "Consolidated Balance Sheets"),
    (r"consolidated\s+statements?\s+of\s+financial\s+(condition|position)",
        "fs_balance_sheet", "Consolidated Statements of Financial Condition"),
    (r"consolidated\s+statements?\s+of\s+cash\s+flows?",
        "fs_cash_flow",     "Consolidated Statements of Cash Flows"),
    (r"consolidated\s+statements?\s+of\s+(stockholders|shareholders|changes\s+in\s+equity)",
        "fs_equity",        "Consolidated Statements of Equity"),
    (r"notes?\s+to\s+(consolidated\s+)?financial\s+statements?",
        "fs_notes",         "Notes to Financial Statements"),

    # ── Standalone title fallbacks (companies that omit "Item N") ──────────
    (r"^risk\s+factors$",                           "item_1a_risk_factors", "Item 1A: Risk Factors"),
    (r"management.s\s+discussion\s+and\s+analysis", "item_7_mda",           "Item 7: MD&A"),
    (r"^quantitative\s+and\s+qualitative",          "item_7a_market_risk",  "Item 7A: Quantitative Disclosures"),
    (r"financial\s+statements\s+and\s+supplementary","item_8_financials",   "Item 8: Financial Statements"),
    (r"^controls\s+and\s+procedures$",              "item_9a_controls",     "Item 9A: Controls and Procedures"),
]

# Expected ordering of section_ids — used to discard out-of-order detections
# (cross-references near the end of filings that fool "keep last" strategy).
# Anchored patterns for the bank-style recovery pass.
# These are stricter than SECTION_PATTERNS: the line must START and END with
# the section title so that cross-references like "Consolidated balance sheets
# analysis" or "Impact of derivatives on the Consolidated statements of income"
# don't produce false matches.
_FS_RECOVERY_PATTERNS: List[Tuple[str, str, str]] = [
    # Standard naming (JPM, GS, most companies)
    (r"^consolidated\s+statements?\s+of\s+(income|earnings)\s*$",
        "fs_income_stmt",   "Consolidated Statements of Income"),
    (r"^consolidated\s+(income|earnings)\s+statements?\s*$",
        "fs_income_stmt",   "Consolidated Statements of Income"),
    (r"^consolidated\s+balance\s+sheets?\s*$",
        "fs_balance_sheet", "Consolidated Balance Sheets"),
    (r"^consolidated\s+statements?\s+of\s+financial\s+(condition|position)\s*$",
        "fs_balance_sheet", "Consolidated Statements of Financial Condition"),
    (r"^consolidated\s+statements?\s+of\s+cash\s+flows?\s*$",
        "fs_cash_flow",     "Consolidated Statements of Cash Flows"),
    (r"^notes?\s+to\s+(consolidated\s+)?financial\s+statements?\s*$",
        "fs_notes",         "Notes to Financial Statements"),
    (r"^consolidated\s+statements?\s+of\s+(stockholders|shareholders|changes\s+in\s+equity)\s*$",
        "fs_equity",        "Consolidated Statements of Equity"),
    # BAC / WFC variants — all-caps or abbreviated headers
    (r"^consolidated\s+statement\s+of\s+income\s*$",
        "fs_income_stmt",   "Consolidated Statement of Income"),
    (r"^consolidated\s+balance\s+sheet\s*$",
        "fs_balance_sheet", "Consolidated Balance Sheet"),
    (r"^consolidated\s+statement\s+of\s+cash\s+flows?\s*$",
        "fs_cash_flow",     "Consolidated Statement of Cash Flows"),
    (r"^financial\s+statements?\s*$",
        "fs_income_stmt",   "Financial Statements"),
]

_SECTION_PRIORITY: Dict[str, int] = {
    "item_1_business":     10,  "item_1a_risk_factors": 15,
    "item_1b_staff":       16,  "item_2_properties":    20,
    "item_3_legal":        30,  "item_4_mine":          40,
    "item_5_market":       50,  "item_6_selected":      60,
    "item_7_mda":          70,  "item_7a_market_risk":  75,
    "item_8_financials":   80,
    "fs_income_stmt":      81,  "fs_balance_sheet":     82,
    "fs_cash_flow":        83,  "fs_equity":            84,
    "fs_notes":            85,
    "item_9_accountants":  90,  "item_9a_controls":     91,
    "item_9b_other":       92,  "item_governance":      100,
    "item_15_exhibits":    150,
}


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def _strip_ixbrl(html: str) -> str:
    html = re.sub(r"^\s*<\?xml[^?]*\?>", "", html, count=1, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)ix:[^>]*>",      "", html, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)xbrli:[^>]*>",   "", html, flags=re.IGNORECASE)
    html = re.sub(r"<(/?)xbrldi:[^>]*>",  "", html, flags=re.IGNORECASE)
    return html


# ---------------------------------------------------------------------------
# Table utilities
# ---------------------------------------------------------------------------

def _is_data_table(tag: Tag) -> bool:
    rows = tag.find_all("tr", recursive=True)
    if len(rows) < 2:
        return False
    first_row_cells = rows[0].find_all(["td", "th"])
    if len(first_row_cells) < 2:
        return False
    return len(tag.get_text(strip=True)) > 60


def _table_to_markdown(df: pd.DataFrame) -> str:
    df       = df.fillna("").astype(str)
    headers  = list(df.columns)
    rows     = df.values.tolist()
    # Pandas names XBRL columns as integers (0, 1, 2, ...) when no <th>
    # headers are present.  Replace them with empty strings so the
    # markdown doesn't start with "| 0 | 1 | 2 | ..." which dominates
    # BM25 tokens and makes table chunks impossible to retrieve.
    clean_headers = ["" if str(h).lstrip("-").isdigit() else str(h) for h in headers]

    h_line   = "| " + " | ".join(clean_headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in clean_headers)  + " |"
    d_lines  = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([h_line, sep_line] + d_lines)


def _extract_tables(soup: BeautifulSoup) -> Dict[str, Tuple[str, List]]:
    tables: Dict[str, Tuple[str, List]] = {}
    idx = 0

    for tag in soup.find_all("table"):
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
                tag.decompose()
        except ValueError as exc:
            logger.debug(f"Table {idx}: pd.read_html ValueError — {exc}")
            tag.decompose()
        except Exception as exc:
            logger.debug(f"Table {idx}: {type(exc).__name__}: {exc}")
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
    Hybrid boundary detection — different strategies for Item-level vs
    financial-statement sub-sections.

    item_* sections  → FIRST occurrence past the 15 % TOC zone.
      Rationale: Item-level headers sometimes appear as cross-references
      deep inside financial footnotes (e.g. "see Item 7A"), causing
      "keep last" to misplace them and balloon section sizes.

    fs_* sub-sections → LAST occurrence inside the valid window.
      Rationale: fs_* headers ("Consolidated Statements of Operations")
      appear first in the Item 8 mini-table-of-contents, then again as
      the actual statement header. "Keep first" would pick the TOC
      listing (tiny content); "keep last" picks the actual statement.

    Pass 2 enforces monotonic Item ordering so misdetections
    (cross-refs, TOC stragglers) are dropped.
    """
    from collections import defaultdict

    total      = len(lines)
    skip_start = int(total * 0.15)
    skip_end   = int(total * 0.97)

    all_occurrences: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    # Sentinels injected by _annotate_fs_header_tables take absolute priority —
    # they point directly at the table that is the section header, bypassing the
    # last/first occurrence heuristics that work on plain text.
    sentinel_hits: Dict[str, Tuple[int, str, str]] = {}

    for i, line in enumerate(lines):
        if i < skip_start or i > skip_end:
            continue
        stripped = line.strip()
        if not stripped or len(stripped) > 250:
            continue

        if stripped.startswith(_FS_SENTINEL_PREFIX):
            title = stripped[len(_FS_SENTINEL_PREFIX):]
            for _, sid, t in _FS_RECOVERY_PATTERNS:
                if t.lower() == title.lower():
                    sentinel_hits[sid] = (i, sid, t)
                    break
            continue

        match = _match_section(stripped)

        # Some filers (AMZN, WFC) split "Item N." and the description across
        # two lines.  When a short "Item N" line doesn't match on its own,
        # join it with the next non-empty line and retry.
        if not match and len(stripped) < 35 and re.search(r"^item\s*\d", stripped.lower()):
            for j in range(i + 1, min(i + 5, total)):
                next_stripped = lines[j].strip()
                if next_stripped and len(next_stripped) < 100:
                    combined = stripped + " " + next_stripped
                    match = _match_section(combined)
                    break

        if match:
            section_id, title = match
            all_occurrences[section_id].append((i, section_id, title))

    selected: Dict[str, Tuple[int, str, str]] = {}
    for section_id, occurrences in all_occurrences.items():
        if section_id in sentinel_hits:
            selected[section_id] = sentinel_hits[section_id]   # sentinel wins
        elif section_id.startswith("fs_"):
            selected[section_id] = occurrences[-1]   # last = actual statement
        else:
            selected[section_id] = occurrences[0]    # first = content header

    # Any sentinel sections not found via text patterns also get included
    for sid, entry in sentinel_hits.items():
        if sid not in selected:
            selected[sid] = entry

    candidates = sorted(selected.values(), key=lambda x: x[0])

    # Monotonic priority filter — drop out-of-order misdetections
    validated: List[Tuple[int, str, str]] = []
    max_priority = -1

    for entry in candidates:
        priority = _SECTION_PRIORITY.get(entry[1], 500)
        if priority > max_priority:
            validated.append(entry)
            max_priority = priority

    # --- Recovery pass for bank-style filings --------------------------------
    # Banks (JPM, GS, BAC, WFC) do not place audited financial statements
    # inline within Item 8.  Instead they appear either:
    #   (a) after Item 15 — the monotonic filter drops them (priority 81 < 150)
    #   (b) inside a large item_* section such as item_governance
    #
    # Strategy: scan within every section that is at least 1,000 lines long
    # for anchored fs_* headers.  The anchored patterns in _FS_RECOVERY_PATTERNS
    # require the line to start AND end with the section title, which eliminates
    # cross-references like "Consolidated balance sheets analysis" (fails $ anchor)
    # or "Impact of derivatives on the Consolidated statements of income" (fails ^).
    already_found: set = {e[1] for e in validated}
    missing_fs = [
        (p, sid, t) for p, sid, t in _FS_RECOVERY_PATTERNS
        if sid not in already_found
    ]

    if missing_fs:
        recovered: List[Tuple[int, str, str]] = []
        found_fs: set = set()

        # Build (start, end) pairs for each validated section
        section_ranges = [
            (validated[i][0],
             validated[i + 1][0] if i + 1 < len(validated) else skip_end)
            for i in range(len(validated))
        ]

        for start_ln, end_ln in section_ranges:
            if end_ln - start_ln < 1_000:
                continue
            for j in range(start_ln, end_ln):
                stripped = lines[j].strip()
                if not stripped or len(stripped) > 120:
                    continue
                for pattern, section_id, title in missing_fs:
                    if section_id in found_fs:
                        continue
                    if re.search(pattern, stripped.lower()):
                        recovered.append((j, section_id, title))
                        found_fs.add(section_id)
                        break
                if len(found_fs) == len(missing_fs):
                    break

        if recovered:
            validated.extend(recovered)
            validated.sort(key=lambda x: x[0])

    return validated


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

        for para in re.split(r"\n{2,}", part):
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
# Pre-extraction annotation
# ---------------------------------------------------------------------------

# Sentinel prefix written into the soup before table extraction so that
# _find_section_boundaries can pick up the section_id from plain text.
_FS_SENTINEL_PREFIX = "FS_SECTION_HEADER:"


def _annotate_fs_header_tables(soup: BeautifulSoup) -> None:
    """
    Scan every <table> for a cell (within the first 5 rows) whose text
    matches an fs_* header pattern. When found, insert a sentinel text node
    immediately before the <table> so the header survives as a plain-text
    line after _extract_tables replaces the table with a placeholder.

    Targets issuers like BAC/WFC where section headers such as
    "Consolidated Statement of Income" live inside <td> cells — sometimes
    after leading empty spacer rows — and are otherwise lost when the table
    is replaced by a <<<TABLE_N>>> placeholder.
    """
    for table_tag in soup.find_all("table"):
        rows = table_tag.find_all("tr", recursive=True)
        matched = False
        for row in rows[:5]:                              # scan first 5 rows
            for cell in row.find_all(["td", "th"]):
                cell_text = cell.get_text(separator=" ", strip=True).lower()
                if not cell_text or len(cell_text) > 120:
                    continue
                for pattern, section_id, title in _FS_RECOVERY_PATTERNS:
                    if re.search(pattern, cell_text):
                        sentinel = soup.new_string(f"\n{_FS_SENTINEL_PREFIX}{title}\n")
                        table_tag.insert_before(sentinel)
                        logger.debug(f"  Pre-annotated table: '{title}' ({cell_text[:50]})")
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break


# ---------------------------------------------------------------------------
# Public entry points
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

    # Pre-annotate financial statement header tables before extraction.
    # Some issuers (BAC, WFC) put headers like "Consolidated Statement of
    # Income" inside <td> cells of the statement tables rather than as
    # stand-alone <div> or <p> elements.  After _extract_tables replaces those
    # tables with <<<TABLE_N>>> placeholders, the text disappears from the
    # plain-text stream and boundary detection misses it.  By inserting a
    # sentinel text node BEFORE the table here, the marker survives into
    # get_text() output so _find_section_boundaries can detect it normally.
    _annotate_fs_header_tables(soup)

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
        logger.warning(f"  No sections found for {ticker} FY{fiscal_year}; using full document")
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
                file_path        = Path(record["file_path"]),
                company          = record["company"],
                ticker           = record["ticker"],
                fiscal_year      = record["fiscal_year"],
                accession_number = record["accession_number"],
                filing_date      = record.get("filing_date"),
            )
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(doc.model_dump_json(indent=2))
            documents.append(doc)
        except Exception as exc:
            logger.error(f"Failed to parse {record['ticker']} FY{record['fiscal_year']}: {exc}")

    logger.success(f"Parsed {len(documents)} / {len(manifest)} filings")
    return documents
