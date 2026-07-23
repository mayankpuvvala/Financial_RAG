import os
import re
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger

from config import settings
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
    (r"item\s*1a[\.\s:—–-]*risk\s*factor",          "item_1a_risk_factors", "Item 1A: Risk Factors"),
    (r"item\s*1b[\.\s:—–-]*unresolved",              "item_1b_staff",        "Item 1B: Unresolved Staff Comments"),
    (r"item\s*1c[\.\s:—–-]*cyber",                   "item_1c_cyber",        "Item 1C: Cybersecurity"),
    (r"item\s*1[\.\s:—–-]*business",                 "item_1_business",      "Item 1: Business"),
    (r"item\s*2[\.\s:—–-]*propert",                  "item_2_properties",    "Item 2: Properties"),
    (r"item\s*3[\.\s:—–-]*legal",                    "item_3_legal",         "Item 3: Legal Proceedings"),
    (r"item\s*4[\.\s:—–-]*mine",                     "item_4_mine",          "Item 4: Mine Safety"),
    (r"item\s*5[\.\s:—–-]*market",                   "item_5_market",        "Item 5: Market for Equity"),
    (r"item\s*6[\.\s:—–-]*selected",                 "item_6_selected",      "Item 6: Selected Financial Data"),
    (r"item\s*7a[\.\s:—–-]*quantitative",            "item_7a_market_risk",  "Item 7A: Quantitative Disclosures"),
    (r"item\s*7[\.\s:—–-]*management",               "item_7_mda",           "Item 7: MD&A"),
    (r"item\s*8[\.\s:—–-]*financial",                "item_8_financials",    "Item 8: Financial Statements"),
    (r"item\s*9a[\.\s:—–-]*controls",                "item_9a_controls",     "Item 9A: Controls and Procedures"),
    (r"item\s*9b[\.\s:—–-]*other",                   "item_9b_other",        "Item 9B: Other Information"),
    (r"item\s*9[\.\s:—–-]*change",                   "item_9_accountants",   "Item 9: Disagreements with Accountants"),
    (r"item\s*1[0-4][\.\s:]",                     "item_governance",      "Items 10-14: Corporate Governance"),
    (r"item\s*15[\.\s:—–-]*exhibit",                 "item_15_exhibits",     "Item 15: Exhibits"),

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
    # Some bank annual-report exhibits (e.g. WFC's EX-13) label MD&A "Financial
    # Review" instead — must be anchored ($) so it doesn't match compound
    # headings like "Financial Review — Risk Factors" in a table of contents.
    (r"^financial\s+review$",                       "item_7_mda",           "Item 7: MD&A"),
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

# Anchored recovery patterns for Item-level headers that some filers (TROW,
# IVZ, ...) render as a <table><td> cell — often a table-of-contents hyperlink
# — rather than plain paragraph text. _extract_tables() decomposes non-data
# tables entirely, so without a sentinel this text is lost before it ever
# reaches the plain-text stream, and no amount of post-hoc line scanning
# (unlike the plain-TOC-zone case AAPL/NVDA hit) can recover it.
_ITEM_RECOVERY_PATTERNS: List[Tuple[str, str, str]] = [
    (r"^item\s*1\.?\s*business$", "item_1_business", "Item 1: Business"),
]

_SECTION_PRIORITY: Dict[str, int] = {
    "item_1_business":     10,  "item_1a_risk_factors": 15,
    "item_1b_staff":       16,  "item_1c_cyber":        17,
    "item_2_properties":    20,
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


def _extract_tables(soup: BeautifulSoup, start_idx: int = 0) -> Tuple[Dict[str, Tuple[str, List]], int]:
    """Extract data tables, returning (placeholder -> (markdown, raw), next_free_idx).

    start_idx/next_free_idx let callers extract tables from multiple
    documents into one shared placeholder namespace (see
    _split_embedded_documents) without collisions.
    """
    tables: Dict[str, Tuple[str, List]] = {}
    idx = start_idx

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

    return tables, idx


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _match_section(text: str) -> Optional[Tuple[str, str]]:
    t = text.lower().strip()
    for pattern, section_id, title in SECTION_PATTERNS:
        if re.search(pattern, t):
            return section_id, title
    return None


def _match_section_at(lines: List[str], i: int, total: int) -> Optional[Tuple[str, str]]:
    """
    _match_section on lines[i], with a lookahead fallback: some filers split
    "Item N." and its description across two separate lines/text nodes
    (AMZN, WFC, TROW, ...) — a short "Item N" line that doesn't match alone
    is joined with the next non-empty line and retried.

    Some filers (MSFT, BLK) go further and split the description WORD
    itself across two adjacent inline tags (e.g. "ITEM 1. B" / "USINESS"),
    so a space-joined retry still fails ("B usiness" != "business"). If the
    space join doesn't match, also retry with no separator at all.
    """
    stripped = lines[i].strip()
    match = _match_section(stripped)

    if not match and len(stripped) < 35 and re.search(r"^item\s*\d", stripped.lower()):
        for j in range(i + 1, min(i + 5, total)):
            next_stripped = lines[j].strip()
            if next_stripped and len(next_stripped) < 100:
                match = _match_section(stripped + " " + next_stripped)
                if not match:
                    match = _match_section(stripped + next_stripped)
                break

    return match


def _collect_occurrences(
    lines: List[str], skip_start: int, skip_end: int
) -> Tuple[Dict[str, List[Tuple[int, str, str]]], Dict[str, Tuple[int, str, str]]]:
    """Scan lines[skip_start:skip_end] for section-header candidates."""
    total = len(lines)

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
            for _, sid, t in _HEADER_TABLE_RECOVERY_PATTERNS:
                if t.lower() == title.lower():
                    sentinel_hits[sid] = (i, sid, t)
                    break
            continue

        match = _match_section_at(lines, i, total)

        if match:
            section_id, title = match
            all_occurrences[section_id].append((i, section_id, title))

    return all_occurrences, sentinel_hits


def _select_and_validate(
    lines: List[str],
    all_occurrences: Dict[str, List[Tuple[int, str, str]]],
    sentinel_hits: Dict[str, Tuple[int, str, str]],
    skip_start: int,
    skip_end: int,
) -> List[Tuple[int, str, str]]:
    """
    Hybrid boundary selection — different strategies for Item-level vs
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

    # --- Recovery pass for Item 1: Business ------------------------------
    # It's almost always the very first substantive heading right after the
    # cover page, which commonly lands it INSIDE the 15% TOC-skip window
    # that every other item_* section relies on to avoid TOC/cover-page
    # false positives. Recover it by scanning exactly that excluded window;
    # take the LAST match there (TOC entries like a bare "Item 1." appear
    # first, the real "Item 1. Business" heading follows shortly after).
    if "item_1_business" not in {e[1] for e in validated} and skip_start > 0:
        toc_zone_hits: List[Tuple[int, str, str]] = []
        for i in range(skip_start):
            stripped = lines[i].strip()
            if not stripped or len(stripped) > 250:
                continue
            match = _match_section_at(lines, i, len(lines))
            if match and match[0] == "item_1_business":
                toc_zone_hits.append((i, match[0], match[1]))
        if toc_zone_hits:
            validated.append(toc_zone_hits[-1])
            validated.sort(key=lambda x: x[0])

    return validated


# ---------------------------------------------------------------------------
# Embedded-document splitting
# ---------------------------------------------------------------------------

# Filers occasionally incorporate content BY REFERENCE to a separately-filed
# exhibit (e.g. Wells Fargo's Item 1A/7/8 point to EX-13, its Annual Report
# to Shareholders) instead of including it inline in the 10-K. The downloader
# concatenates such exhibits into one HTML file, delimited by this marker.
# Each embedded document is a full standalone <html>...</html> tree, so they
# must be parsed as SEPARATE BeautifulSoup documents — concatenating the raw
# HTML and parsing it as one soup causes lxml to silently drop everything
# after the first </html> close tag.
_EMBEDDED_DOC_MARKER = re.compile(r"<!--\s*=====\s*embedded document:.*?=====\s*-->")


def _split_embedded_documents(html: str) -> List[str]:
    parts = [p for p in _EMBEDDED_DOC_MARKER.split(html) if p.strip()]
    return parts if parts else [html]


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
# _collect_occurrences can pick up the section_id from plain text.
_FS_SENTINEL_PREFIX = "FS_SECTION_HEADER:"


_HEADER_TABLE_RECOVERY_PATTERNS = _FS_RECOVERY_PATTERNS + _ITEM_RECOVERY_PATTERNS


def _annotate_fs_header_tables(soup: BeautifulSoup) -> None:
    """
    Scan every <table> for a cell (within the first 5 rows) whose text
    matches an fs_* or Item-level header pattern. When found, insert a
    sentinel text node immediately before the <table> so the header
    survives as a plain-text line after _extract_tables replaces the table
    with a placeholder.

    Targets issuers like BAC/WFC where section headers such as
    "Consolidated Statement of Income" live inside <td> cells — sometimes
    after leading empty spacer rows — and issuers like TROW/IVZ where even
    "Item 1. Business" is a table-of-contents hyperlink cell rather than
    plain paragraph text. Either way the text is otherwise lost when the
    table is replaced by a <<<TABLE_N>>> placeholder or decomposed outright.
    A sentinel this early in the document (e.g. a genuine TOC entry) is
    still subject to the normal skip_start/skip_end window in
    _collect_occurrences, so a TOC hyperlink match doesn't win over a real
    later heading — it's just one more candidate line.
    """
    for table_tag in soup.find_all("table"):
        rows = table_tag.find_all("tr", recursive=True)
        matched = False

        # fs_* headers sit near the top of a financial-statement table — keep
        # this scoped to the original 5-row window so a wide table containing
        # BOTH an fs_ cell and (further down) something matching an item-level
        # pattern still resolves to the fs_ header, not gets skipped past it.
        for row in rows[:5]:
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

        if matched:
            continue

        # Item-level TOC entries (TROW, IVZ) can be dozens of rows into a long
        # TOC table — only fall back to this wider scan if no fs_ header
        # already claimed this table above.
        for row in rows[:60]:
            cells = row.find_all(["td", "th"])
            cell_texts = [c.get_text(separator=" ", strip=True).lower() for c in cells]

            for cell_text in cell_texts:
                if not cell_text or len(cell_text) > 120:
                    continue
                for pattern, section_id, title in _ITEM_RECOVERY_PATTERNS:
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

            # AMZN/WFC split the heading across ADJACENT cells in the same row
            # ("Item 1." in one <td>, "Business" in the next) rather than one
            # cell holding the whole title. Concatenating non-empty cell texts
            # with no separator recovers it; the anchored ^...$ pattern still
            # rejects TOC rows that carry a trailing page-number cell (e.g.
            # "item 1." + "business" + "3" -> "item 1.business3" fails "$").
            joined = "".join(t for t in cell_texts if t)
            if joined and len(joined) <= 120:
                for pattern, section_id, title in _ITEM_RECOVERY_PATTERNS:
                    if re.search(pattern, joined):
                        sentinel = soup.new_string(f"\n{_FS_SENTINEL_PREFIX}{title}\n")
                        table_tag.insert_before(sentinel)
                        logger.debug(f"  Pre-annotated table (joined cells): '{title}' ({joined[:50]})")
                        matched = True
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
    segments     = _split_embedded_documents(cleaned_html)

    # Each embedded document (the 10-K wrapper, plus any incorporated-by-
    # reference exhibit like EX-13) is parsed as its own BeautifulSoup tree
    # AND boundary-validated independently, then the resulting section lists
    # are concatenated in document order. This matters because monotonic
    # Item-priority validation must not span documents: the 10-K wrapper's
    # own stub sections (e.g. "Items 10-14: Corporate Governance") already
    # advance the priority watermark, which would wrongly reject the exhibit's
    # real, lower-priority sections (Risk Factors, MD&A) as "out of order" if
    # validated together. Scoping validation per segment avoids that, and
    # also keeps each document's own 15%-97% TOC-skip window correctly local.
    tables: Dict[str, Tuple[str, List]] = {}
    next_table_idx = 0
    lines: List[str] = []
    boundaries: List[Tuple[int, str, str]] = []

    for seg_html in segments:
        soup = BeautifulSoup(seg_html, "lxml")
        for tag in soup(["script", "style", "meta", "link", "head", "noscript"]):
            tag.decompose()

        # Pre-annotate financial statement header tables before extraction.
        # Some issuers (BAC, WFC) put headers like "Consolidated Statement of
        # Income" inside <td> cells of the statement tables rather than as
        # stand-alone <div> or <p> elements.  After _extract_tables replaces
        # those tables with <<<TABLE_N>>> placeholders, the text disappears
        # from the plain-text stream and boundary detection misses it.  By
        # inserting a sentinel text node BEFORE the table here, the marker
        # survives into get_text() output so boundary detection finds it.
        _annotate_fs_header_tables(soup)

        seg_tables, next_table_idx = _extract_tables(soup, start_idx=next_table_idx)
        tables.update(seg_tables)

        seg_text = soup.get_text(separator="\n")
        seg_text = re.sub(r"\n{4,}", "\n\n\n", seg_text)
        seg_lines = seg_text.split("\n")

        seg_total      = len(seg_lines)
        seg_skip_start = int(seg_total * 0.15)
        seg_skip_end   = int(seg_total * 0.97)
        seg_occurrences, seg_sentinels = _collect_occurrences(
            seg_lines, seg_skip_start, seg_skip_end
        )
        seg_boundaries = _select_and_validate(
            seg_lines, seg_occurrences, seg_sentinels, seg_skip_start, seg_skip_end
        )

        offset = len(lines)
        boundaries.extend((i + offset, s, t) for (i, s, t) in seg_boundaries)
        lines.extend(seg_lines)

    # Disambiguate section_ids that recur across segments (e.g. the 10-K
    # wrapper's stub "Notes to Financial Statements" pointer vs. the real one
    # in the exhibit) so each stays independently addressable by parent_id —
    # otherwise ParsedDocument.section_by_id() would always resolve to the
    # first (possibly stub) match regardless of which section a chunk came
    # from. Display titles are untouched; only the internal id gets suffixed.
    seen_ids: Dict[str, int] = {}
    disambiguated: List[Tuple[int, str, str]] = []
    for line_no, sid, title in boundaries:
        seen_ids[sid] = seen_ids.get(sid, 0) + 1
        if seen_ids[sid] > 1:
            sid = f"{sid}__{seen_ids[sid]}"
        disambiguated.append((line_no, sid, title))
    boundaries = disambiguated

    logger.info(
        f"  → {len(boundaries)} sections detected, "
        f"{len(tables)} tables extracted"
    )

    sections: List[ParsedSection] = []

    if not boundaries:
        logger.warning(f"  No sections found for {ticker} FY{fiscal_year}; using full document")
        full_text = "\n".join(lines)
        sections.append(_build_section("full_document", "Full Document", full_text, tables, 0))
    else:
        for i, (start_line, section_id, title) in enumerate(boundaries):
            end_line   = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            slice_text = "\n".join(lines[start_line + 1 : end_line])
            section    = _build_section(section_id, title, slice_text, tables, i)
            if section.content_blocks:
                sections.append(section)

    doc = ParsedDocument(
        # Deterministic, not the random-UUID default: chunks embedded from an
        # earlier parse of the same filing reference doc_id as a foreign key
        # into ParentStore for full-section context. A random doc_id would
        # orphan every already-embedded chunk each time a filing gets
        # re-parsed (e.g. a parser bug fix) without also being re-embedded —
        # ParentStore silently falls back to the smaller child-chunk text,
        # degrading (not breaking) answers in a way that's easy to miss.
        doc_id=f"{ticker}_{fiscal_year}",
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


def _parse_record_worker(record: dict, parsed_dir: Path) -> Optional[ParsedDocument]:
    """Top-level worker so ProcessPoolExecutor can pickle it on Windows (spawn)."""
    out_file = parsed_dir / f"{record['ticker']}_{record['fiscal_year']}.json"
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
        return doc
    except Exception as exc:
        logger.error(f"Failed to parse {record['ticker']} FY{record['fiscal_year']}: {exc}")
        return None


def parse_all_filings(
    manifest:   List[dict],
    parsed_dir: Path,
) -> List[ParsedDocument]:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    documents:  List[ParsedDocument] = []
    to_parse:   List[dict]           = []

    for record in manifest:
        out_file = parsed_dir / f"{record['ticker']}_{record['fiscal_year']}.json"
        if out_file.exists():
            logger.info(f"Skipping {record['ticker']} FY{record['fiscal_year']} (already parsed)")
            with open(out_file, encoding="utf-8") as f:
                documents.append(ParsedDocument.model_validate(json.load(f)))
        else:
            to_parse.append(record)

    if to_parse:
        # Each ProcessPoolExecutor worker is a whole separate Python process
        # (lxml/BeautifulSoup imports and all) — settings.parse_workers
        # defaults to 1 on the assumption of a memory-capped host, in which
        # case skip the pool entirely rather than pay for even one extra
        # process. See config.py's parse_workers for why os.cpu_count() is
        # never used to size this.
        max_workers = min(len(to_parse), settings.parse_workers)
        if max_workers <= 1:
            logger.info(f"Parsing {len(to_parse)} filings sequentially …")
            for rec in to_parse:
                doc = _parse_record_worker(rec, parsed_dir)
                if doc is not None:
                    documents.append(doc)
        else:
            logger.info(f"Parsing {len(to_parse)} filings in parallel (workers={max_workers}) …")
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_parse_record_worker, rec, parsed_dir): rec
                    for rec in to_parse
                }
                for future in as_completed(futures):
                    rec = futures[future]
                    try:
                        doc = future.result()
                        if doc is not None:
                            documents.append(doc)
                    except Exception as exc:
                        logger.error(
                            f"Worker failed for {rec['ticker']} FY{rec['fiscal_year']}: {exc}"
                        )

    logger.success(f"Parsed {len(documents)} / {len(manifest)} filings")
    return documents
