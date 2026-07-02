"""
Hierarchical chunker: ParsedDocument → List[Chunk]

Parent nodes  = ParsedSection objects (stored separately, fetched at retrieval time)
Child nodes   = Chunk objects (embedded + stored in vector DB)

Rules per block type:
  text/footnote → sentence-boundary chunking, 400-token max, 2-sentence overlap
  table         → one chunk if ≤ 400 tokens; split by row groups otherwise
                  (header row repeated in every sub-chunk)
"""

import json
import re
from pathlib import Path
from typing import List, Generator

import nltk
import tiktoken
from loguru import logger

from config import settings
from models import Chunk, ContentBlock, ParsedDocument, ParsedSection

# ---------------------------------------------------------------------------
# Bootstrap NLTK sentence tokenizer (downloads once, silent after that).
# punkt_tab is the new name in NLTK >= 3.8.1; older versions use punkt.
# We download both so either version of NLTK works.
# ---------------------------------------------------------------------------
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except (LookupError, OSError):
        # OSError fires when the directory exists but a file inside it is missing
        nltk.download(_resource, quiet=True)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Table splitting
# ---------------------------------------------------------------------------

def _split_table(text: str, max_tokens: int) -> List[str]:
    """
    If a markdown table exceeds max_tokens, split it into row-group sub-chunks.

    The markdown header + separator are repeated in every sub-chunk.
    Additionally, leading "context rows" (date headers, year labels) that
    contain no financial figures (5+ digit integers) are also repeated so
    every sub-chunk knows which fiscal year its numbers belong to.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    lines = text.strip().split("\n")
    if len(lines) < 3:
        return [text]

    header    = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    if not data_rows:
        return [text]

    # Identify leading "context rows": rows with no large financial integers.
    # These carry the year/date headers (e.g. "September 28, 2024") and must
    # appear in every sub-chunk so the LLM can match numbers to fiscal years.
    context_rows: List[str] = []
    for row in data_rows[:8]:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        has_financial_data = any(
            re.match(r"^\d{5,}$", c)                    # bare 5+ digit int
            or re.match(r"^\d{1,3}(?:,\d{3}){2,}$", c) # e.g. 391,035
            for c in cells
        )
        if not has_financial_data:
            context_rows.append(row)
        else:
            break

    effective_rows = data_rows[len(context_rows):]

    fixed_header = "\n".join([header, separator] + context_rows) + "\n"
    header_tokens = _count_tokens(fixed_header)

    sample      = "\n".join(effective_rows[:10])
    avg_row_tok = max(1, _count_tokens(sample) // max(1, min(10, len(effective_rows))))
    rows_per_chunk = max(1, (max_tokens - header_tokens) // avg_row_tok)

    sub_chunks = []
    for i in range(0, len(effective_rows), rows_per_chunk):
        batch      = effective_rows[i : i + rows_per_chunk]
        chunk_text = "\n".join([header, separator] + context_rows + batch)
        sub_chunks.append(chunk_text)

    return sub_chunks if sub_chunks else [text]


# ---------------------------------------------------------------------------
# Text splitting (sentence-boundary)
# ---------------------------------------------------------------------------

def _split_text(text: str, max_tokens: int, overlap_sentences: int) -> List[str]:
    """
    Split text into chunks that fit within max_tokens using sentence boundaries.
    The last `overlap_sentences` sentences of each chunk are carried into the next.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks   : List[str] = []
    current  : List[str] = []
    cur_toks : int       = 0

    for sent in sentences:
        sent_toks = _count_tokens(sent)

        # Single sentence longer than max — emit it alone
        if sent_toks > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current  = current[-overlap_sentences:]
                cur_toks = _count_tokens(" ".join(current))
            chunks.append(sent)
            current  = []
            cur_toks = 0
            continue

        if cur_toks + sent_toks > max_tokens and current:
            chunks.append(" ".join(current))
            # Carry overlap into next chunk
            current  = current[-overlap_sentences:]
            cur_toks = _count_tokens(" ".join(current))

        current.append(sent)
        cur_toks += sent_toks

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _chunks_from_block(
    block    : ContentBlock,
    section  : ParsedSection,
    doc      : ParsedDocument,
    position : int,
    max_tokens: int,
    overlap_sentences: int,
) -> Generator[Chunk, None, None]:
    """Yield one or more Chunk objects from a single ContentBlock."""

    base_meta = dict(
        parent_id    = section.section_id,
        doc_id       = doc.doc_id,
        company      = doc.company,
        ticker       = doc.ticker,
        filing_type  = doc.filing_type,
        fiscal_year  = doc.fiscal_year,
        section_name = section.title,
        chunk_type   = block.block_type,
    )

    if block.block_type == "table":
        # Prepend a context header so table chunks are discoverable by both
        # BM25 and dense search.  Without this, XBRL tables start with
        # "| 0 | 1 | 2 | ..." (numeric column indices) which dominates
        # tokenisation and makes the chunk invisible to financial queries.
        context_header = (
            f"{doc.company} ({doc.ticker}) FY{doc.fiscal_year} — "
            f"{section.title}\n"
        )
        header_toks  = _count_tokens(context_header)
        table_budget = max(50, max_tokens - header_toks)
        sub_chunks   = _split_table(block.text, table_budget)
        for sub in sub_chunks:
            enriched = context_header + sub
            yield Chunk(
                **base_meta,
                text        = enriched,
                token_count = _count_tokens(sub),
                position    = position,
            )
            position += 1

    else:  # text or footnote
        sub_chunks = _split_text(block.text, max_tokens, overlap_sentences)
        for sub in sub_chunks:
            yield Chunk(
                **base_meta,
                text        = sub,
                token_count = _count_tokens(sub),
                position    = position,
            )
            position += 1


def chunk_document(
    doc              : ParsedDocument,
    max_tokens       : int = settings.max_chunk_tokens,
    overlap_sentences: int = settings.chunk_overlap_sentences,
) -> List[Chunk]:
    """Convert a ParsedDocument into a flat list of Chunk objects."""
    chunks   : List[Chunk] = []
    position : int         = 0

    for section in doc.sections:
        if not section.content_blocks:
            continue

        for block in section.content_blocks:
            if not block.text.strip():
                continue

            for chunk in _chunks_from_block(
                block, section, doc, position, max_tokens, overlap_sentences
            ):
                chunks.append(chunk)
                position += 1

    return chunks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_all_documents(
    documents  : List[ParsedDocument],
    chunks_dir : Path = settings.chunks_dir,
) -> List[Chunk]:
    """
    Chunk every document, save per-document JSON files, return all chunks.

    Output: chunks_dir/<TICKER>_<YEAR>_chunks.json
    Each file is a JSON array of Chunk objects.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Chunk] = []

    for doc in documents:
        out_file = chunks_dir / f"{doc.ticker}_{doc.fiscal_year}_chunks.json"

        if out_file.exists():
            logger.info(f"Skipping {doc.ticker} FY{doc.fiscal_year} (chunks already exist)")
            with open(out_file) as f:
                loaded = [Chunk.model_validate(c) for c in json.load(f)]
            all_chunks.extend(loaded)
            continue

        doc_chunks = chunk_document(doc)

        with open(out_file, "w") as f:
            json.dump([c.model_dump() for c in doc_chunks], f, indent=2)

        table_n = sum(1 for c in doc_chunks if c.chunk_type == "table")
        text_n  = sum(1 for c in doc_chunks if c.chunk_type == "text")
        foot_n  = sum(1 for c in doc_chunks if c.chunk_type == "footnote")

        logger.success(
            f"{doc.ticker} FY{doc.fiscal_year}: "
            f"{len(doc_chunks)} chunks  "
            f"(text={text_n}, table={table_n}, footnote={foot_n})"
        )
        all_chunks.extend(doc_chunks)

    logger.success(f"Chunking complete — {len(all_chunks)} total chunks across {len(documents)} documents")
    return all_chunks


def load_chunks(chunks_dir: Path = settings.chunks_dir) -> List[Chunk]:
    """Load all previously saved chunk files from chunks_dir."""
    all_chunks: List[Chunk] = []
    for path in sorted(chunks_dir.glob("*_chunks.json")):
        with open(path) as f:
            all_chunks.extend(Chunk.model_validate(c) for c in json.load(f))
    return all_chunks
