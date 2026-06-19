from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import uuid


class ContentBlock(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    block_type: Literal["text", "table", "footnote"]
    text: str
    raw_table: Optional[List[List[Optional[str]]]] = None
    position: int = 0


class ParsedSection(BaseModel):
    section_id: str
    title: str
    content_blocks: List[ContentBlock] = []
    order: int = 0

    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.content_blocks)


class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str
    company: str
    ticker: str
    filing_type: str = "10-K"
    fiscal_year: int
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    sections: List[ParsedSection] = []
    parsed_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def section_by_id(self, section_id: str) -> Optional[ParsedSection]:
        return next((s for s in self.sections if s.section_id == section_id), None)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str           # ParsedSection.section_id
    doc_id: str              # ParsedDocument.doc_id
    text: str
    company: str
    ticker: str
    filing_type: str
    fiscal_year: int
    section_name: str
    chunk_type: Literal["text", "table", "footnote"]
    token_count: int
    position: int = 0


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    parent_text: str         # full section text — passed to LLM as context


class QueryResult(BaseModel):
    query: str
    answer: str
    citations: List[dict]
    chunks_used: List[RetrievedChunk]
    query_type: str
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
