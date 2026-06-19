"""
ParentStore — in-memory index of all ParsedDocuments.

At retrieval time, the reranker picks the best child chunks (precise retrieval).
We then look up the full parent section text to pass to the LLM (rich context).

All 36 parsed JSON files are loaded once on first access and kept in RAM.
36 documents at ~1-5 MB each is well within normal machine memory.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from config import settings
from models import ParsedDocument


class ParentStore:
    def __init__(self, parsed_dir: Path = settings.parsed_dir):
        self._parsed_dir = parsed_dir
        self._docs: Dict[str, ParsedDocument] = {}   # doc_id  → ParsedDocument
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        for path in self._parsed_dir.glob("*.json"):
            try:
                with open(path, encoding="utf-8") as f:
                    doc = ParsedDocument(**json.load(f))
                self._docs[doc.doc_id] = doc
            except Exception as exc:
                logger.warning(f"ParentStore: could not load {path.name}: {exc}")
        self._loaded = True
        logger.debug(f"ParentStore: {len(self._docs)} documents loaded")

    # ------------------------------------------------------------------

    def get_section_text(self, doc_id: str, section_id: str) -> str:
        """Return the full concatenated text of a section (the 'parent' context)."""
        self._load_all()
        doc = self._docs.get(doc_id)
        if not doc:
            return ""
        section = doc.section_by_id(section_id)
        return section.full_text() if section else ""

    def get_doc(self, doc_id: str) -> Optional[ParsedDocument]:
        self._load_all()
        return self._docs.get(doc_id)

    def reload(self) -> None:
        """Force a fresh load — call after adding new parsed files."""
        self._docs.clear()
        self._loaded = False
        self._load_all()


# Module-level singleton — imported by retriever and generation layers
parent_store = ParentStore()
