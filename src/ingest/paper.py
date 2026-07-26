"""PDF paper ingestion — turns research papers into searchable text chunks.

Papers have no time dimension (unlike video transcripts), so t_start/t_end are
always 0.0 — the caller uses page numbers to locate content instead. Text is
extracted via pymupdf (no OCR); scanned-image PDFs silently yield no chunks
rather than blowing up.

Chunking strategy: split long pages into ~500-word pieces so each chunk stays
within reasonable embedding context. Short pages (< 50 words, e.g. a title page
or a blank separator) get merged forward into the next page's text to avoid
near-empty chunks.
"""
from __future__ import annotations

from pathlib import Path

# Target words per chunk — long pages are split, short pages are merged.
_CHUNK_WORDS = 500
_SHORT_PAGE_WORDS = 50


def _split_text(text: str, page: int, max_words: int = _CHUNK_WORDS) -> list[dict]:
    """Break *text* into chunks of ~max_words, each tagged with *page*."""
    words = text.split()
    if not words:
        return []
    chunks: list[dict] = []
    for i in range(0, len(words), max_words):
        chunk_text = " ".join(words[i : i + max_words])
        chunks.append({
            "text": chunk_text,
            "page": page,
            "t_start": 0.0,
            "t_end": 0.0,
        })
    return chunks


def parse_paper(path: Path) -> list[dict]:
    """Extract text from a PDF paper and return chunked passages.

    Each chunk carries its 1-indexed page number. Pages with fewer than
    50 words are merged into the next page so we don't produce trivially
    small chunks (common for title pages and section dividers).
    """
    import fitz  # pymupdf

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        print(f"[paper] failed to open {path.name} ({type(exc).__name__}: {exc})")
        return []

    chunks: list[dict] = []
    carry_text = ""
    carry_page: int | None = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            text = (page.get_text() or "").strip()
        except Exception:
            text = ""

        if not text:
            continue

        # Merge short pages forward.
        if len(text.split()) < _SHORT_PAGE_WORDS:
            if carry_text:
                carry_text += "\n" + text
            else:
                carry_text = text
                carry_page = page_num + 1  # 1-indexed
            continue

        # Prepend any carried-over text from previous short page(s).
        if carry_text:
            text = carry_text + "\n" + text
            effective_page = carry_page or (page_num + 1)
            carry_text = ""
            carry_page = None
        else:
            effective_page = page_num + 1

        chunks.extend(_split_text(text, effective_page))

    # Flush any trailing carried text (e.g. paper ends with a short page).
    if carry_text and carry_page is not None:
        chunks.extend(_split_text(carry_text, carry_page))

    doc.close()
    return chunks
