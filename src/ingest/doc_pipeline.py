"""Document ingest pipeline — papers (PDF) and decks (PDF/PPTX).

pending -> parsing -> embedding -> indexed | failed

Documents are text-only: no frames, no CLIP, no ffmpeg. Chunks land in the
text collection (moments_text) alongside video transcript chunks, tagged with
kind=paper|deck and page/slide locators so the search path can build
cross-source citations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from prefect import flow, task

from .. import db, storage
from ..config import DOC_KEY_PREFIX, TEXT_EMBED_VERSION
from ..rag import vector_store
from ..rag.embeddings import embed_docs
from .paper import parse_paper
from .deck import parse_deck


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@task(name="fetch-doc", retries=2, retry_delay_seconds=[30, 120])
def t_fetch_doc(doc_id: str, user_id: str) -> str:
    """Download document from storage to scratch; duplicate check."""
    db.set_doc_status(doc_id, "parsing")
    row = db.get_document(doc_id)
    if row is None:
        raise ValueError(f"no manifest row for {doc_id}")

    scratch = Path(f"/tmp/{doc_id}{Path(row['filename'] or 'doc').suffix}")
    storage.download_to(row["storage_key"], scratch)

    source_hash = _sha256(scratch)
    db.set_doc_status(doc_id, "parsing", progress=0.1)

    dup = db.find_doc_duplicate(user_id, source_hash, exclude_id=doc_id)
    if dup:
        scratch.unlink(missing_ok=True)
        db.set_doc_status(doc_id, "failed", error=f"duplicate of {dup['id']}")
        return ""
    return str(scratch)


@task(name="parse-embed-doc", retries=1, retry_delay_seconds=30)
def t_parse_embed(doc_id: str, user_id: str, path: str, kind: str) -> int:
    """Parse document -> embed chunks -> upsert to text collection."""
    p = Path(path)
    if kind == "deck":
        chunks = parse_deck(p)
    else:
        chunks = parse_paper(p)

    if not chunks:
        raise RuntimeError(f"No text could be extracted from {p.name}")

    db.set_doc_status(doc_id, "embedding", progress=0.3)

    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)

    texts = [c["text"] for c in chunks]
    vecs = embed_docs(texts)

    payloads = []
    for i, c in enumerate(chunks):
        payloads.append({
            "user_id": user_id,
            "video_id": doc_id,
            "modality": "text",
            "kind": kind,
            "page": c.get("page"),
            "slide": c.get("slide"),
            "t_start": 0.0,
            "t_end": 0.0,
            "ms": 0,
            "text": c["text"],
            "embed_version": TEXT_EMBED_VERSION,
        })

    vector_store.upsert_chunks(user_id, doc_id, vecs, payloads=payloads)
    db.set_doc_status(doc_id, "indexed", chunk_count=len(chunks),
                      embed_version=TEXT_EMBED_VERSION, progress=1.0)
    return len(chunks)


@flow(name="ms-ingest-document", log_prints=True, timeout_seconds=1800)
def ingest_document(doc_id: str, user_id: str) -> dict:
    attempt = db.bump_doc_attempts(doc_id)
    path: str | None = None
    try:
        row = db.get_document(doc_id)
        if row is None:
            raise ValueError(f"no manifest row for {doc_id}")
        kind = row["kind"]

        path = t_fetch_doc(doc_id, user_id)
        if not path:
            print(f"[ingest-doc] {doc_id} skipped (duplicate)")
            return {"doc_id": doc_id, "skipped": True}

        n = t_parse_embed(doc_id, user_id, path, kind)
        print(f"[ingest-doc] {doc_id} indexed: {n} chunks (attempt {attempt})")
        return {"doc_id": doc_id, "chunks": n}
    except Exception as exc:
        db.set_doc_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if path:
            Path(path).unlink(missing_ok=True)
