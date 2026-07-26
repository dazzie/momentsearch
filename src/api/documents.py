"""Document registration API — papers (PDF) and decks (PDF/PPTX).

Upload flow (same presign pattern as videos):
  1. POST /api/documents/presign  -> scoped PUT URL
  2. browser PUTs the file to storage
  3. POST /api/documents          -> HEAD-verify, insert pending row,
                                     schedule Prefect run, 202
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .. import config, db, jobs, storage
from ..config import (
    ADMIN_TOKEN,
    ALLOWED_DOC_TYPES,
    DEFAULT_USER_ID,
    DOC_KEY_PREFIX,
    MAX_UPLOAD_MB,
)
from ..rag import vector_store

router = APIRouter(prefix="/api/documents", tags=["documents"])

_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
_KIND_MAP = {".pdf": "paper", ".pptx": "deck"}


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not ADMIN_TOKEN:
        return
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "Missing or invalid bearer token.")


def user_id(x_user_id: str | None = Header(default=None)) -> str:
    uid = (x_user_id or DEFAULT_USER_ID).strip()
    if not _USER_RE.match(uid):
        raise HTTPException(400, "Invalid X-User-Id.")
    return uid


def _doc_key(user_id: str, doc_id: str, ext: str) -> str:
    return f"{DOC_KEY_PREFIX}{user_id}/{doc_id}{ext}"


# ── Presign ───────────────────────────────────────────────────────────────────

class PresignRequest(BaseModel):
    filename: str
    content_type: str
    size: int


@router.post("/presign", dependencies=[Depends(require_auth)])
def presign(req: PresignRequest, uid: str = Depends(user_id)):
    if req.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB}MB limit.")
    if not any(req.content_type.startswith(t) for t in ALLOWED_DOC_TYPES):
        raise HTTPException(415, "Only PDF and PPTX files are accepted.")
    ext = Path(req.filename or "document.pdf").suffix.lower() or ".pdf"
    if not _EXT_RE.match(ext):
        ext = ".pdf"
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    key = _doc_key(uid, doc_id, ext)
    if not storage.presign_capable():
        return {"mode": "direct", "doc_id": doc_id, "key": key,
                "url": f"/api/documents/{doc_id}/content?key={key}",
                "headers": {"Content-Type": req.content_type}}
    signed = storage.presign_put(key, req.content_type)
    return {"mode": "presigned", "doc_id": doc_id, "key": key, **signed}


@router.put("/{doc_id}/content", dependencies=[Depends(require_auth)])
async def upload_direct(doc_id: str, key: str, request: Request,
                        uid: str = Depends(user_id)):
    """Dev-only direct upload (STORAGE_PROVIDER=local)."""
    if storage.presign_capable():
        raise HTTPException(400, "Use the presigned URL to upload.")
    if not key.startswith(f"{DOC_KEY_PREFIX}{uid}/{doc_id}"):
        raise HTTPException(403, "Key does not belong to this upload.")
    dest = storage.local_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB}MB limit.")
            out.write(chunk)
    return {"ok": True, "key": key, "size": size}


# ── Register ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    doc_id: str
    key: str
    title: str | None = None
    kind: str | None = None  # paper | deck — auto-detected from extension if omitted


@router.post("", status_code=202, dependencies=[Depends(require_auth)])
def register(req: RegisterRequest, uid: str = Depends(user_id)):
    if not req.key.startswith(f"{DOC_KEY_PREFIX}{uid}/{req.doc_id}"):
        raise HTTPException(403, "Key does not belong to this user/upload.")
    meta = storage.head(req.key)
    if meta is None:
        raise HTTPException(404, "Object not found — did the upload finish?")
    if meta["size"] > MAX_UPLOAD_MB * 1024 * 1024:
        storage.delete_key(req.key)
        raise HTTPException(413, f"Object exceeds the {MAX_UPLOAD_MB}MB limit.")

    ext = Path(req.key).suffix.lower()
    kind = req.kind or _KIND_MAP.get(ext, "paper")
    if kind not in ("paper", "deck"):
        raise HTTPException(400, "kind must be 'paper' or 'deck'.")

    title = req.title or Path(req.key).stem
    filename = Path(req.key).name
    row = db.upsert_pending_doc({
        "id": req.doc_id, "user_id": uid, "kind": kind,
        "title": title, "filename": filename,
        "storage_key": req.key, "source_hash": None,
    })

    flow_run_id = jobs.enqueue_document(row["id"], uid)
    return {"doc_id": row["id"], "status": row["status"], "flow_run_id": flow_run_id}


# ── Status / lifecycle ────────────────────────────────────────────────────────

_PUBLIC_FIELDS = ("id", "kind", "title", "filename", "status", "error",
                  "chunk_count", "progress", "attempts", "created_at", "updated_at")


def _public(row: dict) -> dict:
    return {k: row.get(k) for k in _PUBLIC_FIELDS}


@router.get("")
def list_documents(uid: str = Depends(user_id), status: str | None = None):
    return {"documents": [_public(r) for r in db.list_documents(uid, status=status)]}


@router.get("/{doc_id}")
def get_document(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    return _public(row)


@router.delete("/{doc_id}", dependencies=[Depends(require_auth)])
def delete(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    vector_store.delete_video(uid, doc_id)
    if row.get("storage_key"):
        storage.delete_key(row["storage_key"])
    db.delete_document(doc_id)
    return {"ok": True, "doc_id": doc_id}


@router.post("/{doc_id}/retry", status_code=202, dependencies=[Depends(require_auth)])
def retry(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    db.set_doc_status(doc_id, "pending", error=None)
    flow_run_id = jobs.enqueue_document(doc_id, uid)
    return {"doc_id": doc_id, "status": "pending", "flow_run_id": flow_run_id}
