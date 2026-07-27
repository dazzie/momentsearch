"""Admin API — unified /admin/sources view across videos and documents.

Provides the assignment-expected /admin/* endpoints that present all
ingested sources (videos + documents) through a single interface.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Header, HTTPException

from .. import db
from ..config import ADMIN_TOKEN, DEFAULT_USER_ID
from ..rag import vector_store
from .. import storage
from ..samples import is_sample

router = APIRouter(prefix="/admin", tags=["admin"])

_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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


def _video_to_source(row: dict) -> dict:
    return {
        "id": row["id"],
        "type": "video",
        "kind": row.get("source", "upload"),
        "title": row.get("title") or row["id"],
        "status": row.get("status"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "is_sample": is_sample(row["id"]),
    }


def _doc_to_source(row: dict) -> dict:
    return {
        "id": row["id"],
        "type": "document",
        "kind": row.get("kind", "paper"),
        "title": row.get("title") or row["id"],
        "filename": row.get("filename"),
        "status": row.get("status"),
        "error": row.get("error"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


@router.get("/sources", dependencies=[Depends(require_auth)])
def list_sources(uid: str = Depends(user_id), status: str | None = None,
                 type: str | None = None):
    """Unified listing of all ingested sources (videos + documents)."""
    sources: list[dict] = []
    if type is None or type == "video":
        sources.extend(_video_to_source(r) for r in db.list_videos(uid, status=status))
    if type is None or type == "document":
        sources.extend(_doc_to_source(r) for r in db.list_documents(uid, status=status))
    sources.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return {"sources": sources, "count": len(sources)}


@router.get("/sources/{source_id}", dependencies=[Depends(require_auth)])
def get_source(source_id: str, uid: str = Depends(user_id)):
    """Get a single source by ID (video or document)."""
    if source_id.startswith("doc_"):
        row = db.get_document(source_id)
        if row is None or row["user_id"] != uid:
            raise HTTPException(404, "Source not found.")
        return _doc_to_source(row)
    row = db.get_video(source_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Source not found.")
    return _video_to_source(row)


@router.delete("/sources/{source_id}", dependencies=[Depends(require_auth)])
def delete_source(source_id: str, uid: str = Depends(user_id)):
    """Delete a source (video or document) and all its indexed data."""
    if source_id.startswith("doc_"):
        row = db.get_document(source_id)
        if row is None or row["user_id"] != uid:
            raise HTTPException(404, "Source not found.")
        vector_store.delete_video(uid, source_id)
        if row.get("storage_key"):
            storage.delete_key(row["storage_key"])
        db.delete_document(source_id)
        return {"ok": True, "source_id": source_id}
    if is_sample(source_id):
        raise HTTPException(403, "Sample videos cannot be deleted.")
    row = db.get_video(source_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Source not found.")
    vector_store.delete_video(uid, source_id)
    storage.delete_prefix(storage.frame_prefix(uid, source_id))
    if row.get("storage_key"):
        storage.delete_key(row["storage_key"])
    db.delete_video(source_id)
    return {"ok": True, "source_id": source_id}
