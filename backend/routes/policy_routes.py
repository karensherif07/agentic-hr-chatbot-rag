"""
policy_routes.py — Lets an admin add/remove HR policy PDFs from the UI
instead of hardcoding ARABIC_PDF_ENTRIES / ENGLISH_PDF_ENTRIES in setup.py.

Requires the `policy_documents` table (see migration in
backend/migrations/001_policy_documents.sql) and the corresponding change
in setup.py that reads from this table instead of the hardcoded lists.
"""

import os
import re
import shutil
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text

from database import get_db
from deps import require_admin, load_models_once, models
from policy_sync import sync_active_policies_to_disk
from routes.admin_routes import log_admin_action

router = APIRouter(prefix="/api/admin/policies", tags=["policies"])

POLICY_DIR = os.environ.get("POLICY_PDF_DIR", "policies")
os.makedirs(POLICY_DIR, exist_ok=True)


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(name))
    return f"{uuid.uuid4().hex[:8]}_{base}"


@router.get("")
def list_policies(emp: dict = Depends(require_admin)):
    with get_db() as db:
        rows = db.execute(text("""
            SELECT pd.id, pd.file_path, pd.doc_name, pd.lang, pd.is_active, pd.uploaded_at,
                   e.full_name AS uploaded_by_name
            FROM policy_documents pd
            LEFT JOIN employees e ON e.id = pd.uploaded_by
            ORDER BY lang, doc_name
        """)).fetchall()
    data = [dict(r._mapping) for r in rows]
    for r in data:
        r["uploaded_at"] = r["uploaded_at"].isoformat() if r["uploaded_at"] else None
    return {"rows": data}


@router.post("")
async def upload_policy(
    file: UploadFile = File(...),
    doc_name: str = Form(...),
    lang: str = Form(...),   # "arabic" | "english"
    emp: dict = Depends(require_admin),
):
    if lang not in ("arabic", "english"):
        raise HTTPException(400, "lang must be 'arabic' or 'english'.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file.")

    stored_name = _safe_filename(file.filename)
    file_path = os.path.join(POLICY_DIR, stored_name)
    with open(file_path, "wb") as f:
        f.write(contents)

    with get_db() as db:
        row = db.execute(text("""
            INSERT INTO policy_documents (file_path, doc_name, lang, is_active, uploaded_by, uploaded_at, file_data)
            VALUES (:fp, :dn, :lang, TRUE, :uid, NOW(), :data)
            RETURNING id
        """), {"fp": file_path, "dn": doc_name, "lang": lang, "uid": emp["id"], "data": contents}).fetchone()

    log_admin_action(emp["id"], "upload_policy", "policy_documents", row[0], notes=doc_name)
    return {"id": row[0], "file_path": file_path, "note": "Uploaded. Click 'Rebuild Index' to activate it in retrieval."}


@router.post("/{policy_id}/deactivate")
def deactivate_policy(policy_id: int, emp: dict = Depends(require_admin)):
    with get_db() as db:
        result = db.execute(text(
            "UPDATE policy_documents SET is_active = FALSE WHERE id = :id RETURNING id"
        ), {"id": policy_id}).fetchone()
    if not result:
        raise HTTPException(404, "Policy not found.")
    log_admin_action(emp["id"], "deactivate_policy", "policy_documents", policy_id)
    return {"ok": True, "note": "Deactivated. Click 'Rebuild Index' to remove it from retrieval."}


@router.post("/{policy_id}/activate")
def activate_policy(policy_id: int, emp: dict = Depends(require_admin)):
    with get_db() as db:
        result = db.execute(text(
            "UPDATE policy_documents SET is_active = TRUE WHERE id = :id RETURNING id"
        ), {"id": policy_id}).fetchone()
    if not result:
        raise HTTPException(404, "Policy not found.")
    log_admin_action(emp["id"], "activate_policy", "policy_documents", policy_id)
    return {"ok": True}


@router.delete("/{policy_id}")
def delete_policy(policy_id: int, emp: dict = Depends(require_admin)):
    with get_db() as db:
        row = db.execute(text(
            "SELECT file_path FROM policy_documents WHERE id = :id"
        ), {"id": policy_id}).fetchone()
        if not row:
            raise HTTPException(404, "Policy not found.")
        db.execute(text("DELETE FROM policy_documents WHERE id = :id"), {"id": policy_id})

    try:
        if os.path.exists(row[0]):
            os.remove(row[0])
    except Exception as e:
        print(f"[policy] file delete failed: {e}")

    log_admin_action(emp["id"], "delete_policy", "policy_documents", policy_id)
    return {"ok": True, "note": "Deleted. Click 'Rebuild Index' to remove it from retrieval."}


@router.get("/rebuild-status")
def rebuild_status_endpoint(emp: dict = Depends(require_admin)):
    """
    Lightweight status the frontend polls while a rebuild is in progress
    (the /rebuild-index POST itself blocks until done, which can take
    minutes on free-tier CPU — this lets the UI show live stage text
    instead of a bare spinner the whole time).
    """
    from deps import rebuild_status
    return dict(rebuild_status)


@router.post("/rebuild-index")
def rebuild_index(emp: dict = Depends(require_admin)):
    """
    Clears the in-memory model bundle and rebuilds FAISS/BM25 indexes from
    the current active rows in policy_documents. This is synchronous and
    can take a while (embeddings + reranker reload) — the frontend polls
    GET /rebuild-status concurrently to show live progress during this call.
    """
    models.ready = False
    try:
        sync_active_policies_to_disk()
        load_models_once()
    except Exception as e:
        raise HTTPException(500, f"Index rebuild failed: {e}")
    log_admin_action(emp["id"], "rebuild_index")
    return {"ok": True, "note": "Index rebuilt successfully."}