"""
policy_sync.py — Repopulates the local `policies/` folder from the database
on every backend startup.

Why this exists: Render's free tier has no persistent disk, so any files
written to local disk (including uploaded PDFs) are wiped whenever the
container restarts or redeploys. Postgres rows, on the other hand, always
persist. So the database is now the source of truth for PDF *contents*
(stored as bytes in policy_documents.file_data), and local disk is treated
as a disposable cache that gets rebuilt from the database every time the
app boots.
"""

import os
from sqlalchemy import text
from database import get_db

POLICY_DIR = os.environ.get("POLICY_PDF_DIR", "policies")


def sync_active_policies_to_disk() -> int:
    """Writes every active policy PDF's bytes from the DB to local disk.
    Returns the number of files written. Safe to call repeatedly."""
    os.makedirs(POLICY_DIR, exist_ok=True)
    written = 0
    try:
        with get_db() as db:
            rows = db.execute(text("""
                SELECT file_path, file_data FROM policy_documents
                WHERE is_active = TRUE AND file_data IS NOT NULL
            """)).fetchall()
        for file_path, file_data in rows:
            try:
                os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(bytes(file_data))
                written += 1
            except Exception as e:
                print(f"[policy_sync] Failed writing {file_path}: {e}")
    except Exception as e:
        print(f"[policy_sync] Could not query policy_documents ({e}) — "
              f"this is expected if migrations/002_add_pdf_bytes.sql hasn't "
              f"been run yet.")
    print(f"[policy_sync] Wrote {written} policy PDF(s) to disk from the database.")
    return written


def fetch_single_file_from_db(file_path: str) -> bool:
    """On-demand fallback: if a specific file is missing locally (e.g. a
    request arrives in the narrow window before startup sync finishes),
    fetch just that one file from the DB. Returns True if it wrote a file."""
    try:
        with get_db() as db:
            row = db.execute(text("""
                SELECT file_data FROM policy_documents
                WHERE file_path = :fp AND file_data IS NOT NULL
                LIMIT 1
            """), {"fp": file_path}).fetchone()
        if not row or row[0] is None:
            return False
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(bytes(row[0]))
        return True
    except Exception as e:
        print(f"[policy_sync] fetch_single_file_from_db failed for {file_path}: {e}")
        return False