"""
 — Persistent chat history per employee via PostgreSQL.

Schema (run once in pgAdmin, add to schema.sql):

    CREATE TABLE IF NOT EXISTS chat_sessions (
        employee_id     INT PRIMARY KEY REFERENCES employees(id) ON DELETE CASCADE,
        history_json    TEXT NOT NULL DEFAULT '[]',
        summary_text    TEXT NOT NULL DEFAULT '',
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

This keeps ONE session per employee — the latest conversation.
Saving is intentionally lightweight: just a single UPSERT with the
last 20 message pairs (40 rows max) so the stored JSON stays small.
"""

import json
from datetime import datetime, timezone
from sqlalchemy import text
from database import get_db


_MAX_STORED_MESSAGES = 40   # keep last 20 Q+A pairs in the DB


def save_session(employee_id: int, chat_history: list, summary: str = "") -> None:
    """Upsert the employee's chat history and summary to the DB."""
    # Keep only the most recent messages to bound storage size
    trimmed = chat_history[-_MAX_STORED_MESSAGES:] if chat_history else []
    payload = json.dumps(trimmed, ensure_ascii=False)
    with get_db() as db:
        db.execute(text("""
            INSERT INTO chat_sessions (employee_id, history_json, summary_text, updated_at)
            VALUES (:eid, :hist, :summ, :now)
            ON CONFLICT (employee_id)
            DO UPDATE SET
                history_json = EXCLUDED.history_json,
                summary_text = EXCLUDED.summary_text,
                updated_at   = EXCLUDED.updated_at
        """), {
            "eid":  employee_id,
            "hist": payload,
            "summ": summary or "",
            "now":  datetime.now(timezone.utc),
        })


def load_session(employee_id: int) -> tuple[list, str]:
    """
    Returns (chat_history, summary_text) for the employee.
    Returns ([], '') if no saved session exists.
    """
    with get_db() as db:
        row = db.execute(text("""
            SELECT history_json, summary_text
            FROM chat_sessions
            WHERE employee_id = :eid
        """), {"eid": employee_id}).fetchone()
    if not row:
        return [], ""
    try:
        history = json.loads(row.history_json or "[]")
    except (json.JSONDecodeError, TypeError):
        history = []
    return history, (row.summary_text or "")


def clear_session(employee_id: int) -> None:
    """Delete the stored session (e.g. when employee clicks 'Clear chat')."""
    with get_db() as db:
        db.execute(text(
            "DELETE FROM chat_sessions WHERE employee_id = :eid"
        ), {"eid": employee_id})
