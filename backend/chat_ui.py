"""
chat_ui.py (backend) — Only the DB-writing logic survives here.
All rendering (answer bubbles, expanders, source evidence images) moved to
the React frontend — see frontend/src/components/ChatBubble.tsx,
SourceEvidence.tsx, PersonalDataPanel.tsx.
"""

from database import get_db
from sqlalchemy import text


def log_query(employee_id, intent, topic, lang, dialect, is_no_info, question):
    if lang == "arabic":
        log_lang = "arabic_egyptian" if dialect == "egyptian" else "arabic_msa"
    elif lang == "franco":
        log_lang = "franco"
    else:
        log_lang = "english"
    try:
        with get_db() as db:
            db.execute(text("""
                INSERT INTO analytics_log
                    (employee_id, intent, topic, language, unanswered, question_text, asked_at)
                VALUES (:eid, :intent, :topic, :lang, :unans, :q, NOW())
            """), {
                "eid": employee_id, "intent": intent, "topic": topic or "",
                "lang": log_lang, "unans": is_no_info, "q": question[:300],
            })
    except Exception as e:
        print(f"[analytics] {e}")
