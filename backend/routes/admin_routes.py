from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text

from database import get_db
from deps import require_admin

router = APIRouter(prefix="/api/admin", tags=["admin"])


def log_admin_action(admin_id, action, resource_type=None, resource_id=None, notes=None):
    try:
        with get_db() as db:
            db.execute(text("""
                INSERT INTO admin_audit_log
                (admin_id, action, resource_type, resource_id, performed_at, notes)
                VALUES (:admin_id, :action, :rtype, :rid, NOW(), :notes)
            """), {"admin_id": admin_id, "action": action,
                    "rtype": resource_type, "rid": resource_id, "notes": notes})
    except Exception as e:
        print("Audit log failed:", e)


@router.get("/analytics")
def analytics(
    date_from: date = Query(default_factory=lambda: date.today() - timedelta(days=30)),
    date_to: date = Query(default_factory=date.today),
    emp: dict = Depends(require_admin),
):
    with get_db() as db:
        rows = db.execute(text("""
            SELECT al.id, al.asked_at, e.full_name, e.department, e.grade,
                   al.intent, al.topic, al.language, al.unanswered, al.question_text
            FROM analytics_log al
            LEFT JOIN employees e ON e.id = al.employee_id
            WHERE al.asked_at::date BETWEEN :df AND :dt
            ORDER BY al.asked_at DESC
        """), {"df": str(date_from), "dt": str(date_to)}).fetchall()
    data = [dict(r._mapping) for r in rows]
    for r in data:
        r["asked_at"] = r["asked_at"].isoformat() if r["asked_at"] else None

    total = len(data)
    unanswered = sum(1 for r in data if r["unanswered"])
    lang_counts, intent_counts = {}, {}
    for r in data:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
        intent_counts[r["intent"]] = intent_counts.get(r["intent"], 0) + 1

    return {
        "rows": data,
        "total": total,
        "unanswered": unanswered,
        "unanswered_pct": round(unanswered / total * 100, 1) if total else 0,
        "by_language": lang_counts,
        "by_intent": intent_counts,
    }


@router.get("/escalations")
def escalations(emp: dict = Depends(require_admin)):
    with get_db() as db:
        rows = db.execute(text("""
            SELECT al.id, al.asked_at, e.full_name, e.email, e.department,
                   al.language, al.question_text
            FROM analytics_log al
            LEFT JOIN employees e ON e.id = al.employee_id
            WHERE al.unanswered = TRUE
              AND (al.resolved IS FALSE OR al.resolved IS NULL)
            ORDER BY al.asked_at DESC
            LIMIT 50
        """)).fetchall()
    data = [dict(r._mapping) for r in rows]
    for r in data:
        r["asked_at"] = r["asked_at"].isoformat() if r["asked_at"] else None
    return {"rows": data}


@router.post("/escalations/{log_id}/resolve")
def resolve_escalation(log_id: int, emp: dict = Depends(require_admin)):
    with get_db() as db:
        db.execute(text("UPDATE analytics_log SET resolved = TRUE WHERE id = :id"), {"id": log_id})
    log_admin_action(emp["id"], "resolve_escalation", "analytics_log", log_id)
    return {"ok": True}


@router.get("/audit-log")
def audit_log(emp: dict = Depends(require_admin)):
    with get_db() as db:
        rows = db.execute(text("""
            SELECT admin_id, action, resource_type, resource_id, performed_at, notes
            FROM admin_audit_log ORDER BY performed_at DESC LIMIT 100
        """)).fetchall()
    data = [dict(r._mapping) for r in rows]
    for r in data:
        r["performed_at"] = r["performed_at"].isoformat() if r["performed_at"] else None
    return {"rows": data}
