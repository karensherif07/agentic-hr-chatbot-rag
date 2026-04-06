"""
personal_data.py
All functions that fetch a specific employee's live data from PostgreSQL.
Every function returns a plain dict or list of dicts — ready to be
serialised into an LLM prompt.

Never returns raw SQLAlchemy Row objects. Always converts to Python types.
"""

from datetime import date, datetime
from sqlalchemy import text
from database import get_db


# ── Helpers ────────────────────────────────────────────────────
def _row(r) -> dict:
    """Convert a SQLAlchemy Row to a plain dict, serialising non-JSON types."""
    d = dict(r._mapping)
    for k, v in d.items():
        if isinstance(v, (date, datetime)):
            d[k] = v.isoformat()
    return d

def _rows(rs) -> list[dict]:
    return [_row(r) for r in rs]


# ── 1. Employee profile ────────────────────────────────────────
def get_employee_profile(employee_id: int) -> dict:
    """Returns the employee's own profile row."""
    with get_db() as db:
        r = db.execute(text("""
            SELECT e.id, e.full_name, e.full_name_ar, e.email, e.grade,
                   e.job_title, e.department, e.hire_date, e.employment_type,
                   e.work_model, e.probation_end_date,
                   m.full_name AS manager_name, m.email AS manager_email
            FROM employees e
            LEFT JOIN employees m ON m.id = e.manager_id
            WHERE e.id = :eid
        """), {"eid": employee_id}).fetchone()
    return _row(r) if r else {}


# ── 2. Leave balances ──────────────────────────────────────────
def get_leave_balance(employee_id: int, year: int = None) -> list[dict]:
    """
    Returns leave balance rows for the employee.
    Defaults to current year. Pass year=None for all years.
    Uses the leave_balances_view which adds the computed 'remaining_days'.
    """
    if year is None:
        year = date.today().year
    with get_db() as db:
        rows = db.execute(text("""
            SELECT leave_type, year, entitled_days, carried_over_days,
                   used_days, pending_days, encashed_days, remaining_days
            FROM leave_balances_view
            WHERE employee_id = :eid AND year = :yr
            ORDER BY leave_type
        """), {"eid": employee_id, "yr": year}).fetchall()
    return _rows(rows)


def get_annual_leave_summary(employee_id: int, year: int = None) -> dict | None:
    """Focused helper: just the annual leave row with remaining days."""
    if year is None:
        year = date.today().year
    with get_db() as db:
        r = db.execute(text("""
            SELECT entitled_days, carried_over_days, used_days,
                   pending_days, encashed_days, remaining_days
            FROM leave_balances_view
            WHERE employee_id = :eid AND year = :yr AND leave_type = 'annual'
        """), {"eid": employee_id, "yr": year}).fetchone()
    return _row(r) if r else None


# ── 3. Leave requests ──────────────────────────────────────────
def get_leave_requests(employee_id: int, status: str = None, limit: int = 10) -> list[dict]:
    """
    Returns recent leave requests.
    status filter: 'pending' | 'approved' | 'rejected' | 'cancelled' | None (all)
    """
    with get_db() as db:
        rows = db.execute(text("""
            SELECT lr.id, lr.leave_type, lr.start_date, lr.end_date,
                   lr.days_count, lr.status, lr.notes,
                   lr.requested_at, lr.approved_at,
                   a.full_name AS approved_by_name,
                   lr.rejection_reason
            FROM leave_requests lr
            LEFT JOIN employees a ON a.id = lr.approved_by
            WHERE lr.employee_id = :eid
              AND (:status IS NULL OR lr.status = :status)
            ORDER BY lr.requested_at DESC
            LIMIT :lim
        """), {"eid": employee_id, "status": status, "lim": limit}).fetchall()
    return _rows(rows)


def get_pending_leave(employee_id: int) -> list[dict]:
    return get_leave_requests(employee_id, status='pending')


# ── 4. Performance reviews ─────────────────────────────────────
def get_performance_history(employee_id: int, limit: int = 6) -> list[dict]:
    """Returns performance reviews newest-first."""
    with get_db() as db:
        rows = db.execute(text("""
            SELECT pr.review_period, pr.review_date, pr.rating,
                   pr.rating_label, pr.salary_increment_pct,
                   pr.bonus_multiplier, pr.strengths,
                   pr.areas_for_growth, pr.overall_comments,
                   r.full_name AS reviewer_name
            FROM performance_reviews pr
            LEFT JOIN employees r ON r.id = pr.reviewer_id
            WHERE pr.employee_id = :eid
            ORDER BY pr.review_date DESC
            LIMIT :lim
        """), {"eid": employee_id, "lim": limit}).fetchall()
    return _rows(rows)


def get_latest_review(employee_id: int) -> dict | None:
    reviews = get_performance_history(employee_id, limit=1)
    return reviews[0] if reviews else None


def get_performance_trend(employee_id: int) -> dict:
    """
    Returns a summary dict with trend info:
    - all ratings ordered by date
    - average rating
    - direction: improving | declining | stable
    """
    reviews = get_performance_history(employee_id, limit=6)
    if not reviews:
        return {"trend": "no data", "ratings": []}
    ratings = [r["rating"] for r in reviews][::-1]  # oldest first
    avg = round(sum(ratings) / len(ratings), 1)
    if len(ratings) >= 2:
        if ratings[-1] > ratings[0]:
            direction = "improving"
        elif ratings[-1] < ratings[0]:
            direction = "declining"
        else:
            direction = "stable"
    else:
        direction = "stable"
    return {
        "ratings_oldest_first": ratings,
        "average_rating": avg,
        "trend_direction": direction,
        "review_periods": [r["review_period"] for r in reviews][::-1],
        "labels_oldest_first": [r["rating_label"] for r in reviews][::-1],
    }


# ── 5. OKRs ────────────────────────────────────────────────────
def get_okrs(employee_id: int, period: str = None) -> list[dict]:
    """
    Returns OKRs. If period is None, returns the most recent period.
    """
    with get_db() as db:
        if period:
            rows = db.execute(text("""
                SELECT title, period, description, key_results,
                       overall_progress_pct, status, set_date, due_date
                FROM okrs
                WHERE employee_id = :eid AND period = :period
                ORDER BY set_date DESC
            """), {"eid": employee_id, "period": period}).fetchall()
        else:
            # Get the most recent period automatically
            rows = db.execute(text("""
                SELECT title, period, description, key_results,
                       overall_progress_pct, status, set_date, due_date
                FROM okrs
                WHERE employee_id = :eid
                  AND period = (
                      SELECT period FROM okrs
                      WHERE employee_id = :eid
                      ORDER BY set_date DESC LIMIT 1
                  )
                ORDER BY set_date DESC
            """), {"eid": employee_id}).fetchall()
    return _rows(rows)


# ── 6. Payroll ─────────────────────────────────────────────────
def get_latest_salary(employee_id: int) -> dict | None:
    """Returns the most recent payroll record."""
    with get_db() as db:
        r = db.execute(text("""
            SELECT month, base_salary, transport_allowance, housing_allowance,
                   mobile_allowance, remote_allowance, shift_allowance,
                   on_call_allowance, gross_salary, income_tax,
                   social_insurance, other_deductions, net_salary
            FROM payroll_records
            WHERE employee_id = :eid
            ORDER BY month DESC
            LIMIT 1
        """), {"eid": employee_id}).fetchone()
    return _row(r) if r else None


def get_payroll_history(employee_id: int, months: int = 6) -> list[dict]:
    with get_db() as db:
        rows = db.execute(text("""
            SELECT month, base_salary, gross_salary, net_salary
            FROM payroll_records
            WHERE employee_id = :eid
            ORDER BY month DESC
            LIMIT :m
        """), {"eid": employee_id, "m": months}).fetchall()
    return _rows(rows)


# ── 7. Training ────────────────────────────────────────────────
def get_training_record(employee_id: int, year: int = None) -> dict | None:
    if year is None:
        year = date.today().year
    with get_db() as db:
        r = db.execute(text("""
            SELECT year, budget_total_usd, budget_used_usd,
                   training_days_total, training_days_used,
                   courses, scholarship_applied,
                   scholarship_amount, scholarship_status,
                   GREATEST(0, budget_total_usd - budget_used_usd) AS budget_remaining_usd,
                   GREATEST(0, training_days_total - training_days_used) AS days_remaining
            FROM training_records
            WHERE employee_id = :eid AND year = :yr
        """), {"eid": employee_id, "yr": year}).fetchone()
    return _row(r) if r else None


# ── 8. Disciplinary ────────────────────────────────────────────
def get_active_disciplinary(employee_id: int) -> list[dict]:
    """Returns only active (non-expired) disciplinary records."""
    with get_db() as db:
        rows = db.execute(text("""
            SELECT action_type, issued_date, expiry_date, reason, outcome
            FROM disciplinary_records
            WHERE employee_id = :eid AND is_active = TRUE
            ORDER BY issued_date DESC
        """), {"eid": employee_id}).fetchall()
    return _rows(rows)


# ── 9. Full personal context bundle ───────────────────────────
def get_full_personal_context(employee_id: int) -> dict:
    """
    Fetches ALL personal data in one call.
    Used for the LLM prompt when intent is 'personal' or 'hybrid'.
    Returns a single dict that the prompt template can reference.
    """
    today = date.today()
    return {
        "profile":             get_employee_profile(employee_id),
        "leave_balances":      get_leave_balance(employee_id, today.year),
        "pending_leaves":      get_pending_leave(employee_id),
        "latest_review":       get_latest_review(employee_id),
        "performance_trend":   get_performance_trend(employee_id),
        "current_okrs":        get_okrs(employee_id),
        "latest_salary":       get_latest_salary(employee_id),
        "training":            get_training_record(employee_id, today.year),
        "active_disciplinary": get_active_disciplinary(employee_id),
        "fetched_at":          today.isoformat(),
    }


# ── 10. Intent-specific fetch (lightweight) ───────────────────
def fetch_for_intent(employee_id: int, intent_topic: str) -> dict:
    """
    Fetches only the data relevant to the classified topic.
    Avoids running all queries when only one is needed.

    intent_topic values:
        leave | performance | salary | training | okr | profile | disciplinary | all
    """
    today = date.today()
    fetchers = {
        "leave":        lambda: {
                            "leave_balances": get_leave_balance(employee_id, today.year),
                            "pending_leaves": get_pending_leave(employee_id),
                            "recent_requests": get_leave_requests(employee_id, limit=5),
                        },
        "performance":  lambda: {
                            "performance_history": get_performance_history(employee_id),
                            "performance_trend":   get_performance_trend(employee_id),
                            "latest_review":       get_latest_review(employee_id),
                        },
        "salary":       lambda: {
                            "latest_salary":  get_latest_salary(employee_id),
                            "salary_history": get_payroll_history(employee_id),
                        },
        "training":     lambda: {
                            "training": get_training_record(employee_id, today.year),
                        },
        "okr":          lambda: {
                            "current_okrs": get_okrs(employee_id),
                        },
        "profile":      lambda: {
                            "profile": get_employee_profile(employee_id),
                        },
        "disciplinary": lambda: {
                            "active_disciplinary": get_active_disciplinary(employee_id),
                        },
        "all":          lambda: get_full_personal_context(employee_id),
    }
    fn = fetchers.get(intent_topic, fetchers["all"])
    data = fn()
    data["profile"] = get_employee_profile(employee_id)  # always include profile
    data["fetched_at"] = today.isoformat()
    return data