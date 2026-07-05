"""
auth.py (backend) — Same token scheme as your original Streamlit auth.py,
just with st.session_state / st.query_params removed. FastAPI routes call
these functions directly and set/read an httpOnly cookie instead.
"""

import base64, bcrypt, hashlib, hmac, json, os, time
from sqlalchemy import text
from database import get_db

SESSION_TTL_SEC = 7 * 24 * 60 * 60   # 7 days
ADMIN_ROLES = {"hr_admin", "super_admin"}


def _secret() -> bytes:
    s = (os.environ.get("HR_AUTH_SECRET") or "").strip()
    return s.encode() if s else b"hr-chatbot-dev-secret-change-me"


def make_token(employee_id: int) -> str:
    exp = int(time.time()) + SESSION_TTL_SEC
    payload = json.dumps({"id": employee_id, "exp": exp}, separators=(",", ":")).encode()
    p64 = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    sig = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    return f"{p64}.{sig}"


def validate_token(token: str) -> "int | None":
    token = (token or "").strip()
    if not token or "." not in token:
        return None
    p64, sig = token.rsplit(".", 1)
    want = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, sig):
        return None
    pad = "=" * (-len(p64) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode((p64 + pad).encode()).decode())
    except Exception:
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    return int(data["id"])


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


_SELECT_COLS = """
    id, full_name, full_name_ar, email, password_hash,
    grade, job_title, department, manager_id,
    hire_date, employment_type, work_model,
    is_active, probation_end_date, admin_role
"""


def fetch_by_email(email: str) -> "dict | None":
    with get_db() as db:
        row = db.execute(text(f"""
            SELECT {_SELECT_COLS} FROM employees
            WHERE email = :e AND is_active = TRUE
        """), {"e": email.strip().lower()}).fetchone()
    return _serialize(row) if row else None


def fetch_by_id(emp_id: int) -> "dict | None":
    with get_db() as db:
        row = db.execute(text(f"""
            SELECT {_SELECT_COLS} FROM employees
            WHERE id = :id AND is_active = TRUE
        """), {"id": emp_id}).fetchone()
    return _serialize(row) if row else None


def _serialize(row) -> dict:
    """Convert Row -> plain JSON-safe dict (dates as ISO strings), drop password hash
    for anything that leaves the auth layer via the API response."""
    d = dict(row._mapping)
    for k, v in list(d.items()):
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
    d["is_admin"] = d.get("admin_role") in ADMIN_ROLES
    d["in_probation"] = d.get("probation_end_date") is not None
    return d


def public_employee(emp: dict) -> dict:
    """Strip sensitive fields before sending to the frontend."""
    safe = dict(emp)
    safe.pop("password_hash", None)
    return safe
