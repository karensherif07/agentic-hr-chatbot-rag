import base64, bcrypt, hashlib, hmac, json, os, time
import streamlit as st
from sqlalchemy import text
from database import get_db

SESSION_TTL_SEC = 7 * 24 * 60 * 60   # 7 days
_QPARAM_KEY     = "_token"

# ── Admin role constants ────────────────────────────────────────
# admin_role values stored in DB: None | 'hr_admin' | 'super_admin'
# Both admin roles can see the analytics page.
# Regular employees (None) only see the chatbot.
ADMIN_ROLES = {"hr_admin", "super_admin"}


def init_cookie_manager():
    """No-op — kept for API compatibility."""
    pass


# ── Token helpers ───────────────────────────────────────────────
def _secret() -> bytes:
    s = (os.environ.get("HR_AUTH_SECRET") or "").strip()
    return s.encode() if s else b"hr-chatbot-dev-secret-change-me"


def _make_token(employee_id: int) -> str:
    exp     = int(time.time()) + SESSION_TTL_SEC
    payload = json.dumps({"id": employee_id, "exp": exp}, separators=(",", ":")).encode()
    p64 = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    sig = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    return f"{p64}.{sig}"


def _validate_token(token: str) -> "int | None":
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


# ── DB helpers ──────────────────────────────────────────────────
def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _fetch_by_email(email: str) -> "dict | None":
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date, admin_role
            FROM employees WHERE email = :e AND is_active = TRUE
        """), {"e": email.strip().lower()}).fetchone()
    return dict(row._mapping) if row else None


def _fetch_by_id(emp_id: int) -> "dict | None":
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date, admin_role
            FROM employees WHERE id = :id AND is_active = TRUE
        """), {"id": emp_id}).fetchone()
    return dict(row._mapping) if row else None


def _hydrate(emp: dict) -> None:
    """Write employee data into session state."""
    st.session_state.logged_in        = True
    st.session_state.employee_id      = emp["id"]
    st.session_state.employee_email   = emp["email"]
    st.session_state.employee_name    = emp["full_name"]
    st.session_state.employee_name_ar = emp.get("full_name_ar", "")
    st.session_state.employee_grade   = emp["grade"]
    st.session_state.employee_title   = emp["job_title"]
    st.session_state.employee_dept    = emp["department"]
    st.session_state.manager_id       = emp["manager_id"]
    st.session_state.hire_date        = emp["hire_date"]
    st.session_state.employment_type  = emp["employment_type"]
    st.session_state.work_model       = emp["work_model"]
    st.session_state.in_probation     = emp["probation_end_date"] is not None
    # Admin role — None for regular employees, 'hr_admin' or 'super_admin' for admins
    st.session_state.admin_role       = emp.get("admin_role")
    st.session_state.is_admin         = emp.get("admin_role") in ADMIN_ROLES


def _try_restore_from_qparam() -> bool:
    """Read ?_token from URL, validate it, hydrate session."""
    try:
        token = st.query_params.get(_QPARAM_KEY)
    except Exception:
        return False
    if not token:
        return False

    emp_id = _validate_token(token)
    if emp_id is None:
        try:
            st.query_params.clear()
        except Exception:
            pass
        return False

    emp = _fetch_by_id(emp_id)
    if emp is None:
        try:
            st.query_params.clear()
        except Exception:
            pass
        return False

    _hydrate(emp)
    # Do NOT re-set query_params here — writing it again triggers an extra
    # Streamlit rerun (v1.32+) which wipes session_state before the page renders.
    return True


# ── Public API ──────────────────────────────────────────────────
def require_login() -> None:
    if st.session_state.get("logged_in"):
        return
    if _try_restore_from_qparam():
        return

    st.markdown("<style>.lw{max-width:420px;margin:80px auto 0}</style>",
                unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lw">', unsafe_allow_html=True)
        st.title("HR Policy Assistant")
        st.caption("Sign in with your Horizon Tech email to continue.")
        with st.form("login_form"):
            email    = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign in", use_container_width=True)
        if submit:
            emp = _fetch_by_email(email)
            if emp and _verify_password(password, emp["password_hash"]):
                _hydrate(emp)
                token = _make_token(emp["id"])
                st.query_params[_QPARAM_KEY] = token   # written once on login
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def is_admin() -> bool:
    """Convenience helper — True if the logged-in user has any admin role."""
    return bool(st.session_state.get("is_admin", False))


def require_admin() -> None:
    """Call at the top of admin-only pages. Stops non-admins."""
    require_login()
    if not is_admin():
        st.error("⛔ Access denied. This page is for HR administrators only.")
        st.stop()


def logout() -> None:
    try:
        st.query_params.clear()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()