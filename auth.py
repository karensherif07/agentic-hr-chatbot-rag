
import base64
import bcrypt
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timedelta, timezone

import streamlit as st
from sqlalchemy import text
from database import get_db

try:
    from extra_streamlit_components import CookieManager
except ImportError:
    CookieManager = None

COOKIE_NAME = "hr_policy_session_v1"
COOKIE_MGR_KEY = "hr_policy_cookie_manager_v1"
_SESSION_COOKIE_MGR = "_hr_policy_cookie_manager_singleton"
SESSION_TTL_SEC = 7 * 24 * 60 * 60


def _cookie_manager():
    """
    One CookieManager per browser session (session_state). Do not use @st.cache_* —
    CookieManager embeds Streamlit components and triggers CachedWidgetWarning.
    """
    if CookieManager is None:
        return None
    if _SESSION_COOKIE_MGR not in st.session_state:
        st.session_state[_SESSION_COOKIE_MGR] = CookieManager(key=COOKIE_MGR_KEY)
    return st.session_state[_SESSION_COOKIE_MGR]


def _read_session_token() -> str | None:
    """
    Read auth cookie using CookieManager.
    """
    cm = _cookie_manager()
    if cm is None:
        return None
    t = cm.get(COOKIE_NAME)
    return str(t).strip() if t else None


def _auth_secret() -> bytes:
    s = (os.environ.get("HR_AUTH_SECRET") or "").strip()
    if not s:
        return b"hr-chatbot-dev-secret-set-hr-auth-secret-in-env"
    return s.encode("utf-8")


def _create_session_token(employee_id: int) -> str:
    exp = int(time.time()) + SESSION_TTL_SEC
    payload = json.dumps({"id": employee_id, "exp": exp}, separators=(",", ":")).encode("utf-8")
    p_b64 = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    sig = hmac.new(_auth_secret(), p_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{p_b64}.{sig}"


def _parse_session_token(token: str) -> int | None:
    if not token or "." not in token:
        return None
    p_b64, sig = token.rsplit(".", 1)
    want = hmac.new(_auth_secret(), p_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, sig):
        return None
    pad = "=" * (-len(p_b64) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode((p_b64 + pad).encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    return int(data["id"])


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _fetch_employee(email: str) -> dict | None:
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date
            FROM employees
            WHERE email = :email AND is_active = TRUE
        """), {"email": email.strip().lower()}).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


def _fetch_employee_by_id(emp_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date
            FROM employees
            WHERE id = :id AND is_active = TRUE
        """), {"id": emp_id}).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


def _hydrate_session(emp: dict) -> None:
    st.session_state.logged_in        = True
    st.session_state.employee_id      = emp["id"]
    st.session_state.employee_email   = emp["email"]
    st.session_state.employee_name    = emp["full_name"]
    st.session_state.employee_name_ar = emp["full_name_ar"]
    st.session_state.employee_grade   = emp["grade"]
    st.session_state.employee_title  = emp["job_title"]
    st.session_state.employee_dept   = emp["department"]
    st.session_state.manager_id      = emp["manager_id"]
    st.session_state.hire_date       = emp["hire_date"]
    st.session_state.employment_type = emp["employment_type"]
    st.session_state.work_model      = emp["work_model"]
    st.session_state.in_probation    = emp["probation_end_date"] is not None


def _set_auth_cookie(employee_id: int) -> None:
    cm = _cookie_manager()
    if cm is None:
        return
    token = _create_session_token(employee_id)
    exp = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SEC)
    try:
        cm.set(
            COOKIE_NAME,
            token,
            expires_at=exp,
            max_age=float(SESSION_TTL_SEC),
            path="/",
            same_site="lax",
        )
    except TypeError:
        cm.set(COOKIE_NAME, token, expires_at=exp, same_site="lax")


def _clear_auth_cookie() -> None:
    cm = _cookie_manager()
    if cm is None:
        return
    try:
        cm.delete(COOKIE_NAME)
    except Exception:
        try:
            cm.set(COOKIE_NAME, "", expires_at=datetime(1970, 1, 1, tzinfo=timezone.utc))
        except Exception:
            pass


def require_login():
    """
    Call this at the very top of app.py before any other content.
    Stops execution with st.stop() if not logged in.
    After a successful sign-in, a signed cookie keeps the session across refreshes
    until sign-out (requires: pip install extra-streamlit-components).
    """
    if st.session_state.get("logged_in"):
        return

    token = _read_session_token()
    if token:
        emp_id = _parse_session_token(token)
        emp = _fetch_employee_by_id(emp_id) if emp_id else None
        if emp:
            _hydrate_session(emp)
            return
        _clear_auth_cookie()

    st.markdown("""
    <style>
    .login-wrap { max-width: 420px; margin: 80px auto 0; }
    </style>
    """, unsafe_allow_html=True)

    if CookieManager is None:
        st.warning(
            "Install **extra-streamlit-components** for sign-in to persist across page refresh: "
            "`pip install extra-streamlit-components`"
        )

    with st.container():
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.title("HR Policy Assistant")
        st.caption("Sign in with your Horizon Tech email to continue.")

        with st.form("login_form"):
            email    = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign in", use_container_width=True)

        if submit:
            if not email or not password:
                st.error("Please enter your email and password.")
                st.stop()

            emp = _fetch_employee(email)
            if emp is None:
                st.error("Email not found or account is inactive.")
                st.stop()

            if not _verify_password(password, emp["password_hash"]):
                st.error("Incorrect password.")
                st.stop()

            _hydrate_session(emp)
            _set_auth_cookie(emp["id"])
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


def logout():
    """Call from a sidebar button."""
    _clear_auth_cookie()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
