
import bcrypt
import streamlit as st
from sqlalchemy import text
from database import get_db


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


def require_login():
    """
    Call this at the very top of app.py before any other content.
    Stops execution with st.stop() if not logged in.
    """
    if st.session_state.get("logged_in"):
        return  # already authenticated — continue

    st.markdown("""
    <style>
    .login-wrap { max-width: 420px; margin: 80px auto 0; }
    </style>
    """, unsafe_allow_html=True)

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

            # ── Store in session ──
            st.session_state.logged_in       = True
            st.session_state.employee_id     = emp["id"]
            st.session_state.employee_email  = emp["email"]
            st.session_state.employee_name   = emp["full_name"]
            st.session_state.employee_name_ar= emp["full_name_ar"]
            st.session_state.employee_grade  = emp["grade"]
            st.session_state.employee_title  = emp["job_title"]
            st.session_state.employee_dept   = emp["department"]
            st.session_state.manager_id      = emp["manager_id"]
            st.session_state.hire_date       = emp["hire_date"]
            st.session_state.employment_type = emp["employment_type"]
            st.session_state.work_model      = emp["work_model"]
            st.session_state.in_probation    = emp["probation_end_date"] is not None
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()  # halt execution until login succeeds


def logout():
    """Call from a sidebar button."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()