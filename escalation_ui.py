"""
escalation_ui.py
Handles escalation when the bot cannot answer:
- get_hr_email()        — resolves HR email from env or DB
- send_escalation_email() — sends SMTP notification to HR
- render_escalation_ui()  — shows the Notify HR / Dismiss buttons
"""

import os
import streamlit as st


def get_hr_email() -> str:
    """Resolve HR email from HR_EMAIL env var, or from DB admin users."""
    hr_email = os.environ.get("HR_EMAIL", "").strip()
    if hr_email:
        return hr_email
    try:
        from database import get_db
        from sqlalchemy import text
        with get_db() as db:
            row = db.execute(text("""
                SELECT email FROM employees
                WHERE admin_role IN ('hr_admin', 'super_admin') AND is_active = TRUE
                ORDER BY admin_role DESC, id ASC LIMIT 1
            """)).fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    return ""


def send_escalation_email(employee_name: str, hr_email: str, question: str) -> bool:
    """Send SMTP email to HR team about an unanswered question."""
    import smtplib
    from email.mime.text import MIMEText

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, hr_email]):
        return False

    try:
        body = (
            f"Hi HR Team,\n\n"
            f"The HR chatbot could not answer the following question "
            f"from {employee_name}:\n\n"
            f'  "{question}"\n\n'
            f"Please follow up with them directly.\n\n"
            f"— HR Assistant (automated)"
        )
        msg            = MIMEText(body)
        msg["Subject"] = f"HR Chatbot: Unanswered query from {employee_name}"
        msg["From"]    = from_addr
        msg["To"]      = hr_email
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[escalation] {e}")
        return False


def render_escalation_ui():
    """
    Render the Notify HR / Dismiss UI block.
    Only shown when st.session_state._show_esc is True.
    """
    if not st.session_state.get("_show_esc"):
        return

    st.info(
        "I could not find an answer in the policy documents. "
        "Would you like me to notify the HR team?"
    )
    col_a, col_b = st.columns([1, 5])

    with col_a:
        if st.button("📧 Notify HR", key="esc_btn"):
            sent = send_escalation_email(
                st.session_state.get("_esc_name", ""),
                st.session_state.get("_esc_hr_email", ""),
                st.session_state.get("_esc_question", ""),
            )
            st.session_state["_show_esc"] = False
            if sent:
                st.success("✅ The HR team has been notified.")
            else:
                st.warning(
                    "Email not sent. Check SMTP settings in .env or set HR_EMAIL."
                )

    with col_b:
        if st.button("Dismiss", key="esc_dismiss"):
            st.session_state["_show_esc"] = False
            st.rerun()