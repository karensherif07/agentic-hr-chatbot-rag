
"""
escalation.py (backend) — email-sending logic only.
The "Notify HR / Dismiss" buttons now live in React
(frontend/src/components/EscalationBanner.tsx) and call
POST /api/chat/escalate, which uses these functions.
"""

import os
from sqlalchemy import text
from database import get_db


def get_hr_email(exclude_employee_id: int | None = None) -> str:
    """
    Resolve who unanswered questions should be escalated to.

    exclude_employee_id: if the person asking is themself the top-ranked
    admin, we skip them and fall through to the next admin (or HR_EMAIL)
    instead of "notifying HR" by emailing the requester themselves. This
    matters now that admins use the same chat interface as everyone else.
    """
    hr_email = os.environ.get("HR_EMAIL", "").strip()
    if hr_email:
        return hr_email
    try:
        with get_db() as db:
            row = db.execute(text("""
                SELECT email FROM employees
                WHERE admin_role IN ('hr_admin', 'super_admin') AND is_active = TRUE
                  AND (:exclude_id IS NULL OR id != :exclude_id)
                ORDER BY admin_role DESC, id ASC LIMIT 1
            """), {"exclude_id": exclude_employee_id}).fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    return ""


def send_escalation_email(employee_name: str, hr_email: str, question: str) -> bool:
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
            f"Hi HR Team,\n\nThe HR chatbot could not answer the following question "
            f"from {employee_name}:\n\n  \"{question}\"\n\n"
            f"Please follow up with them directly.\n\n— HR Assistant (automated)"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"HR Chatbot: Unanswered query from {employee_name}"
        msg["From"] = from_addr
        msg["To"] = hr_email
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[escalation] {e}")
        return False


def send_contact_hr_email(employee_name, employee_email, hr_email, subject, body) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, hr_email]):
        return False
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"[Employee Message] {subject}"
        msg["From"] = from_addr
        msg["To"] = hr_email
        msg["Reply-To"] = employee_email
        full_body = f"Message from: {employee_name} ({employee_email})\n{'-'*40}\n\n{body}"
        msg.attach(MIMEText(full_body, "plain"))
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[contact_hr] {e}")
        return False