"""
pages/admin_portal.py
Admin-only portal accessible via sidebar button.
Complete HR admin dashboard with analytics, escalations, audit log, and system config.
"""

import streamlit as st
import pandas as pd
import os
from datetime import date, timedelta
from sqlalchemy import text
from database import get_db
from auth import require_admin

st.set_page_config(page_title="HR Admin Portal", layout="wide")
require_admin()   # blocks non-admins with error message


def log_admin_action(action, resource_type=None, resource_id=None, notes=None):
    """Log admin actions for audit trail."""
    try:
        with get_db() as db:
            db.execute(text("""
                INSERT INTO admin_audit_log
                (admin_id, action, resource_type, resource_id, performed_at, notes)
                VALUES (:admin_id, :action, :rtype, :rid, NOW(), :notes)
            """), {
                "admin_id": st.session_state.employee_id,
                "action": action,
                "rtype": resource_type,
                "rid": resource_id,
                "notes": notes
            })
            db.commit()
    except Exception as e:
        print("Audit log failed:", e)


st.title("⚙️ HR Admin Portal")
st.caption(f"Logged in as: {st.session_state.employee_name} ({st.session_state.admin_role})")

# ─── Tab navigation ─────────────────────────────────────────
tabs = st.tabs([
    "📊 Analytics",
    "🔴 Escalations",
    "📋 Audit Log",
    "⚙️ Config",
    "💚 System Health"  
])

# ═══════════════════════════════════════════════════════════
# TAB 1: ANALYTICS
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Query Analytics")

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From", value=date.today() - timedelta(days=30), key="ana_from")
    with col2:
        date_to = st.date_input("To", value=date.today(), key="ana_to")

    @st.cache_data(ttl=60)
    def load_analytics(df, dt):
        with get_db() as db:
            rows = db.execute(text("""
                SELECT asked_at, full_name, department, grade, intent, topic,
                       language, unanswered, question_text
                FROM analytics_log al
                LEFT JOIN employees e ON e.id = al.employee_id
                WHERE al.asked_at::date BETWEEN :df AND :dt
                ORDER BY al.asked_at DESC
            """), {"df": str(df), "dt": str(dt)}).fetchall()

        return pd.DataFrame([dict(r._mapping) for r in rows])

    df_ana = load_analytics(date_from, date_to)

    if df_ana.empty:
        st.info("No data in this range.")
    else:
        total = len(df_ana)
        unanswered = int(df_ana["unanswered"].sum())
        unan_pct = round(unanswered / total * 100, 1) if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total queries", total)
        m2.metric("Unanswered", unanswered)
        m3.metric("Unanswered %", f"{unan_pct}%")

        st.divider()

        # Charts by language and intent
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("By Language")
            label_map = {
                "english": "English",
                "arabic_msa": "Arabic (MSA)",
                "arabic_egyptian": "Arabic (Egyptian)",
                "franco": "Franco Arabic",
            }
            lang_counts = df_ana["language"].value_counts().reset_index()
            lang_counts.columns = ["language", "count"]
            lang_counts["language"] = lang_counts["language"].map(lambda x: label_map.get(x, x))
            st.bar_chart(lang_counts.set_index("language"))

        with c2:
            st.subheader("By Intent")
            intent_counts = df_ana["intent"].value_counts()
            st.bar_chart(intent_counts)

        st.divider()
        st.subheader("🔴 Unanswered Questions")
        udf = df_ana[df_ana["unanswered"] == True][
            ["asked_at", "full_name", "department", "language", "question_text"]
        ].head(20)

        st.dataframe(udf, use_container_width=True, hide_index=True)

        csv = df_ana.to_csv(index=False).encode("utf-8")
        if st.download_button("⬇️ Download CSV", csv, "analytics.csv", "text/csv"):
            log_admin_action("export_analytics_csv")

# ═══════════════════════════════════════════════════════════
# TAB 2: ESCALATION QUEUE
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Escalation Queue")
    st.caption("Questions the bot couldn't answer — review and follow up with employees.")

    @st.cache_data(ttl=30)
    def load_escalations():
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

        return pd.DataFrame([dict(r._mapping) for r in rows])

    df_esc = load_escalations()

    if df_esc.empty:
        st.success("✅ No pending escalations")
    else:
        st.metric("Pending escalations", len(df_esc))

        for _, row in df_esc.iterrows():
            with st.expander(f"#{row['id']} — {row['question_text'][:80]}..."):
                st.write(f"👤 {row['full_name']} ({row['department']})")
                st.write(f"📧 {row['email']}")
                st.write(f"🌐 {row['language']}")
                st.write(f"🕒 {row['asked_at']}")
                st.info(row["question_text"])

                if st.button(f"Resolve #{row['id']}", key=f"res_{row['id']}"):
                    try:
                        with get_db() as db:
                            db.execute(text("""
                                UPDATE analytics_log
                                SET resolved = TRUE
                                WHERE id = :id
                            """), {"id": row["id"]})
                            db.commit()

                        log_admin_action(
                            action="resolve_escalation",
                            resource_type="analytics_log",
                            resource_id=row["id"]
                        )

                        st.success("✅ Resolved")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════
# TAB 3: AUDIT LOG
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Admin Audit Log")
    st.caption("Track all admin actions for compliance and accountability.")

    @st.cache_data(ttl=60)
    def load_audit():
        with get_db() as db:
            rows = db.execute(text("""
                SELECT admin_id, action, resource_type, resource_id, performed_at, notes
                FROM admin_audit_log
                ORDER BY performed_at DESC
                LIMIT 100
            """)).fetchall()

        return pd.DataFrame([dict(r._mapping) for r in rows])

    try:
        df_audit = load_audit()
        if df_audit.empty:
            st.info("No audit entries yet.")
        else:
            st.dataframe(df_audit, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Audit table may not be initialized: {e}")
        st.info("Run the migration to create admin_audit_log table")

# ═══════════════════════════════════════════════════════════
# TAB 4: SYSTEM CONFIG
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("System Configuration")
    st.caption("Manage HR contact, SMTP settings, and system parameters.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Configuration:**")
        st.code(f"""
HR_EMAIL: {os.environ.get('HR_EMAIL', '(not set)')}
SMTP_HOST: {os.environ.get('SMTP_HOST', '(not set)')}
SMTP_USER: {os.environ.get('SMTP_USER', '(not set)')}
LLM_MODEL: llama-3.3-70b-versatile
RETRIEVAL_TOP_N: 5
CONFIDENCE_THRESHOLD: -9.0
ADMIN_ROLE_COLUMN: employees.admin_role
""")

    with col2:
        st.write("**How to Update:**")
        st.info("""
1. Edit your `.env` file in the root directory:
   ```
   HR_EMAIL=hr@horizontech.com
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=noreply@horizontech.com
   SMTP_PASS=your-app-password
   SMTP_FROM=noreply@horizontech.com
   ```

2. Restart the Streamlit app for changes to take effect.

3. To grant admin access to an employee:
   ```sql
   UPDATE employees
   SET admin_role = 'hr_admin'
   WHERE email = 'hr.person@horizontech.com';
   ```

4. Or use super_admin for unrestricted access:
   ```sql
   UPDATE employees
   SET admin_role = 'super_admin'
   WHERE email = 'admin@horizontech.com';
   ```
""")

    st.divider()
    st.write("**Database Status:**")
    try:
        with get_db() as db:
            row = db.execute(text("SELECT COUNT(*) as cnt FROM analytics_log")).fetchone()
            log_count = row[0] if row else 0
            row2 = db.execute(text("SELECT COUNT(*) as cnt FROM employees WHERE admin_role IS NOT NULL")).fetchone()
            admin_count = row2[0] if row2 else 0
        st.success(f"✅ Database connected")
        st.write(f"   • {log_count} analytics log entries")
        st.write(f"   • {admin_count} admin users configured")
    except Exception as e:
        st.error(f"❌ Database error: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 5: SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("💚 System Health Dashboard")
    st.caption("Monitor system performance, failures, and response behavior.")

    @st.cache_data(ttl=60)
    def load_health():
        with get_db() as db:
            rows = db.execute(text("""
                SELECT asked_at, unanswered
                FROM analytics_log
                WHERE asked_at >= NOW() - INTERVAL '30 days'
                ORDER BY asked_at
            """)).fetchall()

        df = pd.DataFrame([dict(r._mapping) for r in rows])

        if not df.empty:
            # ✅ FIX timezone issue (your error)
            df["asked_at"] = pd.to_datetime(df["asked_at"], utc=True).dt.tz_convert(None)

        return df

    df = load_health()

    if df.empty:
        st.info("No data available.")
    else:
        total = len(df)
        failures = int(df["unanswered"].sum())
        failure_rate = round(failures / total * 100, 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Queries (30d)", total)
        c2.metric("Failures", failures)
        c3.metric("Failure Rate", f"{failure_rate}%")

        st.divider()

        # ─── Query Volume Trend ───
        st.subheader("📈 Queries Over Time")
        trend = df.groupby(df["asked_at"].dt.date).size()
        st.line_chart(trend)

        # ─── Failure Trend ───
        st.subheader("⚠️ Failure Rate Over Time")
        fail_trend = df.groupby(df["asked_at"].dt.date)["unanswered"].mean()
        st.line_chart(fail_trend)