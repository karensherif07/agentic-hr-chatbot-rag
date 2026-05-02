"""
pages/admin_portal.py
HR Admin Dashboard — Analytics, Escalations, Audit Log, System Config, Health
"""

import streamlit as st
import pandas as pd
import os
from datetime import date, timedelta
from sqlalchemy import text
from database import get_db
from auth import require_admin

st.set_page_config(page_title="HR Admin Portal", layout="wide")
require_admin()


def log_admin_action(action, resource_type=None, resource_id=None, notes=None):
    try:
        with get_db() as db:
            db.execute(text("""
                INSERT INTO admin_audit_log
                (admin_id, action, resource_type, resource_id, performed_at, notes)
                VALUES (:admin_id, :action, :rtype, :rid, NOW(), :notes)
            """), {
                "admin_id": st.session_state.employee_id,
                "action": action, "rtype": resource_type,
                "rid": resource_id, "notes": notes,
            })
            db.commit()
    except Exception as e:
        print("Audit log failed:", e)


st.title("⚙️ HR Admin Portal")
st.caption(f"Logged in as: {st.session_state.employee_name} ({st.session_state.admin_role})")

tabs = st.tabs([
    "📊 Analytics",
    "🔴 Escalations",
    "📋 Audit Log",
    "⚙️ Config",
    "💚 System Health",
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
                SELECT al.id, al.asked_at, e.full_name, e.department, e.grade,
                       al.intent, al.topic, al.language, al.unanswered, al.question_text
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
        total      = len(df_ana)
        unanswered = int(df_ana["unanswered"].sum())
        unan_pct   = round(unanswered / total * 100, 1) if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total queries", total)
        m2.metric("Unanswered", unanswered)
        m3.metric("Unanswered %", f"{unan_pct}%")

        st.divider()

        # ── Charts ───────────────────────────────────────────
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
            st.bar_chart(df_ana["intent"].value_counts())

        st.divider()

        # ── Unanswered questions ──────────────────────────────
        st.subheader("🔴 Unanswered Questions")
        udf = df_ana[df_ana["unanswered"] == True][
            ["asked_at", "full_name", "department", "language", "question_text"]
        ].head(20)
        st.dataframe(udf, use_container_width=True, hide_index=True)

        st.divider()

        # ── ALL queries log ───────────────────────────────────
        # This is the full log with search, filter, and download
        st.subheader("📋 All Queries Log")

        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_lang = st.selectbox(
                "Filter by language",
                ["All", "english", "arabic_msa", "arabic_egyptian", "franco"],
                key="filter_lang",
            )
        with fc2:
            filter_intent = st.selectbox(
                "Filter by intent",
                ["All", "policy", "personal", "hybrid"],
                key="filter_intent",
            )
        with fc3:
            filter_unanswered = st.selectbox(
                "Filter by answered",
                ["All", "Answered", "Unanswered"],
                key="filter_unans",
            )

        search_text = st.text_input("🔍 Search question text", key="search_q", placeholder="Type to search...")

        # Apply filters
        df_filtered = df_ana.copy()
        if filter_lang != "All":
            df_filtered = df_filtered[df_filtered["language"] == filter_lang]
        if filter_intent != "All":
            df_filtered = df_filtered[df_filtered["intent"] == filter_intent]
        if filter_unanswered == "Answered":
            df_filtered = df_filtered[df_filtered["unanswered"] == False]
        elif filter_unanswered == "Unanswered":
            df_filtered = df_filtered[df_filtered["unanswered"] == True]
        if search_text:
            df_filtered = df_filtered[
                df_filtered["question_text"].str.contains(search_text, case=False, na=False)
            ]

        st.caption(f"Showing {len(df_filtered)} of {total} queries")

        # Display table with colour-coded unanswered column
        display_cols = ["asked_at", "full_name", "department", "grade",
                        "intent", "topic", "language", "unanswered", "question_text"]
        available_cols = [c for c in display_cols if c in df_filtered.columns]

        st.dataframe(
            df_filtered[available_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "asked_at":      st.column_config.DatetimeColumn("Time", format="DD/MM/YY HH:mm"),
                "full_name":     st.column_config.TextColumn("Employee", width="medium"),
                "department":    st.column_config.TextColumn("Dept", width="small"),
                "grade":         st.column_config.TextColumn("Grade", width="small"),
                "intent":        st.column_config.TextColumn("Intent", width="small"),
                "topic":         st.column_config.TextColumn("Topic", width="small"),
                "language":      st.column_config.TextColumn("Lang", width="small"),
                "unanswered":    st.column_config.CheckboxColumn("Unanswered?", width="small"),
                "question_text": st.column_config.TextColumn("Question", width="large"),
            },
        )

        # Download buttons
        dl1, dl2 = st.columns(2)
        with dl1:
            csv_all = df_ana.to_csv(index=False).encode("utf-8")
            if st.download_button("⬇️ Download All (CSV)", csv_all, "analytics_all.csv", "text/csv"):
                log_admin_action("export_analytics_csv", notes="all queries")
        with dl2:
            csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
            if st.download_button("⬇️ Download Filtered (CSV)", csv_filtered, "analytics_filtered.csv", "text/csv"):
                log_admin_action("export_analytics_csv", notes="filtered queries")


# ═══════════════════════════════════════════════════════════
# TAB 2: ESCALATION QUEUE
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Escalation Queue")
    st.caption("Questions the bot could not answer — review and follow up with employees.")

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
                            db.execute(text(
                                "UPDATE analytics_log SET resolved = TRUE WHERE id = :id"
                            ), {"id": row["id"]})
                            db.commit()
                        log_admin_action("resolve_escalation", "analytics_log", row["id"])
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
                FROM admin_audit_log ORDER BY performed_at DESC LIMIT 100
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
HR_EMAIL:        {os.environ.get('HR_EMAIL', '(not set)')}
SMTP_HOST:       {os.environ.get('SMTP_HOST', '(not set)')}
SMTP_USER:       {os.environ.get('SMTP_USER', '(not set)')}
ROUTING_LLM:     llama-3.3-70b-versatile
EN_LLM:          llama-3.3-70b-versatile
AR_LLM:          qwen/qwen3-32b  (thinking disabled)
CRITIQUE_LLM:    llama-3.1-8b-instant
RETRIEVAL_TOP_N: 5
""")
    with col2:
        st.write("**How to Update:**")
        st.info("""
1. Edit `.env` in the root directory:
   ```
   HR_EMAIL=hr@horizontech.com
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=noreply@horizontech.com
   SMTP_PASS=your-app-password
   ```

2. Restart the Streamlit app for changes to take effect.

3. To grant admin access:
   ```sql
   UPDATE employees SET admin_role = 'hr_admin'
   WHERE email = 'hr.person@horizontech.com';
   ```
""")

    st.divider()
    st.write("**Database Status:**")
    try:
        with get_db() as db:
            log_count   = db.execute(text("SELECT COUNT(*) FROM analytics_log")).fetchone()[0]
            admin_count = db.execute(text(
                "SELECT COUNT(*) FROM employees WHERE admin_role IS NOT NULL"
            )).fetchone()[0]
        st.success("✅ Database connected")
        st.write(f"   • {log_count} analytics log entries")
        st.write(f"   • {admin_count} admin users configured")
    except Exception as e:
        st.error(f"❌ Database error: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 5: SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("💚 System Health Dashboard")
    st.caption("Monitor system performance, failures, and response behaviour.")

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
            df["asked_at"] = pd.to_datetime(df["asked_at"], utc=True).dt.tz_convert(None)
        return df

    df = load_health()

    if df.empty:
        st.info("No data available.")
    else:
        total        = len(df)
        failures     = int(df["unanswered"].sum())
        failure_rate = round(failures / total * 100, 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Queries (30d)", total)
        c2.metric("Failures", failures)
        c3.metric("Failure Rate", f"{failure_rate}%")

        st.divider()
        st.subheader("📈 Queries Over Time")
        st.line_chart(df.groupby(df["asked_at"].dt.date).size())

        st.subheader("⚠️ Failure Rate Over Time")
        st.line_chart(df.groupby(df["asked_at"].dt.date)["unanswered"].mean())