"""
pages/admin_analytics.py
HR Admin analytics dashboard — visible only to admin_role users.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import text
from database import get_db
from auth import require_admin   # uses the new role-based gate

st.set_page_config(page_title="HR Analytics", layout="wide")
require_admin()   # blocks anyone without hr_admin or super_admin role

st.title("📊 HR Chatbot Analytics")
st.caption(f"Logged in as: {st.session_state.employee_name} ({st.session_state.admin_role})")

# ── Date range filter ──────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    from datetime import date, timedelta
    date_from = st.date_input("From", value=date.today() - timedelta(days=30))
with col2:
    date_to = st.date_input("To", value=date.today())

# ── Load data ──────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_analytics(date_from, date_to):
    with get_db() as db:
        rows = db.execute(text("""
            SELECT
                al.asked_at,
                e.full_name,
                e.department,
                e.grade,
                al.intent,
                al.topic,
                al.language,
                al.unanswered,
                al.question_text
            FROM analytics_log al
            LEFT JOIN employees e ON e.id = al.employee_id
            WHERE al.asked_at::date BETWEEN :df AND :dt
            ORDER BY al.asked_at DESC
        """), {"df": str(date_from), "dt": str(date_to)}).fetchall()
    return pd.DataFrame([dict(r._mapping) for r in rows])

df = load_analytics(date_from, date_to)

if df.empty:
    st.info("No queries logged in this date range yet.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────
total      = len(df)
unanswered = int(df["unanswered"].sum())
unans_rate = round(unanswered / total * 100, 1) if total else 0
top_lang   = df["language"].value_counts().idxmax() if total else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total queries",   total)
m2.metric("Unanswered",      unanswered)
m3.metric("Unanswered rate", f"{unans_rate}%")
m4.metric("Top language",    top_lang)

st.divider()

# ── Charts ─────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Queries by topic")
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["topic", "count"]
    topic_counts = topic_counts[topic_counts["topic"] != ""]
    st.bar_chart(topic_counts.set_index("topic"))

with right:
    # 4 distinct language values:
    # english | arabic_msa | arabic_egyptian | franco
    st.subheader("Queries by language")
    label_map = {
        "english":         "English",
        "arabic_msa":      "Arabic (MSA)",
        "arabic_egyptian": "Arabic (Egyptian)",
        "franco":          "Franco Arabic",
        "arabic":          "Arabic (legacy)",   # rows before the fix
    }
    lang_counts = df["language"].value_counts().reset_index()
    lang_counts.columns = ["language", "count"]
    lang_counts["language"] = lang_counts["language"].map(lambda x: label_map.get(x, x))
    st.bar_chart(lang_counts.set_index("language"))

left2, right2 = st.columns(2)

with left2:
    st.subheader("Queries by department")
    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["department", "count"]
    st.bar_chart(dept_counts.set_index("department"))

with right2:
    st.subheader("Intent distribution")
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "count"]
    st.bar_chart(intent_counts.set_index("intent"))

st.divider()

# ── Unanswered questions ───────────────────────────────────────
st.subheader("🔴 Unanswered questions")
st.caption("Questions the bot could not answer — review to identify policy gaps.")
udf = df[df["unanswered"] == True][
    ["asked_at", "full_name", "department", "language", "question_text"]
]
if udf.empty:
    st.success("No unanswered questions in this period.")
else:
    st.dataframe(udf, use_container_width=True, hide_index=True)

st.divider()

# ── Full log ───────────────────────────────────────────────────
with st.expander("📋 Full query log"):
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "hr_chatbot_analytics.csv", "text/csv")