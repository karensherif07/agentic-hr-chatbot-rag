"""
chat_ui.py
Handles everything inside the assistant chat bubble:
- Rendering the answer with correct RTL/LTR styling
- Confidence badge
- Query Info expander (intent, lang, tools called)
- Your data used expander (personal + hybrid)
- Source Evidence expander (policy + hybrid)
- Saving chat history + session summary
- Analytics logging
- Triggering escalation state
"""

import streamlit as st
from setup import render_page_to_image
from utils import (
    is_no_info_answer, confidence_badge,
    summarize_history,
)
from sessions import save_session

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


def log_query(employee_id, intent, topic, lang, dialect, is_no_info, question):
    if lang == "arabic":
        log_lang = "arabic_egyptian" if dialect == "egyptian" else "arabic_msa"
    elif lang == "franco":
        log_lang = "franco"
    else:
        log_lang = "english"
    try:
        from database import get_db
        from sqlalchemy import text
        with get_db() as db:
            db.execute(text("""
                INSERT INTO analytics_log
                    (employee_id, intent, topic, language, unanswered, question_text, asked_at)
                VALUES (:eid, :intent, :topic, :lang, :unans, :q, NOW())
            """), {
                "eid":    employee_id,
                "intent": intent,
                "topic":  topic or "",
                "lang":   log_lang,
                "unans":  is_no_info,
                "q":      question[:300],
            })
    except Exception as e:
        print(f"[analytics] {e}")


def render_answer(
    answer: str,
    intent: str,
    lang: str,
    dialect: str | None,
    tools_called: list,
    cited_docs: list,
    scores_dict: dict,
    personal_data_str: str,
    _no_info: bool,
):
    """Render the full assistant response block inside a chat bubble."""

    is_arabic_script = (lang == "arabic")
    css = "rtl-answer" if is_arabic_script else "ltr-answer"

    # ── Answer text ───────────────────────────────────────────
    st.markdown(
        f'<div class="{css}">'
        f'{answer.replace(chr(10), "<br>")}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Confidence badge (policy + hybrid only) ───────────────
    if scores_dict and intent in ("policy", "hybrid") and not _no_info:
        peer_scores  = list(scores_dict.values())
        raw_score    = max(scores_dict.values())
        label, color = confidence_badge(raw_score, peer_scores)
        st.markdown(
            f'<span class="conf-badge" style="background:{color}">'
            f"{label}</span>",
            unsafe_allow_html=True,
        )

    # ── Query Info expander ───────────────────────────────────
    with st.expander("🔍 Query Info"):
        st.info(
            f"**Intent:** {intent} | **Lang:** {lang}"
            + (f" | **Dialect:** {dialect}" if dialect else "")
        )
        if tools_called:
            badges = " ".join(
                f'<span class="tool-badge">🔧 {t}</span>'
                for t in tools_called
            )
            st.markdown(f"**Tools used:** {badges}", unsafe_allow_html=True)

    # ── Your data used (personal + hybrid only) ───────────────
    if personal_data_str and intent in ("personal", "hybrid") and not _no_info:
        with st.expander("📋 Your data used to answer this"):
            st.code(personal_data_str, language=None)

    # ── Source Evidence (policy + hybrid only) ────────────────
    if cited_docs and intent in ("policy", "hybrid") and not _no_info:
        with st.expander("📄 Source Evidence"):
            unique_pages: dict = {}
            for d in cited_docs:
                p_no = d.metadata.get("page", 0) + 1
                src  = (
                    ARABIC_PDF_PATH
                    if ARABIC_PDF_PATH in d.metadata.get("source", "")
                    else ENGLISH_PDF_PATH
                )
                if (src, p_no) not in unique_pages:
                    unique_pages[(src, p_no)] = d

            for (pdf, p_no), d in unique_pages.items():
                src_label = "Arabic PDF" if pdf == ARABIC_PDF_PATH else "English PDF"
                st.markdown(f"**📄 Page {p_no} — {src_label}**")
                try:
                    st.image(render_page_to_image(pdf, p_no), width=700)
                except Exception:
                    direction = "rtl" if pdf == ARABIC_PDF_PATH else "ltr"
                    align     = "right" if direction == "rtl" else "left"
                    st.markdown(
                        f'<div style="direction:{direction};text-align:{align};'
                        f'background:#f9f9f9;padding:12px;'
                        f'border-left:3px solid #1976d2;">'
                        f"{d.page_content}</div>",
                        unsafe_allow_html=True,
                    )


def save_and_summarise(
    question: str,
    answer: str,
    lang: str,
    is_franco: bool,
    is_arabic_script: bool,
    en_llm,
    employee_id: int,
):
    """Append messages to history, summarise every 8 turns, persist to DB."""
    st.session_state.chat_history.append({
        "role":      "user",
        "content":   question,
        "is_arabic": False,
        "is_franco": is_franco,
    })
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "is_arabic": is_arabic_script,
        "is_franco": is_franco,
    })

    n = len(st.session_state.chat_history)
    if n % 8 == 0 or n <= 4:
        st.session_state.conversation_summary = summarize_history(
            en_llm,
            st.session_state.chat_history,
            st.session_state.conversation_summary,
        )

    save_session(
        employee_id,
        st.session_state.chat_history,
        st.session_state.conversation_summary,
    )


def maybe_trigger_escalation(is_no_info: bool, question: str, employee_name: str):
    """Set escalation session state flags if answer was not found."""
    if not is_no_info:
        return
    from escalation_ui import get_hr_email
    hr_email = get_hr_email()
    if hr_email:
        st.session_state["_esc_hr_email"] = hr_email
        st.session_state["_esc_question"] = question
        st.session_state["_esc_name"]     = employee_name
        st.session_state["_show_esc"]     = True