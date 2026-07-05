import base64

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from deps import get_current_employee, get_models, ModelBundle
from agent import run_agent
from nlp_utils import detect_language_type, get_semantic_dialect
from utils import build_history_str, is_no_info_answer, translate, strip_citations
from sessions import load_session, save_session, clear_session
from speech import text_to_speech, tts_audio_format, tts_available
from escalation import get_hr_email, send_escalation_email, send_contact_hr_email

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _summarize_and_save(employee_id, chat_history, conversation_summary, summary_llm):
    from utils import summarize_history
    n = len(chat_history)
    if n % 8 == 0 or n <= 4:
        conversation_summary = summarize_history(summary_llm, chat_history, conversation_summary)
    save_session(employee_id, chat_history, conversation_summary)
    return conversation_summary


@router.get("/history")
def get_history(emp: dict = Depends(get_current_employee)):
    history, summary = load_session(emp["id"])
    return {"history": history, "summary": summary}


@router.delete("/history")
def delete_history(emp: dict = Depends(get_current_employee)):
    clear_session(emp["id"])
    return {"ok": True}


class MessageBody(BaseModel):
    question: str
    # Frontend keeps its own copy of history/summary and sends it back each
    # turn — same pattern app.py used with st.session_state, just explicit.
    chat_history: list = []
    conversation_summary: str = ""


def _doc_to_json(d):
    return {
        "page": d.metadata.get("page", 0) + 1,
        "source": d.metadata.get("source", ""),
        "doc_name": d.metadata.get("doc_name") or d.metadata.get("source", ""),
        "lang": "arabic" if "ar_" in (d.metadata.get("source") or "") else "english",
        "content": d.page_content,
    }


@router.post("/message")
def send_message(
    body: MessageBody,
    emp: dict = Depends(get_current_employee),
    m: ModelBundle = Depends(get_models),
):
    question = body.question.strip()
    lang = detect_language_type(question)
    dialect = get_semantic_dialect(question, m.dialect_pipe) if lang == "arabic" else None

    history_str = build_history_str(body.chat_history, body.conversation_summary)

    result = run_agent(
        question=question,
        employee_id=emp["id"],
        lang=lang,
        dialect=dialect,
        history_str=history_str,
        ar_index=m.ar_index,
        en_index=m.en_index,
        routing_llm=m.routing_llm,
        en_llm=m.en_llm,
        ar_llm=m.ar_llm,
        critique_llm=m.critique_llm,
        reranker=m.reranker,
        ara_tokenizer=m.ara_tokenizer,
        dialect_pipe=m.dialect_pipe,
    )

    answer = result["answer"]
    is_franco = lang == "franco"
    is_arabic_script = lang == "arabic"
    no_info = is_no_info_answer(answer)

    new_history = body.chat_history + [
        {"role": "user", "content": question, "is_arabic": False, "is_franco": is_franco},
        {"role": "assistant", "content": answer, "is_arabic": is_arabic_script, "is_franco": is_franco},
    ]
    summary_llm = m.en_llm if lang == "english" else m.ar_llm
    new_summary = _summarize_and_save(emp["id"], new_history, body.conversation_summary, summary_llm)

    # personal data string: strip backend-only ELIGIBILITY PRE-CHECK block
    pdata = result["personal_data"]
    if pdata and "ELIGIBILITY PRE-CHECK" in pdata:
        lines, out, skipping = pdata.splitlines(), [], False
        for line in lines:
            if line.strip().startswith("ELIGIBILITY PRE-CHECK"):
                skipping = True
                continue
            if skipping:
                if line.strip() == "":
                    skipping = False
                continue
            out.append(line)
        pdata = "\n".join(out).strip()

    from chat_ui import log_query
    log_query(emp["id"], result["intent"], result["topic"], lang, dialect, no_info, question)

    return {
        "answer": answer,
        "lang": lang,
        "dialect": dialect,
        "intent": result["intent"],
        "topic": result["topic"],
        "tools_called": result["tools_called"],
        "no_info": no_info,
        "personal_data": pdata if result["intent"] in ("personal", "hybrid") and not no_info else "",
        "cited_docs": [_doc_to_json(d) for d in result["cited_docs"]] if result["intent"] in ("policy", "hybrid") and not no_info else [],
        "chat_history": new_history,
        "conversation_summary": new_summary,
    }


class TranslateBody(BaseModel):
    text: str
    source_lang: str  # "english" | "arabic" | "franco"


@router.post("/translate")
def translate_answer(body: TranslateBody, m: ModelBundle = Depends(get_models), emp: dict = Depends(get_current_employee)):
    target = "Arabic" if body.source_lang == "english" else "English"
    llm = m.ar_llm if body.source_lang == "english" else m.en_llm
    return {"translated": translate(llm, body.text, target)}


class TTSBody(BaseModel):
    text: str
    lang: str


@router.post("/tts")
def tts(body: TTSBody, emp: dict = Depends(get_current_employee)):
    if not tts_available():
        return {"available": False}
    audio = text_to_speech(strip_citations(body.text), lang=body.lang)
    if not audio:
        return {"available": False}
    return {
        "available": True,
        "mime": tts_audio_format(body.lang),
        "audio_base64": base64.b64encode(audio).decode(),
    }


class EscalateBody(BaseModel):
    question: str


@router.post("/escalate")
def escalate(body: EscalateBody, emp: dict = Depends(get_current_employee)):
    hr_email = get_hr_email(exclude_employee_id=emp["id"])
    if not hr_email:
        return {"sent": False, "reason": "HR email not configured."}
    sent = send_escalation_email(emp["full_name"], hr_email, body.question)
    return {"sent": sent}


class ContactHRBody(BaseModel):
    subject: str
    body: str


@router.post("/contact-hr")
def contact_hr(body: ContactHRBody, emp: dict = Depends(get_current_employee)):
    hr_email = get_hr_email(exclude_employee_id=emp["id"])
    if not hr_email:
        return {"sent": False, "reason": "HR email not configured."}
    sent = send_contact_hr_email(
        emp["full_name"], emp["email"], hr_email, body.subject, body.body
    )
    return {"sent": sent}
