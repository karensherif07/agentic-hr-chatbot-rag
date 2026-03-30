import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

ARABIC_PDF_PATH = "ar_policy.pdf"
ENGLISH_PDF_PATH = "eng_policy.pdf"

# ─── Language Detection ────────────────────────────────────────

FRANCO_TIER1 = {
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma",
    "msh", "mesh", "mish", "mafish",
    "leh", "leih", "fein", "fen", "emta", "ezay", "meen", "eih", "eh",
    "ya3ni", "3ashan", "momken", "3ayz", "3ayza", "yenfa3", "ynfa3",
    "tayeb", "tamam", "keda", "kidda", "bas", "ad" , "2ad",
    "7aga", "haga", "feeh", "fieh", "la2", "aywa", "aiwa",
    "wenta", "wenti", "bs", "delwa2ty", "badein", "b3dein",
    "el", "di", "da", "dol", "aho", "ahi",
}

ENGLISH_STOP_WORDS = {"the", "is", "are", "what", "how", "who", "where", "of", "and", "to", "for"}


def detect_language_type(text: str) -> str:
    # Arabic unicode check — must come first
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"

    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    token_set = set(tokens)

    # Priority 1: Known Franco slang vocabulary
    if token_set & FRANCO_TIER1:
        return "franco"

    # Priority 2: Digit-letter combos (e.g. 3yoon, 7elw, ma3lesh)
    franco_hits = 0
    for tok in tokens:
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok):
            franco_hits += 1
    if franco_hits >= 1:
        return "franco"

    # Priority 3: English structural words — only reached after ruling out Franco
    if token_set & ENGLISH_STOP_WORDS:
        return "english"

    return "english"


def get_semantic_dialect(llm, arabic_text: str) -> str:
    try:
        prompt = (
            "You are an Arabic linguist. Classify the following Arabic text as either "
            "'egyptian' (Egyptian colloquial / عامية مصرية) or 'msa' (Modern Standard Arabic / فصحى). "
            "Reply with ONE word only: egyptian or msa.\n\nText: " + arabic_text
        )
        res = llm.invoke(prompt)
        return "egyptian" if "egyptian" in res.content.strip().lower() else "msa"
    except Exception:
        return "msa"


# ─── Text Cleaning & Normalization ────────────────────────────

def clean_pdf(text: str) -> str:
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u200e\u200f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text.lower()


def normalize_english(text: str) -> str:
    return text.lower()


# ─── Franco → Arabic Conversion ───────────────────────────────

FRANCO_MAP = {"2":"ء", "3":"ع", "4":"ش", "5":"خ","7":"ح", "8":"غ"}

FRANCO_WORDS = {
    "3ayz": "عايز", "3ayza": "عايزة", "a3raf": "اعرف", "ezay": "ازاي", "fein": "فين", 
    "leh": "ليه", "leih": "ليه", "msh": "مش", "mesh": "مش", "mish": "مش", "ana": "انا", 
    "enta": "انت", "enti": "انتي", "el": "ال", "ya3ni": "يعني", "3ashan": "عشان", 
    "tayeb": "طيب", "tamam": "تمام", "keda": "كده", "kidda": "كده", "bas": "بس", "bs": "بس",
    "la2": "لأ", "aywa": "ايوه", "aiwa": "ايوه", "momken": "ممكن", "7aga": "حاجة", 
    "haga": "حاجة", "emta": "امتى", "meen": "مين", "eih": "ايه", "eh": "ايه", "fen": "فين", 
    "mafish": "مافيش", "yenfa3": "ينفع", "ynfa3": "ينفع", "feeh": "فيه", "fieh": "فيه", 
    "delwa2ty": "دلوقتي", "badein": "بعدين", "b3dein": "بعدين", "da": "ده", "di": "دي", 
    "dol": "دول", "aho": "اهو", "ahi": "اهي", "ehna": "احنا", "ento": "انتوا", 
    "howa": "هو", "hya": "هي", "homma": "هما"
}

def franco_to_arabic(text: str) -> str:
    """
    FIX: Word-level dictionary lookup MUST happen before digit substitution.
    In the original code, FRANCO_MAP digit replacement ran first across the whole
    string — so "3ayz" became "عayz" before the FRANCO_WORDS dict could match it,
    meaning the dict lookup NEVER worked. Now we do word-level lookup first, then
    apply digit substitution only to words not found in the dictionary.
    """
    words = text.lower().split()
    converted = []
    for w in words:
        if w in FRANCO_WORDS:
            converted.append(FRANCO_WORDS[w])
        else:
            # Apply digit → Arabic-letter substitution for unknown words
            result = w
            for digit, arabic_char in FRANCO_MAP.items():
                result = result.replace(digit, arabic_char)
            converted.append(result)
    return " ".join(converted)


# ─── Retrieval Utilities ──────────────────────────────────────

def tokenize(text: str) -> list:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    scores = {}

    def add(docs):
        for rank, d in enumerate(docs):
            key = d.page_content[:120]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    add(docs1)
    add(docs2)
    doc_map = {d.page_content[:120]: d for d in docs1 + docs2}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    faiss_docs = vs.similarity_search(raw_query, k=k)
    normalized_q = normalize_fn(raw_query)
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


def translate(llm, text: str, target_language: str) -> str:
    try:
        prompt = (
            f"Translate the following text to {target_language}. "
            f"Return ONLY the translation, nothing else.\n\nText: {text}"
        )
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception:
        return text


def build_context(docs: list) -> str:
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        source = d.metadata.get("source", "")
        lang_tag = "AR" if ARABIC_PDF_PATH in source else "EN"
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


# ─── Validation & Citation Utilities ─────────────────────────

def validate(ans: str, lang: str, has_citations: bool = False) -> str:
    """
    Warn if the LLM produced no citations at all.
    `has_citations` is pre-computed from the raw answer before [Page N] stripping,
    so this function does not need to re-scan for markers.
    """
    ans = ans.strip()

    not_found_phrases = [
        "not available in the policy",
        "غير متوفرة في وثائق",
        "not found in",
    ]
    is_not_found = any(p.lower() in ans.lower() for p in not_found_phrases)

    if not has_citations and not is_not_found:
        if lang == "arabic":
            ans += "\n\n⚠️ لم يتم ذكر أرقام الصفحات في هذه الإجابة."
        else:
            ans += "\n\n⚠️ Page citations were not produced for this answer."

    return ans


def get_cited_pages(answer: str) -> set:
    return {int(n) for n in re.findall(r"\[Page\s*(\d+)", answer, re.IGNORECASE)}


def strip_citations(answer: str) -> str:
    """Remove all [Page N] / [Page N | AR] / [Page N | EN] markers from displayed answer."""
    cleaned = re.sub(r"\s*\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", answer, flags=re.IGNORECASE)
    # Collapse any double spaces left behind
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.strip()


def filter_cited_chunks(docs: list, cited_pages: set) -> list:
    if not cited_pages:
        return docs
    return [
        d for d in docs
        if (d.metadata.get("page", 0) + 1) in cited_pages
    ]


# ─── Prompt Templates ─────────────────────────────────────────

CITATION_EXAMPLE_EN_2ND = """
EXAMPLES of correct cited answers:
Q: Can I take unpaid leave?
A: Yes, you can apply for unpaid leave after exhausting your annual leave balance [Page 8].

Q: How many annual leave days do employees get?
A: Employees are entitled to 21 working days of annual leave per year [Page 5].
   This increases to 30 days after 10 years of continuous service [Page 5].
"""

CITATION_EXAMPLE_AR_2ND = """
أمثلة على إجابات صحيحة:
س: هل يمكنني أخذ إجازة بدون راتب؟
ج: نعم، يمكنك التقدم بطلب إجازة بدون راتب بعد استنفاد رصيد إجازتك السنوية [Page 8].

س: كم يوم إجازة سنوية يحق للموظف؟
ج: يحق للموظف 21 يوم عمل إجازة سنوية في السنة [Page 5]. وترتفع إلى 30 يوماً بعد 10 سنوات [Page 5].
"""

BASE_EN = (
    "You are an HR policy assistant.\n\n"
    "RULES — follow exactly:\n"
    "1. Answer ONLY from the provided context. Never use outside knowledge.\n"
    "2. Every sentence in your answer MUST end with a citation in the format [Page N].\n"
    "   Use the page number from the context label, e.g. [Page 5] or [Page 5 | AR].\n"
    "3. If the same fact appears on multiple pages, cite all of them: [Page 3] [Page 7].\n"
    "4. If the answer is not in the context, respond with exactly:\n"
    "   \"This information is not available in the policy documents.\"\n"
    "5. Do not add information not present in the context.\n"
    "6. MIRROR the person/voice of the question:\n"
    "   - If the question uses 'I' or 'can I' or 'my' → answer with 'you' / 'you can' / 'your'.\n"
    "   - If the question uses 'employees' or 'he/she/they' → answer in third person.\n"
    + CITATION_EXAMPLE_EN_2ND
)

BASE_AR = (
    "أنت مساعد سياسة الموارد البشرية.\n\n"
    "القواعد — اتبعها بدقة:\n"
    "1. أجب فقط من السياق المقدم. لا تستخدم أي معلومات خارجية.\n"
    "2. كل جملة في إجابتك يجب أن تنتهي باقتباس بالتنسيق [Page N].\n"
    "   استخدم رقم الصفحة من تسمية السياق، مثلاً [Page 5] أو [Page 5 | AR].\n"
    "3. إذا لم تكن المعلومات في السياق، قل بالضبط:\n"
    "   \"هذه المعلومات غير متوفرة في وثائق السياسة.\"\n"
    "4. لا تضف معلومات غير موجودة في السياق.\n"
    "5. طابق ضمير السؤال في الإجابة:\n"
    "   - إذا كان السؤال بصيغة المتكلم (أنا / هل أستطيع / ممكن آخذ) → أجب بصيغة المخاطب (أنت / يمكنك / تستحق).\n"
    "   - إذا كان السؤال عن الموظف أو الغائب → أجب بصيغة الغائب (يستحق الموظف / تنص السياسة).\n"
    + CITATION_EXAMPLE_AR_2ND
)

english_prompt = PromptTemplate(
    template=BASE_EN + "\nRespond in English.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

msa_prompt = PromptTemplate(
    template=BASE_AR + "\nأجب باللغة العربية الفصحى.\n\nالسياق:\n{context}\n\nالسؤال: {question}\nالإجابة:",
    input_variables=["context", "question"]
)

egy_prompt = PromptTemplate(
    template=BASE_AR + "\nأجب باللهجة المصرية العامية.\n\nالسياق:\n{context}\n\nالسؤال: {question}\nالإجابة:",
    input_variables=["context", "question"]
)

CITATION_EXAMPLE_FRANCO = """
Amtela sa7:
So2al (2nd person): momken akhod agaza bel mabla3?
Egaba: aywa, momken ta5od agaza bel mabla3 lw 3andak raseed kafi [Page 8].

So2al (3rd person): el mowazaf bya5od kam yom agaza?
Egaba: el mowazaf bystahel 21 yom 3amal agaza f el sana [Page 5]. lw 3amal 10 sneen, byb2a 3ando 30 yom [Page 5].
"""

FRANCO_BASE = (
    "Enta mosa3ed siyaset el HR.\n\n"
    "El rules — etba3ha:\n"
    "1. Egib bass men el context el maktub tala7t. Matesta5dimsh ay ma3lomat bara.\n"
    "2. Kol gomla lazem tet7et 3aleha citation bel format [Page N].\n"
    "3. Lw el ma3loma mesh mawgoda f el context, 2ol:\n"
    "   \"El ma3loma di mesh mawgoda f el policy.\"\n"
    "4. Matdifsh 7aga mesh f el context.\n"
    "5. Etba3 damir el so2al:\n"
    "   - Lw el so2al bel mutakalem (ana / momken akhod / 3ayez a3raf) → egib bel muka5ab (enta / momken ta5od / 3andak).\n"
    "   - Lw el so2al 3an el mowazaf aw el gha2eb → egib bel gha2eb (el mowazaf bystahel...).\n"
    "6. Egib bel Franco 3arabi bass: kelmaat 3arabiyya maktuba bel 7oroof el latiniyya wel arqam "
    "(3 = ع, 7 = ح, 2 = ء, 5 = خ, 6 = ط, 8 = ق, 9 = ص). Matektibsh 3arabi fa9i7 wala inglizi.\n"
    + CITATION_EXAMPLE_FRANCO
)

franco_prompt = PromptTemplate(
    template=FRANCO_BASE + "\nEl context:\n{context}\n\nEl so2al: {question}\nEl egaba:",
    input_variables=["context", "question"]
)


# ─── Setup (cached) ───────────────────────────────────────────

@st.cache_resource
def setup():
    emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    def build_index(path: str, normalize_fn):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        pages = PyMuPDFLoader(path).load()
        for d in pages:
            d.page_content = clean_pdf(d.page_content)
            d.metadata["doc_type"] = "policy"
            d.metadata["lang"] = "arabic" if ARABIC_PDF_PATH in path else "english"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        docs = splitter.split_documents(pages)
        bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in docs]
        vs = FAISS.from_documents(docs, emb)
        bm25 = BM25Okapi(bm25_corpus)
        return vs, bm25, docs

    ar_index = build_index(ARABIC_PDF_PATH, normalize_arabic)
    en_index = build_index(ENGLISH_PDF_PATH, normalize_english)

    ar_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    en_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",
        temperature=0
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return ar_index, en_index, ar_llm, en_llm, reranker


# ─── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="HR Policy Assistant", layout="wide")

# RTL/LTR CSS for proper Arabic and Franco rendering
st.markdown("""
<style>
.rtl-answer {
    direction: rtl;
    text-align: right;
    font-size: 1rem;
    line-height: 1.9;
    padding: 0.5rem 0;
}
.ltr-answer {
    direction: ltr;
    text-align: left;
    font-size: 1rem;
    line-height: 1.9;
    padding: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Policy Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

question = st.text_input("Ask your question:")

try:
    ar_index, en_index, ar_llm, en_llm, reranker = setup()
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

if question:
    with st.spinner("Searching policy documents..."):
        try:
            lang = detect_language_type(question)

            # ── Build bilingual queries ──────────────────────────────
            if lang == "english":
                q_en = question
                q_ar = translate(ar_llm, question, "Arabic")

            elif lang == "arabic":
                q_ar = question
                q_en = translate(en_llm, question, "English")

            elif lang == "franco":
                # FIX: Do NOT apply normalize_arabic to q_ar here.
                # normalize_arabic is called inside retrieve() via normalize_fn,
                # so FAISS gets clean Arabic while BM25 gets normalized Arabic.
                # Applying normalize_arabic here would corrupt q_ar before FAISS sees it.
                franco_arabic = franco_to_arabic(question)
                q_ar = translate(ar_llm, franco_arabic, "Modern Standard Arabic")
                q_en = translate(en_llm, q_ar, "English")

            # ── Dialect detection ────────────────────────────────────
            dialect = None
            if lang == "arabic":
                dialect = get_semantic_dialect(ar_llm, question)
            elif lang == "franco":
                dialect = "franco"

            # ── Unpack indices ───────────────────────────────────────
            ar_vs, ar_bm25, ar_docs = ar_index
            en_vs, en_bm25, en_docs = en_index

            # ── Retrieve from both indices ───────────────────────────
            docs_ar = retrieve(q_ar, ar_vs, ar_bm25, ar_docs, normalize_arabic)
            docs_en = retrieve(q_en, en_vs, en_bm25, en_docs, normalize_english)

            combined = rrf(docs_ar, docs_en)

            rerank_query = q_ar if lang in ("arabic", "franco") else q_en
            top_docs = rerank(rerank_query, combined, reranker, top_n=5)

            if not top_docs:
                st.warning("No relevant documents found.")
                st.stop()

            context = build_context(top_docs)

            # ── Select prompt and LLM ────────────────────────────────
            if lang == "english":
                prompt = english_prompt
                llm = en_llm
            elif lang == "franco":
                prompt = franco_prompt
                llm = ar_llm
            else:  # arabic
                prompt = egy_prompt if dialect == "egyptian" else msa_prompt
                llm = ar_llm

            final_prompt = prompt.format(context=context, question=question)
            res = llm.invoke(final_prompt)

            raw_answer = res.content

            # Parse citations from the raw LLM output BEFORE stripping markers
            cited_pages = get_cited_pages(raw_answer)
            cited_docs = filter_cited_chunks(top_docs, cited_pages)

            # Strip [Page N] markers so they don't appear in the displayed answer
            clean_answer = strip_citations(raw_answer)

            # Validate on the clean answer (citation check uses cited_pages already parsed above)
            answer = validate(clean_answer, lang, has_citations=bool(cited_pages))

            # ── Layout ───────────────────────────────────────────────
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Answer")
                css_class = "rtl-answer" if lang == "arabic" else "ltr-answer"
                st.markdown(
                    f'<div class="{css_class}">{answer.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
                st.divider()
                if st.button("🔄 Translate answer"):
                    if lang == "english":
                        translated = translate(ar_llm, answer, "Arabic")
                        st.markdown(
                            f'<div class="rtl-answer">{translated.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        translated = translate(en_llm, answer, "English")
                        st.markdown(
                            f'<div class="ltr-answer">{translated.replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True
                        )

            with col2:
                st.subheader("Query Info")
                st.info(f"**Detected language:** {lang}")
                if dialect:
                    st.info(f"**Dialect:** {dialect}")
                st.info(f"**Arabic query:** {q_ar}")
                st.info(f"**English query:** {q_en}")

            # ── Source chunks ────────────────────────────────────────
            display_docs = cited_docs if cited_docs else top_docs
            chunk_label = (
                f"📄 Source Chunks ({len(cited_docs)} cited)"
                if cited_docs
                else "📄 Source Chunks (retrieved — no citations parsed)"
            )
            with st.expander(chunk_label):
                for i, d in enumerate(display_docs, 1):
                    source = (
                        "Arabic PDF"
                        if ARABIC_PDF_PATH in d.metadata.get("source", "")
                        else "English PDF"
                    )
                    page_no = d.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {i} — Page {page_no} ({source})**")
                    st.write(d.page_content)

        except Exception as e:
            st.error(f"Error: {str(e)}")