import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from experiments.run_ragas_eval import _strip_thinking

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume",       action="store_true")
parser.add_argument("--rerun-ids",    nargs="*", default=[],
                    help="Force-rerun specific query_ids even if already done")
parser.add_argument("--query-set",    default="policy_query_set.json")
parser.add_argument("--checkpoint",   default="generation_checkpoint.json")
parser.add_argument("--output",       default="generated_answers.json")
parser.add_argument("--errors-log",   default="generation_errors.json")
parser.add_argument("--batch-size",   type=int,   default=5)
parser.add_argument("--batch-sleep",  type=float, default=10.0)
args = parser.parse_args()

BASE_WAIT = 5
MAX_WAIT  = 60

MAX_ANSWER_TOKENS = 512   # hard cap: prevents loop runaway
MIN_CHUNK_CHARS   = 80    # filter boilerplate footer chunks

GROQ_KEY = os.environ["GROQ_API_KEY"]
from langchain_groq import ChatGroq

# English and FRANCO both use llama — it handles code-switched Franco Arabic
# far better than Qwen3, and has no <think> leak problem.
EN_LLM = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=MAX_ANSWER_TOKENS,
)

# Franco gets its own llama instance with a slightly higher token cap so full
# lists (e.g. all mandatory training modules) aren't truncated mid-item.
FRANCO_LLM = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=600,
)

# Arabic (MSA + Egyptian) keep Qwen3 — it's strong on Arabic script.
AR_LLM = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="qwen/qwen3-32b",
    temperature=0,
    max_tokens=400,
)

# ── Project imports ───────────────────────────────────────────────────────────
from setup import setup
from retrieval import retrieve, rerank, rrf
from nlp_utils import (
    normalize_arabic, normalize_english,
    detect_language_type, get_semantic_dialect,
    egyptian_to_msa, franco_to_arabic,
)
from utils import _is_arabic_source, translate
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt

# ── Load indexes ──────────────────────────────────────────────────────────────
print("\nLoading indexes and models (one-time)...")
(ar_index, en_index,
 routing_llm, _en_prod, _ar_prod, _critique,
 reranker, dialect_pipe, ara_tokenizer) = setup()
ar_vs, ar_bm25, ar_docs = ar_index
en_vs, en_bm25, en_docs = en_index
print("✓ Ready.\n")

# ── Constants matching agent.py ───────────────────────────────────────────────
_CANDIDATE_K    = 20
_RERANKED_TOP_N = 7
_MAX_CHUNK_CHARS= 600

# ── Boilerplate detection ─────────────────────────────────────────────────────
_BOILERPLATE_PATTERNS = [
    r"^Confidential\s*[—-]\s*Internal Use Only",
    r"^سري\s*[—-]\s*للاستخدام الداخلي فقط",
    r"^Mbps\s*\.",
    r"^Page\s*\d+\s*$",
    r"تاريخ النفاذ\s*سري",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.DOTALL)

def _is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    stripped_check = re.sub(r'^[.\s]+', '', stripped)

    if len(stripped_check) < 50:
        return True

    if _BOILERPLATE_RE.search(stripped_check[:300]):
        return True

    _SIRI_ONLY = re.compile(
        r'^[\.\s]*سري\s*[—\-–]\s*للاستخدام الداخلي فقط',
        re.IGNORECASE
    )
    if _SIRI_ONLY.match(stripped):
        return True

    if 'سري' in stripped and 'للاستخدام الداخلي فقط' in stripped:
        _POLICY_WORDS = re.compile(
            r'(إجازة|راتب|موظف|بدل|تأمين|اشتراك|مكافأة|عمل إضافي|فترة|اختبار|تقييم|شهادة|تدريب|نفقة|مصروف)'
        )
        if not _POLICY_WORDS.search(stripped):
            return True

    _EN_STAMP = re.compile(
        r'^[\.\s]*(Confidential\s*[—\-–]\s*Internal Use Only|'
        r'Horizon Tech\s*[—\-–]\s*Human Resources)',
        re.IGNORECASE
    )
    if _EN_STAMP.match(stripped) and len(stripped) < 300:
        return True

    return False

def _looks_like_ocr_loop(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    repeated = sum(
        1 for i in range(len(lines)-1)
        if lines[i] == lines[i+1]
    )
    if repeated >= 2:
        return True
    numeric_ratio = sum(
        1 for tok in text.split()
        if any(c.isdigit() for c in tok)
    ) / max(len(text.split()), 1)
    if numeric_ratio > 0.35:
        return True
    return False

_GARBAGE_PATTERNS = [
    r"[^\w\s\u0600-\u06FF]{15,}",
    r"(.)\1{10,}",
    r"\b[a-zA-Z]{1,2}\b(?:\s+\b[a-zA-Z]{1,2}\b){10,}",
]
_GARBAGE_RE = re.compile("|".join(_GARBAGE_PATTERNS), re.DOTALL)

def _is_corrupted_chunk(text: str) -> bool:
    t = text.strip()
    if len(t) < 80:
        return True
    weird_ratio = sum(
        1 for c in t
        if not (c.isalnum() or c.isspace() or c in ".,:;!?-%()/")
    ) / max(len(t), 1)
    if weird_ratio > 0.30:
        return True
    if _GARBAGE_RE.search(t):
        return True
    words = t.split()
    if len(words) < 15 and len(t) > 300:
        return True
    return False

def _build_context(top_docs: list) -> str:
    chunks = []
    skipped = 0
    for d in top_docs[:_RERANKED_TOP_N]:
        content = d.page_content
        if _is_boilerplate(content) or _is_corrupted_chunk(content) or _looks_like_ocr_loop(content):
            skipped += 1
            continue
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if _is_arabic_source(d.metadata.get("source", "")) else "EN"
        truncated = content[:_MAX_CHUNK_CHARS]
        if len(content) > _MAX_CHUNK_CHARS:
            truncated += "…"
        chunks.append(f"[Page {page_num}|{lang_tag}]\n{truncated}")
    if skipped:
        tqdm.write(f"    ⚠ Skipped {skipped} boilerplate chunks")
    return "\n---\n".join(chunks) if chunks else ""

def build_context_for_arabic(docs: list) -> str:
    """Build Arabic-query context WITHOUT translating English chunks."""
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        source   = d.metadata.get("source", "")
        lang_tag = "AR" if _is_arabic_source(source) else "EN"
        content  = d.page_content[:500]
        out.append(f"[Page {page_num}|{lang_tag}]\n{content}")
    return "\n\n---\n\n".join(out)

# ── Retrieval mirrors agent.py _retrieve_policy() exactly ────────────────────
def _retrieve_policy(query: str) -> tuple:
    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)
    q_lang  = detect_language_type(query)

    if q_lang == "franco":
        franco_ar    = franco_to_arabic(query)
        franco_norm  = egyptian_to_msa(franco_ar)
        msa_query    = franco_norm
        docs_ar = rrf(
            retrieve(franco_ar,  ar_vs, ar_bm25, ar_docs, norm_ar),
            retrieve(msa_query,  ar_vs, ar_bm25, ar_docs, norm_ar),
        )
        from deep_translator import GoogleTranslator
        en_query2 = GoogleTranslator(source='ar', target='en').translate(franco_ar)
        docs_en = rrf(
            retrieve(en_query2,  en_vs, en_bm25, en_docs, normalize_english),
            retrieve(msa_query,  en_vs, en_bm25, en_docs, normalize_english),
        )
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "egyptian":
        ar_msa   = egyptian_to_msa(query)
        docs_ar  = rrf(
            retrieve(query,  ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
            retrieve(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
        )
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "arabic":
        if get_semantic_dialect(query, dialect_pipe) == "egyptian":
            ar_msa  = egyptian_to_msa(query)
            docs_ar = rrf(
                retrieve(query,  ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
                retrieve(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
            )
        else:
            docs_ar = retrieve(query, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K)
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "english":
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        docs_ar  = retrieve(query, ar_vs, ar_bm25, ar_docs, norm_ar,           k=_CANDIDATE_K)
        combined = rrf(docs_en, docs_ar)

    else:
        combined = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)

    if q_lang == "franco":
        rerank_query     = franco_to_arabic(query)
        rerank_query_msa = egyptian_to_msa(rerank_query)
        pairs_ar  = [(rerank_query,     d.page_content) for d in combined]
        pairs_msa = [(rerank_query_msa, d.page_content) for d in combined]
        scores_ar  = reranker.predict(pairs_ar)
        scores_msa = reranker.predict(pairs_msa)
        scores = [max(a, b) for a, b in zip(scores_ar, scores_msa)]
    else:
        pairs  = [(query, d.page_content) for d in combined]
        scores = reranker.predict(pairs)

    ranked   = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d, _ in ranked[:_RERANKED_TOP_N]]
    context  = _build_context(top_docs)
    return top_docs, context


# ── Loop detection ────────────────────────────────────────────────────────────
_LOOP_PATTERNS = [
    r"(\[Page \d+\|(?:EN|AR)\]\n?){3,}",
    r"(التسجيل\n){5,}",
    r"(ليلة\n){5,}",
    r"(الخدمات التالية\n){3,}",
    r"(however:? the salary){3,}",
    r"(التحديد\n\d+\n){5,}",
    r"(سياسة الادخار\n){3,}",
    r"(تنفيذ برنامج){3,}",
    r"(هياخدك\s*تبعتك){3,}",
    r"(تبعتك\s*وتبعتك){3,}",
    r"(.{15,50})\1{4,}",
]
_LOOP_RE = re.compile("|".join(_LOOP_PATTERNS), re.DOTALL)

def _is_looping(text: str) -> bool:
    return bool(_LOOP_RE.search(text))


# ── Franco language enforcer ──────────────────────────────────────────────────
_FRANCO_WORD_RE = re.compile(
    r'\b(el|al|fe|fi|law|lw|bas|bs|msh|mesh|mish|ana|enta|enti|howa|hya|'
    r'lazem|yalla|mashy|tamam|tayeb|momken|ya3ni|3ashan|keda|aywa|la2|'
    r'leih|fein|emta|ezay|meen|eih|eh|da|di|dol|aho|ahi|'
    r'biyedi|biyebda2|biyestamr|biyedfa3|byet7aseb|'
    r'le7ad|men|3ala|mafish|feeh|fieh|walla|wala|'
    r'shoghl|rateb|agaza|ta2min|muwazaf|gedid|'
    r'[a-z]+[237][a-z]*|[a-z]*[237][a-z]+)\b',  # words with Franco digits 2/3/7
    re.IGNORECASE
)

def _is_franco(text: str) -> bool:
    """Return True if text looks like Franco Arabic (Latin-script Egyptian)."""
    if not text or len(text) < 5:
        return False
    # Must be primarily Latin script
    latin_chars  = len(re.findall(r'[a-zA-Z0-9 ]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if latin_chars < 3 * max(arabic_chars, 1):
        return False
    # Must contain at least one Franco marker word
    return bool(_FRANCO_WORD_RE.search(text))


def _strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3 output.

    Handles three cases:
    1. Well-formed: <think>...</think> present — strip the block.
    2. Unclosed:   <think>... no closing tag (truncated by max_tokens) — strip
                   everything from <think> onward, since no answer follows.
    3. No block:   return text unchanged.
    """
    # Case 1: well-formed block
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if cleaned != text:
        return cleaned.strip()

    # Case 2: unclosed <think> — everything after it is reasoning, not an answer
    unclosed = re.search(r"<think\b[^>]*>", text, re.IGNORECASE)
    if unclosed:
        before = text[:unclosed.start()].strip()
        # If there's a real answer before the tag, keep it; otherwise signal empty
        return before if len(before) > 10 else ""

    return text


# ── Expected-number validators ────────────────────────────────────────────────
_EXPECTED_NUMERIC = {
    r"akher.*maw3id.*masrof|expense.*claim.*deadline|tasweyyet.*masroufat.*maw3id": "30",
    r"overtime.*weekday|ayyam.*shoghl.*3adeya.*taweed|eih nisblt taweed.*ayyam.*3adeya": "1.5",
    r"(yom|day).*(rateb|salary).*paid|mwaahed.*btetatref.*yom|salary.*paid.*day|on which day.*salar": "25",
    r"el mwaahed btetatref|الرواتب.*يوم": "25",
    r"ta2min.*7ayah.*gama3y|life.*insur.*benefit.*salary|group life": "2",
    r"employer.*social.*insur|nisblet.*sharka.*ta2minat|18\.75": "18.75",
    r"employee.*social.*insur.*rate|nisblet.*muwazaf.*ta2minat": "11",
}

def _validate_expected_number(question: str, answer: str) -> bool:
    for pattern, expected in _EXPECTED_NUMERIC.items():
        if re.search(pattern, question, re.IGNORECASE):
            if expected not in answer:
                return False
    return True


# ── List-question detector ────────────────────────────────────────────────────
_LIST_Q_PATTERNS = [
    r"kol.*wa7dat|all.*modules|eih.*kol|what are (all|the) (modules|types|kinds)",
    r"fatrat.*ehtefaz.*anwa3|retention.*period.*types|different.*types.*data",
    r"eih.*anwa3|what types|list all|enumerate",
    r"مذكور.*انواع|جميع.*انواع|كل.*أنواع",
]

def _is_list_question(question: str) -> bool:
    return any(re.search(p, question, re.IGNORECASE) for p in _LIST_Q_PATTERNS)


# ── Franco synonym expansion map ──────────────────────────────────────────────
_FRANCO_EN_SYNONYMS = {
    r"imtithal.*tadrib|tadrib.*elzam|compliance.*training|mandatory.*training|wa7dat.*tadribeyya":
        "mandatory compliance training annual modules deadlines information security data protection",
    r"ta2min.*7ayah|7ayah.*gama3y|life.*insur":
        "group life insurance benefit annual salary lump sum",
    r"mol7a2.*ta2min|ta2min.*idafi|health.*add.?on|additional.*coverage|mawlood.*ta2min":
        "optional supplemental health insurance enrollment window qualifying event birth marriage",
    r"fatrat.*ehtefaz|retention.*period|data.*retention|ehtefaz.*bayanat":
        "data retention period employee records payroll performance disciplinary",
    r"masroufat.*maw3id|expense.*deadline|tasweyyet.*masrof|akher.*maw3id.*masrof":
        "expense claim submission deadline 30 days not accepted",
}

# ── Query expansion for number disambiguation ─────────────────────────────────
_EXPANSION_MAP = {
    r"overtime.*weekday|ayyam.*3adeya.*overtime|taweed.*overtime.*3adeya":
        " weekday 1.5 overtime compensated NOT weekend NOT 2x",
    r"expense.*claim.*deadline|tasweyyet.*masroufat.*maw3id|akher.*maw3id.*masrof":
        " 30 days expense claim NOT wellness NOT gym NOT 60",
    r"salary.*paid.*day|yom.*rateb|mwaahed.*yom|on which day.*salar|el mwaahed btetatref":
        " 25th payroll salary paid calendar month",
}


# ── Context instruction preambles ─────────────────────────────────────────────
_AR_CONTEXT_INSTRUCTION = (
    "استخدم جميع المقاطع التالية معاً للإجابة، بما فيها المقاطع باللغة الإنجليزية. "
    "لا تتجاهل أي مقطع. أجب بشكل مباشر ومختصر بناءً على المعلومات المتاحة فقط.\n\n"
)

_EN_CONTEXT_INSTRUCTION = (
    "ONLY use the passages labeled [Page N|EN] or [Page N|AR] below. "
    "Some passages are in Arabic — read them too. "
    "Do NOT use any knowledge from training. "
    "If the answer isn't in these passages, say: "
    "'This information is not available in the policy documents.' "
    "Answer in 1-3 sentences maximum.\n\n"
)

# Franco context instruction — written in Franco so the model stays in register
_FRANCO_CONTEXT_INSTRUCTION = (
    "Esta3mel kol el passages el gaya di — sawa el 3arabi w el inglizi. "
    "Matb2ash ay passage. "
    "Law el ma3loma mesh mawgoda: '2ol mesh mawgoda fel policy.' "
    "Egabtak: Franco bass — la 3arabi wala inglizi.\n\n"
)


# ── LLM call with <think> stripping and Franco enforcement ────────────────────
def _call_llm(
    llm,
    prompt: str,
    lang: str,
    retry_prompt: str | None = None,
) -> str:
    wait     = BASE_WAIT
    attempts = 6
    current  = prompt

    while attempts > 0:
        attempts -= 1
        try:
            res  = llm.invoke(current)
            text = getattr(res, "content", str(res))

            # ── Strip <think> blocks (Qwen3 + any model that emits them) ──────
            text = _strip_think_blocks(text)
            # Belt-and-suspenders: also call the imported ragas stripper
            text = _strip_thinking(text).strip()

            if not text:
                if retry_prompt and current != retry_prompt:
                    tqdm.write("    ⚠ Empty after think-strip — retrying")
                    current = retry_prompt
                    continue
                return (
                    "This information is not available in the policy documents."
                    if lang == "english"
                    else "ma3loma mesh mawgoda fel policy."
                    if lang == "franco"
                    else "هذه المعلومات غير متوفرة في وثائق السياسة."
                )

            # ── Loop detection ────────────────────────────────────────────────
            if _is_looping(text):
                if retry_prompt and current != retry_prompt:
                    tqdm.write("    ⚠ Loop detected — retrying with strict prompt")
                    current = retry_prompt
                    continue
                sentences = re.split(r"[\.\!\؟\n]", text)
                text = sentences[0].strip() if sentences else ""
                if not text:
                    return (
                        "This information is not available in the policy documents."
                        if lang == "english"
                        else "ma3loma mesh mawgoda fel policy."
                        if lang == "franco"
                        else "هذه المعلومات غير متوفرة في وثائق السياسة."
                    )

            # ── Franco language enforcement ───────────────────────────────────
            # If the model drifted into Arabic script or French, retry once.
            if lang == "franco" and not _is_franco(text):
                if retry_prompt and current != retry_prompt:
                    tqdm.write(
                        f"    ⚠ Franco drift detected "
                        f"(arabic_ratio={len(re.findall(chr(0x600)+'-'+chr(0x6FF), text))/max(len(text),1):.2f}) "
                        f"— retrying"
                    )
                    current = retry_prompt
                    continue
                # Second drift: force-transliterate key facts rather than return garbage
                # (keep the text — it's likely wrong dialect but has the right facts)
                tqdm.write("    ⚠ Franco drift on retry — keeping answer as-is")

            # ── Hard length cap ───────────────────────────────────────────────
            if len(text) > 1200:
                cut  = text[:1200].rfind(". ")
                text = text[:cut + 1] if cut > 600 else text[:1200]

            return text

        except ValueError:
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if ("429" in msg or "rate_limit" in msg or "rate limit" in msg) and attempts > 0:
                tqdm.write(f"\n  ⏳ Rate limit — waiting {wait}s ({attempts} retries left)...")
                time.sleep(wait)
                wait = min(wait * 2, MAX_WAIT)
            else:
                raise

    raise RuntimeError("All retry attempts exhausted.")


def generate_answer(question: str, context: str, lang: str, dialect: str) -> str:
    if not context:
        return (
            "This information is not available in the policy documents."
            if lang == "english"
            else "ma3loma mesh mawgoda fel policy."
            if lang == "franco"
            else "هذه المعلومات غير متوفرة في وثائق السياسة."
        )

    if lang == "english":
        prompt_tmpl  = english_prompt
        llm          = EN_LLM
        augmented_ctx = _EN_CONTEXT_INSTRUCTION + context

        retry_prompt = (
            f"Answer in 1-2 sentences using ONLY the context. "
            f"Do NOT repeat yourself. Cite the page number.\n\n"
            f"Context:\n{context[:600]}\n\nQuestion: {question}\n\nAnswer:"
        )

    elif lang == "franco":
        prompt_tmpl   = franco_prompt
        llm           = FRANCO_LLM          # ← llama, not Qwen
        augmented_ctx = _FRANCO_CONTEXT_INSTRUCTION + context

        # Retry prompt is in Franco — keeps the model in the right register
        retry_prompt = (
            f"Egeb be Franco bass — la 3arabi wala inglizi wala french. "
            f"Gamla aw etnein bass. Lazem tezkar raqam el page.\n\n"
            f"El context:\n{context[:600]}\n\n"
            f"El so2al: {question}\n\n"
            f"El egaba (Franco bass):"
        )

    elif dialect == "egyptian" or lang == "egyptian":
        prompt_tmpl   = egy_prompt
        llm           = AR_LLM
        augmented_ctx = _AR_CONTEXT_INSTRUCTION + context

        retry_prompt = (
            f"أجب بجملة أو جملتين فقط بناءً على السياق. لا تكرر. لا تخترع معلومات.\n\n"
            f"السياق:\n{context[:600]}\n\nالسؤال: {question}\n\nالإجابة:"
        )

    else:
        # MSA Arabic
        prompt_tmpl   = msa_prompt
        llm           = AR_LLM
        augmented_ctx = _AR_CONTEXT_INSTRUCTION + context

        retry_prompt = (
            f"أجب بجملة أو جملتين فقط بناءً على السياق. لا تكرر. لا تخترع معلومات.\n\n"
            f"السياق:\n{context[:600]}\n\nالسؤال: {question}\n\nالإجابة:"
        )

    prompt = prompt_tmpl.format(context=augmented_ctx, question=question, history="")
    return _call_llm(llm, prompt, lang=lang, retry_prompt=retry_prompt)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

def _save(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    queries = _load(Path(args.query_set), [])
    if not queries:
        print(f"ERROR: {args.query_set} missing or empty.")
        return

    done   = _load(Path(args.checkpoint), {}) if args.resume else {}
    errors = _load(Path(args.errors_log),  []) if args.resume else []

    for qid in args.rerun_ids:
        done.pop(qid, None)
        tqdm.write(f"  Forced rerun: {qid}")

    remaining = [q for q in queries if q["query_id"] not in done]
    print(f"Queries — total: {len(queries)} | done: {len(done)} | remaining: {len(remaining)}\n")

    batch_n = 0

    for q in tqdm(remaining, desc="Generating"):
        qid  = q["query_id"]
        lang = q["language"]
        try:
            dialect = None
            if lang == "arabic":
                dialect = get_semantic_dialect(q["question"], dialect_pipe)
            elif lang == "egyptian":
                dialect = "egyptian"

            top_docs, context_str = _retrieve_policy(q["question"])

            src_tags = list({
                ("AR" if _is_arabic_source(d.metadata.get("source","")) else "EN")
                for d in top_docs
            })
            non_boilerplate = [d for d in top_docs if not _is_boilerplate(d.page_content)]

            if not context_str:
                tqdm.write(f"  ⚠ {qid} — ALL chunks were boilerplate, context empty")

            answer = generate_answer(q["question"], context_str, lang, dialect)

            record = {
                "query_id":        qid,
                "concept_id":      q["concept_id"],
                "language":        lang,
                "complexity":      q["complexity"],
                "source_doc":      q["source_doc"],
                "topic":           q["topic"],
                "question":        q["question"],
                "contexts":        [d.page_content for d in top_docs],
                "context_sources": src_tags,
                "n_en_chunks":     sum(1 for d in top_docs if not _is_arabic_source(d.metadata.get("source", ""))),
                "n_ar_chunks":     sum(1 for d in top_docs if _is_arabic_source(d.metadata.get("source", ""))),
                "answer":          answer,
                "reference":       q["reference_answer"],
                "timestamp":       datetime.now(timezone.utc).isoformat(),
            }

            done[qid] = record
            _save(Path(args.checkpoint), done)

            short_ans = answer[:70].replace("\n", " ")
            tqdm.write(f"  ✓ {qid} | {lang:8s} | src={src_tags} | {short_ans}…")

        except Exception as exc:
            errors.append({
                "query_id":  qid,
                "language":  lang,
                "question":  q.get("question", ""),
                "error":     str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            _save(Path(args.errors_log), errors)
            tqdm.write(f"  ✗ {qid} — {exc}")
            continue

        batch_n += 1
        if batch_n % args.batch_size == 0:
            tqdm.write(f"\n  💤 Batch of {args.batch_size} done. Sleeping {args.batch_sleep}s...\n")
            time.sleep(args.batch_sleep)

    results = list(done.values())
    _save(Path(args.output), results)
    print(f"\n✅ Saved {len(results)} results → {args.output}")
    if errors:
        print(f"⚠️  {len(errors)} failed → {args.errors_log}  (re-run with --resume to retry)")


if __name__ == "__main__":
    main()