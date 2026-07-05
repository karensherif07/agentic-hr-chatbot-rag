import os
import re

from langchain_groq import ChatGroq

# ── Franco tier-1: high-confidence Egyptian Arabic words written in Latin ──
FRANCO_TIER1 = {
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma",
    "wenta", "wenti", "bs", "bas", "ad", "2ad",
    "msh", "mesh", "mish", "mafish", "la2", "aywa", "aiwa",
    "leh", "leih", "fein", "fen", "emta", "ezay", "meen", "eih", "eh",
    "ya3ni", "3ashan", "momken", "tayeb", "tamam", "keda", "kidda",
    "feeh", "fieh", "7aga", "haga", "delwa2ty", "badein", "ba3dein",
    "el", "di", "da", "dol", "aho", "ahi",
    "3andi", "3andak", "3andik", "3andena",
    "agaza", "egazti", "egazat", "egaza",
    "raseed",
    "shoghl", "shoghli",
    "mawa3id", "maw3id",
    "sa3a", "sa3at",
    "rateb", "ratbi",
    "bonus", "bta3i", "bta3ak", "bta3ti", "bta3na",
    "mashy", "ta3ala", "yalla",
    "talab", "talabat",
    "mawgood", "mawgooda",
    "segelak", "segelty",
    "lw", "law",
    "ayh", "kamet", "kamt",
    "hakhod", "ha5od",
    "lazem",
    "2olly", "2ol",
    "walla", "wala",
    "inzar", "okrs",
    "lesa", "lessa",
    "wla", "wlla",
    "3amela", "3amel",
    "byiji", "byigi",
    # New additions
    "sa7afi", "rager3", "abawa", "mol7a2", "mawlood", "ganeby",
    "taqyemat", "ada2", "ekhtebar", "ehtefaz", "bayanat",
    "masroufat", "tasweyyet", "eddekhar", "taw3i",
}

FRANCO_MAP = {"2": "ء", "3": "ع", "4": "ش", "5": "خ", "7": "ح", "8": "غ"}

FRANCO_WORDS = {
    "3ayz": "عايز", "3ayza": "عايزة", "a3raf": "اعرف", "ezay": "ازاي",
    "fein": "فين", "law": "لو", "lw": "لو", "2ad": "قد",
    "leh": "ليه", "leih": "ليه", "msh": "مش", "mesh": "مش", "mish": "مش",
    "ana": "انا", "enta": "انت", "enti": "انتي", "el": "ال",
    "ya3ni": "يعني", "3ashan": "عشان", "tayeb": "طيب", "tamam": "تمام",
    "keda": "كده", "kidda": "كده", "bas": "بس", "bs": "بس",
    "la2": "لأ", "aywa": "ايوه", "aiwa": "ايوه", "momken": "ممكن",
    "7aga": "حاجة", "haga": "حاجة", "emta": "امتى", "meen": "مين",
    "eih": "ايه", "eh": "ايه", "fen": "فين", "mafish": "مافيش",
    "yenfa3": "ينفع", "ynfa3": "ينفع", "feeh": "فيه", "fieh": "فيه",
    "delwa2ty": "دلوقتي", "badein": "بعدين", "b3dein": "بعدين",
    "da": "ده", "di": "دي", "dol": "دول", "aho": "اهو", "ahi": "اهي",
    "ehna": "احنا", "ento": "انتوا", "howa": "هو", "hya": "هي",
    "homma": "هما", "egaza": "اجازة", "gawaz": "جواز",
    "a5od": "اخد", "a3mel": "اعمل", "egazah": "اجازة", "egazt": "اجازة",
    "3ayez": "عايز",
    "3andi": "عندي", "3andak": "عندك", "3andik": "عندك", "3andena": "عندنا",
    "agaza": "اجازة", "egazati": "اجازتي", "raseed": "رصيد",
    "shoghl": "شغل", "shoghli": "شغلي", "mawa3id": "مواعيد",
    "sa3a": "ساعة", "sa3at": "ساعات", "rateb": "راتب", "ratbi": "راتبي",
    "bta3i": "بتاعي", "bta3ak": "بتاعك", "bta3ti": "بتاعتي",
    "bta3na": "بتاعنا", "talab": "طلب", "talabat": "طلبات",
    "mashy": "ماشي", "yalla": "يلا", "ta3ala": "تعالى",
    "mawgood": "موجود", "mawgooda": "موجودة",
    "hakhod": "هاخد", "ha5od": "هاخد",
    "lazem": "لازم", "kamet": "كام", "kamt": "كام",
    "walla": "ولا", "wala": "ولا",
    "inzar": "إنذار", "2ol": "قول", "2olly": "قولي",
    "ayh": "ايه",
    "ashtaghalt": "اشتغلت", "5aragt": "خرجت", "5arabt": "خرجت",
    "ashtaghal": "اشتغل", "beshtaghal": "بيشتغل",
    "fasl": "فصل", "fawry": "فوري", "ye2ady": "يؤدي",
    "ely": "اللي", "yewdi": "يودي",
    "lesa": "لسه", "lessa": "لسه",
    "wla": "ولا", "probation": "فترة التجربة",
    "3amela": "عاملة", "3amel": "عامل",
    "byiji": "بيجي", "byigi": "بيجي",
    "segelak": "سجلك", "segelti": "سجلتي",
    "okrs": "أهداف", "m4": "مش",
    "3aiz": "عايز", "3ayz": "عايز", "3ayez": "عايز", "3ayza": "عايزة",
    "2ywa": "ايوه", "ah": "اه", "aha": "اه",
    "fe": "في", "fi": "في", "fy": "في",
    "kolo": "كله", "kollo": "كله",
    "y3ni": "يعني", "3shan": "عشان",
    # HR/policy terms
    "ta2min": "تأمين",
    "ta2men": "تأمين",
    "se77i": "صحي",
    "se7i": "صحي",
    "overtime": "عمل إضافي",
    "bonus": "مكافأة",
    "PIP": "خطة تحسين أداء",
    "ma3ash": "راتب",
    "badal": "بدل",
    "ta2meen": "تأمين",
    "nisblt": "نسبة", "taweed": "تعويض", "ayyam": "أيام", "3adeya": "عادية",
    "biyebda2": "يبدأ", "muwazaf": "موظف", "muwazafin": "موظفين",
    "gedid": "جديد", "bedl": "بدل", "sakan": "سكن",
    "men": "من", "bya5od": "يأخذ", "bya5odo": "يأخذون",
    "modet": "مدة", "esh3ar": "إشعار", "yenhi": "ينهي", "3a2d": "عقد",
    "fatret": "فترة", "ekhtebar": "اختبار", "2emet": "قيمة",
    "da3m": "دعم", "gym": "جيم", "biyedi": "يعطي",
    "7ayah": "حياة", "gama3y": "جماعي",
    "mizaneyyet": "ميزانية", "ta3allom": "تعلم", "sanaweyya": "سنوية",
    "mwaahed": "مرتب", "btetatref": "يتصرف", "yom": "يوم", "shahr": "شهر",
    "nesblet": "نسبة", "estered": "استرداد", "rasoom": "رسوم",
    "shahadet": "شهادة", "re3adet": "إعادة", "bettetghatta": "تتغطى",
    "3agz": "عجز", "taweel": "طويل", "amad": "أمد",
    "biyestamr": "يستمر", "yesta2red": "يقترض",
    "tare2": "طارئ", "fawa2ed": "فوائد", "sedad": "سداد",
    "ishtrak": "اشتراك", "ta2minat": "تأمينات",
    "egtema3eyya": "اجتماعية", "shorut": "شروط", "matluba": "مطلوبة",
    "indemam": "انضمام", "khtet": "خطة", "eddekhar": "ادخار",
    "taw3i": "طوعي", "wa7dat": "وحدات", "imtithal": "امتثال",
    "tadribeyya": "تدريبية", "elzameyya": "إلزامية", "mawa3eedha": "مواعيدها",
    "mokhalafat": "مخالفات", "gasima": "جسيمة", "btwassal": "توصل",
    "ehtefaz": "احتفاظ", "bayanat": "بيانات", "sagalat": "سجلات",
    "mokhtalefa": "مختلفة", "anwa3": "أنواع", "masroufat": "مصروفات",
    "btetrod": "تترد", "7ododha": "حدودها",
    "sa7afi": "صحفي", "ta3asal": "تواصل", "ma3aya": "معاي",
    "ta3li2": "تعليق", "a3mel eih": "أعمل إيه",
    "rager3": "راجع", "abawa": "أبوة", "a2dar": "أقدر",
    "a2al": "أقل", "kamil": "كامل", "ganeby": "جانبي",
    "2adeem": "قديم", "yetadakhel": "يتداخل", "a3mal": "أعمال",
    "mawlood": "مولود", "ashtarak": "أشترك",
    "mol7a2": "ملحق", "idafi": "إضافي", "maw3id": "موعد",
    # newly added
    "taqyemat": "تقييم", "taqyeem": "تقييم", "ada2": "أداء",
    "2ard": "قرض", "solfet": "سلفة", "solf": "سلفة",
    "don": "بدون", "fasl": "فصل",
    "rago3": "عودة", "tadregi": "تدريجي",
    "tadreg": "تدريج", "3awda": "عودة",
    "omuma": "أمومة",
    "ad eih": "كم",
    "byet7aseb": "يحسب", "dif3": "ضعف",
}

# ── Franco synonym expansion: maps franco query patterns → EN search terms ─────
# Used in _retrieve_policy to widen retrieval for weak franco_to_arabic output
FRANCO_EN_SYNONYMS = {
    r"imtithal.*tadrib|tadrib.*elzam|compliance.*training|mandatory.*training|wa7dat.*tadribeyya":
        "mandatory compliance training annual modules deadlines information security data protection",
    r"ta2min.*7ayah|7ayah.*gama3y|life.*insur":
        "group life insurance benefit annual salary lump sum",
    r"mol7a2.*ta2min|ta2min.*idafi|health.*add.?on|additional.*coverage|mawlood.*ta2min":
        "optional supplemental health insurance enrollment window qualifying event birth marriage 30 days",
    r"fatrat.*ehtefaz|retention.*period|data.*retention|ehtefaz.*bayanat":
        "data retention period employee records payroll performance disciplinary",
    r"masroufat.*maw3id|expense.*deadline|tasweyyet.*masrof|akher.*maw3id.*masrof":
        "expense claim submission deadline 30 days not accepted",
    # NEW additions
    r"sa7afi|journalist|media.*contact|ta3li2.*sharka|press.*enquiry":
        "journalist media enquiry forward communications team respond press communications@horizontech.com",
    r"rager3.*agaza|agaza.*abawa|agaza.*omuma|phased.*return|3awda.*tadreg|rago3.*tadregi":
        "phased return parental leave maternity minimum 60 percent hours full pay 8 weeks",
    r"taqyemat.*ada2.*ekhtebar|taqyeem.*ekhtebar|probation.*review|ada2.*fatret.*ekhtebar":
        "performance review probation first month third month evaluation timing",
    r"2ard.*tare2|emergency.*loan|solfet.*tare2a|solf.*don.*fawa2ed|interest.?free.*loan":
        "emergency interest-free loan two months net salary probation completed repaid installments",
    r"khtet.*eddekhar|savings.*plan|taw3i.*eddekhar|voluntary.*saving":
        "voluntary savings plan eligibility grade minimum service matching contribution enrollment",
}

# HR-Specific Mapping: Normalizes colloquial Egyptian HR terms to formal MSA
EGY_TO_MSA_WORDS = {
    "عايز": "أريد", "عايزة": "أريد", "عاوز": "أريد", "عاوزه": "أريد",
    "أجازة": "إجازة", "اجازه": "إجازة", "اجازة": "إجازة", "أجازتي": "إجازتي",
    "مرتب": "راتب", "المرتب": "الراتب", "بقبض": "أستلم راتب",
    "بياخد": "يأخذ", "هياخد": "سيأخذ",
    "ازاي": "كيف", "إزاي": "كيف", "فين": "أين",
    "امتى": "متى", "إمتى": "متى", "إيه": "ماذا", "ايه": "ماذا",
    "دلوقتي": "الآن", "مش": "ليس", "ينفع": "هل يمكن", "ممكن": "هل يمكن",
    "بتاعي": "الخاص بي", "بتاعتي": "الخاصة بي", "شغلي": "عملي",
    "برضه": "أيضاً", "كمان": "أيضاً", "لسه": "ما زال",
    "ده": "هذا", "دي": "هذه", "دول": "هؤلاء",
    "هو": "هو", "هي": "هي", "احنا": "نحن",
    "عشان": "لأن", "علشان": "لأن", "فيه": "يوجد", "مفيش": "لا يوجد",
    "ليه": "لماذا", "كده": "هكذا",
    "يلا": "هيا", "ماشي": "حسناً", "تمام": "حسناً",
    "شغل": "عمل", "شغلي": "عملي", "شغلك": "عملك",
    "راتبي": "راتبي", "بيشتغل": "يعمل", "اشتغل": "عمل", "اشتغلت": "عملت",
    "اجازتي": "إجازتي", "مواعيد": "مواعيد", "ساعة": "ساعة", "ساعات": "ساعات",
    "فترة": "فترة", "تجربة": "تجربة", "فترة التجربة": "فترة الاختبار",
    "عايز اعرف": "أريد أن أعرف", "عايزة اعرف": "أريد أن أعرف",
    "ممكن اعرف": "هل يمكنني معرفة", "عايز اسأل": "أريد أن أسأل",
    "عايزة اسأل": "أريد أن أسأل",
}

EGY_PHRASES = {
    "مش عارف": "لا أعلم",
    "مش فاهم": "لا أفهم",
    "عايز اعرف": "أريد أن أعرف",
    "عايزة اعرف": "أريد أن أعرف",
    "ممكن اعرف": "هل يمكنني معرفة",
    "فيه مشكلة": "هناك مشكلة",
    "مفيش مشكلة": "لا توجد مشكلة",
    "عايز اسأل": "أريد أن أسأل",
    "عايزة اسأل": "أريد أن أسأل",
    "ايه ده": "ما هذا",
    "ليه كده": "لماذا هكذا",
}

ENGLISH_STOP_WORDS = {
    "the", "is", "are", "what", "how", "who", "where", "of", "and",
    "to", "for", "can", "i", "if", "do", "does", "will", "my", "me",
    "on", "a", "an", "in", "at", "be", "get", "have", "has", "am",
    "not", "yes", "no", "was", "were", "it", "its", "this", "that",
    "with", "from", "or", "but", "so", "than", "then", "when", "which",
    "there", "their", "they", "we", "you", "he", "she", "about",
    "would", "could", "should", "may", "might", "must", "shall",
    "any", "all", "some", "more", "also", "too", "very", "just",
    "did", "been", "being", "had", "having", "by", "as", "up",
}

EGY_MARKERS = {
    "مش", "عايز", "عايزة", "فين", "إيه", "ايه", "ده", "دي", "احنا", "إحنا",
    "عشان", "بتاع", "دلوقتي", "هو", "ليه", "ازاي", "كده", "لسه", "برضه", "كمان",
    "بيشتغل", "هياخد", "بياخد", "بتاعتي", "بتاعي", "شغلي", "ولا", "إيه ده",
    "بتاعك", "شغلك", "مش عارف", "عايز أعرف", "ممكن",
}


def detect_language_type(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    token_set = set(tokens)
    en_hits = token_set & ENGLISH_STOP_WORDS
    if len(en_hits) >= 2:
        return "english"
    if token_set & FRANCO_TIER1:
        return "franco"
    franco_hits = sum(
        1 for tok in tokens
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok)
    )
    if franco_hits >= 1:
        return "franco"
    if en_hits:
        return "english"
    return "english"


def get_semantic_dialect(text: str, dialect_pipe) -> str:
    if not isinstance(text, str) or len(text) < 15:
        return "msa"
    tokens = set(re.findall(r"[\u0600-\u06FF]+", text))
    if tokens & EGY_MARKERS:
        return "egyptian"
    try:
        res = dialect_pipe(text)[0]
        if res['score'] < 0.75:
            return "msa"
        label = res['label'].upper()
        if any(k in label for k in ("EGY", "EGYPT", "CAI", "DIAL", "DA")):
            return "egyptian"
    except Exception:
        pass
    return "msa"


from functools import lru_cache
from deep_translator import GoogleTranslator


@lru_cache(maxsize=2000)
def egyptian_to_msa(query: str, llm=None) -> str:
    """
    Convert Egyptian Arabic (script) to Modern Standard Arabic.
    Uses Google Translate (ar→ar) with fallback to hardcoded rules.
    """
    if not query:
        return query

    text = query
    for k, v in EGY_PHRASES.items():
        text = re.sub(rf"\b{k}\b", v, text)
    for k, v in EGY_TO_MSA_WORDS.items():
        text = re.sub(rf"\b{k}\b", v, text)

    try:
        translator = GoogleTranslator(source='ar', target='ar')
        translated = translator.translate(text)
        if translated and len(translated) > 5:
            return translated
    except Exception:
        pass

    return text


def clean_pdf(text: str) -> str:
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u200e\u200f]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_arabic(text: str, ara_tokenizer) -> str:
    try:
        tokens = ara_tokenizer.tokenize(text)
        segmented = " ".join(tokens).replace(" ##", "")
    except Exception:
        segmented = text
    segmented = re.sub(r"[أإآ]", "ا", segmented)
    segmented = segmented.replace("ة", "ه").replace("ى", "ي")
    segmented = re.sub(r"[\u064B-\u065F]", "", segmented)
    return segmented.lower()


def normalize_english(text: str) -> str:
    return text.lower()


def franco_to_arabic(text: str) -> str:
    words = text.lower().split()
    converted = []
    for w in words:
        if w in FRANCO_WORDS:
            converted.append(FRANCO_WORDS[w])
        else:
            result = w
            for digit, arabic_char in FRANCO_MAP.items():
                result = result.replace(digit, arabic_char)
            converted.append(result)
    return " ".join(converted)


def apply_franco_en_synonyms(query: str) -> str:
    """
    Given a franco query, return extra English search terms based on
    FRANCO_EN_SYNONYMS patterns.  Returns empty string if no match.
    """
    q = query.lower()
    for pattern, expansion in FRANCO_EN_SYNONYMS.items():
        if re.search(pattern, q, re.IGNORECASE):
            return expansion
    return ""


def tokenize(text: str) -> list:
    text = re.sub(r"[\"']", "", text)
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())