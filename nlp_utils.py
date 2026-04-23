import re

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
    "okrs": "أهداف",
}

# ── English stop words — enough coverage to distinguish English from Franco ──
# Threshold: >= 2 hits → English. This catches normal English sentences
# even if they happen to contain digit-like characters.
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
    # Arabic script → arabic
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    token_set = set(tokens)

    # >= 2 English stop words → English (catches "Can I get a bonus if I am on a PIP?")
    en_hits = token_set & ENGLISH_STOP_WORDS
    if len(en_hits) >= 2:
        return "english"

    # Franco tier-1 vocabulary hit
    if token_set & FRANCO_TIER1:
        return "franco"

    # Franco digit-in-word heuristic (e.g. 3ayz, 7aga, 2ad)
    franco_hits = sum(
        1 for tok in tokens
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok)
    )
    if franco_hits >= 1:
        return "franco"

    # Single English stop word
    if en_hits:
        return "english"

    return "english"


def get_semantic_dialect(text: str, dialect_pipe) -> str:
    if not isinstance(text, str):
        return "msa"

    tokens = set(re.findall(r"[\u0600-\u06FF]+", text))
    if tokens & EGY_MARKERS:
        return "egyptian"

    try:
        res = dialect_pipe(text)[0]
        label = res['label'].upper()
        if any(k in label for k in ("EGY", "EGYPT", "CAI", "DIAL", "DA")):
            return "egyptian"
    except Exception:
        pass

    return "msa"


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


def tokenize(text: str) -> list:
    text = re.sub(r"[\"']", "", text)
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())