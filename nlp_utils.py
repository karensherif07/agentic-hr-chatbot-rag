import re

# ── Franco tier-1: high-confidence Egyptian Arabic words written in Latin ──
# Expanded with HR-domain terms commonly used in Franco chat.
FRANCO_TIER1 = {
    # Pronouns & discourse
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma",
    "wenta", "wenti", "bs", "bas", "ad", "2ad",
    # Negation / affirmation
    "msh", "mesh", "mish", "mafish", "la2", "aywa", "aiwa",
    # Question words
    "leh", "leih", "fein", "fen", "emta", "ezay", "meen", "eih", "eh",
    # Common connectors / particles
    "ya3ni", "3ashan", "momken", "tayeb", "tamam", "keda", "kidda",
    "feeh", "fieh", "7aga", "haga", "delwa2ty", "badein", "ba3dein",
    "el", "di", "da", "dol", "aho", "ahi",
    # HR / workplace Franco terms (very common in Egyptian office chat)
    "3andi", "3andak", "3andik", "3andena",          # I have / you have
    "agaza", "egazti", "egazat", "egaza",             # leave / vacation
    "raseed", "raseed_agaza",                          # balance
    "shoghl", "shoghli",                               # work
    "mawa3id", "maw3id",                               # schedule / timing
    "sa3a", "sa3at",                                   # hour / hours
    "rateb", "ratbi",                                  # salary
    "bonus", "bta3i", "bta3ak", "bta3ti", "bta3na",  # possessive particle
    "mashy", "tamam_keda",                             # ok / alright
    "ta3ala", "yalla",                                 # come on / let's go
    "talab", "talabat",                                # request(s)
    "mawgood", "mawgooda",                             # available / exists
    "segelak", "segelty",                              # your record
}

FRANCO_MAP = {"2": "ء", "3": "ع", "4": "ش", "5": "خ", "7": "ح", "8": "غ"}

FRANCO_WORDS = {
    "3ayz": "عايز", "3ayza": "عايزة", "a3raf": "اعرف", "ezay": "ازاي",
    "fein": "فين", "law": "لو", "2ad": "قد", "leh": "ليه", "leih": "ليه",
    "msh": "مش", "mesh": "مش", "mish": "مش", "ana": "انا",
    "enta": "انت", "enti": "انتي", "el": "ال", "ya3ni": "يعني",
    "3ashan": "عشان", "tayeb": "طيب", "tamam": "تمام", "keda": "كده",
    "kidda": "كده", "bas": "بس", "bs": "بس", "la2": "لأ",
    "aywa": "ايوه", "aiwa": "ايوه", "momken": "ممكن", "7aga": "حاجة",
    "haga": "حاجة", "emta": "امتى", "meen": "مين", "eih": "ايه",
    "eh": "ايه", "fen": "فين", "mafish": "مافيش", "yenfa3": "ينفع",
    "ynfa3": "ينفع", "feeh": "فيه", "fieh": "فيه",
    "delwa2ty": "دلوقتي", "badein": "بعدين", "b3dein": "بعدين",
    "da": "ده", "di": "دي", "dol": "دول", "aho": "اهو", "ahi": "اهي",
    "ehna": "احنا", "ento": "انتوا", "howa": "هو", "hya": "هي",
    "homma": "هما", "egaza": "اجازة", "gawaz": "جواز", "a5od": "اخد",
    "a3mel": "اعمل", "egazah": "اجازة", "egazt": "اجازة",
    "3ayez": "عايز",
    # HR-domain additions
    "3andi": "عندي", "3andak": "عندك", "3andik": "عندك", "3andena": "عندنا",
    "agaza": "اجازة", "egazati": "اجازتي", "raseed": "رصيد",
    "shoghl": "شغل", "shoghli": "شغلي", "mawa3id": "مواعيد",
    "sa3a": "ساعة", "sa3at": "ساعات", "rateb": "راتب", "ratbi": "راتبي",
    "bta3i": "بتاعي", "bta3ak": "بتاعك", "bta3ti": "بتاعتي",
    "bta3na": "بتاعنا", "talab": "طلب", "talabat": "طلبات",
    "mashy": "ماشي", "yalla": "يلا", "ta3ala": "تعالى",
    "mawgood": "موجود", "mawgooda": "موجودة",
}

ENGLISH_STOP_WORDS = {"the", "is", "are", "what", "how", "who", "where", "of", "and", "to", "for"}

EGY_MARKERS = {
    "مش", "عايز", "عايزة", "فين", "إيه", "ايه", "ده", "دي", "احنا", "إحنا",
    "عشان", "بتاع", "دلوقتي", "هو", "ليه", "ازاي", "كده", "لسه", "برضه", "كمان"
}


def detect_language_type(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    token_set = set(tokens)

    if token_set & ENGLISH_STOP_WORDS:
        return "english"

    if token_set & FRANCO_TIER1:
        return "franco"

    # Franco digit-in-word heuristic (e.g. 3ayz, 7aga, 2ad)
    franco_hits = sum(
        1 for tok in tokens
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok)
    )
    if franco_hits >= 1:
        return "franco"

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


# ─── Franco → Arabic Conversion ───────────────────────────────

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