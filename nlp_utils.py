import re

FRANCO_TIER1 = {
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma", "msh", "mesh", "mish", "mafish","leh", "leih", "fein", "fen", "emta", "ezay", 
    "meen", "eih", "eh","ya3ni", "3ashan", "momken", "3ayz", "3ayza", "yenfa3", "ynfa3","tayeb", "tamam", "keda", "kidda", "bas", "ad", "2ad",
    "7aga", "haga", "feeh", "fieh", "la2", "aywa", "aiwa","wenta", "wenti", "bs", "delwa2ty", "badein", "ba3dein", "el", "di", "da", "dol", "aho", "ahi",
}
FRANCO_MAP = {"2": "ء", "3": "ع", "4": "ش", "5": "خ", "7": "ح", "8": "غ"}

FRANCO_WORDS = {
    "3ayz": "عايز", "3ayza": "عايزة", "a3raf": "اعرف", "ezay": "ازاي", "fein": "فين", "law": "لو", "2ad": "قد",
    "leh": "ليه", "leih": "ليه", "msh": "مش", "mesh": "مش", "mish": "مش", "ana": "انا",
    "enta": "انت", "enti": "انتي", "el": "ال", "ya3ni": "يعني", "3ashan": "عشان",
    "tayeb": "طيب", "tamam": "تمام", "keda": "كده", "kidda": "كده", "bas": "بس", "bs": "بس",
    "la2": "لأ", "aywa": "ايوه", "aiwa": "ايوه", "momken": "ممكن", "7aga": "حاجة",
    "haga": "حاجة", "emta": "امتى", "meen": "مين", "eih": "ايه", "eh": "ايه", "fen": "فين",
    "mafish": "مافيش", "yenfa3": "ينفع", "ynfa3": "ينفع", "feeh": "فيه", "fieh": "فيه",
    "delwa2ty": "دلوقتي", "badein": "بعدين", "b3dein": "بعدين", "da": "ده", "di": "دي",
    "dol": "دول", "aho": "اهو", "ahi": "اهي", "ehna": "احنا", "ento": "انتوا",
    "howa": "هو", "hya": "هي", "homma": "هما",    "law": "لو", "egaza": "اجازة", "gawaz": "جواز", "a5od": "اخد", "a3mel": "اعمل",
    "eh": "ايه", "gawaz": "جواز", "egazah": "اجازة", "egazt": "اجازة", "3ayez": "عايز",}

ENGLISH_STOP_WORDS = {"the", "is", "are", "what", "how", "who", "where", "of", "and", "to", "for"}

EGY_MARKERS = {
    "مش", "عايز", "عايزة", "فين", "إيه", "ايه", "ده", "دي", "احنا", "إحنا", "عشان",
     "بتاع", "دلوقتي", "هو","ليه", "ازاي", "كده", "لسه", "برضه", "كمان"
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

    # Rule-based fast path — check known Egyptian markers first
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
    text = re.sub(r"[\"']", "", text)  # remove quotes
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())
