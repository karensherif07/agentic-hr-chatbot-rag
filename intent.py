import re

# ─────────────────────────────────────────────────────────────
# intent.py  —  personal | hybrid | policy classifier
# ─────────────────────────────────────────────────────────────
#
# KEY FIX: English personal patterns use \bmy\b.{0,20}\b<keyword>\b
# so "What is my current net salary?" still matches even though
# adjectives sit between "my" and the keyword.
# Old pattern r"my (salary|...)" required them to be adjacent.
# ─────────────────────────────────────────────────────────────

PERSONAL_PATTERNS = [

    # ── Leave ────────────────────────────────────────────────
    # English — bounded wildcard between "my" and keyword
    (r"\bmy\b.{0,25}\b(leave|vacation|days off|annual|sick|maternity|paternity|hajj|bereavement|absence|study leave)\b", "leave"),
    (r"\bmy\b.{0,25}\b(leave balance|remaining days|remaining leave)\b", "leave"),
    (r"(how many|how much).{0,30}\b(leave|days|vacation)\b.{0,30}\b(i have|left|remaining|got|balance|mine)\b", "leave"),
    (r"\b(my leave|leave balance|my balance|my remaining)\b", "leave"),
    # Arabic — first-person possessives only (رصيد إجازتي, not كم يوم)
    (r"(رصيد إجازتي|إجازتي السنوية|إجازتي|أيامي المتبقية|يتبقى لي|رصيدي)", "leave"),
    (r"كم يوم.{0,15}(لدي|عندي|متبقٍّ لي|تبقّى لي|لي)", "leave"),
    # Franco
    (r"(agazti|raseed bta3i|raseed agaza bta3i|kam yom 3andi|el balance bta3i|agazat bta3ti)", "leave"),
    (r"(pending leave bta3i|talab agaza bta3i|my leave request)", "leave"),

    # ── Performance ──────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(performance|rating|review|score|appraisal|evaluation)\b", "performance"),
    (r"(describe|summarise|show).{0,30}(my performance|my ratings|how i.ve been doing)", "performance"),
    (r"(تقييمي|أدائي|تقرير أدائي|مراجعتي الأخيرة|درجتي في التقييم)", "performance"),
    (r"(taqyimi|adai|el rating bta3i|el review bta3i|el taqyim bta3i)", "performance"),
    (r"(am i improving|my trend|my scores over time)", "performance"),

    # ── Salary ───────────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(salary|pay|payslip|payroll|income|net salary|gross salary)\b", "salary"),
    (r"\b(my allowance|my allowances|my compensation)\b", "salary"),
    (r"(راتبي|مرتبي|بدلاتي|صافي راتبي|إجمالي راتبي|راتبي الصافي|راتبي الشهري)", "salary"),
    (r"(ratbi|el rateb bta3i|flous bta3ti|salary bta3i|el net bta3i|el rateb el net)", "salary"),

    # ── Training ─────────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(training|training budget|courses|learning|development|certification)\b", "training"),
    (r"(how much|how many).{0,30}(training|budget|courses).{0,30}(left|remaining|used|spent|i have)", "training"),
    (r"(ميزانيتي التدريبية|دوراتي|رصيد تدريبي|ميزانية التدريب بتاعتي)", "training"),
    (r"(budget bta3i|training bta3i|courses bta3ti|el training bta3i)", "training"),

    # ── OKRs ─────────────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(okr|okrs|goal|goals|objective|target|kpi|key result)\b", "okr"),
    (r"(أهدافي|مؤشراتي|okr بتاعتي|أهدافي الحالية)", "okr"),
    (r"(okrs bta3ti|goals bta3ti|el okr bta3i|el okrs bta3ti)", "okr"),

    # ── Profile ──────────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(profile|grade|job grade|department|manager|hire date|role|job title|title|employment type|work model)\b", "profile"),
    (r"(when did i join|when did i start working)", "profile"),
    (r"(درجتي الوظيفية|قسمي|مديري|تاريخ التحاقي|نموذج عملي)", "profile"),
    (r"(grade bta3i|manager bta3i|my info|my details|el grade bta3i|el department bta3i)", "profile"),

    # ── Disciplinary ─────────────────────────────────────────
    (r"\bmy\b.{0,25}\b(warning|pip|disciplinary)\b", "disciplinary"),
    (r"(any action against me|am i on pip|do i have a warning)", "disciplinary"),
    (r"(إنذاراتي|هل عليّ إنذار|خطة تحسين أدائي|هل أنا على PIP)", "disciplinary"),
    (r"(el inzar bta3i|ana 3ala pip|3andi inzar 3alaya)", "disciplinary"),

    # ── Probation ────────────────────────────────────────────
    (r"\bmy\b.{0,25}\bprobation\b", "profile"),
    (r"(am i on probation|am i still on probation|probation status|is my probation over)", "profile"),
    (r"(فترة تجربتي|هل أنا في فترة التجربة|هل انتهت فترة تجربتي)", "profile"),
    # Franco — fixed: was "lesa fi el probation" but user writes "lesa fi probation"
    (r"(lesa fi probation|probation bta3i|ana fi probation|lessa fi probation|el probation bta3i)", "profile"),
    (r"(lesa|lessa).{0,10}probation", "profile"),
]

HYBRID_PATTERNS = [
    # Working hours — needs work_model from DB + shift policy
    r"(what are my working hours|what are my hours|my working hours|my work hours|hours do i work"
    r"|when do i (start|finish|clock in|work)|my shift|my schedule|my start time|my work schedule"
    r"|am i (remote|hybrid|office|on-?site)|work from home"
    r"|ساعات عملي|مواعيد عملي|عدد ساعات عملي|جدول عملي|ساعات شغلي|مواعيد شغلي"
    r"|mawa3id shoghl bta3i|working hours bta3i|kam sa3a bashtaghal|sa3at el shoghl bta3ti"
    r"|mawa3id shoghl bta3ti|sa3at shoghl bta3i)",

    # Leave eligibility — needs remaining balance from DB + policy entitlement
    # "Can I take MORE leave" or "Can I take ANY leave" → hybrid
    # Removed: "Can I take study leave" — that's a pure policy question
    r"can i (take|get|have|request|apply).{0,10}(more|additional|extra|another).{0,15}(leave|agaza)",
    r"can i (take|get).{0,10}(annual|sick|maternity|paternity|hajj|bereavement).{0,15}(leave|agaza)",
    r"(هل أقدر آخد|ممكن آخد|هل يمكنني أخذ).{0,20}(إجازة|أجازة)",
    r"(a2dar akhod|momken akhod).{0,20}agaza",

    # Eligibility questions that need DB check + policy
    r"(am i eligible|do i qualify).{0,30}(leave|bonus|scholarship|promotion|training|raise)",
    r"(هل يحق لي|هل أستحق|هل يمكنني الحصول على).{0,20}(إجازة|مكافأة|منحة|ترقية|علاوة)",
    r"(momken|ayezna3raf|a3raf).{0,20}(eligible|a5od|ahkel).{0,20}(leave|bonus|promotion|scholarship)",

    # Bonus eligibility — needs disciplinary/tenure from DB + policy
    r"(am i eligible for a bonus|do i get a bonus|will i get a bonus|هل سأحصل على مكافأة|momken akhod bonus)",

    # Promotion eligibility — needs rating/tenure from DB + policy criteria
    r"(am i eligible for.{0,10}promot|can i be promoted|هل يحق لي.{0,10}ترق|هل أستحق.{0,10}ترق|momken at2addam lel tar2eyya)",

    # How much more leave / remaining eligibility
    r"(how much more|كم يتبقى|remaining).{0,20}(allowed|policy|يسمح|entitl)",
    r"(can i still get|هل يمكنني الحصول على|momken a5od).{0,20}(bonus|scholarship|leave|promotion)",
]


def _match(text: str, patterns: list) -> tuple[bool, str | None]:
    t = text.lower()
    for entry in patterns:
        if isinstance(entry, tuple):
            pattern, topic = entry
            if re.search(pattern, t):
                return True, topic
        else:
            if re.search(entry, t):
                return True, None
    return False, None


def classify_intent(question: str) -> tuple[str, str | None]:
    """
    Returns (intent, topic).
    intent : "personal" | "hybrid" | "policy"
    topic  : "leave" | "performance" | "salary" | "training" |
             "okr" | "profile" | "disciplinary" | "all" | None
    """
    # Hybrid check first
    is_hybrid, _ = _match(question, HYBRID_PATTERNS)
    if is_hybrid:
        _, topic = _match(question, PERSONAL_PATTERNS)
        return "hybrid", (topic or "all")

    # Personal check
    is_personal, topic = _match(question, PERSONAL_PATTERNS)
    if is_personal:
        return "personal", (topic or "all")

    # Default: policy RAG
    return "policy", None