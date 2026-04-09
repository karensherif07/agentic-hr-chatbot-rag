import json
from langchain_core.prompts import PromptTemplate


def format_personal_data(data: dict) -> str:
    """
    Converts personal data dict to a compact readable block.
    Only emits sections that have actual data — no empty headers.
    Skips zero-value leave types to reduce noise.
    """
    lines = []

    profile = data.get("profile", {})
    if profile:
        lines.append("PROFILE")
        lines.append(f"Name: {profile.get('full_name')} | Grade: {profile.get('grade')} | Title: {profile.get('job_title')}")
        lines.append(f"Dept: {profile.get('department')} | Manager: {profile.get('manager_name', 'N/A')}")
        lines.append(f"Hire date: {profile.get('hire_date')} | Type: {profile.get('employment_type')} | Work model: {profile.get('work_model')}")
        # Explicit probation status — never let the LLM infer from other fields
        if profile.get("probation_end_date"):
            lines.append(f"Probation status: ACTIVE — ends {profile['probation_end_date']}")
        else:
            lines.append("Probation status: NOT IN PROBATION")
        # Working hours note (policy-based, injected here so personal answers can state it)
        wm = profile.get("work_model", "")
        if wm == "remote":
            lines.append("Working hours note: Standard corporate hours 9:00 AM – 5:00 PM apply. Remote allowance 800 EGP/month.")
        elif wm == "hybrid":
            lines.append("Working hours note: Standard corporate hours 9:00 AM – 5:00 PM. 3 days in-office, 2 remote.")
        else:
            lines.append("Working hours note: Standard corporate hours 9:00 AM – 5:00 PM, 5 days in-office.")
        lines.append("")

    leave_balances = data.get("leave_balances", [])
    non_zero = [lb for lb in leave_balances if float(lb.get("remaining_days", 0)) > 0 or float(lb.get("used_days", 0)) > 0]
    if non_zero:
        lines.append("LEAVE BALANCES (current year)")
        for lb in non_zero:
            lines.append(
                f"  {lb['leave_type'].upper()}: entitled={lb['entitled_days']} "
                f"carried={lb['carried_over_days']} used={lb['used_days']} "
                f"pending={lb['pending_days']} remaining={lb['remaining_days']}"
            )
        lines.append("")

    pending = data.get("pending_leaves", [])
    if pending:
        lines.append("PENDING LEAVE REQUESTS")
        for p in pending:
            lines.append(f"  {p['leave_type']} {p['start_date']}→{p['end_date']} ({p['days_count']} days)")
        lines.append("")

    latest = data.get("latest_review")
    if latest:
        lines.append("LATEST PERFORMANCE REVIEW")
        lines.append(f"  Period: {latest['review_period']} | Rating: {latest['rating']}/5 ({latest['rating_label']})")
        lines.append(f"  Salary increment: {latest['salary_increment_pct']}% | Bonus multiplier: {latest['bonus_multiplier']}x")
        if latest.get("strengths"):    lines.append(f"  Strengths: {latest['strengths']}")
        if latest.get("areas_for_growth"): lines.append(f"  Areas for growth: {latest['areas_for_growth']}")
        if latest.get("overall_comments"): lines.append(f"  Comments: {latest['overall_comments']}")
        lines.append("")

    trend = data.get("performance_trend", {})
    if trend.get("ratings_oldest_first"):
        lines.append("PERFORMANCE TREND")
        for period, rating, label in zip(trend["review_periods"], trend["ratings_oldest_first"], trend["labels_oldest_first"]):
            lines.append(f"  {period}: {rating}/5 ({label})")
        lines.append(f"  Average: {trend['average_rating']} | Direction: {trend['trend_direction']}")
        lines.append("")

    okrs = data.get("current_okrs", [])
    if okrs:
        lines.append("CURRENT OKRs")
        for okr in okrs:
            lines.append(f"  [{okr['status'].upper()}] {okr['title']} — {okr['overall_progress_pct']}%")
            krs = okr.get("key_results", [])
            if isinstance(krs, str):
                krs = json.loads(krs)
            for kr in krs:
                lines.append(f"    • {kr.get('kr')} → {kr.get('progress_pct', 0)}% ({kr.get('status', '')})")
        lines.append("")

    salary = data.get("latest_salary")
    if salary:
        lines.append("LATEST SALARY")
        lines.append(f"  Base: {float(salary['base_salary']):,.0f} EGP | Gross: {float(salary['gross_salary']):,.0f} | Net: {float(salary['net_salary']):,.0f}")
        allowances = []
        for k, label in [("transport_allowance","Transport"),("housing_allowance","Housing"),("mobile_allowance","Mobile"),("remote_allowance","Remote")]:
            v = float(salary.get(k, 0))
            if v > 0:
                allowances.append(f"{label}={v:,.0f}")
        if allowances:
            lines.append(f"  Allowances: {' | '.join(allowances)}")
        lines.append("")

    training = data.get("training")
    if training:
        lines.append("TRAINING (current year)")
        lines.append(f"  Budget: ${float(training['budget_total_usd']):,.0f} total | ${float(training['budget_used_usd']):,.0f} used | ${float(training['budget_remaining_usd']):,.0f} remaining")
        lines.append(f"  Days: {training['training_days_total']} total | {training['training_days_used']} used | {training['days_remaining']} remaining")
        courses = training.get("courses", [])
        if isinstance(courses, str):
            courses = json.loads(courses)
        for c in courses:
            cert = " [certified]" if c.get("certificate") else ""
            lines.append(f"  • {c['name']} ({c['provider']}, {c['date']}, ${c.get('cost_usd',0):,.0f}){cert}")
        lines.append("")

    disc = data.get("active_disciplinary", [])
    if disc:
        lines.append("ACTIVE DISCIPLINARY")
        for d in disc:
            lines.append(f"  {d['action_type']} issued {d['issued_date']} expires {d.get('expiry_date','N/A')}: {d['reason']}")
        lines.append("")

    return "\n".join(lines).strip()


# ── Shared personal rules (compact, per language) ─────────────
_PERSONAL_RULES_EN = (
    "RULES:\n"
    "1. Answer ONLY from the personal data block below. Never guess.\n"
    "2. State exact numbers — days, amounts, ratings.\n"
    "3. PROBATION: Use the 'Probation status' field. Never infer probation from OKR grace periods.\n"
    "4. WORKING HOURS: Use the 'Working hours note' field in the profile.\n"
    "5. If a field is missing, say: 'This information is not available in your record.'\n"
    "6. Mirror pronouns: I/my → you/your.\n"
    "7. No page citations — this data is from your live record.\n"
    "8. LANGUAGE LOCK: Reply in English only.\n"
)
_PERSONAL_RULES_AR = (
    "القواعد:\n"
    "1. أجب من بيانات الموظف فقط. لا تخمّن.\n"
    "2. اذكر الأرقام بدقة — أيام، مبالغ، تقييمات.\n"
    "3. التجربة: استخدم حقل 'Probation status'. لا تستنتج الوضع من فترات OKR.\n"
    "4. ساعات العمل: استخدم حقل 'Working hours note' في الملف الشخصي.\n"
    "5. إذا غاب حقل: 'هذه المعلومة غير متوفرة في سجلك.'\n"
    "6. طابق الضمائر: أنا/لي → أنت/لك.\n"
    "7. لا أرقام صفحات — البيانات من السجل الحي.\n"
    "8. قفل اللغة: أجب بالعربية فقط.\n"
)
_PERSONAL_RULES_EGY = (
    "القواعد:\n"
    "1. جاوب من البيانات دي بس. متخمنش.\n"
    "2. اذكر الأرقام بدقة.\n"
    "3. التجربة: استخدم حقل 'Probation status'. متستنتجش من OKR.\n"
    "4. ساعات الشغل: استخدم حقل 'Working hours note'.\n"
    "5. لو المعلومة ناقصة: 'المعلومة دي مش متاحة في سجلك.'\n"
    "6. الموظف 'أنا/بتاعي' → إنت 'إنت/بتاعك'.\n"
    "7. متستخدمش أرقام صفحات.\n"
    "8. قفل اللغة: أجب بالعامية المصرية فقط. لا فصحى.\n"
)
_PERSONAL_RULES_FRANCO = (
    # English meta so LLM reliably follows it
    "IMPORTANT: Reply EXCLUSIVELY in Franco Arabic — Egyptian Arabic written in Latin script "
    "with numbers for Arabic letters (3=ع, 7=ح, 2=ء, 5=خ). "
    "Do NOT write English sentences. Do NOT write Modern Standard Arabic.\n"
    "RULES:\n"
    "1. Egib men el bayanat el shakhsiyya bass. Matkhminsh.\n"
    "2. 2ol el arqam beld2a — ayam, mablag, taqyim.\n"
    "3. Probation: esta5dem 7aql 'Probation status' bass. Matkstantij4sh men OKR.\n"
    "4. Mawa3id el shoghl: esta5dem 7aql 'Working hours note'.\n"
    "5. Lw el ma3loma na2sa: '(Da) mesh mawgod f segelak.'\n"
    "6. Ana/bta3i → enta/bta3ak.\n"
    "7. Mafish arqam sa7fat — el bayanat men el segell el 7ay.\n"
)

# ── Personal prompts ───────────────────────────────────────────
PERSONAL_EN = PromptTemplate(
    template=(
        "You are a personal HR assistant for Horizon Tech.\n"
        + _PERSONAL_RULES_EN
        + "\nRecent conversation:\n{history}\n\n"
        "=== EMPLOYEE DATA ===\n{personal_data}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "question", "history"]
)
PERSONAL_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي في شركة أفق التقنية.\n"
        + _PERSONAL_RULES_AR
        + "\nالمحادثة الأخيرة:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)
PERSONAL_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي في أفق التقنية.\n"
        + _PERSONAL_RULES_EGY
        + "\nالمحادثة اللي فاتت:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)
PERSONAL_FRANCO = PromptTemplate(
    template=(
        "You are a personal HR assistant for Horizon Tech.\n"
        + _PERSONAL_RULES_FRANCO
        + "\nEl kalam el fat:\n{history}\n\n"
        "=== El bayanat el shakhsiyya ===\n{personal_data}\n\n"
        "El so2al: {question}\nEl egaba (Franco bass):"
    ),
    input_variables=["personal_data", "question", "history"]
)

# ── Hybrid rules (compact) ─────────────────────────────────────
_HYBRID_RULES_EN = (
    "RULES:\n"
    "1. State the employee's actual data FIRST (balance, rating, etc.).\n"
    "2. Then state the relevant policy rule with [Page N | AR/EN] citations.\n"
    "3. Combine both to give a direct, actionable answer.\n"
    "4. No page citations for personal data.\n"
    "5. LANGUAGE LOCK: Reply in English only.\n"
)
_HYBRID_RULES_AR = (
    "القواعد:\n"
    "1. اذكر بيانات الموظف الفعلية أولاً.\n"
    "2. ثم اذكر القاعدة من السياسة مع [Page N | AR/EN].\n"
    "3. ادمجهما في إجابة عملية مباشرة.\n"
    "4. لا أرقام صفحات للبيانات الشخصية.\n"
    "5. قفل اللغة: أجب بالعربية فقط.\n"
)
_HYBRID_RULES_EGY = (
    "القواعد:\n"
    "1. اذكر بيانات الموظف الأول.\n"
    "2. بعدين القاعدة من السياسة مع [Page N | AR/EN].\n"
    "3. ادمجهم في إجابة بالعامية المصرية.\n"
    "4. لا أرقام صفحات للبيانات الشخصية.\n"
    "5. قفل اللغة: أجب بالعامية المصرية فقط. لا فصحى.\n"
)
_HYBRID_RULES_FRANCO = (
    "IMPORTANT: Reply EXCLUSIVELY in Franco Arabic — Egyptian Arabic in Latin script "
    "with number substitutions (3=ع, 7=ح, 2=ء, 5=خ). No English sentences, no فصحى.\n"
    "RULES:\n"
    "1. 2ol bayanat el mowazaf awwel (balance, taqyim, etc.).\n"
    "2. Ba3den el rule men el policy ma3 [Page N | AR/EN].\n"
    "3. Edmezhom f egaba wa7da — sara7a w 3amaleya.\n"
    "4. Mafish cite lel bayanat el shakhsiyya.\n"
)

HYBRID_EN = PromptTemplate(
    template=(
        "You are an HR assistant for Horizon Tech.\n"
        + _HYBRID_RULES_EN
        + "\nRecent conversation:\n{history}\n\n"
        "=== EMPLOYEE DATA ===\n{personal_data}\n\n"
        "=== POLICY CONTEXT ===\n{policy_context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)
HYBRID_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية في شركة أفق التقنية.\n"
        + _HYBRID_RULES_AR
        + "\nالمحادثة الأخيرة:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)
HYBRID_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية في أفق التقنية.\n"
        + _HYBRID_RULES_EGY
        + "\nالمحادثة اللي فاتت:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)
HYBRID_FRANCO = PromptTemplate(
    template=(
        "You are an HR assistant for Horizon Tech.\n"
        + _HYBRID_RULES_FRANCO
        + "\nEl kalam el fat:\n{history}\n\n"
        "=== El bayanat el shakhsiyya ===\n{personal_data}\n\n"
        "=== El policy context ===\n{policy_context}\n\n"
        "El so2al: {question}\nEl egaba (Franco bass):"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

def get_personal_prompt(lang: str, dialect: str = None):
    if lang == "english": return PERSONAL_EN
    if lang == "franco":  return PERSONAL_FRANCO
    if dialect == "egyptian": return PERSONAL_EGY
    return PERSONAL_AR

def get_hybrid_prompt(lang: str, dialect: str = None):
    if lang == "english": return HYBRID_EN
    if lang == "franco":  return HYBRID_FRANCO
    if dialect == "egyptian": return HYBRID_EGY
    return HYBRID_AR