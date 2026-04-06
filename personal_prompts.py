"""
personal_prompts.py
Prompt templates for personal and hybrid queries.
These are separate from the RAG prompts in prompts.py.
"""

import json
from langchain_core.prompts import PromptTemplate


def format_personal_data(data: dict) -> str:
    """
    Converts the personal data dict into a clean readable block
    that the LLM can parse without confusion.
    """
    lines = []

    profile = data.get("profile", {})
    if profile:
        lines.append("=== EMPLOYEE PROFILE ===")
        lines.append(f"Name: {profile.get('full_name')} ({profile.get('full_name_ar', '')})")
        lines.append(f"Grade: {profile.get('grade')} | Title: {profile.get('job_title')}")
        lines.append(f"Department: {profile.get('department')}")
        lines.append(f"Manager: {profile.get('manager_name', 'N/A')}")
        lines.append(f"Hire date: {profile.get('hire_date')}")
        lines.append(f"Employment type: {profile.get('employment_type')}")
        lines.append(f"Work model: {profile.get('work_model')}")
        if profile.get("probation_end_date"):
            lines.append(f"Probation ends: {profile.get('probation_end_date')} (still in probation)")
        lines.append("")

    leave_balances = data.get("leave_balances", [])
    if leave_balances:
        lines.append("=== LEAVE BALANCES (current year) ===")
        for lb in leave_balances:
            lines.append(
                f"  {lb['leave_type'].upper()}: "
                f"Entitled={lb['entitled_days']} | "
                f"Carried over={lb['carried_over_days']} | "
                f"Used={lb['used_days']} | "
                f"Pending={lb['pending_days']} | "
                f"Remaining={lb['remaining_days']}"
            )
        lines.append("")

    pending = data.get("pending_leaves", [])
    if pending:
        lines.append("=== PENDING LEAVE REQUESTS ===")
        for p in pending:
            lines.append(f"  {p['leave_type']} — {p['start_date']} to {p['end_date']} ({p['days_count']} days) — submitted {p['requested_at'][:10]}")
        lines.append("")

    latest = data.get("latest_review")
    if latest:
        lines.append("=== LATEST PERFORMANCE REVIEW ===")
        lines.append(f"Period: {latest['review_period']} | Date: {latest['review_date']}")
        lines.append(f"Rating: {latest['rating']}/5 — {latest['rating_label']}")
        lines.append(f"Salary increment given: {latest['salary_increment_pct']}%")
        lines.append(f"Bonus multiplier: {latest['bonus_multiplier']}x")
        if latest.get("strengths"):
            lines.append(f"Strengths: {latest['strengths']}")
        if latest.get("areas_for_growth"):
            lines.append(f"Areas for growth: {latest['areas_for_growth']}")
        if latest.get("overall_comments"):
            lines.append(f"Overall: {latest['overall_comments']}")
        lines.append("")

    trend = data.get("performance_trend", {})
    if trend.get("ratings_oldest_first"):
        lines.append("=== PERFORMANCE TREND ===")
        pairs = zip(trend["review_periods"], trend["ratings_oldest_first"], trend["labels_oldest_first"])
        for period, rating, label in pairs:
            lines.append(f"  {period}: {rating}/5 ({label})")
        lines.append(f"Average rating: {trend['average_rating']} | Trend: {trend['trend_direction']}")
        lines.append("")

    okrs = data.get("current_okrs", [])
    if okrs:
        lines.append("=== CURRENT OKRs ===")
        for okr in okrs:
            lines.append(f"  [{okr['status'].upper()}] {okr['title']} — {okr['overall_progress_pct']}% complete")
            krs = okr.get("key_results", [])
            if isinstance(krs, str):
                krs = json.loads(krs)
            for kr in krs:
                lines.append(f"    • {kr.get('kr')} → {kr.get('progress_pct', 0)}% ({kr.get('status', '')})")
        lines.append("")

    salary = data.get("latest_salary")
    if salary:
        lines.append("=== LATEST SALARY (most recent month) ===")
        lines.append(f"Month: {salary['month']}")
        lines.append(f"Base: {salary['base_salary']:,.0f} EGP")
        lines.append(f"Transport: {salary['transport_allowance']:,.0f} | Housing: {salary['housing_allowance']:,.0f} | Mobile: {salary['mobile_allowance']:,.0f} | Remote: {salary['remote_allowance']:,.0f}")
        lines.append(f"Gross: {salary['gross_salary']:,.0f} EGP | Net (after tax + insurance): {salary['net_salary']:,.0f} EGP")
        lines.append("")

    training = data.get("training")
    if training:
        lines.append("=== TRAINING & DEVELOPMENT (current year) ===")
        lines.append(f"Budget: ${training['budget_total_usd']:,.0f} total | ${training['budget_used_usd']:,.0f} used | ${training['budget_remaining_usd']:,.0f} remaining")
        lines.append(f"Training days: {training['training_days_total']} total | {training['training_days_used']} used | {training['days_remaining']} remaining")
        courses = training.get("courses", [])
        if isinstance(courses, str):
            courses = json.loads(courses)
        if courses:
            lines.append("Courses completed:")
            for c in courses:
                cert = " ✓ certified" if c.get("certificate") else ""
                lines.append(f"  • {c['name']} ({c['provider']}, {c['date']}, ${c.get('cost_usd', 0):,.0f}){cert}")
        lines.append("")

    disc = data.get("active_disciplinary", [])
    if disc:
        lines.append("=== ACTIVE DISCIPLINARY RECORDS ===")
        for d in disc:
            lines.append(f"  {d['action_type']} — issued {d['issued_date']} | expires {d.get('expiry_date', 'N/A')}")
            lines.append(f"  Reason: {d['reason']}")
        lines.append("")

    lines.append(f"Data fetched: {data.get('fetched_at', 'today')}")
    return "\n".join(lines)


# ── Personal prompt (EN) ────────────────────────────────────────
PERSONAL_EN = PromptTemplate(
    template=(
        "You are a personal HR assistant for Horizon Tech. "
        "The employee currently logged in is asking you a question about their own data.\n\n"
        "RULES:\n"
        "1. Answer ONLY using the personal data block below. Never guess or generalise.\n"
        "2. Be specific with every number — state exact days, amounts, ratings.\n"
        "3. If a field is missing from the data, say 'this information is not available in your record.'\n"
        "4. MIRROR pronouns: the employee says 'I/my' → you say 'you/your'.\n"
        "5. Do NOT cite page numbers — this answer comes from the employee's live record, not the policy PDF.\n"
        "6. If the question requires both personal data AND a policy rule "
        "(e.g. 'can I take more leave?'), state their current balance first, "
        "then state the policy rule clearly.\n\n"
        "Conversation so far:\n{history}\n\n"
        "=== EMPLOYEE'S PERSONAL DATA ===\n{personal_data}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "question", "history"]
)

# ── Personal prompt (MSA Arabic) ───────────────────────────────
PERSONAL_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي في شركة أفق التقنية. "
        "الموظف المسجّل دخوله يسألك عن بياناته الشخصية.\n\n"
        "القواعد:\n"
        "1. أجب فقط بناءً على بيانات الموظف أدناه. لا تخمّن ولا تعمّم.\n"
        "2. كن دقيقاً في كل رقم — اذكر الأيام والمبالغ والتقييمات بدقة.\n"
        "3. إذا كانت المعلومة غير موجودة في البيانات، قل: 'هذه المعلومة غير متوفرة في سجلك.'\n"
        "4. طابق الضمائر: الموظف يقول 'أنا/لي' → أنت تقول 'أنت/لك'.\n"
        "5. لا تستخدم أرقام الصفحات — هذه الإجابة من السجل الحي للموظف.\n"
        "6. إذا تطلّب السؤال بيانات شخصية وقاعدة من السياسة، اذكر البيانات أولاً ثم القاعدة.\n\n"
        "المحادثة السابقة:\n{history}\n\n"
        "=== البيانات الشخصية للموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)

# ── Personal prompt (Egyptian Arabic) ─────────────────────────
PERSONAL_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي في شركة أفق التقنية. "
        "الموظف اللي عامل login بيسألك عن بياناته الشخصية. "
        "أجب بالعامية المصرية.\n\n"
        "القواعد:\n"
        "1. جاوب بس من البيانات الشخصية اللي تحت. متخمنش.\n"
        "2. اذكر كل رقم بدقة — أيام، مبالغ، تقييمات.\n"
        "3. لو المعلومة مش موجودة في البيانات، قول: 'المعلومة دي مش متاحة في سجلك.'\n"
        "4. الموظف بيقول 'أنا/بتاعي' → إنت بتقول 'إنت/بتاعك'.\n"
        "5. متستخدمش أرقام صفحات — الإجابة دي من السجل الحي.\n\n"
        "المحادثة اللي فاتت:\n{history}\n\n"
        "=== البيانات الشخصية للموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)

# ── Personal prompt (Franco Arabic) ───────────────────────────
PERSONAL_FRANCO = PromptTemplate(
    template=(
        "Enta mosa3ed HR shakhsy f Horizon Tech. "
        "El mowazaf el logged in byes2al 3an bayanatu el shakhsiyya. "
        "Egib bel Franco 3arabi.\n\n"
        "El rules:\n"
        "1. Egib bass men el bayanat el shakhsiyya tala7t. Matkhminsh.\n"
        "2. 2ol kol raqam beld2a — ayam, mablag, rating.\n"
        "3. Law el ma3loma mesh f el bayanat, 2ol: 'El ma3loma di mesh mawgoda f segelak.'\n"
        "4. El mowazaf bey2ol 'ana/bta3i' → enta bet2ol 'enta/bta3ak'.\n"
        "5. Matesta5dimsh argam sa7fat — el egaba di men el segel el live.\n\n"
        "El kalam elli fat:\n{history}\n\n"
        "=== El bayanat el shakhsiyya lel mowazaf ===\n{personal_data}\n\n"
        "El so2al: {question}\nEl egaba:"
    ),
    input_variables=["personal_data", "question", "history"]
)


# ── Hybrid prompt (EN) — both personal data + policy context ──
HYBRID_EN = PromptTemplate(
    template=(
        "You are an HR assistant for Horizon Tech.\n"
        "This question needs BOTH the employee's personal data AND the policy rules.\n\n"
        "RULES:\n"
        "1. State the employee's actual data first (their balance, rating, etc.).\n"
        "2. Then state the relevant policy rule from the policy context.\n"
        "3. Combine both to give a direct, actionable answer.\n"
        "4. Cite policy pages with [Page N | AR/EN]. Do NOT cite page numbers for personal data.\n\n"
        "Conversation so far:\n{history}\n\n"
        "=== EMPLOYEE'S PERSONAL DATA ===\n{personal_data}\n\n"
        "=== POLICY CONTEXT (from PDF) ===\n{policy_context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية في شركة أفق التقنية.\n"
        "هذا السؤال يحتاج بيانات الموظف الشخصية ونصّ السياسة معاً.\n\n"
        "القواعد:\n"
        "1. اذكر أولاً البيانات الفعلية للموظف.\n"
        "2. ثم اذكر القاعدة من السياسة.\n"
        "3. ادمج الاثنين لتقديم إجابة عملية مباشرة.\n"
        "4. وثّق صفحات السياسة بـ [Page N | AR/EN]. لا توثّق أرقام صفحات للبيانات الشخصية.\n\n"
        "المحادثة السابقة:\n{history}\n\n"
        "=== البيانات الشخصية للموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة (من ملف PDF) ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية في أفق التقنية.\n"
        "السؤال ده محتاج بيانات الموظف الشخصية وكمان قاعدة من السياسة.\n\n"
        "القواعد:\n"
        "1. اذكر بيانات الموظف الفعلية الأول.\n"
        "2. بعدين اذكر القاعدة من السياسة.\n"
        "3. ادمجهم في إجابة واضحة بالعامية المصرية.\n"
        "4. وثّق السياسة بـ [Page N | AR/EN]. متوثقش البيانات الشخصية.\n\n"
        "المحادثة اللي فاتت:\n{history}\n\n"
        "=== البيانات الشخصية للموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_FRANCO = PromptTemplate(
    template=(
        "Enta mosa3ed HR f Horizon Tech.\n"
        "El so2al da me7tag bayanat el mowazaf el shakhsiyya w kaman el policy.\n\n"
        "El rules:\n"
        "1. 2ol el bayanat el ha2i2iyya bta3et el mowazaf awwel.\n"
        "2. Ba3den 2ol el rule men el policy.\n"
        "3. Edmezhom f egaba wa7da bel Franco.\n"
        "4. Cite el policy bel [Page N | AR/EN]. Matktebishsh cite lel bayanat el shakhsiyya.\n\n"
        "El kalam elli fat:\n{history}\n\n"
        "=== El bayanat el shakhsiyya lel mowazaf ===\n{personal_data}\n\n"
        "=== El policy context ===\n{policy_context}\n\n"
        "El so2al: {question}\nEl egaba:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)


# ── Prompt selector ────────────────────────────────────────────
def get_personal_prompt(lang: str, dialect: str = None):
    if lang == "english":
        return PERSONAL_EN
    if lang == "franco":
        return PERSONAL_FRANCO
    if dialect == "egyptian":
        return PERSONAL_EGY
    return PERSONAL_AR

def get_hybrid_prompt(lang: str, dialect: str = None):
    if lang == "english":
        return HYBRID_EN
    if lang == "franco":
        return HYBRID_FRANCO
    if dialect == "egyptian":
        return HYBRID_EGY
    return HYBRID_AR