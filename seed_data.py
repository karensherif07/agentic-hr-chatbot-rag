"""
Creates:
  - 13 employees across all grades (G1–G5), departments, and work models
  - Leave balances for 2025 and 2026
  - Leave requests (approved, pending, rejected)
  - Performance reviews for 2024 and 2025 (H1 + H2)
  - OKRs for H1 2026
  - Payroll records for Jan–Mar 2026
  - Training records for 2025 and 2026
"""

import os, json, bcrypt
from datetime import date, timedelta
from decimal import Decimal
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("DB_URL"))

def h(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def run(conn, sql, params=None):
    conn.execute(text(sql), params or {})

# ── Helper: insert employee and return its id ──────────────────
def insert_employee(conn, data: dict) -> int:
    result = conn.execute(text("""
        INSERT INTO employees
            (full_name, full_name_ar, email, password_hash, grade, job_title,
             department, manager_id, hire_date, employment_type, work_model,
             is_active, probation_end_date, phone)
        VALUES
            (:full_name, :full_name_ar, :email, :password_hash, :grade, :job_title,
             :department, :manager_id, :hire_date, :employment_type, :work_model,
             :is_active, :probation_end_date, :phone)
        RETURNING id
    """), data)
    return result.fetchone()[0]

# ── Leave balances helper ──────────────────────────────────────
def insert_leave_balance(conn, employee_id, year, leave_type,
                          entitled, carried=0, used=0, pending=0, encashed=0):
    conn.execute(text("""
        INSERT INTO leave_balances
            (employee_id, year, leave_type, entitled_days, carried_over_days,
             used_days, pending_days, encashed_days)
        VALUES
            (:eid, :year, :lt, :en, :co, :ud, :pd, :ec)
        ON CONFLICT (employee_id, year, leave_type) DO NOTHING
    """), dict(eid=employee_id, year=year, lt=leave_type,
               en=entitled, co=carried, ud=used, pd=pending, ec=encashed))

# ── Leave request helper ───────────────────────────────────────
def insert_leave_request(conn, employee_id, leave_type, start, end,
                          days, status, approved_by=None, notes=None):
    conn.execute(text("""
        INSERT INTO leave_requests
            (employee_id, leave_type, start_date, end_date, days_count,
             status, approved_by, document_submitted, notes, requested_at)
        VALUES
            (:eid, :lt, :sd, :ed, :dc, :st, :ab, TRUE, :notes, NOW())
    """), dict(eid=employee_id, lt=leave_type, sd=start, ed=end,
               dc=days, st=status, ab=approved_by, notes=notes))

# ── Performance review helper ──────────────────────────────────
RATING_LABELS = {
    1: 'Unsatisfactory',
    2: 'Needs Improvement',
    3: 'Meets Expectations',
    4: 'Exceeds Expectations',
    5: 'Exceptional'
}
RATING_RAISE = {1: 0, 2: 0, 3: 8, 4: 15, 5: 20}
RATING_BONUS = {1: 0, 2: 0, 3: 1.0, 4: 1.25, 5: 1.5}

def insert_review(conn, employee_id, period, review_date, rating,
                   reviewer_id, strengths='', areas='', comments=''):
    conn.execute(text("""
        INSERT INTO performance_reviews
            (employee_id, review_period, review_date, rating, rating_label,
             salary_increment_pct, bonus_multiplier, reviewer_id,
             strengths, areas_for_growth, overall_comments)
        VALUES
            (:eid, :period, :rd, :rating, :label, :raise_pct, :bonus,
             :rev, :s, :a, :c)
        ON CONFLICT (employee_id, review_period) DO NOTHING
    """), dict(eid=employee_id, period=period, rd=review_date,
               rating=rating, label=RATING_LABELS[rating],
               raise_pct=RATING_RAISE[rating], bonus=RATING_BONUS[rating],
               rev=reviewer_id, s=strengths, a=areas, c=comments))

# ── OKR helper ────────────────────────────────────────────────
def insert_okr(conn, employee_id, period, title, key_results, progress, status, set_date):
    conn.execute(text("""
        INSERT INTO okrs
            (employee_id, period, title, key_results, overall_progress_pct,
             status, set_date)
        VALUES
            (:eid, :period, :title, CAST(:kr AS JSONB), :prog, :status, :sd)
    """), dict(eid=employee_id, period=period, title=title,
               kr=json.dumps(key_results), prog=progress, status=status, sd=set_date))

# ── Payroll helper ─────────────────────────────────────────────
def insert_payroll(conn, employee_id, month, base, transport=1500,
                    housing=0, mobile=0, remote=0, tax_rate=0.15):
    gross = base + transport + housing + mobile + remote
    tax = round(gross * tax_rate, 2)
    social = round(base * 0.11, 2)
    net = round(gross - tax - social, 2)
    conn.execute(text("""
        INSERT INTO payroll_records
            (employee_id, month, base_salary, transport_allowance, housing_allowance,
             mobile_allowance, remote_allowance, gross_salary,
             income_tax, social_insurance, net_salary)
        VALUES
            (:eid, :month, :base, :tr, :ho, :mo, :re, :gross, :tax, :soc, :net)
        ON CONFLICT (employee_id, month) DO NOTHING
    """), dict(eid=employee_id, month=month, base=base, tr=transport,
               ho=housing, mo=mobile, re=remote, gross=gross,
               tax=tax, soc=social, net=net))

# ── Training helper ────────────────────────────────────────────
def insert_training(conn, employee_id, year, budget, used, days_total,
                     days_used, courses):
    conn.execute(text("""
        INSERT INTO training_records
            (employee_id, year, budget_total_usd, budget_used_usd,
             training_days_total, training_days_used, courses)
        VALUES
            (:eid, :yr, :bt, :bu, :dt, :du, CAST(:courses AS JSONB))
        ON CONFLICT (employee_id, year) DO NOTHING
    """), dict(eid=employee_id, yr=year, bt=budget, bu=used,
               dt=days_total, du=days_used, courses=json.dumps(courses)))


# ══════════════════════════════════════════════════════════════
# MAIN SEED
# ══════════════════════════════════════════════════════════════
with engine.begin() as conn:

    # ── EMPLOYEES ─────────────────────────────────────────────
    # G5 — CEO (no manager)
    ceo_id = insert_employee(conn, dict(
        full_name='Layla Hassan', full_name_ar='ليلى حسن',
        email='layla.hassan@horizontech.com', password_hash=h('pass1234'),
        grade='G5', job_title='Chief Executive Officer',
        department='Executive', manager_id=None,
        hire_date=date(2017, 3, 1), employment_type='full-time',
        work_model='in-office', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0001'
    ))

    # G5 — HR Director
    hr_dir_id = insert_employee(conn, dict(
        full_name='Amr Youssef', full_name_ar='عمرو يوسف',
        email='amr.youssef@horizontech.com', password_hash=h('pass1234'),
        grade='G5', job_title='HR Director',
        department='Human Resources', manager_id=ceo_id,
        hire_date=date(2018, 6, 15), employment_type='full-time',
        work_model='in-office', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0002'
    ))

    # G4 — Engineering Manager
    eng_mgr_id = insert_employee(conn, dict(
        full_name='Sara El-Masry', full_name_ar='سارة المصري',
        email='sara.elmasry@horizontech.com', password_hash=h('pass1234'),
        grade='G4', job_title='Engineering Manager',
        department='Engineering', manager_id=ceo_id,
        hire_date=date(2019, 9, 1), employment_type='full-time',
        work_model='hybrid', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0003'
    ))

    # G4 — Product Manager
    prod_mgr_id = insert_employee(conn, dict(
        full_name='Omar Nabil', full_name_ar='عمر نبيل',
        email='omar.nabil@horizontech.com', password_hash=h('pass1234'),
        grade='G4', job_title='Product Manager',
        department='Product', manager_id=ceo_id,
        hire_date=date(2020, 2, 1), employment_type='full-time',
        work_model='hybrid', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0004'
    ))

    # G3 — Senior Software Engineer (reports to eng_mgr)
    sen_eng_id = insert_employee(conn, dict(
        full_name='Nour Khalil', full_name_ar='نور خليل',
        email='nour.khalil@horizontech.com', password_hash=h('pass1234'),
        grade='G3', job_title='Senior Software Engineer',
        department='Engineering', manager_id=eng_mgr_id,
        hire_date=date(2021, 4, 15), employment_type='full-time',
        work_model='remote', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0005'
    ))
    # G3 — Senior Software Engineer (Karen Sherif)
    karen_id = insert_employee(conn, dict(
        full_name='Karen Sherif', full_name_ar='كارين شريف',
        email='karen.sherif@horizontech.com', password_hash=h('pass1234'),
        grade='G3', job_title='Senior Software Engineer',
        department='Engineering', manager_id=eng_mgr_id, # Reports to Sara
        hire_date=date(2024, 1, 10), employment_type='full-time',
        work_model='remote', is_active=True,
        probation_end_date=None, phone='+20-10-1234-5678'
    ))

    # G3 — HR Lead
    hr_lead_id = insert_employee(conn, dict(
        full_name='Mona Saad', full_name_ar='منى سعد',
        email='mona.saad@horizontech.com', password_hash=h('pass1234'),
        grade='G3', job_title='HR Team Lead',
        department='Human Resources', manager_id=hr_dir_id,
        hire_date=date(2020, 7, 1), employment_type='full-time',
        work_model='in-office', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0006'
    ))

    # G2 — Software Engineer
    eng2_id = insert_employee(conn, dict(
        full_name='Ahmed Mostafa', full_name_ar='أحمد مصطفى',
        email='ahmed.mostafa@horizontech.com', password_hash=h('pass1234'),
        grade='G2', job_title='Software Engineer',
        department='Engineering', manager_id=sen_eng_id,
        hire_date=date(2022, 11, 1), employment_type='full-time',
        work_model='hybrid', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0007'
    ))

    # G2 — Data Analyst
    analyst_id = insert_employee(conn, dict(
        full_name='Dina Ramadan', full_name_ar='دينا رمضان',
        email='dina.ramadan@horizontech.com', password_hash=h('pass1234'),
        grade='G2', job_title='Data Analyst',
        department='Product', manager_id=prod_mgr_id,
        hire_date=date(2023, 3, 15), employment_type='full-time',
        work_model='hybrid', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0008'
    ))

    # G2 — HR Specialist
    hr_spec_id = insert_employee(conn, dict(
        full_name='Yara Fathy', full_name_ar='يارا فتحي',
        email='yara.fathy@horizontech.com', password_hash=h('pass1234'),
        grade='G2', job_title='HR Specialist',
        department='Human Resources', manager_id=hr_lead_id,
        hire_date=date(2023, 8, 1), employment_type='full-time',
        work_model='in-office', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0009'
    ))

    # G1 — Junior Engineer (recently hired, in probation)
    junior_id = insert_employee(conn, dict(
        full_name='Karim Adel', full_name_ar='كريم عادل',
        email='karim.adel@horizontech.com', password_hash=h('pass1234'),
        grade='G1', job_title='Junior Software Engineer',
        department='Engineering', manager_id=eng2_id,
        hire_date=date(2026, 1, 15), employment_type='full-time',
        work_model='in-office', is_active=True,
        probation_end_date=date(2026, 4, 15), phone='+20-10-0000-0010'
    ))

    # G1 — Intern
    intern_id = insert_employee(conn, dict(
        full_name='Salma Ibrahim', full_name_ar='سلمى إبراهيم',
        email='salma.ibrahim@horizontech.com', password_hash=h('pass1234'),
        grade='G1', job_title='Engineering Intern',
        department='Engineering', manager_id=eng2_id,
        hire_date=date(2026, 2, 1), employment_type='intern',
        work_model='in-office', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0011'
    ))

    # Part-time contractor
    contractor_id = insert_employee(conn, dict(
        full_name='Tarek Zaki', full_name_ar='طارق زكي',
        email='tarek.zaki@horizontech.com', password_hash=h('pass1234'),
        grade='G2', job_title='UX Designer (Contractor)',
        department='Product', manager_id=prod_mgr_id,
        hire_date=date(2025, 6, 1), employment_type='contractor',
        work_model='remote', is_active=True,
        probation_end_date=None, phone='+20-10-0000-0012'
    ))

    print("✓ Employees inserted")

    # ── LEAVE BALANCES 2025 & 2026 ────────────────────────────
    # Format: (employee_id, year, leave_type, entitled, carried, used, pending, encashed)

    # Layla Hassan — G5, 8+ yrs → 30 annual days
    insert_leave_balance(conn, ceo_id,    2025, 'annual', 30, 0, 20, 0, 5)
    insert_leave_balance(conn, ceo_id,    2026, 'annual', 30, 5, 3, 0, 0)
    insert_leave_balance(conn, ceo_id,    2026, 'sick',   10, 0, 0, 0, 0)

    # Amr Youssef — G5, 7+ yrs → 25 annual days
    insert_leave_balance(conn, hr_dir_id, 2025, 'annual', 25, 0, 18, 0, 0)
    insert_leave_balance(conn, hr_dir_id, 2026, 'annual', 25, 7, 5, 2, 0)
    insert_leave_balance(conn, hr_dir_id, 2026, 'sick',   10, 0, 1, 0, 0)

    # Sara El-Masry — G4, 6+ yrs → 25 annual days
    insert_leave_balance(conn, eng_mgr_id, 2025, 'annual', 25, 0, 22, 0, 0)
    insert_leave_balance(conn, eng_mgr_id, 2026, 'annual', 25, 3, 4, 0, 0)
    insert_leave_balance(conn, eng_mgr_id, 2026, 'sick',   10, 0, 2, 0, 0)
    insert_leave_balance(conn, eng_mgr_id, 2026, 'maternity', 90, 0, 0, 0, 0)

    # Karen Sherif — G3, 2+ yrs → 21 annual days
    insert_leave_balance(conn, karen_id, 2025, 'annual', 21, 0, 15, 0, 0)
    insert_leave_balance(conn, karen_id, 2026, 'annual', 21, 6, 0, 0, 0)
    insert_leave_balance(conn, karen_id, 2026, 'sick',   10, 0, 0, 0, 0)

    # Omar Nabil — G4, 5+ yrs → 25 annual days
    insert_leave_balance(conn, prod_mgr_id, 2025, 'annual', 25, 0, 15, 0, 3)
    insert_leave_balance(conn, prod_mgr_id, 2026, 'annual', 25, 7, 6, 3, 0)
    insert_leave_balance(conn, prod_mgr_id, 2026, 'hajj',    20, 0, 0, 0, 0)

    # Nour Khalil — G3, 4+ yrs → 21 annual days
    insert_leave_balance(conn, sen_eng_id, 2025, 'annual', 21, 0, 14, 0, 0)
    insert_leave_balance(conn, sen_eng_id, 2026, 'annual', 21, 7, 5, 0, 0)
    insert_leave_balance(conn, sen_eng_id, 2026, 'sick',   10, 0, 0, 0, 0)
    insert_leave_balance(conn, sen_eng_id, 2026, 'paternity', 10, 0, 10, 0, 0)

    # Mona Saad — G3, 5+ yrs → 21 annual days
    insert_leave_balance(conn, hr_lead_id, 2025, 'annual', 21, 0, 16, 0, 0)
    insert_leave_balance(conn, hr_lead_id, 2026, 'annual', 21, 5, 3, 2, 0)

    # Ahmed Mostafa — G2, 3+ yrs → 21 annual days
    insert_leave_balance(conn, eng2_id, 2025, 'annual', 21, 0, 10, 0, 0)
    insert_leave_balance(conn, eng2_id, 2026, 'annual', 21, 7, 2, 5, 0)
    insert_leave_balance(conn, eng2_id, 2026, 'sick',   10, 0, 3, 0, 0)

    # Dina Ramadan — G2, 2+ yrs → 21 annual days
    insert_leave_balance(conn, analyst_id, 2025, 'annual', 21, 0, 8, 0, 0)
    insert_leave_balance(conn, analyst_id, 2026, 'annual', 21, 7, 0, 0, 0)
    insert_leave_balance(conn, analyst_id, 2026, 'study',  10, 0, 5, 0, 0)
    insert_leave_balance(conn, analyst_id, 2026, 'marriage', 5, 0, 5, 0, 0)

    # Yara Fathy — G2, 2+ yrs → 21 annual days
    insert_leave_balance(conn, hr_spec_id, 2025, 'annual', 21, 0, 11, 0, 0)
    insert_leave_balance(conn, hr_spec_id, 2026, 'annual', 21, 7, 4, 0, 0)

    # Karim Adel — G1, <1 yr, in probation → 14 annual (can't take yet)
    insert_leave_balance(conn, junior_id, 2026, 'annual', 14, 0, 0, 0, 0)

    print("✓ Leave balances inserted")

    # ── LEAVE REQUESTS ────────────────────────────────────────
    insert_leave_request(conn, sen_eng_id,  'annual',    date(2026,2,2),  date(2026,2,6),  5, 'approved', eng_mgr_id, 'Family trip')
    insert_leave_request(conn, eng2_id,     'annual',    date(2026,3,10), date(2026,3,12), 3, 'pending',  None,       'Personal errands')
    insert_leave_request(conn, eng2_id,     'annual',    date(2026,2,1),  date(2026,2,2),  2, 'approved', sen_eng_id, None)
    insert_leave_request(conn, analyst_id,  'study',     date(2026,1,20), date(2026,1,24), 5, 'approved', prod_mgr_id,'Masters exams')
    insert_leave_request(conn, hr_spec_id,  'annual',    date(2026,1,5),  date(2026,1,8),  4, 'approved', hr_lead_id, None)
    insert_leave_request(conn, prod_mgr_id, 'annual',    date(2026,2,15), date(2026,2,20), 6, 'approved', ceo_id,     'Vacation')
    insert_leave_request(conn, prod_mgr_id, 'annual',    date(2026,3,1),  date(2026,3,3),  3, 'pending',  None,       None)
    insert_leave_request(conn, hr_dir_id,   'sick',      date(2026,1,28), date(2026,1,28), 1, 'approved', ceo_id,     'Medical visit')
    insert_leave_request(conn, eng2_id,     'sick',      date(2026,1,15), date(2026,1,17), 3, 'approved', sen_eng_id, None)
    insert_leave_request(conn, analyst_id,  'marriage',  date(2025,11,10),date(2025,11,14),5, 'approved', prod_mgr_id,'Marriage documents submitted')
    insert_leave_request(conn, sen_eng_id,  'paternity', date(2025,12,1), date(2025,12,10),10,'approved', eng_mgr_id, 'New baby')
    insert_leave_request(conn, junior_id,   'annual',    date(2026,3,20), date(2026,3,21), 2, 'rejected', eng_mgr_id, 'Still in probation')

    print("✓ Leave requests inserted")

    # ── PERFORMANCE REVIEWS ───────────────────────────────────
    # CEO reviews managers; managers review their reports
    insert_review(conn, eng_mgr_id, 'H1 2024', date(2024,6,30), 4, ceo_id,    'Delivered platform redesign on time', 'Delegation skills', 'Strong leader')
    insert_review(conn, eng_mgr_id, 'H2 2024', date(2024,12,31),5, ceo_id,    'Led 0→1 new product', 'Work-life balance', 'Outstanding year')
    insert_review(conn, eng_mgr_id, 'H1 2025', date(2025,6,30), 4, ceo_id,    'Mentored 3 engineers to promotion', '', 'Consistently excellent')

    insert_review(conn, prod_mgr_id,'H1 2024', date(2024,6,30), 3, ceo_id,    'Delivered roadmap on time', 'Data-driven decisions', 'Solid performance')
    insert_review(conn, prod_mgr_id,'H2 2024', date(2024,12,31),4, ceo_id,    'Launched 2 major features', 'Stakeholder communication', 'Good growth')
    insert_review(conn, prod_mgr_id,'H1 2025', date(2025,6,30), 3, ceo_id,    'Maintained product quality', 'Speed of execution', 'Meets all expectations')

    insert_review(conn, sen_eng_id, 'H1 2024', date(2024,6,30), 3, eng_mgr_id,'Clean code, good PR reviews', 'Technical leadership', 'Good engineer')
    insert_review(conn, sen_eng_id, 'H2 2024', date(2024,12,31),4, eng_mgr_id,'Architected new microservice', 'Documentation', 'Impressive H2')
    insert_review(conn, sen_eng_id, 'H1 2025', date(2025,6,30), 5, eng_mgr_id,'Led team through migration', '', 'Best review yet')

    insert_review(conn, hr_lead_id, 'H1 2024', date(2024,6,30), 4, hr_dir_id, 'Reduced time-to-hire by 30%', 'Reporting depth', 'Very effective')
    insert_review(conn, hr_lead_id, 'H2 2024', date(2024,12,31),4, hr_dir_id, 'Policy rollout went smoothly', '', 'Consistent high performer')
    insert_review(conn, hr_lead_id, 'H1 2025', date(2025,6,30), 3, hr_dir_id, 'Managed leave audit well', 'Cross-team collaboration', 'Solid')

    insert_review(conn, eng2_id,    'H1 2024', date(2024,6,30), 3, sen_eng_id,'Met all sprint commitments', 'Code review depth', 'Good start')
    insert_review(conn, eng2_id,    'H2 2024', date(2024,12,31),3, sen_eng_id,'Reliable delivery', 'Taking ownership', 'Steady')
    insert_review(conn, eng2_id,    'H1 2025', date(2025,6,30), 4, sen_eng_id,'Owned full feature end-to-end', 'Estimation accuracy', 'Clear improvement')

    insert_review(conn, analyst_id, 'H1 2024', date(2024,6,30), 2, prod_mgr_id,'Reports lacked depth', 'SQL proficiency, storytelling', 'Needs to grow')
    insert_review(conn, analyst_id, 'H2 2024', date(2024,12,31),3, prod_mgr_id,'Improved data storytelling', 'Proactiveness', 'Good recovery')
    insert_review(conn, analyst_id, 'H1 2025', date(2025,6,30), 4, prod_mgr_id,'Built self-serve dashboard', '', 'Strong progress')

    insert_review(conn, hr_spec_id, 'H1 2025', date(2025,6,30), 3, hr_lead_id, 'Managed onboarding well', 'Policy knowledge depth', 'Good first year+')

    print("✓ Performance reviews inserted")

    # ── OKRs H1 2026 ──────────────────────────────────────────
    insert_okr(conn, eng_mgr_id, 'H1 2026',
        'Scale engineering team to 20 FTEs',
        [{'kr':'Hire 4 engineers by March','target':'4','progress_pct':75,'status':'on-track'},
         {'kr':'Reduce time-to-hire below 45 days','target':'45 days','progress_pct':60,'status':'on-track'}],
        68, 'on-track', date(2026,1,10))

    insert_okr(conn, sen_eng_id, 'H1 2026',
        'Migrate legacy service to microservices',
        [{'kr':'Complete service decomposition doc','target':'100%','progress_pct':100,'status':'completed'},
         {'kr':'Deploy 3 new services to production','target':'3','progress_pct':33,'status':'on-track'},
         {'kr':'Zero critical bugs in migrated services','target':'0','progress_pct':90,'status':'on-track'}],
        74, 'on-track', date(2026,1,12))

    insert_okr(conn, eng2_id, 'H1 2026',
        'Improve API response time by 40%',
        [{'kr':'Profile and fix top 5 slow endpoints','target':'5','progress_pct':80,'status':'on-track'},
         {'kr':'Add caching layer to search API','target':'done','progress_pct':50,'status':'at-risk'}],
        65, 'on-track', date(2026,1,15))

    insert_okr(conn, analyst_id, 'H1 2026',
        'Build real-time product analytics dashboard',
        [{'kr':'Connect 3 data sources','target':'3','progress_pct':100,'status':'completed'},
         {'kr':'Launch dashboard to 5 stakeholders','target':'5','progress_pct':20,'status':'at-risk'}],
        60, 'at-risk', date(2026,1,10))

    insert_okr(conn, hr_spec_id, 'H1 2026',
        'Digitise onboarding process fully',
        [{'kr':'Upload all onboarding docs to portal','target':'100%','progress_pct':90,'status':'on-track'},
         {'kr':'Reduce onboarding time from 5 to 3 days','target':'3 days','progress_pct':40,'status':'at-risk'}],
        65, 'on-track', date(2026,1,8))

    print("✓ OKRs inserted")

    # ── PAYROLL Jan–Mar 2026 ──────────────────────────────────
    # Grade → base salary midpoint
    grade_base = {
        ceo_id:       100000,
        hr_dir_id:     90000,
        eng_mgr_id:    57000,
        prod_mgr_id:   55000,
        sen_eng_id:    36000,
        karen_id:      36000, 
        hr_lead_id:    34000,
        eng2_id:       21000,
        analyst_id:    20000,
        hr_spec_id:    19000,
        junior_id:     11000,
    }
    for month in [date(2026,1,1), date(2026,2,1), date(2026,3,1)]:
        for eid, base in grade_base.items():
            emp_grade = {
                ceo_id: 'G5', hr_dir_id: 'G5', eng_mgr_id: 'G4', prod_mgr_id: 'G4',
                sen_eng_id: 'G3', hr_lead_id: 'G3', eng2_id: 'G2', karen_id: 'G3',
                analyst_id: 'G2', hr_spec_id: 'G2', junior_id: 'G1'
            }[eid]
            housing  = round(base * 0.20, 2) if emp_grade in ('G4','G5') else 0
            mobile   = 500 if emp_grade != 'G1' else 0
            remote   = 800 if eid in (sen_eng_id,) else 0
            insert_payroll(conn, eid, month, base,
                           transport=1500, housing=housing,
                           mobile=mobile, remote=remote)

    print("✓ Payroll inserted")

    # ── TRAINING 2025 & 2026 ──────────────────────────────────
    insert_training(conn, eng_mgr_id, 2025, 8000, 6200, 10, 8,
        [{'name':'AWS Solutions Architect','provider':'AWS','date':'2025-03-10','cost_usd':3200,'days':3,'certificate':True},
         {'name':'Engineering Leadership','provider':'Pluralsight','date':'2025-07-20','cost_usd':3000,'days':5,'certificate':True}])
    insert_training(conn, eng_mgr_id, 2026, 8000, 0, 10, 0, [])

    insert_training(conn, sen_eng_id, 2025, 5000, 4800, 8, 7,
        [{'name':'System Design Masterclass','provider':'Educative','date':'2025-02-15','cost_usd':1500,'days':4,'certificate':False},
         {'name':'Kubernetes for Developers','provider':'Linux Foundation','date':'2025-09-01','cost_usd':3300,'days':3,'certificate':True}])
    insert_training(conn, sen_eng_id, 2026, 5000, 1200, 8, 2,
        [{'name':'Rust Programming','provider':'O\'Reilly','date':'2026-01-20','cost_usd':1200,'days':2,'certificate':False}])

    insert_training(conn, eng2_id, 2025, 5000, 3100, 8, 5,
        [{'name':'React Advanced Patterns','provider':'Frontend Masters','date':'2025-05-10','cost_usd':800,'days':2,'certificate':False},
         {'name':'PostgreSQL for Developers','provider':'Udemy','date':'2025-10-01','cost_usd':2300,'days':3,'certificate':True}])
    insert_training(conn, eng2_id, 2026, 5000, 0, 8, 0, [])

    insert_training(conn, analyst_id, 2025, 5000, 4500, 8, 7,
        [{'name':'dbt Core Certification','provider':'dbt Labs','date':'2025-04-01','cost_usd':2000,'days':3,'certificate':True},
         {'name':'Tableau Desktop Specialist','provider':'Tableau','date':'2025-11-15','cost_usd':2500,'days':4,'certificate':True}])
    insert_training(conn, analyst_id, 2026, 5000, 0, 8, 0, [],)

    insert_training(conn, hr_spec_id, 2026, 5000, 0, 8, 0, [])

    print("✓ Training records inserted")
    print("\n✅ Seed complete. Database is ready.")