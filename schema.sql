-- ============================================================
-- Employee Database Schema
-- Run once in pgAdmin Query Tool on the 'horizonhr' database
-- ============================================================

-- ─── 1. EMPLOYEES ────────────────────────────────────────────
-- Core identity record. One row per employee.
-- manager_id is self-referential: every employee points to their manager.
CREATE TABLE employees (
    id                  SERIAL PRIMARY KEY,
    full_name           VARCHAR(120)    NOT NULL,
    full_name_ar        VARCHAR(120),                        -- Arabic name for Arabic UI
    email               VARCHAR(120)    NOT NULL UNIQUE,     -- used as login username
    password_hash       VARCHAR(255)    NOT NULL,            -- bcrypt hash
    grade               VARCHAR(10)     NOT NULL,            -- G1, G2, G3, G4, G5
    job_title           VARCHAR(120)    NOT NULL,
    department          VARCHAR(80)     NOT NULL,
    manager_id          INT             REFERENCES employees(id) ON DELETE SET NULL,
    hire_date           DATE            NOT NULL,
    employment_type     VARCHAR(30)     NOT NULL DEFAULT 'full-time',
    -- CHECK: full-time | part-time | contractor | intern
    work_model          VARCHAR(30)     NOT NULL DEFAULT 'hybrid',
    -- CHECK: in-office | hybrid | remote
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    probation_end_date  DATE,
    national_id         VARCHAR(30),
    phone               VARCHAR(30),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ─── 2. LEAVE BALANCES ───────────────────────────────────────
-- One row per employee per year per leave type.
-- Recalculated at year start; decremented when leave is approved.
-- "remaining" is always: entitled + carried_over - used - pending
CREATE TABLE leave_balances (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    year                INT             NOT NULL,
    leave_type          VARCHAR(40)     NOT NULL,
    -- annual | sick | maternity | paternity | marriage |
    -- bereavement | hajj | study | unpaid
    entitled_days       NUMERIC(6,1)    NOT NULL DEFAULT 0,
    carried_over_days   NUMERIC(6,1)    NOT NULL DEFAULT 0,
    used_days           NUMERIC(6,1)    NOT NULL DEFAULT 0,
    pending_days        NUMERIC(6,1)    NOT NULL DEFAULT 0,  -- requested, not yet approved
    encashed_days       NUMERIC(6,1)    NOT NULL DEFAULT 0,  -- days paid out instead of taken
    UNIQUE (employee_id, year, leave_type)
);

-- Computed column view so the app can always ask for "remaining" cleanly
CREATE OR REPLACE VIEW leave_balances_view AS
SELECT *,
       GREATEST(0, entitled_days + carried_over_days - used_days - pending_days - encashed_days)
           AS remaining_days
FROM leave_balances;

-- ─── 3. LEAVE REQUESTS ───────────────────────────────────────
-- Every individual leave request submitted by an employee.
-- status transitions: pending → approved | rejected | cancelled
CREATE TABLE leave_requests (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    leave_type          VARCHAR(40)     NOT NULL,
    start_date          DATE            NOT NULL,
    end_date            DATE            NOT NULL,
    days_count          NUMERIC(6,1)    NOT NULL,
    status              VARCHAR(20)     NOT NULL DEFAULT 'pending',
    approved_by         INT             REFERENCES employees(id),
    approved_at         TIMESTAMPTZ,
    rejection_reason    TEXT,
    notes               TEXT,           -- employee's note when submitting
    document_submitted  BOOLEAN         NOT NULL DEFAULT FALSE,
    requested_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_dates CHECK (end_date >= start_date),
    CONSTRAINT chk_status CHECK (status IN ('pending','approved','rejected','cancelled'))
);

-- ─── 4. PERFORMANCE REVIEWS ──────────────────────────────────
-- One row per review per employee. Horizon Tech does bi-annual reviews.
-- review_period examples: 'H1 2025', 'H2 2025', 'Annual 2024'
CREATE TABLE performance_reviews (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    review_period       VARCHAR(20)     NOT NULL,            -- e.g. 'H1 2025'
    review_date         DATE            NOT NULL,
    rating              INT             NOT NULL,            -- 1 to 5
    rating_label        VARCHAR(40)     NOT NULL,
    -- 1=Unsatisfactory | 2=Needs Improvement | 3=Meets Expectations
    -- 4=Exceeds Expectations | 5=Exceptional
    salary_increment_pct NUMERIC(5,2)   NOT NULL DEFAULT 0,  -- % raise applied
    bonus_multiplier    NUMERIC(4,2)    NOT NULL DEFAULT 0,
    reviewer_id         INT             REFERENCES employees(id),
    strengths           TEXT,
    areas_for_growth    TEXT,
    overall_comments    TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_rating CHECK (rating BETWEEN 1 AND 5),
    UNIQUE (employee_id, review_period)
);

-- ─── 5. OKRs / GOALS ─────────────────────────────────────────
-- One row per objective. Each objective can have multiple key results
-- stored as JSONB so the schema stays flat.
-- period examples: 'H1 2026', 'H2 2026'
CREATE TABLE okrs (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    period              VARCHAR(20)     NOT NULL,
    title               VARCHAR(200)    NOT NULL,
    description         TEXT,
    key_results         JSONB           NOT NULL DEFAULT '[]',
    -- [{kr: "...", target: "...", progress_pct: 75, status: "on-track"}]
    overall_progress_pct INT            NOT NULL DEFAULT 0,  -- 0–100
    status              VARCHAR(20)     NOT NULL DEFAULT 'in-progress',
    -- in-progress | on-track | at-risk | completed | cancelled
    set_date            DATE            NOT NULL,
    due_date            DATE,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_progress CHECK (overall_progress_pct BETWEEN 0 AND 100)
);

-- ─── 6. PAYROLL RECORDS ──────────────────────────────────────
-- Monthly salary snapshot. One row per employee per month.
-- month is always the first day of the month (e.g. 2026-03-01).
CREATE TABLE payroll_records (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    month               DATE            NOT NULL,            -- always 1st of month
    base_salary         NUMERIC(12,2)   NOT NULL,
    transport_allowance NUMERIC(10,2)   NOT NULL DEFAULT 0,
    housing_allowance   NUMERIC(10,2)   NOT NULL DEFAULT 0,
    mobile_allowance    NUMERIC(10,2)   NOT NULL DEFAULT 0,
    remote_allowance    NUMERIC(10,2)   NOT NULL DEFAULT 0,
    shift_allowance     NUMERIC(10,2)   NOT NULL DEFAULT 0,
    on_call_allowance   NUMERIC(10,2)   NOT NULL DEFAULT 0,
    gross_salary        NUMERIC(12,2)   NOT NULL,            -- sum of all above
    income_tax          NUMERIC(10,2)   NOT NULL DEFAULT 0,
    social_insurance    NUMERIC(10,2)   NOT NULL DEFAULT 0,
    other_deductions    NUMERIC(10,2)   NOT NULL DEFAULT 0,
    net_salary          NUMERIC(12,2)   NOT NULL,
    notes               TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (employee_id, month)
);

-- ─── 7. TRAINING RECORDS ─────────────────────────────────────
-- One row per employee per year tracking their L&D budget and courses.
-- courses is JSONB: [{name, provider, date, cost_usd, days, certificate}]
CREATE TABLE training_records (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    year                INT             NOT NULL,
    budget_total_usd    NUMERIC(10,2)   NOT NULL DEFAULT 0,
    budget_used_usd     NUMERIC(10,2)   NOT NULL DEFAULT 0,
    training_days_total INT             NOT NULL DEFAULT 0,
    training_days_used  INT             NOT NULL DEFAULT 0,
    courses             JSONB           NOT NULL DEFAULT '[]',
    scholarship_applied BOOLEAN         NOT NULL DEFAULT FALSE,
    scholarship_amount  NUMERIC(10,2),
    scholarship_status  VARCHAR(20),
    -- pending | approved | rejected | completed
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    UNIQUE (employee_id, year)
);

-- ─── 8. DISCIPLINARY RECORDS ─────────────────────────────────
-- One row per disciplinary action. Visibility restricted:
-- employee can only see their own; managers see their reports.
CREATE TABLE disciplinary_records (
    id                  SERIAL PRIMARY KEY,
    employee_id         INT             NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    action_type         VARCHAR(40)     NOT NULL,
    -- verbal_warning | written_warning_1 | written_warning_final | pip | termination
    issued_date         DATE            NOT NULL,
    expiry_date         DATE,                                -- null for termination
    issued_by           INT             NOT NULL REFERENCES employees(id),
    reason              TEXT            NOT NULL,
    outcome             TEXT,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,-- false when expired or overturned
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ─── INDEXES ─────────────────────────────────────────────────
-- Every foreign key that will be used in WHERE clauses gets an index.
CREATE INDEX idx_leave_balances_employee   ON leave_balances(employee_id, year);
CREATE INDEX idx_leave_requests_employee   ON leave_requests(employee_id, status);
CREATE INDEX idx_performance_employee      ON performance_reviews(employee_id);
CREATE INDEX idx_okrs_employee             ON okrs(employee_id, period);
CREATE INDEX idx_payroll_employee          ON payroll_records(employee_id, month);
CREATE INDEX idx_training_employee         ON training_records(employee_id, year);
CREATE INDEX idx_disciplinary_employee     ON disciplinary_records(employee_id);
CREATE INDEX idx_employees_email           ON employees(email);
CREATE INDEX idx_employees_manager        ON employees(manager_id);