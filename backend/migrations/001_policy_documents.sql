-- Run this once against your existing DB (psql -f 001_policy_documents.sql, or paste into pgAdmin).

CREATE TABLE IF NOT EXISTS policy_documents (
    id           SERIAL PRIMARY KEY,
    file_path    TEXT        NOT NULL,
    doc_name     TEXT        NOT NULL,
    lang         VARCHAR(10) NOT NULL CHECK (lang IN ('arabic', 'english')),
    is_active    BOOLEAN     NOT NULL DEFAULT TRUE,
    uploaded_by  INT         REFERENCES employees(id) ON DELETE SET NULL,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_policy_documents_active ON policy_documents(is_active, lang);

-- Seed with the 7 PDFs that were previously hardcoded in setup.py, so nothing
-- breaks on first deploy. Adjust file_path if your policies/ folder differs.
INSERT INTO policy_documents (file_path, doc_name, lang, is_active) VALUES
    ('policies/ar_policy.pdf',               'ar_policy.pdf',               'arabic',  TRUE),
    ('policies/ar_recruitment.pdf',          'ar_recruitment.pdf',          'arabic',  TRUE),
    ('policies/ar_payroll_finance.pdf',      'ar_payroll_finance.pdf',      'arabic',  TRUE),
    ('policies/eng_policy.pdf',              'eng_policy.pdf',              'english', TRUE),
    ('policies/eng_wellness_benefits.pdf',   'eng_wellness_benefits.pdf',   'english', TRUE),
    ('policies/eng_training_development.pdf','eng_training_development.pdf','english', TRUE),
    ('policies/eng_workplace_conduct.pdf',   'eng_workplace_conduct.pdf',   'english', TRUE)
ON CONFLICT DO NOTHING;
