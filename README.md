# Horizon Tech HR Assistant — React + FastAPI Edition

This replaces the Streamlit app with a proper split:

- **`backend/`** — FastAPI, wraps your existing `agent.py` / `retrieval.py` /
  `personal_data.py` / `setup.py` logic (barely changed) behind a REST API.
- **`frontend/`** — React + Vite + TypeScript, a from-scratch UI (navy/brass
  "Horizon" design system) that replaces every Streamlit widget.

Your core AI/retrieval logic did **not** change — `agent.py`, `retrieval.py`,
`nlp_utils.py`, `prompts.py`, `personal_prompts.py`, `personal_data.py` are
copied over as-is. What changed:

| File | What happened |
|---|---|
| `app.py`, `chat_ui.py`, `voice_ui.py`, `escalation_ui.py`, `admin_portal.py` | **Removed.** Replaced by React pages/components + FastAPI routes. |
| `auth.py` | Rewritten: same HMAC token scheme, but sets an httpOnly cookie instead of `st.query_params`. |
| `setup.py` | `st.cache_resource`/`st.cache_data` → `functools.lru_cache`. PDF list now read from a new `policy_documents` DB table instead of hardcoded. |
| `speech.py` | Dropped the one `st.cache_resource` on the Whisper model loader (now `functools.lru_cache`). Transcription logic unchanged. |
| `database.py`, `sessions.py`, `schema.sql`, `seed_data.py` | Unchanged — copy as-is. |

## 1. Database migration

Run this once against your existing Postgres DB — it adds the new
`policy_documents` table (for the admin PDF manager) and seeds it with your
current 7 PDFs so nothing breaks on first boot:

```bash
psql "$DB_URL" -f backend/migrations/001_policy_documents.sql
```

## 2. Backend setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in DB_URL, GROQ_API_KEY, HR_AUTH_SECRET, etc.
mkdir -p policies            # put your existing PDF files here (same as before)
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health` — it returns `{"status":"ok","models_ready":false}`
until the embeddings/FAISS/reranker finish loading in the background (same
cold-start cost you had before, just non-blocking now).

## 3. Frontend setup

```bash
cd frontend
npm install
cp .env.example .env         # VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`. Log in with an existing employee email/password
from your `employees` table.

## 4. What's new: Admin → Manage Policies

Admins (`hr_admin` / `super_admin`) now have a **📁 Manage Policies** tab:

- Upload a new PDF, tag it English/Arabic, give it a display name.
- Activate/deactivate or permanently delete existing policy PDFs.
- Click **🔄 Rebuild Index** to re-run FAISS/BM25 indexing against the
  current active set — no redeploy or code edit required.

This works by moving `ARABIC_PDF_ENTRIES`/`ENGLISH_PDF_ENTRIES` out of
`setup.py` and into the `policy_documents` table; `setup.py` now queries it
on every index build (`_load_pdf_entries_from_db`), with a hardcoded fallback
if the table doesn't exist yet (e.g. before you've run the migration).

Uploaded PDFs are stored under `backend/policies/` (configurable via
`POLICY_PDF_DIR` in `.env`).

## 5. Production notes (not done for you — do before shipping)

- Set `secure=True` on the auth cookie in `backend/routes/auth_routes.py`
  once you're behind HTTPS.
- Rebuild-index is synchronous and can take 30s–2min depending on model
  size and PDF count — consider making it a background job with a
  status-polling endpoint if your PDF set grows large.
- Add request size limits / virus scanning on PDF uploads if this is
  internet-facing.
- The chat endpoint sends the full `chat_history` array on every turn
  (mirrors the original `st.session_state` pattern) — fine for a few dozen
  turns, but consider trimming client-side for very long sessions the same
  way `sessions.py` already trims server-side storage to 40 messages.
