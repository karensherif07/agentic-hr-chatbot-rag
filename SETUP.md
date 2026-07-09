# Local Development Setup

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (local install, or a free hosted instance like [Neon](https://neon.tech))
- A [Groq](https://console.groq.com) API key

## 1. Database

Run these two SQL files once against your database, in order — via pgAdmin's
Query Tool (open the file, copy contents, paste into Query Tool, execute) or
`psql -f <file>` if you have it on your PATH:

```
backend/schema.sql
backend/migrations/001_policy_documents.sql
backend/migrations/002_add_pdf_bytes.sql
```

## 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in real values — see the table below.

```bash
mkdir -p policies   # place your HR policy PDFs here
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/health` — it should show
`{"status":"ok","models_ready":false}` at first, then `true` once the
embedding/reranker models finish loading (this can take a minute or two).

## 3. Frontend

In a **separate terminal**:

```bash
cd frontend
npm install
cp .env.example .env    # VITE_API_URL=http://localhost:8000
npm run dev
```

Visit `http://localhost:5173` and log in with an employee email/password
that exists in your database.

## Environment variables (`backend/.env`)

| Variable | Purpose | Required? |
|---|---|---|
| `DB_URL` | Postgres connection string | Yes |
| `HR_AUTH_SECRET` | Random string used to sign login session tokens | Yes |
| `GROQ_API_KEY` | Groq API key for all LLM calls | Yes |
| `DEEPGRAM_API_KEY` | Voice transcription | No — falls back to local Whisper |
| `HR_EMAIL` | Fixed HR contact for escalations | No — auto-resolves to an admin |
| `SMTP_HOST` / `PORT` / `USER` / `PASS` / `FROM` | Outbound email for escalations and "Contact HR" | No — those features silently no-op without it |
| `POLICY_PDF_DIR` | Where uploaded policy PDFs live | No — defaults to `policies` |
| `CORS_ORIGINS` | Allowed frontend origin(s), comma-separated | Yes |

## Running both at once

Keep two terminals open: backend on port 8000, frontend on port 5173. The
frontend's `VITE_API_URL` tells it where to find the backend.