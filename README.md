# Horizon Tech — Agentic HR Assistant

An agentic, multilingual (English / MSA Arabic / Egyptian Arabic / Franco-Arabic)
RAG chatbot for HR policy and employee self-service, built for a Bachelor's thesis
at the German University in Cairo.

**Architecture:** React (frontend) + FastAPI (backend) + PostgreSQL. The
retrieval/agent pipeline (hybrid FAISS+BM25 retrieval, LLM-based intent
routing, per-language generation models, self-critique, dialect detection)
is unchanged from the original research/eval implementation — only the UI
layer was rebuilt.

```
hr_chatbot/
├── backend/        FastAPI app — agent, retrieval, DB, auth, admin API
└── frontend/       React + Vite + TypeScript UI
```

## Features

- Multilingual chat (English, MSA Arabic, Egyptian Arabic, Franco-Arabic) with
  voice input (mic recording or file upload) and text-to-speech playback
- Personal HR self-service: leave balances, payroll, performance, OKRs,
  training budget — scoped to the logged-in employee
- Policy Q&A grounded in citable PDF pages, with the actual source page
  rendered inline as an image
- HR escalation: unanswered questions can notify HR directly (never the
  requester themselves, even if they're the admin on duty)
- Admin portal: analytics dashboard, escalation queue, audit log, and a
  **Policy PDF manager** — upload/deactivate/delete policy PDFs and rebuild
  the retrieval index, no code changes required
- Suggested-question chips on first load so new users know what to ask

## Tech stack

- **Frontend:** React 18, TypeScript, Vite, Zustand, React Router
- **Backend:** FastAPI, SQLAlchemy, LangChain + Groq (Llama 4 Scout / Llama
  3.3 70B / Qwen3 32B / Llama 3.1 8B), FAISS, BM25, sentence-transformers
  reranker, Deepgram Nova-3 / local Whisper for STT, gTTS for TTS
- **Database:** PostgreSQL

---

## Part 1 — Local development

### 1. Database

Run once against your Postgres instance (via pgAdmin's Query Tool, or `psql`):

```
backend/schema.sql
backend/migrations/001_policy_documents.sql
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # fill in real values, see table below
mkdir -p policies                # put your policy PDFs here
uvicorn main:app --reload --port 8000
```

Check `http://localhost:8000/health` → `{"status":"ok","models_ready":true}`
once the embedding/reranker models finish loading.

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env             # VITE_API_URL=http://localhost:8000
npm run dev
```

Open `http://localhost:5173`.

### Environment variables (`backend/.env`)

| Variable | Purpose |
|---|---|
| `DB_URL` | Postgres connection string |
| `HR_AUTH_SECRET` | Random secret used to sign login session tokens |
| `GROQ_API_KEY` | Groq API key (LLM inference) |
| `DEEPGRAM_API_KEY` | Optional — voice transcription (falls back to local Whisper if unset) |
| `HR_EMAIL` | Fallback HR contact email for escalations (optional — auto-resolves to an admin if unset) |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASS` / `SMTP_FROM` | Outbound email for escalations + "Contact HR" |
| `POLICY_PDF_DIR` | Where uploaded policy PDFs are stored (default `policies`) |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |

---

## Part 2 — Deploying

This app has two moving pieces that deploy separately: a **static frontend**
and a **long-running backend** (it holds ML models in memory, so it needs a
real server, not a serverless function). Below is the simplest free/cheap
path: **Vercel** for the frontend, **Render** for the backend, and a managed
Postgres (Render's own, or Neon/Supabase).

### Step 1 — Database (Neon, free tier)

1. Go to [neon.tech](https://neon.tech) → sign up → **Create a project**.
2. Once created, copy the **connection string** it gives you (starts with
   `postgresql://...`). This is your `DB_URL` — Neon's format already works
   with SQLAlchemy's `psycopg2` driver, just prefix it:
   ```
   postgresql+psycopg2://<rest of the connection string>
   ```
3. Open Neon's **SQL Editor** (in their dashboard) and paste + run, in order:
   - the full contents of `backend/schema.sql`
   - the full contents of `backend/migrations/001_policy_documents.sql`
4. Keep this tab open — you'll need the connection string again in Step 2.

### Step 2 — Backend (Render)

1. Push your `hr_chatbot` repo to GitHub if it isn't already (you've done this).
2. Go to [render.com](https://render.com) → sign up → **New +** → **Web Service**.
3. Connect your GitHub account, select the `agentic-rag-hr-chatbot` repo.
4. Configure:
   - **Root Directory:** `backend`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** at least the smallest paid tier — the free tier's RAM
     is usually too small once FAISS + the reranker + embedding model are
     all loaded. Check current specs at render.com/pricing.
5. Under **Environment**, add every variable from the table above (`DB_URL`
   from Step 1, `GROQ_API_KEY`, `HR_AUTH_SECRET`, etc.). For `CORS_ORIGINS`,
   put a placeholder for now — you'll update it in Step 4 with your real
   frontend URL.
6. Under **Disks**, add a persistent disk (e.g. 1 GB) mounted at
   `/opt/render/project/src/backend/policies` — otherwise uploaded PDFs are
   wiped on every redeploy, since Render's filesystem is ephemeral by default.
7. Click **Create Web Service**. First deploy will take a while (installing
   `torch`, `transformers`, etc.). Once live, note your backend URL, e.g.
   `https://hr-chatbot-backend.onrender.com`.
8. Visit `https://<your-backend>.onrender.com/health` and confirm it returns
   `{"status":"ok","models_ready":true}` (give it a minute after first boot).

### Step 3 — Upload your policy PDFs

The persistent disk starts empty. Either:
- Log into the deployed app once the frontend is live (Step 4) as an admin,
  and use **Admin → Manage Policies → Upload** for each PDF, then click
  **Rebuild Index**, or
- Use Render's **Shell** tab on your service to `scp`/paste files directly
  into the `policies/` folder, then hit the rebuild-index endpoint.

The first route is easier and is exactly what that admin feature was built for.

### Step 4 — Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → sign up → **Add New** → **Project**.
2. Import the same GitHub repo.
3. Configure:
   - **Root Directory:** `frontend`
   - **Framework Preset:** Vite (should auto-detect)
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Under **Environment Variables**, add:
   ```
   VITE_API_URL=https://<your-backend>.onrender.com
   ```
   (the exact URL from Step 2.7)
5. Click **Deploy**. Once live, note your frontend URL, e.g.
   `https://hr-chatbot.vercel.app`.

### Step 5 — Connect the two

Go back to Render → your backend service → **Environment** → update:
```
CORS_ORIGINS=https://hr-chatbot.vercel.app
```
Save — this triggers a redeploy. Without this, the browser will block every
request from your frontend with a CORS error.

Also open `backend/routes/auth_routes.py` in your repo and change:
```python
secure=False,   # set True in production behind HTTPS
```
to `secure=True` (both Render and Vercel serve over HTTPS by default, so
this is required for the login cookie to actually be sent/received
correctly in production) — commit and push, Render redeploys automatically.

### Step 6 — Test end to end

1. Visit your Vercel URL.
2. Log in with a real employee email/password from your Neon database.
3. Ask a policy question, confirm source PDF pages render.
4. Log in as an admin, confirm Analytics/Escalations/Audit Log/Policy
   Manager all load real data.

### Costs (rough, subject to change — check current pricing)

- **Neon:** free tier is fine for a thesis-scale demo.
- **Vercel:** free tier is fine for the frontend.
- **Render:** the backend needs enough RAM for FAISS + reranker + embedding
  model simultaneously in memory — check Render's current plans and pick
  the smallest one that doesn't OOM on startup (watch the deploy logs).

### Common deployment issues

| Symptom | Likely cause |
|---|---|
| Frontend loads but login fails silently | `CORS_ORIGINS` on backend doesn't exactly match your Vercel URL, or cookie `secure=False` in production |
| `/health` never reaches `models_ready: true` | Backend instance too small (OOM killed mid-load) — check Render logs, upgrade instance |
| Uploaded PDFs disappear after redeploy | Persistent disk not configured, or mounted at the wrong path |
| 401 errors on every request after login | Cookie not being sent — check `credentials: "include"` is intact in `frontend/src/api/client.ts` (it is, by default) and that `CORS_ORIGINS` isn't `*` (must be your exact origin when using credentials) |