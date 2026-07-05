# Deployment Guide

This app has two parts that deploy separately:

- **Frontend** — a static site once built (React/Vite) → deploy to **Vercel**
- **Backend** — a long-running Python server that holds ML models in memory
  (FAISS index, reranker, embedding model) → needs a real server, not a
  serverless function → deploy to **Render**
- **Database** — PostgreSQL → use a managed free-tier instance from **Neon**

Total cost for a thesis-scale demo: likely $0–7/month depending on the
Render instance size you need (see Step 2, note on RAM).

Estimated total time: 45–75 minutes, most of it waiting for the backend's
first build (installing `torch`, `transformers`, etc. takes a while).

---

## Before you start

Make sure your latest code is pushed to GitHub:
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

Decide now whether your policy PDFs are committed to the repo or not (see
the note in the main README about `.gitignore`). If they're **not**
committed, you'll upload them manually in Step 6 — that's fine, just don't
skip that step.

---

## Step 1 — Create the database (Neon)

1. Go to **[neon.tech](https://neon.tech)** and sign up (GitHub login is
   fastest).
2. Click **Create a project**. Give it a name (e.g. `horizon-hr`), pick a
   region close to you, and click **Create Project**.
3. Once it's created, you'll land on a dashboard showing a **connection
   string** that looks like:
   ```
   postgresql://neondb_owner:AbC123xyz@ep-cool-name-12345.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
   Click the copy icon next to it. **Save this somewhere temporarily** (a
   scratch text file) — you'll need it twice.
4. Your actual `DB_URL` for the app needs a small prefix change. Take what
   you copied and change:
   ```
   postgresql://...
   ```
   to:
   ```
   postgresql+psycopg2://...
   ```
   (just add `+psycopg2` after `postgresql`). Save this final version — this
   exact string is your `DB_URL`.
5. In the left sidebar of the Neon dashboard, click **SQL Editor**.
6. Open `backend/schema.sql` from your project in a text editor, select all
   the text, copy it.
7. Paste it into Neon's SQL Editor and click **Run** (or the ▶ button).
   You should see a success message and a list of created tables.
8. Repeat steps 6–7 with `backend/migrations/001_policy_documents.sql`.
9. Optional sanity check: in the sidebar, click **Tables** — you should see
   `employees`, `leave_balances`, `policy_documents`, and the rest.

---

## Step 2 — Deploy the backend (Render)

1. Go to **[render.com](https://render.com)** and sign up (GitHub login
   recommended — it makes repo connection automatic).
2. Click **New +** (top right) → **Web Service**.
3. If prompted, connect your GitHub account and grant access to the
   `agentic-rag-hr-chatbot` repository.
4. Select that repo from the list.
5. Fill in the configuration form:
   - **Name:** `horizon-hr-backend` (or anything you like — this becomes
     part of your URL)
   - **Region:** pick one close to your users
   - **Branch:** `main`
   - **Root Directory:** `backend`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Scroll to **Instance Type**. Check Render's current plan list at
   render.com/pricing — you need enough RAM to hold FAISS + the
   cross-encoder reranker + the embedding model simultaneously. The
   free/smallest tier commonly fails here with an out-of-memory error
   during startup; if that happens, this is the setting to upgrade.
7. Scroll to **Environment Variables**. Click **Add Environment Variable**
   for each of these (values from your own setup):

   | Key | Value |
   |---|---|
   | `DB_URL` | the `postgresql+psycopg2://...` string from Step 1 |
   | `HR_AUTH_SECRET` | any long random string you make up |
   | `GROQ_API_KEY` | your Groq API key |
   | `DEEPGRAM_API_KEY` | your Deepgram key (optional) |
   | `HR_EMAIL` | HR contact email (optional) |
   | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM` | your email provider's SMTP details (optional — skip if you don't need escalation emails yet) |
   | `POLICY_PDF_DIR` | `policies` |
   | `CORS_ORIGINS` | `http://localhost:5173` for now — you'll change this in Step 5 |

8. Scroll to **Disks** (you may need to click "Advanced" to find it). Click
   **Add Disk**:
   - **Name:** `policies-storage`
   - **Mount Path:** `/opt/render/project/src/backend/policies`
   - **Size:** 1 GB is plenty

   This matters because Render's filesystem resets on every deploy — without
   a persistent disk, any PDFs uploaded through the admin panel would vanish
   the next time you push code.
9. Click **Create Web Service** at the bottom.
10. You'll be taken to a logs page — watch it. The first build takes a
    while (5–15 minutes is normal, since it's installing `torch` and
    friends). Wait for a line like `Uvicorn running on http://0.0.0.0:$PORT`.
11. At the top of the page, note your service URL — something like
    `https://horizon-hr-backend.onrender.com`. **Save this.**
12. Open `https://horizon-hr-backend.onrender.com/health` in a browser tab.
    You should see:
    ```json
    {"status":"ok","models_ready":false}
    ```
    Refresh every 15–30 seconds — once model loading finishes (can take a
    couple of minutes after the server starts), it flips to
    `"models_ready":true`. If it never flips and the Render logs show the
    process restarting repeatedly, that's the out-of-memory issue from
    step 6 — go back and pick a larger instance.

---

## Step 3 — Deploy the frontend (Vercel)

1. Go to **[vercel.com](https://vercel.com)** and sign up (GitHub login
   recommended).
2. Click **Add New...** → **Project**.
3. Find and import the same `agentic-rag-hr-chatbot` repo.
4. On the configuration screen:
   - **Framework Preset:** should auto-detect as **Vite** — if not, select
     it manually
   - **Root Directory:** click **Edit** next to it, choose `frontend`
   - **Build Command:** `npm run build` (should be pre-filled)
   - **Output Directory:** `dist` (should be pre-filled)
5. Expand **Environment Variables** and add:
   - **Key:** `VITE_API_URL`
   - **Value:** your Render URL from Step 2.11, e.g.
     `https://horizon-hr-backend.onrender.com`
6. Click **Deploy**.
7. Wait for the build to finish (1–3 minutes, much faster than the backend).
8. Once done, Vercel shows you the live URL, e.g.
   `https://horizon-hr-chatbot.vercel.app`. **Save this.**

---

## Step 4 — Fix the login cookie for HTTPS

Both Render and Vercel serve over HTTPS, which requires one small code
change so login sessions actually persist:

1. In your local project, open `backend/routes/auth_routes.py`.
2. Find this line:
   ```python
   secure=False,   # set True in production behind HTTPS
   ```
3. Change it to:
   ```python
   secure=True,
   ```
4. Commit and push:
   ```bash
   git add backend/routes/auth_routes.py
   git commit -m "Enable secure cookies for production"
   git push origin main
   ```
5. Render will automatically detect the push and redeploy (watch the Render
   logs page — you'll see a new build start).

---

## Step 5 — Connect frontend and backend (CORS)

1. Go back to your Render dashboard → your backend service → **Environment**
   tab.
2. Find the `CORS_ORIGINS` variable and edit it. Change the value to your
   actual Vercel URL from Step 3.8, exactly as shown (no trailing slash):
   ```
   https://horizon-hr-chatbot.vercel.app
   ```
3. Click **Save Changes** — this triggers another redeploy. Wait for it to
   finish.

This step matters a lot: without it, every request from your deployed
frontend will fail with a CORS error in the browser console, even though
both services are technically "up."

---

## Step 6 — Upload your policy PDFs (if not committed to git)

Skip this step if your PDFs are already in `backend/policies/` in your repo
and got picked up automatically on deploy.

Otherwise:
1. Visit your live Vercel URL.
2. Log in with an **admin** account (one with `admin_role` set in your
   `employees` table).
3. Go to **Admin Portal → Manage Policies**.
4. For each PDF: choose the file, give it a display name, pick its
   language, click **Upload**.
5. Once all PDFs are uploaded, click **Rebuild Index** — this can take
   30 seconds to a couple of minutes depending on how many documents you
   have.

---

## Step 7 — Full test pass

Go through this checklist on your live URL:

- [ ] Load the site — login page appears, styled correctly
- [ ] Log in as a regular employee
- [ ] Ask a policy question in English — get a grounded answer with a
      "Source Evidence" section that shows a real PDF page image
- [ ] Ask a personal question ("how many leave days do I have left?") —
      get an answer with a "Your data used" panel
- [ ] Try the mic button and the file-upload voice button
- [ ] Click a suggested-question chip on the empty chat screen
- [ ] Log out, log in as an admin
- [ ] Check Admin → Analytics loads real data and the CSV download works
- [ ] Check Admin → Escalations and Audit Log load without errors
- [ ] Check Admin → Manage Policies shows your uploaded PDFs

If every box checks out, you're deployed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Frontend loads, but login silently fails / redirects back to login | `CORS_ORIGINS` doesn't exactly match your Vercel URL, or cookie `secure` flag wasn't updated | Re-check Step 4 and Step 5 |
| `/health` never reaches `models_ready: true`, Render logs show repeated restarts | Instance out of memory | Upgrade Render instance size |
| Uploaded PDFs vanish after a new deploy | Persistent disk missing or mounted at wrong path | Re-check Step 2.8's exact mount path |
| Browser console shows a CORS error | `CORS_ORIGINS` env var not updated, or has a trailing slash / wrong protocol | Must match your frontend URL exactly, including `https://`, no trailing slash |
| 401 Unauthorized on every action right after logging in | Cookie not being accepted cross-origin | Confirm `secure=True` was actually pushed and Render redeployed after Step 4 |
| Backend builds fail on `torch` or `transformers` install | Render's Python version mismatch, or instance ran out of disk during install | Check the Render build logs for the exact error; may need to pin versions in `requirements.txt` |