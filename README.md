# Horizon Assistant — Agentic Multilingual HR Chatbot

An AI-powered HR assistant built for **Horizon Tech**, a fictional MENA-region
company created for this project. It answers HR policy questions and gives
employees secure, self-service access to their own HR data — leave balance,
payroll, performance reviews, OKRs, and training budget — in four languages.

Built as a Bachelor's thesis project at the German University in Cairo,
Media Engineering and Technology department.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## What it does

**For employees:**
- Ask HR policy questions in **English, Modern Standard Arabic, Egyptian
  Arabic, or Franco-Arabic** — the assistant detects the language and
  dialect automatically and responds in kind
- Every policy answer is grounded in the actual source document — click
  "Source Evidence" to see the real PDF page the answer came from
- Ask personal questions ("How many leave days do I have left?", "What was
  my last performance rating?") and get answers pulled live from your own
  HR record — no one else's data is ever exposed
- Speak your question instead of typing it (microphone or upload an audio
  file), and have answers read back to you
- Not sure what to ask? Quick-start suggestion chips cover common topics —
  leave, payroll, performance, training, and policies
- If a question can't be answered from policy or your data, you can notify
  HR directly with one click

**For HR administrators:**
- A live analytics dashboard — query volume, language breakdown, intent
  breakdown, and an unanswered-questions report, exportable to CSV
- An escalation queue for questions the bot couldn't answer, with one-click
  resolution tracking
- A full audit log of every admin action
- A **policy document manager** — upload, tag, activate/deactivate, or
  remove HR policy PDFs directly from the browser, and rebuild the
  retrieval index on demand. No code changes or redeployments needed to
  update what the assistant knows.

---

## How it works, briefly

Underneath the chat interface is an agentic retrieval-augmented generation
(RAG) pipeline:

1. A lightweight routing model classifies each question's **intent**
   (policy question, personal data question, both, or out of scope) and
   picks which tools to call.
2. Policy questions run through **hybrid retrieval** — dense semantic
   search (FAISS) combined with lexical search (BM25) — followed by a
   cross-encoder reranker that guarantees balanced English/Arabic source
   coverage.
3. Personal questions query the employee's own records directly from the
   HR database, scoped strictly to their `employee_id`.
4. A language-appropriate generation model (separate models are used for
   English/Franco-Arabic vs. MSA/Egyptian Arabic) produces the final answer,
   with every factual claim citing its source page.
5. A self-critique pass checks the answer before it's returned.

Full technical documentation of the pipeline, evaluation methodology, and
architectural decisions lives in the thesis document itself.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | FastAPI, SQLAlchemy |
| Database | PostgreSQL |
| LLMs | Llama 4 Scout, Llama 3.3 70B, Qwen3 32B, Llama 3.1 8B (via Groq) |
| Retrieval | FAISS, BM25 (rank-bm25), multilingual-e5-large embeddings |
| Reranking | bge-reranker-v2-m3 (cross-encoder) |
| Dialect detection | MARBERTv2 |
| Speech-to-text | Deepgram Nova-3, with local Whisper fallback |
| Text-to-speech | gTTS |

---

## Getting started

Want to run this locally or deploy it yourself? See **[SETUP.md](./SETUP.md)**
for local development and **[DEPLOYMENT.md](./DEPLOYMENT.md)** for a full,
step-by-step production deployment guide.

---

## Project structure

```
hr_chatbot/
├── backend/     FastAPI app — agent, retrieval, database, auth, admin API
└── frontend/    React + Vite UI
```

## Author

Karen Sherif — German University in Cairo, Media Engineering and Technology.
Thesis supervised by Dr. Nada Sharaf.