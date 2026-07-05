"""
main.py — FastAPI entrypoint.

Run with:
    uvicorn main:app --reload --port 8000

Frontend dev server (Vite, default port 5173) talks to this over
http://localhost:8000 — see frontend/src/api/client.ts for the base URL
and CORS_ORIGINS below for what's allowed.
"""

import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deps import load_models_once
from routes import auth_routes, chat_routes, voice_routes, admin_routes, policy_routes, pdf_routes

app = FastAPI(title="Horizon Tech HR Assistant API")

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,   # required so the httpOnly auth cookie is sent/received
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(chat_routes.router)
app.include_router(voice_routes.router)
app.include_router(admin_routes.router)
app.include_router(policy_routes.router)
app.include_router(pdf_routes.router)


@app.on_event("startup")
def _startup():
    # Model loading (embeddings, FAISS, reranker, LLM clients) is heavy —
    # do it in a background thread so the API can report /health immediately
    # while /api/chat/message returns 503 until models.ready is True.
    threading.Thread(target=load_models_once, daemon=True).start()


@app.get("/health")
def health():
    from deps import models
    return {"status": "ok", "models_ready": models.ready}
