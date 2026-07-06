"""
deps.py — Shared FastAPI dependencies.

Replaces what Streamlit gave you for free:
  - st.session_state   -> Bearer token (validated per request) + DB lookup
  - st.cache_resource   -> module-level singleton, built once at startup

Note: this was originally cookie-based, but Hugging Face Spaces sits behind
a shared proxy that was silently stripping the
`Access-Control-Allow-Credentials` header, which cookie-based cross-origin
auth depends on. Switched to a standard Bearer token instead — the frontend
sends `Authorization: Bearer <token>` on each request, which doesn't need
that header at all and works regardless of what any intermediate proxy does
with cookies.
"""

import os
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status

import auth  # your existing auth.py logic (token make/validate + DB fetch), trimmed of st.*
from setup import setup as build_models
from policy_sync import sync_active_policies_to_disk


# ─────────────────────────────────────────────────────────────
# Model singleton — built once, shared across all requests.
# In Streamlit this was st.cache_resource; here it's just module state
# populated once in main.py's startup event.
# ─────────────────────────────────────────────────────────────
class ModelBundle:
    ar_index = None
    en_index = None
    routing_llm = None
    en_llm = None
    ar_llm = None
    critique_llm = None
    reranker = None
    dialect_pipe = None
    ara_tokenizer = None
    ready = False


models = ModelBundle()


def load_models_once():
    if models.ready:
        return
    sync_active_policies_to_disk()
    (models.ar_index, models.en_index,
     models.routing_llm, models.en_llm, models.ar_llm, models.critique_llm,
     models.reranker, models.dialect_pipe, models.ara_tokenizer) = build_models()
    models.ready = True


def get_models() -> ModelBundle:
    if not models.ready:
        raise HTTPException(503, "Models still loading, try again shortly.")
    return models


# ─────────────────────────────────────────────────────────────
# Auth — reads the Bearer token from the Authorization header,
# validates it with the same HMAC scheme as your original auth.py,
# and loads the employee row fresh from the DB every request.
# ─────────────────────────────────────────────────────────────
def _extract_bearer_token(request: Request) -> "str | None":
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer "):].strip()


def get_current_employee(request: Request) -> dict:
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in.")
    emp_id = auth.validate_token(token)
    if emp_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired.")
    emp = auth.fetch_by_id(emp_id)
    if emp is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account not found or deactivated.")
    return emp


def require_admin(emp: dict = Depends(get_current_employee)) -> dict:
    if emp.get("admin_role") not in auth.ADMIN_ROLES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return emp