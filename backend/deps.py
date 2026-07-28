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

from fastapi import Depends, HTTPException, Request, Response, status
from slowapi import Limiter
from slowapi.util import get_remote_address

import auth  # your existing auth.py logic (token make/validate + DB fetch), trimmed of st.*
from setup import setup as build_models
from policy_sync import sync_active_policies_to_disk


def _rate_limit_key(request: Request) -> str:
    """
    Key rate limits by employee id when the request carries a valid token,
    so limits are per-person rather than per-IP (several employees could
    share a NAT'd IP, e.g. all on the same office network). Falls back to
    IP address for unauthenticated requests (e.g. a bad/expired token still
    needs *some* key to rate-limit against).
    """
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        token = header[len("Bearer "):].strip()
        emp_id = auth.validate_token(token)
        if emp_id is not None:
            return f"employee:{emp_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


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

# Shared progress tracker for model (re)builds — read by
# GET /api/admin/policies/rebuild-status so the frontend can show live
# stage text instead of a bare spinner during "Rebuild Index" (which can
# take minutes on free-tier CPU hardware).
rebuild_status = {"stage": "idle", "error": None}


def load_models_once():
    if models.ready:
        return
    rebuild_status["stage"] = "Syncing policy PDFs from database…"
    rebuild_status["error"] = None
    sync_active_policies_to_disk()

    def _on_progress(stage):
        rebuild_status["stage"] = stage

    try:
        (models.ar_index, models.en_index,
         models.routing_llm, models.en_llm, models.ar_llm, models.critique_llm,
         models.reranker, models.dialect_pipe, models.ara_tokenizer) = build_models(
            on_progress=_on_progress
        )
        models.ready = True
        rebuild_status["stage"] = "idle"
    except Exception as e:
        rebuild_status["stage"] = "idle"
        rebuild_status["error"] = str(e)
        raise


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


# Refresh a token once it has less than this much life left, so an active
# user never gets logged out mid-session just because 7 days passed while
# they were using the app — they only get logged out after 7 days of
# actual inactivity (no requests at all).
_REFRESH_THRESHOLD_SEC = 24 * 60 * 60  # 1 day


def get_current_employee(request: Request, response: Response) -> dict:
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in.")
    emp_id = auth.validate_token(token)
    if emp_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired.")
    emp = auth.fetch_by_id(emp_id)
    if emp is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account not found or deactivated.")

    remaining = auth.token_seconds_remaining(token)
    if remaining is not None and remaining < _REFRESH_THRESHOLD_SEC:
        fresh_token = auth.make_token(emp_id)
        # Exposed via CORS's default allowed-response-headers is NOT
        # guaranteed for custom headers — see main.py's CORS config, which
        # explicitly allows this one via expose_headers.
        response.headers["X-Refreshed-Token"] = fresh_token

    return emp


def require_admin(emp: dict = Depends(get_current_employee)) -> dict:
    if emp.get("admin_role") not in auth.ADMIN_ROLES:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required.")
    return emp