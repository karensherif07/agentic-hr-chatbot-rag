"""
auth.py — persistent login with HMAC-signed browser cookie.

ARCHITECTURE:
  Two-layer cookie strategy for reliability:
  1. extra-streamlit-components CookieManager (primary) — reads on every run.
  2. JS/document.cookie injection via st.components.v1.html (write fallback)
     because the CookieManager write can silently race on the first rerun.

  init_cookie_manager() MUST be called before st.set_page_config() in app.py
  is NOT the right order — st.set_page_config() must be first among st.* calls.
  Instead, init_cookie_manager() is called right after set_page_config() and
  before require_login(), which is the correct and safe order.

LOGIN FLOW (2-step, race-free):
  Step A — form submit:
    • verify credentials → hydrate session_state (logged_in=True)
    • write cookie via CookieManager.set()
    • ALSO inject cookie via JS as belt-and-suspenders
    • st.rerun()
  Step B — next run: logged_in is already True → pass through immediately.
    No extra rerun needed; the JS cookie write has already committed to the
    browser by the time the page reloads.
"""

import base64, bcrypt, hashlib, hmac, json, os, time
from datetime import datetime, timedelta, timezone

import streamlit as st
import streamlit.components.v1 as stc
from sqlalchemy import text
from database import get_db

try:
    from extra_streamlit_components import CookieManager
    _HAS_COOKIE_MGR = True
except ImportError:
    CookieManager   = None
    _HAS_COOKIE_MGR = False

COOKIE_NAME     = "hr_policy_session_v1"
_COOKIE_MGR_KEY = "hr_policy_cookie_manager_v2"
_CM_STATE_KEY   = "_hr_cm_instance"
SESSION_TTL_SEC = 7 * 24 * 60 * 60   # 7 days


# ─── Must be called once near the top of app.py ───────────────
def init_cookie_manager():
    """
    Instantiate (or reuse) the CookieManager.
    Call this AFTER st.set_page_config() but BEFORE require_login().
    """
    if not _HAS_COOKIE_MGR:
        print("[auth] init_cookie_manager: extra-streamlit-components not available")
        return
    if _CM_STATE_KEY not in st.session_state:
        print("[auth] init_cookie_manager: creating new CookieManager instance")
        st.session_state[_CM_STATE_KEY] = CookieManager(key=_COOKIE_MGR_KEY)
    else:
        print("[auth] init_cookie_manager: reusing existing CookieManager")


def _cm():
    return st.session_state.get(_CM_STATE_KEY)


# ─── Token helpers ─────────────────────────────────────────────
def _secret() -> bytes:
    s = (os.environ.get("HR_AUTH_SECRET") or "").strip()
    return s.encode() if s else b"hr-chatbot-dev-secret-change-me"


def _make_token(employee_id: int) -> str:
    exp     = int(time.time()) + SESSION_TTL_SEC
    payload = json.dumps({"id": employee_id, "exp": exp}, separators=(",", ":")).encode()
    p64     = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    sig     = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    return f"{p64}.{sig}"


def _read_token() -> int | None:
    """Returns employee_id from valid cookie, or None."""
    cm = _cm()
    if cm is None:
        print(f"[auth] _read_token: CookieManager is None")
        return None
    try:
        raw = cm.get(COOKIE_NAME)
        token = str(raw).strip() if raw else ""
        print(f"[auth] _read_token: got raw token, len={len(token)}")
    except Exception as e:
        print(f"[auth] _read_token: cm.get() failed: {e}")
        return None
    if not token or "." not in token:
        print(f"[auth] _read_token: token invalid or empty")
        return None
    p64, sig = token.rsplit(".", 1)
    want = hmac.new(_secret(), p64.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(want, sig):
        print(f"[auth] _read_token: signature mismatch")
        return None
    pad = "=" * (-len(p64) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode((p64 + pad).encode()).decode())
    except Exception as e:
        print(f"[auth] _read_token: decode failed: {e}")
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        print(f"[auth] _read_token: token expired")
        return None
    emp_id = int(data["id"])
    print(f"[auth] _read_token: token valid, emp_id={emp_id}")
    return emp_id


def _write_cookie(employee_id: int) -> None:
    """
    Write the session cookie two ways:
    1. CookieManager.set() — works reliably when the component has rendered.
    2. JS document.cookie injection — fires immediately on the current page
       load, guaranteeing the cookie exists before the next Streamlit rerun.
    """
    token = _make_token(employee_id)
    exp   = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SEC)

    # --- Method 1: CookieManager ---
    cm = _cm()
    if cm is not None:
        try:
            cm.set(COOKIE_NAME, token, expires_at=exp,
                   max_age=float(SESSION_TTL_SEC), path="/", same_site="lax")
            print("[auth] _write_cookie: CookieManager.set() succeeded")
        except TypeError:
            try:
                cm.set(COOKIE_NAME, token, expires_at=exp, same_site="lax")
                print("[auth] _write_cookie: CookieManager.set() (no max_age) succeeded")
            except Exception as e:
                print(f"[auth] _write_cookie: CookieManager.set() failed: {e}")
        except Exception as e:
            print(f"[auth] _write_cookie: CookieManager.set() failed: {e}")
    else:
        print("[auth] _write_cookie: CookieManager is None, skipping method 1")

    # --- Method 2: JS injection (belt-and-suspenders) ---
    # Encode token as base64 to avoid any quote/semicolon issues in the JS string.
    token_b64 = base64.urlsafe_b64encode(token.encode()).decode()
    js = f"""
    <script>
    (function() {{
        var token = atob("{token_b64}".replace(/-/g,'+').replace(/_/g,'/'));
        var d = new Date();
        d.setTime(d.getTime() + {SESSION_TTL_SEC} * 1000);
        document.cookie = "{COOKIE_NAME}=" + encodeURIComponent(token)
            + "; expires=" + d.toUTCString()
            + "; path=/; SameSite=Lax";
        console.log("[auth] JS cookie write done");
    }})();
    </script>
    """
    stc.html(js, height=0)


def _delete_cookie() -> None:
    cm = _cm()
    if cm is None:
        return
    try:
        cm.delete(COOKIE_NAME)
    except Exception:
        try:
            cm.set(COOKIE_NAME, "",
                   expires_at=datetime(1970, 1, 1, tzinfo=timezone.utc))
        except Exception:
            pass


# ─── DB helpers ────────────────────────────────────────────────
def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _fetch_by_email(email: str) -> dict | None:
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date
            FROM employees WHERE email = :e AND is_active = TRUE
        """), {"e": email.strip().lower()}).fetchone()
    return dict(row._mapping) if row else None


def _fetch_by_id(emp_id: int) -> dict | None:
    with get_db() as db:
        row = db.execute(text("""
            SELECT id, full_name, full_name_ar, email, password_hash,
                   grade, job_title, department, manager_id,
                   hire_date, employment_type, work_model,
                   is_active, probation_end_date
            FROM employees WHERE id = :id AND is_active = TRUE
        """), {"id": emp_id}).fetchone()
    return dict(row._mapping) if row else None


def _hydrate(emp: dict) -> None:
    st.session_state.logged_in        = True
    st.session_state.employee_id      = emp["id"]
    st.session_state.employee_email   = emp["email"]
    st.session_state.employee_name    = emp["full_name"]
    st.session_state.employee_name_ar = emp.get("full_name_ar", "")
    st.session_state.employee_grade   = emp["grade"]
    st.session_state.employee_title   = emp["job_title"]
    st.session_state.employee_dept    = emp["department"]
    st.session_state.manager_id       = emp["manager_id"]
    st.session_state.hire_date        = emp["hire_date"]
    st.session_state.employment_type  = emp["employment_type"]
    st.session_state.work_model       = emp["work_model"]
    st.session_state.in_probation     = emp["probation_end_date"] is not None


# ─── Public API ────────────────────────────────────────────────
def require_login() -> None:
    """
    Call AFTER init_cookie_manager() in app.py.
    Passes through immediately if already logged in (session_state).
    On a fresh page load, tries to restore from cookie.
    Shows login form and calls st.stop() if not authenticated.
    """
    # Already authenticated this run — fast path
    if st.session_state.get("logged_in"):
        return

    # NOTE: The old "_pending_cookie_write" 3-step dance has been removed.
    # It was the root cause of logout-on-refresh: the extra rerun was
    # clearing session_state before the cookie could be read back.
    # The JS injection in _write_cookie() now handles the write reliably
    # without needing a dedicated "write rerun".

    # Try restoring from cookie
    emp_id = _read_token()
    if emp_id:
        emp = _fetch_by_id(emp_id)
        if emp:
            _hydrate(emp)
            print(f"[auth] require_login: restored session for emp_id={emp_id}")
            return
        _delete_cookie()   # stale/invalid token

    # Show login form
    if not _HAS_COOKIE_MGR:
        st.info("💡 `pip install extra-streamlit-components` to persist session across refreshes.")

    st.markdown("<style>.lw{max-width:420px;margin:80px auto 0}</style>",
                unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="lw">', unsafe_allow_html=True)
        st.title("HR Policy Assistant")
        st.caption("Sign in with your Horizon Tech email to continue.")
        with st.form("login_form"):
            email    = st.text_input("Work email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Sign in", use_container_width=True)
        if submit:
            if not email or not password:
                st.error("Please enter your email and password.")
                st.stop()
            emp = _fetch_by_email(email)
            if emp is None:
                st.error("Email not found or account is inactive.")
                st.stop()
            if not _verify_password(password, emp["password_hash"]):
                st.error("Incorrect password.")
                st.stop()
            # Hydrate session and write cookie BEFORE rerun.
            # logged_in=True is already in session_state, so the next run
            # takes the fast path above without needing to re-read the cookie.
            _hydrate(emp)
            _write_cookie(emp["id"])
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def logout() -> None:
    _delete_cookie()
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()