const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_STORAGE_KEY = "hr_auth_token";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// Token-based auth: stored in localStorage, sent as an Authorization header.
// (Previously this was an httpOnly cookie + credentials:"include", but
// Hugging Face Spaces' shared proxy was silently stripping the
// Access-Control-Allow-Credentials header that cross-origin cookies
// require. A Bearer token sidesteps that entirely — no credentialed
// requests, so nothing for a proxy to strip.)
export function getToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setToken(token: string) {
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  // Sliding-session refresh: if the backend decided this token is getting
  // close to expiry (see deps.py's _REFRESH_THRESHOLD_SEC), it sends a
  // fresh one here — swap it in silently so an active user never gets
  // logged out mid-session just from time passing while they're using it.
  const refreshed = res.headers.get("X-Refreshed-Token");
  if (refreshed) setToken(refreshed);

  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data.detail || message;
    } catch {
      /* ignore */
    }

    // A 401 here means the token is missing/expired/invalid — the backend
    // will never accept it again as-is. Rather than let every caller show
    // a confusing generic error, clear it and bounce to login once, so the
    // user gets a clear "please sign in again" instead of a dead end.
    if (res.status === 401 && token) {
      clearToken();
      if (typeof window !== "undefined" && window.location.pathname !== "/login") {
        window.location.href = "/login";
      }
    }

    throw new ApiError(res.status, message);
  }

  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

export const api = {
  get: <T>(path: string) => request<T>(path, { method: "GET" }),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : body !== undefined ? JSON.stringify(body) : undefined,
    }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

export { ApiError, BASE_URL };