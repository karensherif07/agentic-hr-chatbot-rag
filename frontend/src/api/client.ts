const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    credentials: "include", // send/receive the httpOnly session cookie
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    let message = res.statusText;
    try {
      const data = await res.json();
      message = data.detail || message;
    } catch {
      /* ignore */
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
