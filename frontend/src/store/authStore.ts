import { create } from "zustand";
import { api, ApiError, getToken, setToken, clearToken } from "../api/client";
import type { Employee } from "../api/types";

interface AuthState {
  employee: Employee | null;
  loading: boolean;
  error: string | null;
  init: () => Promise<void>;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  employee: null,
  loading: true,
  error: null,

  init: async () => {
    // No token stored at all → skip the network call entirely, avoids an
    // unnecessary 401 on every fresh page load for logged-out visitors.
    if (!getToken()) {
      set({ employee: null, loading: false });
      return;
    }
    try {
      const { employee } = await api.get<{ employee: Employee }>("/api/auth/me");
      set({ employee, loading: false });
    } catch {
      clearToken();
      set({ employee: null, loading: false });
    }
  },

  login: async (email, password) => {
    set({ error: null });
    try {
      const { employee, token } = await api.post<{ employee: Employee; token: string }>(
        "/api/auth/login",
        { email, password }
      );
      setToken(token);
      set({ employee });
      return true;
    } catch (e) {
      set({ error: e instanceof ApiError ? e.message : "Login failed." });
      return false;
    }
  },

  logout: async () => {
    try {
      await api.post("/api/auth/logout");
    } finally {
      clearToken();
      set({ employee: null });
    }
  },
}));