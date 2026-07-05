import { create } from "zustand";
import { api, ApiError } from "../api/client";
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
    try {
      const { employee } = await api.get<{ employee: Employee }>("/api/auth/me");
      set({ employee, loading: false });
    } catch {
      set({ employee: null, loading: false });
    }
  },

  login: async (email, password) => {
    set({ error: null });
    try {
      const { employee } = await api.post<{ employee: Employee }>("/api/auth/login", {
        email,
        password,
      });
      set({ employee });
      return true;
    } catch (e) {
      set({ error: e instanceof ApiError ? e.message : "Login failed." });
      return false;
    }
  },

  logout: async () => {
    await api.post("/api/auth/logout");
    set({ employee: null });
  },
}));
