// Thin typed fetch wrapper. The admin token lives in localStorage (a dashboard
// convenience, not a security boundary — the gateway still validates it server-side).
import type { CacheEntry, Stats, Timeseries, Window } from "./types";

const TOKEN_KEY = "gw_admin_token";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.error?.message ?? body?.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  stats: (w: Window) => request<Stats>(`/admin/stats?window=${w}`),
  timeseries: (w: Window) =>
    request<Timeseries>(`/admin/stats/timeseries?window=${w}&bucket_minutes=${bucketFor(w)}`),
  cacheEntries: () => request<{ items: CacheEntry[] }>("/admin/cache/entries?limit=10"),
};

// Pick a bucket size so every window renders ~40-60 points.
function bucketFor(w: Window): number {
  switch (w) {
    case "1h":
      return 2; // 30 points
    case "24h":
      return 30; // 48 points
    case "7d":
      return 240; // 42 points
  }
}
