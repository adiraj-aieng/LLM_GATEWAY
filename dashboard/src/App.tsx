import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError, getToken, setToken } from "./api/client";
import type { CacheEntry, Stats, Timeseries, Window } from "./api/types";
import { Panel, StatCard } from "./components/StatCard";

const WINDOWS: Window[] = ["1h", "24h", "7d"];

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function tsLabel(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export default function App() {
  const [win, setWin] = useState<Window>("24h");
  const [token, setTokenInput] = useState(getToken());
  const [stats, setStats] = useState<Stats | null>(null);
  const [series, setSeries] = useState<Timeseries | null>(null);
  const [entries, setEntries] = useState<CacheEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (w: Window) => {
    setLoading(true);
    setError(null);
    try {
      const [s, t, e] = await Promise.all([api.stats(w), api.timeseries(w), api.cacheEntries()]);
      setStats(s);
      setSeries(t);
      setEntries(e.items);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        setError("Admin token rejected — set it above.");
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (getToken()) void load(win);
    else setError("Enter the admin token to load the dashboard.");
  }, [win, load]);

  function saveToken(e: FormEvent) {
    e.preventDefault();
    setToken(token.trim());
    void load(win);
  }

  const chartData = (series?.series ?? []).map((p) => ({
    ...p,
    label: tsLabel(p.ts),
    hit_rate_pct: Math.round(p.hit_rate * 100),
  }));

  return (
    <div className="app">
      <header>
        <h1>LLM Gateway</h1>
        <div className="controls">
          <form onSubmit={saveToken} className="token-form">
            <input
              type="password"
              placeholder="admin token"
              value={token}
              onChange={(e) => setTokenInput(e.target.value)}
            />
            <button type="submit">Connect</button>
          </form>
          <div className="window-picker">
            {WINDOWS.map((w) => (
              <button
                key={w}
                className={w === win ? "active" : ""}
                onClick={() => setWin(w)}
                type="button"
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error ? <div className="banner error">{error}</div> : null}
      {loading && !stats ? <div className="banner">Loading…</div> : null}

      {stats ? (
        <>
          <div className="cards">
            <StatCard
              label="Requests"
              value={fmt(stats.requests)}
              sub={`${stats.rate_limit.rejected} rate-limited`}
            />
            <StatCard
              label="Cache hit rate"
              value={pct(stats.cache.hit_rate)}
              sub={`${stats.cache.exact} exact · ${stats.cache.semantic} semantic`}
              accent
            />
            <StatCard
              label="Tokens from cache"
              value={fmt(stats.tokens.served_from_cache)}
              sub={`of ${fmt(stats.tokens.total)} total`}
              accent
            />
            <StatCard
              label="p95 latency"
              value={`${stats.latency.p95} ms`}
              sub={`hit ${stats.latency.hit_p95} ms · miss ${stats.latency.miss_p95} ms`}
            />
          </div>

          <div className="charts">
            <Panel title="Hit rate & requests">
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                  <XAxis dataKey="label" stroke="#8b93a7" fontSize={11} />
                  <YAxis yAxisId="l" stroke="#8b93a7" fontSize={11} domain={[0, 100]} />
                  <YAxis yAxisId="r" orientation="right" stroke="#8b93a7" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: "#141821", border: "1px solid #2a2f3a" }}
                  />
                  <Line
                    yAxisId="l"
                    type="monotone"
                    dataKey="hit_rate_pct"
                    name="hit rate %"
                    stroke="#4ade80"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="r"
                    type="monotone"
                    dataKey="requests"
                    name="requests"
                    stroke="#60a5fa"
                    dot={false}
                    strokeWidth={1.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            <Panel title="Latency: hit vs miss (avg per bucket)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                  <XAxis dataKey="label" stroke="#8b93a7" fontSize={11} />
                  <YAxis stroke="#8b93a7" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: "#141821", border: "1px solid #2a2f3a" }}
                  />
                  <Bar dataKey="avg_latency" name="avg latency ms" fill="#60a5fa" />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>

          <div className="two-col">
            <Panel title="Top cached prompts">
              <table>
                <thead>
                  <tr>
                    <th>Prompt</th>
                    <th>Model</th>
                    <th>Hits</th>
                    <th>Best sim</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="empty">
                        No cache hits in the last 7 days
                      </td>
                    </tr>
                  ) : (
                    entries.map((e) => (
                      <tr key={e.prompt_hash}>
                        <td className="preview" title={e.preview ?? ""}>
                          {e.preview ?? e.prompt_hash.slice(0, 12)}
                        </td>
                        <td>{e.model}</td>
                        <td>{e.hits}</td>
                        <td>{e.best_similarity != null ? e.best_similarity.toFixed(3) : "—"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </Panel>

            <Panel title="By model">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Requests</th>
                    <th>Hit rate</th>
                    <th>Tokens</th>
                    <th>p95</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.by_model.map((m) => (
                    <tr key={m.model}>
                      <td>{m.model}</td>
                      <td>{fmt(m.requests)}</td>
                      <td>{pct(m.cache_hit_rate)}</td>
                      <td>{fmt(m.tokens)}</td>
                      <td>{m.p95} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          </div>

          <footer>
            Savings basis: <code>{stats.saved.basis}</code> —{" "}
            {stats.saved.note ??
              "local inference has no marginal $ cost; token and latency savings are the measured figures."}
          </footer>
        </>
      ) : null}
    </div>
  );
}
